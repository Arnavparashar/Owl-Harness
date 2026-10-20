"""Background execution of runs started from the dashboard.

A job is one request to test one task on one or more models. Jobs wait in a queue and a
single worker thread runs them, and the models inside a job, one at a time: local models
share the same CPU/GPU, so running them side by side would distort the latency numbers.
Everything a run produces is stored by `run_task` exactly as for `owl run`; the job only
tracks progress while it is happening.
"""

from __future__ import annotations

import asyncio
import queue
import secrets
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from owl.custom import CustomTaskSpec, DatasetLibrary, build_task
from owl.providers.base import get_provider_class, split_model_string, validate_model_args
from owl.runner import run_task
from owl.store import SampleRecord, Store
from owl.tasks import Task, get_task

MAX_JOBS_KEPT = 50


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ModelSpec(BaseModel):
    """A model to test: a saved one (`model_id`) or provider, model and settings given here."""

    model_id: str | None = None
    provider: str | None = Field(default=None, min_length=1, max_length=64)
    model: str | None = Field(default=None, min_length=1, max_length=400)
    args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _given(self) -> "ModelSpec":
        if not self.model_id and not (self.provider and self.model):
            raise ValueError("each model needs either model_id or provider and model")
        return self


class RunRequest(BaseModel):
    """Test one task on one or more models."""

    task: str | None = None  # a built-in or plugin task
    custom: CustomTaskSpec | None = None  # or a test defined in the dashboard
    models: list[ModelSpec] = Field(min_length=1, max_length=10)
    limit: int | None = Field(default=None, ge=1)
    max_concurrency: int | None = Field(default=None, ge=1, le=64)
    timeout_s: float = Field(default=120.0, gt=0, le=3600)
    retries: int = Field(default=2, ge=0, le=10)
    # Generation settings for this run; unset keeps the test's own (temperature 0 and seed 42 by default).
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def _one_task(self) -> "RunRequest":
        if (self.task is None) == (self.custom is None):
            raise ValueError("give either 'task' (a built-in test) or 'custom' (a test you defined), not both")
        return self


ItemStatus = Literal["queued", "running", "completed", "completed_with_errors", "failed", "cancelled"]
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class JobItem:
    """Progress of one model within a job."""

    def __init__(self, provider: str, model: str, args: dict[str, Any]) -> None:
        self.provider = provider
        self.model = model
        self.args = args
        self.status: ItemStatus = "queued"
        self.run_id: str | None = None
        self.done = 0
        self.total = 0
        self.correct = 0
        self.errors = 0
        self.error: str | None = None
        self.started_at: str | None = None
        self.finished_at: str | None = None
        # set while the item runs, so a cancel request can reach its event loop
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None

    @property
    def name(self) -> str:
        return f"{self.provider}/{self.model}"

    def to_dict(self, store: Store) -> dict[str, Any]:
        metrics = store.get_metrics(self.run_id) if self.run_id and self.status.startswith("completed") else {}
        main = None
        if "mean_score" in metrics:
            main = {"name": "mean score", "value": metrics["mean_score"]}
        elif "pass_rate" in metrics:
            main = {"name": "pass rate", "value": metrics["pass_rate"]}
        elif "accuracy" in metrics:
            main = {"name": "accuracy", "value": metrics["accuracy"]}
        return {
            "model": self.name,
            "status": self.status,
            "run_id": self.run_id,
            "done": self.done,
            "total": self.total,
            "correct": self.correct,
            "errors": self.errors,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "accuracy": metrics.get("accuracy"),
            "main_metric": main,
        }


class Job:
    def __init__(self, request: RunRequest, task: Task, items: list[JobItem]) -> None:
        self.id = secrets.token_hex(5)
        self.request = request
        self.task = task
        self.items = items
        self.status: JobStatus = "queued"
        self.created_at = _now()
        self.finished_at: str | None = None
        self.cancel_requested = False
        self.warnings: list[str] = []

    def to_dict(self, store: Store) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "task": self.task.name,
            "description": self.task.description,
            "scoring_mode": self.task.mode,
            "warnings": self.warnings,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "cancel_requested": self.cancel_requested,
            "limit": self.request.limit,
            "items": [i.to_dict(store) for i in self.items],
        }


class JobManager:
    def __init__(self, store: Store, library: DatasetLibrary) -> None:
        self.store = store
        self.library = library
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._idle = threading.Condition(self._lock)

    # --- public ---------------------------------------------------------------

    def _resolve(self, spec: Any) -> tuple[str, str, dict[str, Any]]:
        """(provider, model, settings) for a model spec, looking saved models up by id."""
        if spec.model_id:
            saved = self.store.get_model(spec.model_id)
            if saved is None:
                raise KeyError(f"no saved model '{spec.model_id}'")
            provider, model, args = saved["provider"], saved["model"], {**saved["settings"], **spec.args}
        else:
            provider, model, args = spec.provider, spec.model, spec.args
        if "/" in provider:
            raise ValueError(f"invalid provider name '{provider}'")
        cls = get_provider_class(provider)  # KeyError if unknown
        split_model_string(f"{provider}/{model}")
        return provider, model, validate_model_args(cls, args)

    def submit(self, request: RunRequest) -> Job:
        """Validate everything up front (so mistakes fail fast), then queue the job."""
        warnings: list[str] = []
        judge_model: tuple[str, str] | None = None
        if request.task:
            task = get_task(request.task)
        else:
            custom = request.custom
            assert custom is not None
            if custom.judge is not None:
                provider, model, args = self._resolve(custom.judge)
                if get_provider_class(provider).binary_only:
                    raise ValueError(f"{provider} only answers yes/no questions and cannot be a judge")
                judge_model = (provider, model)
                resolved = custom.judge.model_copy(update={"model_id": None, "provider": provider, "model": model, "args": args})
                custom = custom.model_copy(update={"judge": resolved})
            task = build_task(custom, self.library, timeout_s=request.timeout_s)

        items: list[JobItem] = []
        for m in request.models:
            provider, model, args = self._resolve(m)
            if get_provider_class(provider).binary_only and task.positive_class is None:
                raise ValueError(f"{provider} only answers yes/no questions and cannot run this test")
            items.append(JobItem(provider, model, args))
            if judge_model == (provider, model):
                # Allowed, but worth knowing: self-grading tends to be lenient.
                warnings.append(f"{provider}/{model} is also the judge. A model grading its own answers tends to be lenient.")

        job = Job(request, task, items)
        job.warnings = warnings
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > MAX_JOBS_KEPT:
                oldest = next((j for j in self._jobs.values() if j.status not in ("queued", "running")), None)
                if oldest is None:
                    break
                del self._jobs[oldest.id]
            self._queue.put(job.id)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, name="owl-jobs", daemon=True)
                self._worker.start()
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"no job '{job_id}'")
            return self._jobs[job_id]

    def list(self) -> list[Job]:
        with self._lock:
            return list(reversed(self._jobs.values()))

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        with self._lock:
            if job.status in ("completed", "failed", "cancelled"):
                return job
            job.cancel_requested = True
            if job.status == "queued":  # never started: finish it right away
                for item in job.items:
                    item.status = "cancelled"
                self._finish(job, "cancelled")
                return job
            for item in job.items:
                if item.status == "queued":
                    item.status = "cancelled"
                elif item.status == "running" and item._loop is not None and item._task is not None:
                    item._loop.call_soon_threadsafe(item._task.cancel)
        return job

    def active_run_ids(self) -> set[str]:
        """Runs a job is still working on; these must not be deleted."""
        with self._lock:
            return {i.run_id for j in self._jobs.values() for i in j.items if i.run_id and i.status == "running"}

    def wait(self, job_id: str, timeout: float = 30.0) -> Job:
        """Block until a job finishes. Used by tests and scripts."""
        job = self.get(job_id)
        with self._idle:
            self._idle.wait_for(lambda: job.status in ("completed", "failed", "cancelled"), timeout=timeout)
        return job

    # --- worker ---------------------------------------------------------------

    def _finish(self, job: Job, status: JobStatus) -> None:
        with self._idle:
            job.status = status
            job.finished_at = _now()
            self._idle.notify_all()

    def _work(self) -> None:
        while True:
            job_id = self._queue.get()
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job.status != "queued":
                    continue
                job.status = "running"
            try:
                self._run_job(job)
            except Exception as e:  # a bug here must not kill the worker or leave the job hanging
                for item in job.items:
                    if item.status in ("queued", "running"):
                        item.status, item.error = "failed", f"internal error: {type(e).__name__}: {e}"
                self._finish(job, "failed")

    def _run_job(self, job: Job) -> None:
        for item in job.items:
            with self._lock:
                if job.cancel_requested:
                    item.status = "cancelled"
                    continue
                item.status = "running"
                item.started_at = _now()
            try:
                asyncio.run(self._run_item(job, item))
            except asyncio.CancelledError:
                item.status = "cancelled"
            except Exception as e:
                item.status = "failed"
                item.error = f"{type(e).__name__}: {e}"[:500]
            finally:
                item._loop = item._task = None
                item.finished_at = _now()
        with self._lock:
            failed = any(i.status == "failed" for i in job.items)
        self._finish(job, "cancelled" if job.cancel_requested else "failed" if failed else "completed")

    async def _run_item(self, job: Job, item: JobItem) -> None:
        item._loop = asyncio.get_running_loop()
        item._task = asyncio.current_task()
        if job.cancel_requested:  # cancelled between the status change and here
            raise asyncio.CancelledError

        def on_start(run_id: str, total: int) -> None:
            item.run_id, item.total = run_id, total

        def on_progress(done: int, total: int, rec: SampleRecord) -> None:
            item.done, item.total = done, total
            if rec.error:
                item.errors += 1
            elif rec.correct:
                item.correct += 1

        req = job.request
        run_id = await run_task(
            job.task,
            item.name,
            store=self.store,
            model_args=item.args,
            limit=req.limit,
            max_concurrency=req.max_concurrency,
            timeout_s=req.timeout_s,
            retries=req.retries,
            on_progress=on_progress,
            on_start=on_start,
            generate_overrides={"temperature": req.temperature, "max_tokens": req.max_tokens, "seed": req.seed},
        )
        item.run_id = run_id
        run = self.store.get_run(run_id)
        item.status = run["status"] if run else "completed"  # completed or completed_with_errors

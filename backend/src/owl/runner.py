"""Async runner: executes a task against one model and persists results as it goes."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from owl.metrics import compute_metrics
from owl.providers.base import Provider, get_provider_class, split_model_string
from owl.solvers import SolverContext, initial_state
from owl.store import SampleRecord, Store
from owl.tasks import Task, get_task
from owl.types import Sample, RunConfig

ProgressFn = Callable[[int, int, SampleRecord], None]
StartFn = Callable[[str, int], None]  # called with the run id and sample count as soon as the run exists


class DatasetSizeError(RuntimeError):
    pass


async def _score_sample(task: Task, sample: Sample, ctx: SolverContext, timeout_s: float) -> SampleRecord:
    state = initial_state(sample)
    state = await asyncio.wait_for(task.solver(state, ctx), timeout=timeout_s)
    if state.output is None:
        raise RuntimeError("solver finished without producing a model output")
    out = state.output
    score = task.scorer(sample, out)
    return SampleRecord(
        sample_id=sample.id,
        input=sample.input,
        target=sample.target,
        output=out.text,
        pred=score.pred,
        correct=score.correct,
        probability=out.probability,
        latency_ms=out.latency_ms,
        stop_reason=out.stop_reason,
        metadata={
            "explanation": score.explanation,
            "score": score.metadata,
            "sample": sample.metadata,
            "usage": out.usage.model_dump(),
            "raw": out.raw,
        },
    )


async def run_task(
    task: Task | str,
    model: str,
    *,
    store: Store,
    model_args: dict[str, Any] | None = None,
    limit: int | None = None,
    max_concurrency: int | None = None,
    timeout_s: float = 120.0,
    retries: int = 2,
    resume_run_id: str | None = None,
    on_progress: ProgressFn | None = None,
    on_start: StartFn | None = None,
) -> str:
    """Run `task` on `model` ("provider/model"). Returns the run id.

    Every sample is written to the store the moment it finishes. Passing
    `resume_run_id` continues an interrupted run, skipping samples that already
    have a result (samples that errored are retried).
    """
    if isinstance(task, str):
        task = get_task(task)
    provider_name, model_name = split_model_string(model)
    provider_cls = get_provider_class(provider_name)
    args = {**task.provider_args.get(provider_name, {}), **(model_args or {})}

    dataset = task.load_dataset()
    if task.expected_size is not None and len(dataset) != task.expected_size:
        raise DatasetSizeError(
            f"task '{task.name}' expects {task.expected_size} samples but the dataset has {len(dataset)}"
        )
    samples = dataset.samples[:limit] if limit else dataset.samples

    config = RunConfig(
        task=task.name,
        model=model,
        model_args=args,
        generate=task.generate,
        positive_class=task.positive_class,
        limit=limit,
        max_concurrency=max_concurrency or provider_cls.max_concurrency,
        timeout_s=timeout_s,
        retries=retries,
        system_prompt=task.system_prompt,
        task_spec=task.spec,
    )

    # Build the provider first so bad model args fail before any run is recorded.
    provider: Provider = provider_cls(model_name, **args)
    ctx = SolverContext(provider=provider, config=task.generate)

    if resume_run_id:
        existing = store.get_run(resume_run_id)
        if existing is None:
            raise KeyError(f"no run '{resume_run_id}' to resume")
        if existing["model"] != model or existing["task"] != task.name:
            raise ValueError("resume target is a different task or model")
        if existing["dataset_hash"] != dataset.content_hash:
            raise ValueError("dataset changed since the original run; refusing to resume")
        run_id = resume_run_id
        store.set_status(run_id, "running")
        done = store.completed_sample_ids(run_id)
    else:
        run_id = store.create_run(config, provider_name, dataset.content_hash)
        done = set()

    todo = [s for s in samples if s.id not in done]
    sem = asyncio.Semaphore(config.max_concurrency)
    finished = len(done)
    total = len(samples)
    if on_start:
        on_start(run_id, total)

    async def process(sample: Sample) -> None:
        nonlocal finished
        last_error = "unknown error"
        record: SampleRecord | None = None
        for attempt in range(retries + 1):
            try:
                async with sem:
                    record = await _score_sample(task, sample, ctx, timeout_s)
                break
            except asyncio.CancelledError:
                raise
            except Exception as e:  # provider failure, timeout, scorer bug: retry then record
                last_error = f"{type(e).__name__}: {e}"
                if attempt < retries:
                    await asyncio.sleep(min(0.5 * 2**attempt, 5.0))
        if record is None:
            record = SampleRecord(sample_id=sample.id, input=sample.input, target=sample.target, error=last_error)
        store.add_sample(run_id, record)
        finished += 1
        if on_progress:
            on_progress(finished, total, record)

    try:
        await provider.setup()  # weights, warm-up, server check: not part of timed work
        t0 = time.perf_counter()
        await asyncio.gather(*(process(s) for s in todo))
        wall = time.perf_counter() - t0
    except BaseException as e:
        store.set_status(run_id, "interrupted" if isinstance(e, (asyncio.CancelledError, KeyboardInterrupt)) else "failed")
        raise
    finally:
        await provider.close()

    rows = store.rows_for_metrics(run_id)
    metrics = compute_metrics(rows, task.positive_class, wall_time_s=wall)
    counts = store.count_samples(run_id)
    metrics["errors"] = float(counts["errors"])
    store.set_metrics(run_id, metrics)
    store.set_status(run_id, "completed" if counts["errors"] == 0 else "completed_with_errors", finished=True)
    return run_id

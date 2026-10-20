"""Async runner: executes a task against one model and persists results as it goes.

The evaluation standard is applied here, the same for every provider and scoring mode:
- the dataset's content hash is stored on the run;
- generation is deterministic by default (temperature 0 and a fixed seed) unless the task
  or the caller sets otherwise;
- the prompt templates, scorer configuration, provider settings, model version facts and
  Owl version are stored on the run;
- headline metrics are stored with their standard errors and 95% intervals;
- unparseable answers are kept and count as wrong;
- latency is measured by the provider around its own call, after setup and warm-up.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from typing import Any

from owl.metrics import compute_metrics, headline_metrics
from owl.providers.base import Provider, get_provider_class, split_model_string
from owl.solvers import SolverContext, initial_state
from owl.store import SampleRecord, Store
from owl.tasks import Task, get_task
from owl.types import RunConfig, Sample

ProgressFn = Callable[[int, int, SampleRecord], None]
StartFn = Callable[[str, int], None]  # called with the run id and sample count as soon as the run exists

DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42


class DatasetSizeError(RuntimeError):
    pass


async def _score_sample(task: Task, sample: Sample, ctx: SolverContext, timeout_s: float) -> SampleRecord:
    state = initial_state(sample)
    state = await asyncio.wait_for(task.solver(state, ctx), timeout=timeout_s)
    if state.output is None:
        raise RuntimeError("solver finished without producing a model output")
    out = state.output
    score = task.scorer(sample, out)
    if inspect.isawaitable(score):  # judge scorers call a model
        score = await score
    sample_meta = {k: v for k, v in sample.metadata.items() if k != "row"}
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
            "sample": sample_meta,
            "prompt": [m.model_dump(exclude={"metadata"}) for m in state.messages if m.role != "assistant"],
            "usage": out.usage.model_dump(),
            "raw": out.raw,
        },
    )


def resolve_model_args(task: Task, provider_name: str, provider_cls: type[Provider], model_args: dict[str, Any] | None) -> dict[str, Any]:
    """Task defaults, then values the provider asked for via `task_inputs`, then the user's args."""
    args = dict(task.provider_args.get(provider_name, {}))
    for name in provider_cls.task_inputs:
        value = task.task_input(name)
        if value is not None:
            args[name] = value
    args.update(model_args or {})
    return args


def _scorer_config(task: Task) -> dict[str, Any]:
    config = dict(task.scorer_config)
    for resource in task.resources:  # a judge reports the model version it found during setup
        if hasattr(resource, "describe"):
            config.update(resource.describe())
    if task.labels:
        config.setdefault("labels", task.labels)
    return config


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
    generate_overrides: dict[str, Any] | None = None,
) -> str:
    """Run `task` on `model` ("provider/model"). Returns the run id.

    Every sample is written to the store the moment it finishes. Passing
    `resume_run_id` continues an interrupted run, skipping samples that already
    have a result (samples that errored are retried). `generate_overrides` sets
    temperature, max_tokens or seed for this run.
    """
    if isinstance(task, str):
        task = get_task(task)
    provider_name, model_name = split_model_string(model)
    provider_cls = get_provider_class(provider_name)
    args = resolve_model_args(task, provider_name, provider_cls, model_args)

    gen = task.generate.model_copy()
    if gen.temperature is None:
        gen.temperature = DEFAULT_TEMPERATURE
    if gen.seed is None:
        gen.seed = DEFAULT_SEED
    for key, value in (generate_overrides or {}).items():
        if value is not None:
            setattr(gen, key, value)

    dataset = task.load_dataset()
    if task.expected_size is not None and len(dataset) != task.expected_size:
        raise DatasetSizeError(
            f"task '{task.name}' expects {task.expected_size} samples but the dataset has {len(dataset)}"
        )
    samples = dataset.samples[:limit] if limit else dataset.samples

    if resume_run_id:
        existing = store.get_run(resume_run_id)
        if existing is None:
            raise KeyError(f"no run '{resume_run_id}' to resume")
        if existing["model"] != model or existing["task"] != task.name:
            raise ValueError("resume target is a different task or model")
        if existing["dataset_hash"] != dataset.content_hash:
            raise ValueError("dataset changed since the original run; refusing to resume")

    # Build the provider first so bad model args fail before any run is recorded.
    provider: Provider = provider_cls(model_name, **args)
    ctx = SolverContext(provider=provider, config=gen)

    run_id: str | None = None
    ready: list[Any] = []  # resources that finished setup and need closing
    try:
        # Setup and warm-up (weights, server checks, the judge) are not part of the timed run.
        await provider.setup()
        for resource in task.resources:
            await resource.setup()
            ready.append(resource)

        config = RunConfig(
            task=task.name,
            model=model,
            model_args=args,
            generate=gen,
            positive_class=task.positive_class,
            limit=limit,
            max_concurrency=max_concurrency or provider_cls.max_concurrency,
            timeout_s=timeout_s,
            retries=retries,
            system_prompt=task.system_prompt,
            task_spec=task.spec,
            scoring_mode=task.mode,
            prompt=task.prompt or ({"system": task.system_prompt, "user": "{input}"} if task.system_prompt else {"user": "{input}"}),
            scorer=_scorer_config(task),
            model_info=provider.model_info(),
        )
        if resume_run_id:
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
                    last_error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                    if attempt < retries:
                        await asyncio.sleep(min(0.5 * 2**attempt, 5.0))
            if record is None:
                record = SampleRecord(sample_id=sample.id, input=sample.input, target=sample.target, error=last_error)
            store.add_sample(run_id, record)  # type: ignore[arg-type]
            finished += 1
            if on_progress:
                on_progress(finished, total, record)

        t0 = time.perf_counter()
        await asyncio.gather(*(process(s) for s in todo))
        wall = time.perf_counter() - t0

        # Some providers only learn the exact model version from their responses.
        info = provider.model_info()
        scorer_config = _scorer_config(task)
        if info != config.model_info or scorer_config != config.scorer:
            store.update_config(run_id, config.model_copy(update={"model_info": info, "scorer": scorer_config}))
    except BaseException as e:
        if run_id is not None:
            store.set_status(run_id, "interrupted" if isinstance(e, (asyncio.CancelledError, KeyboardInterrupt)) else "failed")
        raise
    finally:
        for resource in reversed(ready):
            await resource.close()
        await provider.close()

    rows = store.rows_for_metrics(run_id)
    metrics = compute_metrics(rows, task.positive_class, wall_time_s=wall)
    metrics.update(headline_metrics(rows, task.mode, task.positive_class, task.labels))
    counts = store.count_samples(run_id)
    metrics["errors"] = float(counts["errors"])
    store.set_metrics(run_id, metrics)
    store.set_status(run_id, "completed" if counts["errors"] == 0 else "completed_with_errors", finished=True)
    return run_id

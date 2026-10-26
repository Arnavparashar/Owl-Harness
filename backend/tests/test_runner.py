import pytest

from conftest import TEXTS, make_task
from owl.runner import DatasetSizeError, run_task


async def test_full_run_with_mock_provider(store, toy_task):
    seen = []
    run_id = await run_task(toy_task, "mock/keyword", store=store, on_progress=lambda d, t, r: seen.append((d, t)))
    run = store.get_run(run_id)
    assert run["status"] == "completed" and run["finished_at"] is not None
    assert run["provider"] == "mock" and run["config"]["positive_class"] == "1"
    m = store.get_metrics(run_id)
    assert m["n"] == len(TEXTS) and m["accuracy"] == 1.0 and m["f1"] == 1.0
    assert m["tp"] == 6 and m["tn"] == 6 and m["unparseable"] == 0 and m["errors"] == 0
    assert m["latency_median_ms"] > 0 and m["wall_time_s"] >= 0
    assert sorted(d for d, _ in seen) == list(range(1, len(TEXTS) + 1))
    sample = store.all_samples(run_id)[0]
    assert sample["output"] == "1" and sample["probability"] is not None
    assert sample["metadata"]["explanation"].startswith("pred=1")


async def test_models_can_be_run_one_after_another(store, toy_task):
    a = await run_task(toy_task, "mock/always1", store=store)
    b = await run_task(toy_task, "mock/always0", store=store)
    assert store.get_metrics(a)["recall"] == 1.0 and store.get_metrics(a)["accuracy"] == 0.5
    assert store.get_metrics(b)["recall"] == 0.0
    assert len(store.list_runs()) == 2


async def test_think_blocks_and_garbage_are_scored(store, toy_task):
    t = await run_task(toy_task, "mock/think", store=store)
    assert store.get_metrics(t)["accuracy"] == 1.0
    g = await run_task(toy_task, "mock/garbage", store=store)
    m = store.get_metrics(g)
    assert m["unparseable"] == len(TEXTS) and m["accuracy"] == 0.0
    assert m["fn"] == 6 and m["fp"] == 6  # unparseable counted as the wrong class


async def test_limit(store, toy_task):
    run_id = await run_task(toy_task, "mock/keyword", store=store, limit=5)
    assert store.get_metrics(run_id)["n"] == 5


async def test_resume_after_partial_run(store, toy_task):
    # an earlier run that only got through the first 5 samples
    run_id = await run_task(toy_task, "mock/keyword", store=store, limit=5)
    store.set_status(run_id, "interrupted")
    assert len(store.completed_sample_ids(run_id)) == 5

    resumed = await run_task(toy_task, "mock/keyword", store=store, resume_run_id=run_id)
    assert resumed == run_id
    assert len(store.list_runs()) == 1
    assert store.get_run(run_id)["status"] == "completed"
    assert store.get_metrics(run_id)["n"] == len(TEXTS)
    assert len(store.all_samples(run_id)) == len(TEXTS)


async def test_resume_skips_finished_samples(store, toy_task):
    run_id = await run_task(toy_task, "mock/keyword", store=store, limit=4)
    before = {s["sample_id"]: s for s in store.all_samples(run_id)}
    calls: list[str] = []
    await run_task(
        toy_task, "mock/keyword", store=store, resume_run_id=run_id, on_progress=lambda d, t, r: calls.append(r.sample_id)
    )
    assert sorted(calls) == sorted(str(i) for i in range(4, len(TEXTS)))  # only the unfinished ones
    after = {s["sample_id"]: s for s in store.all_samples(run_id)}
    assert all(after[k] == v for k, v in before.items())


async def test_failed_samples_are_recorded_and_retried_on_resume(store, toy_task):
    run_id = await run_task(
        toy_task, "mock/keyword", store=store, model_args={"fail_if_contains": "cats"}, retries=1
    )
    run = store.get_run(run_id)
    assert run["status"] == "completed_with_errors"
    m = store.get_metrics(run_id)
    assert m["errors"] == 1 and m["n"] == len(TEXTS) - 1
    failed = [s for s in store.all_samples(run_id) if s["error"]]
    assert len(failed) == 1 and "mock failure" in failed[0]["error"]

    await run_task(toy_task, "mock/keyword", store=store, resume_run_id=run_id)
    assert store.get_run(run_id)["status"] == "completed"
    assert store.get_metrics(run_id)["errors"] == 0 and store.get_metrics(run_id)["n"] == len(TEXTS)


async def test_resume_rejects_a_different_model_or_dataset(store, toy_task):
    run_id = await run_task(toy_task, "mock/keyword", store=store, limit=3)
    with pytest.raises(ValueError, match="different"):
        await run_task(toy_task, "mock/always1", store=store, resume_run_id=run_id)
    with pytest.raises(ValueError, match="dataset changed"):
        await run_task(make_task(n=8), "mock/keyword", store=store, resume_run_id=run_id)
    with pytest.raises(KeyError):
        await run_task(toy_task, "mock/keyword", store=store, resume_run_id="nope")


async def test_bad_model_args_leave_no_run_behind(store, toy_task):
    with pytest.raises(ValueError, match="bogus"):
        await run_task(toy_task, "mock/keyword", store=store, model_args={"bogus": 1})
    assert store.list_runs() == []


async def test_expected_dataset_size_is_enforced(store):
    task = make_task(n=8)
    task.expected_size = 116
    with pytest.raises(DatasetSizeError, match="116"):
        await run_task(task, "mock/keyword", store=store)
    assert store.list_runs() == []


async def test_task_provider_args_are_defaults_that_user_args_override(store, toy_task):
    toy_task.provider_args = {"mock": {"fail_if_contains": "cats"}}
    r1 = await run_task(toy_task, "mock/keyword", store=store, retries=0)
    assert store.get_metrics(r1)["errors"] == 1
    r2 = await run_task(toy_task, "mock/keyword", store=store, model_args={"fail_if_contains": None})
    assert store.get_metrics(r2)["errors"] == 0


async def test_per_sample_timeout_is_recorded_as_error(store, toy_task, monkeypatch):
    import asyncio

    from owl.providers.mock import MockProvider

    class Slow(MockProvider):
        name = "slow"

        async def generate(self, messages, config):
            await asyncio.sleep(1)
            return await super().generate(messages, config)

    monkeypatch.setattr("owl.runner.get_provider_class", lambda name: Slow)
    run_id = await run_task(toy_task, "slow/keyword", store=store, timeout_s=0.05, retries=0, limit=2)
    errs = [s["error"] for s in store.all_samples(run_id)]
    assert len(errs) == 2 and all("Timeout" in e for e in errs)
    assert store.get_run(run_id)["status"] == "completed_with_errors"

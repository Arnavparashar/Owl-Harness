from owl.store import SampleRecord, Store
from owl.types import RunConfig


def new_run(store: Store) -> str:
    return store.create_run(RunConfig(task="t", model="mock/keyword", positive_class="1"), "mock", "hash123")


def rec(i: int, **kw) -> SampleRecord:
    base = dict(sample_id=str(i), input=f"in {i}", target="1", output="1", pred="1", correct=True)
    base.update(kw)
    return SampleRecord(**base)


def test_run_round_trip(store):
    run_id = new_run(store)
    run = store.get_run(run_id)
    assert run["task"] == "t" and run["model"] == "mock/keyword" and run["provider"] == "mock"
    assert run["status"] == "running" and run["finished_at"] is None
    assert run["dataset_hash"] == "hash123" and run["config"]["positive_class"] == "1"
    store.set_status(run_id, "completed", finished=True)
    run = store.get_run(run_id)
    assert run["status"] == "completed" and run["finished_at"] is not None
    assert [r["id"] for r in store.list_runs()] == [run_id]
    assert store.get_run("missing") is None


def test_sample_round_trip(store):
    run_id = new_run(store)
    store.add_sample(
        run_id,
        rec(0, probability=0.97, latency_ms=12.5, stop_reason="stop", metadata={"explanation": "e", "n": [1, 2]}),
    )
    (item,) = store.all_samples(run_id)
    assert item["sample_id"] == "0" and item["input"] == "in 0" and item["pred"] == "1"
    assert item["correct"] is True and item["probability"] == 0.97 and item["latency_ms"] == 12.5
    assert item["metadata"] == {"explanation": "e", "n": [1, 2]} and item["error"] is None


def test_metrics_round_trip_and_replace(store):
    run_id = new_run(store)
    store.set_metrics(run_id, {"accuracy": 0.5, "n": 4})
    assert store.get_metrics(run_id) == {"accuracy": 0.5, "n": 4.0}
    store.set_metrics(run_id, {"accuracy": 0.75})
    assert store.get_metrics(run_id) == {"accuracy": 0.75}


def test_add_sample_replaces_same_id(store):
    run_id = new_run(store)
    store.add_sample(run_id, rec(0, error="boom", output=None, pred=None, correct=False))
    store.add_sample(run_id, rec(0))
    assert store.count_samples(run_id) == {"total": 1, "errors": 0}


def test_completed_ids_exclude_errors(store):
    run_id = new_run(store)
    store.add_sample(run_id, rec(0))
    store.add_sample(run_id, rec(1, error="boom", output=None, pred=None, correct=False))
    assert store.completed_sample_ids(run_id) == {"0"}
    assert store.count_samples(run_id) == {"total": 2, "errors": 1}
    assert len(store.rows_for_metrics(run_id)) == 1  # errored samples are not scored


def test_sample_filters_and_search(store):
    run_id = new_run(store)
    store.add_sample(run_id, rec(0, target="1", pred="1", correct=True))  # TP
    store.add_sample(run_id, rec(1, target="0", pred="1", correct=False, input="ignore me"))  # FP
    store.add_sample(run_id, rec(2, target="1", pred="0", correct=False))  # FN
    store.add_sample(run_id, rec(3, target="0", pred="0", correct=True))  # TN
    store.add_sample(run_id, rec(4, target="1", pred=None, correct=False, output="hmm"))  # unparseable

    def ids(**kw):
        total, items = store.query_samples(run_id, "1", **kw)
        assert total == len(items)
        return [i["sample_id"] for i in items]

    assert ids() == ["0", "1", "2", "3", "4"]
    assert ids(filter="correct") == ["0", "3"]
    assert ids(filter="incorrect") == ["1", "2", "4"]
    assert ids(filter="false_positive") == ["1"]
    assert ids(filter="false_negative") == ["2"]
    assert ids(filter="unparseable") == ["4"]
    assert ids(q="IGNORE") == ["1"]  # case-insensitive, searches input
    assert ids(q="hmm") == ["4"]  # and output
    assert ids(filter="incorrect", q="in 2") == ["2"]


def test_pagination_orders_numerically(store):
    run_id = new_run(store)
    for i in [10, 2, 1, 11, 0]:
        store.add_sample(run_id, rec(i))
    total, page1 = store.query_samples(run_id, "1", page=1, page_size=3)
    _, page2 = store.query_samples(run_id, "1", page=2, page_size=3)
    assert total == 5
    assert [s["sample_id"] for s in page1 + page2] == ["0", "1", "2", "10", "11"]


def test_data_persists_across_connections(tmp_path):
    path = tmp_path / "owl.db"
    s1 = Store(path)
    run_id = new_run(s1)
    s1.add_sample(run_id, rec(0))
    s1.close()
    s2 = Store(path)
    assert s2.get_run(run_id) is not None
    assert s2.completed_sample_ids(run_id) == {"0"}
    s2.close()

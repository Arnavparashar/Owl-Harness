import pytest
from fastapi.testclient import TestClient

from conftest import TEXTS, make_task
from owl.api import create_app
from owl.runner import run_task


@pytest.fixture
async def seeded(store):
    task = make_task()
    ids = {
        "keyword": await run_task(task, "mock/keyword", store=store),
        "always1": await run_task(task, "mock/always1", store=store),
        "always0": await run_task(task, "mock/always0", store=store),
    }
    return ids


@pytest.fixture
def client(store, seeded, tmp_path):
    return TestClient(create_app(store, frontend_dist=tmp_path / "no-dist"))


def test_list_runs(client, seeded):
    runs = client.get("/api/runs").json()
    assert {r["id"] for r in runs} == set(seeded.values())
    r = next(r for r in runs if r["id"] == seeded["keyword"])
    assert r["model"] == "mock/keyword" and r["metrics"]["accuracy"] == 1.0 and r["status"] == "completed"


def test_run_detail(client, seeded):
    d = client.get(f"/api/runs/{seeded['keyword']}").json()
    assert d["config"]["task"] == "toy" and d["metrics"]["n"] == len(TEXTS)
    assert d["has_probabilities"] is True and d["sample_counts"] == {"total": len(TEXTS), "errors": 0}
    assert sum(b["count"] for b in d["latency_histogram"]) == len(TEXTS)
    assert client.get(f"/api/runs/{seeded['always1']}").json()["has_probabilities"] is False
    assert client.get("/api/runs/missing").status_code == 404


def test_samples_filters_search_and_pagination(client, seeded):
    rid = seeded["always1"]  # predicts 1 for everything: 6 FP, 6 TP
    get = lambda **p: client.get(f"/api/runs/{rid}/samples", params=p).json()
    assert get()["total"] == len(TEXTS)
    assert get(filter="false_positive")["total"] == 6
    assert get(filter="false_negative")["total"] == 0
    assert get(filter="correct")["total"] == 6 and get(filter="incorrect")["total"] == 6
    assert get(q="marathon")["items"][0]["input"].startswith("How do I train")
    page = get(page=2, page_size=5)
    assert [i["sample_id"] for i in page["items"]] == ["5", "6", "7", "8", "9"]
    assert client.get(f"/api/runs/{rid}/samples", params={"filter": "bogus"}).status_code == 422
    assert client.get(f"/api/runs/{seeded['always0']}/samples", params={"filter": "false_negative"}).json()["total"] == 6


def test_threshold_curve(client, seeded):
    body = client.get(f"/api/runs/{seeded['keyword']}/threshold-curve").json()
    assert body["positive_class"] == "1" and body["n"] == len(TEXTS) and len(body["points"]) == 101
    assert body["points"][0]["threshold"] == 0.0 and body["points"][-1]["threshold"] == 1.0
    at = {p["threshold"]: p for p in body["points"]}
    assert at[0.5]["f1"] == 1.0  # injections score >= 0.85, benign <= 0.14
    assert client.get(f"/api/runs/{seeded['always1']}/threshold-curve").status_code == 404


def test_compare(client, seeded):
    ids = f"{seeded['keyword']},{seeded['always1']}"
    body = client.get("/api/compare", params={"run_ids": ids}).json()
    assert [r["id"] for r in body["runs"]] == ids.split(",")
    assert body["shared_samples"] == len(TEXTS)
    # always1 disagrees with keyword on the 6 benign samples
    assert body["total_disagreements"] == 6
    d = body["disagreements"][0]
    assert d["preds"] == {seeded["keyword"]: "0", seeded["always1"]: "1"} and d["target"] == "0"
    assert d["correct"] == {seeded["keyword"]: True, seeded["always1"]: False}
    assert client.get("/api/compare", params={"run_ids": f"{ids},nope"}).status_code == 404


def test_frontend_is_served_when_built(store, seeded, tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>owl</html>")
    (dist / "assets" / "a.js").write_text("console.log(1)")
    c = TestClient(create_app(store, frontend_dist=dist))
    assert c.get("/").text == "<html>owl</html>"
    assert c.get("/runs/abc").text == "<html>owl</html>"  # client-side route falls back to index
    assert c.get("/assets/a.js").text == "console.log(1)"
    assert c.get("/api/nope").status_code == 404
    assert c.get("/../backend/pyproject.toml").text == "<html>owl</html>"


def test_summary_carries_accuracy_interval(client, seeded):
    run = next(r for r in client.get("/api/runs").json() if r["id"] == seeded["always1"])
    lo, hi = run["accuracy_ci"]  # 6 of 12 correct
    assert 0.25 < lo < 0.30 and 0.70 < hi < 0.75


def test_analysis_for_probability_run(client, seeded):
    body = client.get(f"/api/runs/{seeded['keyword']}/analysis").json()
    assert body["roc"]["auc"] == 1.0 and body["pr"]["average_precision"] == 1.0
    assert body["pr"]["baseline"] == 0.5
    dist = body["score_distribution"]
    assert sum(b["positive"] for b in dist) == 6 and sum(b["negative"] for b in dist) == 6
    assert body["calibration"]["brier"] < 0.05 and body["calibration"]["ece"] < 0.15
    assert len(body["latency_points"]) == len(TEXTS) and body["latency_points"][0]["index"] == 0
    assert body["latency_by_outcome"]["correct"]["n"] == len(TEXTS) and body["latency_by_outcome"]["incorrect"] is None
    assert len(body["length_slices"]) == 1  # 12 samples: fewer than 8 per bucket for two buckets
    assert client.get("/api/runs/missing/analysis").status_code == 404


def test_analysis_without_probabilities(client, seeded):
    body = client.get(f"/api/runs/{seeded['always1']}/analysis").json()
    assert body["roc"] is None and body["pr"] is None and body["score_distribution"] is None and body["calibration"] is None
    assert body["latency_by_outcome"]["incorrect"]["n"] == 6


def test_export_csv(client, seeded, store):
    res = client.get(f"/api/runs/{seeded['keyword']}/export.csv")
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/csv")
    assert f'owl-{seeded["keyword"]}.csv' in res.headers["content-disposition"]
    lines = res.text.strip().splitlines()
    assert lines[0].startswith("sample_id,input,target,output,pred,correct,probability,latency_ms")
    assert len(lines) == len(TEXTS) + 1
    assert client.get("/api/runs/missing/export.csv").status_code == 404


def test_export_csv_neutralises_formulas(store, tmp_path):
    from owl.api import _csv_text

    assert _csv_text("=HYPERLINK(1)") == "'=HYPERLINK(1)"
    assert _csv_text("@SUM(A1)") == "'@SUM(A1)" and _csv_text("-1+1") == "'-1+1" and _csv_text("\tx") == "'\tx"
    assert _csv_text("plain") == "plain" and _csv_text(None) is None


def test_compare_pairwise_and_coverage(client, seeded):
    k, a1, a0 = seeded["keyword"], seeded["always1"], seeded["always0"]
    body = client.get("/api/compare", params={"run_ids": f"{k},{a1},{a0}"}).json()
    assert body["sample_ids"] == [str(i) for i in range(len(TEXTS))]
    assert body["outcomes"][k] == [True] * 12
    assert sum(body["outcomes"][a1]) == 6 and sum(body["outcomes"][a0]) == 6
    # every sample is right for keyword, so nobody is wrong everywhere; always1 and always0 are never both right
    assert body["coverage"] == {"all_correct": 0, "all_wrong": 0, "any_correct": 12}
    pair = next(p for p in body["pairwise"] if (p["a"], p["b"]) == (k, a1))
    assert pair["a_only_correct"] == 6 and pair["b_only_correct"] == 0
    assert pair["agreement"] == 0.5 and pair["mcnemar_p"] == pytest.approx(2 / 64)
    assert len(body["pairwise"]) == 3
    assert set(body["roc"]) == {k}  # only the run with probabilities has a ROC curve

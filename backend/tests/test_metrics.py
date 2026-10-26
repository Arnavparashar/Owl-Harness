import math

import pytest

from owl.metrics import (
    Row,
    compute_metrics,
    confusion,
    latency_histogram,
    percentile,
    precision_recall_f1,
    threshold_curve,
)


def rows_case_a() -> list[Row]:
    # (target, pred): 2 TP, 1 FN, 1 FP, 3 TN, plus one unparseable on a negative target (-> FP)
    spec = [
        ("1", "1"), ("1", "1"), ("1", "0"),
        ("0", "1"),
        ("0", "0"), ("0", "0"), ("0", "0"),
        ("0", None),
    ]
    return [Row(target=t, pred=p, correct=(p == t)) for t, p in spec]


def test_confusion_counts_unparseable_as_wrong_class():
    c = confusion(rows_case_a(), positive="1")
    assert c == {"tp": 2, "fp": 2, "fn": 1, "tn": 3}


def test_unparseable_on_positive_target_is_false_negative():
    rows = [Row("1", None, False), Row("1", "1", True)]
    assert confusion(rows, "1") == {"tp": 1, "fp": 0, "fn": 1, "tn": 0}


def test_precision_recall_f1_hand_computed():
    p, r, f = precision_recall_f1(tp=2, fp=2, fn=1)
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(2 / 3)
    assert f == pytest.approx(4 / 7)


def test_zero_denominators_give_zero():
    assert precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)
    assert precision_recall_f1(0, 3, 2) == (0.0, 0.0, 0.0)


def test_compute_metrics_case_a():
    m = compute_metrics(rows_case_a(), positive_class="1", wall_time_s=1.5)
    assert m["n"] == 8
    assert m["accuracy"] == pytest.approx(5 / 8)
    # sample variance (n-1) of five 1s and three 0s, divided by n
    assert m["stderr"] == pytest.approx(math.sqrt(1.875 / 7 / 8))
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(4 / 7)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 2, 1, 3)
    assert m["unparseable"] == 1
    assert m["wall_time_s"] == 1.5


def test_alternative_positive_class():
    rows = [Row("0", "0", True), Row("0", "1", False), Row("1", "1", True)]
    c = confusion(rows, positive="0")
    assert c == {"tp": 1, "fp": 0, "fn": 1, "tn": 1}


def test_binary_metrics_absent_without_positive_class():
    m = compute_metrics([Row("a", "a", True)], positive_class=None)
    assert "precision" not in m and "tp" not in m
    assert m["accuracy"] == 1.0


def test_stderr_single_row_is_zero():
    assert compute_metrics([Row("1", "1", True)], "1")["stderr"] == 0.0


def test_latency_median_and_p95():
    rows = [Row("1", "1", True, latency_ms=v) for v in [1, 2, 3, 4, 5]]
    m = compute_metrics(rows, "1")
    assert m["latency_median_ms"] == 3
    assert m["latency_p95_ms"] == pytest.approx(4.8)  # 4 + 0.8 * (5 - 4)


def test_percentile_edges():
    assert percentile([], 95) == 0.0
    assert percentile([7], 95) == 7.0
    assert percentile([1, 2], 50) == pytest.approx(1.5)


def test_threshold_curve_hand_computed():
    rows = [
        Row("1", "1", True, probability=0.9),
        Row("1", "1", True, probability=0.6),
        Row("0", "0", True, probability=0.4),
        Row("0", "1", False, probability=0.7),
    ]
    curve = {pt["threshold"]: pt for pt in threshold_curve(rows, "1")}
    assert len(curve) == 101
    at0 = curve[0.0]  # everything predicted positive
    assert (at0["tp"], at0["fp"]) == (2, 2) and at0["recall"] == 1.0
    assert at0["precision"] == pytest.approx(0.5)
    at50 = curve[0.5]  # 0.9, 0.6, 0.7 positive
    assert at50["precision"] == pytest.approx(2 / 3) and at50["recall"] == 1.0
    at80 = curve[0.8]  # only 0.9 positive
    assert at80["precision"] == 1.0 and at80["recall"] == pytest.approx(0.5)
    assert at80["f1"] == pytest.approx(2 / 3)
    assert curve[1.0]["recall"] == 0.0  # nothing reaches P >= 1.0


def test_threshold_curve_includes_boundary():
    rows = [Row("1", "1", True, probability=0.5)]
    curve = {pt["threshold"]: pt for pt in threshold_curve(rows, "1")}
    assert curve[0.5]["tp"] == 1  # P >= t is positive
    assert curve[0.51]["fn"] == 1


def test_threshold_curve_ignores_rows_without_probability():
    rows = [Row("1", "1", True), Row("1", "1", True, probability=0.9)]
    assert threshold_curve(rows, "1")[0]["tp"] == 1


def test_latency_histogram():
    h = latency_histogram([0, 5, 10], bins=2)
    assert [b["count"] for b in h] == [1, 2]
    assert latency_histogram([]) == []
    assert latency_histogram([3, 3])[0]["count"] == 2


# --- interval estimates, curves and significance ---------------------------------

from owl.metrics import (  # noqa: E402
    accuracy_by_length,
    calibration,
    mcnemar_exact,
    pr_curve,
    roc_curve,
    score_distribution,
    wilson_interval,
)


def _prob_rows(pairs):
    """(target, P(positive)) pairs -> rows; the prediction is irrelevant to the curves."""
    return [Row(target=t, pred=None, correct=False, probability=p) for t, p in pairs]


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(8, 10)  # textbook value: (0.490, 0.943)
    assert (round(lo, 3), round(hi, 3)) == (0.490, 0.943)
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and 0.72 < lo < 0.73
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.27 < hi < 0.28


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)  # 2 * C(10,0) / 2**10
    assert mcnemar_exact(3, 1) == pytest.approx(2 * (1 + 4) / 16)  # n=4, min=1 -> 0.625
    assert mcnemar_exact(10, 0) == mcnemar_exact(0, 10)


def test_roc_perfect_random_and_inverted():
    perfect = roc_curve(_prob_rows([("1", 0.9), ("1", 0.8), ("0", 0.2), ("0", 0.1)]), "1")
    assert perfect["auc"] == 1.0 and perfect["points"][0] == {"fpr": 0.0, "tpr": 0.0, "threshold": None}
    inverted = roc_curve(_prob_rows([("1", 0.1), ("1", 0.2), ("0", 0.8), ("0", 0.9)]), "1")
    assert inverted["auc"] == 0.0
    tied = roc_curve(_prob_rows([("1", 0.5), ("0", 0.5)]), "1")  # one tie group -> straight diagonal
    assert tied["auc"] == 0.5 and len(tied["points"]) == 2


def test_roc_hand_computed_auc():
    # positives 0.9, 0.4; negatives 0.6, 0.2. Pairs where positive outranks negative: 3 of 4.
    rows = _prob_rows([("1", 0.9), ("0", 0.6), ("1", 0.4), ("0", 0.2)])
    assert roc_curve(rows, "1")["auc"] == pytest.approx(0.75)
    assert roc_curve(_prob_rows([("1", 0.9), ("1", 0.4)]), "1") is None  # no negatives


def test_pr_curve_average_precision():
    # ranking: pos, neg, pos -> AP = 0.5 * 1.0 + 0.5 * (2/3)
    rows = _prob_rows([("1", 0.9), ("0", 0.6), ("1", 0.4)])
    pr = pr_curve(rows, "1")
    assert pr["average_precision"] == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert pr["baseline"] == pytest.approx(2 / 3)
    assert [round(p["precision"], 3) for p in pr["points"]] == [1.0, 0.5, 0.667]
    assert pr_curve(_prob_rows([("0", 0.3)]), "1") is None


def test_score_distribution_bins_and_edges():
    rows = _prob_rows([("1", 0.0), ("1", 1.0), ("0", 0.04), ("0", 0.5), ("1", 0.95)])
    bins = score_distribution(rows, "1", bins=20)
    assert len(bins) == 20 and sum(b["positive"] + b["negative"] for b in bins) == 5
    assert bins[0] == {"lo": 0.0, "hi": 0.05, "positive": 1.0, "negative": 1.0}
    assert bins[19]["positive"] == 2.0  # 0.95 and 1.0 both land in the last bin
    assert bins[10]["negative"] == 1.0
    assert score_distribution([], "1") == []


def test_calibration_brier_and_ece():
    rows = _prob_rows([("1", 1.0), ("0", 0.0), ("1", 0.5), ("0", 0.5)])
    cal = calibration(rows, "1")
    assert cal["brier"] == pytest.approx((0 + 0 + 0.25 + 0.25) / 4)
    assert cal["ece"] == pytest.approx(0.0)  # the 0.5 bin holds one positive of two
    assert [b["n"] for b in cal["bins"]] == [1.0, 2.0, 1.0]
    off = calibration(_prob_rows([("0", 0.9), ("0", 0.9)]), "1")
    assert off["ece"] == pytest.approx(0.9)


def test_accuracy_by_length_buckets():
    pairs = [(i, i >= 8) for i in range(16)]  # short half wrong, long half right
    out = accuracy_by_length(pairs, buckets=2)
    assert [b["accuracy"] for b in out] == [0.0, 1.0]
    assert out[0]["min_chars"] == 0 and out[0]["max_chars"] == 7 and out[1]["n"] == 8
    assert len(accuracy_by_length(pairs, buckets=4)) == 2  # at least 8 samples per bucket
    assert len(accuracy_by_length([(1, True)] * 3)) == 1
    assert accuracy_by_length([]) == []

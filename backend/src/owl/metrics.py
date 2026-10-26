"""Metric computation over scored sample rows.

Conventions
- `stderr` is the standard error of the mean accuracy: sample std (n-1) / sqrt(n).
- Precision, recall and F1 return 0.0 when their denominator is zero.
- An unparseable prediction counts as the wrong class: a false negative when the
  target is the positive class, a false positive otherwise. This keeps the
  confusion matrix consistent with accuracy.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Row:
    target: str
    pred: str | None
    correct: bool
    probability: float | None = None
    latency_ms: float | None = None


def other_class(positive: str) -> str:
    """The negative label of a 0/1 task."""
    return "0" if positive == "1" else "1"


def confusion(rows: Sequence[Row], positive: str) -> dict[str, int]:
    tp = fp = fn = tn = 0
    for r in rows:
        actual_pos = r.target == positive
        if r.pred is None:  # unparseable: counted as the wrong class
            if actual_pos:
                fn += 1
            else:
                fp += 1
            continue
        pred_pos = r.pred == positive
        if pred_pos and actual_pos:
            tp += 1
        elif pred_pos:
            fp += 1
        elif actual_pos:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return precision, recall, f1


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, `q` in [0, 100]."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q / 100
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def accuracy_with_stderr(rows: Sequence[Row]) -> tuple[float, float]:
    n = len(rows)
    if n == 0:
        return 0.0, 0.0
    values = [1.0 if r.correct else 0.0 for r in rows]
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def compute_metrics(
    rows: Sequence[Row],
    positive_class: str | None,
    wall_time_s: float | None = None,
) -> dict[str, float]:
    """All run-level metrics. Binary metrics appear only when a positive class is set."""
    acc, se = accuracy_with_stderr(rows)
    out: dict[str, float] = {"n": float(len(rows)), "accuracy": acc, "stderr": se}

    if positive_class is not None:
        c = confusion(rows, positive_class)
        p, r, f = precision_recall_f1(c["tp"], c["fp"], c["fn"])
        out.update({"precision": p, "recall": r, "f1": f})
        out.update({k: float(v) for k, v in c.items()})

    out["unparseable"] = float(sum(1 for r in rows if r.pred is None))

    latencies = [r.latency_ms for r in rows if r.latency_ms is not None]
    out["latency_median_ms"] = float(statistics.median(latencies)) if latencies else 0.0
    out["latency_p95_ms"] = percentile(latencies, 95)
    if wall_time_s is not None:
        out["wall_time_s"] = wall_time_s
    return out


def threshold_curve(rows: Sequence[Row], positive_class: str, step: float = 0.01) -> list[dict[str, float]]:
    """Precision/recall/F1 (and counts) when predicting positive iff P(positive) >= t.

    Only rows that carry a probability take part. Thresholds run 0.00..1.00.
    """
    scored = [r for r in rows if r.probability is not None]
    steps = round(1 / step)
    curve: list[dict[str, float]] = []
    for i in range(steps + 1):
        t = round(i * step, 10)
        re_rows = [
            Row(
                target=r.target,
                pred=positive_class if r.probability >= t else other_class(positive_class),  # type: ignore[operator]
                correct=False,
            )
            for r in scored
        ]
        c = confusion(re_rows, positive_class)
        p, rc, f = precision_recall_f1(c["tp"], c["fp"], c["fn"])
        n = len(re_rows)
        curve.append(
            {
                "threshold": round(t, 2),
                "precision": p,
                "recall": rc,
                "f1": f,
                "accuracy": (c["tp"] + c["tn"]) / n if n else 0.0,
                **{k: float(v) for k, v in c.items()},
            }
        )
    return curve


def latency_histogram(values: Sequence[float], bins: int = 20) -> list[dict[str, float]]:
    """Equal-width histogram: [{lo, hi, count}, ...]."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [{"lo": lo, "hi": hi, "count": float(len(values))}]
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        counts[min(int((v - lo) / width), bins - 1)] += 1
    return [{"lo": lo + i * width, "hi": lo + (i + 1) * width, "count": float(c)} for i, c in enumerate(counts)]


# --- interval estimates and significance ---------------------------------------


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (95% by default). (0, 0) when n is 0."""
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts of two paired classifiers.

    `a_only` is the number of samples only A got right, `b_only` those only B got right.
    Under the null hypothesis each discordant sample is equally likely to go either way.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


# --- analysis of rows that carry P(positive) -----------------------------------


def _scored(rows: Sequence[Row], positive: str) -> list[tuple[float, bool]]:
    return [(r.probability, r.target == positive) for r in rows if r.probability is not None]


def roc_curve(rows: Sequence[Row], positive: str) -> dict | None:
    """ROC points and AUC (trapezoid rule, ties grouped). None unless both classes are present."""
    scored = sorted(_scored(rows, positive), key=lambda t: -t[0])
    pos = sum(1 for _, y in scored if y)
    neg = len(scored) - pos
    if pos == 0 or neg == 0:
        return None
    points: list[dict[str, float | None]] = [{"fpr": 0.0, "tpr": 0.0, "threshold": None}]
    tp = fp = 0
    i = 0
    while i < len(scored):
        p = scored[i][0]
        while i < len(scored) and scored[i][0] == p:
            if scored[i][1]:
                tp += 1
            else:
                fp += 1
            i += 1
        points.append({"fpr": fp / neg, "tpr": tp / pos, "threshold": p})
    auc = sum(
        (points[k]["fpr"] - points[k - 1]["fpr"]) * (points[k]["tpr"] + points[k - 1]["tpr"]) / 2  # type: ignore[operator]
        for k in range(1, len(points))
    )
    return {"auc": auc, "points": points}


def pr_curve(rows: Sequence[Row], positive: str) -> dict | None:
    """Precision/recall at each distinct score, average precision, and the prevalence baseline."""
    scored = sorted(_scored(rows, positive), key=lambda t: -t[0])
    pos = sum(1 for _, y in scored if y)
    if pos == 0:
        return None
    points: list[dict[str, float]] = []
    tp = fp = 0
    prev_recall = 0.0
    average_precision = 0.0
    i = 0
    while i < len(scored):
        p = scored[i][0]
        while i < len(scored) and scored[i][0] == p:
            if scored[i][1]:
                tp += 1
            else:
                fp += 1
            i += 1
        precision, recall = tp / (tp + fp), tp / pos
        average_precision += (recall - prev_recall) * precision
        prev_recall = recall
        points.append({"recall": recall, "precision": precision, "threshold": p})
    return {"average_precision": average_precision, "baseline": pos / len(scored), "points": points}


def score_distribution(rows: Sequence[Row], positive: str, bins: int = 20) -> list[dict[str, float]]:
    """Histogram of P(positive), split by the true class: [{lo, hi, positive, negative}, ...]."""
    scored = _scored(rows, positive)
    if not scored:
        return []
    out = [{"lo": i / bins, "hi": (i + 1) / bins, "positive": 0.0, "negative": 0.0} for i in range(bins)]
    for p, y in scored:
        out[min(int(min(max(p, 0.0), 1.0) * bins), bins - 1)]["positive" if y else "negative"] += 1
    return out


def calibration(rows: Sequence[Row], positive: str, bins: int = 10) -> dict | None:
    """Reliability bins (non-empty only), Brier score and expected calibration error."""
    scored = _scored(rows, positive)
    if not scored:
        return None
    groups: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for p, y in scored:
        groups[min(int(min(max(p, 0.0), 1.0) * bins), bins - 1)].append((p, y))
    n = len(scored)
    out = []
    ece = 0.0
    for g in groups:
        if not g:
            continue
        mean_p = sum(p for p, _ in g) / len(g)
        frac = sum(1 for _, y in g if y) / len(g)
        ece += len(g) / n * abs(mean_p - frac)
        out.append({"mean_probability": mean_p, "fraction_positive": frac, "n": float(len(g))})
    brier = sum((p - (1.0 if y else 0.0)) ** 2 for p, y in scored) / n
    return {"brier": brier, "ece": ece, "bins": out}


def accuracy_by_length(pairs: Sequence[tuple[int, bool]], buckets: int = 4) -> list[dict[str, float]]:
    """Accuracy per input-length bucket. Buckets hold equal numbers of samples (shortest first)."""
    n = len(pairs)
    buckets = max(1, min(buckets, n // 8))  # keep at least 8 samples per bucket
    if n == 0:
        return []
    ordered = sorted(pairs, key=lambda t: t[0])
    out = []
    for b in range(buckets):
        chunk = ordered[b * n // buckets : (b + 1) * n // buckets]
        correct = sum(1 for _, ok in chunk if ok)
        lo, hi = wilson_interval(correct, len(chunk))
        out.append(
            {
                "min_chars": float(chunk[0][0]),
                "max_chars": float(chunk[-1][0]),
                "n": float(len(chunk)),
                "accuracy": correct / len(chunk),
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return out


# --- multi-class -------------------------------------------------------------------

UNPARSEABLE = "(unparseable)"


def multiclass_report(rows: Sequence[Row], labels: Sequence[str]) -> dict:
    """Per-class precision, recall, F1 and support, macro and weighted F1, and the confusion matrix.

    Labels compare case-insensitively. An unparseable answer is wrong: it is a false negative
    for its true class and a false positive for no class, and it gets its own matrix column.
    Rows whose target is not in `labels` still count toward accuracy but have no class row.
    """
    canon = {l.casefold(): l for l in labels}

    def key(v: str | None) -> str | None:
        return None if v is None else canon.get(v.casefold(), v)

    tp = {l: 0 for l in labels}
    fp = {l: 0 for l in labels}
    fn = {l: 0 for l in labels}
    columns = [*labels, UNPARSEABLE]
    matrix = {l: {c: 0 for c in columns} for l in labels}
    for r in rows:
        t, p = key(r.target), key(r.pred)
        if t in matrix:
            matrix[t][p if p in matrix[t] else UNPARSEABLE] += 1
        if p is not None and p == t and t in tp:
            tp[t] += 1
            continue
        if t in fn:
            fn[t] += 1
        if p is not None and p in fp:
            fp[p] += 1

    per_class = []
    for l in labels:
        precision, recall, f1 = precision_recall_f1(tp[l], fp[l], fn[l])
        per_class.append(
            {"label": l, "precision": precision, "recall": recall, "f1": f1, "support": tp[l] + fn[l]}
        )
    support = sum(c["support"] for c in per_class)
    macro = sum(c["f1"] for c in per_class) / len(per_class) if per_class else 0.0
    weighted = sum(c["f1"] * c["support"] for c in per_class) / support if support else 0.0
    return {
        "labels": list(labels),
        "per_class": per_class,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "matrix": [[matrix[l][c] for c in columns] for l in labels],
        "matrix_columns": columns,
    }


# --- uncertainty for metrics without a closed form ------------------------------------


def bootstrap(
    rows: Sequence[Row],
    statistic,  # Callable[[Sequence[Row]], float]
    resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(stderr, ci_low, ci_high) of `statistic` by the percentile bootstrap.

    The generator is seeded, so the same rows always give the same interval.
    """
    import random

    n = len(rows)
    if n < 2:
        v = statistic(rows) if n else 0.0
        return 0.0, v, v
    rng = random.Random(seed)
    values = sorted(statistic([rows[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))
    mean = sum(values) / resamples
    se = math.sqrt(sum((v - mean) ** 2 for v in values) / (resamples - 1))
    return se, percentile(values, 2.5), percentile(values, 97.5)


def mean_with_ci(values: Sequence[float], z: float = 1.96) -> tuple[float, float, float, float]:
    """(mean, stderr, ci_low, ci_high) with the normal approximation; stderr uses n-1."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0, mean, mean
    se = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1) / n)
    return mean, se, mean - z * se, mean + z * se


def judge_scores(rows: Sequence[Row], kind: str) -> list[float]:
    """Numeric judge scores. An unparseable verdict counts as the worst score (0 or 1)."""
    worst = 0.0 if kind == "passfail" else 1.0
    out = []
    for r in rows:
        try:
            out.append(float(r.pred) if r.pred is not None else worst)
        except ValueError:
            out.append(worst)
    return out


def headline_metrics(
    rows: Sequence[Row],
    mode: str,
    positive_class: str | None = None,
    labels: Sequence[str] | None = None,
) -> dict[str, float]:
    """The headline numbers for a scoring mode, each with stderr and a 95% interval.

    accuracy (every mode except the 1-5 scale): Wilson interval.
    F1 (binary) and macro-F1 (multi-class): percentile bootstrap.
    mean score (1-5 judge): normal approximation.
    """
    out: dict[str, float] = {}
    n = len(rows)
    correct = sum(1 for r in rows if r.correct)
    lo, hi = wilson_interval(correct, n)
    out["accuracy_ci_low"], out["accuracy_ci_high"] = lo, hi

    if mode == "binary" and positive_class is not None:

        def f1(rs: Sequence[Row]) -> float:
            c = confusion(rs, positive_class)
            return precision_recall_f1(c["tp"], c["fp"], c["fn"])[2]

        out["f1_stderr"], out["f1_ci_low"], out["f1_ci_high"] = bootstrap(rows, f1)
    elif mode == "multiclass" and labels:
        report = multiclass_report(rows, labels)
        out["macro_f1"] = report["macro_f1"]
        out["weighted_f1"] = report["weighted_f1"]
        se, a, b = bootstrap(rows, lambda rs: multiclass_report(rs, labels)["macro_f1"])
        out["macro_f1_stderr"], out["macro_f1_ci_low"], out["macro_f1_ci_high"] = se, a, b
    elif mode == "judge_passfail":
        out["pass_rate"] = correct / n if n else 0.0
    elif mode == "judge_scale":
        mean, se, a, b = mean_with_ci(judge_scores(rows, "scale"))
        out["mean_score"], out["mean_score_stderr"] = mean, se
        out["mean_score_ci_low"], out["mean_score_ci_high"] = max(1.0, a), min(5.0, b)
    return out

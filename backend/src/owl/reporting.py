"""How a run is summarised: its scoring mode and its headline numbers.

Runs recorded before v0.2 have no stored scoring mode or intervals; this module derives them
from what those runs did store, so old and new runs are reported the same way.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from owl.metrics import Row, bootstrap, confusion, multiclass_report, precision_recall_f1, wilson_interval

MAIN_LABELS = {
    "binary": "Accuracy",
    "multiclass": "Accuracy",
    "exact_match": "Accuracy",
    "judge_passfail": "Pass rate",
    "judge_scale": "Mean score",
}


def scoring_mode(run: dict[str, Any]) -> str:
    config = run.get("config", {})
    mode = config.get("scoring_mode")
    if mode:
        return mode
    return "binary" if config.get("positive_class") is not None else "exact_match"


def labels_of(run: dict[str, Any]) -> list[str] | None:
    config = run.get("config", {})
    labels = (config.get("scorer") or {}).get("labels")
    if labels:
        return list(labels)
    spec = config.get("task_spec") or {}
    return spec.get("labels")


def _stat(key: str, label: str, m: dict[str, float], value: float | None = None) -> dict[str, Any] | None:
    v = m.get(key) if value is None else value
    if v is None:
        return None
    return {
        "key": key,
        "label": label,
        "value": v,
        "stderr": m.get(f"{key}_stderr"),
        "ci_low": m.get(f"{key}_ci_low"),
        "ci_high": m.get(f"{key}_ci_high"),
    }


def headline(run: dict[str, Any], m: dict[str, float]) -> dict[str, Any]:
    """{"mode", "main", "secondary"}: the main metric for the scoring mode and, where it applies,
    F1 (binary) or macro-F1 (multi-class). Each has value, stderr and a 95% interval."""
    mode = scoring_mode(run)
    n = int(m.get("n", 0))
    main: dict[str, Any] | None
    if mode == "judge_scale" and "mean_score" in m:
        main = _stat("mean_score", "Mean score", m)
        if main:
            main["range"] = [1, 5]
    else:
        acc = m.get("accuracy")
        if acc is None:
            main = None
        else:
            lo, hi = m.get("accuracy_ci_low"), m.get("accuracy_ci_high")
            if lo is None or hi is None:  # recorded before intervals were stored
                lo, hi = wilson_interval(round(acc * n), n) if n else (acc, acc)
            main = {
                "key": "pass_rate" if mode == "judge_passfail" else "accuracy",
                "label": MAIN_LABELS.get(mode, "Accuracy"),
                "value": m.get("pass_rate", acc) if mode == "judge_passfail" else acc,
                "stderr": m.get("stderr"),
                "ci_low": lo,
                "ci_high": hi,
                "range": [0, 1],
            }
    secondary = None
    if mode == "binary" and "f1" in m:
        secondary = _stat("f1", "F1", m)
    elif mode == "multiclass" and "macro_f1" in m:
        secondary = _stat("macro_f1", "Macro-F1", m)
    return {"mode": mode, "main": main, "secondary": secondary}


def fill_secondary_interval(run: dict[str, Any], head: dict[str, Any], rows: Sequence[Row]) -> None:
    """Compute a missing F1 / macro-F1 interval for runs recorded before intervals were stored."""
    sec = head.get("secondary")
    mode = head["mode"]
    if mode == "binary":
        positive = run["config"].get("positive_class")
        if positive is None or (sec and sec.get("stderr") is not None):
            return

        def f1(rs: Sequence[Row]) -> float:
            c = confusion(rs, positive)
            return precision_recall_f1(c["tp"], c["fp"], c["fn"])[2]

        se, lo, hi = bootstrap(rows, f1)
        head["secondary"] = {"key": "f1", "label": "F1", "value": f1(rows), "stderr": se, "ci_low": lo, "ci_high": hi}
    elif mode == "multiclass":
        labels = labels_of(run)
        if not labels or (sec and sec.get("stderr") is not None):
            return
        report = multiclass_report(rows, labels)
        se, lo, hi = bootstrap(rows, lambda rs: multiclass_report(rs, labels)["macro_f1"])
        head["secondary"] = {"key": "macro_f1", "label": "Macro-F1", "value": report["macro_f1"], "stderr": se, "ci_low": lo, "ci_high": hi}

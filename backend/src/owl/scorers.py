"""Scorers: turn a model output into a `Score`."""

from __future__ import annotations

import re
from collections.abc import Callable

from owl.types import ModelOutput, Sample, Score

Scorer = Callable[[Sample, ModelOutput], Score]

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_OPEN = re.compile(r"<think>.*", re.DOTALL)  # block cut off before it closed
_BINARY_DIGIT = re.compile(r"[01]")


def strip_think(text: str) -> str:
    """Remove `<think>...</think>` blocks, including one that never closed."""
    return _THINK_OPEN.sub("", _THINK_BLOCK.sub("", text))


def binary_label(positive: str = "1") -> Scorer:
    """Score a 0/1 classification.

    The prediction is the last `0` or `1` character left after think blocks are
    removed. If there is none the sample is incorrect and flagged `unparseable`.
    A probability on the output is carried into the score metadata so threshold
    analysis can use it.
    """
    if positive not in ("0", "1"):
        raise ValueError("binary_label positive class must be '0' or '1'")

    def score(sample: Sample, output: ModelOutput) -> Score:
        found = _BINARY_DIGIT.findall(strip_think(output.text))
        pred = found[-1] if found else None
        unparseable = pred is None
        metadata: dict = {"unparseable": unparseable}
        if output.probability is not None:
            metadata["probability"] = output.probability
        explanation = f"pred={pred if pred is not None else 'unparseable'} true={sample.target}"
        if output.probability is not None:
            explanation += f" P(positive)={output.probability:.4f}"
        return Score(
            value=1.0 if pred == sample.target else 0.0,
            pred=pred,
            explanation=explanation,
            metadata=metadata,
        )

    return score


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize_text(text: str, *, case: bool = True, whitespace: bool = True, punctuation: bool = False) -> str:
    """Normalise for comparison. Each flag removes one kind of difference:
    `case` ignores letter case, `whitespace` trims and collapses runs of spaces,
    `punctuation` drops punctuation characters."""
    if punctuation:
        text = _PUNCT.sub("", text)
    if whitespace:
        text = _SPACE.sub(" ", text).strip()
    if case:
        text = text.casefold()
    return text


def exact_match(
    case_sensitive: bool = False,
    *,
    ignore_case: bool | None = None,
    ignore_whitespace: bool = True,
    ignore_punctuation: bool = False,
) -> Scorer:
    """Score by comparing the output with the target after the chosen normalisation.

    `case_sensitive` is the original option and is kept for existing callers;
    `ignore_case` overrides it when given.
    """
    fold_case = (not case_sensitive) if ignore_case is None else ignore_case

    def norm(s: str) -> str:
        return normalize_text(s, case=fold_case, whitespace=ignore_whitespace, punctuation=ignore_punctuation)

    def score(sample: Sample, output: ModelOutput) -> Score:
        pred = strip_think(output.text).strip()
        correct = norm(pred) == norm(sample.target)
        return Score(
            value=1.0 if correct else 0.0,
            pred=pred,
            explanation=f"pred={pred!r} true={sample.target!r}",
            metadata={"unparseable": False},
        )

    return score


def match_label(text: str, labels: list[str]) -> str | None:
    """The single label an answer names, or None.

    Case-insensitive, after removing think blocks and trimming. An answer that is exactly a
    label wins. Otherwise every label is searched for as a whole word or phrase; a match that
    sits inside a longer matched label ("Tech" inside "Sci/Tech") does not count. Exactly one
    label must remain; none or several means the answer is unparseable.
    """
    answer = strip_think(text).strip()
    folded = answer.casefold().strip(" .!\"'`*")
    for label in labels:
        if folded == label.casefold().strip():
            return label
    spans: list[tuple[int, int, str]] = []
    for label in labels:
        needle = label.casefold().strip()
        if not needle:
            continue
        pattern = re.compile(r"(?<![\w])" + re.escape(needle) + r"(?![\w])")
        for m in pattern.finditer(answer.casefold()):
            spans.append((m.start(), m.end(), label))
    kept = {
        label
        for (a, b, label) in spans
        if not any((a2 <= a and b <= b2) and (b2 - a2) > (b - a) for (a2, b2, _l) in spans)
    }
    return kept.pop() if len(kept) == 1 else None


def label_match(labels: list[str]) -> Scorer:
    """Multi-class scorer: the prediction is the one label the answer names (see `match_label`)."""
    if not labels:
        raise ValueError("multi-class scoring needs at least one label")

    def score(sample: Sample, output: ModelOutput) -> Score:
        pred = match_label(output.text, labels)
        return Score(
            value=1.0 if pred is not None and pred.casefold() == sample.target.casefold() else 0.0,
            pred=pred,
            explanation=f"pred={pred if pred is not None else 'unparseable'} true={sample.target}",
            metadata={"unparseable": pred is None},
        )

    return score


# --- judge replies ------------------------------------------------------------------

_VERDICT = re.compile(r"\b(PASS|FAIL)\b", re.IGNORECASE)
_SCORE_LABEL = re.compile(r"score\s*[:=]?\s*\**\s*([1-5])\b", re.IGNORECASE)
_SCORE_DIGIT = re.compile(r"(?<![\d.])([1-5])(?!\d|\.\d)")
_OUT_OF = re.compile(r"\s*(/|out of)\s*5\b", re.IGNORECASE)  # "4/5" and "4 out of 5" mean 4


def parse_judge_reply(text: str, kind: str) -> int | None:
    """Read a judge's verdict.

    pass/fail: the last standalone PASS or FAIL (1 or 0).
    1-5 scale: the number after "score" if present, otherwise the last standalone 1-5.
    Returns None when nothing usable is found.
    """
    body = strip_think(text)
    if kind == "passfail":
        found = _VERDICT.findall(body)
        return (1 if found[-1].upper() == "PASS" else 0) if found else None
    if kind == "scale":
        labelled = _SCORE_LABEL.findall(body)
        if labelled:
            return int(labelled[-1])
        digits = _SCORE_DIGIT.findall(_OUT_OF.sub("", body))
        return int(digits[-1]) if digits else None
    raise ValueError(f"unknown judge kind '{kind}'")

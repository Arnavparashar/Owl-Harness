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


def exact_match(case_sensitive: bool = False) -> Scorer:
    """Score by comparing the whitespace-stripped output with the target."""

    def norm(s: str) -> str:
        s = s.strip()
        return s if case_sensitive else s.lower()

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

"""Judge-graded scoring: a second model grades each answer against a rubric.

The judge is a run-scoped resource: its provider is created and warmed up before the first
sample and closed after the last, like the model under test. Its calls are not part of the
measured latency, which covers only the evaluated model's own call.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from owl.providers.base import Provider, get_provider_class
from owl.scorers import parse_judge_reply, strip_think
from owl.templates import render
from owl.types import GenerateConfig, Message, ModelOutput, Sample, Score

RubricKind = Literal["passfail", "scale"]

_PASSFAIL_WITH_REFERENCE = """You are grading an answer to a question.

Question:
{input}

Reference answer:
{target}

Answer to grade:
{output}

Is the answer correct and consistent with the reference answer? Wording may differ; the facts must agree.
Explain your reasoning in one or two sentences, then end with a final line that is exactly PASS or FAIL."""

_PASSFAIL_NO_REFERENCE = """You are grading an answer to a question.

Question:
{input}

Answer to grade:
{output}

Is the answer correct, relevant and complete?
Explain your reasoning in one or two sentences, then end with a final line that is exactly PASS or FAIL."""

_SCALE_WITH_REFERENCE = """You are grading an answer to a question.

Question:
{input}

Reference answer:
{target}

Answer to grade:
{output}

Rate the answer from 1 to 5 against the reference: 1 = wrong or unrelated, 2 = mostly wrong, 3 = partly right,
4 = right with minor gaps, 5 = fully correct and complete.
Explain your reasoning in one or two sentences, then end with a final line of the form: Score: N"""

_SCALE_NO_REFERENCE = """You are grading an answer to a question.

Question:
{input}

Answer to grade:
{output}

Rate the answer from 1 to 5: 1 = wrong or unrelated, 2 = mostly wrong, 3 = partly right,
4 = right with minor gaps, 5 = fully correct and complete.
Explain your reasoning in one or two sentences, then end with a final line of the form: Score: N"""


def default_rubric(kind: RubricKind, has_reference: bool) -> str:
    if kind == "passfail":
        return _PASSFAIL_WITH_REFERENCE if has_reference else _PASSFAIL_NO_REFERENCE
    return _SCALE_WITH_REFERENCE if has_reference else _SCALE_NO_REFERENCE


class JudgeScorer:
    """An async scorer backed by a judge model."""

    def __init__(
        self,
        provider: str,
        model: str,
        args: dict[str, Any],
        kind: RubricKind,
        template: str,
        timeout_s: float = 120.0,
        max_tokens: int = 512,
        seed: int | None = 42,
    ) -> None:
        self.provider_name = provider
        self.model = model
        self.args = args
        self.kind = kind
        self.template = template
        self.timeout_s = timeout_s
        self.config = GenerateConfig(temperature=0.0, max_tokens=max_tokens, seed=seed)
        self._provider: Provider | None = None
        self._lock = asyncio.Lock()
        self.info: dict[str, Any] = {}

    async def setup(self) -> None:
        cls = get_provider_class(self.provider_name)
        self._provider = cls(self.model, **self.args)
        await self._provider.setup()
        self.info = self._provider.model_info()

    async def close(self) -> None:
        if self._provider is not None:
            await self._provider.close()
            self._provider = None

    def describe(self) -> dict[str, Any]:
        return {
            "judge": {
                "provider": self.provider_name,
                "model": self.model,
                "settings": self.args,
                "model_info": self.info,
                "generate": self.config.model_dump(),
            },
            "rubric": self.kind,
            "rubric_template": self.template,
        }

    async def __call__(self, sample: Sample, output: ModelOutput) -> Score:
        if self._provider is None:
            raise RuntimeError("judge used before setup")
        row = sample.metadata.get("row") or {}
        values = {**row, "input": sample.input, "target": sample.target, "output": strip_think(output.text).strip()}
        prompt = render(self.template, values)
        async with self._lock:  # one judge call at a time
            reply = await asyncio.wait_for(
                self._provider.generate([Message(role="user", content=prompt)], self.config), timeout=self.timeout_s
            )
        verdict = parse_judge_reply(reply.text, self.kind)
        if verdict is None:
            correct, pred = False, None
        elif self.kind == "passfail":
            correct, pred = verdict == 1, str(verdict)
        else:
            correct, pred = verdict >= 4, str(verdict)  # 4 or 5 counts as a good answer
        return Score(
            value=1.0 if correct else 0.0,
            pred=pred,
            explanation=reply.text,
            metadata={
                "unparseable": verdict is None,
                "judge_reply": reply.text,
                "judge_latency_ms": reply.latency_ms,
                "verdict": verdict,
            },
        )

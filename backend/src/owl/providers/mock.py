"""Deterministic provider for tests and demos. Makes no network calls; it only waits if `delay_ms` is set.

The model name picks a behaviour:
  mock/keyword   answers 1 when the text contains injection-like phrases, with a probability
  mock/always1   always answers "1"
  mock/always0   always answers "0"
  mock/think     answers with a <think> block before the digit
  mock/garbage   answers with text that has no 0/1 digit
"""

from __future__ import annotations

import asyncio
from typing import Any

from owl.providers.base import ArgSpec, Provider
from owl.types import GenerateConfig, Message, ModelOutput, Usage

_TRIGGERS = ("ignore", "override", "jailbreak", "previous instructions", "disregard")
_MODES = ("keyword", "always1", "always0", "think", "garbage")


class MockProvider(Provider):
    name = "mock"
    max_concurrency = 4
    description = "Deterministic fake models for trying the app without a real model. Answers by keyword."
    arg_specs = (
        ArgSpec("fail_if_contains", "string", None, "Fail on inputs containing this text, to see how errors are recorded."),
        ArgSpec("delay_ms", "integer", 0, "Wait this long before each answer, to watch progress or try cancelling."),
    )

    @classmethod
    async def list_models(cls, **_: Any) -> list[str]:
        return list(_MODES)

    def __init__(self, model: str, fail_if_contains: str | None = None, delay_ms: int = 0, **model_args: Any) -> None:
        super().__init__(model, **model_args)
        if model not in _MODES:
            raise ValueError(f"unknown mock model '{model}', expected one of {_MODES}")
        # Raise on inputs containing this text, to exercise error handling.
        self.fail_if_contains = fail_if_contains
        self.delay_ms = max(0, int(delay_ms))

    async def generate(self, messages: list[Message], config: GenerateConfig) -> ModelOutput:
        text = next((m.content for m in reversed(messages) if m.role == "user"), "")
        if self.delay_ms:
            await asyncio.sleep(self.delay_ms / 1000)
        if self.fail_if_contains and self.fail_if_contains in text:
            raise RuntimeError("mock failure")

        lowered = text.lower()
        hit = any(t in lowered for t in _TRIGGERS)
        jitter = (len(text) % 10) / 100
        probability: float | None = None

        if self.model == "keyword":
            probability = (0.85 if hit else 0.05) + jitter
            answer = "1" if hit else "0"
            out = answer
        elif self.model == "always1":
            out = "1"
        elif self.model == "always0":
            out = "0"
        elif self.model == "think":
            out = f"<think>checking {len(text)} chars</think>{'1' if hit else '0'}"
        else:
            out = "I am not sure."

        return ModelOutput(
            text=out,
            probability=probability,
            latency_ms=float(1 + len(text) % 7),
            stop_reason="stop",
            usage=Usage(input_tokens=len(text.split()), output_tokens=1),
        )

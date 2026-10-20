"""Owl provider plugin for the Laya decision model (https://pypi.org/project/laya/).

Laya is an encoder, not a text generator: for a yes/no ("noul") question it returns
P(true). This provider turns that into an answer of "1" (P >= threshold) or "0" and
exposes the raw P(true) as `ModelOutput.probability`, so Owl shows ROC and threshold
analysis for it.

It only answers yes/no tests. Owl passes the test's question in as `question`
(declared through `task_inputs`); `instruction` is accepted as an older name for it.
Runs on CPU with at most 8 torch threads.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from owl.providers import ArgSpec, Provider
from owl.types import GenerateConfig, Message, ModelOutput

__version__ = "0.1.0"

_QUESTION_ID = "q"


class LayaProvider(Provider):
    name = "laya"
    display_name = "Laya (plugin)"
    max_concurrency = 1
    binary_only = True
    task_inputs = ("question",)
    description = (
        "Laya, a small encoder that answers yes/no questions with a probability. Runs on CPU. "
        "Yes/no tests only. Weights for a model you have not used before are downloaded on first run."
    )
    arg_specs = (ArgSpec("threshold", "number", 0.5, "Answer yes (1) when P(yes) is at least this value.", label="Threshold"),)

    @classmethod
    async def list_models(cls, **_: Any) -> list[str]:
        try:
            from laya import DEFAULT_MODELS
        except ImportError:
            return []
        return sorted(DEFAULT_MODELS)

    def __init__(
        self,
        model: str,
        question: str | None = None,
        instruction: str | None = None,
        threshold: float = 0.5,
        **model_args: Any,
    ) -> None:
        super().__init__(model, **model_args)
        self.instruction = question or instruction
        if not self.instruction:
            raise ValueError("laya needs the yes/no question; it is only usable with yes/no tests")
        self.threshold = float(threshold)
        self._router: Any = None
        self._lock = threading.Lock()
        self._info: dict[str, Any] = {}

    def _load(self) -> None:
        import torch
        import laya
        from laya import Router

        torch.set_num_threads(8)
        self._router = Router(max_loaded=1, device="cpu")
        repo, _variant = laya.DEFAULT_MODELS.get(self.model, (None, None))
        self._info = {
            "laya": getattr(laya, "__version__", None),
            "revision": laya.PINNED_REVISIONS.get(repo) if repo else None,
            "torch": torch.__version__,
        }

    def _predict(self, text: str) -> tuple[float, float]:
        # One forward pass at a time; time only the inference itself.
        with self._lock:
            t0 = time.perf_counter()
            result = self._router.predict(
                text,
                {_QUESTION_ID: {"type": "noul", "instructions": self.instruction}},
                model=self.model,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
        return float(result["answers"][_QUESTION_ID]["noul"]), latency_ms

    async def setup(self) -> None:
        def load_and_warm() -> None:
            self._load()
            self._predict("warm-up")  # loads the checkpoint so it is not counted in latency

        await asyncio.to_thread(load_and_warm)

    def model_info(self) -> dict[str, Any]:
        return dict(self._info)

    async def close(self) -> None:
        self._router = None  # let the weights be freed between runs

    async def generate(self, messages: list[Message], config: GenerateConfig) -> ModelOutput:
        text = next((m.content for m in reversed(messages) if m.role == "user"), "")
        p_true, latency_ms = await asyncio.to_thread(self._predict, text)
        return ModelOutput(
            text="1" if p_true >= self.threshold else "0",
            probability=p_true,
            latency_ms=latency_ms,
            stop_reason="stop",
            raw={"threshold": self.threshold},
        )

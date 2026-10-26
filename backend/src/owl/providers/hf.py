"""Provider for Hugging Face models run locally with `transformers`.

Model string: `hf/<hub id or local folder>`, e.g. `hf/deepset/deberta-v3-base-injection`.

Settings
  mode       causal-lm (default): an instruct model; the conversation goes through its chat
             template and the reply is generated.
             sequence-classification: a classifier head; the answer is the predicted label.
  label_map  sequence-classification only. JSON object mapping the model's labels to the
             test's answers, e.g. {"INJECTION": "1", "LEGIT": "0"}. Yes/no tests answer 1 or 0;
             multi-class tests answer with the label names. For a yes/no test the probability
             of the labels mapped to 1 is stored as P(positive).
  device     cpu (default), or cuda / cuda:0 when a CUDA build of torch is installed
  revision   optional Hub revision (branch, tag or commit) to pin

Requires the `hf-models` extra. The model loads once per run, gets one warm-up call outside
the timed samples, and calls are serialised with a lock; torch uses at most 8 threads.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from owl.providers.base import ArgSpec, Provider
from owl.types import GenerateConfig, Message, ModelOutput

MODES = ("causal-lm", "sequence-classification")


class TransformersBackend:
    """The only place that touches transformers and torch. Tests replace it with a fake."""

    def __init__(self, model: str, mode: str, device: str, revision: str | None) -> None:
        try:
            import torch
            import transformers
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as e:  # pragma: no cover - depends on installed extras
            raise RuntimeError("the hf provider needs the 'hf-models' extra: pip install 'owl-harness[hf-models]'") from e
        torch.set_num_threads(min(8, torch.get_num_threads()))
        self._torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        cls = AutoModelForSequenceClassification if mode == "sequence-classification" else AutoModelForCausalLM
        self.model = cls.from_pretrained(model, revision=revision).to(device)
        self.model.eval()
        config = self.model.config
        self.labels: list[str] = [config.id2label[i] for i in sorted(config.id2label)] if mode == "sequence-classification" else []
        self.info = {
            "revision": getattr(config, "_commit_hash", None) or revision or "local",
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "device": device,
        }

    def classify(self, text: str) -> dict[str, float]:
        torch = self._torch
        enc = self.tokenizer(text, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits[0], dim=-1).tolist()
        return {self.labels[i]: float(p) for i, p in enumerate(probs)}

    def chat(self, messages: list[dict[str, str]], max_new_tokens: int, temperature: float | None, seed: int | None) -> tuple[str, int, int]:
        torch = self._torch
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if not isinstance(ids, torch.Tensor):  # some tokenizers return a BatchEncoding
            ids = ids["input_ids"]
        ids = ids.to(self.device)
        sample = bool(temperature)
        if seed is not None:
            torch.manual_seed(seed)
        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": sample}
        if sample:
            kwargs["temperature"] = temperature
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        with torch.no_grad():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), pad_token_id=pad, **kwargs)
        new = out[0][ids.shape[-1] :]
        return self.tokenizer.decode(new, skip_special_tokens=True), int(ids.shape[-1]), int(new.shape[-1])


class HFProvider(Provider):
    name = "hf"
    display_name = "Hugging Face (local)"
    max_concurrency = 1
    task_inputs = ("positive_class",)
    description = (
        "Runs a Hugging Face model on this machine: an instruct model (causal-lm) or a classifier "
        "(sequence-classification). Use a Hub id or a local folder as the model name."
    )
    arg_specs = (
        ArgSpec("mode", "string", "causal-lm", "What kind of model this is.", label="Mode", choices=MODES),
        ArgSpec(
            "label_map",
            "json",
            None,
            'Classifiers only: map the model\'s labels to the test\'s answers, e.g. {"INJECTION": "1", "LEGIT": "0"}. Yes/no tests answer 1 or 0.',
            label="Label map",
        ),
        ArgSpec("device", "string", "cpu", "cpu, or cuda when a CUDA build of torch is installed.", label="Device"),
        ArgSpec("revision", "string", None, "Optional Hub revision to pin (branch, tag or commit).", label="Revision"),
    )
    backend_class: type = TransformersBackend

    def __init__(
        self,
        model: str,
        mode: str = "causal-lm",
        label_map: dict[str, str] | None = None,
        device: str = "cpu",
        revision: str | None = None,
        positive_class: str | None = None,
        **model_args: Any,
    ) -> None:
        super().__init__(model, **model_args)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        if mode == "sequence-classification" and not label_map:
            raise ValueError("a sequence-classification model needs label_map: the model's labels mapped to the test's answers")
        self.mode = mode
        self.label_map = {str(k): str(v) for k, v in (label_map or {}).items()}
        self.device = device or "cpu"
        self.revision = revision or None
        self.positive_class = positive_class
        self._backend: Any = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        self._backend = self.backend_class(self.model, self.mode, self.device, self.revision)
        if self.mode == "sequence-classification":
            unknown = [k for k in self.label_map if k not in self._backend.labels]
            missing = [l for l in self._backend.labels if l not in self.label_map]
            if unknown or missing:
                raise ValueError(
                    f"label_map must name every model label exactly. the model's labels are: {', '.join(self._backend.labels)}"
                )

    async def setup(self) -> None:
        def load_and_warm() -> None:
            self._load()
            if self.mode == "sequence-classification":
                self._backend.classify("warm-up")
            else:
                self._backend.chat([{"role": "user", "content": "Reply with OK."}], 1, None, None)

        await asyncio.to_thread(load_and_warm)

    def _classify(self, text: str) -> tuple[dict[str, float], float]:
        with self._lock:
            t0 = time.perf_counter()
            probs = self._backend.classify(text)
            return probs, (time.perf_counter() - t0) * 1000

    def _chat(self, messages: list[dict[str, str]], config: GenerateConfig) -> tuple[str, int, int, float]:
        with self._lock:
            t0 = time.perf_counter()
            text, n_in, n_out = self._backend.chat(messages, config.max_tokens or 256, config.temperature, config.seed)
            return text, n_in, n_out, (time.perf_counter() - t0) * 1000

    async def generate(self, messages: list[Message], config: GenerateConfig) -> ModelOutput:
        from owl.types import Usage

        if self.mode == "sequence-classification":
            text = next((m.content for m in reversed(messages) if m.role == "user"), "")
            probs, latency_ms = await asyncio.to_thread(self._classify, text)
            top = max(probs, key=probs.__getitem__)
            probability = None
            if self.positive_class is not None:
                probability = sum(p for label, p in probs.items() if self.label_map.get(label) == self.positive_class)
            return ModelOutput(
                text=self.label_map[top],
                probability=probability,
                latency_ms=latency_ms,
                stop_reason="stop",
                raw={"label": top, "scores": probs},
            )

        chat = [{"role": m.role, "content": m.content} for m in messages if m.role in ("system", "user", "assistant")]
        text, n_in, n_out, latency_ms = await asyncio.to_thread(self._chat, chat, config)
        limit = config.max_tokens or 256
        return ModelOutput(
            text=text,
            latency_ms=latency_ms,
            stop_reason="max_tokens" if n_out >= limit else "stop",
            usage=Usage(input_tokens=n_in, output_tokens=n_out),
        )

    def model_info(self) -> dict[str, Any]:
        return dict(self._backend.info) if self._backend is not None else {}

    async def close(self) -> None:
        self._backend = None

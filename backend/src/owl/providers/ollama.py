"""Ollama provider using the native `/api/chat` endpoint.

Thinking is switched off with the documented top-level `think: false` request field
(https://docs.ollama.com/capabilities/thinking). The OpenAI-compatible `/v1` endpoint
ignores that field, which is why this provider talks to the native API.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from owl.providers.base import ArgSpec, Provider
from owl.types import GenerateConfig, Message, ModelOutput, Usage

DEFAULT_BASE_URL = "http://localhost:11434"


def _base_url(value: str | None) -> str:
    url = (value or os.environ.get("OLLAMA_HOST") or DEFAULT_BASE_URL).rstrip("/")
    return url if url.startswith("http") else f"http://{url}"


class OllamaProvider(Provider):
    name = "ollama"
    max_concurrency = 1
    display_name = "Ollama"
    description = "Local models served by Ollama, called through its native chat API."
    arg_specs = (
        ArgSpec("base_url", "string", DEFAULT_BASE_URL, "Ollama server address.", affects_models=True),
        ArgSpec("think", "boolean", False, "Let thinking models reason before answering. Off by default so answers stay short."),
    )

    @classmethod
    async def list_models(cls, base_url: str | None = None, **_: Any) -> list[str]:
        """Models already pulled on the server (never downloads anything)."""
        async with httpx.AsyncClient(base_url=_base_url(base_url), timeout=httpx.Timeout(5.0)) as client:
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            return sorted(m["name"] for m in resp.json().get("models", []))

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        think: bool = False,
        **model_args: Any,
    ) -> None:
        super().__init__(model, **model_args)
        self.base_url = _base_url(base_url)
        self.think = bool(think)
        self._info: dict[str, Any] = {}
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(300.0))

    async def setup(self) -> None:
        # Check the server, then load the model into memory so the first sample is not slow.
        try:
            await self._client.get("/api/version")
        except httpx.HTTPError as e:
            raise RuntimeError(f"cannot reach Ollama at {self.base_url}: {e}") from e
        try:  # record exactly which build of the model ran
            version = (await self._client.get("/api/version")).json().get("version")
            tags = (await self._client.get("/api/tags")).json().get("models", [])
            entry = next((m for m in tags if m.get("name") == self.model or m.get("model") == self.model), None)
            self._info = {"ollama_version": version}
            if entry:
                self._info["digest"] = entry.get("digest")
                details = entry.get("details") or {}
                for k in ("parameter_size", "quantization_level", "family"):
                    if details.get(k):
                        self._info[k] = details[k]
        except (httpx.HTTPError, ValueError):
            self._info = {}
        await self._chat(
            [{"role": "user", "content": "Reply with 0."}],
            {"num_predict": 1, "temperature": 0},
        )

    async def _chat(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "options": options,
        }
        resp = await self._client.post("/api/chat", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"ollama /api/chat returned {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    async def generate(self, messages: list[Message], config: GenerateConfig) -> ModelOutput:
        options: dict[str, Any] = {}
        if config.temperature is not None:
            options["temperature"] = config.temperature
        if config.max_tokens is not None:
            options["num_predict"] = config.max_tokens
        if config.seed is not None:
            options["seed"] = config.seed
        payload_messages = [{"role": m.role, "content": m.content} for m in messages]

        t0 = time.perf_counter()
        data = await self._chat(payload_messages, options)
        latency_ms = (time.perf_counter() - t0) * 1000

        message = data.get("message", {})
        done_reason = data.get("done_reason")
        raw: dict[str, Any] = {"done_reason": done_reason}
        if message.get("thinking"):
            raw["thinking"] = message["thinking"]
        return ModelOutput(
            text=message.get("content", ""),
            latency_ms=latency_ms,
            stop_reason="max_tokens" if done_reason == "length" else "stop" if done_reason else "unknown",
            usage=Usage(input_tokens=data.get("prompt_eval_count"), output_tokens=data.get("eval_count")),
            raw=raw,
        )

    def model_info(self) -> dict[str, Any]:
        return dict(self._info)

    async def close(self) -> None:
        await self._client.aclose()

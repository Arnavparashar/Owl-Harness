"""Provider for any OpenAI-compatible `/chat/completions` endpoint.

Configuration (model args win over environment variables):
  base_url      or env OPENAI_BASE_URL   (default https://api.openai.com/v1)
  api_key_env   name of the env var holding the key (default OPENAI_API_KEY)
  max_tokens_param   request field for the token limit (default "max_tokens";
                     some newer models want "max_completion_tokens")
The key is read from the environment at call time and never stored in the run config.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from owl.providers.base import ArgSpec, Provider
from owl.types import GenerateConfig, Message, ModelOutput, Usage

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIProvider(Provider):
    name = "openai"
    max_concurrency = 4
    description = "Any OpenAI-compatible chat endpoint: OpenAI itself, LM Studio, vLLM, a llama.cpp server and others."
    arg_specs = (
        ArgSpec("base_url", "string", DEFAULT_BASE_URL, "Endpoint base URL, including /v1.", affects_models=True),
        ArgSpec(
            "api_key_env",
            "string",
            "OPENAI_API_KEY",
            "Name of the environment variable that holds the API key. The key itself is never entered or stored here.",
            affects_models=True,
        ),
        ArgSpec("max_tokens_param", "string", "max_tokens", "Request field for the token limit; some newer models need max_completion_tokens."),
    )

    @classmethod
    async def list_models(cls, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY", **_: Any) -> list[str]:
        url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        key = os.environ.get(api_key_env)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
            resp = await client.get(f"{url}/models", headers=headers)
            resp.raise_for_status()
            return sorted(m["id"] for m in resp.json().get("data", []))

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        max_tokens_param: str = "max_tokens",
        **model_args: Any,
    ) -> None:
        super().__init__(model, **model_args)
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key_env = api_key_env
        self.max_tokens_param = max_tokens_param
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))

    def _headers(self) -> dict[str, str]:
        key = os.environ.get(self.api_key_env)
        return {"Authorization": f"Bearer {key}"} if key else {}

    async def setup(self) -> None:
        # Hosted OpenAI needs a key; local OpenAI-compatible servers often do not.
        if self.base_url == DEFAULT_BASE_URL and not os.environ.get(self.api_key_env):
            raise RuntimeError(f"environment variable {self.api_key_env} is not set")

    async def generate(self, messages: list[Message], config: GenerateConfig) -> ModelOutput:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if config.temperature is not None:
            payload["temperature"] = config.temperature
        if config.max_tokens is not None:
            payload[self.max_tokens_param] = config.max_tokens

        t0 = time.perf_counter()
        resp = await self._client.post(f"{self.base_url}/chat/completions", json=payload, headers=self._headers())
        latency_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code != 200:
            raise RuntimeError(f"chat/completions returned {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        choice = data["choices"][0]
        finish = choice.get("finish_reason")
        usage = data.get("usage") or {}
        return ModelOutput(
            text=choice["message"].get("content") or "",
            latency_ms=latency_ms,
            stop_reason="max_tokens" if finish == "length" else "stop" if finish else "unknown",
            usage=Usage(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens")),
            raw={"finish_reason": finish, "id": data.get("id")},
        )

    async def close(self) -> None:
        await self._client.aclose()

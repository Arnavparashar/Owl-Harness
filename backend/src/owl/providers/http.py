"""Provider for any model behind an HTTP endpoint, configured entirely by settings.

Settings
  url               endpoint URL (may use template variables)
  method            POST (default), PUT or GET
  headers           JSON object. Values may reference environment variables as
                    ${ENV:NAME}; they are resolved here, on the backend, at request time.
                    Headers that carry credentials must use such a reference.
  body              JSON request template (ignored for GET)
  answer_path       where the answer text is in the JSON response, e.g. message.content
                    or choices[0].message.content
  probability_path  optional: where P(positive) is, for models that return one
  timeout_s         request timeout in seconds (default 60)
  warmup            send one request before the timed samples (default on), so a server
                    that loads the model on first use does not inflate the first latency

Template variables, written {{name}}:
  {{input}}     the user message (the rendered prompt)
  {{system}}    the system message, or "" when there is none
  {{messages}}  the conversation as a JSON array of {role, content}
  {{model}}     the model part of the model string (http/<model>)
  {{temperature}} {{max_tokens}} {{seed}}   the run's generation settings
  {{column}}    any column of the dataset row, e.g. {{question}}

A string that is exactly one variable is replaced by the value itself, keeping its JSON type
(so "{{messages}}" becomes an array and "{{temperature}}" a number). Inside a longer string
the value is inserted as text.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any
from urllib.parse import quote

import httpx

from owl.providers.base import ArgSpec, Provider
from owl.types import GenerateConfig, Message, ModelOutput

_VAR = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_ .-]*?)\s*\}\}")
_ENV = re.compile(r"\$\{ENV:([A-Za-z_][A-Za-z0-9_]*)\}")
_SENSITIVE_HEADER = re.compile(r"authorization|api[-_]?key|token|secret|password|cookie", re.IGNORECASE)
_PATH_PART = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def get_path(data: Any, path: str) -> Any:
    """Follow a dotted path with optional indices: `a.b[0].c`, `a.b.0.c` or `$.a.b`."""
    path = path.strip()
    if path.startswith("$"):
        path = path[1:].lstrip(".")
    current = data
    for m in _PATH_PART.finditer(path):
        key = m.group(1) if m.group(1) is not None else m.group(2)
        if isinstance(current, list):
            try:
                current = current[int(key)]
            except (ValueError, IndexError):
                raise KeyError(f"no index '{key}' at '{path}'") from None
        elif isinstance(current, dict):
            if key not in current:
                raise KeyError(f"no field '{key}' at '{path}'")
            current = current[key]
        else:
            raise KeyError(f"cannot look up '{key}' in a {type(current).__name__} at '{path}'")
    return current


def resolve_env(text: str) -> str:
    """Replace ${ENV:NAME} references with the environment value; error if one is unset."""

    def sub(m: re.Match[str]) -> str:
        value = os.environ.get(m.group(1))
        if value is None:
            raise RuntimeError(f"environment variable {m.group(1)} is not set")
        return value

    return _ENV.sub(sub, text)


def render_value(template: Any, variables: dict[str, Any]) -> Any:
    """Fill {{name}} variables throughout a JSON-like template."""
    if isinstance(template, dict):
        return {k: render_value(v, variables) for k, v in template.items()}
    if isinstance(template, list):
        return [render_value(v, variables) for v in template]
    if not isinstance(template, str):
        return template

    whole = _VAR.fullmatch(template.strip())
    if whole:
        return _lookup(whole.group(1), variables)

    def sub(m: re.Match[str]) -> str:
        v = _lookup(m.group(1), variables)
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)

    return _VAR.sub(sub, template)


class _Lenient(dict):
    """Variables for the warm-up request: any name the template uses resolves to ""."""

    def __contains__(self, key: object) -> bool:
        return True

    def __missing__(self, key: str) -> str:
        return ""


def _lookup(name: str, variables: dict[str, Any]) -> Any:
    if name not in variables:
        raise RuntimeError(f"the request template uses {{{{{name}}}}} but there is no such value. available: {', '.join(sorted(variables))}")
    return variables[name]


class HTTPProvider(Provider):
    name = "http"
    display_name = "Custom HTTP endpoint"
    max_concurrency = 1
    description = (
        "Any model with a REST API: describe the request as a JSON template and say where the answer is in the "
        "response. Credentials go in headers as ${ENV:NAME} references, read on the server."
    )
    arg_specs = (
        ArgSpec("url", "string", None, "Endpoint URL.", label="URL"),
        ArgSpec("method", "string", "POST", "HTTP method.", label="Method", choices=("POST", "PUT", "GET")),
        ArgSpec(
            "headers",
            "json",
            None,
            'JSON object, e.g. {"Authorization": "Bearer ${ENV:MY_API_KEY}"}. Credentials must be ${ENV:NAME} references.',
            label="Headers",
        ),
        ArgSpec(
            "body",
            "json",
            None,
            'JSON request template, e.g. {"model": "{{model}}", "messages": "{{messages}}"}. '
            "Variables: {{input}}, {{system}}, {{messages}}, {{model}}, {{temperature}}, {{max_tokens}}, {{seed}} and any dataset column.",
            label="Request body template",
        ),
        ArgSpec("answer_path", "string", None, "Where the answer text is in the JSON response, e.g. choices[0].message.content.", label="Answer path"),
        ArgSpec("probability_path", "string", None, "Optional: where P(positive) is in the response.", label="Probability path"),
        ArgSpec("timeout_s", "number", 60, "Request timeout in seconds.", label="Timeout (s)"),
        ArgSpec("warmup", "boolean", True, "Send one request before timing starts, so a cold model load is not measured.", label="Warm-up request"),
    )

    def __init__(
        self,
        model: str,
        url: str | None = None,
        method: str = "POST",
        headers: dict[str, Any] | None = None,
        body: Any = None,
        answer_path: str | None = None,
        probability_path: str | None = None,
        timeout_s: float = 60.0,
        warmup: bool = True,
        **model_args: Any,
    ) -> None:
        super().__init__(model, **model_args)
        if not url:
            raise ValueError("the http provider needs a url")
        if not answer_path:
            raise ValueError("the http provider needs answer_path: where the answer text is in the response")
        self.method = method.upper()
        if self.method not in ("POST", "PUT", "GET"):
            raise ValueError("method must be POST, PUT or GET")
        if isinstance(headers, str):
            headers = json.loads(headers)
        if isinstance(body, str):
            body = json.loads(body)
        self.headers = {str(k): str(v) for k, v in (headers or {}).items()}
        for key, value in self.headers.items():
            # A literal credential would be stored in the run config; insist on a reference instead.
            if _SENSITIVE_HEADER.search(key) and not _ENV.search(value):
                raise ValueError(
                    f"header '{key}' looks like a credential; write it as an environment reference such as "
                    "${ENV:MY_API_KEY} so the secret is never stored"
                )
        self.url = url
        self.body = body
        self.answer_path = answer_path
        self.probability_path = probability_path or None
        self.timeout_s = float(timeout_s)
        self.warmup = bool(warmup)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_s))

    async def setup(self) -> None:
        if self.warmup:
            warm = [Message(role="user", content="Reply with OK.")]
            await self._request(warm, GenerateConfig(temperature=0.0, max_tokens=4), lenient=True)

    def _variables(self, messages: list[Message], config: GenerateConfig) -> dict[str, Any]:
        user = next((m for m in reversed(messages) if m.role == "user"), None)
        system = next((m.content for m in messages if m.role == "system"), "")
        row = (user.metadata.get("row") if user else None) or {}
        variables: dict[str, Any] = {str(k): v for k, v in row.items()}
        variables.update(
            {
                "input": user.content if user else "",
                "system": system,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "model": self.model,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "seed": config.seed,
            }
        )
        return variables

    async def generate(self, messages: list[Message], config: GenerateConfig) -> ModelOutput:
        return await self._request(messages, config)

    async def _request(self, messages: list[Message], config: GenerateConfig, lenient: bool = False) -> ModelOutput:
        variables: dict[str, Any] = self._variables(messages, config)
        if lenient:
            variables = _Lenient(variables)
        url = _VAR.sub(lambda m: quote(str(_lookup(m.group(1), variables)), safe=""), self.url)
        headers = {k: resolve_env(v) for k, v in self.headers.items()}
        payload = render_value(self.body, variables) if self.body is not None and self.method != "GET" else None

        t0 = time.perf_counter()
        try:
            resp = await self._client.request(self.method, url, json=payload, headers=headers)
        except httpx.TimeoutException:
            raise RuntimeError(f"no response within {self.timeout_s:g} s") from None
        latency_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code >= 400:
            raise RuntimeError(f"endpoint returned {resp.status_code}: {resp.text[:300]}")
        try:
            data = resp.json()
        except ValueError:
            raise RuntimeError(f"response is not JSON: {resp.text[:200]}") from None

        try:
            answer = get_path(data, self.answer_path)
        except KeyError as e:
            raise RuntimeError(f"answer_path: {e.args[0]}") from None
        probability = None
        if self.probability_path:
            try:
                probability = float(get_path(data, self.probability_path))
            except (KeyError, TypeError, ValueError) as e:
                raise RuntimeError(f"probability_path: {e}") from None
            if not 0.0 <= probability <= 1.0:
                raise RuntimeError(f"probability {probability} is outside 0..1")
        return ModelOutput(
            text=answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False),
            probability=probability,
            latency_ms=latency_ms,
            stop_reason="stop",
            raw={"status": resp.status_code},
        )

    def model_info(self) -> dict[str, Any]:
        return {"endpoint": self.url.split("?")[0]}

    async def close(self) -> None:
        await self._client.aclose()

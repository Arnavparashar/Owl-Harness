"""Provider interface and the plugin registry.

A provider turns a list of messages into a `ModelOutput`. Third-party packages add
providers by exposing an entry point in the `owl.providers` group:

    [project.entry-points."owl.providers"]
    myprovider = "my_package.module:MyProvider"

Models are addressed as `provider/model`, e.g. `ollama/qwen3.5:4b`.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from importlib.metadata import entry_points
from typing import Any, ClassVar, Literal

from owl.types import GenerateConfig, Message, ModelOutput

ENTRY_POINT_GROUP = "owl.providers"

# Fallback for source checkouts that were not installed (so no entry points exist).
_BUILTIN: dict[str, str] = {
    "mock": "owl.providers.mock:MockProvider",
    "ollama": "owl.providers.ollama:OllamaProvider",
    "openai": "owl.providers.openai:OpenAIProvider",
    "laya": "owl.providers.laya:LayaProvider",
}


@dataclass(frozen=True)
class ArgSpec:
    """One user-settable model argument. The dashboard builds its form from these."""

    name: str
    type: Literal["string", "number", "integer", "boolean"] = "string"
    default: Any = None
    description: str = ""
    # True when the value changes which models are available (for example the server URL).
    affects_models: bool = False


class Provider(ABC):
    """Abstract async model provider."""

    name: ClassVar[str]
    # Calls allowed in flight at once. Local models default to 1.
    max_concurrency: ClassVar[int] = 1
    # Shown in the dashboard's provider picker.
    description: ClassVar[str] = ""
    # Settable model args (everything the constructor accepts, except ones a task supplies).
    arg_specs: ClassVar[tuple[ArgSpec, ...]] = ()
    # True for providers that can only answer yes/no questions (they cannot do exact match).
    binary_only: ClassVar[bool] = False

    @classmethod
    async def list_models(cls, **model_args: Any) -> list[str]:
        """Models this provider can run right now. Empty means "not discoverable: type a name"."""
        return []

    def __init__(self, model: str, **model_args: Any) -> None:
        self.model = model
        if model_args:
            raise ValueError(
                f"provider '{self.name}' got unknown model args: {', '.join(sorted(model_args))}"
            )

    async def setup(self) -> None:
        """Called once before the first sample (load weights, warm up, check the server)."""

    @abstractmethod
    async def generate(self, messages: list[Message], config: GenerateConfig) -> ModelOutput:
        """Produce one completion. Must fill `latency_ms` with the model-call wall time."""

    async def close(self) -> None:
        """Release resources."""


def split_model_string(model_string: str) -> tuple[str, str]:
    provider, sep, model = model_string.partition("/")
    if not sep or not provider or not model:
        raise ValueError(f"model must look like 'provider/model', got '{model_string}'")
    return provider, model


def _load(target: str) -> type[Provider]:
    module_name, _, attr = target.partition(":")
    cls = getattr(importlib.import_module(module_name), attr)
    if not (isinstance(cls, type) and issubclass(cls, Provider)):
        raise TypeError(f"{target} is not a Provider subclass")
    return cls


def available_providers() -> dict[str, str]:
    """Map provider name -> 'module:Class' from entry points, plus built-ins."""
    found = dict(_BUILTIN)
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        found[ep.name] = ep.value
    return found


def get_provider_class(name: str) -> type[Provider]:
    providers = available_providers()
    if name not in providers:
        raise KeyError(f"unknown provider '{name}'. available: {', '.join(sorted(providers))}")
    return _load(providers[name])


def describe_provider(name: str, cls: type[Provider]) -> dict[str, Any]:
    return {
        "name": name,
        "description": cls.description,
        "max_concurrency": cls.max_concurrency,
        "binary_only": cls.binary_only,
        "args": [asdict(a) for a in cls.arg_specs],
    }


def validate_model_args(cls: type[Provider], args: dict[str, Any]) -> dict[str, Any]:
    """Check user-supplied model args against the provider's specs and coerce their types.

    Blank values are dropped (the provider default applies). Raises ValueError for an
    unknown setting or a value of the wrong type.
    """
    specs = {a.name: a for a in cls.arg_specs}
    out: dict[str, Any] = {}
    for key, value in args.items():
        spec = specs.get(key)
        if spec is None:
            raise ValueError(f"provider '{cls.name}' has no setting '{key}'. settings: {', '.join(specs) or 'none'}")
        if value is None or value == "":
            continue
        try:
            if spec.type == "number":
                if isinstance(value, bool):
                    raise ValueError
                value = float(value)
            elif spec.type == "integer":
                if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
                    raise ValueError
                value = int(value)
            elif spec.type == "boolean":
                if not isinstance(value, bool):
                    raise ValueError
            else:
                value = str(value)
        except (TypeError, ValueError):
            raise ValueError(f"setting '{key}' must be a {spec.type}") from None
        out[key] = value
    return out


def create_provider(model_string: str, **model_args: Any) -> Provider:
    name, model = split_model_string(model_string)
    return get_provider_class(name)(model, **model_args)

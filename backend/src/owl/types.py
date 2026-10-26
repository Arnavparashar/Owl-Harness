"""Core data types shared by every part of Owl.

These are deliberately small. `Message` carries a role and text today; the `tool`
role and the free-form `metadata` dict leave room for a multi-turn agent solver
later without changing the provider interface (`generate(messages, config)`).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]
StopReason = Literal["stop", "max_tokens", "error", "unknown"]


class Message(BaseModel):
    role: Role
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Sample(BaseModel):
    id: str
    input: str
    target: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class ModelOutput(BaseModel):
    text: str
    # P(positive class) for decision models such as Laya; None for text generators.
    probability: float | None = None
    latency_ms: float = 0.0
    stop_reason: StopReason = "stop"
    usage: Usage = Field(default_factory=Usage)
    # Provider-specific payload kept for inspection (never interpreted by Owl).
    raw: dict[str, Any] = Field(default_factory=dict)


class Score(BaseModel):
    value: float  # 1.0 correct, 0.0 incorrect
    pred: str | None  # None when no prediction could be extracted
    explanation: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def correct(self) -> bool:
        return self.value >= 1.0

    @property
    def unparseable(self) -> bool:
        return bool(self.metadata.get("unparseable", False))


class GenerateConfig(BaseModel):
    temperature: float | None = None
    max_tokens: int | None = None
    # Passed to providers that support seeded sampling (Ollama, OpenAI-compatible, Hugging Face, HTTP).
    seed: int | None = None


ScoringMode = Literal["binary", "multiclass", "exact_match", "judge_passfail", "judge_scale"]


class RunConfig(BaseModel):
    """Everything needed to understand (and reproduce) a run."""

    task: str
    model: str  # "provider/model"
    model_args: dict[str, Any] = Field(default_factory=dict)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    positive_class: str | None = None
    limit: int | None = None
    max_concurrency: int = 1
    timeout_s: float = 120.0
    retries: int = 2
    system_prompt: str | None = None
    # How a custom (user-defined) task was set up; None for built-in tasks.
    task_spec: dict[str, Any] | None = None
    # Added in v0.2. Runs recorded earlier leave these empty and are read with sensible fallbacks.
    scoring_mode: ScoringMode | None = None
    # The resolved prompt templates: {"system": ..., "user": ...}
    prompt: dict[str, Any] = Field(default_factory=dict)
    # Scorer configuration: labels, normalisation, judge and rubric, ...
    scorer: dict[str, Any] = Field(default_factory=dict)
    # Version facts the provider reported: digest, revision, library versions.
    model_info: dict[str, Any] = Field(default_factory=dict)

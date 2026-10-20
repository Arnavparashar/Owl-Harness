"""Tasks: a dataset, a solver chain and a scorer, plus a small registry.

Tasks come from three places: the examples shipped with Owl (`owl.examples`), any installed
package exposing an `owl.tasks` entry point, and tests defined in the dashboard on an
uploaded dataset (`owl.custom`), which are built on demand and not registered.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Protocol

from owl.datasets import Dataset
from owl.scorers import Scorer
from owl.solvers import Solver
from owl.types import GenerateConfig, ScoringMode


class Resource(Protocol):
    """Something a task needs for the length of a run, such as a judge model."""

    async def setup(self) -> None: ...

    async def close(self) -> None: ...

    def describe(self) -> dict[str, Any]: ...


@dataclass
class Task:
    name: str
    load_dataset: Callable[[], Dataset]
    solver: Solver
    scorer: Scorer
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    # Set for binary tasks; enables precision/recall/F1 and the confusion matrix.
    positive_class: str | None = None
    # If set, loading a dataset of a different size is an error.
    expected_size: int | None = None
    # Default model args per provider name. User model args win. Prefer `task_inputs`
    # on the provider, which works for every task without naming providers here.
    provider_args: dict[str, dict[str, Any]] = field(default_factory=dict)
    system_prompt: str | None = None
    # Shown in the dashboard's test picker.
    description: str = ""
    # How a user-defined task was configured (recorded on each run); None for built-in tasks.
    spec: dict[str, Any] | None = None
    # --- added in v0.2 ---
    # None means "infer": binary when positive_class is set, otherwise exact match.
    scoring_mode: ScoringMode | None = None
    # The yes/no question of a binary test, for providers that are not prompted.
    question: str | None = None
    # The label set of a multi-class test.
    labels: list[str] | None = None
    # Prompt templates recorded on the run: {"system": ..., "user": ...}
    prompt: dict[str, Any] = field(default_factory=dict)
    # Scorer configuration recorded on the run (normalisation, rubric, ...).
    scorer_config: dict[str, Any] = field(default_factory=dict)
    # Set up before the first sample and closed after the last, for each model run.
    resources: list[Resource] = field(default_factory=list)
    # Shipped as an example rather than as part of the general tool.
    example: bool = False

    @property
    def mode(self) -> ScoringMode:
        if self.scoring_mode is not None:
            return self.scoring_mode
        return "binary" if self.positive_class is not None else "exact_match"

    def task_input(self, name: str) -> Any:
        """Values a provider can ask for through `Provider.task_inputs`."""
        return {"question": self.question, "labels": self.labels, "positive_class": self.positive_class}.get(name)


ENTRY_POINT_GROUP = "owl.tasks"

# Appended to a yes/no question to tell the model how to answer.
BINARY_ANSWER_INSTRUCTION = "Answer with only a single character: 1 for yes, 0 for no."

_TASKS: dict[str, Callable[[], Task]] = {}


def register_task(name: str) -> Callable[[Callable[[], Task]], Callable[[], Task]]:
    def deco(factory: Callable[[], Task]) -> Callable[[], Task]:
        _TASKS[name] = factory
        return factory

    return deco


def _factories() -> dict[str, Callable[[], Task]]:
    """Registered tasks (including the examples) plus `owl.tasks` entry points.

    An entry point maps a task name to a zero-argument function returning a `Task`.
    """
    import owl.examples  # noqa: F401  (registers the example tasks)

    found = dict(_TASKS)
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        found.setdefault(ep.name, ep.load())
    return found


def get_task(name: str) -> Task:
    factories = _factories()
    if name not in factories:
        raise KeyError(f"unknown task '{name}'. available: {', '.join(sorted(factories))}")
    return factories[name]()


def list_tasks() -> list[str]:
    return sorted(_factories())


def is_example(name: str) -> bool:
    factory = _factories().get(name)
    return bool(factory and factory().example)


def describe_tasks() -> list[dict[str, Any]]:
    """What the dashboard needs to show each task without loading its dataset."""
    out = []
    for name in list_tasks():
        t = get_task(name)
        out.append(
            {
                "name": name,
                "description": t.description,
                "example": t.example,
                "scoring_mode": t.mode,
                "positive_class": t.positive_class,
                "binary": t.positive_class is not None,
                "labels": t.labels,
                "expected_size": t.expected_size,
                "system_prompt": t.system_prompt,
                "prompt": t.prompt,
                "temperature": t.generate.temperature,
                "max_tokens": t.generate.max_tokens,
            }
        )
    return out

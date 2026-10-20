"""Solvers: composable steps that turn a sample into a model output.

A solver is an async callable `(state, ctx) -> state`. `state.messages` is the full
conversation so far, which is what a future multi-turn solver would extend.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from owl.providers.base import Provider
from owl.templates import render
from owl.types import GenerateConfig, Message, ModelOutput, Sample


@dataclass
class TaskState:
    sample: Sample
    messages: list[Message] = field(default_factory=list)
    output: ModelOutput | None = None


@dataclass
class SolverContext:
    provider: Provider
    config: GenerateConfig


Solver = Callable[[TaskState, SolverContext], Awaitable[TaskState]]


def system_message(text: str) -> Solver:
    """Prepend a system message to the conversation."""

    async def solve(state: TaskState, ctx: SolverContext) -> TaskState:
        state.messages.insert(0, Message(role="system", content=text))
        return state

    return solve


def generate(config: GenerateConfig | None = None) -> Solver:
    """Call the model on the current messages. `config` overrides the run's config."""

    async def solve(state: TaskState, ctx: SolverContext) -> TaskState:
        cfg = config or ctx.config
        output = await ctx.provider.generate(state.messages, cfg)
        state.output = output
        state.messages.append(Message(role="assistant", content=output.text))
        return state

    return solve


def chain(*solvers: Solver) -> Solver:
    """Run solvers in order."""

    async def solve(state: TaskState, ctx: SolverContext) -> TaskState:
        for s in solvers:
            state = await s(state, ctx)
        return state

    return solve


def templated_prompt(system: str | None, user: str, extra: dict[str, str] | None = None) -> Solver:
    """Build the conversation from templates.

    `{name}` placeholders are filled from the sample's dataset row, `{input}` (the input
    column) and `extra` (such as `{labels}`). The row travels on the user message as
    metadata, so providers such as `http` can use any column.
    """

    async def solve(state: TaskState, ctx: SolverContext) -> TaskState:
        row = state.sample.metadata.get("row") or {}
        values = {**row, "input": state.sample.input, **(extra or {})}
        messages: list[Message] = []
        rendered_system = render(system, values).strip() if system else ""
        if rendered_system:
            messages.append(Message(role="system", content=rendered_system))
        messages.append(Message(role="user", content=render(user, values), metadata={"row": row}))
        state.messages = messages
        return state

    return solve


def initial_state(sample: Sample) -> TaskState:
    """Start a state from a sample: the sample input becomes the first user message."""
    row = sample.metadata.get("row")
    metadata = {"row": row} if row else {}
    return TaskState(sample=sample, messages=[Message(role="user", content=sample.input, metadata=metadata)])

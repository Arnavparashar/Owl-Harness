"""Solvers: composable steps that turn a sample into a model output.

A solver is an async callable `(state, ctx) -> state`. `state.messages` is the full
conversation so far, which is what a future multi-turn solver would extend.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from owl.providers.base import Provider
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


def initial_state(sample: Sample) -> TaskState:
    """Start a state from a sample: the sample input becomes the first user message."""
    return TaskState(sample=sample, messages=[Message(role="user", content=sample.input)])

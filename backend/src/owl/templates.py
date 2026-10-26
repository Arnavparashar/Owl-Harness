"""Prompt templates.

Prompt templates use `{name}` placeholders: `{input}`, `{labels}`, or any dataset column such
as `{question}`. Only names that are known are replaced, so literal braces in a prompt
(JSON examples, code) are left alone. Write `{{` or `}}` for a literal brace next to a name.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_PLACEHOLDER = re.compile(r"\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_ .-]*)\}")


def render(template: str, values: Mapping[str, Any]) -> str:
    """Fill `{name}` placeholders whose name is in `values`; leave everything else as written."""

    def sub(m: re.Match[str]) -> str:
        token = m.group(0)
        if token == "{{":
            return "{"
        if token == "}}":
            return "}"
        name = m.group(1)
        if name in values:
            v = values[name]
            return "" if v is None else str(v)
        return token

    return _PLACEHOLDER.sub(sub, template)


def placeholders(template: str) -> list[str]:
    """Names referenced by a template, in order of first use."""
    seen: list[str] = []
    for m in _PLACEHOLDER.finditer(template):
        name = m.group(1)
        if name and name not in seen:
            seen.append(name)
    return seen

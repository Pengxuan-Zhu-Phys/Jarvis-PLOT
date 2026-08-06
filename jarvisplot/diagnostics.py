#!/usr/bin/env python3

"""Structured diagnostics shared by every Jarvis-PLOT validation surface.

This module is deliberately dependency-free (stdlib only) so any layer -- config
loading, schema validation, column planning, render health -- can report through
it without creating an import cycle, and so ``jplot validate`` can run without
importing matplotlib.

A diagnostic is not just an error message. Every one of them carries the four
things an agent (or a human) needs to act:

- ``path``       where in the YAML the problem is (``$.Figures[0].layers[1]``)
- ``message``    what constraint was violated
- ``suggestion`` what edit to make next
- ``fix``        the same edit in machine-applicable form, when it is mechanical
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

__all__ = [
    "Diagnostic",
    "DiagnosticBag",
    "Fix",
    "LEVELS",
    "did_you_mean",
    "join_path",
    "path_of",
]


LEVELS = ("error", "warning", "info")

#: Mechanical repairs are applied by default; heuristic ones need an opt-in flag.
CONFIDENCE = ("certain", "heuristic")


# --------------------------------------------------------------------------- #
# YAML path helpers
# --------------------------------------------------------------------------- #


def join_path(*parts: Any) -> str:
    """Build a JSONPath-ish YAML location from mapping keys and list indices.

    ``join_path("Figures", 0, "layers", 1, "coordinates")`` ->
    ``"$.Figures[0].layers[1].coordinates"``.

    Callers may pass an already-built path as the first part; it is extended
    rather than re-rooted.
    """
    out = ""
    for part in parts:
        if isinstance(part, int):
            out += f"[{part}]"
            continue
        text = str(part)
        if not out and text.startswith("$"):
            out = text
            continue
        out += f".{text}" if out else f"$.{text}"
    return out or "$"


def path_of(base: str, *parts: Any) -> str:
    """Extend an existing path. Convenience wrapper over :func:`join_path`."""
    return join_path(base or "$", *parts)


# --------------------------------------------------------------------------- #
# did-you-mean
# --------------------------------------------------------------------------- #


def did_you_mean(
    word: str,
    candidates: Iterable[str],
    *,
    limit: int = 3,
    cutoff: float = 0.6,
) -> list[str]:
    """Closest spellings of ``word`` among ``candidates``.

    Case-insensitive first pass catches the ``Layers:`` / ``layers:`` class of
    mistake, which plain ``difflib`` ratio ranking can rank below unrelated keys.
    """
    pool = [str(c) for c in candidates]
    if not word or not pool:
        return []

    lowered = word.lower()
    exact_case_fold = [c for c in pool if c.lower() == lowered and c != word]
    if exact_case_fold:
        return exact_case_fold[:limit]

    return difflib.get_close_matches(word, pool, n=limit, cutoff=cutoff)


# --------------------------------------------------------------------------- #
# Diagnostic model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Fix:
    """A machine-applicable repair for one diagnostic.

    ``op`` is one of ``rename_key`` / ``set_value`` / ``remove_key`` /
    ``move_key``. ``confidence="certain"`` fixes are applied by ``--fix``;
    ``heuristic`` ones require ``--fix-unsafe``.
    """

    op: str
    path: str
    to: Any = None
    old: Any = None
    confidence: str = "certain"

    def __post_init__(self) -> None:
        if self.confidence not in CONFIDENCE:
            raise ValueError(
                f"Fix.confidence must be one of {CONFIDENCE}, got {self.confidence!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "path": self.path,
            "from": self.old,
            "to": self.to,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class Diagnostic:
    """One actionable finding, addressed to a specific place in the YAML."""

    code: str
    level: str
    path: str
    message: str
    suggestion: str = ""
    example: str | None = None
    fix: Fix | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"Diagnostic.level must be one of {LEVELS}, got {self.level!r}")

    @property
    def is_error(self) -> bool:
        return self.level == "error"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "level": self.level,
            "path": self.path,
            "message": self.message,
            "suggestion": self.suggestion,
        }
        if self.example is not None:
            payload["example"] = self.example
        if self.fix is not None:
            payload["fix"] = self.fix.to_dict()
        if self.context:
            payload["context"] = self.context
        return payload

    def render_human(self) -> str:
        """One diagnostic as the human console renders it."""
        lines = [f"{self.code} [{self.level}] {self.path}", f"  {self.message}"]
        if self.suggestion:
            lines.append(f"  -> {self.suggestion}")
        if self.example:
            indented = "\n".join(f"     {line}" for line in self.example.splitlines())
            lines.append("  example:")
            lines.append(indented)
        return "\n".join(lines)


class DiagnosticBag:
    """Collector that lets one pass report every problem instead of the first.

    The whole point of ``jplot validate`` is that a ten-figure config converges
    in one round instead of ten, so validation code must add to a bag and keep
    going rather than raise on first contact.
    """

    def __init__(self, items: Sequence[Diagnostic] | None = None) -> None:
        self._items: list[Diagnostic] = list(items or ())

    # -- collection ------------------------------------------------------- #

    def add(self, diagnostic: Diagnostic) -> Diagnostic:
        self._items.append(diagnostic)
        return diagnostic

    def error(self, code: str, path: str, message: str, **kwargs: Any) -> Diagnostic:
        return self.add(Diagnostic(code=code, level="error", path=path, message=message, **kwargs))

    def warning(self, code: str, path: str, message: str, **kwargs: Any) -> Diagnostic:
        return self.add(Diagnostic(code=code, level="warning", path=path, message=message, **kwargs))

    def info(self, code: str, path: str, message: str, **kwargs: Any) -> Diagnostic:
        return self.add(Diagnostic(code=code, level="info", path=path, message=message, **kwargs))

    def extend(self, items: Iterable[Diagnostic]) -> None:
        self._items.extend(items)

    # -- inspection ------------------------------------------------------- #

    def __iter__(self) -> Iterator[Diagnostic]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self._items if d.level == "error"]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [d for d in self._items if d.level == "warning"]

    @property
    def ok(self) -> bool:
        """True when nothing blocks a render."""
        return not self.errors

    def fixable(self, *, include_heuristic: bool = False) -> list[Diagnostic]:
        return [
            d
            for d in self._items
            if d.fix is not None
            and (include_heuristic or d.fix.confidence == "certain")
        ]

    # -- output ----------------------------------------------------------- #

    def to_list(self) -> list[dict[str, Any]]:
        return [d.to_dict() for d in self.sorted()]

    def sorted(self) -> list[Diagnostic]:
        """Errors first, then by YAML path so output order is deterministic."""
        rank = {"error": 0, "warning": 1, "info": 2}
        return sorted(self._items, key=lambda d: (rank[d.level], d.path, d.code))

    def summary_rows(self) -> list[tuple[str, str, str]]:
        """Compact ``(code, path, message)`` table shown before full diagnostics."""
        return [(d.code, d.path, d.message) for d in self.sorted()]

    def render_human(self) -> str:
        return "\n\n".join(d.render_human() for d in self.sorted())

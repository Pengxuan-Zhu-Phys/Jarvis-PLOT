#!/usr/bin/env python3

"""Verb routing for the agent-facing side of ``jplot``.

``jplot <file>`` is the **render** path (not a verb). That is intentional:
``Jarvis2 plot`` forwards argv wholesale to ``jplot``, so a ``run`` verb would
surface as ``Jarvis2 plot run …`` and collide with HEP's ``Jarvis2 run``
(scan) muscle memory. See DR-08 in ``docs/roadmap/V2_DEV_LEDGER.md``.

This package sits *in front* of the legacy flat parser and claims argv only
when the first token is a registered non-render verb. Handlers are imported
lazily so ``jplot validate`` never pulls in matplotlib.
"""

from __future__ import annotations

import importlib
import sys
from typing import Callable, Sequence

__all__ = [
    "RESERVED_NON_VERBS",
    "VERBS",
    "is_verb",
    "route",
    "verb_names",
]


#: verb name -> "module:function". The function takes ``(argv, *, prog)`` and
#: returns a process exit code.
VERBS: dict[str, str] = {
    "validate": "jarvisplot.verbs.validate:run",
    "cap": "jarvisplot.verbs.cap:run",
    "data": "jarvisplot.verbs.data:run",
    "dryrun": "jarvisplot.verbs.dryrun:run",
    "doctor": "jarvisplot.verbs.doctor:run",
    "template": "jarvisplot.verbs.template:run",
    "suggest": "jarvisplot.verbs.suggest:run",
    "explain": "jarvisplot.verbs.explain:run",
    "config": "jarvisplot.verbs.config_cmd:run",
    "man": "jarvisplot.verbs.man:run",
}

#: Tokens that must never become verbs or silent aliases. Map to a short
#: user-facing reason (printed on stderr, exit 2).
RESERVED_NON_VERBS: dict[str, str] = {
    "run": (
        "Render is the bare path: `jplot <file>` "
        "(same as `Jarvis2 plot <file>`). "
        "There is no `jplot run` — under Jarvis2, `plot run` would blur "
        "'run a scan' with 'render a figure'."
    ),
    "context": (
        "unknown command 'context'. "
        "Use the jplot CLI for full usage and information "
        "(`jplot -h`, `jplot man --json`, `jplot cap --json`)."
    ),
}

#: Bare tokens that are not agent verbs but are still legal CLI heads owned by
#: the render/legacy path (must not be "unknown command").
LEGACY_COMMANDS: frozenset[str] = frozenset({"flowchart"})


def verb_names() -> list[str]:
    return sorted(VERBS)


def is_verb(token: str) -> bool:
    return token in VERBS


def _load(target: str) -> Callable[..., int]:
    module_name, _, attr = target.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def route(argv: Sequence[str], *, prog: str = "jplot") -> tuple[bool, int]:
    """Dispatch ``argv`` if it starts with a registered verb.

    Returns ``(handled, exit_code)``. ``handled=False`` means the caller should
    fall through to the legacy ``jplot <file>`` path -- including for
    ``flowchart``, which is still owned by :mod:`jarvisplot.core`.

    A config file whose name collides with a verb is reachable as
    ``jplot ./validate``; the bare token always wins for the verb, because a
    silently-different meaning is the exact failure mode this track exists to
    remove.

    The reserved token ``run`` is **rejected** (not aliased) so
    ``Jarvis2 plot run scene.yaml`` cannot silently mean render.

    Bare unknown words (e.g. ``jplot whaat``) are rejected with did-you-mean
    instead of being treated as a missing YAML path.
    """
    tokens = list(argv)
    if not tokens:
        return False, 0

    head = tokens[0]
    if head in RESERVED_NON_VERBS:
        print(f"{prog}: {RESERVED_NON_VERBS[head]}", file=sys.stderr)
        return True, 2

    if is_verb(head):
        handler = _load(VERBS[head])
        return True, handler(tokens[1:], prog=f"{prog} {head}")

    # flowchart (and any other legacy head) falls through to core.
    if head in LEGACY_COMMANDS:
        return False, 0

    if _looks_like_unknown_command(head):
        from ..diagnostics import did_you_mean

        near = did_you_mean(
            head, verb_names() + list(RESERVED_NON_VERBS) + sorted(LEGACY_COMMANDS)
        )
        hint = f"; did you mean {near[0]!r}?" if near else ""
        print(
            f"{prog}: unknown command {head!r}{hint}\n"
            f"  try `{prog} -h` or `{prog} man --json`",
            file=sys.stderr,
        )
        return True, 2

    return False, 0


def _looks_like_unknown_command(token: str) -> bool:
    """True for bare tokens that are almost certainly not config paths."""
    from pathlib import Path

    text = str(token).strip()
    if not text or text in {"-h", "--help", "-v", "--version"}:
        return False
    if text.startswith("-"):
        return False
    # Path-like → let the render path handle existence errors.
    if any(sep in text for sep in ("/", "\\")):
        return False
    if text.startswith("."):
        return False
    lower = text.lower()
    if any(
        lower.endswith(ext)
        for ext in (
            ".yaml",
            ".yml",
            ".json",
            ".csv",
            ".parquet",
            ".h5",
            ".hdf5",
            ".hdf",
        )
    ):
        return False
    try:
        if Path(text).expanduser().exists():
            return False
    except Exception:
        pass
    # Single bare word without extension and not on disk → unknown command.
    return True

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
}


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
    """
    tokens = list(argv)
    if not tokens:
        return False, 0

    head = tokens[0]
    if head in RESERVED_NON_VERBS:
        print(f"{prog}: {RESERVED_NON_VERBS[head]}", file=sys.stderr)
        return True, 2

    if not is_verb(head):
        return False, 0

    handler = _load(VERBS[head])
    return True, handler(tokens[1:], prog=f"{prog} {head}")

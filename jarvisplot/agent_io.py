#!/usr/bin/env python3

"""The single JSON wire format every agent-facing ``jplot`` verb speaks.

Contract (``docs/specs/AGENT_DATA_API.md`` §2): each agent verb prints **exactly
one** JSON object on stdout; humans and logs go to stderr. This module owns the
envelope shape and the exit-code mapping so no verb hand-rolls either.

One extension over the frozen §2 shape: a top-level ``diagnostics`` array.
The spec's singular ``error`` cannot express "the config has nine problems",
which is the whole point of one-round validation. ``error`` is kept and still
means "the verb itself could not run".
"""

from __future__ import annotations

import json
import sys
from typing import Any, IO, Iterable, Sequence

from .diagnostics import Diagnostic, DiagnosticBag

__all__ = [
    "API_VERSION",
    "EXIT_FAILED",
    "EXIT_OK",
    "EXIT_USAGE",
    "emit",
    "envelope",
    "error_payload",
    "exit_code_for",
    "system_exit_code",
]


#: Bumped only on a breaking wire change. Agents cache it per session and refuse
#: on mismatch rather than falling back to text parsing.
API_VERSION = 1

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def system_exit_code(exc: BaseException) -> int:
    """Map ``SystemExit`` from argparse (including ``-h`` → 0) to a process code.

    Note: ``exc.code or EXIT_USAGE`` is wrong because successful help exits with
    code ``0``, which is falsy in Python.
    """
    code = getattr(exc, "code", None)
    if code is None:
        return EXIT_OK
    if isinstance(code, int):
        return code
    # argparse may pass a string message with status implied non-zero
    return EXIT_USAGE


def error_payload(exc_or_type: Any, message: str | None = None) -> dict[str, str]:
    """Normalize an exception or a type name into the ``{type, message}`` shape."""
    if isinstance(exc_or_type, BaseException):
        return {
            "type": type(exc_or_type).__name__,
            "message": str(exc_or_type) or type(exc_or_type).__name__,
        }
    return {"type": str(exc_or_type), "message": message or str(exc_or_type)}


def _normalize_diagnostics(
    diagnostics: DiagnosticBag | Iterable[Diagnostic] | Sequence[dict] | None,
) -> list[dict[str, Any]]:
    if diagnostics is None:
        return []
    if isinstance(diagnostics, DiagnosticBag):
        return diagnostics.to_list()
    out: list[dict[str, Any]] = []
    for item in diagnostics:
        out.append(item.to_dict() if isinstance(item, Diagnostic) else dict(item))
    return out


def envelope(
    kind: str,
    ok: bool | None,
    data: Any = None,
    diagnostics: DiagnosticBag | Iterable[Diagnostic] | Sequence[dict] | None = None,
    error: dict[str, str] | BaseException | None = None,
) -> dict[str, Any]:
    """Build the one object a verb is allowed to print.

    ``ok`` is the verb's own verdict; it is *not* derived from ``diagnostics``,
    because some verbs (``describe``) legitimately succeed while reporting
    warnings, and others (``validate``) report ``ok=false`` purely because the
    config has errors.

    Tri-state ``ok``:
    - ``True``  — full pass
    - ``False`` — real failure (bad config / empty ledger that dryrun fully checked)
    - ``None``  — partial coverage (e.g. heavy transforms skipped); not a failure
    """
    if isinstance(error, BaseException):
        error = error_payload(error)
    if ok is None:
        ok_value: bool | None = None
    else:
        ok_value = bool(ok)
    return {
        "api_version": API_VERSION,
        "kind": kind,
        "ok": ok_value,
        "data": data if data is not None else {},
        "diagnostics": _normalize_diagnostics(diagnostics),
        "error": error,
    }


def exit_code_for(env: dict[str, Any]) -> int:
    """Map an envelope to a process exit code (0 ok|partial / 1 failed / 2 usage)."""
    # Partial (ok is null) is not a process failure — agent should read status/coverage.
    if env.get("ok") is not False:
        return EXIT_OK
    error = env.get("error") or {}
    if str(error.get("type", "")) in {"UsageError", "ArgumentError"}:
        return EXIT_USAGE
    return EXIT_FAILED


def emit(env: dict[str, Any], stream: IO[str] | None = None) -> int:
    """Write the envelope as one line of JSON and return the exit code.

    ``default=str`` keeps numpy scalars and Paths from turning a successful verb
    into a serialization crash; agents get a string rather than nothing.
    """
    handle = stream if stream is not None else sys.stdout
    json.dump(env, handle, ensure_ascii=False, default=str)
    handle.write("\n")
    handle.flush()
    return exit_code_for(env)

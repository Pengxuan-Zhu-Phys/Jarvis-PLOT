#!/usr/bin/env python3

"""Turn ``jsonschema`` validation errors into diagnostics an agent can act on.

Raw jsonschema messages are addressed to a schema author, not to whoever wrote
the YAML: ``Additional properties are not allowed ('Layers' was unexpected)``
names the violated keyword but not the edit to make. Every translation here owes
the caller three things -- the YAML path, the *nearest legal spelling*, and,
when the repair is mechanical, a machine-applicable ``fix``.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from jsonschema.exceptions import ValidationError

from .diagnostics import Diagnostic, Fix, did_you_mean, join_path

__all__ = ["diagnostics_for_errors", "diagnostic_for_error"]


_UNEXPECTED_KEYS = re.compile(r"\('([^']+(?:', '[^']+)*)' (?:was|were) unexpected\)")
_MAX_LISTED_KEYS = 12


def _path(error: ValidationError, *extra: Any) -> str:
    return join_path(*list(error.absolute_path), *extra)


def _allowed_keys(schema: Mapping[str, Any]) -> list[str]:
    return sorted(schema.get("properties", {}))


def _format_allowed(keys: Iterable[str]) -> str:
    keys = list(keys)
    if not keys:
        return "(none)"
    if len(keys) > _MAX_LISTED_KEYS:
        return ", ".join(keys[:_MAX_LISTED_KEYS]) + f", … ({len(keys)} total)"
    return ", ".join(keys)


def _example_of(schema: Mapping[str, Any]) -> str | None:
    example = schema.get("x-jarvis-example")
    return example if isinstance(example, str) else None


# --------------------------------------------------------------------------- #
# Per-keyword translations
# --------------------------------------------------------------------------- #


def _unknown_key(error: ValidationError) -> list[Diagnostic]:
    match = _UNEXPECTED_KEYS.search(error.message)
    keys = match.group(1).split("', '") if match else []
    allowed = _allowed_keys(error.schema)
    example = _example_of(error.schema)

    out: list[Diagnostic] = []
    for key in keys:
        near = did_you_mean(key, allowed)
        # A key that differs only in case is a certain rename; anything else is
        # a guess, and `--fix` must not apply guesses by default.
        case_only = bool(near) and near[0].lower() == key.lower()
        suggestion = (
            f"Rename it to {near[0]!r}."
            if near
            else f"Remove it. Allowed keys here: {_format_allowed(allowed)}."
        )
        fix = (
            Fix(
                op="rename_key",
                path=_path(error, key),
                old=key,
                to=near[0],
                confidence="certain" if case_only else "heuristic",
            )
            if near
            else None
        )
        out.append(
            Diagnostic(
                code="JP-SCH-001",
                level="error",
                path=_path(error, key),
                message=f"unknown key {key!r}",
                suggestion=suggestion,
                example=example,
                fix=fix,
                context={"allowed_keys": allowed, "did_you_mean": near},
            )
        )
    return out


def _missing_key(error: ValidationError) -> Diagnostic:
    match = re.search(r"'([^']+)' is a required property", error.message)
    key = match.group(1) if match else "the required key"
    return Diagnostic(
        code="JP-SCH-002",
        level="error",
        path=_path(error),
        message=f"missing required key {key!r}",
        suggestion=f"Add {key} here.",
        example=_example_of(error.schema),
        context={"allowed_keys": _allowed_keys(error.schema)},
    )


def _bad_enum(error: ValidationError) -> Diagnostic:
    allowed = [str(v) for v in error.validator_value]
    value = error.instance
    near = did_you_mean(str(value), allowed)
    hint = f" Did you mean {near[0]!r}?" if near else ""
    return Diagnostic(
        code="JP-SCH-003",
        level="error",
        path=_path(error),
        message=f"{value!r} is not a valid value here.{hint}",
        suggestion=(
            f"Use one of: {_format_allowed(allowed)}."
            if not near
            else f"Change it to {near[0]!r}; spelling and case matter."
        ),
        fix=(
            Fix(op="set_value", path=_path(error), old=value, to=near[0], confidence="heuristic")
            if near
            else None
        ),
        context={"allowed_values": allowed, "did_you_mean": near},
    )


_TYPE_ADVICE = {
    "object": "Rewrite this as an indented YAML mapping (key: value pairs).",
    "array": "Rewrite this as a YAML list whose items start with '- '.",
    "string": "Quote the value so YAML reads it as text.",
    "number": "Use a plain numeric scalar.",
    "integer": "Use a whole number.",
    "boolean": "Use true or false.",
}


def _bad_type(error: ValidationError) -> Diagnostic:
    expected = error.validator_value
    names = [expected] if isinstance(expected, str) else list(expected)
    advice = _TYPE_ADVICE.get(names[0], "Correct the value type.")
    return Diagnostic(
        code="JP-SCH-004",
        level="error",
        path=_path(error),
        message=f"expected {' or '.join(names)}, got {type(error.instance).__name__}",
        suggestion=advice,
        example=_example_of(error.schema),
    )


def _no_branch_matched(error: ValidationError) -> Diagnostic:
    """``oneOf`` degrades to 'not valid under any of the given schemas'.

    Useless on its own, so describe the branches instead of quoting the keyword.
    """
    branches = error.validator_value or []
    shapes: list[str] = []
    for branch in branches:
        if not isinstance(branch, Mapping):
            continue
        if "$ref" in branch:
            shapes.append(str(branch["$ref"]).rsplit("/", 1)[-1])
        elif "type" in branch:
            shapes.append(str(branch["type"]))
    described = " or ".join(dict.fromkeys(shapes)) or "one of the documented shapes"
    return Diagnostic(
        code="JP-SCH-005",
        level="error",
        path=_path(error),
        message=f"value does not match any accepted shape here (expected {described})",
        suggestion=f"Rewrite the value as {described}.",
        example=_example_of(error.schema),
    )


def _generic(error: ValidationError) -> Diagnostic:
    return Diagnostic(
        code="JP-SCH-009",
        level="error",
        path=_path(error),
        message=error.message,
        suggestion="Correct the value so it satisfies the constraint stated above.",
        example=_example_of(error.schema),
    )


_HANDLERS = {
    "required": _missing_key,
    "enum": _bad_enum,
    "type": _bad_type,
    "oneOf": _no_branch_matched,
    "anyOf": _no_branch_matched,
}


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def diagnostic_for_error(error: ValidationError) -> list[Diagnostic]:
    if error.validator == "additionalProperties":
        return _unknown_key(error)
    handler = _HANDLERS.get(str(error.validator))
    return [handler(error) if handler else _generic(error)]


def diagnostics_for_errors(errors: Iterable[ValidationError]) -> list[Diagnostic]:
    """Flatten a validator's errors, deduplicated by (code, path, message).

    Sub-schema errors nested under ``oneOf`` are dropped: reporting both the
    branch failures and the union failure buries the actionable one.
    """
    out: list[Diagnostic] = []
    seen: set[tuple[str, str, str]] = set()
    for error in sorted(errors, key=lambda e: (list(map(str, e.absolute_path)), str(e.validator))):
        for diagnostic in diagnostic_for_error(error):
            key = (diagnostic.code, diagnostic.path, diagnostic.message)
            if key in seen:
                continue
            seen.add(key)
            out.append(diagnostic)
    return out

#!/usr/bin/env python3

"""Apply :class:`~jarvisplot.diagnostics.Fix` operations to a parsed config.

Only structural edits that do not need ruamel (comment-preserving rewrite is
``jplot config set``, still DR-02). B8's ``--fix`` rewrites with PyYAML and is
honest about dropping comments.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Iterable, Sequence

from .diagnostics import Diagnostic, Fix

__all__ = [
    "apply_fixes",
    "parse_yaml_path",
    "planned_fixes",
]


_SEGMENT = re.compile(
    r"""
    \.([A-Za-z_][A-Za-z0-9_]*)   # .key
    | \[(\d+)\]                    # [0]
    | ^\$                          # root marker
    | ^([A-Za-z_][A-Za-z0-9_]*)    # bare leading key (rare)
    """,
    re.VERBOSE,
)


def parse_yaml_path(path: str) -> list[str | int]:
    """Turn ``$.Figures[0].layers[1].method`` into ``['Figures', 0, 'layers', 1, 'method']``."""
    text = str(path or "").strip()
    if not text or text == "$":
        return []
    if text.startswith("$"):
        text = text[1:]
    parts: list[str | int] = []
    pos = 0
    while pos < len(text):
        if text[pos] == ".":
            pos += 1
            m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", text[pos:])
            if not m:
                raise ValueError(f"bad path segment in {path!r} at {pos}")
            parts.append(m.group(1))
            pos += len(m.group(1))
            continue
        if text[pos] == "[":
            m = re.match(r"\[(\d+)\]", text[pos:])
            if not m:
                raise ValueError(f"bad index in {path!r} at {pos}")
            parts.append(int(m.group(1)))
            pos += len(m.group(0))
            continue
        raise ValueError(f"cannot parse path {path!r} at {pos}: {text[pos:]!r}")
    return parts


def planned_fixes(
    diagnostics: Iterable[Diagnostic],
    *,
    include_heuristic: bool = False,
) -> list[Fix]:
    """Fixes that ``--fix`` would apply, in path order for stable diffs."""
    fixes: list[Fix] = []
    for diagnostic in diagnostics:
        fix = diagnostic.fix
        if fix is None:
            continue
        if fix.confidence == "heuristic" and not include_heuristic:
            continue
        if fix.confidence == "certain" or include_heuristic:
            fixes.append(fix)
    # deeper paths first so renames under a parent still resolve if a sibling moves
    fixes.sort(key=lambda f: (-len(parse_yaml_path(f.path)), f.path, f.op))
    return fixes


def apply_fixes(config: Any, fixes: Sequence[Fix]) -> tuple[Any, list[dict[str, Any]]]:
    """Return ``(new_config, applied)`` where ``applied`` lists successful ops."""
    tree = copy.deepcopy(config)
    applied: list[dict[str, Any]] = []
    for fix in fixes:
        try:
            _apply_one(tree, fix)
            applied.append(fix.to_dict())
        except Exception as exc:
            applied.append({**fix.to_dict(), "error": str(exc)})
    return tree, applied


def _apply_one(tree: Any, fix: Fix) -> None:
    parts = parse_yaml_path(fix.path)
    if not parts:
        raise ValueError("cannot apply a fix at the document root")

    if fix.op == "rename_key":
        *parent_parts, old_key = parts
        if not isinstance(old_key, str):
            raise ValueError(f"rename_key path must end in a key, got {fix.path!r}")
        parent = _resolve(tree, parent_parts)
        if not isinstance(parent, dict):
            raise ValueError(f"parent of {fix.path!r} is not a mapping")
        key = old_key if old_key in parent else (fix.old if fix.old in parent else None)
        if key is None or key not in parent:
            raise KeyError(f"key {old_key!r} not found at {fix.path}")
        new_key = fix.to
        if new_key is None:
            raise ValueError("rename_key requires Fix.to")
        if new_key in parent and new_key != key:
            raise KeyError(f"target key {new_key!r} already exists")
        # preserve key order: rebuild
        rebuilt: dict[str, Any] = {}
        for k, v in parent.items():
            if k == key:
                rebuilt[str(new_key)] = v
            else:
                rebuilt[k] = v
        parent.clear()
        parent.update(rebuilt)
        return

    if fix.op == "set_value":
        *parent_parts, leaf = parts
        parent = _resolve(tree, parent_parts)
        if isinstance(leaf, int):
            if not isinstance(parent, list):
                raise ValueError(f"index path into non-list at {fix.path!r}")
            parent[leaf] = fix.to
            return
        if not isinstance(parent, dict):
            raise ValueError(f"parent of {fix.path!r} is not a mapping")
        parent[leaf] = fix.to
        return

    if fix.op == "remove_key":
        *parent_parts, leaf = parts
        parent = _resolve(tree, parent_parts)
        if isinstance(leaf, int):
            if not isinstance(parent, list):
                raise ValueError(f"index path into non-list at {fix.path!r}")
            del parent[leaf]
            return
        if not isinstance(parent, dict):
            raise ValueError(f"parent of {fix.path!r} is not a mapping")
        key = leaf if leaf in parent else fix.old
        if key not in parent:
            raise KeyError(f"key {leaf!r} not found at {fix.path}")
        del parent[key]
        return

    if fix.op == "move_key":
        # move_key: path is source; to is destination path string or key name
        raise NotImplementedError("move_key is reserved for config set; not in B8")

    raise ValueError(f"unknown fix op {fix.op!r}")


def _resolve(tree: Any, parts: Sequence[str | int]) -> Any:
    node = tree
    for part in parts:
        if isinstance(part, int):
            node = node[part]
        else:
            node = node[part]
    return node

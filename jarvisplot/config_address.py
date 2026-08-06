#!/usr/bin/env python3

"""Named YAML addressing for ``jplot config get`` (F3).

Addresses prefer stable names over list indices so reordering layers does not
break agent edits::

    Figures[EggBox].layers[_density].style.cmap
    DataSet[samples].path
    Figures[0].layers[1].method          # numeric still accepted

List items are located by their ``name`` field when the selector is not an
integer. Unnamed items are addressable as ``_layer0``, ``_layer1``, … in
document order (auto-alias, not written back).
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "AddressError",
    "parse_address",
    "resolve_address",
    "format_address",
]


class AddressError(KeyError):
    """Raised when an address cannot be resolved."""


_SEGMENT = re.compile(
    r"""
    (?P<key>[A-Za-z_][A-Za-z0-9_]*)
    (?:\[(?P<sel>[^\]]+)\])?
    """,
    re.VERBOSE,
)


def parse_address(address: str) -> list[tuple[str, str | None]]:
    """Parse ``Figures[f1].layers[pts].style`` into ``[(key, selector|None), …]``."""
    text = str(address or "").strip()
    if text.startswith("$."):
        text = text[2:]
    elif text.startswith("$"):
        text = text[1:].lstrip(".")
    if not text:
        raise AddressError("empty address")

    parts: list[tuple[str, str | None]] = []
    pos = 0
    while pos < len(text):
        if text[pos] == ".":
            pos += 1
            continue
        m = _SEGMENT.match(text, pos)
        if not m:
            raise AddressError(f"cannot parse address {address!r} at {text[pos:]!r}")
        key = m.group("key")
        sel = m.group("sel")
        parts.append((key, sel))
        pos = m.end()
    if not parts:
        raise AddressError(f"cannot parse address {address!r}")
    return parts


def resolve_address(config: Any, address: str) -> Any:
    """Return the value at ``address`` inside a parsed config mapping."""
    node = config
    trail: list[str] = []
    for key, sel in parse_address(address):
        trail.append(f"{key}[{sel}]" if sel is not None else key)
        if not isinstance(node, dict):
            raise AddressError(
                f"cannot enter {key!r} under non-mapping at {'.'.join(trail[:-1]) or '$'}"
            )
        if key not in node:
            raise AddressError(f"missing key {key!r} at {'.'.join(trail[:-1]) or '$'}")
        node = node[key]
        if sel is None:
            continue
        node = _select(node, sel, path=".".join(trail))
    return node


def _select(node: Any, selector: str, *, path: str) -> Any:
    sel = str(selector).strip()
    # name=foo form
    if sel.startswith("name="):
        sel = sel[5:].strip().strip("'\"")

    if isinstance(node, list):
        if re.fullmatch(r"-?\d+", sel):
            idx = int(sel)
            try:
                return node[idx]
            except IndexError as exc:
                raise AddressError(f"index {idx} out of range at {path}") from exc
        # match by name / auto alias
        for index, item in enumerate(node):
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if isinstance(name, str) and name == sel:
                return item
            if not (isinstance(name, str) and name.strip()):
                if sel == f"_layer{index}" or sel == f"L{index}":
                    return item
        available = _list_names(node)
        raise AddressError(
            f"no item named {sel!r} at {path}; available: {available}"
        )

    if isinstance(node, dict):
        if sel in node:
            return node[sel]
        raise AddressError(f"missing key {sel!r} at {path}")

    raise AddressError(f"cannot apply selector [{sel}] at {path}")


def _list_names(items: list[Any]) -> list[str]:
    out: list[str] = []
    for index, item in enumerate(items):
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"].strip():
            out.append(item["name"])
        else:
            out.append(f"_layer{index}")
    return out


def format_address(parts: list[tuple[str, str | None]]) -> str:
    chunks: list[str] = []
    for key, sel in parts:
        if sel is None:
            chunks.append(key)
        else:
            chunks.append(f"{key}[{sel}]")
    return ".".join(chunks)

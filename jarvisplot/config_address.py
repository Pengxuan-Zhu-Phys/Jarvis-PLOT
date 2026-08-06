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
    "set_address",
    "delete_address",
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


def set_address(config: Any, address: str, value: Any) -> None:
    """Mutate ``config`` so ``address`` holds ``value`` (in place)."""
    parent, last_key, last_sel = _parent_and_leaf(config, address, create=True)
    if last_sel is None:
        if not isinstance(parent, dict):
            raise AddressError(f"cannot set {last_key!r} on non-mapping")
        parent[last_key] = value
        return
    container = parent[last_key] if isinstance(parent, dict) else parent
    if isinstance(container, list):
        idx = _index_in_list(container, last_sel)
        container[idx] = value
        return
    if isinstance(container, dict):
        container[last_sel] = value
        return
    raise AddressError(f"cannot set selector [{last_sel}] under {last_key!r}")


def delete_address(config: Any, address: str) -> None:
    """Remove the value at ``address`` (in place)."""
    parent, last_key, last_sel = _parent_and_leaf(config, address, create=False)
    if last_sel is None:
        if not isinstance(parent, dict) or last_key not in parent:
            raise AddressError(f"missing key {last_key!r}")
        del parent[last_key]
        return
    container = parent[last_key] if isinstance(parent, dict) else parent
    if isinstance(container, list):
        idx = _index_in_list(container, last_sel)
        del container[idx]
        return
    if isinstance(container, dict):
        if last_sel not in container:
            raise AddressError(f"missing key {last_sel!r}")
        del container[last_sel]
        return
    raise AddressError(f"cannot delete selector [{last_sel}] under {last_key!r}")


def _parent_and_leaf(
    config: Any, address: str, *, create: bool
) -> tuple[Any, str, str | None]:
    parts = parse_address(address)
    if not parts:
        raise AddressError("empty address")
    *head, (last_key, last_sel) = parts
    node: Any = config
    trail: list[str] = []
    for key, sel in head:
        trail.append(f"{key}[{sel}]" if sel is not None else key)
        if not isinstance(node, dict):
            raise AddressError(
                f"cannot enter {key!r} under non-mapping at {'.'.join(trail[:-1]) or '$'}"
            )
        if key not in node:
            if not create:
                raise AddressError(f"missing key {key!r} at {'.'.join(trail[:-1]) or '$'}")
            node[key] = [] if sel is not None else {}
        node = node[key]
        if sel is None:
            continue
        node = _select(node, sel, path=".".join(trail))
    if not isinstance(node, dict) and last_sel is None:
        raise AddressError(f"parent of {last_key!r} is not a mapping")
    if last_sel is not None and last_key not in node and create and isinstance(node, dict):
        node[last_key] = {}
    if last_sel is not None and isinstance(node, dict) and last_key not in node:
        raise AddressError(f"missing key {last_key!r}")
    return node, last_key, last_sel


def _index_in_list(items: list[Any], selector: str) -> int:
    sel = str(selector).strip()
    if sel.startswith("name="):
        sel = sel[5:].strip().strip("'\"")
    if re.fullmatch(r"-?\d+", sel):
        idx = int(sel)
        if idx < 0 or idx >= len(items):
            raise AddressError(f"index {idx} out of range")
        return idx
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name == sel:
            return index
        if not (isinstance(name, str) and name.strip()):
            if sel in {f"_layer{index}", f"L{index}"}:
                return index
    raise AddressError(
        f"no item named {sel!r}; available: {_list_names(items)}"
    )


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

#!/usr/bin/env python3

"""YAML load/dump with optional comment preservation (ruamel).

Falls back to PyYAML when ``ruamel.yaml`` is not installed; callers must treat
``comments_preserved`` honestly in that case.
"""

from __future__ import annotations

from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Any

__all__ = ["load_yaml_doc", "dump_yaml_doc", "has_ruamel"]


def _is_yaml_container(value: Any) -> bool:
    return isinstance(value, Mapping) or (
        isinstance(value, (list, tuple))
        and not isinstance(value, (str, bytes, bytearray))
    )


def _is_leaf_collection(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(not _is_yaml_container(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(not _is_yaml_container(item) for item in value)
    return False


def _mark_ruamel_leaf_flow(value: Any) -> None:
    """Mark scalar-only mappings/sequences as YAML flow-style in place."""
    if isinstance(value, Mapping):
        for item in value.values():
            _mark_ruamel_leaf_flow(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _mark_ruamel_leaf_flow(item)

    if _is_leaf_collection(value):
        format_attributes = getattr(value, "fa", None)
        if format_attributes is not None:
            format_attributes.set_flow_style()


def _to_ruamel_node(value: Any) -> Any:
    """Convert plain containers to ruamel containers and mark leaf flow style."""
    from ruamel.yaml.comments import CommentedMap, CommentedSeq

    if isinstance(value, Mapping):
        node = CommentedMap(
            (key, _to_ruamel_node(item)) for key, item in value.items()
        )
    elif isinstance(value, (list, tuple)):
        node = CommentedSeq(_to_ruamel_node(item) for item in value)
    else:
        return value

    if _is_leaf_collection(node):
        node.fa.set_flow_style()
    return node


class _FlowDict(dict):
    """PyYAML-only marker for a scalar-only mapping."""


class _FlowList(list):
    """PyYAML-only marker for a scalar-only sequence."""


def _to_pyyaml_node(value: Any) -> Any:
    """Copy plain containers with markers for leaf flow-style collections."""
    if isinstance(value, Mapping):
        converted = {
            key: _to_pyyaml_node(item) for key, item in value.items()
        }
        return _FlowDict(converted) if _is_leaf_collection(value) else converted
    if isinstance(value, (list, tuple)):
        converted = [_to_pyyaml_node(item) for item in value]
        return _FlowList(converted) if _is_leaf_collection(value) else converted
    return value


def has_ruamel() -> bool:
    try:
        import ruamel.yaml  # noqa: F401

        return True
    except Exception:
        return False


def load_yaml_doc(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Return ``(document, meta)`` where meta has ``engine`` and ``comments_preserved``."""
    text = Path(path).expanduser().read_text(encoding="utf-8")
    if has_ruamel():
        from ruamel.yaml import YAML

        yaml = YAML(typ="rt")
        yaml.preserve_quotes = True
        yaml.default_flow_style = False
        doc = yaml.load(StringIO(text))
        return doc, {"engine": "ruamel", "comments_preserved": True, "raw_text": text}
    import yaml as pyyaml

    doc = pyyaml.safe_load(text)
    return doc, {"engine": "pyyaml", "comments_preserved": False, "raw_text": text}


def dump_yaml_doc(doc: Any, *, meta: dict[str, Any] | None = None) -> str:
    """Serialize ``doc`` using the same engine preference as load."""
    prefer_ruamel = bool(meta is None or meta.get("engine") == "ruamel")
    if prefer_ruamel and has_ruamel():
        from ruamel.yaml import YAML

        if hasattr(doc, "fa"):
            _mark_ruamel_leaf_flow(doc)
        else:
            doc = _to_ruamel_node(doc)
        yaml = YAML(typ="rt")
        yaml.preserve_quotes = True
        yaml.default_flow_style = False
        yaml.width = 4096
        buf = StringIO()
        yaml.dump(doc, buf)
        return buf.getvalue()
    import yaml as pyyaml

    class _FlowDumper(pyyaml.SafeDumper):
        pass

    _FlowDumper.add_representer(
        _FlowDict,
        lambda dumper, value: dumper.represent_mapping(
            "tag:yaml.org,2002:map", value, flow_style=True
        ),
    )
    _FlowDumper.add_representer(
        _FlowList,
        lambda dumper, value: dumper.represent_sequence(
            "tag:yaml.org,2002:seq", value, flow_style=True
        ),
    )
    return pyyaml.dump(
        _to_pyyaml_node(doc),
        Dumper=_FlowDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )

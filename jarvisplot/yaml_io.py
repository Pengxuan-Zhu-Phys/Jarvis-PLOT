#!/usr/bin/env python3

"""YAML load/dump with optional comment preservation (ruamel).

Falls back to PyYAML when ``ruamel.yaml`` is not installed; callers must treat
``comments_preserved`` honestly in that case.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

__all__ = ["load_yaml_doc", "dump_yaml_doc", "has_ruamel"]


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

        yaml = YAML(typ="rt")
        yaml.preserve_quotes = True
        yaml.default_flow_style = False
        yaml.width = 4096
        buf = StringIO()
        yaml.dump(doc, buf)
        return buf.getvalue()
    import yaml as pyyaml

    return pyyaml.safe_dump(
        doc,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )

#!/usr/bin/env python3

"""Load declarative ``manual_cards`` for ``jplot man``.

Cards are pure data (YAML). Renderers must not invent prose here.

Live topics (methods catalog + per-method pages) are assembled from
:mod:`jarvisplot.man_methods` / capabilities so they cannot drift from the
runtime registry.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .diagnostics import did_you_mean

__all__ = [
    "ManCatalogError",
    "cards_dir",
    "index_payload",
    "list_topics",
    "load_card",
    "load_manifest",
    "resolve_topic",
]


class ManCatalogError(Exception):
    """Card missing, corrupt, or topic unknown."""


def cards_dir() -> Path:
    return Path(__file__).resolve().parent / "manual_cards"


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    path = cards_dir() / "manifest.yaml"
    if not path.is_file():
        raise ManCatalogError(f"manual manifest missing: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ManCatalogError("manifest.yaml must be a mapping")
    topics = data.get("topics")
    if not isinstance(topics, list) or not topics:
        raise ManCatalogError("manifest.topics must be a non-empty list")
    data.setdefault("aliases", {})
    data.setdefault("schema_version", 1)
    return data


def list_topics() -> list[str]:
    """Static card topics + live method/transform catalogs and per-item pages."""
    from .man_methods import list_method_names, method_topic_id
    from .man_transforms import list_transform_names, transform_topic_id

    static = [str(t) for t in load_manifest()["topics"]]
    for extra in ("methods", "transforms"):
        if extra not in static:
            static.append(extra)
    return (
        static
        + [method_topic_id(n) for n in list_method_names()]
        + [transform_topic_id(n) for n in list_transform_names()]
    )


def list_index_topics() -> list[str]:
    """Topics shown on the man index (no per-method/transform explosion)."""
    static = [str(t) for t in load_manifest()["topics"]]
    for extra in ("methods", "transforms"):
        if extra not in static:
            static.append(extra)
    return static


def resolve_topic(raw: str | None) -> str | None:
    """Map user token → canonical topic id. ``None`` means index."""
    if raw is None or str(raw).strip() == "":
        return None
    token = str(raw).strip().lower()
    manifest = load_manifest()
    topics = {str(t).lower(): str(t) for t in manifest["topics"]}
    # inject live catalog ids
    topics.setdefault("methods", "methods")
    topics.setdefault("transforms", "transforms")

    if token in topics:
        return topics[token]

    aliases = manifest.get("aliases") or {}
    if isinstance(aliases, dict) and token in aliases:
        target = str(aliases[token])
        if target.lower() in topics:
            return topics[target.lower()]
        from .man_methods import resolve_method_token
        from .man_transforms import resolve_transform_token

        for resolver in (resolve_method_token, resolve_transform_token):
            try:
                live = resolver(target)
                if live is not None:
                    return live
            except ValueError as exc:
                raise ManCatalogError(str(exc)) from exc
        return target

    # Live method / transform pages
    from .man_methods import list_method_names, resolve_method_token
    from .man_transforms import list_transform_names, resolve_transform_token

    for resolver in (resolve_transform_token, resolve_method_token):
        try:
            live = resolver(token)
        except ValueError as exc:
            raise ManCatalogError(str(exc)) from exc
        if live is not None:
            return live

    known = (
        list(topics.keys())
        + [str(a) for a in (aliases or {})]
        + ["methods", "transforms"]
        + list_method_names()
        + [f"method.{n}" for n in list_method_names()]
        + list_transform_names()
        + [f"transform.{n}" for n in list_transform_names()]
    )
    suggestions = did_you_mean(token, known)
    hint = f"; did you mean {suggestions[0]!r}?" if suggestions else ""
    raise ManCatalogError(
        f"unknown man topic {raw!r}{hint}. "
        "Use the jplot CLI for full usage and information "
        "(`jplot -h`, `jplot man --json`, `jplot cap --json`)."
    )


def load_card(topic: str) -> dict[str, Any]:
    topic_id = resolve_topic(topic)
    if topic_id is None:
        raise ManCatalogError("load_card requires a topic id")

    # Live method pages (no YAML card file required)
    from .man_methods import (
        is_method_topic,
        load_method_card,
        load_methods_catalog_card,
        parse_method_topic,
    )
    from .man_transforms import (
        is_transform_topic,
        load_transform_card,
        load_transforms_catalog_card,
        parse_transform_topic,
    )

    if is_method_topic(topic_id):
        if topic_id == "methods":
            return load_methods_catalog_card()
        name = parse_method_topic(topic_id)
        if not name:
            raise ManCatalogError(f"invalid method topic {topic_id!r}")
        try:
            return load_method_card(name)
        except KeyError as exc:
            raise ManCatalogError(str(exc)) from exc

    if is_transform_topic(topic_id):
        if topic_id == "transforms":
            return load_transforms_catalog_card()
        name = parse_transform_topic(topic_id)
        if not name:
            raise ManCatalogError(f"invalid transform topic {topic_id!r}")
        try:
            return load_transform_card(name)
        except KeyError as exc:
            raise ManCatalogError(str(exc)) from exc

    path = cards_dir() / f"{topic_id}.yaml"
    if not path.is_file():
        raise ManCatalogError(f"manual card missing for topic {topic_id!r}: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ManCatalogError(f"card {topic_id!r} must be a mapping")
    data.setdefault("id", topic_id)
    if str(data.get("id")) != topic_id:
        data["id"] = topic_id
    for required in ("title", "summary", "role"):
        if required not in data or data[required] in (None, ""):
            raise ManCatalogError(f"card {topic_id!r} missing required field {required!r}")
    data.setdefault("priority", 100)
    data.setdefault("see_also", [])
    data.setdefault("related_cli", [])
    data.setdefault("live_sources", [])
    data.setdefault("human", {})
    data.setdefault("agent", {})
    return data


def index_payload() -> dict[str, Any]:
    """Structured index for agent + shared metadata for human index."""
    from .man_methods import list_method_names, method_topic_id
    from .man_transforms import list_transform_names, transform_topic_id

    manifest = load_manifest()
    topics_out = []
    for tid in list_index_topics():
        card = load_card(str(tid))
        topics_out.append(
            {
                "id": card["id"],
                "title": card["title"],
                "summary": card["summary"],
                "priority": int(card.get("priority", 100)),
                "role": card["role"],
            }
        )
    topics_out.sort(key=lambda t: (t["priority"], t["id"]))

    method_names = list_method_names()
    transform_names = list_transform_names()
    return {
        "schema_version": manifest.get("schema_version", 1),
        "aliases": dict(manifest.get("aliases") or {}),
        "topics": topics_out,
        "methods": [
            {"id": method_topic_id(n), "name": n, "man": f"jplot man {n}"}
            for n in method_names
        ],
        "transforms": [
            {
                "id": transform_topic_id(n),
                "name": n,
                "man": f"jplot man transform.{n}",
            }
            for n in transform_names
        ],
    }

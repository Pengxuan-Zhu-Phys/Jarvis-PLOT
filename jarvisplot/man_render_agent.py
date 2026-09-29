#!/usr/bin/env python3

"""Assemble agent JSON payloads for ``jplot man --json``."""

from __future__ import annotations

from typing import Any

from .man_catalog import index_payload, load_card

__all__ = ["agent_index_data", "agent_topic_data"]


def _slug(text: str) -> str:
    """Lowercase, hyphenated id from a panel title."""
    out = []
    for ch in str(text).strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")


def agent_index_data() -> dict[str, Any]:
    idx = index_payload()
    return {
        "topic": None,
        "audience": "agent",
        "title": "Jarvis-PLOT manual index",
        "summary": "Discover with data/cap; write YAML yourself; judge with validate/dryrun/doctor.",
        "role": "catalog",
        "priority": 0,
        "topics": idx["topics"],
        "aliases": idx["aliases"],
        "methods": idx.get("methods") or [],
        "transforms": idx.get("transforms") or [],
        "related_cli": [
            {
                "argv": ["jplot", "man", "workflow", "--json"],
                "why": "recommended coding-agent loop",
            },
            {
                "argv": ["jplot", "man", "methods", "--json"],
                "why": "all layer drawing methods",
            },
            {
                "argv": ["jplot", "man", "transforms", "--json"],
                "why": "all pipeline transform contracts",
            },
            {
                "argv": ["jplot", "man", "transform.make_interp_2d", "--json"],
                "why": "example heavy transform page",
            },
            {
                "argv": ["jplot", "man", "scatter", "--json"],
                "why": "example per-method man page",
            },
            {
                "argv": ["jplot", "data", "describe", "<file>", "--json"],
                "why": "column-name whitelist",
            },
            {
                "argv": ["jplot", "cap", "all", "--json"],
                "why": "string whitelist",
            },
            {
                "argv": ["jplot", "cap", "datasets", "--json"],
                "why": "dataset and generated-data contracts",
            },
            {
                "argv": ["jplot", "doctor", "<yaml>", "--json"],
                "why": "validate + dryrun gate",
            },
        ],
        "live_sources": [
            {
                "verb": "cap.all",
                "truth": "methods/styles/cmaps/funcs/transforms/types/datasets/cli",
            },
            {"verb": "cap.methods", "truth": "drawing method contracts"},
            {"verb": "cap.transforms", "truth": "pipeline transform contracts"},
            {"verb": "cap.datasets", "truth": "dataset and generated-data contracts"},
            {"verb": "data.describe", "truth": "real file columns only"},
        ],
        "write_yaml": False,
        "card_version": idx.get("schema_version", 1),
    }


def agent_topic_data(topic: str) -> dict[str, Any]:
    card = load_card(topic)
    agent = card.get("agent") if isinstance(card.get("agent"), dict) else {}
    human = card.get("human") if isinstance(card.get("human"), dict) else {}

    sections = list(agent.get("sections") or [])
    if not sections and isinstance(human.get("panels"), list):
        # Project short human panels into structured sections.
        seen_ids: dict[str, int] = {}
        for block in human["panels"]:
            if not isinstance(block, dict):
                continue
            kind = str(block.get("kind") or "text")
            title = str(block.get("title") or kind)
            # A card with three `notes` panels needs three addressable ids,
            # so the title carries the id when there is one.
            base = _slug(title) or kind
            seen_ids[base] = seen_ids.get(base, 0) + 1
            sec: dict[str, Any] = {
                "id": base if seen_ids[base] == 1 else f"{base}-{seen_ids[base]}",
                "title": title,
                "kind": kind,
            }
            if "body" in block:
                sec["body"] = block["body"]
            if "items" in block:
                sec["items"] = block["items"]
            sections.append(sec)

    data: dict[str, Any] = {
        "topic": card["id"],
        "audience": "agent",
        "title": card["title"],
        "summary": card["summary"],
        "role": card.get("role"),
        "priority": int(card.get("priority", 100)),
        "see_also": list(card.get("see_also") or []),
        "related_cli": list(card.get("related_cli") or []),
        "live_sources": list(card.get("live_sources") or []),
        "sections": sections,
        "body_markdown": agent.get("body_markdown") or "",
        "examples": list(agent.get("examples") or []),
        "diagnostics": list(agent.get("diagnostics") or []),
        "anti_patterns": list(agent.get("anti_patterns") or []),
        "schema_ids": list(agent.get("schema_ids") or []),
        "write_yaml": False,
        "card_version": int(card.get("card_version", 1)),
    }
    if isinstance(agent.get("verification"), dict) and agent["verification"]:
        data["verification"] = agent["verification"]
    # Live method / transform pages attach structured contract blobs.
    if "method" in card:
        data["method"] = card["method"]
    if "methods" in card:
        data["methods"] = card["methods"]
    if "transform" in card:
        data["transform"] = card["transform"]
    if "transforms" in card:
        data["transforms"] = card["transforms"]
    if card["id"] == "legend":
        from .legend_cards import legend_card_catalog

        data["legend_cards"] = legend_card_catalog()
    return data

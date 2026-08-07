#!/usr/bin/env python3

"""Live transform manuals: ``jplot man transforms`` / ``jplot man transform.filter``."""

from __future__ import annotations

from typing import Any

from .diagnostics import did_you_mean
from .transform_contracts import TRANSFORM_NAMES, contract_for, list_contracts

__all__ = [
    "TRANSFORM_TOPIC_PREFIX",
    "is_transform_topic",
    "list_transform_names",
    "load_transform_card",
    "load_transforms_catalog_card",
    "parse_transform_topic",
    "resolve_transform_token",
    "transform_topic_id",
]

TRANSFORM_TOPIC_PREFIX = "transform."
_CATALOG_ID = "transforms"


def list_transform_names() -> list[str]:
    return list(TRANSFORM_NAMES)


def transform_topic_id(name: str) -> str:
    return f"{TRANSFORM_TOPIC_PREFIX}{name}"


def is_transform_topic(topic_id: str) -> bool:
    return topic_id == _CATALOG_ID or str(topic_id).startswith(TRANSFORM_TOPIC_PREFIX)


def parse_transform_topic(topic_id: str) -> str | None:
    if topic_id == _CATALOG_ID:
        return None
    if topic_id.startswith(TRANSFORM_TOPIC_PREFIX):
        return topic_id[len(TRANSFORM_TOPIC_PREFIX) :]
    return None


def resolve_transform_token(token: str) -> str | None:
    text = str(token).strip().lower()
    if not text:
        return None
    if text in {_CATALOG_ID, "transform", "pipeline", "pipeline-steps"}:
        return _CATALOG_ID
    if text.startswith(TRANSFORM_TOPIC_PREFIX):
        name = text[len(TRANSFORM_TOPIC_PREFIX) :]
        if name in TRANSFORM_NAMES:
            return transform_topic_id(name)
        near = did_you_mean(name, list(TRANSFORM_NAMES))
        hint = f"; did you mean {near[0]!r}?" if near else ""
        raise ValueError(
            f"unknown transform {name!r}{hint}. "
            "Use the jplot CLI for full usage and information "
            "(`jplot -h`, `jplot man --json`, `jplot cap --json`)."
        )
    if text in TRANSFORM_NAMES:
        return transform_topic_id(text)
    return None


def load_transforms_catalog_card() -> dict[str, Any]:
    rows = []
    for c in list_contracts():
        name = c["name"]
        req = ", ".join(c.get("required") or {}) or "—"
        form = c.get("form") or "—"
        rows.append(
            {
                "name": name,
                "form": form,
                "required_keys": list(c.get("required") or {}),
                "optional_keys": list(c.get("optional") or {}),
                "input": c.get("input"),
                "output": c.get("output"),
                "topic": transform_topic_id(name),
                "description": c.get("description") or "",
            }
        )
    list_items = [
        f"{r['name']} [{r['form']}] required={r['required_keys'] or '—'} "
        f"→ jplot man transform.{r['name']}"
        for r in rows
    ]
    return {
        "id": _CATALOG_ID,
        "title": "Transforms (all)",
        "summary": (
            f"All {len(rows)} pipeline transform steps with live contracts. "
            "Open one: jplot man transform.profile / man filter."
        ),
        "role": "catalog",
        "priority": 37,
        "see_also": ["layer-method", "methods", "type-posterior-2d", "type-profile-2d"],
        "related_cli": [
            {"argv": ["jplot", "cap", "transforms", "--json"], "why": "same live contracts"},
            {
                "argv": ["jplot", "man", "transform.make_interp_2d", "--json"],
                "why": "example heavy transform page",
            },
            {"argv": ["jplot", "man", "transform.filter", "--json"], "why": "example light step"},
        ],
        "live_sources": [
            {"verb": "cap.transforms", "truth": "transform_contracts + schema vocabulary"},
        ],
        "human": {
            "panels": [
                {
                    "kind": "overview",
                    "title": "Authority",
                    "body": (
                        "Transform steps sit on DataSet or layer data[].transform.\n"
                        "Contracts are live (not delegated stubs). "
                        "Heavy steps (profile / density / interp) are skipped in dryrun."
                    ),
                },
                {
                    "kind": "steps",
                    "title": f"Transforms ({len(rows)})",
                    "items": list_items,
                },
            ]
        },
        "agent": {
            "body_markdown": (
                "## Transforms\n\n"
                "Use `jplot man transform.<name> --json` for required/optional/defaults/enums/examples.\n"
            ),
            "sections": [
                {
                    "id": "transforms",
                    "title": "All transforms",
                    "kind": "transforms_table",
                    "items": rows,
                }
            ],
            "anti_patterns": [
                "Inventing transform names not in this list",
                "Assuming dryrun fully executes profile/posterior_density/make_interp_2d",
            ],
        },
        "transforms": rows,
        "card_version": 1,
        "_live": True,
    }


def load_transform_card(name: str) -> dict[str, Any]:
    c = contract_for(name)
    if c is None:
        near = did_you_mean(name, list(TRANSFORM_NAMES))
        hint = f"; did you mean {near[0]!r}?" if near else ""
        raise KeyError(
            f"unknown transform {name!r}{hint}. "
            "Use the jplot CLI for full usage and information "
            "(`jplot -h`, `jplot man --json`, `jplot cap --json`)."
        )

    topic = transform_topic_id(name)
    required = c.get("required") or {}
    optional = c.get("optional") or {}
    defaults = c.get("defaults") or {}
    enums = c.get("enums") or {}
    examples = c.get("examples") or []
    notes = list(c.get("notes") or [])
    if c.get("owner"):
        notes = notes + [f"owner: {c['owner']}"]

    yaml_example = ""
    if examples:
        yaml_example = str(examples[0].get("yaml") or "")

    contract_body = (
        f"transform: {name}\n"
        f"form: {c.get('form')}\n"
        f"input → output: {c.get('input')} → {c.get('output')}\n"
        f"required: {', '.join(required) or '—'}\n"
        f"optional: {', '.join(optional) or '—'}\n"
        f"defaults: {defaults or '—'}\n"
        f"enums: {enums or '—'}"
    )
    if c.get("value"):
        contract_body += f"\nvalue schema: {c.get('value')}"

    return {
        "id": topic,
        "title": f"transform: {name}",
        "summary": c.get("description") or f"Pipeline step {name!r}.",
        "role": "reference",
        "priority": 37,
        "see_also": ["transforms", "layer-method", "methods"],
        "related_cli": [
            {"argv": ["jplot", "cap", "transforms", "--json"], "why": "full transform table"},
            {"argv": ["jplot", "man", "transforms", "--json"], "why": "catalog"},
            {"argv": ["jplot", "data", "eval", "<expr>", "--data", "<file>", "--json"], "why": "sandbox expressions"},
        ],
        "live_sources": [
            {"verb": "cap.transforms", "truth": f"contract for {name}"},
        ],
        "human": {
            "panels": [
                {"kind": "overview", "title": "Contract", "body": contract_body},
                *(
                    [
                        {
                            "kind": "yaml",
                            "title": "Example",
                            "lexer": "yaml",
                            "body": yaml_example,
                        }
                    ]
                    if yaml_example
                    else []
                ),
                {
                    "kind": "notes",
                    "title": "Notes",
                    "items": notes
                    or [
                        "Place under DataSet[].transform or layers[].data[].transform.",
                    ],
                },
            ]
        },
        "agent": {
            "body_markdown": (
                f"## transform `{name}`\n\n{c.get('description')}\n\n"
                f"- form: `{c.get('form')}`\n"
                f"- input/output: `{c.get('input')}` → `{c.get('output')}`\n"
                f"- owner: `{c.get('owner')}`\n"
            ),
            "sections": [
                {
                    "id": "contract",
                    "title": "Contract",
                    "kind": "mapping",
                    "body": {
                        "name": name,
                        "form": c.get("form"),
                        "required": required,
                        "optional": optional,
                        "defaults": defaults,
                        "enums": enums,
                        "value": c.get("value") or {},
                        "input": c.get("input"),
                        "output": c.get("output"),
                    },
                }
            ],
            "examples": examples,
            "anti_patterns": [
                f"Using undeclared keys for {name}",
                "Expecting dryrun to execute heavy transforms",
            ],
        },
        "transform": {
            "name": name,
            "form": c.get("form"),
            "required": required,
            "optional": optional,
            "defaults": defaults,
            "enums": enums,
            "value": c.get("value") or {},
            "input": c.get("input"),
            "output": c.get("output"),
            "owner": c.get("owner"),
            "examples": examples,
        },
        "card_version": 1,
        "_live": True,
    }

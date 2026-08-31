#!/usr/bin/env python3

"""Live method manuals for ``jplot man methods`` / ``jplot man scatter``.

Content is assembled from :func:`jarvisplot.capabilities.section` (``methods``),
so the man pages cannot drift from the runtime registry. Optional prose tips
live in :mod:`jarvisplot.manual_cards.method_notes` (YAML map); missing tips
are fine — contract data alone is enough.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .diagnostics import did_you_mean

__all__ = [
    "METHOD_TOPIC_PREFIX",
    "is_method_topic",
    "list_method_names",
    "load_method_card",
    "load_methods_catalog_card",
    "method_topic_id",
    "parse_method_topic",
    "resolve_method_token",
]

METHOD_TOPIC_PREFIX = "method."
_CATALOG_ID = "methods"


def list_method_names() -> list[str]:
    from .capabilities import section

    return [str(m["name"]) for m in section("methods") if m.get("name")]


@lru_cache(maxsize=1)
def _methods_by_name() -> dict[str, dict[str, Any]]:
    from .capabilities import section

    return {str(m["name"]): dict(m) for m in section("methods") if m.get("name")}


@lru_cache(maxsize=1)
def _method_notes() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "manual_cards" / "method_notes.yaml"
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def method_topic_id(name: str) -> str:
    return f"{METHOD_TOPIC_PREFIX}{name}"


def is_method_topic(topic_id: str) -> bool:
    return topic_id == _CATALOG_ID or str(topic_id).startswith(METHOD_TOPIC_PREFIX)


def parse_method_topic(topic_id: str) -> str | None:
    """Return bare method name for ``method.scatter``; None for catalog."""
    if topic_id == _CATALOG_ID:
        return None
    if topic_id.startswith(METHOD_TOPIC_PREFIX):
        return topic_id[len(METHOD_TOPIC_PREFIX) :]
    return None


def resolve_method_token(token: str) -> str | None:
    """Map user token → topic id if it names methods catalog or a method."""
    text = str(token).strip().lower()
    if not text:
        return None
    if text in {_CATALOG_ID, "method", "drawing-methods", "plot-methods"}:
        return _CATALOG_ID
    if text.startswith(METHOD_TOPIC_PREFIX):
        name = text[len(METHOD_TOPIC_PREFIX) :]
        if name in _methods_by_name():
            return method_topic_id(name)
        near = did_you_mean(name, list_method_names())
        hint = f"; did you mean {near[0]!r}?" if near else ""
        raise ValueError(
            f"unknown drawing method {name!r}{hint}. "
            "Use the jplot CLI for full usage and information "
            "(`jplot -h`, `jplot man --json`, `jplot cap --json`)."
        )
    # bare method name: scatter, pcolormesh, …
    if text in _methods_by_name():
        return method_topic_id(text)
    return None


def load_methods_catalog_card() -> dict[str, Any]:
    methods = list(_methods_by_name().values())
    methods.sort(key=lambda m: str(m.get("name") or ""))
    rows = []
    for m in methods:
        name = str(m["name"])
        coords = m.get("coordinates") or {}
        req = ", ".join(coords.get("required") or ()) or "—"
        opt = ", ".join(coords.get("optional") or ()) or "—"
        axes = ", ".join(m.get("axes_types") or ()) or "—"
        rows.append(
            {
                "name": name,
                "required": req,
                "optional": opt,
                "axes_types": axes,
                "mpl_method": m.get("mpl_method") or name,
                "topic": method_topic_id(name),
            }
        )

    list_items = [
        f"{r['name']}: required [{r['required']}]  optional [{r['optional']}]  "
        f"axes={r['axes_types']}  → jplot man {r['name']}"
        for r in rows
    ]

    return {
        "id": _CATALOG_ID,
        "title": "Drawing methods (all)",
        "summary": (
            f"All {len(rows)} layer method strings from the live registry. "
            "Open one with jplot man <method> (e.g. man scatter)."
        ),
        "role": "catalog",
        "priority": 38,
        "see_also": ["layer-method", "type-posterior-2d", "style-axes"],
        "related_cli": [
            {"argv": ["jplot", "cap", "methods", "--json"], "why": "same live method contracts"},
            {"argv": ["jplot", "man", "scatter", "--json"], "why": "example single-method page"},
            {"argv": ["jplot", "man", "layer-method", "--json"], "why": "coordinates contracts / traps"},
        ],
        "live_sources": [
            {"verb": "cap.methods", "truth": "METHOD_DISPATCH + method_contracts"},
        ],
        "human": {
            "panels": [
                {
                    "kind": "overview",
                    "title": "Authority",
                    "body": (
                        "This list is live from the runtime registry (same as "
                        "`jplot cap methods`). Do not invent method names.\n"
                        "Per-method pages: `jplot man scatter`, `jplot man pcolormesh`, …"
                    ),
                },
                {
                    "kind": "steps",
                    "title": f"Methods ({len(rows)})",
                    "items": list_items,
                },
                {
                    "kind": "notes",
                    "title": "How to use",
                    "items": [
                        "layer.method must be one of the names below.",
                        "coordinates keys must cover required axes for that method.",
                        "For type: macros (posterior_2d / profile_2d) you rarely name methods by hand.",
                    ],
                },
            ]
        },
        "agent": {
            "body_markdown": (
                "## Drawing methods\n\n"
                "Source of truth: `jplot cap methods --json`.\n"
                "Per method: `jplot man <name> --json` or `jplot man method.<name> --json`.\n"
            ),
            "sections": [
                {
                    "id": "methods",
                    "title": "All methods",
                    "kind": "methods_table",
                    "items": rows,
                }
            ],
            "examples": [],
            "anti_patterns": [
                "Inventing method strings not in this list",
                "Omitting required coordinates for a method",
            ],
        },
        "methods": rows,
        "card_version": 1,
        "_live": True,
    }


def load_method_card(name: str) -> dict[str, Any]:
    entry = _methods_by_name().get(name)
    if entry is None:
        near = did_you_mean(name, list_method_names())
        hint = f"; did you mean {near[0]!r}?" if near else ""
        raise KeyError(
            f"unknown drawing method {name!r}{hint}. "
            "Use the jplot CLI for full usage and information "
            "(`jplot -h`, `jplot man --json`, `jplot cap --json`)."
        )

    coords = entry.get("coordinates") or {}
    required = list(coords.get("required") or ())
    optional = list(coords.get("optional") or ())
    axes_types = list(entry.get("axes_types") or ())
    mpl = str(entry.get("mpl_method") or name)
    notes_map = _method_notes()
    tip = notes_map.get(name) if isinstance(notes_map.get(name), dict) else {}
    tip_summary = str(tip.get("summary") or "").strip()
    tip_notes = [str(x) for x in (tip.get("notes") or []) if str(x).strip()]
    tip_traps = [str(x) for x in (tip.get("traps") or []) if str(x).strip()]

    summary = tip_summary or (
        f"Layer method {name!r} → matplotlib {mpl!r}. "
        f"Required coordinates: {', '.join(required) or 'none'}; "
        f"optional: {', '.join(optional) or 'none'}."
    )

    yaml_body = _example_yaml(name, required=required, optional=optional)
    human_notes = tip_notes or [
        f"mpl backend: {mpl}",
        f"axes families: {', '.join(axes_types) or 'any'}",
        "Coordinate values are usually `{expr: <column_or_expression>}`.",
        "Do not put lim/scale on layer.coordinates — use frame or type: slots.",
    ]
    human_traps = tip_traps or [
        "Missing a required coordinate key → validate / render error.",
        "Inventing method aliases (e.g. 'scatterplot') will fail — use the exact name.",
    ]

    agent_examples = [{"title": "minimal layer", "yaml": yaml_body}]
    agent_sections = [
        {
            "id": "contract",
            "title": "Contract",
            "kind": "mapping",
            "body": {
                "name": name,
                "mpl_method": mpl,
                "axes_types": axes_types,
                "coordinates": {"required": required, "optional": optional},
            },
        },
        {
            "id": "notes",
            "title": "Notes",
            "kind": "notes",
            "items": human_notes,
        },
    ]
    verification: dict[str, Any] = {}
    if name == "corrplot":
        agent_examples = _corrplot_examples()
        verification = {
            "validate": {
                "argv": ["jplot", "validate", "<yaml>", "--json"],
                "purpose": "schema, references, data paths, columns and method contract",
            },
            "doctor": {
                "argv": ["jplot", "doctor", "<yaml>", "--json"],
                "purpose": "validate plus data loading and type expansion without rendering",
            },
            "render": {
                "argv": ["jplot", "<yaml>", "--report"],
                "purpose": "actual matplotlib render and render-health report",
            },
        }
        agent_sections.append(
            {
                "id": "verification",
                "title": "Verification",
                "kind": "commands",
                "items": [
                    {
                        "name": "validate",
                        "argv": ["jplot", "validate", "<yaml>", "--json"],
                        "purpose": "schema, references, data paths, columns, and method contract",
                    },
                    {
                        "name": "doctor",
                        "argv": ["jplot", "doctor", "<yaml>", "--json"],
                        "purpose": "validate plus data loading and type expansion without rendering",
                    },
                    {
                        "name": "render",
                        "argv": ["jplot", "<yaml>", "--report"],
                        "purpose": "actual matplotlib render and render-health report",
                    },
                ],
            }
        )

    related_cli = [
        {"argv": ["jplot", "cap", "methods", "--json"], "why": "full method table"},
        {"argv": ["jplot", "man", "methods", "--json"], "why": "all methods catalog"},
        {"argv": ["jplot", "data", "describe", "<file>", "--json"], "why": "column names for expr"},
    ]
    if name == "corrplot":
        related_cli.extend(
            [
                {"argv": ["jplot", "cap", "types", "--json"], "why": "correlation_matrix type and its compatible cards"},
                {"argv": ["jplot", "cap", "styles", "--json"], "why": "matrix/diamond axes and card options"},
                {"argv": ["jplot", "man", "type-correlation-matrix", "--json"], "why": "type-first YAML examples and card rules"},
                {"argv": ["jplot", "validate", "<yaml>", "--json"], "why": "cheap schema/reference/column gate"},
                {"argv": ["jplot", "doctor", "<yaml>", "--json"], "why": "validate plus data/type expansion without rendering"},
            ]
        )

    topic = method_topic_id(name)
    return {
        "id": topic,
        "title": f"method: {name}",
        "summary": summary,
        "role": "reference",
        "priority": 39,
        "see_also": ["methods", "layer-method", "style-axes"],
        "related_cli": related_cli,
        "live_sources": [
            {"verb": "cap.methods", "truth": f"contract for {name}"},
        ],
        "human": {
            "panels": [
                {
                    "kind": "overview",
                    "title": "Contract",
                    "body": (
                        f"method: {name}\n"
                        f"matplotlib: {mpl}\n"
                        f"axes_types: {', '.join(axes_types) or '—'}\n"
                        f"coordinates.required: {', '.join(required) or '—'}\n"
                        f"coordinates.optional: {', '.join(optional) or '—'}"
                    ),
                },
                {
                    "kind": "yaml",
                    "title": "Minimal layer",
                    "lexer": "yaml",
                    "body": yaml_body,
                },
                {
                    "kind": "notes",
                    "title": "Notes",
                    "items": human_notes,
                },
                {
                    "kind": "traps",
                    "title": "Traps",
                    "items": human_traps,
                },
            ]
        },
        "agent": {
            "body_markdown": (
                f"## method `{name}`\n\n"
                f"- matplotlib: `{mpl}`\n"
                f"- axes_types: {axes_types}\n"
                f"- required coordinates: {required}\n"
                f"- optional coordinates: {optional}\n\n"
                "Live source: `jplot cap methods --json`.\n"
                + (
                    "For correlation figures, use `jplot man type-correlation-matrix --json` "
                    "for the type contract and choose a matrix or diamond YAML example below.\n"
                    if name == "corrplot"
                    else ""
                )
            ),
            "sections": agent_sections,
            "examples": agent_examples,
            "anti_patterns": human_traps,
            "verification": verification,
        },
        "method": {
            "name": name,
            "mpl_method": mpl,
            "axes_types": axes_types,
            "coordinates": {"required": required, "optional": optional},
        },
        "card_version": 1,
        "_live": True,
    }


def _example_yaml(name: str, *, required: list[str], optional: list[str]) -> str:
    """Synthesize a minimal layer sketch from the coordinate contract."""
    if name == "corrplot":
        return _corrplot_examples()[0]["yaml"]
    # Prefer common HEP-ish placeholders by axis key.
    defaults = {
        "x": "m_A",
        "y": "tanb",
        "z": "density",
        "c": "LogL",
        "s": "1.0",
        "marker": "'o'",
        "height": "counts",
        "width": "counts",
        "bottom": "0",
        "left": "0",
        "u": "u",
        "v": "v",
    }
    lines = [
        "layers:",
        "  - name: layer0",
        "    data: [{source: samples}]",
        f"    method: {name}",
        "    coordinates:",
    ]
    for key in required:
        sample = defaults.get(key, key)
        if key == "marker":
            lines.append(f"      {key}: {sample}")
        else:
            lines.append(f"      {key}: {{expr: {sample}}}")
    # Show at most two optional keys as comments in agent body via including them
    for key in optional[:2]:
        sample = defaults.get(key, key)
        if key == "marker":
            lines.append(f"      # {key}: {sample}   # optional")
        else:
            lines.append(f"      # {key}: {{expr: {sample}}}   # optional")
    lines.append("    style: {}")
    return "\n".join(lines) + "\n"


def _corrplot_examples() -> list[dict[str, str]]:
    """Canonical type-first YAML few-shots for both corrplot cards."""
    return [
        {
            "title": "square correlation matrix",
            "yaml": r'''version: '0.3'
DataSet:
  - {name: samples, path: ./samples.csv, type: csv}
Figures:
  - name: correlation
    type: correlation_matrix
    data: samples
    variables: {exclude: [weight, label]}
    corrplot:
      method: circle
      type: upper
      order: hclust
      addrect: 3
    colorbar: {label: '$\rho$'}
''',
        },
        {
            "title": "diamond correlation matrix",
            "yaml": r'''version: '0.3'
DataSet:
  - {name: samples, path: ./samples.csv, type: csv}
Figures:
  - name: correlation
    type: correlation_matrix
    style: [corrplot, diamond]
    data: samples
    variables: {exclude: [weight, label]}
    corrplot:
      method: circle
      side: right
      stripe: alternate
      order: hclust
    colorbar: {label: '$\rho$'}
''',
        },
    ]

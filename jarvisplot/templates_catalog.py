#!/usr/bin/env python3

"""Type-first YAML templates + slot schemas for agent drafting (F1)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

__all__ = [
    "TEMPLATE_KINDS",
    "get_template",
    "list_templates",
    "render_template_yaml",
]


def list_templates() -> list[dict[str, Any]]:
    return [
        {
            "kind": kind,
            "title": spec["title"],
            "family": spec["family"],
            "requires": list(spec["requires"]),
            "description": spec["description"],
        }
        for kind, spec in TEMPLATE_KINDS.items()
    ]


def get_template(kind: str) -> dict[str, Any]:
    key = str(kind or "").strip()
    if key not in TEMPLATE_KINDS:
        raise KeyError(
            f"unknown template kind {kind!r}; choose one of: "
            + ", ".join(sorted(TEMPLATE_KINDS))
        )
    spec = deepcopy(TEMPLATE_KINDS[key])
    spec["kind"] = key
    return spec


def render_template_yaml(kind: str, *, values: dict[str, Any] | None = None) -> str:
    """Fill slot defaults with ``values`` and return YAML text."""
    import yaml

    spec = get_template(kind)
    slots = {s["name"]: s.get("default") for s in spec.get("slots") or []}
    if values:
        slots.update({k: v for k, v in values.items() if v is not None})
    body = deepcopy(spec["skeleton"])
    _apply_slots(body, slots)
    return yaml.safe_dump(
        body,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def _apply_slots(node: Any, slots: dict[str, Any]) -> None:
    """Replace string tokens like ``${x}`` in the skeleton tree."""
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                slot = value[2:-1]
                if slot in slots:
                    node[key] = slots[slot]
            else:
                _apply_slots(value, slots)
    elif isinstance(node, list):
        for item in node:
            _apply_slots(item, slots)


TEMPLATE_KINDS: dict[str, dict[str, Any]] = {
    "posterior_2d": {
        "title": "2D posterior density (type: posterior_2d)",
        "family": "rect",
        "requires": ["data", "x", "y", "weight"],
        "description": (
            "Voronoi/grid posterior density on a rect frame with optional HPD contours. "
            "Primary agent-facing figure type."
        ),
        "slots": [
            {
                "name": "name",
                "type": "string",
                "default": "posterior",
                "description": "Figure name / output stem",
            },
            {
                "name": "data",
                "type": "dataset_ref",
                "default": "samples",
                "description": "DataSet.name referenced by the figure",
            },
            {
                "name": "path",
                "type": "path",
                "default": "./samples.csv",
                "description": "Dataset file path",
            },
            {
                "name": "dtype",
                "type": "enum",
                "default": "csv",
                "enum": ["csv", "hdf5", "parquet"],
                "description": "DataSet.type",
            },
            {
                "name": "x",
                "type": "column",
                "default": "m_A",
                "source_hint": "parameter",
                "description": "X parameter column / expression",
            },
            {
                "name": "y",
                "type": "column",
                "default": "tanb",
                "source_hint": "parameter",
                "description": "Y parameter column / expression",
            },
            {
                "name": "weight",
                "type": "expression",
                "default": "exp(LogL)",
                "source_hint": "weight",
                "description": "Posterior weight expression (usually exp(LogL))",
            },
            {
                "name": "style",
                "type": "style_tokens",
                "default": ["a4paper_2x1", "rectcmap"],
                "description": "Style card tokens from jplot cap styles",
            },
            {
                "name": "xlim",
                "type": "lim",
                "default": None,
                "description": "Optional [lo, hi] for x; omit to let the type choose",
            },
            {
                "name": "ylim",
                "type": "lim",
                "default": None,
                "description": "Optional [lo, hi] for y",
            },
            {
                "name": "xscale",
                "type": "enum",
                "default": "linear",
                "enum": ["linear", "log"],
            },
            {
                "name": "yscale",
                "type": "enum",
                "default": "linear",
                "enum": ["linear", "log"],
            },
        ],
        "skeleton": {
            "version": "0.3",
            "DataSet": [
                {
                    "name": "${data}",
                    "path": "${path}",
                    "type": "${dtype}",
                }
            ],
            "Figures": [
                {
                    "name": "${name}",
                    "type": "posterior_2d",
                    "style": "${style}",
                    "data": "${data}",
                    "x": {"expr": "${x}"},
                    "y": {"expr": "${y}"},
                    "weight": {"expr": "${weight}"},
                    "frame": {
                        "ax": {
                            "xscale": "${xscale}",
                            "yscale": "${yscale}",
                        }
                    },
                }
            ],
            "output": {"dir": "./plots"},
        },
    },
    "profile_2d": {
        "title": "2D profile likelihood (type: profile_2d)",
        "family": "rect",
        "requires": ["data", "x", "y", "z"],
        "description": "Profile-likelihood reduction on a rect frame with optional CL contours.",
        "slots": [
            {"name": "name", "type": "string", "default": "profile", "description": "Figure name"},
            {"name": "data", "type": "dataset_ref", "default": "samples"},
            {"name": "path", "type": "path", "default": "./samples.csv"},
            {
                "name": "dtype",
                "type": "enum",
                "default": "csv",
                "enum": ["csv", "hdf5", "parquet"],
            },
            {"name": "x", "type": "column", "default": "m_A", "source_hint": "parameter"},
            {"name": "y", "type": "column", "default": "tanb", "source_hint": "parameter"},
            {
                "name": "z",
                "type": "column",
                "default": "LogL",
                "source_hint": "log_likelihood",
                "description": "Objective column (usually LogL)",
            },
            {
                "name": "style",
                "type": "style_tokens",
                "default": ["a4paper_2x1", "rectcmap"],
            },
            {"name": "xscale", "type": "enum", "default": "linear", "enum": ["linear", "log"]},
            {"name": "yscale", "type": "enum", "default": "linear", "enum": ["linear", "log"]},
        ],
        "skeleton": {
            "version": "0.3",
            "DataSet": [
                {"name": "${data}", "path": "${path}", "type": "${dtype}"}
            ],
            "Figures": [
                {
                    "name": "${name}",
                    "type": "profile_2d",
                    "style": "${style}",
                    "data": "${data}",
                    "x": {"expr": "${x}"},
                    "y": {"expr": "${y}"},
                    "z": {"expr": "${z}"},
                    "frame": {
                        "ax": {
                            "xscale": "${xscale}",
                            "yscale": "${yscale}",
                        }
                    },
                }
            ],
            "output": {"dir": "./plots"},
        },
    },
    "scatter_2d": {
        "title": "2D scatter with optional colour",
        "family": "rect",
        "requires": ["data", "x", "y"],
        "description": "Hand-written layer stack (no type macro) for quick diagnostics.",
        "slots": [
            {"name": "name", "type": "string", "default": "scatter"},
            {"name": "data", "type": "dataset_ref", "default": "samples"},
            {"name": "path", "type": "path", "default": "./samples.csv"},
            {
                "name": "dtype",
                "type": "enum",
                "default": "csv",
                "enum": ["csv", "hdf5", "parquet"],
            },
            {"name": "x", "type": "column", "default": "m_A"},
            {"name": "y", "type": "column", "default": "tanb"},
            {
                "name": "c",
                "type": "column",
                "default": "LogL",
                "description": "Optional colour column; set empty to disable",
            },
            {
                "name": "style",
                "type": "style_tokens",
                "default": ["a4paper_2x1", "rectcmap"],
            },
            {"name": "xscale", "type": "enum", "default": "linear", "enum": ["linear", "log"]},
            {"name": "yscale", "type": "enum", "default": "linear", "enum": ["linear", "log"]},
        ],
        "skeleton": {
            "version": "0.3",
            "DataSet": [
                {"name": "${data}", "path": "${path}", "type": "${dtype}"}
            ],
            "Figures": [
                {
                    "name": "${name}",
                    "style": "${style}",
                    "frame": {
                        "ax": {
                            "xscale": "${xscale}",
                            "yscale": "${yscale}",
                        }
                    },
                    "layers": [
                        {
                            "name": "pts",
                            "data": [{"source": "${data}"}],
                            "axes": "ax",
                            "method": "scatter",
                            "coordinates": {
                                "x": {"expr": "${x}"},
                                "y": {"expr": "${y}"},
                                "c": {"expr": "${c}"},
                            },
                            "style": {"s": 6, "marker": "."},
                            "colorbar": "axc",
                        }
                    ],
                }
            ],
            "output": {"dir": "./plots"},
        },
    },
}

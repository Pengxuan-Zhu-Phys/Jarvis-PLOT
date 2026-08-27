#!/usr/bin/env python3

"""Per-method coordinate contracts for YAML layers.

Authoritative table of which ``coordinates`` axes each ``method`` requires.
Schema files under ``schema/methods/`` and ``jplot validate`` both read this
module so the two cannot drift. Axes families still come from
:mod:`jarvisplot.Figure.method_registry`.
"""

from __future__ import annotations

from typing import Any, Mapping

from .Figure.method_registry import METHOD_DISPATCH, REGISTRY, normalize_method_key

__all__ = [
    "METHOD_COORDINATES",
    "contract_for",
    "missing_coordinates",
    "schema_payload",
]


# required / optional coordinate axis names (layer coordinates keys).
# ``alternatives`` is a list of full axis packs that also satisfy the method
# (ternary layers use left/right/bottom instead of x/y).
METHOD_COORDINATES: dict[str, dict[str, Any]] = {
    "plot": {
        "required": ("x", "y"),
        "optional": (),
        "alternatives": (("left", "right", "bottom"),),
    },
    "scatter": {
        "required": ("x", "y"),
        "optional": ("c", "s", "marker"),
        "alternatives": (("left", "right", "bottom"),),
    },
    "hist": {"required": ("x",), "optional": ("weights",)},
    "hist2d": {"required": ("x", "y"), "optional": ("weights",)},
    "stairs": {"required": ("x_lo", "x_hi", "y"), "optional": ()},
    "errorbar": {"required": ("x", "y"), "optional": ("xerr", "yerr")},
    "fill": {"required": ("x", "y"), "optional": ()},
    "fill_between": {"required": ("x", "y1", "y2"), "optional": ()},
    "fill_betweenx": {"required": ("y", "x1", "x2"), "optional": ()},
    "bar": {"required": ("x", "height"), "optional": ("bottom",)},
    "barh": {"required": ("y", "width"), "optional": ("left",)},
    "step": {"required": ("x", "y"), "optional": ()},
    "quiver": {"required": ("x", "y", "u", "v"), "optional": ("C",)},
    "imshow": {"required": ("z",), "optional": ()},
    "pcolormesh": {"required": ("x", "y", "z"), "optional": ()},
    "pcolor": {"required": ("x", "y", "z"), "optional": ()},
    "contour": {"required": ("x", "y", "z"), "optional": ()},
    "contourf": {"required": ("x", "y", "z"), "optional": ()},
    "jpcontour": {"required": ("x", "y", "z"), "optional": ()},
    "jpcontourf": {"required": ("x", "y", "z"), "optional": ()},
    "jpfield": {"required": ("x", "y", "z"), "optional": ()},
    "dynesty_runplot": {"required": (), "optional": ()},
    "tripcolor": {"required": ("x", "y", "z"), "optional": ()},
    "tripcolor_axes": {"required": ("x", "y", "z"), "optional": ()},
    "tricontour": {"required": ("x", "y", "z"), "optional": ()},
    "tricontourf": {"required": ("x", "y", "z"), "optional": ()},
    "triplot": {"required": ("x", "y"), "optional": ()},
    "voronoi": {"required": ("x", "y"), "optional": ("z",)},
    "voronoif": {"required": ("x", "y", "z"), "optional": ()},
}


def contract_for(method: str) -> dict[str, Any] | None:
    key = normalize_method_key(method)
    if key not in METHOD_DISPATCH:
        return None
    return METHOD_COORDINATES.get(key, {"required": (), "optional": ()})


def missing_coordinates(method: str, coordinates: Any) -> list[str]:
    """Return required axis names absent from the layer's coordinates block."""
    contract = contract_for(method)
    if contract is None:
        return []
    required = tuple(contract.get("required") or ())
    if not required:
        return []
    present = _present_axes(coordinates)
    if all(axis in present for axis in required):
        return []
    for pack in contract.get("alternatives") or ():
        if pack and all(axis in present for axis in pack):
            return []
    return [axis for axis in required if axis not in present]


def _present_axes(coordinates: Any) -> set[str]:
    if not isinstance(coordinates, Mapping):
        return set()
    out: set[str] = set()
    for key, value in coordinates.items():
        name = str(key).strip()
        if not name:
            continue
        # Empty mapping still counts as "axis declared but incomplete";
        # missing means the key is absent entirely.
        if value is None:
            continue
        out.add(name)
    return out


def schema_payload(method: str) -> dict[str, Any]:
    """JSON-serialisable body for ``schema/methods/<method>.json``."""
    key = normalize_method_key(method)
    mpl = METHOD_DISPATCH[key]
    contract = METHOD_COORDINATES.get(key, {"required": (), "optional": ()})
    try:
        spec, _ = REGISTRY.resolve(key, strict=False)
        axes_types = list(spec.axes_types)
    except Exception:
        axes_types = ["any"]
    required = list(contract.get("required") or ())
    optional = list(contract.get("optional") or ())
    alternatives = [list(pack) for pack in (contract.get("alternatives") or ())]
    example_axes = "\n".join(f"  {a}: {{expr: {a}}}" for a in (required or ["x", "y"]))
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://jarvis-plot.org/schema/v2/methods/{key}.json",
        "title": f"method: {key}",
        "description": (
            f"Coordinate contract for layers[].method={key!r} "
            f"(matplotlib: {mpl}). Required axes must appear under coordinates."
        ),
        "x-jarvis-zone": "closed",
        "x-jarvis-method": key,
        "x-jarvis-mpl": mpl,
        "x-jarvis-axes-types": axes_types,
        "x-jarvis-coordinates": {
            "required": required,
            "optional": optional,
            "alternatives": alternatives,
        },
        "x-jarvis-example": f"method: {key}\ncoordinates:\n{example_axes}",
        "type": "object",
        "properties": {
            "method": {"const": key},
            "coordinates": {
                "type": "object",
                "x-jarvis-zone": "delegated",
                "additionalProperties": True,
                "description": (
                    f"Required axes for {key}: {', '.join(required) or '(none)'}. "
                    f"Optional: {', '.join(optional) or '(none)'}."
                ),
            },
        },
        "additionalProperties": True,
    }

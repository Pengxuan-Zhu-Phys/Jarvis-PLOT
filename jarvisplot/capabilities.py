#!/usr/bin/env python3

"""Everything Jarvis-PLOT will accept, as data.

This module exists to answer one question an agent otherwise has to guess at:
**what are the legal strings?** Column names already have an authoritative
source (``jplot data describe``); every *other* string in a config -- method,
figure type, style card, axes name, colormap, transform step, expression
function -- has had none, so agents fall back on memory and invent plausible
names like ``jarvis_rainbow2_r`` or ``a4paper_2x1``.

Two rules keep this honest:

- **Report what the code does, not what the docs claim.** Every list here is
  derived from the registry, card directory or schema that the runtime actually
  consults, and ``tests/test_capabilities.py`` fails if the two drift apart.
- **Stay cheap.** Agents call this constantly, so the collectors read JSON and
  module-level dicts; nothing here imports matplotlib.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "CAPABILITY_SECTIONS",
    "capabilities",
    "digest",
    "section",
]

_CARDS_DIR = Path(__file__).with_name("cards")
_STYLE_PREFERENCE = _CARDS_DIR / "style_preference.json"
_COLORMAPS = _CARDS_DIR / "colors" / "colormaps.json"
_ARGS = _CARDS_DIR / "args.json"

CAPABILITY_SECTIONS = (
    "methods",
    "transforms",
    "types",
    "styles",
    "cmaps",
    "funcs",
    "cli",
)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def _methods() -> list[dict[str, Any]]:
    """Rendering primitives, with the axes families each one works on."""
    from .Figure.method_registry import METHOD_DISPATCH, MethodRegistry

    registry = MethodRegistry()
    out: list[dict[str, Any]] = []
    for key in sorted(METHOD_DISPATCH):
        entry: dict[str, Any] = {"name": key, "mpl_method": METHOD_DISPATCH[key]}
        try:
            spec = registry.resolve(key)
        except Exception:
            spec = None
        axes_types = getattr(spec, "axes_types", None)
        if axes_types:
            entry["axes_types"] = sorted(str(a) for a in axes_types)
        out.append(entry)
    return out


def _transforms() -> list[dict[str, Any]]:
    """Pipeline steps, from the schema that pins their vocabulary."""
    from .schema_catalog import subschema

    schema = subschema("https://jarvis-plot.org/schema/v2/core/transform.json")
    discriminated = set(schema["$defs"]["discriminatedStepName"]["enum"])
    out: list[dict[str, Any]] = []
    for name, spec in schema.get("properties", {}).items():
        if name == "type":
            continue
        entry: dict[str, Any] = {
            "name": name,
            "form": "single-key mapping",
            "description": str(spec.get("description", "")),
        }
        if name in discriminated:
            entry["form"] = "single-key mapping, or {type: %s, ...}" % name
        if spec.get("x-jarvis-zone") == "delegated":
            entry["keys"] = "delegated -- see the runtime module named in the description"
        out.append(entry)
    return sorted(out, key=lambda e: e["name"])


def _types() -> list[dict[str, Any]]:
    """Figure-type shorthands that expand into a layer stack before rendering.

    Only what ``Figure/figure_types.py::expand_figure_type`` actually dispatches
    on. Listing an aspirational type here would be worse than listing none.
    """
    from .Figure import figure_types

    names = sorted(
        attr[len("expand_") :]
        for attr in dir(figure_types)
        if attr.startswith("expand_")
        and attr not in {"expand_figure_type", "expand_figure_types_in_config"}
    )
    return [
        {
            "name": name,
            "expands_to": "layers",
            "explain": f"jplot explain {name}",
        }
        for name in names
    ]


def _styles() -> list[dict[str, Any]]:
    """Style cards, and -- the part agents cannot otherwise discover -- their axes.

    A layer's ``axes:`` must name one of these. Nothing in the YAML says where a
    name like ``axc`` comes from; it comes from the chosen card's ``Frame.axes``.
    """
    preference = _load_json(_STYLE_PREFERENCE)
    out: list[dict[str, Any]] = []
    for bundle in sorted(preference):
        for token in sorted(preference[bundle]):
            entry: dict[str, Any] = {
                "bundle": bundle,
                "token": token,
                "style": [bundle, token],
            }
            card_path = _resolve_card(preference[bundle][token])
            if card_path is None or not card_path.exists():
                entry.update(axes=[], usable=False, error="card file not found")
                out.append(entry)
                continue

            card = _load_json(card_path)
            frame = card.get("Frame") if isinstance(card, dict) else None
            if not isinstance(frame, dict):
                # Figure/figure.py:279 dereferences bundle["Frame"] directly, so a
                # card without it raises rather than degrading. Advertising it as
                # available would send an agent straight into a KeyError.
                entry.update(
                    axes=[],
                    usable=False,
                    error=(
                        "card has no 'Frame' block "
                        f"(top-level keys: {sorted(card) if isinstance(card, dict) else type(card).__name__}); "
                        "the renderer requires it"
                    ),
                )
                out.append(entry)
                continue

            axes = frame.get("axes", {})
            entry["usable"] = True
            entry["axes"] = sorted(axes) if isinstance(axes, dict) else []
            figsize = (frame.get("figure") or {}).get("figsize")
            if figsize:
                entry["figsize"] = figsize
            entry["styled_methods"] = sorted(card.get("Style", {}))
            out.append(entry)
    return out


def _resolve_card(value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.startswith("&JP/"):
        return Path(__file__).resolve().parents[1] / text[4:]
    return Path(text)


def _cmaps() -> dict[str, Any]:
    """Colormaps Jarvis-PLOT registers on top of the matplotlib set."""
    spec = _load_json(_COLORMAPS)
    entries = spec.get("colormaps", []) if isinstance(spec, dict) else []
    names = sorted(
        str(entry["name"])
        for entry in entries
        if isinstance(entry, dict) and entry.get("name")
    )
    return {
        "jarvis": names,
        "jarvis_reversed": [f"{name}_r" for name in names],
        "note": (
            "Every matplotlib colormap is also available. Jarvis colormaps are "
            "registered from cards/colors/colormaps.json; each has an _r reverse."
        ),
    }


def _funcs() -> dict[str, Any]:
    """Names an expression may call, beyond plain column references.

    The namespace is assembled at call time by
    ``utils/expression.build_eval_globals()`` -- there is no static table to
    read, and the set is not fixed: ``inner_func.update_funcs`` folds in
    Jarvis-Operas registrations and any externally injected interpolators. So
    this reports what the namespace holds *right now*, and says so.

    Costs a numpy import. That is the one place ``cap`` is not free.
    """
    from .utils.expression import build_eval_globals

    namespace = build_eval_globals()
    names = sorted(
        name
        for name, value in namespace.items()
        if not name.startswith("__") and callable(value)
    )
    modules = sorted(
        name
        for name, value in namespace.items()
        if not name.startswith("__") and hasattr(value, "__name__") and not callable(value)
    )
    return {
        "names": names,
        "namespaces": modules,
        "note": (
            "Assembled at call time; Jarvis-Operas operators and externally "
            "registered interpolators join this set, so it is not a fixed list. "
            "Use `jplot data eval` to check one expression before writing it "
            "into YAML."
        ),
    }


def _cli() -> dict[str, Any]:
    """The CLI, straight from its own spec file -- so it cannot drift."""
    return _load_json(_ARGS)


_COLLECTORS = {
    "methods": _methods,
    "transforms": _transforms,
    "types": _types,
    "styles": _styles,
    "cmaps": _cmaps,
    "funcs": _funcs,
    "cli": _cli,
}


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def section(name: str) -> Any:
    if name not in _COLLECTORS:
        raise KeyError(name)
    return _COLLECTORS[name]()


@lru_cache(maxsize=1)
def capabilities() -> dict[str, Any]:
    """Every section at once, plus a digest an agent can cache against."""
    payload = {name: _COLLECTORS[name]() for name in CAPABILITY_SECTIONS}
    payload["digest"] = _digest_of(payload)
    return payload


def digest() -> str:
    return capabilities()["digest"]


def _digest_of(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

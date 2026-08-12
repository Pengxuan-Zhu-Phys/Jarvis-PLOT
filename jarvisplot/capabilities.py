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
  module-level dicts. The cmap collector also inspects Matplotlib's live
  registry because agents may need the complete built-in cmap vocabulary.
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
    """Rendering primitives, with axes families and coordinate contracts."""
    from .Figure.method_registry import METHOD_DISPATCH, REGISTRY
    from .method_contracts import contract_for

    out: list[dict[str, Any]] = []
    for key in sorted(METHOD_DISPATCH):
        entry: dict[str, Any] = {"name": key, "mpl_method": METHOD_DISPATCH[key]}
        try:
            spec, _ = REGISTRY.resolve(key, strict=False)
            entry["axes_types"] = sorted(str(a) for a in spec.axes_types)
        except Exception:
            pass
        contract = contract_for(key)
        if contract is not None:
            entry["coordinates"] = {
                "required": list(contract.get("required") or ()),
                "optional": list(contract.get("optional") or ()),
            }
        out.append(entry)
    return out


def _transforms() -> list[dict[str, Any]]:
    """Pipeline steps with full contracts (not delegated stubs).

    Vocabulary still must match ``schema/core/transform.json``; field contracts
    come from :mod:`jarvisplot.transform_contracts` (runtime-aligned).
    """
    from .schema_catalog import subschema
    from .transform_contracts import contract_for, list_contracts

    schema = subschema("https://jarvis-plot.org/schema/v2/core/transform.json")
    schema_names = {
        name for name in schema.get("properties", {}) if name != "type"
    }
    discriminated = set(schema["$defs"]["discriminatedStepName"]["enum"])
    out: list[dict[str, Any]] = []
    for contract in list_contracts():
        name = contract["name"]
        if name not in schema_names:
            # Contract ahead of schema is a packaging bug — skip quietly.
            continue
        entry: dict[str, Any] = {
            "name": name,
            "form": contract.get("form") or "object",
            "description": contract.get("description") or "",
            "required": contract.get("required") or {},
            "optional": contract.get("optional") or {},
            "defaults": contract.get("defaults") or {},
            "enums": contract.get("enums") or {},
            "value": contract.get("value") or {},
            "input": contract.get("input") or "table",
            "output": contract.get("output") or "table",
            "owner": contract.get("owner") or "",
            "examples": contract.get("examples") or [],
            "man": f"jplot man transform.{name}",
        }
        if name in discriminated:
            entry["form_note"] = f"single-key mapping, or {{type: {name}, ...}}"
        out.append(entry)
    # Any schema-only names without a contract still appear (degraded).
    known = {e["name"] for e in out}
    for name in sorted(schema_names - known):
        spec = schema["properties"][name]
        entry = {
            "name": name,
            "form": "object",
            "description": str(spec.get("description", "")),
            "required": {},
            "optional": {},
            "defaults": {},
            "enums": {},
            "man": f"jplot man transform.{name}",
            "note": "schema vocabulary only — contract not yet filled in transform_contracts",
        }
        out.append(entry)
    return sorted(out, key=lambda e: e["name"])


def _types() -> list[dict[str, Any]]:
    """Figure-type shorthands that expand into a layer stack before rendering.

    Only names in :data:`KNOWN_FIGURE_TYPES` (actual ``expand_*`` dispatch),
    not helper names like ``typed_figures``.
    """
    from .Figure.figure_types import KNOWN_FIGURE_TYPES

    return [
        {
            "name": name,
            "expands_to": "layers",
            "explain": f"jplot explain {name}",
            "man": f"jplot man type-{name.replace('_', '-')}" if name.endswith("_2d") else f"jplot explain {name}",
        }
        for name in sorted(KNOWN_FIGURE_TYPES)
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
    jarvis_and_reversed = set(names) | {f"{name}_r" for name in names}
    matplotlib_entries: list[dict[str, Any]] = []
    try:
        from matplotlib import colormaps

        matplotlib_names = sorted(
            str(name) for name in colormaps if str(name) not in jarvis_and_reversed
        )
        matplotlib_registry = set(str(name) for name in colormaps)
        for name in matplotlib_names:
            cmap = colormaps[name]
            if name.endswith("_r") and name[:-2] in matplotlib_registry:
                reverse = name[:-2]
            elif f"{name}_r" in matplotlib_registry:
                reverse = f"{name}_r"
            else:
                reverse = None
            samples = getattr(cmap, "N", None)
            matplotlib_entries.append(
                {
                    "name": name,
                    "type": type(cmap).__name__,
                    "N": int(samples) if isinstance(samples, int) else samples,
                    "reverse": reverse,
                }
            )
        matplotlib_note = (
            "Matplotlib registry entries include the normal and _r reverse names."
        )
    except Exception as exc:
        matplotlib_note = f"Matplotlib registry unavailable: {exc}"
    return {
        "jarvis": names,
        "jarvis_reversed": [f"{name}_r" for name in names],
        "matplotlib": matplotlib_entries,
        "matplotlib_note": matplotlib_note,
        "note": (
            "Every matplotlib colormap is also available. Jarvis colormaps are "
            "registered from cards/colors/colormaps.json; each has an _r reverse."
        ),
    }


def _funcs() -> dict[str, Any]:
    """Public expression callables for agents (filtered).

    Full eval globals still include Operas registrations and ephemeral hashed
    helpers; those are noise for agents. We publish:

    1. Stable tokens from :data:`EXPR_IDENTIFIER_IGNORE` that are callable-like.
    2. A short public sample of remaining callables without ``_``/hash junk.

    Full dump remains available under ``names_full`` for debugging only.
    """
    from .expr_names import EXPR_IDENTIFIER_IGNORE
    from .utils.expression import build_eval_globals

    namespace = build_eval_globals()
    all_callable = sorted(
        name
        for name, value in namespace.items()
        if not name.startswith("__") and callable(value)
    )
    modules = sorted(
        name
        for name, value in namespace.items()
        if not name.startswith("__") and hasattr(value, "__name__") and not callable(value)
    )

    def _is_public(name: str) -> bool:
        if name.startswith("_"):
            return False
        # hashed / gensym style: contains long digit runs or looks like uuid crumbs
        if any(ch.isdigit() for ch in name) and sum(ch.isdigit() for ch in name) >= 4:
            return False
        if name.count("_") >= 3 and any(ch.isdigit() for ch in name):
            return False
        return True

    preferred = [
        n
        for n in (
            "exp",
            "log",
            "ln",
            "log10",
            "sqrt",
            "abs",
            "min",
            "max",
            "sin",
            "cos",
            "tan",
            "Heaviside",
            "Gauss",
            "LogGauss",
            "Normal",
        )
        if n in all_callable or n in EXPR_IDENTIFIER_IGNORE
    ]
    public = [n for n in all_callable if _is_public(n)]
    # Prefer short stable names first
    public.sort(key=lambda n: (0 if n in preferred else 1, len(n), n))
    names = []
    for n in preferred + public:
        if n not in names:
            names.append(n)
        if len(names) >= 80:
            break

    return {
        "names": names,
        "namespaces": modules,
        "names_full_count": len(all_callable),
        "note": (
            "Public sample for agents (hashed/internal Operas helpers omitted). "
            "`names_full_count` is the raw callable count in the eval namespace. "
            "Use `jplot data eval` to check one expression; signatures are not "
            "yet published (most are numpy/math-like unary/binary)."
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

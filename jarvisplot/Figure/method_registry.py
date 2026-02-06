

"""JarvisPLOT method registry

This module centralizes the mapping between YAML `method` keys and the
corresponding Matplotlib Axes methods.

Why this exists:
- Keep Figure/renderer code slim.
- Provide a single place to add aliases / deprecations.
- Allow light validation by axes type (rect/tri).

Design notes:
- For backward compatibility, `METHOD_DISPATCH` remains a plain dict mapping
  YAML keys to Matplotlib method names.
- Prefer using `resolve_method(...)` for new code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Set, Tuple


# -----------------------------
# Backward-compatible mapping
# -----------------------------
#
# Keys here should be stable YAML-facing names.
# Values are Matplotlib Axes method names.
#
# NOTE:
# - "tripcolor" / "tricontourf" / "triplot" are tri-axes friendly.
# - Most others are standard rect axes methods.
#
METHOD_DISPATCH: Dict[str, str] = {
    # common 2D primitives
    "plot":         "plot",
    "scatter":      "scatter",
    "hist":         "hist",
    "errorbar":     "errorbar",
    "fill_between": "fill_between",
    "fill_betweenx": "fill_betweenx",
    "bar":          "bar",
    "barh":         "barh",
    "step":         "step",

    # images / grids
    "imshow":       "imshow",
    "pcolormesh":   "pcolormesh",
    "pcolor":       "pcolor",
    "contour":      "contour",
    "contourf":     "contourf",

    # tri related
    "tripcolor":    "tripcolor",
    "tricontour":   "tricontour",
    "tricontourf":  "tricontourf",
    "triplot":      "triplot",
    
    # jarvisplot inplemented Voronoi method 
    "voronoi":      "voronoi", 
    "voronoif":     "voronoif"
}


# -----------------------------
# Registry (new preferred API)
# -----------------------------

_ALLOWED_AX_TYPES = {"rect", "tri", "any"}


@dataclass(frozen=True)
class MethodSpec:
    key: str
    mpl_method: str
    axes_types: Tuple[str, ...] = ("any",)
    aliases: Tuple[str, ...] = ()
    deprecated_aliases: Tuple[str, ...] = ()


class MethodRegistry:
    """A small, explicit registry for YAML method keys."""

    def __init__(self) -> None:
        self._by_key: Dict[str, MethodSpec] = {}
        self._alias_to_key: Dict[str, str] = {}
        self._deprecated_alias_to_key: Dict[str, str] = {}

    def register(
        self,
        key: str,
        mpl_method: str,
        *,
        axes_types: Iterable[str] = ("any",),
        aliases: Iterable[str] = (),
        deprecated_aliases: Iterable[str] = (),
        overwrite: bool = False,
    ) -> None:
        k = normalize_method_key(key)
        ax_types = tuple(normalize_axes_type(t) for t in axes_types)
        if not overwrite and k in self._by_key:
            raise ValueError(f"method key already registered: {k}")

        spec = MethodSpec(
            key=k,
            mpl_method=mpl_method,
            axes_types=ax_types,
            aliases=tuple(normalize_method_key(a) for a in aliases),
            deprecated_aliases=tuple(normalize_method_key(a) for a in deprecated_aliases),
        )
        self._by_key[k] = spec

        for a in spec.aliases:
            if not overwrite and a in self._alias_to_key:
                raise ValueError(f"alias already registered: {a}")
            self._alias_to_key[a] = k

        for a in spec.deprecated_aliases:
            if not overwrite and a in self._deprecated_alias_to_key:
                raise ValueError(f"deprecated alias already registered: {a}")
            self._deprecated_alias_to_key[a] = k

    def resolve(
        self,
        key: str,
        *,
        axes_type: str = "any",
        strict: bool = True,
    ) -> Tuple[MethodSpec, Optional[str]]:
        """Resolve a YAML `method` key.

        Returns:
            (spec, warning)

        warning is a string when a deprecated alias is used; otherwise None.
        """
        k = normalize_method_key(key)
        ax_t = normalize_axes_type(axes_type)

        warn: Optional[str] = None

        if k in self._by_key:
            spec = self._by_key[k]
        elif k in self._alias_to_key:
            spec = self._by_key[self._alias_to_key[k]]
        elif k in self._deprecated_alias_to_key:
            resolved = self._deprecated_alias_to_key[k]
            spec = self._by_key[resolved]
            warn = f"Method key '{k}' is deprecated; use '{spec.key}' instead."
        else:
            if strict:
                raise KeyError(f"Unknown method key: '{key}'")
            # best-effort: treat it as a raw matplotlib method name
            spec = MethodSpec(key=k, mpl_method=k, axes_types=("any",))

        if strict and ("any" not in spec.axes_types) and (ax_t not in spec.axes_types):
            raise ValueError(
                f"Method '{spec.key}' is not supported for axes_type='{ax_t}'. "
                f"Allowed: {spec.axes_types}"
            )

        return spec, warn


def normalize_method_key(key: str) -> str:
    return str(key).strip().lower()


def normalize_axes_type(axes_type: str) -> str:
    t = str(axes_type).strip().lower()
    if t not in _ALLOWED_AX_TYPES:
        # be permissive: unknown types collapse to 'any'
        return "any"
    return t


# default global registry
REGISTRY = MethodRegistry()


def _bootstrap_default_registry() -> None:
    """Populate REGISTRY from METHOD_DISPATCH and known aliases."""

    # --- core mapping from METHOD_DISPATCH
    for k, mpl in METHOD_DISPATCH.items():
        # tri-specific methods
        if k.startswith("tri") or k in {"triplot"}:
            REGISTRY.register(k, mpl, axes_types=("tri", "any"), overwrite=True)
        else:
            REGISTRY.register(k, mpl, axes_types=("rect", "any"), overwrite=True)

    # --- aliases / more user-friendly names
    REGISTRY.register(
        "line",
        "plot",
        axes_types=("rect", "any"),
        aliases=("lines",),
        deprecated_aliases=(),
        overwrite=True,
    )
    REGISTRY.register(
        "points",
        "scatter",
        axes_types=("rect", "any"),
        aliases=("point", "scatterplot"),
        overwrite=True,
    )

    # tri field convenience: allow explicit gouraud naming on same mpl method
    REGISTRY.register(
        "tripcolor_gouraud",
        "tripcolor",
        axes_types=("tri", "any"),
        aliases=("tri_field", "tri_color"),
        overwrite=True,
    )


_bootstrap_default_registry()


def resolve_method(
    method_key: str,
    *,
    axes_type: str = "any",
    strict: bool = True,
) -> Tuple[str, Optional[str]]:
    """Resolve a YAML method key to a Matplotlib method name.

    Returns:
        (mpl_method_name, warning)
    """
    spec, warn = REGISTRY.resolve(method_key, axes_type=axes_type, strict=strict)
    return spec.mpl_method, warn


def resolve_callable(
    ax,
    method_key: str,
    *,
    axes_type: str = "any",
    strict: bool = True,
) -> Tuple[Callable, Optional[str]]:
    """Resolve a YAML method key to a bound Axes callable."""
    mpl_name, warn = resolve_method(method_key, axes_type=axes_type, strict=strict)
    fn = getattr(ax, mpl_name, None)
    if fn is None:
        raise AttributeError(f"Axes has no method '{mpl_name}' (from key '{method_key}')")
    return fn, warn
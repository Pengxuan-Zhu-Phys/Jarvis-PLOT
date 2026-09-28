"""Stdlib-only coordinate contract and column demand for PDF1D / CDF1D."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Any, Mapping

from .expr_names import expr_identifiers

KINDS = ("PDF1D", "CDF1D")
DEFAULT_GRID = 600


@dataclass(frozen=True)
class AnchorSettings:
    method: str = "adaptive"
    tolerance: float = 0.005
    min_mass: float = 0.01
    max_points: int = 256


def anchor_settings(spec: Any = None) -> AnchorSettings:
    if spec is None:
        return AnchorSettings()
    if isinstance(spec, str):
        spec = {"method": spec}
    if not isinstance(spec, Mapping):
        raise ValueError("anchors needs adaptive/all or a configuration mapping")
    extra = set(spec) - {"method", "tolerance", "min_mass", "max_points"}
    if extra:
        raise ValueError(f"anchors has unknown keys: {sorted(extra)}")
    method = spec.get("method", "adaptive")
    if method not in {"adaptive", "all"}:
        raise ValueError("anchors.method must be adaptive or all")
    values = {}
    for key, default in (("tolerance", 0.005), ("min_mass", 0.01)):
        value = spec.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not 0 < value < 1:
            raise ValueError(f"anchors.{key} must be finite and strictly between 0 and 1")
        values[key] = float(value)
    maximum = spec.get("max_points", 256)
    if isinstance(maximum, bool) or not isinstance(maximum, Integral) or maximum < 4:
        raise ValueError("anchors.max_points must be an integer >= 4 (including support endpoints)")
    return AnchorSettings(method, values["tolerance"], values["min_mass"], int(maximum))


def distribution_kind(step: Any) -> str | None:
    if not isinstance(step, Mapping):
        return None
    return next((kind for kind in KINDS if kind in step), None)


def distribution_config(step: Mapping[str, Any]) -> dict:
    kind = distribution_kind(step)
    cfg = step.get(kind) if kind else None
    if not isinstance(cfg, Mapping):
        raise ValueError(f"{kind or 'PDF1D/CDF1D'} needs a configuration mapping")
    return dict(cfg)


def _coordinate(spec: Any, label: str, *, weight: bool = False) -> dict:
    if isinstance(spec, str):
        spec = {"expr": spec}
    if not isinstance(spec, Mapping):
        raise ValueError(f"{label} needs a column string or an expr/name mapping")
    allowed = {"expr", "name"} if weight else {"expr", "name", "lim", "grid", "scale"}
    extra = set(spec) - allowed
    if extra:
        raise ValueError(f"{label} has unknown keys: {sorted(extra)}")
    expr = spec.get("expr", spec.get("name"))
    if isinstance(expr, bool) or not isinstance(expr, (str, int, float)):
        raise ValueError(f"{label} needs expr (or name of an input column)")
    if isinstance(expr, str) and not expr.strip():
        raise ValueError(f"{label}.expr must not be empty")
    return dict(spec, expr=expr)


@dataclass(frozen=True)
class Settings:
    x: dict
    weight: dict | None
    repeat: str | None
    name: str
    grid: int
    scale: str
    limits: tuple[float, float] | None
    anchors: AnchorSettings
    interpolation: str


def settings(cfg: Mapping[str, Any]) -> Settings:
    extra = set(cfg) - {"coordinates", "repeat", "anchors", "interpolation"}
    if extra:
        raise ValueError(f"PDF1D/CDF1D has unknown keys: {sorted(extra)}; grid belongs in coordinates.x")
    coords = cfg.get("coordinates")
    if not isinstance(coords, Mapping) or "x" not in coords:
        raise ValueError("PDF1D/CDF1D requires coordinates.x")
    extra = set(coords) - {"x", "weight"}
    if extra:
        raise ValueError(f"coordinates has unknown keys: {sorted(extra)}")
    x = _coordinate(coords["x"], "coordinates.x")
    weight = _coordinate(coords["weight"], "coordinates.weight", weight=True) if "weight" in coords else None
    name = x.get("name", "x")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("coordinates.x.name must be a nonempty string")
    name = name.strip()
    if name in {"cdf", "pdf", "cdf_std", "pdf_std", "n_repeats", "__jp_row_idx__"}:
        raise ValueError(f"coordinates.x.name {name!r} collides with a reserved output column")
    grid = x.get("grid", DEFAULT_GRID)
    if isinstance(grid, bool) or not isinstance(grid, Integral) or grid < 2:
        raise ValueError("coordinates.x.grid must be an integer >= 2")
    scale = x.get("scale", "linear")
    if scale not in {"linear", "log"}:
        raise ValueError("coordinates.x.scale must be linear or log")
    lim = x.get("lim")
    limits = None
    if lim is not None:
        try:
            arr = tuple(float(v) for v in lim)
        except (TypeError, ValueError) as exc:
            raise ValueError("coordinates.x.lim needs finite [lo, hi] with lo < hi") from exc
        if len(arr) != 2 or not all(isfinite(v) for v in arr) or arr[0] >= arr[1]:
            raise ValueError("coordinates.x.lim needs finite [lo, hi] with lo < hi")
        limits = (float(arr[0]), float(arr[1]))
        if scale == "log" and limits[0] <= 0:
            raise ValueError("coordinates.x.scale: log requires positive limits")
    repeat = cfg.get("repeat")
    if repeat is not None and (not isinstance(repeat, str) or not repeat.strip()):
        raise ValueError("repeat must name a column of batch identifiers")
    anchors = anchor_settings(cfg.get("anchors"))
    interpolation = cfg.get("interpolation", "monotone_c2")
    if interpolation not in {"monotone_c2", "pchip"}:
        raise ValueError("interpolation must be monotone_c2 or pchip")
    return Settings(x, weight, repeat.strip() if repeat is not None else None, name, int(grid), scale, limits,
                    anchors, interpolation)


def distribution_input_columns(cfg: Any) -> set[str]:
    """Analyze expressions without validating; invalid YAML belongs to diagnostics."""
    if not isinstance(cfg, Mapping):
        return set()
    columns = set()
    coords = cfg.get("coordinates")
    if isinstance(coords, Mapping):
        for key in ("x", "weight"):
            spec = coords.get(key)
            expr = spec.get("expr", spec.get("name")) if isinstance(spec, Mapping) else spec
            if isinstance(expr, str):
                columns.update(expr_identifiers(expr))
    repeat = cfg.get("repeat")
    if isinstance(repeat, str) and repeat.strip():
        columns.add(repeat.strip())
    return columns


def distribution_output_columns(kind: str, cfg: Any) -> set[str]:
    if not isinstance(cfg, Mapping):
        return set()
    coords = cfg.get("coordinates")
    x = coords.get("x") if isinstance(coords, Mapping) else None
    name = x.get("name", "x") if isinstance(x, Mapping) else "x"
    name = name.strip() if isinstance(name, str) and name.strip() else "x"
    columns = {name, "cdf", "cdf_std", "n_repeats"}
    if kind == "PDF1D":
        columns.update({"pdf", "pdf_std"})
    return columns

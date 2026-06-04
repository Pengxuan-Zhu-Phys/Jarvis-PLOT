from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np
import pandas as pd

from .density_cell_runtime import (
    _axis_lim as _density_axis_lim,
    _axis_scale as _density_axis_scale,
    _resolve_array as _density_resolve_array,
    density_cell,
)
from .interp_2d_runtime import make_interp_2d
from .posterior_mesh import conservative_density, support_areas


_POSTERIOR_DENSITY_TYPES = {"posterior_density"}


def is_posterior_density_transform(step: Any) -> bool:
    if not isinstance(step, Mapping):
        return False
    if any(name in step for name in _POSTERIOR_DENSITY_TYPES):
        return True
    return str(step.get("type", "")).strip().lower() in _POSTERIOR_DENSITY_TYPES


def posterior_density_config(step: Mapping[str, Any]) -> Mapping[str, Any]:
    for name in _POSTERIOR_DENSITY_TYPES:
        if name in step:
            cfg = step.get(name)
            return cfg if isinstance(cfg, Mapping) else {}
    if str(step.get("type", "")).strip().lower() in _POSTERIOR_DENSITY_TYPES:
        return step
    return {}


def _logger_emit(logger, level: str, message: str) -> None:
    if logger is None:
        return
    try:
        getattr(logger, level)(message)
    except Exception:
        pass


def _coord_with_output_name(cfg: Mapping[str, Any], key: str, output_name: str) -> Mapping[str, Any]:
    spec = cfg.get(key, None)
    if spec is None:
        coors = cfg.get("coordinates", {})
        if isinstance(coors, Mapping):
            spec = coors.get(key, None)
    if isinstance(spec, Mapping):
        out = dict(spec)
    elif spec is None:
        out = {"name": output_name}
    else:
        out = {"expr": spec}
    out["name"] = output_name
    return out


def _output_names(cfg: Mapping[str, Any]) -> Tuple[str, str, str]:
    output = cfg.get("output", "density")
    if isinstance(output, Mapping):
        x_name = str(output.get("x", "x")).strip() or "x"
        y_name = str(output.get("y", "y")).strip() or "y"
        z_name = str(output.get("z", output.get("density", "density"))).strip() or "density"
    else:
        x_name = "x"
        y_name = "y"
        z_name = str(output).strip() or "density"
    return x_name, y_name, z_name


def _axis_domain(df: pd.DataFrame, cfg: Mapping[str, Any], logger) -> Tuple[Tuple[float, float], Tuple[float, float], str, str]:
    x = _density_resolve_array(df, cfg, "x", "x", logger, required=True)
    y = _density_resolve_array(df, cfg, "y", "y", logger, required=True)
    return (
        _density_axis_lim(np.asarray(x, dtype=float), cfg, "x"),
        _density_axis_lim(np.asarray(y, dtype=float), cfg, "y"),
        _density_axis_scale(cfg, "x"),
        _density_axis_scale(cfg, "y"),
    )


def _common_core_cfg(cfg: Mapping[str, Any], *, method: str, x_out: str, y_out: str) -> dict[str, Any]:
    core: dict[str, Any] = {
        "method": method,
        "x": _coord_with_output_name(cfg, "x", x_out),
        "y": _coord_with_output_name(cfg, "y", y_out),
        "weight": _coord_with_output_name(cfg, "weight", "weight"),
        "bins": int(cfg.get("bins", cfg.get("bin", 64))),
        "normalize": bool(cfg.get("normalize", True)),
        "diagnostics": bool(cfg.get("diagnostics", True)),
    }
    for key in ("seed", "_base_dir", "_mesh_debug"):
        if key in cfg:
            core[key] = cfg[key]
    return core


def _bridson_core_cfg(cfg: Mapping[str, Any], *, adaptive: bool, x_out: str, y_out: str) -> dict[str, Any]:
    core = _common_core_cfg(cfg, method="bridson", x_out=x_out, y_out=y_out)
    voronoi = cfg.get("voronoi", {})
    if not isinstance(voronoi, Mapping):
        voronoi = {}
    bins = int(cfg.get("bins", cfg.get("bin", 64)))
    bridson = {
        "bin": bins,
        "seed": cfg.get("seed", None),
        "k": int(voronoi.get("k", voronoi.get("candidates", 30))),
    }
    core["bridson"] = {k: v for k, v in bridson.items() if v is not None}
    if adaptive:
        adaptive_cfg = cfg.get("adaptive", {})
        if not isinstance(adaptive_cfg, Mapping):
            adaptive_cfg = {}
        refinement = dict(adaptive_cfg)
        refinement["enabled"] = bool(refinement.get("enabled", True))
        if "split" in refinement and "split_enabled" not in refinement:
            refinement["split_enabled"] = bool(refinement.pop("split"))
        if "merge" in refinement and "merge_enabled" not in refinement:
            refinement["merge_enabled"] = bool(refinement.pop("merge"))
        if "seed" not in refinement and cfg.get("seed", None) is not None:
            refinement["seed"] = cfg.get("seed")
        core["refinement"] = refinement
    return core


def _kde_core_cfg(cfg: Mapping[str, Any], *, x_out: str, y_out: str) -> dict[str, Any]:
    core = _common_core_cfg(cfg, method="kde", x_out=x_out, y_out=y_out)
    kde = cfg.get("kde", {})
    if isinstance(kde, Mapping) and kde.get("bw_method", None) is not None:
        core["bw_method"] = kde.get("bw_method")
    elif cfg.get("bw_method", None) is not None:
        core["bw_method"] = cfg.get("bw_method")
    return core


def _grid_or_kde_density(
    df: pd.DataFrame,
    cfg: Mapping[str, Any],
    core_cfg: Mapping[str, Any],
    *,
    x_out: str,
    y_out: str,
    z_out: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    logger,
) -> pd.DataFrame:
    core = density_cell(df, core_cfg, logger)
    xs = core[x_out].to_numpy(dtype=float)
    ys = core[y_out].to_numpy(dtype=float)
    mass = core["weight"].to_numpy(dtype=float)
    areas = support_areas(xs, ys, xlim, ylim).areas
    density = conservative_density(mass, areas)
    out = pd.DataFrame({x_out: xs, y_out: ys, z_out: density})
    if bool(cfg.get("diagnostics", True)):
        integral = float(np.nansum(density * areas))
        _logger_emit(
            logger,
            "warning",
            "posterior_density diagnostics:\n"
            f"\t method \t-> {core_cfg.get('method')}\n"
            f"\t output_grid \t-> {len(out)}\n"
            f"\t output_columns -> [{x_out}, {y_out}, {z_out}]\n"
            f"\t integral \t-> {integral:.17g}",
        )
    return out


def posterior_density(df: pd.DataFrame, cfg: Mapping[str, Any], logger=None) -> pd.DataFrame:
    """Build a regular posterior density grid from raw weighted samples."""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if not isinstance(cfg, Mapping):
        cfg = {}

    method = str(cfg.get("method", "voronoi")).strip().lower()
    method = {"bridson": "voronoi", "natural_neighbor": "voronoi"}.get(method, method)
    if method not in {"voronoi", "adaptive", "kde", "grid"}:
        raise ValueError("posterior_density method must be one of: voronoi, adaptive, kde, grid.")

    x_out, y_out, z_out = _output_names(cfg)
    domain_cfg = _common_core_cfg(cfg, method="grid", x_out=x_out, y_out=y_out)
    xlim, ylim, xscale, yscale = _axis_domain(df, domain_cfg, logger)

    if method in {"voronoi", "adaptive"}:
        core_cfg = _bridson_core_cfg(cfg, adaptive=(method == "adaptive"), x_out=x_out, y_out=y_out)
        core = density_cell(df, core_cfg, logger)
        interp_cfg = {
            "method": "natural_neighbor",
            "as_density": True,
            "normalize": bool(cfg.get("normalize", True)),
            "diagnostics": bool(cfg.get("diagnostics", True)),
            "coordinates": {
                "x": {"expr": x_out, "name": x_out, "lim": list(xlim), "scale": xscale},
                "y": {"expr": y_out, "name": y_out, "lim": list(ylim), "scale": yscale},
                "z": {"expr": "weight", "name": z_out},
            },
            "grid": cfg.get("grid", 256),
            "nan_policy": cfg.get("nan_policy", "strict"),
        }
        for key in ("triangulation", "griddata"):
            if key in cfg:
                interp_cfg[key] = cfg[key]
        return make_interp_2d(core, interp_cfg, logger)

    if method == "kde":
        core_cfg = _kde_core_cfg(cfg, x_out=x_out, y_out=y_out)
    else:
        core_cfg = _common_core_cfg(cfg, method="grid", x_out=x_out, y_out=y_out)
    return _grid_or_kde_density(
        df,
        cfg,
        core_cfg,
        x_out=x_out,
        y_out=y_out,
        z_out=z_out,
        xlim=xlim,
        ylim=ylim,
        logger=logger,
    )

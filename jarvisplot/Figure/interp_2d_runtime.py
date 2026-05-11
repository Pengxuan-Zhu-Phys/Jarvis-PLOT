from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils.expression import eval_dataframe_expression
from .interp_natural_neighbor import resolve_backend


_INTERP_2D_TYPES = {"make_interp_2d"}


def is_interp_2d_transform(step: Any) -> bool:
    if not isinstance(step, Mapping):
        return False
    if any(name in step for name in _INTERP_2D_TYPES):
        return True
    return str(step.get("type", "")).strip().lower() in _INTERP_2D_TYPES


def interp_2d_config(step: Mapping[str, Any]) -> Mapping[str, Any]:
    for name in _INTERP_2D_TYPES:
        if name in step:
            cfg = step.get(name)
            return cfg if isinstance(cfg, Mapping) else {}
    if str(step.get("type", "")).strip().lower() in _INTERP_2D_TYPES:
        return step
    return {}


def _logger_emit(logger, level: str, message: str) -> None:
    if logger is None:
        return
    try:
        getattr(logger, level)(message)
    except Exception:
        pass


def _coord_cfg(cfg: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    coors = cfg.get("coordinates", {})
    if not isinstance(coors, Mapping):
        coors = {}
    spec = coors.get(key, cfg.get(key, None))
    if isinstance(spec, Mapping):
        return spec
    if spec is None:
        return {}
    return {"name": str(spec)}


def _output_name(cfg: Mapping[str, Any], key: str, default: str) -> str:
    name = _coord_cfg(cfg, key).get("name", default)
    text = str(name).strip()
    return text or default


def _resolve_array(df: pd.DataFrame, cfg: Mapping[str, Any], key: str, default: str, logger) -> np.ndarray:
    spec = _coord_cfg(cfg, key)
    expr = spec.get("expr")
    if expr is not None:
        return np.asarray(eval_dataframe_expression(df, expr, logger=logger, allow_column=True), dtype=float)
    name = str(spec.get("name", default)).strip()
    if not name:
        raise ValueError(f"make_interp_2d requires a non-empty coordinate for '{key}'.")
    if name not in df.columns:
        raise KeyError(f"make_interp_2d missing required column '{name}'.")
    return np.asarray(df[name], dtype=float)


def _axis_lim(values: np.ndarray, cfg: Mapping[str, Any], key: str) -> Tuple[float, float]:
    spec = _coord_cfg(cfg, key)
    lim = spec.get("lim", spec.get("limits", None))
    if lim is None:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError(f"make_interp_2d cannot infer {key} limits from non-finite data.")
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if lo == hi:
            hi = lo + 1.0
        return lo, hi
    if not isinstance(lim, Sequence) or len(lim) != 2:
        raise ValueError(f"make_interp_2d coordinates.{key}.lim must have two values.")
    lo, hi = float(lim[0]), float(lim[1])
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise ValueError(f"make_interp_2d coordinates.{key}.lim requires finite lo < hi.")
    return lo, hi


def _axis_scale(cfg: Mapping[str, Any], key: str) -> str:
    scale = str(_coord_cfg(cfg, key).get("scale", "linear")).strip().lower()
    if scale in {"", "lin"}:
        return "linear"
    if scale not in {"linear", "log"}:
        raise ValueError(f"make_interp_2d coordinates.{key}.scale only supports linear or log.")
    return scale


def _project_axis(values: np.ndarray, lim: Tuple[float, float], scale: str, key: str) -> np.ndarray:
    lo, hi = lim
    values = np.asarray(values, dtype=float)
    if scale == "log":
        if lo <= 0 or hi <= 0:
            raise ValueError(f"make_interp_2d {key} log scale requires positive limits.")
        out = np.full(values.shape, np.nan, dtype=float)
        mask = values > 0
        out[mask] = (np.log(values[mask]) - np.log(lo)) / (np.log(hi) - np.log(lo))
        return out
    return (values - lo) / (hi - lo)


def _unproject_axis(values: np.ndarray, lim: Tuple[float, float], scale: str) -> np.ndarray:
    lo, hi = lim
    values = np.asarray(values, dtype=float)
    if scale == "log":
        return np.exp(np.log(lo) + values * (np.log(hi) - np.log(lo)))
    return lo + values * (hi - lo)


def _grid_shape(cfg: Mapping[str, Any], default: int = 256) -> Tuple[int, int]:
    grid = cfg.get("grid", None)
    if isinstance(grid, (int, float, str)):
        nx = ny = int(grid)
    elif isinstance(grid, Sequence) and not isinstance(grid, (str, bytes, Mapping)):
        if len(grid) != 2:
            raise ValueError("make_interp_2d grid sequence must be [nx, ny].")
        nx, ny = int(grid[0]), int(grid[1])
    else:
        if not isinstance(grid, Mapping):
            grid = {}
        bins = grid.get("bins", grid.get("bin", cfg.get("bins", cfg.get("bin", None))))
        if bins is not None:
            nx = ny = int(bins)
        else:
            nx = int(grid.get("nx", cfg.get("nx", default)))
            ny = int(grid.get("ny", cfg.get("ny", default)))
    if nx < 2 or ny < 2:
        raise ValueError("make_interp_2d grid requires nx >= 2 and ny >= 2.")
    return nx, ny


def _dedupe_points(x: np.ndarray, y: np.ndarray, z: np.ndarray, *, reducer: str = "mean"):
    points = np.column_stack([x, y])
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    if len(unique) == len(points):
        return x, y, z
    values = np.asarray(z, dtype=float)
    if reducer == "sum":
        out = np.zeros(len(unique), dtype=float)
        np.add.at(out, inverse, values)
    else:
        sums = np.zeros(len(unique), dtype=float)
        counts = np.zeros(len(unique), dtype=float)
        np.add.at(sums, inverse, values)
        np.add.at(counts, inverse, 1.0)
        out = sums / np.maximum(counts, 1.0)
    return unique[:, 0], unique[:, 1], out


def _regular_support_areas(x: np.ndarray, y: np.ndarray, xlim: Tuple[float, float], ylim: Tuple[float, float]):
    ux, ix = np.unique(np.asarray(x, dtype=float), return_inverse=True)
    uy, iy = np.unique(np.asarray(y, dtype=float), return_inverse=True)
    nx = int(ux.size)
    ny = int(uy.size)
    if nx <= 1 or ny <= 1 or nx * ny != len(x):
        return None
    slots = iy * nx + ix
    if np.unique(slots).size != len(x):
        return None

    def _edges(vals: np.ndarray, lim: Tuple[float, float]) -> np.ndarray:
        mids = 0.5 * (vals[:-1] + vals[1:])
        return np.concatenate([[float(lim[0])], mids, [float(lim[1])]])

    x_edges = _edges(ux, xlim)
    y_edges = _edges(uy, ylim)
    dx = np.diff(x_edges)
    dy = np.diff(y_edges)
    if not (np.all(np.isfinite(dx)) and np.all(np.isfinite(dy)) and np.all(dx > 0) and np.all(dy > 0)):
        return None
    area_grid = dy[:, None] * dx[None, :]
    return area_grid[iy, ix]


def _voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if -1 not in vertices:
            new_regions.append(vertices)
            continue
        ridges = all_ridges.get(p1, [])
        new_region = [v for v in vertices if v != -1]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0 and v2 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            norm = np.linalg.norm(t)
            if norm == 0:
                continue
            t /= norm
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_regions.append(np.asarray(new_region)[np.argsort(angles)].tolist())
    return new_regions, np.asarray(new_vertices)


def _clipped_voronoi_areas(x: np.ndarray, y: np.ndarray, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> np.ndarray:
    try:
        from scipy.spatial import QhullError, Voronoi
    except Exception as e:
        raise ImportError("make_interp_2d as_density requires scipy.spatial.Voronoi for irregular support.") from e
    try:
        from shapely.geometry import Polygon, box
    except Exception as e:
        raise ImportError("make_interp_2d as_density requires shapely for clipped irregular Voronoi areas.") from e

    pts = np.column_stack([x, y])
    if pts.shape[0] < 4:
        raise ValueError("make_interp_2d as_density requires at least four unique 2D support points for Voronoi areas.")
    if np.linalg.matrix_rank(pts - pts.mean(axis=0)) < 2:
        raise ValueError("make_interp_2d as_density requires non-collinear 2D support points.")
    try:
        vor = Voronoi(pts)
    except QhullError as e:
        raise ValueError(f"make_interp_2d failed to construct Voronoi areas: {e}") from e
    regions, vertices = _voronoi_finite_polygons_2d(vor, radius=max(xlim[1] - xlim[0], ylim[1] - ylim[0]) * 4)
    bbox = box(float(xlim[0]), float(ylim[0]), float(xlim[1]), float(ylim[1]))
    areas = np.zeros(len(regions), dtype=float)
    for idx, region in enumerate(regions):
        if len(region) < 3:
            continue
        poly = Polygon(vertices[region])
        if not poly.is_valid:
            poly = poly.buffer(0)
        clipped = poly.intersection(bbox)
        areas[idx] = float(clipped.area) if not clipped.is_empty else 0.0
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0):
        raise ValueError("make_interp_2d as_density produced invalid support-cell areas.")
    return areas


def _support_areas(x_phys: np.ndarray, y_phys: np.ndarray, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> Tuple[np.ndarray, str]:
    areas = _regular_support_areas(x_phys, y_phys, xlim, ylim)
    if areas is not None:
        return areas, "regular_grid"
    return _clipped_voronoi_areas(x_phys, y_phys, xlim, ylim), "clipped_voronoi"


def _grid_integral(Z: np.ndarray, x_values: np.ndarray, y_values: np.ndarray) -> float:
    finite = np.asarray(Z, dtype=float)
    mask = np.isfinite(finite)
    if not np.any(mask):
        return float("nan")
    dx = float(np.mean(np.diff(np.asarray(x_values, dtype=float)))) if len(x_values) > 1 else 1.0
    dy = float(np.mean(np.diff(np.asarray(y_values, dtype=float)))) if len(y_values) > 1 else 1.0
    return float(np.sum(finite[mask]) * dx * dy)


def _interpolate_natural_neighbor(x, y, z, X, Y, cfg, logger):
    backend_options = cfg.get("backend_options", {})
    if not isinstance(backend_options, Mapping):
        backend_options = {}
    nan_policy = str(cfg.get("nan_policy", "strict")).strip().lower() or "strict"
    try:
        backend = resolve_backend("natural_neighbor")
    except Exception:
        backend = resolve_backend("natural_neighbor_approx")
    try:
        return np.asarray(
            backend(
                x,
                y,
                z,
                X,
                Y,
                nan_policy=nan_policy,
                diagnostics=bool(cfg.get("diagnostics", True)),
                backend_options=dict(backend_options),
            ),
            dtype=float,
        )
    except Exception as exc:
        try:
            backend = resolve_backend("natural_neighbor_approx")
        except Exception:
            raise
        _logger_emit(
            logger,
            "warning",
            f"make_interp_2d natural_neighbor exact backend failed; using natural_neighbor_approx. reason={exc}",
        )
        return np.asarray(
            backend(
                x,
                y,
                z,
                X,
                Y,
                nan_policy=nan_policy,
                diagnostics=bool(cfg.get("diagnostics", True)),
                backend_options=dict(backend_options),
            ),
            dtype=float,
        )


def _interpolate_triangulation(x, y, z, X, Y, cfg):
    from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator

    tri_cfg = cfg.get("triangulation", {})
    if not isinstance(tri_cfg, Mapping):
        tri_cfg = {}
    kind = str(tri_cfg.get("kind", cfg.get("kind", "linear"))).strip().lower()
    points = np.column_stack([x, y])
    if kind == "linear":
        interp = LinearNDInterpolator(points, z, fill_value=np.nan)
    elif kind == "cubic":
        interp = CloughTocher2DInterpolator(points, z, fill_value=np.nan)
    else:
        raise ValueError("make_interp_2d triangulation.kind must be linear or cubic.")
    return np.asarray(interp(X, Y), dtype=float)


def _mask_outside_hull(x, y, values):
    from scipy.spatial import Delaunay

    points = np.column_stack([x, y])
    tri = Delaunay(points)
    query = np.column_stack([np.asarray(values[0]).ravel(), np.asarray(values[1]).ravel()])
    inside = tri.find_simplex(query) >= 0
    return inside.reshape(np.asarray(values[0]).shape)


def _interpolate_griddata(x, y, z, X, Y, cfg):
    from scipy.interpolate import griddata

    gd_cfg = cfg.get("griddata", {})
    if not isinstance(gd_cfg, Mapping):
        gd_cfg = {}
    kind = str(gd_cfg.get("kind", cfg.get("kind", "linear"))).strip().lower()
    if kind not in {"nearest", "linear", "cubic"}:
        raise ValueError("make_interp_2d griddata.kind must be nearest, linear, or cubic.")
    points = np.column_stack([x, y])
    Z = np.asarray(griddata(points, z, (X, Y), method=kind, fill_value=np.nan), dtype=float)
    if str(cfg.get("nan_policy", "strict")).strip().lower() == "strict":
        inside = _mask_outside_hull(x, y, (X, Y))
        Z = Z.copy()
        Z[~inside] = np.nan
    return Z


def make_interp_2d(df: pd.DataFrame, cfg: Mapping[str, Any], logger=None) -> pd.DataFrame:
    """Interpolate support/core (x, y, z) samples onto a regular 2D field grid.

    With as_density=false, z is treated as a generic scalar. With
    as_density=true, z is treated as conserved support mass and converted to a
    local support density before interpolation.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if not isinstance(cfg, Mapping):
        cfg = {}

    x_raw = _resolve_array(df, cfg, "x", "x", logger)
    y_raw = _resolve_array(df, cfg, "y", "y", logger)
    z_raw = _resolve_array(df, cfg, "z", "z", logger)
    n = min(len(x_raw), len(y_raw), len(z_raw), int(df.shape[0]))
    x_raw = np.asarray(x_raw[:n], dtype=float)
    y_raw = np.asarray(y_raw[:n], dtype=float)
    z_raw = np.asarray(z_raw[:n], dtype=float)

    xlim = _axis_lim(x_raw, cfg, "x")
    ylim = _axis_lim(y_raw, cfg, "y")
    xscale = _axis_scale(cfg, "x")
    yscale = _axis_scale(cfg, "y")

    x = _project_axis(x_raw, xlim, xscale, "x")
    y = _project_axis(y_raw, ylim, yscale, "y")
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z_raw)
    valid &= (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)
    valid_count = int(np.count_nonzero(valid))
    if valid_count < 3:
        raise ValueError("make_interp_2d requires at least three finite support points inside the interpolation limits.")
    if valid_count != n:
        _logger_emit(logger, "warning", f"make_interp_2d dropped {n - valid_count} invalid or outside-domain support points.")

    as_density = bool(cfg.get("as_density", False))
    reducer = "sum" if as_density else "mean"
    x, y, z = _dedupe_points(x[valid], y[valid], z_raw[valid], reducer=reducer)
    if len(x) < 3:
        raise ValueError("make_interp_2d requires at least three unique support points after duplicate merging.")
    x_phys = _unproject_axis(x, xlim, xscale)
    y_phys = _unproject_axis(y, ylim, yscale)

    area_method = "not_performed"
    if as_density:
        areas, area_method = _support_areas(x_phys, y_phys, xlim, ylim)
        z = np.asarray(z, dtype=float) / areas

    nx, ny = _grid_shape(cfg)
    qx_norm = np.linspace(0.0, 1.0, nx, dtype=float)
    qy_norm = np.linspace(0.0, 1.0, ny, dtype=float)
    Xn, Yn = np.meshgrid(qx_norm, qy_norm)

    method = str(cfg.get("method", "natural_neighbor")).strip().lower()
    if method in {"natural_neighbor", "natural-neighbor", "nn"}:
        Z = _interpolate_natural_neighbor(x, y, z, Xn, Yn, cfg, logger)
        method_name = "natural_neighbor"
    elif method in {"triangulation", "triangulate", "delaunay"}:
        Z = _interpolate_triangulation(x, y, z, Xn, Yn, cfg)
        method_name = "triangulation"
    elif method == "griddata":
        Z = _interpolate_griddata(x, y, z, Xn, Yn, cfg)
        method_name = "griddata"
    else:
        raise ValueError("make_interp_2d method must be natural_neighbor, triangulation, or griddata.")

    x_out = _output_name(cfg, "x", "x")
    y_out = _output_name(cfg, "y", "y")
    z_out = _output_name(cfg, "z", "z")
    x_grid = _unproject_axis(qx_norm, xlim, xscale)
    y_grid = _unproject_axis(qy_norm, ylim, yscale)
    Xp, Yp = np.meshgrid(x_grid, y_grid)

    normalize = bool(cfg.get("normalize", False))
    integral_before = _grid_integral(Z, x_grid, y_grid)
    integral_after = integral_before
    if normalize:
        if not np.isfinite(integral_before) or integral_before <= 0:
            raise ValueError(f"make_interp_2d cannot normalize grid with integral={integral_before}.")
        Z = np.asarray(Z, dtype=float).copy()
        mask = np.isfinite(Z)
        Z[mask] = Z[mask] / integral_before
        integral_after = _grid_integral(Z, x_grid, y_grid)

    if bool(cfg.get("diagnostics", True)):
        finite_z = Z[np.isfinite(Z)]
        z_range = "nan"
        if finite_z.size:
            z_range = f"{float(np.min(finite_z)):.17g} -> {float(np.max(finite_z)):.17g}"
        _logger_emit(
            logger,
            "warning",
            "make_interp_2d diagnostics:\n"
            f"\t method \t-> {method_name}\n"
            f"\t input_points \t-> {n}\n"
            f"\t valid_points \t-> {valid_count}\n"
            f"\t unique_points \t-> {len(x)}\n"
            f"\t xlim \t\t-> [{xlim[0]}, {xlim[1]}]\n"
            f"\t ylim \t\t-> [{ylim[0]}, {ylim[1]}]\n"
            f"\t xscale \t-> {xscale}\n"
            f"\t yscale \t-> {yscale}\n"
            f"\t grid_shape \t-> ({ny}, {nx})\n"
            f"\t as_density \t-> {as_density}\n"
            f"\t area_method \t-> {area_method}\n"
            f"\t normalize \t-> {normalize}\n"
            f"\t integral_before -> {integral_before:.17g}\n"
            f"\t integral_after \t-> {integral_after:.17g}\n"
            f"\t output_columns -> [{x_out}, {y_out}, {z_out}]\n"
            f"\t z_minmax \t-> {z_range}",
        )

    return pd.DataFrame(
        {
            x_out: Xp.ravel(),
            y_out: Yp.ravel(),
            z_out: np.asarray(Z, dtype=float).ravel(),
        }
    )

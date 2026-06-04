from __future__ import annotations

import re
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from ..memtrace import memtrace_checkpoint
from ..utils.pathing import resolve_project_path
from ..utils.expression import eval_dataframe_expression
from .posterior_mesh import build_posterior_mesh, conservative_density, mesh_diagnostics_frame


_DENSITY_CELL_TYPES = {"make_density_core"}


class _ScaledBwMethod:
    def __init__(self, factor: float, base: str):
        self.factor = float(factor)
        self.base = str(base).strip().lower()

    def __call__(self, kde) -> float:
        if self.base == "silverman":
            return self.factor * float(kde.silverman_factor())
        return self.factor * float(kde.scotts_factor())

    def __str__(self) -> str:
        return f"{self.factor:g}*{self.base}"


def is_density_cell_transform(step: Any) -> bool:
    if not isinstance(step, Mapping):
        return False
    if any(name in step for name in _DENSITY_CELL_TYPES):
        return True
    return str(step.get("type", "")).strip().lower() in _DENSITY_CELL_TYPES


def density_cell_config(step: Mapping[str, Any]) -> Mapping[str, Any]:
    for name in _DENSITY_CELL_TYPES:
        if name in step:
            cfg = step.get(name)
            return cfg if isinstance(cfg, Mapping) else {}
    if str(step.get("type", "")).strip().lower() in _DENSITY_CELL_TYPES:
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
    output = cfg.get("output", {})
    if isinstance(output, Mapping) and output.get(key) is not None:
        name = output.get(key, default)
    else:
        name = _coord_cfg(cfg, key).get("name", default)
    text = str(name).strip()
    return text or default


def _resolve_array(df: pd.DataFrame, cfg: Mapping[str, Any], key: str, default: str, logger, *, required: bool) -> np.ndarray:
    spec = _coord_cfg(cfg, key)
    expr = spec.get("expr")
    if expr is not None:
        return np.asarray(eval_dataframe_expression(df, expr, logger=logger, allow_column=True), dtype=float)
    name = str(spec.get("name", default)).strip()
    if not name:
        if required:
            raise ValueError(f"make_density_core requires a non-empty coordinate for '{key}'.")
        return np.ones(int(df.shape[0]), dtype=float)
    if name not in df.columns:
        if required:
            raise KeyError(f"make_density_core missing required column '{name}'.")
        return np.ones(int(df.shape[0]), dtype=float)
    return np.asarray(df[name], dtype=float)


def _axis_lim(values: np.ndarray, cfg: Mapping[str, Any], key: str) -> Tuple[float, float]:
    spec = _coord_cfg(cfg, key)
    lim = spec.get("lim", spec.get("limits", None))
    domain = cfg.get("domain", {})
    if not isinstance(domain, Mapping):
        domain = {}
    if lim is None:
        lim = domain.get(f"{key}lim", cfg.get(f"{key}lim", None))
    if lim is None:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError(f"make_density_core cannot infer {key} limits from non-finite data.")
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if lo == hi:
            hi = lo + 1.0
        return lo, hi
    if not isinstance(lim, Sequence) or len(lim) != 2:
        raise ValueError(f"make_density_core coordinates.{key}.lim must have two values.")
    lo, hi = float(lim[0]), float(lim[1])
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise ValueError(f"make_density_core coordinates.{key}.lim requires finite lo < hi.")
    return lo, hi


def _axis_scale(cfg: Mapping[str, Any], key: str) -> str:
    spec = _coord_cfg(cfg, key)
    domain = cfg.get("domain", {})
    if not isinstance(domain, Mapping):
        domain = {}
    return str(spec.get("scale", domain.get(f"{key}scale", cfg.get(f"{key}scale", "linear")))).strip().lower()


def _project_axis(values: np.ndarray, lim: Tuple[float, float], scale: str, key: str) -> np.ndarray:
    lo, hi = lim
    values = np.asarray(values, dtype=float)
    if scale == "log":
        if lo <= 0 or hi <= 0:
            raise ValueError(f"make_density_core {key} log scale requires positive limits.")
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


def _grid_shape(cfg: Mapping[str, Any], default: int = 64) -> Tuple[int, int]:
    grid = cfg.get("grid", {})
    if not isinstance(grid, Mapping):
        grid = {}
    bins = grid.get("bins", grid.get("bin", cfg.get("bins", cfg.get("bin", None))))
    if bins is not None:
        nx = ny = int(bins)
    else:
        nx = int(grid.get("nx", cfg.get("nx", default)))
        ny = int(grid.get("ny", cfg.get("ny", default)))
    if nx < 2 or ny < 2:
        raise ValueError("make_density_core grid requires nx >= 2 and ny >= 2.")
    return nx, ny


def _bw_method(cfg: Mapping[str, Any]):
    bw = cfg.get("bw_method", cfg.get("bandwidth", None))
    if bw is None:
        return None
    if isinstance(bw, str):
        text = bw.strip().lower()
        if text in {"", "default", "none"}:
            return None
        if text in {"scott", "silverman"}:
            return text
        match = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*\*\s*(scott|silverman)", text)
        if match:
            factor = float(match.group(1))
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError("make_density_core scaled bw_method factor must be positive and finite.")
            return _ScaledBwMethod(factor, match.group(2))
        value = float(text)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("make_density_core float bw_method must be positive and finite.")
        return value
    value = float(bw)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("make_density_core float bw_method must be positive and finite.")
    return value


def _prepare_samples(df: pd.DataFrame, cfg: Mapping[str, Any], logger):
    x = _resolve_array(df, cfg, "x", "x", logger, required=True)
    y = _resolve_array(df, cfg, "y", "y", logger, required=True)
    weight_required = bool(_coord_cfg(cfg, "weight"))
    w = _resolve_array(df, cfg, "weight", "weight", logger, required=weight_required)
    n = min(len(x), len(y), len(w), int(df.shape[0]))
    x = np.asarray(x[:n], dtype=float)
    y = np.asarray(y[:n], dtype=float)
    w = np.asarray(w[:n], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if not np.any(valid):
        raise ValueError("make_density_core got no finite samples with positive finite weights.")
    if not np.all(valid):
        _logger_emit(logger, "warning", f"make_density_core dropped {int((~valid).sum())} invalid or non-positive samples.")
    return x[valid], y[valid], w[valid], n


def _normalize(values: np.ndarray, *, enabled: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not enabled:
        return values
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("make_density_core cannot normalize non-positive or non-finite total weight.")
    return values / total


def _grid_density_cell(x, y, w, cfg, logger):
    xlim = _axis_lim(x, cfg, "x")
    ylim = _axis_lim(y, cfg, "y")
    xscale = _axis_scale(cfg, "x")
    yscale = _axis_scale(cfg, "y")
    nx, ny = _grid_shape(cfg)
    xp = _project_axis(x, xlim, xscale, "x")
    yp = _project_axis(y, ylim, yscale, "y")
    inside = np.isfinite(xp) & np.isfinite(yp) & (xp >= 0.0) & (xp <= 1.0) & (yp >= 0.0) & (yp <= 1.0)
    if not np.any(inside):
        raise ValueError("make_density_core grid found no positive-mass samples inside the domain.")
    if not np.all(inside):
        _logger_emit(logger, "warning", f"make_density_core grid ignored {int((~inside).sum())} samples outside the domain.")
    ix = np.clip(xp[inside], 0.0, 1.0 - 1e-12)
    iy = np.clip(yp[inside], 0.0, 1.0 - 1e-12)
    ix = (ix * nx).astype(np.int32)
    iy = (iy * ny).astype(np.int32)
    mass = np.zeros((ny, nx), dtype=float)
    np.add.at(mass, (iy, ix), w[inside])
    gx = (np.arange(nx, dtype=float) + 0.5) / float(nx)
    gy = (np.arange(ny, dtype=float) + 0.5) / float(ny)
    xg, yg = np.meshgrid(_unproject_axis(gx, xlim, xscale), _unproject_axis(gy, ylim, yscale))
    return xg.ravel(), yg.ravel(), mass.ravel()


def _bridson_points(radius: float, *, seed: int | None = None, k: int = 30) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cell = radius / np.sqrt(2.0)
    nx = max(1, int(np.ceil(1.0 / cell)))
    ny = max(1, int(np.ceil(1.0 / cell)))
    grid = -np.ones((ny, nx), dtype=np.int64)
    points: list[tuple[float, float]] = []
    active: list[int] = []

    def _grid_pos(pt):
        gx = min(nx - 1, max(0, int(pt[0] / cell)))
        gy = min(ny - 1, max(0, int(pt[1] / cell)))
        return gx, gy

    def _far_enough(pt) -> bool:
        gx, gy = _grid_pos(pt)
        r2 = radius * radius
        for yy in range(max(0, gy - 2), min(ny, gy + 3)):
            for xx in range(max(0, gx - 2), min(nx, gx + 3)):
                idx = int(grid[yy, xx])
                if idx < 0:
                    continue
                px, py = points[idx]
                if (pt[0] - px) ** 2 + (pt[1] - py) ** 2 < r2:
                    return False
        return True

    first = (float(rng.random()), float(rng.random()))
    points.append(first)
    gx, gy = _grid_pos(first)
    grid[gy, gx] = 0
    active.append(0)
    while active:
        slot = int(rng.integers(0, len(active)))
        idx = active[slot]
        base = points[idx]
        accepted = False
        for _ in range(int(k)):
            rr = radius * (1.0 + rng.random())
            theta = 2.0 * np.pi * rng.random()
            cand = (base[0] + rr * np.cos(theta), base[1] + rr * np.sin(theta))
            if cand[0] < 0.0 or cand[0] > 1.0 or cand[1] < 0.0 or cand[1] > 1.0:
                continue
            if not _far_enough(cand):
                continue
            points.append((float(cand[0]), float(cand[1])))
            new_idx = len(points) - 1
            gx, gy = _grid_pos(cand)
            grid[gy, gx] = new_idx
            active.append(new_idx)
            accepted = True
            break
        if not accepted:
            active.pop(slot)
    return np.asarray(points, dtype=float)


def _bridson_density_cell(x, y, w, cfg, logger):
    xlim = _axis_lim(x, cfg, "x")
    ylim = _axis_lim(y, cfg, "y")
    xscale = _axis_scale(cfg, "x")
    yscale = _axis_scale(cfg, "y")
    xp = _project_axis(x, xlim, xscale, "x")
    yp = _project_axis(y, ylim, yscale, "y")
    inside = np.isfinite(xp) & np.isfinite(yp) & (xp >= 0.0) & (xp <= 1.0) & (yp >= 0.0) & (yp <= 1.0)
    if not np.any(inside):
        raise ValueError("make_density_core bridson found no positive-mass samples inside the domain.")
    if not np.all(inside):
        _logger_emit(logger, "warning", f"make_density_core bridson ignored {int((~inside).sum())} samples outside the domain.")
    bridson = cfg.get("bridson", {})
    if not isinstance(bridson, Mapping):
        bridson = {}
    bin_value = int(bridson.get("bin", bridson.get("bins", cfg.get("bin", cfg.get("bins", 64)))))
    if bin_value <= 0:
        raise ValueError("make_density_core bridson bin must be positive.")
    radius = 1.0 / float(bin_value)
    seed = bridson.get("seed", cfg.get("seed", None))
    seed = None if seed is None else int(seed)
    k = int(bridson.get("k", bridson.get("candidates", 30)))
    support = _bridson_points(radius, seed=seed, k=k)
    samples = np.column_stack([xp[inside], yp[inside]])
    mesh_cfg = cfg.get("_mesh_debug", {})
    if not isinstance(mesh_cfg, Mapping):
        mesh_cfg = {}
    refinement_cfg = cfg.get("refinement", mesh_cfg.get("refinement", None))
    mesh = build_posterior_mesh(
        support,
        samples,
        w[inside],
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        include_polygons=bool(mesh_cfg.get("include_polygons", False)),
        strict_geometry=False,
        geometry_space="normalized",
        refinement=refinement_cfg if isinstance(refinement_cfg, Mapping) else None,
    )
    _bridson_density_cell.last_mesh = mesh
    mass = mesh.masses
    xs = _unproject_axis(mesh.generators[:, 0], xlim, xscale)
    ys = _unproject_axis(mesh.generators[:, 1], ylim, yscale)
    return xs, ys, mass


_bridson_density_cell.last_mesh = None  # type: ignore[attr-defined]


def _kde_core(x, y, w, cfg, logger):
    xlim = _axis_lim(x, cfg, "x")
    ylim = _axis_lim(y, cfg, "y")
    xscale = _axis_scale(cfg, "x")
    yscale = _axis_scale(cfg, "y")
    if xscale != "linear" or yscale != "linear":
        raise ValueError("make_density_core method=kde currently requires linear x/y scales.")
    nx, ny = _grid_shape(cfg)
    if x.shape[0] < 3:
        raise ValueError("make_density_core method=kde requires at least three finite 2D samples.")
    pts = np.vstack([x, y])
    if np.linalg.matrix_rank(pts - pts.mean(axis=1, keepdims=True)) < 2:
        raise ValueError("make_density_core method=kde requires non-collinear 2D samples.")
    weights = _normalize(w, enabled=True)
    try:
        from scipy.stats import gaussian_kde
    except Exception as e:
        raise ImportError("make_density_core method=kde requires scipy.stats.gaussian_kde.") from e
    kde = gaussian_kde(dataset=pts, bw_method=_bw_method(cfg), weights=weights)
    xmin, xmax = xlim
    ymin, ymax = ylim
    dx = (xmax - xmin) / float(nx)
    dy = (ymax - ymin) / float(ny)
    x_centers = np.linspace(xmin + 0.5 * dx, xmax - 0.5 * dx, nx)
    y_centers = np.linspace(ymin + 0.5 * dy, ymax - 0.5 * dy, ny)
    xg, yg = np.meshgrid(x_centers, y_centers)
    density = np.asarray(kde(np.vstack([xg.ravel(), yg.ravel()])), dtype=float).reshape(ny, nx)
    density = np.maximum(np.where(np.isfinite(density), density, 0.0), 0.0)
    mass = density.ravel() * dx * dy
    return xg.ravel(), yg.ravel(), mass


def _diagnostics(*, method: str, input_count: int, output_count: int, normalize: bool, names: Tuple[str, str, str], total: float) -> str:
    return (
        "make_density_core diagnostics:\n"
        f"\t method \t-> {method}\n"
        f"\t input_samples \t-> {input_count}\n"
        f"\t output_support \t-> {output_count}\n"
        f"\t normalize \t-> {normalize}\n"
        f"\t output_columns \t-> {names[0]}, {names[1]}, {names[2]}\n"
        f"\t output_weight_sum \t-> {total:.17g}"
    )


def _mesh_diagnostics(mesh) -> str:
    if mesh is None:
        return ""
    diag = getattr(mesh, "diagnostics", {}) or {}
    return (
        "\nposterior mesh diagnostics:\n"
        f"\t generators \t-> {int(diag.get('generator_count', 0))}\n"
        f"\t nonzero_mass \t-> {int(diag.get('nonzero_mass_generators', 0))}\n"
        f"\t empty_generators -> {int(diag.get('empty_generators', 0))}\n"
        f"\t area_method \t-> {diag.get('area_method', 'not_performed')}\n"
        f"\t geometry_space \t-> {diag.get('geometry_space', 'unknown')}\n"
        f"\t max_centroid_drift -> {float(diag.get('max_centroid_drift', float('nan'))):.17g}\n"
        f"\t max_anisotropy \t-> {float(diag.get('max_anisotropy_ratio', float('nan'))):.17g}\n"
        f"\t mass_vs_geometry_corr -> {float(diag.get('mass_geometry_stress_corr', float('nan'))):.17g}\n"
        f"\t refinement_enabled -> {bool(diag.get('refinement_enabled', False))}\n"
        f"\t refinement_iters -> {int(diag.get('refinement_iterations_completed', 0))}\n"
        f"\t refinement_splits -> {int(diag.get('refinement_splits', 0))}\n"
        f"\t refinement_merges -> {int(diag.get('refinement_merges', 0))}"
    )


def _export_mesh_debug(mesh, cfg: Mapping[str, Any], xs: np.ndarray, ys: np.ndarray, logger) -> None:
    if mesh is None:
        return
    mesh_cfg = cfg.get("_mesh_debug", {})
    if not isinstance(mesh_cfg, Mapping):
        return
    target = mesh_cfg.get("to_csv", mesh_cfg.get("path", None))
    if target is None:
        return
    path = str(target).strip()
    if not path:
        return
    base_dir = cfg.get("_base_dir", None)
    out_path = resolve_project_path(path, base_dir=base_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = mesh_diagnostics_frame(mesh, x=xs, y=ys)
    try:
        xlim = _axis_lim(np.asarray(xs, dtype=float), cfg, "x")
        ylim = _axis_lim(np.asarray(ys, dtype=float), cfg, "y")
        xscale = _axis_scale(cfg, "x")
        yscale = _axis_scale(cfg, "y")
        centroid_x = _unproject_axis(mesh.centroid_proposal[:, 0], xlim, xscale)
        centroid_y = _unproject_axis(mesh.centroid_proposal[:, 1], ylim, yscale)
        frame["centroid_x"] = centroid_x
        frame["centroid_y"] = centroid_y
        frame["drift_x"] = centroid_x - np.asarray(xs, dtype=float)
        frame["drift_y"] = centroid_y - np.asarray(ys, dtype=float)
    except Exception:
        pass
    frame.to_csv(out_path, index=False)
    _logger_emit(logger, "warning", f"posterior mesh diagnostics exported -> {out_path}")


def density_cell(df: pd.DataFrame, cfg: Mapping[str, Any], logger=None) -> pd.DataFrame:
    """
    Construct a minimal posterior density-support table from raw samples.

    The output contains exactly three columns: support x, support y, and
    aggregated posterior mass/weight. Geometry, density division, smoothing, and
    plotting preparation are intentionally left to later transforms/rendering.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("make_density_core expects a pandas DataFrame.")
    cfg = cfg if isinstance(cfg, Mapping) else {}
    method = str(cfg.get("method", "bridson")).strip().lower()
    if method not in {"grid", "bridson", "kde"}:
        raise ValueError("make_density_core method must be one of: grid, bridson, kde.")
    memtrace_checkpoint(logger, "make_density_core.before", df)
    density_cell.last_mesh = None
    x, y, w, input_count = _prepare_samples(df, cfg, logger)
    if method == "grid":
        xs, ys, mass = _grid_density_cell(x, y, w, cfg, logger)
    elif method == "bridson":
        xs, ys, mass = _bridson_density_cell(x, y, w, cfg, logger)
        density_cell.last_mesh = _bridson_density_cell.last_mesh
    else:
        xs, ys, mass = _kde_core(x, y, w, cfg, logger)
    normalize = bool(cfg.get("normalize", True))
    mass = _normalize(mass, enabled=normalize)
    if density_cell.last_mesh is not None:
        density_cell.last_mesh.masses = np.asarray(mass, dtype=float)
        density_cell.last_mesh.densities = conservative_density(
            density_cell.last_mesh.masses,
            density_cell.last_mesh.areas,
        )
        density_cell.last_mesh.diagnostics["mass_sum"] = float(np.sum(density_cell.last_mesh.masses))
        _export_mesh_debug(density_cell.last_mesh, cfg, xs, ys, logger)
    names = (
        _output_name(cfg, "x", "x"),
        _output_name(cfg, "y", "y"),
        _output_name(cfg, "weight", "weight"),
    )
    out = pd.DataFrame({names[0]: np.asarray(xs, dtype=float), names[1]: np.asarray(ys, dtype=float), names[2]: mass})
    if list(out.columns) != list(names):
        out = out.loc[:, list(names)]
    msg = _diagnostics(
        method=method,
        input_count=int(input_count),
        output_count=int(out.shape[0]),
        normalize=normalize,
        names=names,
        total=float(np.sum(out[names[2]].to_numpy(dtype=float))),
    )
    if bool(cfg.get("diagnostics", True)):
        if method == "bridson":
            msg += _mesh_diagnostics(density_cell.last_mesh)
        _logger_emit(logger, "warning", msg)
    memtrace_checkpoint(
        logger,
        "make_density_core.after",
        out,
        extra={"method": method, "rows": int(out.shape[0]), "weight_sum": float(np.sum(out[names[2]]))},
    )
    return out


density_cell.last_mesh = None  # type: ignore[attr-defined]

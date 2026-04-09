#!/usr/bin/env python3
"""Interpolation backend registry.

`natural_neighbor` is reserved for the exact Sibson/Voronoi backend.
`natural_neighbor_approx` keeps the legacy Delaunay/Clough-Tocher approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Any, Callable, Dict, Optional

import numpy as np


try:  # SciPy is a runtime dependency in pyproject, but keep import lazy-ish.
    from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
    from scipy.spatial import Delaunay, cKDTree
    from scipy.spatial import QhullError
except Exception:  # pragma: no cover - exercised only in minimal environments
    CloughTocher2DInterpolator = None
    LinearNDInterpolator = None
    Delaunay = None
    cKDTree = None
    QhullError = Exception


__all__ = [
    "NaturalNeighborDiagnostics",
    "NaturalNeighborInterpolator",
    "NaturalNeighborApproxInterpolator",
    "natural_neighbor_interpolate",
    "natural_neighbor_approx_interpolate",
    "register_backend",
    "resolve_backend",
]


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKENDS: Dict[str, Callable[..., np.ndarray]] = {}


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def register_backend(name: str, fn: Callable[..., np.ndarray], *, overwrite: bool = False) -> None:
    key = _normalize_name(name)
    if (not overwrite) and key in _BACKENDS:
        raise ValueError(f"Interpolation backend already registered: {key}")
    _BACKENDS[key] = fn


def resolve_backend(name: str) -> Callable[..., np.ndarray]:
    key = _normalize_name(name)
    if key not in _BACKENDS:
        raise KeyError(f"Unknown interpolation backend: {name!r}")
    return _BACKENDS[key]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass
class NaturalNeighborDiagnostics:
    backend: str = "natural_neighbor_approx"
    implementation: str = "approximate-delaunay-clough-tocher"
    nan_policy: str = "strict"
    input_points: int = 0
    finite_xy_points: int = 0
    unique_points: int = 0
    unique_finite_points: int = 0
    query_points: int = 0
    inside_hull: int = 0
    outside_hull: int = 0
    exact_hits: int = 0
    masked_by_nan: int = 0
    all_nan_cores: bool = False
    degenerate_input: bool = False
    exact_duplicate_groups: int = 0
    near_duplicate_groups: int = 0
    merged_points: int = 0
    nominal_point_spacing: float = 0.0
    vertex_tolerance: float = 0.0
    interpolator_ready: bool = False


def _warn(msg: str) -> None:
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _as_1d_float(arr: Any, *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=float).reshape(-1)
    if out.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return out


def _as_grid_float(arr: Any, *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim != 2:
        raise ValueError(f"{name} must be a 2D grid")
    return out


def _dedupe_coordinates(
    coords: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Collapse exact duplicate (x, y) coordinates.

    Strict NaN handling:
    - if any duplicate value is NaN, the consolidated value is NaN.
    - duplicate finite values are merged using their mean value.
    """
    if coords.size == 0:
        return coords, values, 0, 0

    groups: dict[tuple[float, float], list[int]] = {}
    order: list[tuple[float, float]] = []
    for idx, pt in enumerate(np.asarray(coords, dtype=float).reshape(-1, 2)):
        key = (float(pt[0]), float(pt[1]))
        if key not in groups:
            groups[key] = [int(idx)]
            order.append(key)
        else:
            groups[key].append(int(idx))

    uniq = np.empty((len(order), 2), dtype=float)
    out = np.empty(len(order), dtype=float)
    exact_duplicate_groups = 0
    exact_duplicate_points = 0

    for out_idx, key in enumerate(order):
        group_idx = groups[key]
        group = values[np.asarray(group_idx, dtype=int)]
        uniq[out_idx] = coords[int(group_idx[0])]
        if group.size == 0:
            out[out_idx] = np.nan
            continue
        if np.isnan(group).any():
            out[out_idx] = np.nan
            continue
        out[out_idx] = float(np.mean(group))
        if len(group_idx) > 1:
            exact_duplicate_groups += 1
            exact_duplicate_points += len(group_idx) - 1

    return uniq, out, exact_duplicate_groups, exact_duplicate_points


def _estimate_nominal_spacing(coords: np.ndarray) -> float:
    if coords.size == 0 or coords.shape[0] < 2:
        return 1.0
    try:
        tree = cKDTree(coords)
        dists, _ = tree.query(coords, k=2)
        nn = np.asarray(dists[:, 1], dtype=float)
        finite = nn[np.isfinite(nn) & (nn > 0)]
        if finite.size == 0:
            return 1.0
        spacing = float(np.median(finite))
        if np.isfinite(spacing) and spacing > 0:
            return spacing
    except Exception:
        pass
    return 1.0


def _effective_scale(coords: np.ndarray) -> float:
    if coords.size == 0:
        return 1.0
    span = np.ptp(coords, axis=0)
    scale = float(np.max(span)) if np.ndim(span) else float(span)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.max(np.abs(coords))) if coords.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return scale


class NaturalNeighborInterpolator:
    """First-pass natural-neighbor-like interpolator.

    This implementation uses a Delaunay triangulation plus a smooth local
    interpolator when SciPy is available. It is not a full Sibson/Voronoi
    implementation yet, but the interface and diagnostics are intentionally
    stable so a true backend can replace it later.
    """

    def __init__(
        self,
        x,
        y,
        z,
        *,
        nan_policy: str = "strict",
        backend_options: Optional[dict[str, Any]] = None,
    ):
        self.nan_policy = self._normalize_nan_policy(nan_policy)
        self.backend_options = dict(backend_options or {})
        self.diagnostics = NaturalNeighborDiagnostics(nan_policy=self.nan_policy)

        self._coords: Optional[np.ndarray] = None
        self._values: Optional[np.ndarray] = None
        self._tree: Any = None
        self._vertex_tol: float = 0.0
        self._tri: Any = None
        self._simplex_has_nan: Optional[np.ndarray] = None
        self._value_interpolator: Any = None

        self._build(x, y, z)

    @staticmethod
    def _normalize_nan_policy(nan_policy: str) -> str:
        key = str(nan_policy).strip().lower()
        if key in {"strict", "propagate", "mask"}:
            return "strict"
        raise ValueError(
            "nan_policy must be one of {'strict', 'propagate', 'mask'} for this backend"
        )

    def _build(self, x, y, z) -> None:
        if Delaunay is None or cKDTree is None:
            raise ImportError(
                "natural_neighbor interpolation requires SciPy (scipy.spatial / scipy.interpolate)."
            )

        x = _as_1d_float(x, name="x")
        y = _as_1d_float(y, name="y")
        z = _as_1d_float(z, name="z")

        n = min(x.size, y.size, z.size)
        self.diagnostics.input_points = int(n)
        if n == 0:
            self.diagnostics.degenerate_input = True
            self.diagnostics.implementation = "empty"
            _warn("natural_neighbor: no input points were provided")
            return

        x = x[:n]
        y = y[:n]
        z = z[:n]

        finite_xy = np.isfinite(x) & np.isfinite(y)
        self.diagnostics.finite_xy_points = int(np.count_nonzero(finite_xy))
        if not np.any(finite_xy):
            self.diagnostics.degenerate_input = True
            self.diagnostics.implementation = "empty"
            _warn("natural_neighbor: no finite (x, y) coordinates are available")
            return

        coords = np.column_stack([x[finite_xy], y[finite_xy]])
        values = z[finite_xy]

        coords, values, exact_duplicate_groups, exact_duplicate_points = _dedupe_coordinates(
            coords, values
        )
        self.diagnostics.exact_duplicate_groups = int(exact_duplicate_groups)
        self.diagnostics.merged_points = int(exact_duplicate_points)

        scale = _effective_scale(coords)
        nominal_spacing = self.backend_options.get("nominal_point_spacing", None)
        if nominal_spacing is None:
            nominal_spacing = _estimate_nominal_spacing(coords)
        try:
            nominal_spacing = float(nominal_spacing)
        except Exception:
            nominal_spacing = 1.0
        if not np.isfinite(nominal_spacing) or nominal_spacing <= 0:
            nominal_spacing = scale
        self._vertex_tol = float(
            self.backend_options.get(
                "vertex_tol",
                max(nominal_spacing / 1.0e5, 64.0 * np.finfo(float).eps * scale),
            )
        )
        if not np.isfinite(self._vertex_tol) or self._vertex_tol <= 0:
            self._vertex_tol = max(nominal_spacing / 1.0e5, 64.0 * np.finfo(float).eps * scale)

        try:
            from .interp_natural_neighbor_exact import _merge_near_duplicates as _merge_near_duplicates_exact
        except Exception:
            _merge_near_duplicates_exact = None

        if _merge_near_duplicates_exact is not None:
            coords, values, near_duplicate_groups, near_duplicate_points = _merge_near_duplicates_exact(
                coords, values, self._vertex_tol
            )
        else:
            near_duplicate_groups = 0
            near_duplicate_points = 0
        self.diagnostics.near_duplicate_groups = int(near_duplicate_groups)
        self.diagnostics.merged_points += int(near_duplicate_points)

        self._coords = coords
        self._values = values

        self.diagnostics.unique_points = int(coords.shape[0])
        finite_value_mask = np.isfinite(values)
        self.diagnostics.unique_finite_points = int(np.count_nonzero(finite_value_mask))
        self.diagnostics.all_nan_cores = bool(coords.size > 0 and not np.any(finite_value_mask))
        self.diagnostics.nominal_point_spacing = float(nominal_spacing)
        self.diagnostics.vertex_tolerance = float(self._vertex_tol)

        self._tree = cKDTree(coords)

        if self.diagnostics.all_nan_cores:
            self.diagnostics.implementation = "exact-core-only"
            self.diagnostics.interpolator_ready = True
            return

        # A Delaunay triangulation over all coordinates is used for domain
        # masking and to keep NaN-valued cores strictly active in the stencil.
        try:
            self._tri = Delaunay(coords)
            simplex_has_nan = np.any(~finite_value_mask[self._tri.simplices], axis=1)
            if np.any(simplex_has_nan):
                expanded = simplex_has_nan.copy()
                neighbor_ids = np.unique(self._tri.neighbors[simplex_has_nan].ravel())
                neighbor_ids = neighbor_ids[neighbor_ids >= 0]
                if neighbor_ids.size:
                    expanded[neighbor_ids] = True
                self._simplex_has_nan = expanded
            else:
                self._simplex_has_nan = simplex_has_nan
        except QhullError:
            self._tri = None
            self._simplex_has_nan = None
            self.diagnostics.degenerate_input = True
            _warn("natural_neighbor: input points are too degenerate to triangulate")
        except Exception as exc:
            self._tri = None
            self._simplex_has_nan = None
            self.diagnostics.degenerate_input = True
            _warn(f"natural_neighbor: Delaunay triangulation failed: {exc}")

        # Build a smooth finite-only interpolator. Fall back to linear if needed.
        finite_coords = coords[finite_value_mask]
        finite_values = values[finite_value_mask]
        if finite_coords.shape[0] >= 3:
            try:
                self._value_interpolator = CloughTocher2DInterpolator(
                    finite_coords,
                    finite_values,
                    fill_value=np.nan,
                )
                self.diagnostics.implementation = "clough-tocher"
            except Exception:
                try:
                    self._value_interpolator = LinearNDInterpolator(
                        finite_coords,
                        finite_values,
                        fill_value=np.nan,
                    )
                    self.diagnostics.implementation = "linear"
                except Exception as exc:
                    self._value_interpolator = None
                    self.diagnostics.implementation = "exact-core-only"
                    _warn(f"natural_neighbor: interpolation backend could not be built: {exc}")
        else:
            self._value_interpolator = None
            self.diagnostics.degenerate_input = True
            _warn("natural_neighbor: too few finite-valued points for interpolation")

        self.diagnostics.interpolator_ready = True

    def __call__(self, X, Y):
        return self.evaluate(X, Y)

    def evaluate(self, X, Y):
        X = _as_grid_float(X, name="X")
        Y = _as_grid_float(Y, name="Y")
        if X.shape != Y.shape:
            raise ValueError("X and Y must have the same shape")

        pts = np.column_stack([X.ravel(), Y.ravel()])
        out = np.full(pts.shape[0], np.nan, dtype=float)
        self.diagnostics.query_points = int(pts.shape[0])

        if self._coords is None or self._values is None or self._coords.size == 0:
            return out.reshape(X.shape)

        # Exact site hits take precedence and preserve core values exactly.
        dist, idx = self._tree.query(pts, k=1)
        exact_mask = np.isfinite(dist) & (dist < self._vertex_tol)
        if np.any(exact_mask):
            out[exact_mask] = self._values[idx[exact_mask]]
        self.diagnostics.exact_hits = int(np.count_nonzero(exact_mask))

        remaining = ~exact_mask
        if not np.any(remaining):
            self.diagnostics.inside_hull = int(self.diagnostics.exact_hits)
            self.diagnostics.outside_hull = 0
            return out.reshape(X.shape)

        if self._tri is None or self._value_interpolator is None or self._simplex_has_nan is None:
            self.diagnostics.inside_hull = int(self.diagnostics.exact_hits)
            self.diagnostics.outside_hull = int(np.count_nonzero(remaining))
            return out.reshape(X.shape)

        tri_idx = np.flatnonzero(remaining)
        simplex = self._tri.find_simplex(pts[remaining], tol=1e-12)
        inside = simplex >= 0
        self.diagnostics.outside_hull = int(np.count_nonzero(remaining) - np.count_nonzero(inside))
        self.diagnostics.inside_hull = int(self.diagnostics.exact_hits + np.count_nonzero(inside))

        if not np.any(inside):
            return out.reshape(X.shape)

        inside_tri_idx = tri_idx[inside]
        inside_simplex = simplex[inside]
        nan_mask = self._simplex_has_nan[inside_simplex]
        self.diagnostics.masked_by_nan = int(np.count_nonzero(nan_mask))

        safe = ~nan_mask
        if np.any(safe):
            try:
                vals = np.asarray(self._value_interpolator(pts[inside_tri_idx[safe]]), dtype=float)
            except Exception:
                vals = np.full(np.count_nonzero(safe), np.nan, dtype=float)
            out[inside_tri_idx[safe]] = vals

        return out.reshape(X.shape)


NaturalNeighborApproxInterpolator = NaturalNeighborInterpolator


def natural_neighbor_approx_interpolate(
    x: Any,
    y: Any,
    z: Any,
    X: Any,
    Y: Any,
    *,
    nan_policy: str = "strict",
    diagnostics: bool = False,
    backend_options: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    interp = NaturalNeighborApproxInterpolator(
        x,
        y,
        z,
        nan_policy=nan_policy,
        backend_options=backend_options,
    )
    result = interp.evaluate(X, Y)
    natural_neighbor_approx_interpolate.last_diagnostics = interp.diagnostics
    return result


natural_neighbor_approx_interpolate.last_diagnostics = None  # type: ignore[attr-defined]
register_backend("natural_neighbor_approx", natural_neighbor_approx_interpolate, overwrite=True)


try:
    from .interp_natural_neighbor_exact import register_exact_backend as _register_exact_backend

    _register_exact_backend(register_backend)
except Exception as exc:  # pragma: no cover - exact backend should be present in normal installs
    warnings.warn(
        f"natural_neighbor exact backend could not be registered: {exc}",
        RuntimeWarning,
        stacklevel=2,
    )


def natural_neighbor_interpolate(
    x: Any,
    y: Any,
    z: Any,
    X: Any,
    Y: Any,
    *,
    nan_policy: str = "strict",
    diagnostics: bool = False,
    backend_options: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """Interpolate scattered (x, y, z) values onto a regular grid.

    Registry default:
    - `natural_neighbor` resolves to the exact Sibson/Voronoi backend.
    - `natural_neighbor_approx` preserves the previous Delaunay/Clough-Tocher
      approximation under an honest name.
    """
    backend = resolve_backend("natural_neighbor")
    result = backend(
        x,
        y,
        z,
        X,
        Y,
        nan_policy=nan_policy,
        diagnostics=diagnostics,
        backend_options=backend_options,
    )
    natural_neighbor_interpolate.last_diagnostics = getattr(backend, "last_diagnostics", None)
    return np.asarray(result, dtype=float)


natural_neighbor_interpolate.last_diagnostics = None  # type: ignore[attr-defined]

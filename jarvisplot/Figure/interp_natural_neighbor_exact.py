#!/usr/bin/env python3
"""Exact 2D Sibson natural-neighbor interpolation backend.

This implementation follows the same Bowyer-Watson cavity and Sibson
weight construction used by Tinfour:
- natural neighbors are found by Delaunay cavity traversal,
- weights are computed from the exact area-difference formula
  (wXY - wThiessen),
- coincident and near-coincident cores are merged with a MeanValue rule
  and a Tinfour-style vertex tolerance,
- query points outside the convex hull return NaN,
- NaN-valued natural neighbors propagate strictly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
import warnings
from typing import Any, Callable, Optional

import numpy as np


try:  # SciPy is required at runtime, but keep import failure explicit and local.
    from scipy.spatial import ConvexHull, Delaunay, cKDTree, QhullError
except Exception:  # pragma: no cover - exercised only in minimal environments
    ConvexHull = None
    Delaunay = None
    cKDTree = None
    QhullError = Exception


__all__ = [
    "NaturalNeighborExactDiagnostics",
    "NaturalNeighborExactInterpolator",
    "natural_neighbor_exact_interpolate",
    "register_exact_backend",
]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass
class NaturalNeighborExactDiagnostics:
    backend: str = "natural_neighbor"
    implementation: str = "exact-sibson-2d"
    nan_policy: str = "strict"
    diagnostics_requested: bool = False
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
    degenerate_queries: int = 0
    exact_duplicate_groups: int = 0
    near_duplicate_groups: int = 0
    merged_points: int = 0
    nominal_point_spacing: float = 0.0
    vertex_tolerance: float = 0.0
    boundary_tolerance: float = 0.0
    cavity_triangles: int = 0
    area_of_embedded_polygon: float = 0.0
    barycentric_coordinate_deviation: float = 0.0
    interpolator_ready: bool = False
    build_seconds: float = 0.0
    eval_seconds: float = 0.0


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


def _normalize_nan_policy(nan_policy: str) -> str:
    key = str(nan_policy).strip().lower()
    if key in {"strict", "propagate", "mask"}:
        return "strict"
    raise ValueError("nan_policy must be one of {'strict', 'propagate', 'mask'}")


def _dedupe_coordinates(
    coords: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Collapse exact duplicate coordinates while preserving strict NaN semantics.

    Tinfour's default coincident-vertex resolution rule is MeanValue, so
    duplicate cores are averaged here before triangulation.
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
    """Estimate nominal spacing from the median nearest-neighbor distance."""
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    if coords.shape[0] < 2:
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


def _merge_near_duplicates(
    coords: np.ndarray,
    values: np.ndarray,
    merge_tol: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Merge points whose separation is within the Tinfour-style vertex tolerance."""
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    values = np.asarray(values, dtype=float).reshape(-1)
    n = coords.shape[0]
    if n < 2 or not np.isfinite(merge_tol) or merge_tol <= 0:
        return coords, values, 0, 0

    try:
        tree = cKDTree(coords)
    except Exception:
        return coords, values, 0, 0

    parent = np.arange(n, dtype=int)
    rank = np.zeros(n, dtype=int)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return int(i)

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri == rj:
            return
        if rank[ri] < rank[rj]:
            parent[ri] = rj
        elif rank[ri] > rank[rj]:
            parent[rj] = ri
        else:
            parent[rj] = ri
            rank[ri] += 1

    for i in range(n):
        neigh = tree.query_ball_point(coords[i], merge_tol)
        for j in neigh:
            j = int(j)
            if j > i:
                d2 = float(np.sum((coords[j] - coords[i]) ** 2))
                if d2 < merge_tol * merge_tol:
                    union(i, j)

    grouped: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        grouped.setdefault(root, []).append(i)

    if len(grouped) == n:
        return coords, values, 0, 0

    ordered_roots = sorted(grouped.keys(), key=lambda root: grouped[root][0])
    merged_coords = np.empty((len(ordered_roots), 2), dtype=float)
    merged_values = np.empty(len(ordered_roots), dtype=float)
    merged_groups = 0
    merged_points = 0

    for out_idx, root in enumerate(ordered_roots):
        idxs = np.asarray(grouped[root], dtype=int)
        merged_coords[out_idx] = coords[int(idxs[0])]
        vals = values[idxs]
        if vals.size == 0:
            merged_values[out_idx] = np.nan
        elif np.isnan(vals).any():
            merged_values[out_idx] = np.nan
        else:
            merged_values[out_idx] = float(np.mean(vals))
        if idxs.size > 1:
            merged_groups += 1
            merged_points += int(idxs.size - 1)

    return merged_coords, merged_values, merged_groups, merged_points


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


def _polygon_area(poly: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=float)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clean_polygon(poly: np.ndarray, tol: float) -> np.ndarray:
    poly = np.asarray(poly, dtype=float)
    if poly.size == 0:
        return np.empty((0, 2), dtype=float)
    poly = poly.reshape(-1, 2)

    cleaned = [poly[0]]
    for pt in poly[1:]:
        if np.linalg.norm(pt - cleaned[-1]) > tol:
            cleaned.append(pt)

    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= tol:
        cleaned.pop()

    out = np.asarray(cleaned, dtype=float)
    if out.shape[0] >= 3 and _polygon_area(out) < 0:
        out = out[::-1].copy()
    return out


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    if poly.size == 0:
        return np.empty((0, 2), dtype=float)
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    if _polygon_area(poly) < 0:
        poly = poly[::-1].copy()
    return poly


def _points_in_convex_polygon(
    points: np.ndarray,
    poly: np.ndarray,
    tol: float,
    *,
    strict: bool = False,
) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    poly = np.asarray(poly, dtype=float)
    if poly.size == 0:
        return np.zeros(points.shape[0], dtype=bool)
    poly = poly.reshape(-1, 2)
    if poly.shape[0] == 1:
        return np.linalg.norm(points - poly[0], axis=1) <= tol
    if poly.shape[0] == 2:
        a = poly[0]
        b = poly[1]
        ab = b - a
        ap = points - a
        cross = ab[0] * ap[:, 1] - ab[1] * ap[:, 0]
        dot = ap @ ab
        return (np.abs(cross) <= tol) & (dot >= -tol) & (dot <= float(np.dot(ab, ab)) + tol)

    inside = np.ones(points.shape[0], dtype=bool)
    for p0, p1 in zip(poly, np.roll(poly, -1, axis=0)):
        edge = p1 - p0
        cross = edge[0] * (points[:, 1] - p0[1]) - edge[1] * (points[:, 0] - p0[0])
        if strict:
            inside &= cross > tol
        else:
            inside &= cross >= -tol
        if not np.any(inside):
            return inside
    return inside


def _triangle_circumcenters(points: np.ndarray, simplices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = np.asarray(points, dtype=float)[np.asarray(simplices, dtype=int)]
    ax = tri[:, 0, 0]
    ay = tri[:, 0, 1]
    bx = tri[:, 1, 0]
    by = tri[:, 1, 1]
    cx = tri[:, 2, 0]
    cy = tri[:, 2, 1]

    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    with np.errstate(divide="ignore", invalid="ignore"):
        ux = (
            (ax * ax + ay * ay) * (by - cy)
            + (bx * bx + by * by) * (cy - ay)
            + (cx * cx + cy * cy) * (ay - by)
        ) / d
        uy = (
            (ax * ax + ay * ay) * (cx - bx)
            + (bx * bx + by * by) * (ax - cx)
            + (cx * cx + cy * cy) * (bx - ax)
        ) / d

    centers = np.column_stack([ux, uy])
    radius2 = np.sum((tri[:, 0, :] - centers) ** 2, axis=1)
    bad = ~np.isfinite(centers).all(axis=1) | ~np.isfinite(radius2)
    if np.any(bad):
        centers[bad] = np.nan
        radius2[bad] = np.nan
    return centers, radius2


class NaturalNeighborExactInterpolator:
    """Exact 2D Sibson natural-neighbor interpolator.

    The query flow mirrors Tinfour's natural-neighbor implementation:
    1. locate the Bowyer-Watson cavity by circumcircle tests,
    2. extract the boundary cycle of natural neighbors,
    3. evaluate exact Sibson weights using the wXY - wThiessen area formula.

    Co-location snapping and point merging follow Tinfour's vertex-tolerance
    philosophy rather than an epsilon-only equality test.
    """

    def __init__(
        self,
        x,
        y,
        z,
        *,
        nan_policy: str = "strict",
        diagnostics: bool = False,
        backend_options: Optional[dict[str, Any]] = None,
    ):
        self.nan_policy = _normalize_nan_policy(nan_policy)
        self.backend_options = dict(backend_options or {})
        self.diagnostics = NaturalNeighborExactDiagnostics(
            nan_policy=self.nan_policy,
            diagnostics_requested=bool(diagnostics),
        )

        self._coords: Optional[np.ndarray] = None
        self._values: Optional[np.ndarray] = None
        self._tree: Any = None
        self._tri: Any = None
        self._hull_polygon: Optional[np.ndarray] = None
        self._neighbor_indptr: Optional[np.ndarray] = None
        self._neighbor_indices: Optional[np.ndarray] = None
        self._simplex_neighbors: Optional[np.ndarray] = None
        self._circumcenters: Optional[np.ndarray] = None
        self._circumradius2: Optional[np.ndarray] = None
        self._vertex_to_simplices: list[np.ndarray] = []
        self._ordered_vertex_simplices: list[np.ndarray] = []

        self._vertex_tol: float = 0.0
        self._boundary_tol: float = 0.0
        self._exact_tol: float = 0.0
        self._area_tol: float = 0.0
        self._circumcircle_tol: float = 0.0

        self._build_start = time.perf_counter()
        self._build(x, y, z)
        self.diagnostics.build_seconds = float(time.perf_counter() - self._build_start)

    @staticmethod
    def _normalize_nan_policy(nan_policy: str) -> str:
        return _normalize_nan_policy(nan_policy)

    def _build(self, x, y, z) -> None:
        if Delaunay is None or ConvexHull is None or cKDTree is None:
            raise ImportError(
                "exact natural_neighbor interpolation requires SciPy (scipy.spatial)."
            )

        x = _as_1d_float(x, name="x")
        y = _as_1d_float(y, name="y")
        z = _as_1d_float(z, name="z")

        n = min(x.size, y.size, z.size)
        self.diagnostics.input_points = int(n)
        if n == 0:
            self.diagnostics.degenerate_input = True
            _warn("natural_neighbor: no input points were provided")
            return

        x = x[:n]
        y = y[:n]
        z = z[:n]

        finite_xy = np.isfinite(x) & np.isfinite(y)
        self.diagnostics.finite_xy_points = int(np.count_nonzero(finite_xy))
        if not np.any(finite_xy):
            self.diagnostics.degenerate_input = True
            _warn("natural_neighbor: no finite (x, y) coordinates are available")
            return

        coords = np.column_stack([x[finite_xy], y[finite_xy]])
        values = z[finite_xy]
        coords, values, exact_duplicate_groups, exact_duplicate_points = _dedupe_coordinates(coords, values)
        self.diagnostics.exact_duplicate_groups = int(exact_duplicate_groups)
        self.diagnostics.merged_points = int(exact_duplicate_points)

        scale = _effective_scale(coords)
        scale = max(scale, 1.0)
        nominal_spacing = self.backend_options.get("nominal_point_spacing", None)
        if nominal_spacing is None:
            nominal_spacing = _estimate_nominal_spacing(coords)
        try:
            nominal_spacing = float(nominal_spacing)
        except Exception:
            nominal_spacing = 1.0
        if not np.isfinite(nominal_spacing) or nominal_spacing <= 0:
            nominal_spacing = scale
        self.diagnostics.nominal_point_spacing = float(nominal_spacing)

        self._vertex_tol = float(
            self.backend_options.get(
                "vertex_tol",
                max(nominal_spacing / 1.0e5, 64.0 * np.finfo(float).eps * scale),
            )
        )
        if not np.isfinite(self._vertex_tol) or self._vertex_tol <= 0:
            self._vertex_tol = max(nominal_spacing / 1.0e5, 64.0 * np.finfo(float).eps * scale)
        self._exact_tol = float(
            self.backend_options.get(
                "geometry_tol",
                max(64.0 * np.finfo(float).eps * scale, 1e-12 * scale),
            )
        )
        if not np.isfinite(self._exact_tol) or self._exact_tol <= 0:
            self._exact_tol = max(64.0 * np.finfo(float).eps * scale, 1e-12 * scale)
        self._boundary_tol = float(
            self.backend_options.get(
                "boundary_tol",
                max(64.0 * np.finfo(float).eps * scale, self._exact_tol * 1.0e-2),
            )
        )
        if not np.isfinite(self._boundary_tol) or self._boundary_tol <= 0:
            self._boundary_tol = max(64.0 * np.finfo(float).eps * scale, self._exact_tol * 1.0e-2)
        self._area_tol = float(
            self.backend_options.get(
                "area_tol",
                max(64.0 * np.finfo(float).eps * scale * scale, 1e-12 * scale * scale),
            )
        )
        self._circumcircle_tol = float(
            self.backend_options.get(
                "circumcircle_tol",
                self.backend_options.get(
                    "halfspace_tol",
                    max(64.0 * np.finfo(float).eps * scale * scale, 1e-12 * scale * scale),
                ),
            )
        )

        coords, values, near_duplicate_groups, near_duplicate_points = _merge_near_duplicates(
            coords, values, self._vertex_tol
        )
        self.diagnostics.near_duplicate_groups = int(near_duplicate_groups)
        self.diagnostics.merged_points += int(near_duplicate_points)

        self._coords = coords
        self._values = values

        self.diagnostics.unique_points = int(coords.shape[0])
        finite_value_mask = np.isfinite(values)
        self.diagnostics.unique_finite_points = int(np.count_nonzero(finite_value_mask))
        self.diagnostics.all_nan_cores = bool(coords.size > 0 and not np.any(finite_value_mask))
        self.diagnostics.vertex_tolerance = float(self._vertex_tol)
        self.diagnostics.boundary_tolerance = float(self._boundary_tol)

        self._tree = cKDTree(coords)

        try:
            if coords.shape[0] >= 3:
                hull = ConvexHull(coords)
                hull_poly = coords[hull.vertices]
                hull_poly = _clean_polygon(hull_poly, self._exact_tol)
                if hull_poly.shape[0] >= 3 and abs(_polygon_area(hull_poly)) > self._area_tol:
                    self._hull_polygon = _ensure_ccw(hull_poly)
                else:
                    self._hull_polygon = None
                    self.diagnostics.degenerate_input = True
                    _warn("natural_neighbor: convex hull is degenerate")
            else:
                self._hull_polygon = None
                self.diagnostics.degenerate_input = True
        except QhullError as exc:
            self._hull_polygon = None
            self.diagnostics.degenerate_input = True
            _warn(f"natural_neighbor: convex hull construction failed: {exc}")

        try:
            if coords.shape[0] >= 3:
                self._tri = Delaunay(coords)
                self._neighbor_indptr, self._neighbor_indices = self._tri.vertex_neighbor_vertices
                self._simplex_neighbors = self._tri.neighbors
                self._circumcenters, self._circumradius2 = _triangle_circumcenters(
                    coords, self._tri.simplices
                )
                self._vertex_to_simplices = [[] for _ in range(coords.shape[0])]
                for simplex_idx, simplex in enumerate(self._tri.simplices):
                    for vertex_idx in simplex:
                        self._vertex_to_simplices[int(vertex_idx)].append(int(simplex_idx))
                self._vertex_to_simplices = [
                    np.asarray(indices, dtype=int) if indices else np.empty((0,), dtype=int)
                    for indices in self._vertex_to_simplices
                ]
                self._ordered_vertex_simplices = []
                for vertex_idx, simplex_ids in enumerate(self._vertex_to_simplices):
                    if simplex_ids.size == 0:
                        self._ordered_vertex_simplices.append(np.empty((0,), dtype=int))
                        continue
                    centers = self._circumcenters[simplex_ids]
                    if not np.isfinite(centers).all():
                        self._ordered_vertex_simplices.append(np.empty((0,), dtype=int))
                        self.diagnostics.degenerate_input = True
                        _warn("natural_neighbor: degenerate triangles produced invalid circumcenters")
                        continue
                    origin = coords[int(vertex_idx)]
                    angles = np.arctan2(centers[:, 1] - origin[1], centers[:, 0] - origin[0])
                    order = np.argsort(angles, kind="mergesort")
                    self._ordered_vertex_simplices.append(np.asarray(simplex_ids[order], dtype=int))
            else:
                self._tri = None
        except QhullError as exc:
            self._tri = None
            self._neighbor_indptr = None
            self._neighbor_indices = None
            self._simplex_neighbors = None
            self._circumcenters = None
            self._circumradius2 = None
            self._vertex_to_simplices = []
            self._ordered_vertex_simplices = []
            self.diagnostics.degenerate_input = True
            _warn(f"natural_neighbor: Delaunay triangulation failed: {exc}")

        if self._tri is None or self._hull_polygon is None:
            self.diagnostics.degenerate_input = True
            self.diagnostics.interpolator_ready = False
            return

        self.diagnostics.interpolator_ready = True

    def __call__(self, X, Y):
        return self.evaluate(X, Y)

    def _point_in_hull(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        if self._hull_polygon is None:
            return np.zeros(pts.shape[0], dtype=bool)
        return _points_in_convex_polygon(pts, self._hull_polygon, self._boundary_tol, strict=True)

    def _cavity_boundary_cycle(self, q: np.ndarray, cavity_idx: np.ndarray) -> Optional[np.ndarray]:
        if self._tri is None or self._simplex_neighbors is None or self._coords is None:
            return None

        cavity_set = {int(s) for s in np.asarray(cavity_idx, dtype=int).reshape(-1)}
        boundary_edges: list[tuple[int, int]] = []
        seen_edges: set[tuple[int, int]] = set()

        for simplex_idx in cavity_set:
            simplex = np.asarray(self._tri.simplices[int(simplex_idx)], dtype=int)
            neighbors = np.asarray(self._simplex_neighbors[int(simplex_idx)], dtype=int)
            for local_vertex in range(3):
                nb = int(neighbors[local_vertex])
                if nb >= 0 and nb in cavity_set:
                    continue
                edge = tuple(sorted(tuple(int(v) for v in np.delete(simplex, local_vertex))))
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    boundary_edges.append(edge)

        if len(boundary_edges) < 3:
            return None

        adjacency: dict[int, list[int]] = {}
        for u, v in boundary_edges:
            adjacency.setdefault(u, []).append(v)
            adjacency.setdefault(v, []).append(u)

        if any(len(neigh) != 2 for neigh in adjacency.values()):
            return None

        def _angle(v_idx: int) -> float:
            p = self._coords[int(v_idx)]
            return math.atan2(float(p[1] - q[1]), float(p[0] - q[0]))

        start = min(adjacency, key=_angle)
        cycle = [int(start)]
        prev: Optional[int] = None
        curr = int(start)

        guard = 0
        limit = len(adjacency) + 2
        while True:
            neigh = adjacency[curr]
            if prev is None:
                next_v = int(neigh[0])
            else:
                next_v = int(neigh[0] if neigh[0] != prev else neigh[1])
            if next_v == start:
                break
            cycle.append(next_v)
            prev, curr = curr, next_v
            guard += 1
            if guard > limit:
                return None

        cycle_arr = np.asarray(cycle, dtype=int)
        if cycle_arr.size < 3:
            return None

        poly = self._coords[cycle_arr]
        if abs(_polygon_area(poly)) <= self._area_tol:
            return None
        if _polygon_area(poly) < 0:
            cycle_arr = cycle_arr[::-1].copy()
        return cycle_arr

    def _find_simplex(self, q: np.ndarray) -> int:
        if self._tri is None:
            return -1
        q = np.asarray(q, dtype=float).reshape(1, 2)
        simplex = int(self._tri.find_simplex(q, tol=self._exact_tol)[0])
        if simplex < 0:
            simplex = int(self._tri.find_simplex(q, tol=self._exact_tol, bruteforce=True)[0])
        return simplex

    def _circumcircle_contains(self, simplex_idx: int, q: np.ndarray) -> bool:
        if self._circumcenters is None or self._circumradius2 is None:
            return False
        center = self._circumcenters[simplex_idx]
        radius2 = self._circumradius2[simplex_idx]
        if not np.isfinite(radius2) or not np.all(np.isfinite(center)):
            return False
        dist2 = float(np.sum((np.asarray(q, dtype=float) - center) ** 2))
        return dist2 <= float(radius2) + self._circumcircle_tol

    def _natural_neighbor_sites(
        self,
        q: np.ndarray,
        *,
        start_simplex: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        simplex = self._find_simplex(q) if start_simplex is None else int(start_simplex)
        if simplex < 0 or self._simplex_neighbors is None or self._tri is None:
            return np.empty((0,), dtype=int), np.empty((0,), dtype=int)

        stack = [int(simplex)]
        visited: set[int] = set()
        cavity: set[int] = set()

        while stack:
            s = int(stack.pop())
            if s in visited:
                continue
            visited.add(s)
            if self._circumcircle_contains(s, q):
                cavity.add(s)
                for nb in self._simplex_neighbors[s]:
                    if nb >= 0 and nb not in visited:
                        stack.append(int(nb))

        if not cavity:
            return np.empty((0,), dtype=int), np.empty((0,), dtype=int)

        cavity_idx = np.fromiter(sorted(cavity), dtype=int)
        boundary_cycle = self._cavity_boundary_cycle(q, cavity_idx)
        if boundary_cycle is None:
            return np.empty((0,), dtype=int), cavity_idx
        return boundary_cycle.astype(int, copy=False), cavity_idx

    @staticmethod
    def _cross2d(u: np.ndarray, v: np.ndarray) -> float:
        return float(u[0] * v[1] - v[0] * u[1])

    def _circumcenter_relative(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        ax = float(a[0])
        ay = float(a[1])
        bx = float(b[0])
        by = float(b[1])
        cx = float(c[0])
        cy = float(c[1])

        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if not np.isfinite(d) or abs(d) <= self._area_tol:
            return np.array([np.nan, np.nan], dtype=float)
        ux = (
            (ax * ax + ay * ay) * (by - cy)
            + (bx * bx + by * by) * (cy - ay)
            + (cx * cx + cy * cy) * (ay - by)
        ) / d
        uy = (
            (ax * ax + ay * ay) * (cx - bx)
            + (bx * bx + by * by) * (ax - cx)
            + (cx * cx + cy * cy) * (bx - ax)
        ) / d
        if not np.isfinite(ux) or not np.isfinite(uy):
            return np.array([np.nan, np.nan], dtype=float)
        return np.array([ux, uy], dtype=float)

    def _simplex_has_edge(self, simplex_idx: int, a: int, b: int) -> bool:
        if self._tri is None:
            return False
        simplex = np.asarray(self._tri.simplices[int(simplex_idx)], dtype=int)
        return bool(np.any(simplex == int(a)) and np.any(simplex == int(b)))

    def _incident_simplex_with_edge(
        self,
        vertex_idx: int,
        other_idx: int,
        cavity_set: set[int],
        q: np.ndarray,
    ) -> Optional[int]:
        if not self._vertex_to_simplices:
            return None
        for simplex_idx in self._vertex_to_simplices[int(vertex_idx)]:
            simplex_id = int(simplex_idx)
            if simplex_id not in cavity_set:
                continue
            if not self._simplex_has_edge(simplex_id, vertex_idx, other_idx):
                continue
            if self._circumcircle_contains(simplex_id, q):
                return simplex_id
        return None

    def _ordered_cavity_simplices_for_vertex(
        self,
        vertex_idx: int,
        left_idx: int,
        right_idx: int,
        cavity_set: set[int],
        q: np.ndarray,
    ) -> Optional[list[int]]:
        if not self._ordered_vertex_simplices:
            return None
        ordered = self._ordered_vertex_simplices[int(vertex_idx)]
        if ordered.size == 0:
            return None

        start = self._incident_simplex_with_edge(vertex_idx, left_idx, cavity_set, q)
        end = self._incident_simplex_with_edge(vertex_idx, right_idx, cavity_set, q)
        if start is None or end is None:
            return None

        start_pos_arr = np.flatnonzero(ordered == int(start))
        end_pos_arr = np.flatnonzero(ordered == int(end))
        if start_pos_arr.size == 0 or end_pos_arr.size == 0:
            return None
        start_pos = int(start_pos_arr[0])
        end_pos = int(end_pos_arr[0])

        def _walk(step: int) -> Optional[list[int]]:
            chain: list[int] = []
            pos = start_pos
            limit = ordered.size + 1
            while True:
                simplex_id = int(ordered[pos])
                if simplex_id not in cavity_set:
                    return None
                if not self._circumcircle_contains(simplex_id, q):
                    return None
                chain.append(simplex_id)
                if pos == end_pos:
                    return chain if chain else None
                pos = (pos + step) % ordered.size
                limit -= 1
                if limit <= 0:
                    return None

        # The cavity block can wrap around the angle discontinuity in the
        # per-vertex circular ordering. Try both traversal directions and keep
        # whichever one stays fully inside the cavity.
        chain = _walk(+1)
        if chain is not None:
            return chain
        chain = _walk(-1)
        if chain is not None:
            return chain
        return None

    def _query_point(
        self,
        q: np.ndarray,
        *,
        simplex: Optional[int] = None,
        skip_exact_hit: bool = False,
        skip_hull_check: bool = False,
    ) -> float:
        assert self._coords is not None and self._values is not None

        q = np.asarray(q, dtype=float).reshape(2)

        if not skip_exact_hit:
            dist, idx = self._tree.query(q, k=1)
            if np.isfinite(dist) and dist < self._vertex_tol:
                return float(self._values[int(idx)])

        if self._hull_polygon is None or self._tri is None:
            self.diagnostics.degenerate_queries += 1
            return np.nan

        if (not skip_hull_check) and (not self._point_in_hull(q.reshape(1, 2))[0]):
            return np.nan

        neighbors, cavity_idx = self._natural_neighbor_sites(q, start_simplex=simplex)
        self.diagnostics.cavity_triangles += int(cavity_idx.size)
        if neighbors.size == 0:
            self.diagnostics.degenerate_queries += 1
            return np.nan

        cavity_set = {int(i) for i in np.asarray(cavity_idx, dtype=int).reshape(-1)}
        if neighbors.size < 3:
            self.diagnostics.degenerate_queries += 1
            return np.nan

        neighbor_values = self._values[np.asarray(neighbors, dtype=int)]
        if np.isnan(neighbor_values).any():
            self.diagnostics.masked_by_nan += 1
            return np.nan

        n_edge = int(neighbors.size)
        weights = np.zeros(n_edge, dtype=float)
        w_sum = 0.0
        q0 = q.reshape(2)

        for i0 in range(n_edge):
            i_prev = (i0 - 1) % n_edge
            i1 = i0
            i_next = (i0 + 1) % n_edge

            a_idx = int(neighbors[i_prev])
            b_idx = int(neighbors[i1])
            c_idx = int(neighbors[i_next])

            a = self._coords[a_idx] - q0
            b = self._coords[b_idx] - q0
            c = self._coords[c_idx] - q0
            if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(c).all():
                self.diagnostics.degenerate_queries += 1
                return np.nan

            mid_ab = 0.5 * (a + b)
            mid_bc = 0.5 * (b + c)
            c0 = self._circumcenter_relative(a, b, np.zeros(2, dtype=float))
            c1 = self._circumcenter_relative(b, c, np.zeros(2, dtype=float))
            if not np.isfinite(c0).all() or not np.isfinite(c1).all():
                self.diagnostics.degenerate_queries += 1
                return np.nan

            cavity_chain = self._ordered_cavity_simplices_for_vertex(
                b_idx, a_idx, c_idx, cavity_set, q0
            )
            if cavity_chain is None:
                self.diagnostics.degenerate_queries += 1
                return np.nan

            if self._circumcenters is None:
                self.diagnostics.degenerate_queries += 1
                return np.nan

            c3 = self._circumcenters[int(cavity_chain[0])] - q0
            if not np.isfinite(c3).all():
                self.diagnostics.degenerate_queries += 1
                return np.nan

            w_xy = self._cross2d(mid_ab, c0) + self._cross2d(c0, c1) + self._cross2d(c1, mid_bc)
            w_thiessen = self._cross2d(mid_ab, c3)
            prev_center = c3
            for simplex_id in cavity_chain[1:]:
                curr_center = self._circumcenters[int(simplex_id)] - q0
                if not np.isfinite(curr_center).all():
                    self.diagnostics.degenerate_queries += 1
                    return np.nan
                w_thiessen += self._cross2d(prev_center, curr_center)
                prev_center = curr_center
            w_thiessen += self._cross2d(prev_center, mid_bc)

            w_delta = w_xy - w_thiessen
            if not np.isfinite(w_delta):
                self.diagnostics.degenerate_queries += 1
                return np.nan
            weights[i1] = w_delta
            w_sum += w_delta

        if not np.isfinite(w_sum) or abs(w_sum) <= self._area_tol:
            self.diagnostics.degenerate_queries += 1
            return np.nan

        weights /= w_sum
        self.diagnostics.area_of_embedded_polygon = float(w_sum / 2.0)

        x_sum = 0.0
        y_sum = 0.0
        for weight, nb in zip(weights, neighbors):
            v = self._coords[int(nb)]
            x_sum += float(weight) * (float(v[0]) - q0[0])
            y_sum += float(weight) * (float(v[1]) - q0[1])
        self.diagnostics.barycentric_coordinate_deviation = float(
            math.sqrt(x_sum * x_sum + y_sum * y_sum)
        )

        weighted = float(np.dot(weights, neighbor_values))
        return weighted

    def evaluate(self, X, Y):
        X = _as_grid_float(X, name="X")
        Y = _as_grid_float(Y, name="Y")
        if X.shape != Y.shape:
            raise ValueError("X and Y must have the same shape")

        pts = np.column_stack([X.ravel(), Y.ravel()])
        out = np.full(pts.shape[0], np.nan, dtype=float)
        self.diagnostics.query_points = int(pts.shape[0])
        self.diagnostics.inside_hull = 0
        self.diagnostics.outside_hull = 0
        self.diagnostics.exact_hits = 0
        self.diagnostics.masked_by_nan = 0
        self.diagnostics.degenerate_queries = 0
        self.diagnostics.cavity_triangles = 0

        if self._coords is None or self._values is None or self._coords.size == 0:
            self.diagnostics.eval_seconds = 0.0
            return out.reshape(X.shape)

        eval_start = time.perf_counter()
        if self._tri is None or self._hull_polygon is None:
            self.diagnostics.degenerate_queries = int(pts.shape[0])
            self.diagnostics.eval_seconds = float(time.perf_counter() - eval_start)
            return out.reshape(X.shape)

        dist, idx = self._tree.query(pts, k=1)
        exact_mask = np.isfinite(dist) & (dist < self._vertex_tol)
        if np.any(exact_mask):
            out[exact_mask] = self._values[np.asarray(idx[exact_mask], dtype=int)]
        self.diagnostics.exact_hits = int(np.count_nonzero(exact_mask))

        remaining = np.flatnonzero(~exact_mask)
        if remaining.size == 0:
            self.diagnostics.eval_seconds = float(time.perf_counter() - eval_start)
            return out.reshape(X.shape)

        inside_mask = self._point_in_hull(pts[remaining])
        inside_count = int(np.count_nonzero(inside_mask))
        self.diagnostics.inside_hull = int(self.diagnostics.exact_hits + inside_count)
        self.diagnostics.outside_hull = int(remaining.size - inside_count)

        inside_idx = remaining[inside_mask]
        if inside_idx.size == 0:
            self.diagnostics.eval_seconds = float(time.perf_counter() - eval_start)
            return out.reshape(X.shape)

        inside_pts = pts[inside_idx]
        simplex = self._tri.find_simplex(inside_pts, tol=self._exact_tol)
        bad_simplex = simplex < 0
        if np.any(bad_simplex):
            brute = self._tri.find_simplex(
                inside_pts[bad_simplex],
                tol=self._exact_tol,
                bruteforce=True,
            )
            simplex = np.asarray(simplex, dtype=int)
            simplex[bad_simplex] = np.asarray(brute, dtype=int)
        else:
            simplex = np.asarray(simplex, dtype=int)

        for flat_idx, simplex_idx in zip(inside_idx, simplex, strict=False):
            val = self._query_point(
                pts[int(flat_idx)],
                simplex=int(simplex_idx),
                skip_exact_hit=True,
                skip_hull_check=True,
            )
            out[int(flat_idx)] = val

        self.diagnostics.eval_seconds = float(time.perf_counter() - eval_start)
        return out.reshape(X.shape)


def natural_neighbor_exact_interpolate(
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
    """Exact Sibson natural-neighbor interpolation in 2D."""
    interp = NaturalNeighborExactInterpolator(
        x,
        y,
        z,
        nan_policy=nan_policy,
        diagnostics=diagnostics,
        backend_options=backend_options,
    )
    result = interp.evaluate(X, Y)
    natural_neighbor_exact_interpolate.last_diagnostics = interp.diagnostics
    return np.asarray(result, dtype=float)


natural_neighbor_exact_interpolate.last_diagnostics = None  # type: ignore[attr-defined]


def register_exact_backend(register_backend: Callable[..., None]) -> None:
    """Register the exact backend into the shared interpolation registry."""
    register_backend("natural_neighbor", natural_neighbor_exact_interpolate, overwrite=True)
    register_backend("natural_neighbor_exact", natural_neighbor_exact_interpolate, overwrite=True)

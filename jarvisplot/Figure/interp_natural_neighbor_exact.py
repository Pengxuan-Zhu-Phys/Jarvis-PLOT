#!/usr/bin/env python3
"""Exact 2D Sibson natural-neighbor interpolation backend.

The interpolant is built from Voronoi geometry:
- natural neighbors are discovered through Delaunay cavity traversal,
- query cells are formed with exact half-plane clipping,
- weights are stolen-area fractions from clipped Voronoi polygons,
- NaN-valued cores propagate strictly and outside-hull queries return NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    cavity_triangles: int = 0
    site_cells_built: int = 0
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


def _dedupe_coordinates(coords: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse exact duplicate coordinates while preserving strict NaN semantics."""
    if coords.size == 0:
        return coords, values

    uniq, inverse = np.unique(coords, axis=0, return_inverse=True)
    out = np.empty(uniq.shape[0], dtype=float)

    for idx in range(uniq.shape[0]):
        group = values[inverse == idx]
        if group.size == 0:
            out[idx] = np.nan
            continue
        if np.isnan(group).any():
            out[idx] = np.nan
            continue
        first = float(group[0])
        if group.size > 1 and not np.allclose(group, first, rtol=0.0, atol=1e-12):
            _warn(
                "natural_neighbor: duplicate coordinates with conflicting finite z values "
                "were collapsed by keeping the first value"
            )
        out[idx] = first

    return uniq, out


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


def _polygon_halfspace_from_edge(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the left-of-edge half-space for a CCW polygon edge."""
    v = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    a = np.array([v[1], -v[0]], dtype=float)
    b = float(v[1] * p0[0] - v[0] * p0[1])
    return a, b


def _bisector_halfspace(center: np.ndarray, neighbor: np.ndarray) -> tuple[np.ndarray, float]:
    """Half-space of points closer to `center` than `neighbor`."""
    center = np.asarray(center, dtype=float)
    neighbor = np.asarray(neighbor, dtype=float)
    a = neighbor - center
    b = 0.5 * (float(np.dot(neighbor, neighbor)) - float(np.dot(center, center)))
    return a, b


def _clip_polygon_halfspace(poly: np.ndarray, a: np.ndarray, b: float, tol: float) -> np.ndarray:
    """Sutherland-Hodgman clipping against the half-space a·x <= b."""
    poly = np.asarray(poly, dtype=float)
    if poly.size == 0:
        return np.empty((0, 2), dtype=float)
    poly = poly.reshape(-1, 2)
    if poly.shape[0] == 0:
        return np.empty((0, 2), dtype=float)

    values = poly @ a - b
    inside = values <= tol
    if not np.any(inside):
        return np.empty((0, 2), dtype=float)

    result: list[np.ndarray] = []

    def _intersect(p0: np.ndarray, p1: np.ndarray, f0: float, f1: float) -> np.ndarray:
        denom = f0 - f1
        if abs(denom) <= tol:
            return np.asarray(p0, dtype=float)
        t = f0 / denom
        return np.asarray(p0, dtype=float) + t * (np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float))

    prev = poly[-1]
    prev_f = float(values[-1])
    prev_in = prev_f <= tol

    for curr, curr_f in zip(poly, values):
        curr_f = float(curr_f)
        curr_in = curr_f <= tol
        if curr_in:
            if not prev_in:
                result.append(_intersect(prev, curr, prev_f, curr_f))
            result.append(np.asarray(curr, dtype=float))
        elif prev_in:
            result.append(_intersect(prev, curr, prev_f, curr_f))
        prev = curr
        prev_f = curr_f
        prev_in = curr_in

    if not result:
        return np.empty((0, 2), dtype=float)
    return _clean_polygon(np.asarray(result, dtype=float), tol)


def _polygon_intersection_convex(subject: np.ndarray, clipper: np.ndarray, tol: float) -> np.ndarray:
    subject = np.asarray(subject, dtype=float)
    clipper = np.asarray(clipper, dtype=float)
    if subject.size == 0 or clipper.size == 0:
        return np.empty((0, 2), dtype=float)
    subject = subject.reshape(-1, 2)
    clipper = clipper.reshape(-1, 2)
    if subject.shape[0] < 3 or clipper.shape[0] < 3:
        return np.empty((0, 2), dtype=float)

    poly = subject
    for p0, p1 in zip(clipper, np.roll(clipper, -1, axis=0)):
        a, b = _polygon_halfspace_from_edge(p0, p1)
        poly = _clip_polygon_halfspace(poly, a, b, tol)
        if poly.size == 0:
            return np.empty((0, 2), dtype=float)
    return _clean_polygon(poly, tol)


def _points_in_convex_polygon(points: np.ndarray, poly: np.ndarray, tol: float) -> np.ndarray:
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

    The local natural-neighbor cavity is found from Delaunay triangle
    circumcircle tests. The actual weights come from exact Voronoi geometry:
    we build the query cell with half-plane clipping and take stolen-area
    fractions against the original clipped Voronoi cells.
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
        self._site_cell_cache: dict[int, np.ndarray] = {}

        self._exact_tol: float = 0.0
        self._halfspace_tol: float = 0.0
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
        coords, values = _dedupe_coordinates(coords, values)

        self._coords = coords
        self._values = values

        self.diagnostics.unique_points = int(coords.shape[0])
        finite_value_mask = np.isfinite(values)
        self.diagnostics.unique_finite_points = int(np.count_nonzero(finite_value_mask))
        self.diagnostics.all_nan_cores = bool(coords.size > 0 and not np.any(finite_value_mask))

        scale = _effective_scale(coords)
        scale = max(scale, 1.0)
        self._exact_tol = float(
            self.backend_options.get(
                "exact_tol",
                max(64.0 * np.finfo(float).eps * scale, 1e-12 * scale),
            )
        )
        self._halfspace_tol = float(
            self.backend_options.get(
                "halfspace_tol",
                max(64.0 * np.finfo(float).eps * scale * scale, 1e-12 * scale * scale),
            )
        )
        self._area_tol = float(
            self.backend_options.get(
                "area_tol",
                max(64.0 * np.finfo(float).eps * scale * scale, 1e-12 * scale * scale),
            )
        )
        self._circumcircle_tol = float(
            self.backend_options.get("circumcircle_tol", self._halfspace_tol)
        )

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
            else:
                self._tri = None
        except QhullError as exc:
            self._tri = None
            self._neighbor_indptr = None
            self._neighbor_indices = None
            self._simplex_neighbors = None
            self._circumcenters = None
            self._circumradius2 = None
            self.diagnostics.degenerate_input = True
            _warn(f"natural_neighbor: Delaunay triangulation failed: {exc}")

        if self._tri is None or self._hull_polygon is None:
            self.diagnostics.degenerate_input = True
            if self._coords is not None and self._coords.size:
                self.diagnostics.interpolator_ready = True
            return

        self.diagnostics.interpolator_ready = True

    def __call__(self, X, Y):
        return self.evaluate(X, Y)

    def _point_in_hull(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        if self._hull_polygon is None:
            return np.zeros(pts.shape[0], dtype=bool)
        return _points_in_convex_polygon(pts, self._hull_polygon, self._exact_tol)

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

    def _natural_neighbor_sites(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        simplex = self._find_simplex(q)
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
        neighbors = np.unique(self._tri.simplices[cavity_idx].ravel())
        return neighbors.astype(int, copy=False), cavity_idx

    def _site_cell(self, site_idx: int) -> np.ndarray:
        cached = self._site_cell_cache.get(int(site_idx))
        if cached is not None:
            return cached

        if self._hull_polygon is None or self._coords is None or self._neighbor_indptr is None:
            poly = np.empty((0, 2), dtype=float)
            self._site_cell_cache[int(site_idx)] = poly
            return poly

        poly = np.asarray(self._hull_polygon, dtype=float).copy()
        center = self._coords[int(site_idx)]
        start = int(self._neighbor_indptr[int(site_idx)])
        stop = int(self._neighbor_indptr[int(site_idx) + 1])
        neighbors = np.asarray(self._neighbor_indices[start:stop], dtype=int)

        for nb in neighbors:
            if nb < 0 or nb == site_idx:
                continue
            a, b = _bisector_halfspace(center, self._coords[int(nb)])
            poly = _clip_polygon_halfspace(poly, a, b, self._halfspace_tol)
            if poly.size == 0:
                break

        poly = _clean_polygon(poly, self._exact_tol)
        if poly.shape[0] >= 3 and abs(_polygon_area(poly)) > self._area_tol:
            poly = _ensure_ccw(poly)
        else:
            poly = np.empty((0, 2), dtype=float)

        if poly.size == 0:
            # Fallback to all sites for rare degeneracies.
            poly = np.asarray(self._hull_polygon, dtype=float).copy()
            for nb in range(self._coords.shape[0]):
                if nb == site_idx:
                    continue
                a, b = _bisector_halfspace(center, self._coords[int(nb)])
                poly = _clip_polygon_halfspace(poly, a, b, self._halfspace_tol)
                if poly.size == 0:
                    break
            poly = _clean_polygon(poly, self._exact_tol)
            if poly.shape[0] >= 3 and abs(_polygon_area(poly)) > self._area_tol:
                poly = _ensure_ccw(poly)
            else:
                poly = np.empty((0, 2), dtype=float)

        self._site_cell_cache[int(site_idx)] = poly
        self.diagnostics.site_cells_built += 1
        return poly

    def _query_point(self, q: np.ndarray) -> float:
        assert self._coords is not None and self._values is not None

        q = np.asarray(q, dtype=float).reshape(2)

        dist, idx = self._tree.query(q, k=1)
        if np.isfinite(dist) and dist <= self._exact_tol:
            return float(self._values[int(idx)])

        if self._hull_polygon is None or self._tri is None:
            self.diagnostics.degenerate_queries += 1
            return np.nan

        if not self._point_in_hull(q.reshape(1, 2))[0]:
            return np.nan

        neighbors, cavity_idx = self._natural_neighbor_sites(q)
        self.diagnostics.cavity_triangles += int(cavity_idx.size)
        if neighbors.size == 0:
            self.diagnostics.degenerate_queries += 1
            return np.nan

        q_cell = np.asarray(self._hull_polygon, dtype=float).copy()
        for nb in neighbors:
            a, b = _bisector_halfspace(q, self._coords[int(nb)])
            q_cell = _clip_polygon_halfspace(q_cell, a, b, self._halfspace_tol)
            if q_cell.size == 0:
                break

        q_cell = _clean_polygon(q_cell, self._exact_tol)
        if q_cell.shape[0] < 3 or abs(_polygon_area(q_cell)) <= self._area_tol:
            # Rare fallback: clip against all sites, which is exact but slower.
            q_cell = np.asarray(self._hull_polygon, dtype=float).copy()
            for nb in range(self._coords.shape[0]):
                a, b = _bisector_halfspace(q, self._coords[int(nb)])
                q_cell = _clip_polygon_halfspace(q_cell, a, b, self._halfspace_tol)
                if q_cell.size == 0:
                    break
            q_cell = _clean_polygon(q_cell, self._exact_tol)

        if q_cell.shape[0] < 3 or abs(_polygon_area(q_cell)) <= self._area_tol:
            self.diagnostics.degenerate_queries += 1
            return np.nan

        q_area = abs(_polygon_area(q_cell))
        if not np.isfinite(q_area) or q_area <= self._area_tol:
            self.diagnostics.degenerate_queries += 1
            return np.nan

        def _accumulate(site_ids, q_poly: np.ndarray) -> tuple[float, float, bool, int]:
            weighted = 0.0
            area_sum = 0.0
            finite_count = 0
            saw_nan = False

            for nb in site_ids:
                site_val = float(self._values[int(nb)])
                site_poly = self._site_cell(int(nb))
                if site_poly.size == 0:
                    continue
                inter = _polygon_intersection_convex(q_poly, site_poly, self._halfspace_tol)
                area = abs(_polygon_area(inter))
                if not np.isfinite(area) or area <= self._area_tol:
                    continue
                if np.isnan(site_val):
                    saw_nan = True
                    break
                weighted += area * site_val
                area_sum += area
                finite_count += 1

            return weighted, area_sum, saw_nan, finite_count

        weighted, area_sum, saw_nan, finite_count = _accumulate(neighbors, q_cell)
        if saw_nan:
            self.diagnostics.masked_by_nan += 1
            return np.nan

        area_mismatch = abs(area_sum - q_area) > max(self._area_tol, 1e-10 * q_area)
        if finite_count == 0 or area_mismatch:
            q_cell = np.asarray(self._hull_polygon, dtype=float).copy()
            for nb in range(self._coords.shape[0]):
                a, b = _bisector_halfspace(q, self._coords[int(nb)])
                q_cell = _clip_polygon_halfspace(q_cell, a, b, self._halfspace_tol)
                if q_cell.size == 0:
                    break
            q_cell = _clean_polygon(q_cell, self._exact_tol)
            if q_cell.shape[0] < 3 or abs(_polygon_area(q_cell)) <= self._area_tol:
                self.diagnostics.degenerate_queries += 1
                return np.nan
            q_area = abs(_polygon_area(q_cell))
            if not np.isfinite(q_area) or q_area <= self._area_tol:
                self.diagnostics.degenerate_queries += 1
                return np.nan
            weighted, area_sum, saw_nan, finite_count = _accumulate(range(self._coords.shape[0]), q_cell)
            if saw_nan:
                self.diagnostics.masked_by_nan += 1
                return np.nan
            if finite_count == 0 or abs(area_sum - q_area) > max(self._area_tol, 1e-10 * q_area):
                self.diagnostics.degenerate_queries += 1
                return np.nan

        return float(weighted / q_area)

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
        dist, idx = self._tree.query(pts, k=1)
        exact_mask = np.isfinite(dist) & (dist <= self._exact_tol)
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

        for flat_idx in inside_idx:
            val = self._query_point(pts[int(flat_idx)])
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

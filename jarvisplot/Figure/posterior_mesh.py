from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class VoronoiCells:
    areas: np.ndarray
    polygons: Optional[list[Any]] = None
    method: str = "clipped_voronoi"


@dataclass
class PosteriorMesh:
    """Internal conservative posterior support mesh.

    This is intentionally not a public transform output. It gathers the mesh
    state needed for diagnostics and future adaptive support-node refinement
    while preserving the existing minimal support-table contract.
    """

    generators: np.ndarray
    ownership: np.ndarray
    masses: np.ndarray
    areas: np.ndarray
    densities: np.ndarray
    sample_counts: np.ndarray
    covariance: np.ndarray
    covariance_eigenvalues: np.ndarray
    covariance_eigenvectors: np.ndarray
    principal_direction: np.ndarray
    local_spread: np.ndarray
    anisotropy_ratio: np.ndarray
    density_gradient_proxy: np.ndarray
    curvature_proxy: np.ndarray
    centroid_proposal: np.ndarray
    centroid_drift: np.ndarray
    centroid_drift_magnitude: np.ndarray
    mass_stress: np.ndarray
    geometry_stress: np.ndarray
    stress_gap: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)
    polygons: Optional[list[Any]] = None

    def support_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.generators[:, 0], self.generators[:, 1], self.masses

    def debug_payload(self) -> dict[str, Any]:
        """Return arrays useful for internal mesh visualization/debug views."""
        return {
            "generators": self.generators.copy(),
            "ownership": self.ownership.copy(),
            "masses": self.masses.copy(),
            "areas": self.areas.copy(),
            "densities": self.densities.copy(),
            "sample_counts": self.sample_counts.copy(),
            "covariance": self.covariance.copy(),
            "covariance_eigenvalues": self.covariance_eigenvalues.copy(),
            "covariance_eigenvectors": self.covariance_eigenvectors.copy(),
            "principal_direction": self.principal_direction.copy(),
            "local_spread": self.local_spread.copy(),
            "anisotropy_ratio": self.anisotropy_ratio.copy(),
            "density_gradient_proxy": self.density_gradient_proxy.copy(),
            "curvature_proxy": self.curvature_proxy.copy(),
            "centroid_proposal": self.centroid_proposal.copy(),
            "centroid_drift": self.centroid_drift.copy(),
            "centroid_drift_magnitude": self.centroid_drift_magnitude.copy(),
            "mass_stress": self.mass_stress.copy(),
            "geometry_stress": self.geometry_stress.copy(),
            "stress_gap": self.stress_gap.copy(),
            "stress": self.mass_stress.copy(),
            "polygons": self.polygons,
            "diagnostics": dict(self.diagnostics),
        }

    def refine(
        self,
        samples: np.ndarray,
        weights: np.ndarray,
        *,
        xlim: Tuple[float, float] = (0.0, 1.0),
        ylim: Tuple[float, float] = (0.0, 1.0),
        refinement: Optional[dict[str, Any]] = None,
        include_polygons: bool = False,
        strict_geometry: bool = False,
        geometry_space: str = "normalized",
    ) -> "PosteriorMesh":
        return refine_posterior_mesh(
            self,
            samples,
            weights,
            xlim=xlim,
            ylim=ylim,
            refinement=refinement,
            include_polygons=include_polygons,
            strict_geometry=strict_geometry,
            geometry_space=geometry_space,
        )


def assign_nearest_ownership(samples: np.ndarray, generators: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=float).reshape(-1, 2)
    generators = np.asarray(generators, dtype=float).reshape(-1, 2)
    if samples.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    if generators.shape[0] == 0:
        raise ValueError("posterior mesh ownership requires at least one generator.")
    try:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(generators).query(samples, k=1)
        return np.asarray(idx, dtype=np.int64)
    except Exception:
        diff = samples[:, None, :] - generators[None, :, :]
        return np.argmin(np.sum(diff * diff, axis=2), axis=1).astype(np.int64, copy=False)


def aggregate_masses(ownership: np.ndarray, weights: np.ndarray, n_generators: int) -> np.ndarray:
    ownership = np.asarray(ownership, dtype=np.int64).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    n = min(ownership.size, weights.size)
    mass = np.zeros(int(n_generators), dtype=float)
    if n == 0:
        return mass
    idx = ownership[:n]
    valid = (idx >= 0) & (idx < int(n_generators)) & np.isfinite(weights[:n])
    if np.any(valid):
        np.add.at(mass, idx[valid], weights[:n][valid])
    return mass


def regular_support_areas(
    x: np.ndarray,
    y: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> Optional[np.ndarray]:
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


def voronoi_finite_polygons_2d(vor, radius=None):
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


def clipped_voronoi_cells(
    x: np.ndarray,
    y: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    *,
    include_polygons: bool = False,
) -> VoronoiCells:
    try:
        from scipy.spatial import QhullError, Voronoi
    except Exception as e:
        raise ImportError("posterior mesh Voronoi cells require scipy.spatial.Voronoi.") from e
    try:
        from shapely.geometry import Polygon, box
    except Exception as e:
        raise ImportError("posterior mesh Voronoi cells require shapely.") from e

    pts = np.column_stack([x, y])
    if pts.shape[0] < 4:
        raise ValueError("posterior mesh Voronoi areas require at least four unique 2D support points.")
    if np.linalg.matrix_rank(pts - pts.mean(axis=0)) < 2:
        raise ValueError("posterior mesh Voronoi areas require non-collinear 2D support points.")
    try:
        vor = Voronoi(pts)
    except QhullError as e:
        raise ValueError(f"posterior mesh failed to construct Voronoi areas: {e}") from e

    regions, vertices = voronoi_finite_polygons_2d(
        vor,
        radius=max(xlim[1] - xlim[0], ylim[1] - ylim[0]) * 4,
    )
    bbox = box(float(xlim[0]), float(ylim[0]), float(xlim[1]), float(ylim[1]))
    areas = np.zeros(len(regions), dtype=float)
    polygons = [] if include_polygons else None
    for idx, region in enumerate(regions):
        clipped = None
        if len(region) >= 3:
            poly = Polygon(vertices[region])
            if not poly.is_valid:
                poly = poly.buffer(0)
            clipped = poly.intersection(bbox)
            areas[idx] = float(clipped.area) if not clipped.is_empty else 0.0
        if polygons is not None:
            polygons.append(clipped)
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0):
        raise ValueError("posterior mesh Voronoi areas are invalid or non-positive.")
    return VoronoiCells(areas=areas, polygons=polygons, method="clipped_voronoi")


def support_areas(
    x: np.ndarray,
    y: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    *,
    include_polygons: bool = False,
) -> VoronoiCells:
    areas = None if include_polygons else regular_support_areas(x, y, xlim, ylim)
    if areas is not None:
        return VoronoiCells(areas=areas, polygons=None, method="regular_grid")
    return clipped_voronoi_cells(
        x,
        y,
        xlim,
        ylim,
        include_polygons=include_polygons,
    )


def conservative_density(masses: np.ndarray, areas: np.ndarray) -> np.ndarray:
    masses = np.asarray(masses, dtype=float)
    areas = np.asarray(areas, dtype=float)
    densities = np.full(masses.shape, np.nan, dtype=float)
    valid = np.isfinite(masses) & np.isfinite(areas) & (areas > 0)
    densities[valid] = masses[valid] / areas[valid]
    return densities


def _covariance_for_group(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return np.full((2, 2), np.nan, dtype=float)
    centered = points - np.mean(points, axis=0)
    return (centered.T @ centered) / float(points.shape[0] - 1)


def _positive_scale(values: np.ndarray, default: float = 1.0) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if finite.size == 0:
        return float(default)
    scale = float(np.median(finite))
    if not np.isfinite(scale) or scale <= 0:
        return float(default)
    return scale


def _principal_direction_from_eigvecs(eigvecs: np.ndarray) -> np.ndarray:
    principal = np.full((eigvecs.shape[0], 2), np.nan, dtype=float)
    if eigvecs.size == 0:
        return principal
    principal[:, 0] = eigvecs[:, 0, 0]
    principal[:, 1] = eigvecs[:, 1, 0]
    return principal


def _ellipse_params(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    *,
    nsigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = 2.0 * float(nsigma) * np.sqrt(np.maximum(eigvals[:, 0], 0.0))
    height = 2.0 * float(nsigma) * np.sqrt(np.maximum(eigvals[:, 1], 0.0))
    angle = np.degrees(np.arctan2(eigvecs[:, 1, 0], eigvecs[:, 0, 0]))
    return width, height, angle


def _mesh_moments(
    generators: np.ndarray,
    samples: np.ndarray,
    weights: np.ndarray,
    ownership: np.ndarray,
    *,
    proposal_weights: Optional[np.ndarray] = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Collect per-generator geometric diagnostics from assigned samples."""
    n_generators = int(generators.shape[0])
    sample_counts = np.zeros(n_generators, dtype=np.int64)
    covariance = np.full((n_generators, 2, 2), np.nan, dtype=float)
    eigvals = np.full((n_generators, 2), np.nan, dtype=float)
    eigvecs = np.full((n_generators, 2, 2), np.nan, dtype=float)
    principal_direction = np.full((n_generators, 2), np.nan, dtype=float)
    local_spread = np.full(n_generators, np.nan, dtype=float)
    anisotropy = np.full(n_generators, np.nan, dtype=float)
    density_gradient_proxy = np.full(n_generators, np.nan, dtype=float)
    curvature_proxy = np.full(n_generators, np.nan, dtype=float)
    centroid = np.full((n_generators, 2), np.nan, dtype=float)
    drift = np.full((n_generators, 2), np.nan, dtype=float)

    samples = np.asarray(samples, dtype=float).reshape(-1, 2)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    proposal_weights = weights if proposal_weights is None else np.asarray(proposal_weights, dtype=float).reshape(-1)
    ownership = np.asarray(ownership, dtype=np.int64).reshape(-1)
    n = min(samples.shape[0], weights.size, proposal_weights.size, ownership.size)
    if n == 0:
        return (
            sample_counts,
            covariance,
            eigvals,
            eigvecs,
            principal_direction,
            local_spread,
            anisotropy,
            density_gradient_proxy,
            curvature_proxy,
            centroid,
            drift,
            np.linalg.norm(drift, axis=1),
            np.full(n_generators, np.nan, dtype=float),
        )

    samples = samples[:n]
    weights = weights[:n]
    proposal_weights = proposal_weights[:n]
    ownership = ownership[:n]
    valid = (ownership >= 0) & (ownership < n_generators) & np.isfinite(weights) & np.isfinite(proposal_weights)
    if not np.any(valid):
        return (
            sample_counts,
            covariance,
            eigvals,
            eigvecs,
            principal_direction,
            local_spread,
            anisotropy,
            density_gradient_proxy,
            curvature_proxy,
            centroid,
            drift,
            np.linalg.norm(drift, axis=1),
            np.full(n_generators, np.nan, dtype=float),
        )

    samples = samples[valid]
    weights = weights[valid]
    proposal_weights = proposal_weights[valid]
    ownership = ownership[valid]
    order = np.argsort(ownership, kind="stable")
    samples = samples[order]
    weights = weights[order]
    proposal_weights = proposal_weights[order]
    ownership = ownership[order]
    counts = np.bincount(ownership, minlength=n_generators)
    sample_counts[:] = counts
    pos = 0
    for idx, count in enumerate(counts):
        if count <= 0:
            continue
        end = pos + int(count)
        pts = samples[pos:end]
        covariance[idx] = _covariance_for_group(pts)
        if np.all(np.isfinite(covariance[idx])):
            vals, vecs = np.linalg.eigh(covariance[idx])
            order_ev = np.argsort(vals)[::-1]
            vals = vals[order_ev]
            vecs = vecs[:, order_ev]
            eigvals[idx] = vals
            eigvecs[idx] = vecs
            principal_direction[idx] = vecs[:, 0]
            local_spread[idx] = float(np.sqrt(max(float(np.sum(vals)), 0.0)))
            if vals[1] > 0:
                anisotropy[idx] = vals[0] / vals[1]
            elif vals[0] > 0:
                anisotropy[idx] = np.inf
            denom = float(np.sqrt(max(float(vals[1]), 0.0)) + 1.0e-12)
            curvature_proxy[idx] = float(np.sqrt(max(float(vals[0]), 0.0)) / denom / (1.0 + local_spread[idx]))
        w = proposal_weights[pos:end]
        valid_w = np.isfinite(w) & (w > 0)
        if np.any(valid_w):
            proposal = np.sum(pts[valid_w] * w[valid_w, None], axis=0) / float(np.sum(w[valid_w]))
        else:
            proposal = np.mean(pts, axis=0)
        centroid[idx] = proposal
        drift[idx] = proposal - generators[idx]
        pos = end
    drift_mag = np.linalg.norm(drift, axis=1)
    density_gradient_proxy = drift_mag / (local_spread + 1.0e-12)
    geometry_stress = geometry_stress_score(
        local_spread=local_spread,
        anisotropy_ratio=anisotropy,
        density_gradient_proxy=density_gradient_proxy,
        curvature_proxy=curvature_proxy,
    )
    return (
        sample_counts,
        covariance,
        eigvals,
        eigvecs,
        principal_direction,
        local_spread,
        anisotropy,
        density_gradient_proxy,
        curvature_proxy,
        centroid,
        drift,
        drift_mag,
        geometry_stress,
    )


def mesh_stress_score(
    densities: np.ndarray,
    anisotropy_ratio: np.ndarray,
    centroid_drift_magnitude: np.ndarray,
) -> np.ndarray:
    density_term = np.asarray(densities, dtype=float)
    finite_density = density_term[np.isfinite(density_term) & (density_term > 0)]
    if finite_density.size:
        scale = float(np.nanmedian(finite_density))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        density_term = density_term / scale
    anisotropy_term = np.asarray(anisotropy_ratio, dtype=float)
    drift_term = np.asarray(centroid_drift_magnitude, dtype=float)
    stress = np.nan_to_num(density_term, nan=0.0, posinf=0.0, neginf=0.0)
    stress *= 1.0 + np.nan_to_num(np.log1p(anisotropy_term), nan=0.0, posinf=0.0)
    stress *= 1.0 + np.nan_to_num(drift_term, nan=0.0, posinf=0.0)
    return stress


def geometry_stress_score(
    *,
    local_spread: np.ndarray,
    anisotropy_ratio: np.ndarray,
    density_gradient_proxy: np.ndarray,
    curvature_proxy: np.ndarray,
) -> np.ndarray:
    spread_scale = _positive_scale(local_spread)
    aniso_scale = _positive_scale(anisotropy_ratio)
    grad_scale = _positive_scale(density_gradient_proxy)
    curvature_scale = _positive_scale(curvature_proxy)

    spread_term = np.log1p(np.asarray(local_spread, dtype=float) / spread_scale)
    aniso_term = np.log1p(np.asarray(anisotropy_ratio, dtype=float) / aniso_scale)
    grad_term = np.log1p(np.asarray(density_gradient_proxy, dtype=float) / grad_scale)
    curvature_term = np.log1p(np.asarray(curvature_proxy, dtype=float) / curvature_scale)

    score = 0.25 * np.nan_to_num(spread_term, nan=0.0, posinf=0.0, neginf=0.0)
    score += 0.40 * np.nan_to_num(aniso_term, nan=0.0, posinf=0.0, neginf=0.0)
    score += 0.20 * np.nan_to_num(curvature_term, nan=0.0, posinf=0.0, neginf=0.0)
    score += 0.15 * np.nan_to_num(grad_term, nan=0.0, posinf=0.0, neginf=0.0)
    return score


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    finite = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    a = a[finite]
    b = b[finite]
    if not np.isfinite(np.nanstd(a)) or not np.isfinite(np.nanstd(b)):
        return float("nan")
    if np.nanstd(a) <= 0.0 or np.nanstd(b) <= 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _domain_scale(xlim: Tuple[float, float], ylim: Tuple[float, float]) -> float:
    return float(max(abs(float(xlim[1]) - float(xlim[0])), abs(float(ylim[1]) - float(ylim[0])), 1.0))


def _normalize_refinement_config(
    refinement: Optional[dict[str, Any]],
    *,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> dict[str, Any]:
    cfg = dict(refinement or {})
    scale = _domain_scale(xlim, ylim)

    def _as_float(name: str, default: float) -> float:
        value = cfg.get(name, default)
        try:
            out = float(value)
        except Exception:
            out = float(default)
        return out

    def _as_int(name: str, default: int) -> int:
        value = cfg.get(name, default)
        try:
            out = int(value)
        except Exception:
            out = int(default)
        return out

    out = {
        "enabled": bool(cfg.get("enabled", False)),
        "iterations": max(0, _as_int("iterations", 10)),
        "alpha": float(np.clip(_as_float("alpha", 0.3), 0.05, 1.0)),
        "eta": float(np.clip(_as_float("eta", 0.5), 0.0, 1.0)),
        "anisotropic": bool(cfg.get("anisotropic", True)),
        "split_enabled": bool(cfg.get("split_enabled", True)),
        "merge_enabled": bool(cfg.get("merge_enabled", True)),
        "record_history": bool(cfg.get("record_history", False)),
        "seed": None if cfg.get("seed", None) is None else int(cfg.get("seed")),
        "split_quantile": float(np.clip(_as_float("split_quantile", 0.80), 0.0, 1.0)),
        "merge_quantile": float(np.clip(_as_float("merge_quantile", 0.20), 0.0, 1.0)),
        "max_splits": cfg.get("max_splits", None),
        "max_merges": cfg.get("max_merges", None),
        "max_generators": cfg.get("max_generators", None),
        "min_generators": max(1, _as_int("min_generators", 4)),
        "drift_clip": max(0.0, _as_float("drift_clip", 2.0)),
        "anisotropy_cap": max(1.0, _as_float("anisotropy_cap", 30.0)),
        "covariance_floor": max(1.0e-12, _as_float("covariance_floor", 1.0e-6 * scale * scale)),
        "split_offset": max(0.0, _as_float("split_offset", 0.45)),
        "min_separation": max(0.0, _as_float("min_separation", 0.02 * scale)),
        "merge_distance": max(0.0, _as_float("merge_distance", 0.035 * scale)),
        "empty_reseed": bool(cfg.get("empty_reseed", True)),
        "sample_jitter": max(0.0, _as_float("sample_jitter", 0.01 * scale)),
        "candidate_k": max(1, _as_int("candidate_k", 8)),
        "ownership_batch_size": max(256, _as_int("ownership_batch_size", 50000)),
    }
    if out["max_splits"] is not None:
        out["max_splits"] = max(0, int(out["max_splits"]))
    if out["max_merges"] is not None:
        out["max_merges"] = max(0, int(out["max_merges"]))
    if out["max_generators"] is not None:
        out["max_generators"] = max(int(out["min_generators"]), int(out["max_generators"]))
    return out


def _regularize_covariance_matrix(
    cov: np.ndarray,
    *,
    covariance_floor: float,
    anisotropy_cap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    cov = np.asarray(cov, dtype=float).reshape(2, 2)
    if not np.all(np.isfinite(cov)):
        eigvals = np.array([covariance_floor * anisotropy_cap, covariance_floor], dtype=float)
        eigvecs = np.eye(2, dtype=float)
        return np.diag(eigvals), eigvals, eigvecs, eigvecs[:, 0], float(eigvals[0] / eigvals[1])
    cov = 0.5 * (cov + cov.T)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except Exception:
        eigvals = np.array([covariance_floor * anisotropy_cap, covariance_floor], dtype=float)
        eigvecs = np.eye(2, dtype=float)
        return np.diag(eigvals), eigvals, eigvecs, eigvecs[:, 0], float(eigvals[0] / eigvals[1])
    order = np.argsort(eigvals)[::-1]
    eigvals = np.asarray(eigvals[order], dtype=float)
    eigvecs = np.asarray(eigvecs[:, order], dtype=float)
    eigvals = np.maximum(eigvals, covariance_floor)
    if eigvals[1] <= 0.0:
        eigvals[1] = covariance_floor
    max_minor = eigvals[0] / float(anisotropy_cap)
    if eigvals[1] < max_minor:
        eigvals[1] = max_minor
    eigvals = np.maximum(eigvals, covariance_floor)
    cov_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
    principal = eigvecs[:, 0]
    anisotropy = float(eigvals[0] / max(eigvals[1], covariance_floor))
    return cov_reg, eigvals, eigvecs, principal, anisotropy


def _anisotropic_ownership(
    samples: np.ndarray,
    generators: np.ndarray,
    covariance: np.ndarray,
    *,
    covariance_floor: float,
    anisotropy_cap: float,
    candidate_k: int = 8,
    batch_size: int = 50000,
    diagnostics: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=float).reshape(-1, 2)
    generators = np.asarray(generators, dtype=float).reshape(-1, 2)
    covariance = np.asarray(covariance, dtype=float)
    if samples.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    if generators.shape[0] == 0:
        raise ValueError("posterior mesh ownership requires at least one generator.")
    if covariance.ndim != 3 or covariance.shape[0] != generators.shape[0]:
        if diagnostics is not None:
            diagnostics["anisotropic_fallback"] = "covariance_shape"
            diagnostics["anisotropic_fallback_count"] = int(samples.shape[0])
        return assign_nearest_ownership(samples, generators)
    try:
        from scipy.spatial import cKDTree
    except Exception:
        if diagnostics is not None:
            diagnostics["anisotropic_fallback"] = "missing_scipy"
            diagnostics["anisotropic_fallback_count"] = int(samples.shape[0])
        return assign_nearest_ownership(samples, generators)

    k = max(1, min(int(candidate_k), int(generators.shape[0])))
    batch = max(1, int(batch_size))
    tree = cKDTree(generators)
    ownership = np.empty(samples.shape[0], dtype=np.int64)
    anisotropy_cap = float(max(1.0, anisotropy_cap))
    covariance_floor = float(max(covariance_floor, 1.0e-15))
    fallback_count = 0

    for start in range(0, samples.shape[0], batch):
        stop = min(samples.shape[0], start + batch)
        pts = samples[start:stop]
        cand = tree.query(pts, k=k)[1]
        cand = np.asarray(cand, dtype=np.int64)
        if cand.ndim == 1:
            cand = cand[:, None]
        cand = np.clip(cand, 0, generators.shape[0] - 1)

        scores = np.full(cand.shape, np.inf, dtype=float)
        for j in range(cand.shape[1]):
            idx = cand[:, j]
            diff = pts - generators[idx]
            cov_batch = covariance[idx]
            try:
                eigvals, eigvecs = np.linalg.eigh(cov_batch)
                order = np.argsort(eigvals, axis=-1)[:, ::-1]
                eigvals_sorted = np.take_along_axis(eigvals, order, axis=-1)
                eigvecs_sorted = np.take_along_axis(eigvecs, order[:, None, :], axis=-1)
                major_vecs = eigvecs_sorted[:, :, 0]
                minor_vecs = eigvecs_sorted[:, :, 1]
                major_norm = np.linalg.norm(major_vecs, axis=-1, keepdims=True)
                major_vecs = np.divide(
                    major_vecs,
                    np.where(major_norm > 0.0, major_norm, 1.0),
                    out=np.zeros_like(major_vecs),
                    where=True,
                )
                minor_norm = np.linalg.norm(minor_vecs, axis=-1, keepdims=True)
                minor_vecs = np.divide(
                    minor_vecs,
                    np.where(minor_norm > 0.0, minor_norm, 1.0),
                    out=np.zeros_like(minor_vecs),
                    where=True,
                )
                major_scale = np.sqrt(np.maximum(eigvals_sorted[:, 0], covariance_floor))
                minor_scale = np.sqrt(np.maximum(eigvals_sorted[:, 1], covariance_floor))
                minor_scale = np.maximum(minor_scale, major_scale / anisotropy_cap)
                proj_major = np.sum(diff * major_vecs, axis=-1)
                proj_minor = np.sum(diff * minor_vecs, axis=-1)
                score = (proj_major / major_scale) ** 2 + (proj_minor / minor_scale) ** 2
                score = np.where(np.isfinite(score), score, np.sum(diff * diff, axis=-1))
            except np.linalg.LinAlgError:
                fallback_count += int(diff.shape[0])
                score = np.sum(diff * diff, axis=-1)
            scores[:, j] = score
        ownership[start:stop] = cand[np.arange(cand.shape[0]), np.argmin(scores, axis=1)]
    if diagnostics is not None:
        diagnostics["anisotropic_fallback"] = "none" if fallback_count == 0 else "lin_alg"
        diagnostics["anisotropic_fallback_count"] = int(fallback_count)
    return ownership


def _clip_generators_to_domain(
    generators: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> np.ndarray:
    gens = np.asarray(generators, dtype=float).reshape(-1, 2).copy()
    gens[:, 0] = np.clip(gens[:, 0], float(xlim[0]), float(xlim[1]))
    gens[:, 1] = np.clip(gens[:, 1], float(ylim[0]), float(ylim[1]))
    return gens


def _greedy_unique_generators(
    generators: np.ndarray,
    priority: np.ndarray,
    *,
    min_separation: float,
    min_generators: int,
) -> np.ndarray:
    gens = np.asarray(generators, dtype=float).reshape(-1, 2)
    if gens.shape[0] <= 1 or min_separation <= 0.0:
        return gens
    priority = np.asarray(priority, dtype=float).reshape(-1)
    if priority.size > gens.shape[0]:
        priority = priority[: gens.shape[0]]
    elif priority.size < gens.shape[0]:
        finite = priority[np.isfinite(priority)]
        pad_value = float(np.min(finite)) - 1.0 if finite.size else -1.0
        priority = np.concatenate([priority, np.full(gens.shape[0] - priority.size, pad_value, dtype=float)])
    order = np.argsort(np.nan_to_num(priority, nan=-np.inf))[::-1]
    keep: list[int] = []
    min_sep2 = float(min_separation) ** 2
    for idx in order:
        if len(keep) >= gens.shape[0]:
            break
        pt = gens[idx]
        if any(float(np.sum((pt - gens[j]) ** 2)) < min_sep2 for j in keep):
            continue
        keep.append(int(idx))
    if len(keep) < min_generators:
        for idx in order:
            if idx not in keep:
                keep.append(int(idx))
            if len(keep) >= min_generators:
                break
    keep = sorted(set(keep))
    return gens[keep]


def _reseed_empty_generators(
    generators: np.ndarray,
    mesh: PosteriorMesh,
    samples: np.ndarray,
    weights: np.ndarray,
    *,
    cfg: dict[str, Any],
    rng: np.random.Generator,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> tuple[np.ndarray, int]:
    empty = np.where(np.asarray(mesh.sample_counts, dtype=int) == 0)[0]
    if empty.size == 0 or not cfg.get("empty_reseed", True):
        return np.asarray(generators, dtype=float).reshape(-1, 2), 0
    samples = np.asarray(samples, dtype=float).reshape(-1, 2)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if samples.shape[0] == 0:
        return np.asarray(generators, dtype=float).reshape(-1, 2), 0
    tempered = np.power(np.clip(weights, 0.0, None), cfg["alpha"]).astype(float, copy=False)
    order = np.argsort(np.nan_to_num(tempered, nan=0.0))[::-1]
    base = np.asarray(generators, dtype=float).reshape(-1, 2).copy()
    jitter_scale = float(cfg["sample_jitter"])
    updates = 0
    used: list[np.ndarray] = []
    for slot, sample_idx in zip(empty, order):
        cand = samples[sample_idx].astype(float, copy=True)
        if jitter_scale > 0.0:
            cand = cand + rng.normal(scale=jitter_scale, size=2)
        cand = _clip_generators_to_domain(cand[None, :], xlim, ylim)[0]
        if used:
            if np.min(np.linalg.norm(np.asarray(used) - cand, axis=1)) < cfg["min_separation"]:
                cand = _clip_generators_to_domain((cand + 0.5 * rng.normal(scale=jitter_scale, size=2))[None, :], xlim, ylim)[0]
        base[int(slot)] = cand
        used.append(cand)
        updates += 1
    return base, updates


def _split_merge_generators(
    generators: np.ndarray,
    mesh: PosteriorMesh,
    *,
    cfg: dict[str, Any],
    rng: np.random.Generator,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> tuple[np.ndarray, dict[str, int]]:
    gens = np.asarray(generators, dtype=float).reshape(-1, 2)
    if gens.shape[0] == 0:
        return gens, {"splits": 0, "merges": 0, "reseed": 0}

    split_count = 0
    merge_count = 0

    finite_geom = np.isfinite(mesh.geometry_stress)
    split_candidates = np.where(finite_geom & (mesh.sample_counts > 0))[0]
    if cfg["split_enabled"] and split_candidates.size:
        split_candidates = split_candidates[np.argsort(mesh.geometry_stress[split_candidates])[::-1]]
        if cfg["split_quantile"] > 0.0:
            threshold = float(np.nanquantile(mesh.geometry_stress[finite_geom], cfg["split_quantile"]))
            split_candidates = split_candidates[np.where(mesh.geometry_stress[split_candidates] >= threshold)[0]]
        max_splits = cfg["max_splits"]
        if max_splits is None:
            max_splits = max(1, int(np.ceil(0.15 * gens.shape[0])))
        max_splits = min(int(max_splits), int(cfg["max_generators"]) - gens.shape[0]) if cfg["max_generators"] is not None else int(max_splits)
        max_splits = max(0, int(max_splits))
        principal = np.asarray(mesh.principal_direction, dtype=float)
        eigvals = np.asarray(mesh.covariance_eigenvalues, dtype=float)
        for idx in split_candidates[:max_splits]:
            direction = principal[idx]
            if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) <= 0.0:
                direction = np.array([1.0, 0.0], dtype=float)
            direction = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
            major = float(np.sqrt(max(float(eigvals[idx, 0]), cfg["covariance_floor"])))
            offset = cfg["split_offset"] * major * direction
            if np.linalg.norm(offset) < cfg["min_separation"]:
                offset = cfg["min_separation"] * direction
            perpendicular = np.array([-direction[1], direction[0]], dtype=float)
            candidates = [
                gens[int(idx)] + offset,
                gens[int(idx)] - offset,
                gens[int(idx)] + 0.75 * offset + 0.25 * cfg["min_separation"] * perpendicular,
                gens[int(idx)] - 0.75 * offset - 0.25 * cfg["min_separation"] * perpendicular,
            ]
            accepted = False
            for cand in candidates:
                cand = _clip_generators_to_domain(cand[None, :], xlim, ylim)[0]
                min_dist = float(np.min(np.linalg.norm(gens - cand, axis=1)))
                if min_dist >= 0.75 * cfg["min_separation"]:
                    gens = np.vstack([gens, cand[None, :]])
                    split_count += 1
                    accepted = True
                    break
            if not accepted and rng.random() < 0.5:
                cand = _clip_generators_to_domain((gens[int(idx)] + 1.5 * offset)[None, :], xlim, ylim)[0]
                if float(np.min(np.linalg.norm(gens - cand, axis=1))) >= 0.75 * cfg["min_separation"]:
                    gens = np.vstack([gens, cand[None, :]])
                    split_count += 1

    finite_geom = np.isfinite(mesh.geometry_stress)
    merge_candidates = np.where(finite_geom & (mesh.sample_counts > 0))[0]
    merge_base_count = min(
        int(gens.shape[0]),
        int(mesh.geometry_stress.shape[0]),
        int(mesh.masses.shape[0]),
    )
    merge_candidates = merge_candidates[merge_candidates < merge_base_count]
    if cfg["merge_enabled"] and merge_candidates.size >= 2 and merge_base_count > cfg["min_generators"]:
        if cfg["merge_quantile"] > 0.0:
            threshold = float(np.nanquantile(mesh.geometry_stress[finite_geom], cfg["merge_quantile"]))
            merge_candidates = merge_candidates[np.where(mesh.geometry_stress[merge_candidates] <= threshold)[0]]
        if merge_candidates.size >= 2:
            try:
                from scipy.spatial import cKDTree

                merge_candidate_set = {int(idx) for idx in merge_candidates}
                tree = cKDTree(gens[:merge_base_count])
                order = merge_candidates[np.argsort(np.nan_to_num(mesh.geometry_stress[merge_candidates], nan=np.inf))]
                removed: set[int] = set()
                max_merges = cfg["max_merges"]
                if max_merges is None:
                    max_merges = max(1, int(np.ceil(0.10 * merge_base_count)))
                for idx in order:
                    if idx in removed or merge_base_count - len(removed) <= cfg["min_generators"]:
                        continue
                    dist, nn = tree.query(gens[idx], k=min(2, merge_base_count))
                    if np.ndim(nn) == 0 or len(np.atleast_1d(nn)) < 2:
                        continue
                    partner = int(nn[1])
                    if partner in removed or partner == idx:
                        continue
                    if partner >= merge_base_count:
                        continue
                    if partner not in merge_candidate_set:
                        continue
                    if float(np.linalg.norm(gens[idx] - gens[partner])) > cfg["merge_distance"]:
                        continue
                    if mesh.geometry_stress[partner] > mesh.geometry_stress[idx]:
                        survivor, loser = partner, idx
                    else:
                        survivor, loser = idx, partner
                    mass_i = float(mesh.masses[idx]) if idx < mesh.masses.shape[0] else 1.0
                    mass_j = float(mesh.masses[partner]) if partner < mesh.masses.shape[0] else 1.0
                    total = max(mass_i + mass_j, 1.0e-12)
                    gens[survivor] = (mass_i * gens[idx] + mass_j * gens[partner]) / total
                    removed.add(int(loser))
                    merge_count += 1
                    if merge_count >= int(max_merges):
                        break
                if removed:
                    keep = [i for i in range(gens.shape[0]) if i not in removed]
                    gens = gens[keep]
            except Exception:
                pass

    priority = np.nan_to_num(mesh.geometry_stress, nan=-np.inf) + 0.1 * np.nan_to_num(mesh.mass_stress, nan=-np.inf)
    gens = _greedy_unique_generators(
        gens,
        priority,
        min_separation=cfg["min_separation"],
        min_generators=cfg["min_generators"],
    )
    gens = _clip_generators_to_domain(gens, xlim, ylim)
    return gens, {"splits": int(split_count), "merges": int(merge_count), "reseed": 0}


def mesh_diagnostics_frame(
    mesh: PosteriorMesh,
    *,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    generators = np.asarray(mesh.generators, dtype=float)
    if x is None:
        x = generators[:, 0]
    if y is None:
        y = generators[:, 1]
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = int(generators.shape[0])
    if x.size != n or y.size != n:
        raise ValueError("mesh diagnostics x/y arrays must match generator count.")

    eigvals = np.asarray(mesh.covariance_eigenvalues, dtype=float)
    eigvecs = np.asarray(mesh.covariance_eigenvectors, dtype=float)
    major_x = eigvecs[:, 0, 0]
    major_y = eigvecs[:, 1, 0]
    major_scale = np.sqrt(np.maximum(eigvals[:, 0], 0.0))
    ellipse_width, ellipse_height, ellipse_angle = _ellipse_params(eigvals, eigvecs)
    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "generator_x_norm": generators[:, 0],
            "generator_y_norm": generators[:, 1],
            "mass": mesh.masses,
            "area": mesh.areas,
            "density": mesh.densities,
            "sample_count": mesh.sample_counts,
            "centroid_x_norm": mesh.centroid_proposal[:, 0],
            "centroid_y_norm": mesh.centroid_proposal[:, 1],
            "drift_x_norm": mesh.centroid_drift[:, 0],
            "drift_y_norm": mesh.centroid_drift[:, 1],
            "drift_magnitude": mesh.centroid_drift_magnitude,
            "cov_xx": mesh.covariance[:, 0, 0],
            "cov_xy": mesh.covariance[:, 0, 1],
            "cov_yy": mesh.covariance[:, 1, 1],
            "eigval_major": eigvals[:, 0],
            "eigval_minor": eigvals[:, 1],
            "principal_dx": mesh.principal_direction[:, 0],
            "principal_dy": mesh.principal_direction[:, 1],
            "local_spread": mesh.local_spread,
            "anisotropy_ratio": mesh.anisotropy_ratio,
            "density_gradient_proxy": mesh.density_gradient_proxy,
            "curvature_proxy": mesh.curvature_proxy,
            "anisotropy_x": major_x * major_scale,
            "anisotropy_y": major_y * major_scale,
            "ellipse_width": ellipse_width,
            "ellipse_height": ellipse_height,
            "ellipse_angle_deg": ellipse_angle,
            "mass_stress": mesh.mass_stress,
            "geometry_stress": mesh.geometry_stress,
            "stress_gap": mesh.stress_gap,
            "stress": mesh.mass_stress,
        }
    )


def _build_mesh_state(
    generators: np.ndarray,
    samples: np.ndarray,
    weights: np.ndarray,
    *,
    ownership: Optional[np.ndarray] = None,
    xlim: Tuple[float, float] = (0.0, 1.0),
    ylim: Tuple[float, float] = (0.0, 1.0),
    include_polygons: bool = False,
    strict_geometry: bool = False,
    geometry_space: str = "normalized",
    compute_areas: bool = True,
) -> PosteriorMesh:
    generators = np.asarray(generators, dtype=float).reshape(-1, 2)
    samples = np.asarray(samples, dtype=float).reshape(-1, 2)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if ownership is None:
        ownership = assign_nearest_ownership(samples, generators)
    else:
        ownership = np.asarray(ownership, dtype=np.int64).reshape(-1)

    masses = aggregate_masses(ownership, weights, int(generators.shape[0]))
    polygons = None
    area_method = "not_performed"
    geometry_error = None
    if compute_areas:
        try:
            cells = support_areas(
                generators[:, 0],
                generators[:, 1],
                xlim,
                ylim,
                include_polygons=include_polygons,
            )
            areas = cells.areas
            polygons = cells.polygons
            area_method = cells.method
        except Exception as exc:
            if strict_geometry:
                raise
            areas = np.full(generators.shape[0], np.nan, dtype=float)
            geometry_error = str(exc)
    else:
        areas = np.full(generators.shape[0], np.nan, dtype=float)

    densities = conservative_density(masses, areas)
    (
        sample_counts,
        covariance,
        eigvals,
        eigvecs,
        principal_direction,
        local_spread,
        anisotropy,
        density_gradient_proxy,
        curvature_proxy,
        centroid,
        drift,
        drift_mag,
        geometry_stress,
    ) = _mesh_moments(generators, samples, weights, ownership)
    mass_stress = mesh_stress_score(densities, anisotropy, drift_mag)
    stress_gap = geometry_stress - mass_stress
    diagnostics = {
        "generator_count": int(generators.shape[0]),
        "sample_count": int(samples.shape[0]),
        "assigned_sample_count": int(np.count_nonzero((ownership >= 0) & (ownership < generators.shape[0]))),
        "nonzero_mass_generators": int(np.count_nonzero(masses > 0)),
        "empty_generators": int(np.count_nonzero(sample_counts == 0)),
        "area_method": area_method,
        "geometry_space": str(geometry_space),
        "geometry_error": geometry_error,
        "mass_sum": float(np.sum(masses)),
        "max_centroid_drift": float(np.nanmax(drift_mag)) if np.isfinite(drift_mag).any() else float("nan"),
        "max_anisotropy_ratio": float(np.nanmax(anisotropy)) if np.isfinite(anisotropy).any() else float("nan"),
        "mass_geometry_stress_corr": _safe_corrcoef(mass_stress, geometry_stress),
    }
    return PosteriorMesh(
        generators=generators,
        ownership=ownership,
        masses=masses,
        areas=areas,
        densities=densities,
        sample_counts=sample_counts,
        covariance=covariance,
        covariance_eigenvalues=eigvals,
        covariance_eigenvectors=eigvecs,
        principal_direction=principal_direction,
        local_spread=local_spread,
        anisotropy_ratio=anisotropy,
        density_gradient_proxy=density_gradient_proxy,
        curvature_proxy=curvature_proxy,
        centroid_proposal=centroid,
        centroid_drift=drift,
        centroid_drift_magnitude=drift_mag,
        mass_stress=mass_stress,
        geometry_stress=geometry_stress,
        stress_gap=stress_gap,
        diagnostics=diagnostics,
        polygons=polygons,
    )


def refine_posterior_mesh(
    mesh: PosteriorMesh,
    samples: np.ndarray,
    weights: np.ndarray,
    *,
    xlim: Tuple[float, float] = (0.0, 1.0),
    ylim: Tuple[float, float] = (0.0, 1.0),
    refinement: Optional[dict[str, Any]] = None,
    include_polygons: bool = False,
    strict_geometry: bool = False,
    geometry_space: str = "normalized",
) -> PosteriorMesh:
    cfg = _normalize_refinement_config(refinement, xlim=xlim, ylim=ylim)
    samples = np.asarray(samples, dtype=float).reshape(-1, 2)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    generators = np.asarray(mesh.generators, dtype=float).reshape(-1, 2).copy()

    if not cfg["enabled"] or cfg["iterations"] <= 0:
        final_mesh = _build_mesh_state(
            generators,
            samples,
            weights,
            ownership=mesh.ownership,
            xlim=xlim,
            ylim=ylim,
            include_polygons=include_polygons,
            strict_geometry=strict_geometry,
            geometry_space=geometry_space,
        )
        final_mesh.diagnostics.update(
            {
                "refinement_enabled": False,
                "refinement_iterations_requested": int(cfg["iterations"]),
                "refinement_iterations_completed": 0,
                "refinement_splits": 0,
                "refinement_merges": 0,
                "refinement_reseeds": 0,
            }
        )
        return final_mesh

    rng = np.random.default_rng(cfg["seed"])
    trace: list[dict[str, Any]] = []
    generator_history: list[np.ndarray] = [generators.copy()] if cfg["record_history"] else []
    totals = {"splits": 0, "merges": 0, "reseed": 0}
    anisotropic_fallback_total = 0
    current_mesh = _build_mesh_state(
        generators,
        samples,
        weights,
        ownership=assign_nearest_ownership(samples, generators),
        xlim=xlim,
        ylim=ylim,
        include_polygons=False,
        strict_geometry=False,
        geometry_space=geometry_space,
    )
    completed = 0

    for iteration in range(int(cfg["iterations"])):
        ownership_diag: dict[str, Any] = {}
        if cfg["anisotropic"] and current_mesh.covariance.shape[0] == generators.shape[0]:
            ownership = _anisotropic_ownership(
                samples,
                generators,
                current_mesh.covariance,
                covariance_floor=cfg["covariance_floor"],
                anisotropy_cap=cfg["anisotropy_cap"],
                candidate_k=cfg["candidate_k"],
                batch_size=cfg["ownership_batch_size"],
                diagnostics=ownership_diag,
            )
        else:
            ownership = assign_nearest_ownership(samples, generators)
            if cfg["anisotropic"]:
                ownership_diag["anisotropic_fallback"] = "covariance_shape"
                ownership_diag["anisotropic_fallback_count"] = int(samples.shape[0])
        anisotropic_fallback_total += int(ownership_diag.get("anisotropic_fallback_count", 0))

        current_mesh = _build_mesh_state(
            generators,
            samples,
            weights,
            ownership=ownership,
            xlim=xlim,
            ylim=ylim,
            include_polygons=False,
            strict_geometry=False,
            geometry_space=geometry_space,
        )

        proposal_weights = np.power(np.clip(weights, 0.0, None), cfg["alpha"])
        proposal_weights = np.where(np.isfinite(proposal_weights), proposal_weights, 0.0)
        (
            _sample_counts,
            _covariance,
            _eigvals,
            _eigvecs,
            _principal_direction,
            local_spread,
            anisotropy,
            _density_gradient_proxy,
            _curvature_proxy,
            centroid,
            drift,
            drift_mag,
            geometry_stress,
        ) = _mesh_moments(
            generators,
            samples,
            weights,
            ownership,
            proposal_weights=proposal_weights,
        )
        mass_stress = mesh_stress_score(current_mesh.densities, anisotropy, drift_mag)

        finite_spread = np.isfinite(local_spread) & (local_spread > 0.0)
        step_limit = np.where(
            finite_spread,
            cfg["drift_clip"] * local_spread,
            cfg["drift_clip"] * 0.05 * _domain_scale(xlim, ylim),
        )
        move = centroid - generators
        finite_move = np.all(np.isfinite(move), axis=1)
        move = np.where(finite_move[:, None], move, 0.0)
        move_norm = np.linalg.norm(move, axis=1)
        clipped = np.ones_like(move_norm)
        valid = np.isfinite(move_norm) & (move_norm > 0.0)
        clipped[valid] = np.minimum(1.0, step_limit[valid] / move_norm[valid])
        move = move * clipped[:, None]
        updated = generators + cfg["eta"] * move
        updated = _clip_generators_to_domain(updated, xlim, ylim)

        moved_ownership = ownership
        if updated.shape[0] == generators.shape[0]:
            moved_diag: dict[str, Any] = {}
            if cfg["anisotropic"] and current_mesh.covariance.shape[0] == updated.shape[0]:
                moved_ownership = _anisotropic_ownership(
                    samples,
                    updated,
                    current_mesh.covariance,
                    covariance_floor=cfg["covariance_floor"],
                    anisotropy_cap=cfg["anisotropy_cap"],
                    candidate_k=cfg["candidate_k"],
                    batch_size=cfg["ownership_batch_size"],
                    diagnostics=moved_diag,
                )
            else:
                moved_ownership = assign_nearest_ownership(samples, updated)
                if cfg["anisotropic"]:
                    moved_diag["anisotropic_fallback"] = "covariance_shape"
                    moved_diag["anisotropic_fallback_count"] = int(samples.shape[0])
            anisotropic_fallback_total += int(moved_diag.get("anisotropic_fallback_count", 0))
        moved_mesh = _build_mesh_state(
            updated,
            samples,
            weights,
            ownership=moved_ownership,
            xlim=xlim,
            ylim=ylim,
            include_polygons=False,
            strict_geometry=False,
            geometry_space=geometry_space,
            compute_areas=False,
        )

        updated, reseed_count = _reseed_empty_generators(
            updated,
            moved_mesh,
            samples,
            weights,
            cfg=cfg,
            rng=rng,
            xlim=xlim,
            ylim=ylim,
        )
        updated, update_counts = _split_merge_generators(
            updated,
            moved_mesh,
            cfg=cfg,
            rng=rng,
            xlim=xlim,
            ylim=ylim,
        )
        updated = _clip_generators_to_domain(updated, xlim, ylim)
        totals["splits"] += int(update_counts.get("splits", 0))
        totals["merges"] += int(update_counts.get("merges", 0))
        totals["reseed"] += int(reseed_count)

        if cfg["record_history"]:
            trace.append(
                {
                    "iteration": int(iteration),
                    "generator_count": int(generators.shape[0]),
                    "mass_stress_mean": float(np.nanmean(mass_stress)) if np.isfinite(mass_stress).any() else float("nan"),
                    "geometry_stress_mean": float(np.nanmean(geometry_stress)) if np.isfinite(geometry_stress).any() else float("nan"),
                    "max_drift": float(np.nanmax(drift_mag)) if np.isfinite(drift_mag).any() else float("nan"),
                    "mean_anisotropy": float(np.nanmean(anisotropy)) if np.isfinite(anisotropy).any() else float("nan"),
                    "splits": int(update_counts.get("splits", 0)),
                    "merges": int(update_counts.get("merges", 0)),
                    "reseed": int(reseed_count),
                    "mass_geometry_stress_corr": _safe_corrcoef(mass_stress, geometry_stress),
                    "anisotropic_fallback": ownership_diag.get("anisotropic_fallback", "not_used"),
                    "anisotropic_fallback_count": int(ownership_diag.get("anisotropic_fallback_count", 0)),
                }
            )

        completed += 1
        stable = (
            updated.shape[0] == generators.shape[0]
            and np.allclose(updated, generators, atol=1.0e-8, rtol=1.0e-8)
            and int(update_counts.get("splits", 0)) == 0
            and int(update_counts.get("merges", 0)) == 0
            and int(reseed_count) == 0
        )
        generators = updated
        if cfg["record_history"]:
            generator_history.append(generators.copy())
        current_mesh = _build_mesh_state(
            generators,
            samples,
            weights,
            ownership=assign_nearest_ownership(samples, generators),
            xlim=xlim,
            ylim=ylim,
            include_polygons=False,
            strict_geometry=False,
            geometry_space=geometry_space,
        )
        if stable:
            break

    if cfg["anisotropic"] and current_mesh.covariance.shape[0] == generators.shape[0]:
        final_diag: dict[str, Any] = {}
        final_ownership = _anisotropic_ownership(
            samples,
            generators,
            current_mesh.covariance,
            covariance_floor=cfg["covariance_floor"],
            anisotropy_cap=cfg["anisotropy_cap"],
            candidate_k=cfg["candidate_k"],
            batch_size=cfg["ownership_batch_size"],
            diagnostics=final_diag,
        )
        anisotropic_fallback_total += int(final_diag.get("anisotropic_fallback_count", 0))
    else:
        final_ownership = assign_nearest_ownership(samples, generators)
        final_diag = {"anisotropic_fallback": "covariance_shape", "anisotropic_fallback_count": int(samples.shape[0])} if cfg["anisotropic"] else {}
        anisotropic_fallback_total += int(final_diag.get("anisotropic_fallback_count", 0))
    final_mesh = _build_mesh_state(
        generators,
        samples,
        weights,
        ownership=final_ownership,
        xlim=xlim,
        ylim=ylim,
        include_polygons=include_polygons,
        strict_geometry=strict_geometry,
        geometry_space=geometry_space,
    )
    final_mesh.diagnostics.update(
        {
            "refinement_enabled": True,
            "refinement_iterations_requested": int(cfg["iterations"]),
            "refinement_iterations_completed": int(completed),
            "refinement_splits": int(totals["splits"]),
            "refinement_merges": int(totals["merges"]),
            "refinement_reseeds": int(totals["reseed"]),
            "anisotropic_fallback": final_diag.get("anisotropic_fallback", "not_used"),
            "anisotropic_fallback_count": int(anisotropic_fallback_total),
        }
    )
    if cfg["record_history"]:
        final_mesh.diagnostics["refinement_trace"] = trace
        final_mesh.diagnostics["generator_history"] = generator_history
    return final_mesh


def build_posterior_mesh(
    generators: np.ndarray,
    samples: np.ndarray,
    weights: np.ndarray,
    *,
    ownership: Optional[np.ndarray] = None,
    xlim: Tuple[float, float] = (0.0, 1.0),
    ylim: Tuple[float, float] = (0.0, 1.0),
    include_polygons: bool = False,
    strict_geometry: bool = False,
    geometry_space: str = "normalized",
    refinement: Optional[dict[str, Any]] = None,
) -> PosteriorMesh:
    mesh = _build_mesh_state(
        generators,
        samples,
        weights,
        ownership=ownership,
        xlim=xlim,
        ylim=ylim,
        include_polygons=include_polygons,
        strict_geometry=strict_geometry,
        geometry_space=geometry_space,
    )
    cfg = _normalize_refinement_config(refinement, xlim=xlim, ylim=ylim)
    if cfg["enabled"] and cfg["iterations"] > 0:
        mesh = refine_posterior_mesh(
            mesh,
            samples,
            weights,
            xlim=xlim,
            ylim=ylim,
            refinement=cfg,
            include_polygons=include_polygons,
            strict_geometry=strict_geometry,
            geometry_space=geometry_space,
        )
    return mesh

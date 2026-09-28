"""Adaptive empirical-CDF anchors and a local, monotone C2 quintic interpolant.

No bins, kernels, or nonlinear solver. The quartic PDF is represented in the
Bernstein basis; nonnegative coefficients certify nonnegativity on each whole
interval, rather than only at sampled output points.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
from scipy.interpolate import BPoly, CubicSpline, PchipInterpolator

from ..distribution_1d_config import AnchorSettings


def monotone_c2(x: np.ndarray, y: np.ndarray) -> BPoly:
    """Exact C2 Hermite quintics with locally limited slopes and curvatures.

    On an interval of width h and secant m, the quartic derivative's Bernstein
    coefficients are [d0, d0+h*a0/4, 5*m-2*(d0+d1)+h*(a1-a0)/4,
    d1-h*a1/4, d1]. Limit curvatures first, then shrink each node's (d,a)
    together so all five coefficients are nonnegative. Each shared node has
    one value, slope, and curvature: the assembled CDF is C2 and PDF is C1.
    The support endpoints have zero slope/curvature, allowing constant CDF
    extension and zero PDF extension without a boundary derivative jump.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or len(x) < 2 or y.shape != x.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("monotone_c2 needs matching finite 1D anchor arrays")
    h, mass = np.diff(x), np.diff(y)
    if np.any(h <= 0) or np.any(mass < 0):
        raise ValueError("monotone_c2 requires strictly increasing x and nondecreasing CDF anchors")
    # A clamped cubic estimates smooth slopes/curvatures in linear time. Its
    # unconstrained interpolant is never returned: the quintic limiter below
    # certifies monotonicity even where the cubic estimate would overshoot.
    seed = CubicSpline(x, y, bc_type=((1, 0.0), (1, 0.0)))
    slopes = np.maximum(seed(x, 1), 0.0)
    slopes[[0, -1]] = 0.0
    curvature = seed(x, 2)
    # Adjacent interval constraints for the first/last PDF control coefficients.
    lower, upper = np.full(len(x), -np.inf), np.full(len(x), np.inf)
    lower[:-1] = -4 * slopes[:-1] / h
    upper[1:] = 4 * slopes[1:] / h
    curvature = np.clip(curvature, lower, upper)
    curvature[[0, -1]] = 0.0
    # Both contributions are nonnegative after clipping. Reducing either node
    # cannot violate another interval's constraint, so one local pass suffices.
    left = 2 * slopes[:-1] + h * curvature[:-1] / 4
    right = 2 * slopes[1:] - h * curvature[1:] / 4
    available = 5 * mass / h
    total = left + right
    factors = np.ones(len(h))
    active = total > available
    factors[active] = (available[active] / total[active]) * (1 - 32 * np.finfo(float).eps)
    node_factors = np.ones(len(x))
    node_factors[:-1] = np.minimum(node_factors[:-1], factors)
    node_factors[1:] = np.minimum(node_factors[1:], factors)
    slopes *= node_factors
    curvature *= node_factors
    if not np.all(np.isfinite(slopes)) or not np.all(np.isfinite(curvature)):
        raise ValueError("monotone_c2 cannot resolve this anchor spacing; inspect support and score precision")
    return BPoly.from_derivatives(x, np.column_stack((y, slopes, curvature)), extrapolate=False)


def interpolate_cdf(x: np.ndarray, y: np.ndarray, method: str):
    if method == "monotone_c2":
        return monotone_c2(x, y)
    if method == "pchip":
        return PchipInterpolator(x, y, extrapolate=False)
    raise ValueError(f"unknown CDF interpolation {method!r}")


@dataclass(frozen=True)
class AnchorFit:
    curve: BPoly | PchipInterpolator
    indices: np.ndarray
    max_error: float
    stop_reason: str


def fit_anchors(x: np.ndarray, y: np.ndarray, config: AnchorSettings, interpolation: str) -> AnchorFit:
    """Select only real empirical nodes plus the supplied support endpoints.

    A bounded set of probability seeds starts the fit. Each pass evaluates
    residuals against all empirical knots once, then scans disjoint intervals
    and refines up to eight worst intervals. min_mass is the required observed
    cumulative mass on EACH side of a proposed new node; it stops the selector
    from chasing individual sample jumps. No output grid participates.
    """
    if config.method == "all":
        curve = interpolate_cdf(x, y, interpolation)
        return AnchorFit(curve, np.arange(len(x)), 0.0, "all")
    # Retain the first empirical value and the first empirical value reaching
    # probability one, so the fit preserves the observed onset and upper tail.
    last = min(int(np.searchsorted(y, 1.0)), len(x) - 1)
    selected = {0, min(1, len(x) - 1), last, len(x) - 1}
    seed_count = min(9, config.max_points - len(selected) + 1)
    for target in np.linspace(0, 1, seed_count)[1:-1]:
        index = min(int(np.searchsorted(y, target)), len(x) - 1)
        ordered = sorted(selected)
        slot = int(np.searchsorted(ordered, index))
        if index in selected or slot == 0 or slot == len(ordered):
            continue
        if min(y[index] - y[ordered[slot - 1]], y[ordered[slot]] - y[index]) >= config.min_mass:
            selected.add(index)
    while True:
        indices = np.array(sorted(selected), dtype=int)
        curve = interpolate_cdf(x[indices], y[indices], interpolation)
        residual = np.abs(curve(x) - y)
        error = float(np.max(residual))
        if error <= config.tolerance:
            return AnchorFit(curve, indices, error, "tolerance")
        if len(selected) >= config.max_points:
            return AnchorFit(curve, indices, error, "max_points")
        queue = []
        for lo, hi in zip(indices[:-1], indices[1:]):
            # Search only nodes with enough probability mass on both sides.
            start = max(lo + 1, int(np.searchsorted(y, y[lo] + config.min_mass, side="left")))
            end = min(hi, int(np.searchsorted(y, y[hi] - config.min_mass, side="right")))
            if start >= end:
                continue
            index = start + int(np.argmax(residual[start:end]))
            if residual[index] > config.tolerance:
                heappush(queue, (-float(residual[index]), index))
        if not queue:
            return AnchorFit(curve, indices, error, "min_mass")
        for _ in range(min(8, config.max_points - len(selected), len(queue))):
            _, index = heappop(queue)
            selected.add(index)

"""Raw weighted samples -> continuous CDF, optionally its analytic derivative.

PDF1D and CDF1D share the reconstruction. No histogram or sample-grid
renormalisation participates in fitting: the grid only evaluates the interpolant.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from ..distribution_1d_config import KINDS, settings
from ..memtrace import memtrace_checkpoint, memtrace_object_inventory
from ..utils.expression import eval_dataframe_expression
from .adaptive_cdf_runtime import fit_anchors

ALGORITHM_REVISION = "weighted-ecdf-adaptive-bernstein-c2-v3"


def _values(df: pd.DataFrame, expr: Any, logger=None) -> np.ndarray:
    values = np.asarray(eval_dataframe_expression(df, expr, logger=logger), dtype=float)
    if values.ndim == 0:
        return np.full(len(df), float(values))
    if values.shape != (len(df),):
        raise ValueError(f"expression {expr!r} must return one value per input row, got {values.shape}")
    return values


def _infer_limits(x: np.ndarray, scale: str) -> tuple[float, float]:
    unique = np.unique(x)
    if len(unique) < 2:
        raise ValueError("one unique sample cannot define a continuous support; set coordinates.x.lim")
    lo = unique[0] - 0.5 * (unique[1] - unique[0])
    hi = unique[-1] + 0.5 * (unique[-1] - unique[-2])
    if scale == "log":
        if unique[0] <= 0:
            raise ValueError("coordinates.x.scale: log requires positive samples")
        lo = max(lo, unique[0] * 0.5)
    if not np.isfinite(lo) or not np.isfinite(hi) or not lo < hi:
        raise ValueError("cannot infer finite support; set coordinates.x.lim")
    return float(lo), float(hi)


def empirical_knots(x: np.ndarray, weights: np.ndarray, limits: tuple[float, float]):
    """Inclusive empirical nodes plus support endpoints; return largest atom too."""
    lo, hi = limits
    order = np.argsort(x, kind="stable")
    x, weights = x[order], weights[order]
    unique, first = np.unique(x, return_index=True)
    # Rescaling before summation avoids overflow without changing probability.
    scaled = weights / weights.max()
    mass = np.add.reduceat(scaled, first)
    mass /= mass.sum()
    cumulative = np.minimum(np.cumsum(mass), 1.0)
    cumulative[-1] = 1.0
    interior = (unique > lo) & (unique < hi)
    knots = np.r_[lo, unique[interior], hi]
    probabilities = np.r_[0.0, cumulative[interior], 1.0]
    return knots, probabilities, float(mass.max())


def build_cdf(x: np.ndarray, weights: np.ndarray, limits: tuple[float, float]) -> PchipInterpolator:
    """Original all-anchor PCHIP path, useful for exact empirical-node comparisons.

    A sample exactly at lo cannot retain its jump with F(lo)=0; its mass is
    carried into the first interval. Other empirical nodes retain their CDF.
    Production reconstruction uses fit_anchors with the selected settings.
    """
    knots, probabilities, _ = empirical_knots(x, weights, limits)
    return PchipInterpolator(knots, probabilities, extrapolate=False)


def distribution_1d(df: pd.DataFrame, cfg: Mapping[str, Any], kind: str, logger=None) -> pd.DataFrame:
    if kind not in KINDS:
        raise ValueError(f"unknown distribution transform {kind!r}")
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"{kind} needs a nonempty pandas sample table")
    opts = settings(cfg)
    memtrace_checkpoint(logger, f"pipeline.{kind}.before", df)
    x = _values(df, opts.x["expr"], logger)
    weights = _values(df, opts.weight["expr"], logger) if opts.weight else np.ones(len(df))
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(weights)):
        raise ValueError(f"{kind} needs finite samples and weights; filter invalid rows first")
    if np.any(weights < 0):
        raise ValueError(f"{kind} needs nonnegative probability weights; signed event weights are not a CDF")
    work = pd.DataFrame({"x": x, "weight": weights})
    if opts.repeat is not None:
        if opts.repeat not in df.columns:
            raise KeyError(f"{kind} missing repeat column {opts.repeat!r}")
        if df[opts.repeat].isna().any():
            raise ValueError(f"{kind} repeat identifiers must not be missing")
        work["repeat"] = df[opts.repeat].to_numpy()
    positive = weights > 0
    if not positive.any():
        raise ValueError(f"{kind} weights sum to zero")
    limits = opts.limits or _infer_limits(x[positive], opts.scale)
    lo, hi = limits
    if np.any((x[positive] < lo) | (x[positive] > hi)):
        raise ValueError(f"{kind} coordinates.x.lim must contain all positive-weight samples; filter first to truncate")
    grid = np.geomspace(lo, hi, opts.grid) if opts.scale == "log" else np.linspace(lo, hi, opts.grid)
    memtrace_object_inventory(logger, f"pipeline.{kind}.samples", {"samples": work})
    curves = ("cdf", "pdf") if kind == "PDF1D" else ("cdf",)
    mean = {key: np.zeros(opts.grid) for key in curves}
    m2 = {key: np.zeros(opts.grid) for key in curves}
    groups = work.groupby("repeat", sort=False, observed=True) if opts.repeat else [(None, work)]
    n = 0
    boundary_mass = False
    diagnostics = []
    for label, block in groups:
        block = block.loc[block.weight > 0]
        if block.empty:
            raise ValueError(f"{kind} repeat {label!r} weights sum to zero")
        bx, bw = block.x.to_numpy(), block.weight.to_numpy()
        boundary_mass |= bool(np.any(bx == lo))
        knots, probabilities, largest_jump = empirical_knots(bx, bw, limits)
        fit = fit_anchors(knots, probabilities, opts.anchors, opts.interpolation)
        cdf = fit.curve
        diagnostics.append({
            "repeat": label.item() if isinstance(label, np.generic) else label,
            "n_anchors": len(fit.indices), "n_candidates": len(knots),
            "max_cdf_error": fit.max_error, "stop_reason": fit.stop_reason,
            "largest_empirical_jump": largest_jump,
        })
        if logger is not None:
            logger.debug(f"{kind} repeat={label!r}: anchors={len(fit.indices)}/{len(knots)}, "
                         f"max CDF node error={fit.max_error:.6g}, stopped={fit.stop_reason}")
            if opts.anchors.method == "adaptive" and fit.max_error > opts.anchors.tolerance:
                logger.warning(f"{kind} repeat={label!r}: CDF error {fit.max_error:.6g} exceeds "
                               f"tolerance {opts.anchors.tolerance:g}; stopped at {fit.stop_reason}. "
                               "Adjust anchors settings if finer CDF reconstruction is needed.")
            if opts.anchors.method == "adaptive" and largest_jump > max(2 * opts.anchors.tolerance, opts.anchors.min_mass):
                logger.warning(f"{kind} repeat={label!r}: empirical probability jump {largest_jump:.6g}; "
                               "a continuous density spreads this discrete mass over an interval")
        values = {"cdf": cdf(grid)}
        if kind == "PDF1D":
            pdf = cdf.derivative()(grid)
            tolerance = 128 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(pdf))))
            if np.min(pdf) < -tolerance:
                raise ValueError(f"{kind} produced a negative density beyond floating-point tolerance")
            values["pdf"] = np.maximum(pdf, 0.0)
        n += 1
        for key in curves:
            if not np.all(np.isfinite(values[key])):
                raise ValueError(f"{kind} produced nonfinite {key}; inspect sample spacing/support")
            delta = values[key] - mean[key]
            mean[key] += delta / n
            m2[key] += delta * (values[key] - mean[key])
    out = {opts.name: grid}
    for key in curves:
        out[key] = mean[key]
        out[f"{key}_std"] = np.sqrt(np.maximum(m2[key], 0.0) / (n - 1)) if n > 1 else np.full(opts.grid, np.nan)
    out["n_repeats"] = n
    table = pd.DataFrame(out)
    table.attrs["distribution_1d"] = {
        "algorithm": ALGORITHM_REVISION, "interpolation": opts.interpolation,
        "anchor_method": opts.anchors.method, "support": limits, "repeats": diagnostics,
    }
    if logger is not None:
        if boundary_mass:
            logger.warning(f"{kind}: mass at the lower support boundary is spread over the first CDF interval")
        logger.debug(f"{kind}: {len(df)} samples -> {opts.grid} points, {n} repeats, support={limits}, std ddof=1")
    memtrace_checkpoint(logger, f"pipeline.{kind}.after", table)
    return table

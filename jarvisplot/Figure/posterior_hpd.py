from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _axis_spacing(axis: np.ndarray, name: str) -> float:
    axis = np.asarray(axis, dtype=float)
    if axis.ndim == 2:
        if name == "x":
            vals = axis[0, :]
        else:
            vals = axis[:, 0]
    else:
        vals = axis.reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        raise ValueError("HPD contour levels require at least two finite grid coordinates per axis.")
    diffs = np.diff(vals)
    diffs = diffs[np.isfinite(diffs)]
    diffs = np.abs(diffs[diffs != 0])
    if diffs.size == 0:
        raise ValueError("HPD contour levels require a non-zero regular grid spacing.")
    return float(np.median(diffs))


def compute_hpd_contour_levels(
    density,
    x_grid,
    y_grid,
    masses: Sequence[float] = (0.6827, 0.9545),
) -> tuple[dict[float, float], dict[str, Any]]:
    """Compute highest-posterior-density contour thresholds from integrated mass.

    Density values are sanitized, normalized over the regular grid, sorted from
    high to low, and converted into thresholds whose enclosed probability mass
    meets or slightly exceeds each requested target mass.
    """
    p = np.asarray(density, dtype=float)
    if p.ndim != 2:
        raise ValueError("HPD contour levels require a 2D regular-grid density array.")
    p = np.where(np.isfinite(p), p, 0.0)
    p = np.maximum(p, 0.0)

    dx = _axis_spacing(np.asarray(x_grid, dtype=float), "x")
    dy = _axis_spacing(np.asarray(y_grid, dtype=float), "y")
    cell_area = dx * dy
    integral_before = float(np.sum(p) * cell_area)
    if not np.isfinite(integral_before) or integral_before <= 0:
        raise ValueError("HPD contour levels require a positive finite density integral.")
    p = p / integral_before

    mass_targets = [float(m) for m in masses]
    for mass in mass_targets:
        if not (0.0 <= mass <= 1.0):
            raise ValueError("HPD credible masses must be inside [0, 1].")

    vals_sorted = np.sort(p.ravel())[::-1]
    cum_mass = np.cumsum(vals_sorted * cell_area)
    levels: dict[float, float] = {}
    actual: dict[float, float] = {}
    for mass in mass_targets:
        idx = int(np.searchsorted(cum_mass, mass, side="left"))
        idx = min(max(idx, 0), vals_sorted.size - 1)
        level = float(vals_sorted[idx])
        levels[mass] = level
        actual[mass] = float(np.sum(p[p >= level]) * cell_area)

    diagnostics = {
        "integral_before": integral_before,
        "density_min": float(np.min(p)),
        "density_max": float(np.max(p)),
        "levels": dict(levels),
        "actual_masses": dict(actual),
        "grid_shape": tuple(int(v) for v in p.shape),
        "cell_area": float(cell_area),
    }
    return levels, diagnostics


def format_hpd_diagnostics(diagnostics: Mapping[str, Any]) -> str:
    levels = diagnostics.get("levels", {})
    actual = diagnostics.get("actual_masses", {})
    lines = [
        "Posterior HPD contour diagnostics:",
        f"\t integral_before \t-> {float(diagnostics.get('integral_before')):.17g}",
        f"\t density_min_after \t-> {float(diagnostics.get('density_min')):.17g}",
        f"\t density_max_after \t-> {float(diagnostics.get('density_max')):.17g}",
        f"\t grid_shape \t-> {diagnostics.get('grid_shape')}",
    ]
    for mass in sorted(levels):
        label = f"{100.0 * float(mass):.2f}%"
        lines.append(f"\t threshold_{label} \t-> {float(levels[mass]):.17g}")
        lines.append(f"\t actual_mass_{label} \t-> {float(actual.get(mass)):.17g}")
    return "\n".join(lines)


def prepare_hpd_contour_style(
    z_grid,
    x_grid,
    y_grid,
    style: dict,
    *,
    logger=None,
) -> dict:
    raw_mode = style.pop("contour_mode", None)
    if raw_mode is None:
        raw_mode = style.pop("mode", "")
    mode = str(raw_mode).strip().lower()
    if mode not in {"posterior_hpd", "hpd", "posterior"}:
        return style

    masses = style.pop("masses", (0.6827, 0.9545))
    labels = style.pop("labels", None)
    clabel = bool(style.pop("clabel", labels is not False))
    levels, diagnostics = compute_hpd_contour_levels(z_grid, x_grid, y_grid, masses=masses)
    plot_scale = float(diagnostics.get("integral_before", 1.0))
    plot_levels = {mass: float(level) * plot_scale for mass, level in levels.items()}

    mass_order = list(levels.keys())
    ordered = sorted(((mass, plot_levels[mass]) for mass in mass_order), key=lambda item: item[1])
    ordered_masses = [mass for mass, _ in ordered]
    style["levels"] = [level for _, level in ordered]

    def _reorder_sequence(key: str) -> None:
        val = style.get(key)
        if isinstance(val, (str, bytes)) or val is None:
            return
        try:
            seq = list(val)
        except Exception:
            return
        if len(seq) != len(mass_order):
            return
        by_mass = {mass: seq[i] for i, mass in enumerate(mass_order)}
        style[key] = [by_mass[mass] for mass in ordered_masses]

    for key in ("colors", "linestyles", "linewidths"):
        _reorder_sequence(key)

    if labels and clabel:
        label_seq = list(labels)
        if len(label_seq) == len(mass_order):
            by_mass = {mass: str(label_seq[i]) for i, mass in enumerate(mass_order)}
            style["_hpd_label_map"] = {
                plot_levels[mass]: by_mass[mass] for mass in ordered_masses
            }

    msg = format_hpd_diagnostics(diagnostics)
    if logger is not None:
        try:
            logger.bind(show_info=True).info(msg)
        except Exception:
            try:
                logger.info(msg)
            except Exception:
                pass
    return style

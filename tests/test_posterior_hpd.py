from __future__ import annotations

import numpy as np
import pytest

from jarvisplot.Figure.posterior_hpd import (
    compute_hpd_contour_levels,
    prepare_hpd_contour_style,
)
from jarvisplot.Figure.preprocessor import DataPreprocessor


def _mesh(n=161, lim=5.0):
    x = np.linspace(-lim, lim, n)
    y = np.linspace(-lim, lim, n)
    return np.meshgrid(x, y, indexing="xy")


def test_hpd_levels_for_normalized_gaussian_enclose_requested_mass():
    x, y = _mesh()
    density = np.exp(-0.5 * (x**2 + y**2)) / (2.0 * np.pi)

    levels, diag = compute_hpd_contour_levels(density, x, y)

    assert diag["actual_masses"][0.6827] == pytest.approx(0.6827, abs=0.015)
    assert diag["actual_masses"][0.9545] == pytest.approx(0.9545, abs=0.015)
    assert levels[0.6827] > levels[0.9545]
    assert diag["grid_shape"] == density.shape


def test_hpd_levels_for_multimodal_density_can_be_disconnected():
    x, y = _mesh(n=121, lim=4.0)
    g1 = np.exp(-0.5 * ((x + 1.4) ** 2 + (y + 0.7) ** 2) / 0.25)
    g2 = 0.8 * np.exp(-0.5 * ((x - 1.2) ** 2 + (y - 0.8) ** 2) / 0.16)
    density = g1 + g2

    levels, diag = compute_hpd_contour_levels(density, x, y)

    assert levels[0.6827] > levels[0.9545]
    assert diag["actual_masses"][0.6827] >= 0.6827
    assert diag["actual_masses"][0.9545] >= 0.9545


def test_hpd_levels_treat_nan_inf_and_tiny_negative_values_as_zero():
    x, y = _mesh(n=41, lim=2.0)
    density = np.exp(-(x**2 + y**2))
    density[0, 0] = np.nan
    density[0, 1] = np.inf
    density[0, 2] = -1e-14

    levels, diag = compute_hpd_contour_levels(density, x, y)

    assert np.isfinite(levels[0.6827])
    assert diag["density_min"] >= 0.0
    assert diag["integral_before"] > 0.0


def test_hpd_levels_normalize_unnormalized_density_internally():
    x, y = _mesh(n=81, lim=3.0)
    density = 17.0 * np.exp(-(x**2 + y**2))

    _, diag = compute_hpd_contour_levels(density, x, y, masses=(0.5,))

    assert diag["actual_masses"][0.5] >= 0.5
    assert diag["density_max"] < np.max(density)


def test_hpd_levels_reject_invalid_inputs():
    x = np.linspace(0.0, 1.0, 8)
    y = np.linspace(0.0, 1.0, 8)
    density = np.zeros((8, 8))

    with pytest.raises(ValueError, match="positive finite density integral"):
        compute_hpd_contour_levels(density, x, y)

    density[3, 3] = 1.0
    with pytest.raises(ValueError, match="inside \\[0, 1\\]"):
        compute_hpd_contour_levels(density, x, y, masses=(1.2,))


def test_prepare_hpd_contour_style_sorts_levels_and_reorders_styles():
    x, y = _mesh(n=61, lim=3.0)
    density = np.exp(-0.5 * (x**2 + y**2))
    style = {
        "contour_mode": "posterior_hpd",
        "masses": [0.6827, 0.9545],
        "labels": ["1sigma", "2sigma"],
        "colors": ["red", "blue"],
        "linestyles": ["solid", "dashed"],
        "linewidths": [2.0, 1.0],
    }

    out = prepare_hpd_contour_style(density, x, y, style)

    assert out["levels"] == sorted(out["levels"])
    assert out["colors"] == ["blue", "red"]
    assert out["linestyles"] == ["dashed", "solid"]
    assert out["linewidths"] == [1.0, 2.0]
    assert out["_hpd_label_map"][out["levels"][0]] == "2sigma"
    assert out["_hpd_label_map"][out["levels"][1]] == "1sigma"


def test_prepare_hpd_contour_style_draws_levels_in_original_density_units():
    x, y = _mesh(n=61, lim=3.0)
    density = 17.0 * np.exp(-0.5 * (x**2 + y**2))
    levels, diag = compute_hpd_contour_levels(density, x, y, masses=(0.6827, 0.9545))

    out = prepare_hpd_contour_style(
        density,
        x,
        y,
        {
            "contour_mode": "posterior_hpd",
            "masses": [0.6827, 0.9545],
            "labels": ["1sigma", "2sigma"],
        },
    )

    expected = sorted(float(levels[m]) * float(diag["integral_before"]) for m in levels)
    assert out["levels"] == pytest.approx(expected)


def test_prepare_hpd_contour_style_uses_logger_instead_of_stdout(capsys):
    x, y = _mesh(n=41, lim=2.0)
    density = np.exp(-0.5 * (x**2 + y**2))
    messages = []

    class Logger:
        def info(self, message):
            messages.append(message)

    prepare_hpd_contour_style(
        density,
        x,
        y,
        {"contour_mode": "posterior_hpd"},
        logger=Logger(),
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert messages
    assert "Posterior HPD contour diagnostics" in messages[0]


def test_contour_layer_demands_grid_metadata_for_shared_density_grid():
    dp = DataPreprocessor(context=None)
    demand = dp.layer_demand_columns(
        {
            "method": "contour",
            "coordinates": {
                "x": {"expr": "x"},
                "y": {"expr": "y"},
                "z": {"expr": "density_qvor"},
            },
            "style": {"contour_mode": "posterior_hpd"},
        }
    )

    assert {"x", "y", "density_qvor"} <= set(demand)
    assert {"__grid_ix__", "__grid_iy__", "__grid_nx__", "__grid_ny__"} <= set(demand)

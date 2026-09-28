from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jarvisplot.Figure.adaptive_cdf_runtime import fit_anchors, monotone_c2
from jarvisplot.Figure.distribution_1d_runtime import distribution_1d, empirical_knots
from jarvisplot.distribution_1d_config import AnchorSettings, settings


def _empirical(seed=421, n=50000):
    scores = np.random.default_rng(seed).beta(3, 6, n)
    x, y, _ = empirical_knots(scores, np.ones(n), (0, 1))
    return x, y


@pytest.mark.parametrize("seed", range(6))
def test_monotone_c2_certifies_full_intervals_and_preserves_values(seed):
    rng = np.random.default_rng(seed)
    x = np.r_[0, np.cumsum(rng.uniform(.01, 1, 30))]
    x /= x[-1]
    mass = rng.uniform(0, 1, 30)
    mass[::5] = 0  # Include flat intervals, not just strictly increasing CDFs.
    y = np.r_[0, np.cumsum(mass) / mass.sum()]
    y[-1] = 1
    cdf = monotone_c2(x, y)
    pdf = cdf.derivative()
    np.testing.assert_allclose(cdf(x), y, atol=1e-14)
    # Nonnegative Bernstein coefficients imply a nonnegative quartic PDF
    # everywhere on the interval, including between output grid points.
    assert np.min(pdf.c) >= -1e-11
    assert pdf.integrate(0, 1) == pytest.approx(1, abs=1e-13)
    for order in (1, 2):
        derivative = cdf.derivative(order)
        np.testing.assert_allclose(derivative.c[-1, :-1], derivative.c[0, 1:], atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(derivative([0, 1]), 0, atol=1e-9)


def test_monotone_c2_handles_extremely_uneven_anchor_spacing():
    x = np.array([0, 1e-8, .01, .1, .10001, .8, 1.])
    y = np.array([0, .001, .01, .02, .7, .99, 1.])
    cdf = monotone_c2(x, y)
    pdf = cdf.derivative()
    tolerance = 256 * np.finfo(float).eps * np.max(np.abs(pdf.c))
    assert np.min(pdf.c) >= -tolerance
    np.testing.assert_allclose(cdf(x), y, atol=1e-13)
    assert pdf.integrate(0, 1) == pytest.approx(1, abs=1e-13)


def test_adaptive_anchors_are_real_and_reduce_pdf_noise_without_losing_mass():
    x, y = _empirical()
    fit = fit_anchors(x, y, AnchorSettings(), "monotone_c2")
    exact = fit_anchors(x, y, AnchorSettings(method="all"), "pchip")
    assert 4 <= len(fit.indices) < 100
    np.testing.assert_allclose(fit.curve(x[fit.indices]), y[fit.indices], atol=1e-14)
    assert fit.max_error == pytest.approx(np.max(np.abs(fit.curve(x) - y)))
    assert fit.max_error < .006
    grid = np.linspace(0, 1, 2001)
    smooth_tv = np.abs(np.diff(fit.curve.derivative()(grid))).sum()
    raw_tv = np.abs(np.diff(exact.curve.derivative()(grid))).sum()
    assert smooth_tv < raw_tv * .01
    assert fit.curve.derivative().integrate(0, 1) == pytest.approx(1, abs=1e-13)


def test_tighter_target_refines_anchors_when_mass_floor_permits():
    x, y = _empirical(n=12000)
    loose = fit_anchors(x, y, AnchorSettings(tolerance=.02, min_mass=.0001), "monotone_c2")
    tight = fit_anchors(x, y, AnchorSettings(tolerance=.002, min_mass=.0001), "monotone_c2")
    assert len(tight.indices) > len(loose.indices)
    assert tight.max_error <= .002
    assert tight.stop_reason == "tolerance"


def test_budget_and_mass_floor_stop_refinement_with_measured_error():
    x, y = _empirical(n=2000)
    budget = fit_anchors(x, y, AnchorSettings(tolerance=1e-6, max_points=4), "monotone_c2")
    assert len(budget.indices) == 4
    assert budget.stop_reason == "max_points" and budget.max_error > 1e-6
    floor = fit_anchors(x, y, AnchorSettings(tolerance=1e-6, min_mass=.4), "monotone_c2")
    assert floor.stop_reason == "min_mass" and floor.max_error > 1e-6


def test_default_grid_density_does_not_change_anchors_or_reconstruction():
    scores = np.random.default_rng(32).beta(3, 6, 10000)
    df = pd.DataFrame({"score": scores})
    def run(grid):
        return distribution_1d(df, {"coordinates": {"x": {"expr": "score", "lim": [0, 1], "grid": grid}}}, "PDF1D")
    coarse, dense = run(101), run(201)
    assert coarse.attrs["distribution_1d"] == dense.attrs["distribution_1d"]
    np.testing.assert_allclose(coarse.pdf, dense.pdf[::2], atol=1e-14)
    np.testing.assert_allclose(coarse.cdf, dense.cdf[::2], atol=1e-14)


@pytest.mark.parametrize("method", ["pchip", "monotone_c2"])
def test_adaptive_weighted_repeat_statistics_are_computed_after_reconstruction(method):
    rng = np.random.default_rng(12)
    df = pd.DataFrame({"score": rng.beta(3, 6, 6000), "w": rng.uniform(.5, 2, 6000),
                       "repeat": np.repeat([1, 2, 3], 2000)})
    base = {"coordinates": {"x": {"expr": "score", "lim": [0, 1], "grid": 501}, "weight": "w"},
            "interpolation": method}
    out = distribution_1d(df, {**base, "repeat": "repeat"}, "PDF1D")
    singles = [distribution_1d(block, base, "PDF1D") for _, block in df.groupby("repeat")]
    for key in ("cdf", "pdf"):
        values = np.stack([single[key] for single in singles])
        np.testing.assert_allclose(out[key], values.mean(axis=0), atol=1e-13)
        np.testing.assert_allclose(out[key + "_std"], values.std(axis=0, ddof=1), atol=1e-13)
    assert len(out.attrs["distribution_1d"]["repeats"]) == 3


def test_atoms_and_unmet_tolerance_are_visible_in_diagnostics_and_warnings():
    messages = []
    logger = SimpleNamespace(debug=lambda *a: None, warning=messages.append)
    df = pd.DataFrame({"score": [.2, .21, .3, .9]})
    cfg = {"coordinates": {"x": {"expr": "score", "lim": [0, 1]}},
           "anchors": {"max_points": 4, "tolerance": 1e-6}}
    out = distribution_1d(df, cfg, "CDF1D", logger)
    assert any("probability jump" in message for message in messages)
    assert any("exceeds tolerance" in message for message in messages)
    diag = out.attrs["distribution_1d"]["repeats"][0]
    assert diag["largest_empirical_jump"] == .25
    assert diag["stop_reason"] == "max_points"
    assert "pdf" not in out


@pytest.mark.parametrize("change,match", [
    ({"anchors": {"method": "unknown"}}, "method"),
    ({"anchors": {"tolerance": 0}}, "tolerance"),
    ({"anchors": {"tolerance": float("nan")}}, "tolerance"),
    ({"anchors": {"min_mass": True}}, "min_mass"),
    ({"anchors": {"max_points": 3}}, "max_points"),
    ({"anchors": {"max_points": 5.5}}, "max_points"),
    ({"anchors": {"max_points": True}}, "max_points"),
    ({"anchors": {"bins": 50}}, "unknown keys"),
    ({"interpolation": "cubic"}, "interpolation"),
])
def test_closed_adaptive_configuration(change, match):
    with pytest.raises(ValueError, match=match):
        settings({"coordinates": {"x": "score"}, **change})

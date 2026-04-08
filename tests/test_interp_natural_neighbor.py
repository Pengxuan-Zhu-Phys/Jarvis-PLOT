from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from jarvisplot.Figure import layer_runtime as layer_runtime_mod
from jarvisplot.Figure.interp_natural_neighbor import (
    natural_neighbor_approx_interpolate,
    natural_neighbor_interpolate,
    resolve_backend,
)
from jarvisplot.Figure.interp_natural_neighbor_exact import natural_neighbor_exact_interpolate


def _logger():
    return SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )


def test_exact_backend_reproduces_core_values():
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    z = np.array([10.0, 20.0, 30.0, 40.0])
    X = np.array([[0.0, 1.0], [0.0, 1.0]])
    Y = np.array([[0.0, 0.0], [1.0, 1.0]])

    Z = natural_neighbor_exact_interpolate(x, y, z, X, Y)

    assert Z.shape == X.shape
    np.testing.assert_allclose(Z, np.array([[10.0, 20.0], [30.0, 40.0]]), rtol=0.0, atol=0.0)

    diag = natural_neighbor_exact_interpolate.last_diagnostics
    assert diag is not None
    assert diag.backend == "natural_neighbor"
    assert diag.implementation == "exact-sibson-2d"
    assert diag.exact_hits == 4
    assert diag.outside_hull == 0


def test_exact_backend_returns_nan_outside_convex_hull():
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    z = np.array([0.0, 1.0, 2.0])
    X = np.array([[1.25]])
    Y = np.array([[1.25]])

    Z = natural_neighbor_exact_interpolate(x, y, z, X, Y)

    assert np.isnan(Z[0, 0])
    diag = natural_neighbor_exact_interpolate.last_diagnostics
    assert diag is not None
    assert diag.outside_hull == 1


def test_exact_backend_strict_nan_propagation():
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    z = np.array([0.0, np.nan, 2.0])
    X = np.array([[0.25]])
    Y = np.array([[0.25]])

    Z = natural_neighbor_exact_interpolate(x, y, z, X, Y)

    assert np.isnan(Z[0, 0])
    diag = natural_neighbor_exact_interpolate.last_diagnostics
    assert diag is not None
    assert diag.masked_by_nan == 1


def test_exact_backend_locality_ignores_distant_non_neighbors():
    x = np.array([0.0, 1.0, 0.0, 10.0])
    y = np.array([0.0, 0.0, 1.0, 10.0])
    z1 = np.array([0.0, 1.0, 2.0, 123.0])
    z2 = np.array([0.0, 1.0, 2.0, -999.0])
    X = np.array([[0.25]])
    Y = np.array([[0.25]])

    v1 = natural_neighbor_exact_interpolate(x, y, z1, X, Y)
    v2 = natural_neighbor_exact_interpolate(x, y, z2, X, Y)

    assert np.isfinite(v1[0, 0])
    np.testing.assert_allclose(v1, v2, rtol=0.0, atol=0.0)


def test_exact_backend_linear_precision_at_symmetric_center():
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    z = x + y
    X = np.array([[0.5]])
    Y = np.array([[0.5]])

    Z = natural_neighbor_exact_interpolate(x, y, z, X, Y)

    assert Z.shape == (1, 1)
    assert Z[0, 0] == pytest.approx(1.0, rel=1e-10, abs=1e-10)


def test_render_layer_uses_exact_natural_neighbor_backend(monkeypatch):
    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0],
            "z": [10.0, 20.0, 30.0, 40.0],
        }
    )

    fig = SimpleNamespace(
        logger=_logger(),
        style={"contourf": {"interp": {"method": "natural_neighbor", "bin": 4}}},
        _ensure_pandas_data=lambda data, reason="render": data,
        _eval_series=lambda data, spec: np.asarray(data[spec["expr"]]),
    )

    ax = SimpleNamespace(
        _type="rect",
        get_xscale=lambda: "linear",
        get_yscale=lambda: "linear",
        get_xlim=lambda: (0.0, 1.0),
        get_ylim=lambda: (0.0, 1.0),
    )

    method_calls = {}

    def dummy_method(*args, **kwargs):
        method_calls["arg_count"] = len(args)
        method_calls["kwargs"] = dict(kwargs)
        method_calls["shapes"] = [np.asarray(arg).shape for arg in args]
        method_calls["z_grid"] = np.asarray(args[2], dtype=float)
        return "ok"

    monkeypatch.setattr(layer_runtime_mod, "resolve_callable", lambda *args, **kwargs: (dummy_method, None))
    monkeypatch.setattr(layer_runtime_mod, "collect_and_attach_colorbar", lambda *args, **kwargs: args[1])

    out = layer_runtime_mod.render_layer(
        fig,
        ax,
        {
            "name": "natural-neighbor-layer",
            "method": "contourf",
            "data": df,
            "coor": {"x": {"expr": "x"}, "y": {"expr": "y"}, "z": {"expr": "z"}},
            "style": {},
        },
    )

    assert out == "ok"
    assert method_calls["arg_count"] == 3
    assert method_calls["shapes"] == [(4, 4), (4, 4), (4, 4)]
    assert "interp" not in method_calls["kwargs"]
    assert np.isfinite(method_calls["z_grid"]).any()


def test_exact_and_approx_backends_are_distinct():
    exact = resolve_backend("natural_neighbor")
    approx = resolve_backend("natural_neighbor_approx")

    assert exact is natural_neighbor_exact_interpolate
    assert approx is natural_neighbor_approx_interpolate
    assert exact is not approx


def test_public_wrapper_routes_to_exact_backend():
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    z = np.array([0.0, 1.0, 2.0])
    X = np.array([[0.25]])
    Y = np.array([[0.25]])

    Z = natural_neighbor_interpolate(x, y, z, X, Y)

    assert Z.shape == (1, 1)
    assert np.isfinite(Z[0, 0])
    diag = natural_neighbor_interpolate.last_diagnostics
    assert diag is not None
    assert diag.implementation == "exact-sibson-2d"

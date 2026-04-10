from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.contour import QuadContourSet
from matplotlib.collections import QuadMesh

import jarvisplot.Figure.adapters_rect as adapters_rect_mod
import jarvisplot.Figure.layer_runtime as layer_runtime_mod
from jarvisplot.Figure.adapters_rect import StdAxesAdapter
from jarvisplot.Figure.adapters_ternary import TernaryAxesAdapter
from jarvisplot.Figure.method_registry import resolve_method


def test_jpcontour_methods_are_exposed_on_the_adapter_classes():
    assert hasattr(StdAxesAdapter, "jpcontour")
    assert hasattr(StdAxesAdapter, "jpcontourf")
    assert hasattr(StdAxesAdapter, "jpfield")
    assert hasattr(TernaryAxesAdapter, "jpcontour")
    assert hasattr(TernaryAxesAdapter, "jpcontourf")
    assert hasattr(TernaryAxesAdapter, "jpfield")


def test_jpcontour_method_keys_are_registered_for_rect_and_tri_axes():
    assert resolve_method("jpcontour", axes_type="rect") == ("jpcontour", None)
    assert resolve_method("jpcontourf", axes_type="rect") == ("jpcontourf", None)
    assert resolve_method("jpfield", axes_type="rect") == ("jpfield", None)
    assert resolve_method("jpcontour", axes_type="tri") == ("jpcontour", None)
    assert resolve_method("jpcontourf", axes_type="tri") == ("jpcontourf", None)
    assert resolve_method("jpfield", axes_type="tri") == ("jpfield", None)


def test_jpcontour_and_jpcontourf_return_quadcontoursets():
    fig, ax = plt.subplots()
    adapter = StdAxesAdapter(ax)

    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    z = np.array([0.0, 1.0, 1.0, 2.0])

    contour = adapter.jpcontour(x, y, z, levels=4, colors="black")
    contourf = adapter.jpcontourf(x, y, z, levels=4, cmap="viridis")

    assert isinstance(contour, QuadContourSet)
    assert isinstance(contourf, QuadContourSet)
    plt.close(fig)


def test_jpfield_returns_quadmesh():
    fig, ax = plt.subplots()
    adapter = StdAxesAdapter(ax)

    def fake_backend(
        x,
        y,
        z,
        X,
        Y,
        *,
        nan_policy="strict",
        diagnostics=False,
        backend_options=None,
    ):
        Z = np.asarray(X + Y, dtype=float)
        return Z

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(adapters_rect_mod, "resolve_backend", lambda name: fake_backend)

    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    z = np.array([0.0, 1.0, 1.0, 2.0])

    mesh = adapter.jpfield(x, y, z, cmap="viridis")

    assert isinstance(mesh, QuadMesh)
    monkeypatch.undo()
    plt.close(fig)


@pytest.mark.parametrize(
    "method_name, mpl_name",
    [("jpcontour", "contour"), ("jpcontourf", "contourf"), ("jpfield", "pcolormesh")],
)
def test_default_jp_sample_grid_is_500_when_bin_is_omitted(monkeypatch, method_name, mpl_name):
    fig, ax = plt.subplots()
    adapter = StdAxesAdapter(ax)

    backend_calls = {}

    def fake_backend(
        x,
        y,
        z,
        X,
        Y,
        *,
        nan_policy="strict",
        diagnostics=False,
        backend_options=None,
    ):
        backend_calls["grid_shape"] = X.shape
        return np.asarray(X + Y, dtype=float)

    monkeypatch.setattr(adapters_rect_mod, "resolve_backend", lambda name: fake_backend)
    monkeypatch.setattr(adapter.ax, mpl_name, lambda *args, **kwargs: SimpleNamespace(tag=method_name))

    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    z = np.array([0.0, 1.0, 1.0, 2.0])

    if method_name == "jpfield":
        out = getattr(adapter, method_name)(x, y, z, cmap="viridis")
    else:
        out = getattr(adapter, method_name)(x, y, z, levels=3, cmap="viridis")

    assert out.tag == method_name
    assert backend_calls["grid_shape"] == (500, 500)
    plt.close(fig)


@pytest.mark.parametrize("method_name", ["jpcontour", "jpcontourf"])
def test_render_layer_dispatches_jpcontour_methods_with_scattered_inputs(monkeypatch, method_name):
    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0],
            "z": [0.0, 1.0, 1.0, 2.0],
        }
    )

    fig = SimpleNamespace(
        logger=SimpleNamespace(debug=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
        style={method_name: {"interp": {"method": "natural_neighbor", "bin": 4, "diagnostics": True}, "levels": 3}},
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
        return "ok"

    monkeypatch.setattr(layer_runtime_mod, "resolve_callable", lambda *args, **kwargs: (dummy_method, None))
    monkeypatch.setattr(layer_runtime_mod, "collect_and_attach_colorbar", lambda *args, **kwargs: args[1])

    out = layer_runtime_mod.render_layer(
        fig,
        ax,
        {
            "name": "jpcontour-layer",
            "method": method_name,
            "data": df,
            "coor": {"x": {"expr": "x"}, "y": {"expr": "y"}, "z": {"expr": "z"}},
            "style": {},
        },
    )

    assert out == "ok"
    assert method_calls["arg_count"] == 0
    assert np.allclose(np.asarray(method_calls["kwargs"]["x"], dtype=float), df["x"].to_numpy())
    assert np.allclose(np.asarray(method_calls["kwargs"]["y"], dtype=float), df["y"].to_numpy())
    assert np.allclose(np.asarray(method_calls["kwargs"]["z"], dtype=float), df["z"].to_numpy())
    assert "interp" not in method_calls["kwargs"]
    assert method_calls["kwargs"]["interp_method"] == "natural_neighbor"
    assert method_calls["kwargs"]["bin"] == 4
    assert method_calls["kwargs"]["levels"] == 3
    assert method_calls["kwargs"]["diagnostics"] is True


@pytest.mark.parametrize("method_name, mpl_name", [("jpcontour", "contour"), ("jpcontourf", "contourf")])
def test_jpcontour_strips_interp_kwargs_and_preserves_masked_nan_grid(monkeypatch, method_name, mpl_name):
    fig, ax = plt.subplots()
    adapter = StdAxesAdapter(ax)

    backend_calls = {}

    def fake_backend(
        x,
        y,
        z,
        X,
        Y,
        *,
        nan_policy="strict",
        diagnostics=False,
        backend_options=None,
    ):
        backend_calls["x"] = np.asarray(x, dtype=float)
        backend_calls["y"] = np.asarray(y, dtype=float)
        backend_calls["nan_policy"] = nan_policy
        backend_calls["diagnostics"] = diagnostics
        backend_calls["backend_options"] = backend_options
        backend_calls["grid_shape"] = X.shape
        backend_calls["x_range"] = (float(X.min()), float(X.max()))
        backend_calls["y_range"] = (float(Y.min()), float(Y.max()))
        Z = np.asarray(X + Y, dtype=float)
        Z[0, 0] = np.nan
        return Z

    monkeypatch.setattr(adapters_rect_mod, "resolve_backend", lambda name: fake_backend)

    mpl_calls = {}

    def recorder(*args, **kwargs):
        mpl_calls["args"] = args
        mpl_calls["kwargs"] = kwargs
        return SimpleNamespace(tag=method_name)

    monkeypatch.setattr(adapter.ax, mpl_name, recorder)

    x = np.array([0.0, 1.0, 2.0, 2.0])
    y = np.array([0.0, 0.5, 1.5, 2.0])
    z = np.array([0.0, 1.0, 1.0, 2.0])

    out = getattr(adapter, method_name)(
        x,
        y,
        z,
        levels=3,
        colors="k",
        linewidths=2.0,
        interp_method="natural_neighbor",
        bin=6,
        nx=4,
        ny=5,
        xlim=(0.0, 2.0),
        ylim=(0.0, 2.0),
        nan_policy="strict",
        diagnostics=True,
        backend_options={"vertex_tol": 1.0e-6},
    )

    assert out.tag == method_name
    assert backend_calls["nan_policy"] == "strict"
    assert backend_calls["diagnostics"] is True
    assert backend_calls["backend_options"] == {"vertex_tol": 1.0e-6}
    assert backend_calls["grid_shape"] == (5, 4)
    assert backend_calls["x_range"] == (0.0, 1.0)
    assert backend_calls["y_range"] == (0.0, 1.0)
    np.testing.assert_allclose(backend_calls["x"], np.array([0.0, 0.5, 1.0, 1.0]))
    np.testing.assert_allclose(backend_calls["y"], np.array([0.0, 0.25, 0.75, 1.0]))

    assert mpl_calls["kwargs"]["levels"] == 3
    assert mpl_calls["kwargs"]["colors"] == "k"
    assert mpl_calls["kwargs"]["linewidths"] == 2.0
    assert mpl_calls["kwargs"]["transform"] is adapter.ax.transAxes
    z_arg = mpl_calls["args"][2]
    assert np.ma.isMaskedArray(z_arg)
    mask = np.ma.getmaskarray(z_arg)
    assert mask.shape == (5, 4)
    assert bool(mask[0, 0]) is True
    assert not np.any(mask[1:, :])
    plt.close(fig)


def test_render_layer_contourf_defaults_to_500_grid_when_interp_bin_missing(monkeypatch):
    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0],
            "z": [0.0, 1.0, 1.0, 2.0],
        }
    )

    fig = SimpleNamespace(
        logger=SimpleNamespace(debug=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
        style={"contourf": {"interp": {"method": "natural_neighbor"}, "levels": 3}},
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

    backend_calls = {}

    def fake_backend(
        x,
        y,
        z,
        X,
        Y,
        *,
        nan_policy="strict",
        diagnostics=False,
        backend_options=None,
    ):
        backend_calls["grid_shape"] = X.shape
        return X + Y

    method_calls = {}

    def dummy_method(*args, **kwargs):
        method_calls["kwargs"] = dict(kwargs)
        return "ok"

    monkeypatch.setattr(layer_runtime_mod, "resolve_backend", lambda name: fake_backend)
    monkeypatch.setattr(layer_runtime_mod, "resolve_callable", lambda *args, **kwargs: (dummy_method, None))
    monkeypatch.setattr(layer_runtime_mod, "collect_and_attach_colorbar", lambda *args, **kwargs: args[1])

    out = layer_runtime_mod.render_layer(
        fig,
        ax,
        {
            "name": "contourf-layer",
            "method": "contourf",
            "data": df,
            "coor": {"x": {"expr": "x"}, "y": {"expr": "y"}, "z": {"expr": "z"}},
            "style": {},
        },
    )

    assert out == "ok"
    assert backend_calls["grid_shape"] == (500, 500)
    assert method_calls["kwargs"]["levels"] == 3


def test_jpfield_strips_interp_kwargs_and_preserves_masked_nan_grid(monkeypatch):
    fig, ax = plt.subplots()
    adapter = StdAxesAdapter(ax)

    backend_calls = {}

    def fake_backend(
        x,
        y,
        z,
        X,
        Y,
        *,
        nan_policy="strict",
        diagnostics=False,
        backend_options=None,
    ):
        backend_calls["x"] = np.asarray(x, dtype=float)
        backend_calls["y"] = np.asarray(y, dtype=float)
        backend_calls["nan_policy"] = nan_policy
        backend_calls["backend_options"] = backend_options
        backend_calls["grid_shape"] = X.shape
        backend_calls["x_range"] = (float(X.min()), float(X.max()))
        backend_calls["y_range"] = (float(Y.min()), float(Y.max()))
        Z = np.asarray(X + Y, dtype=float)
        Z[0, 0] = np.nan
        return Z

    def fake_resolve_backend(name):
        backend_calls["method"] = name
        return fake_backend

    monkeypatch.setattr(adapters_rect_mod, "resolve_backend", fake_resolve_backend)

    mpl_calls = {}

    def recorder(*args, **kwargs):
        mpl_calls["args"] = args
        mpl_calls["kwargs"] = kwargs
        return SimpleNamespace(tag="field")

    monkeypatch.setattr(adapter.ax, "pcolormesh", recorder)

    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    z = np.array([0.0, 1.0, 1.0, 2.0])

    out = adapter.jpfield(
        x,
        y,
        z,
        interp_method="natural_neighbor_approx",
        bin=6,
        nx=4,
        ny=5,
        xlim=(-1.0, 1.0),
        ylim=(-2.0, 2.0),
        nan_policy="strict",
        backend_options={"vertex_tol": 1.0e-6},
        cmap="magma",
        alpha=0.4,
        edgecolors="face",
        levels=7,
        colors="red",
        extend="both",
        linestyles="dashed",
    )

    assert out.tag == "field"
    assert backend_calls["method"] == "natural_neighbor_approx"
    assert backend_calls["nan_policy"] == "strict"
    assert backend_calls["backend_options"] == {"vertex_tol": 1.0e-6}
    assert backend_calls["grid_shape"] == (5, 4)
    np.testing.assert_allclose(backend_calls["x"], np.array([0.5, 1.0, 0.5, 1.0]))
    np.testing.assert_allclose(backend_calls["y"], np.array([0.5, 0.5, 0.75, 0.75]))
    assert backend_calls["x_range"] == (0.0, 1.0)
    assert backend_calls["y_range"] == (0.0, 1.0)

    assert mpl_calls["kwargs"]["shading"] == "flat"
    assert mpl_calls["kwargs"]["transform"] is adapter.ax.transAxes
    assert mpl_calls["kwargs"]["cmap"] == "magma"
    assert mpl_calls["kwargs"]["alpha"] == 0.4
    assert mpl_calls["kwargs"]["edgecolors"] == "face"
    assert mpl_calls["kwargs"]["linewidth"] == 0.0
    assert mpl_calls["kwargs"]["antialiased"] is False
    assert mpl_calls["kwargs"]["snap"] is True
    assert "interp_method" not in mpl_calls["kwargs"]
    assert "bin" not in mpl_calls["kwargs"]
    assert "nx" not in mpl_calls["kwargs"]
    assert "ny" not in mpl_calls["kwargs"]
    assert "xlim" not in mpl_calls["kwargs"]
    assert "ylim" not in mpl_calls["kwargs"]
    assert "nan_policy" not in mpl_calls["kwargs"]
    assert "backend_options" not in mpl_calls["kwargs"]
    assert "levels" not in mpl_calls["kwargs"]
    assert "colors" not in mpl_calls["kwargs"]
    assert "extend" not in mpl_calls["kwargs"]
    assert "linestyles" not in mpl_calls["kwargs"]

    x_arg = mpl_calls["args"][0]
    y_arg = mpl_calls["args"][1]
    z_arg = mpl_calls["args"][2]
    assert x_arg.shape == (6, 5)
    assert y_arg.shape == (6, 5)
    assert np.ma.isMaskedArray(z_arg)
    mask = np.ma.getmaskarray(z_arg)
    assert mask.shape == (5, 4)
    assert bool(mask[0, 0]) is True
    assert not np.any(mask[1:, :])
    plt.close(fig)


def test_jpfield_uses_bin_for_both_axes_when_nx_ny_missing(monkeypatch):
    fig, ax = plt.subplots()
    adapter = StdAxesAdapter(ax)

    backend_calls = {}

    def fake_backend(
        x,
        y,
        z,
        X,
        Y,
        *,
        nan_policy="strict",
        diagnostics=False,
        backend_options=None,
    ):
        backend_calls["grid_shape"] = X.shape
        return np.asarray(X + Y, dtype=float)

    monkeypatch.setattr(adapters_rect_mod, "resolve_backend", lambda name: fake_backend)
    monkeypatch.setattr(adapter.ax, "pcolormesh", lambda *args, **kwargs: SimpleNamespace(tag="field"))

    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    z = np.array([0.0, 1.0, 1.0, 2.0])

    out = adapter.jpfield(x, y, z, bin=7)

    assert out.tag == "field"
    assert backend_calls["grid_shape"] == (7, 7)
    plt.close(fig)


def test_jpfield_defaults_to_current_axes_limits_when_xlim_omitted(monkeypatch):
    fig, ax = plt.subplots()
    adapter = StdAxesAdapter(ax)
    adapter.ax.set_xscale("log")
    adapter.ax.set_xlim(0.1, 5.0)
    adapter.ax.set_ylim(0.0, 5.0)

    backend_calls = {}

    def fake_backend(
        x,
        y,
        z,
        X,
        Y,
        *,
        nan_policy="strict",
        diagnostics=False,
        backend_options=None,
    ):
        backend_calls["x_range"] = (float(X.min()), float(X.max()))
        backend_calls["y_range"] = (float(Y.min()), float(Y.max()))
        return np.asarray(X + Y, dtype=float)

    monkeypatch.setattr(adapters_rect_mod, "resolve_backend", lambda name: fake_backend)
    monkeypatch.setattr(adapter.ax, "pcolormesh", lambda *args, **kwargs: SimpleNamespace(tag="field"))

    x = np.array([1.0e-6, 1.0e-3, 1.0e-1, 1.0e+0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    z = np.array([0.0, 1.0, 2.0, 3.0])

    out = adapter.jpfield(x, y, z, bin=5)

    assert out.tag == "field"
    assert backend_calls["x_range"][0] == pytest.approx(0.0)
    assert backend_calls["x_range"][1] == pytest.approx(1.0)
    assert backend_calls["y_range"][0] == pytest.approx(0.0)
    assert backend_calls["y_range"][1] == pytest.approx(1.0)
    plt.close(fig)


def test_ternary_jpcontourf_accepts_left_right_bottom(monkeypatch):
    fig, ax = plt.subplots()
    adapter = TernaryAxesAdapter(ax)

    backend_calls = {}

    def fake_backend(
        x,
        y,
        z,
        X,
        Y,
        *,
        nan_policy="strict",
        diagnostics=False,
        backend_options=None,
    ):
        backend_calls["x"] = np.asarray(x, dtype=float)
        backend_calls["y"] = np.asarray(y, dtype=float)
        backend_calls["z"] = np.asarray(z, dtype=float)
        return np.asarray(X + Y, dtype=float)

    monkeypatch.setattr(adapters_rect_mod, "resolve_backend", lambda name: fake_backend)

    mpl_calls = {}

    def recorder(*args, **kwargs):
        mpl_calls["args"] = args
        mpl_calls["kwargs"] = kwargs
        return SimpleNamespace(tag="contourf")

    monkeypatch.setattr(adapter.ax, "contourf", recorder)

    left = np.array([0.2, 0.3, 0.5])
    right = np.array([0.3, 0.4, 0.2])
    bottom = np.array([0.5, 0.3, 0.3])
    z = np.array([1.0, 2.0, 3.0])

    out = adapter.jpcontourf(left=left, right=right, bottom=bottom, z=z, levels=4)

    expected_x, expected_y = adapter._lbr_to_xy(left, right, bottom)
    assert out.tag == "contourf"
    np.testing.assert_allclose(backend_calls["x"], expected_x)
    np.testing.assert_allclose(backend_calls["y"], expected_y)
    np.testing.assert_allclose(backend_calls["z"], z)
    assert mpl_calls["kwargs"]["levels"] == 4
    assert mpl_calls["kwargs"]["transform"] is adapter.ax.transAxes
    plt.close(fig)


def test_ternary_jpfield_accepts_left_right_bottom(monkeypatch):
    fig, ax = plt.subplots()
    adapter = TernaryAxesAdapter(ax)

    backend_calls = {}

    def fake_backend(
        x,
        y,
        z,
        X,
        Y,
        *,
        nan_policy="strict",
        diagnostics=False,
        backend_options=None,
    ):
        backend_calls["x"] = np.asarray(x, dtype=float)
        backend_calls["y"] = np.asarray(y, dtype=float)
        backend_calls["z"] = np.asarray(z, dtype=float)
        return np.asarray(X + Y, dtype=float)

    monkeypatch.setattr(adapters_rect_mod, "resolve_backend", lambda name: fake_backend)

    mpl_calls = {}

    def recorder(*args, **kwargs):
        mpl_calls["args"] = args
        mpl_calls["kwargs"] = kwargs
        return SimpleNamespace(tag="field")

    monkeypatch.setattr(adapter.ax, "pcolormesh", recorder)

    left = np.array([0.2, 0.3, 0.5])
    right = np.array([0.3, 0.4, 0.2])
    bottom = np.array([0.5, 0.3, 0.3])
    z = np.array([1.0, 2.0, 3.0])

    out = adapter.jpfield(left=left, right=right, bottom=bottom, z=z, bin=4)

    expected_x, expected_y = adapter._lbr_to_xy(left, right, bottom)
    assert out.tag == "field"
    np.testing.assert_allclose(backend_calls["x"], expected_x)
    np.testing.assert_allclose(backend_calls["y"], expected_y)
    np.testing.assert_allclose(backend_calls["z"], z)
    assert mpl_calls["kwargs"]["transform"] is adapter.ax.transAxes
    assert mpl_calls["kwargs"]["shading"] == "flat"
    plt.close(fig)


def test_render_layer_dispatches_jpfield_with_scattered_inputs(monkeypatch):
    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0],
            "z": [0.0, 1.0, 1.0, 2.0],
        }
    )

    fig = SimpleNamespace(
        logger=SimpleNamespace(debug=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
        style={"jpfield": {"interp": {"method": "natural_neighbor_approx", "bin": 4}, "alpha": 0.5}},
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
        return "ok"

    monkeypatch.setattr(layer_runtime_mod, "resolve_callable", lambda *args, **kwargs: (dummy_method, None))
    monkeypatch.setattr(layer_runtime_mod, "collect_and_attach_colorbar", lambda *args, **kwargs: args[1])

    out = layer_runtime_mod.render_layer(
        fig,
        ax,
        {
            "name": "jpfield-layer",
            "method": "jpfield",
            "data": df,
            "coor": {"x": {"expr": "x"}, "y": {"expr": "y"}, "z": {"expr": "z"}},
            "style": {},
        },
    )

    assert out == "ok"
    assert method_calls["arg_count"] == 0
    assert np.allclose(np.asarray(method_calls["kwargs"]["x"], dtype=float), df["x"].to_numpy())
    assert np.allclose(np.asarray(method_calls["kwargs"]["y"], dtype=float), df["y"].to_numpy())
    assert np.allclose(np.asarray(method_calls["kwargs"]["z"], dtype=float), df["z"].to_numpy())
    assert "interp" not in method_calls["kwargs"]
    assert method_calls["kwargs"]["interp_method"] == "natural_neighbor_approx"
    assert method_calls["kwargs"]["bin"] == 4
    assert method_calls["kwargs"]["alpha"] == 0.5


def test_jpcontour_strips_interpolation_kwargs_from_mpl_call(monkeypatch):
    fig, ax = plt.subplots()
    adapter = StdAxesAdapter(ax)

    backend_seen = {}

    def fake_backend(
        x,
        y,
        z,
        X,
        Y,
        *,
        nan_policy="strict",
        diagnostics=False,
        backend_options=None,
    ):
        backend_seen["nan_policy"] = nan_policy
        backend_seen["backend_options"] = backend_options
        return X + Y

    monkeypatch.setattr(adapters_rect_mod, "resolve_backend", lambda name: fake_backend)

    mpl_calls = {}

    def recorder(*args, **kwargs):
        mpl_calls["args"] = args
        mpl_calls["kwargs"] = kwargs
        return SimpleNamespace(tag="contour")

    monkeypatch.setattr(adapter.ax, "contour", recorder)

    out = adapter.jpcontour(
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 2.0]),
        levels=4,
        colors="red",
        interp_method="natural_neighbor",
        bin=8,
        nx=3,
        ny=4,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        nan_policy="strict",
        backend_options={"geometry_tol": 1.0e-12},
    )

    assert out.tag == "contour"
    assert backend_seen["nan_policy"] == "strict"
    assert backend_seen["backend_options"] == {"geometry_tol": 1.0e-12}
    assert "interp_method" not in mpl_calls["kwargs"]
    assert "bin" not in mpl_calls["kwargs"]
    assert "nx" not in mpl_calls["kwargs"]
    assert "ny" not in mpl_calls["kwargs"]
    assert "xlim" not in mpl_calls["kwargs"]
    assert "ylim" not in mpl_calls["kwargs"]
    assert "nan_policy" not in mpl_calls["kwargs"]
    assert "backend_options" not in mpl_calls["kwargs"]
    assert mpl_calls["kwargs"]["transform"] is adapter.ax.transAxes
    plt.close(fig)

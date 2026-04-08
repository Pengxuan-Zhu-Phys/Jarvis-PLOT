from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jarvisplot.Figure import layer_runtime as layer_runtime_mod
from jarvisplot.Figure import figure as figure_mod
from jarvisplot.Figure.colorbar_runtime import collect_and_attach_colorbar
from jarvisplot.Figure.config_runtime import apply_figure_config
from jarvisplot.Figure.figure import Figure
from jarvisplot.Figure.data_pipelines import DataContext, SharedContent
from jarvisplot.Figure.method_registry import resolve_method
from jarvisplot.Figure.preprocessor import DataPreprocessor
from jarvisplot.Figure.profile_runtime import grid_profile_mesh
from jarvisplot.core import _format_console_record
from jarvisplot.data_loader import JP_ROW_IDX


def _logger():
    return SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )


def test_style_family_single_token_uses_available_variant():
    fig = Figure()
    fig.logger = _logger()
    fig.jpstyles = {
        "a4paper_2x1": {
            "rect": {
                "Frame": {"figure": {"figsize": (1, 1)}, "axes": {}},
                "Style": {"marker": "o"},
            }
        }
    }

    fig.style = ["a4paper_2x1"]

    assert fig.frame["figure"]["figsize"] == (1, 1)
    assert fig.style["marker"] == "o"


def test_console_record_formatter_escapes_braces():
    class _Time:
        def __format__(self, spec):
            return "03-26 03:25:36.050"

    record = {
        "extra": {"module": "JarvisPLOT"},
        "message": "JarvisPLOT: colormaps registered: {'registered': ['x'], 'failed': []} <string>",
        "time": _Time(),
        "level": "DEBUG",
    }

    formatted = _format_console_record(record)

    assert "{{'registered': ['x'], 'failed': []}}" in formatted
    assert "\\<string>" in formatted


def test_apply_figure_config_uses_original_style_tokens_for_mode(monkeypatch):
    monkeypatch.setattr(Figure, "load_axes", lambda self: None)

    fig = Figure()
    fig.logger = _logger()
    fig.jpstyles = {
        "gambit_2x1": {
            "rectcmap": {
                "Frame": {"figure": {}, "axes": {}},
                "Style": {},
            }
        }
    }

    result = apply_figure_config(
        fig,
        {
            "name": "fig",
            "style": ["gambit_2x1", "rectcmap"],
            "frame": {"figure": {}, "axes": {}},
            "layers": [],
        },
    )

    assert result is True
    assert fig.mode == "gambit"
    assert fig._setup_status == "ok"
    assert fig._setup_error is None


def test_apply_figure_config_marks_disabled_figures():
    fig = Figure()
    fig.logger = _logger()

    result = apply_figure_config(
        fig,
        {
            "name": "fig",
            "enable": False,
        },
    )

    assert result is False
    assert fig._setup_status == "disabled"
    assert fig._setup_error is None


def test_apply_figure_config_marks_setup_failures(monkeypatch):
    monkeypatch.setattr(Figure, "load_axes", lambda self: None)

    fig = Figure()
    fig.logger = _logger()

    result = apply_figure_config(
        fig,
        {
            "name": "fig",
            "frame": {},
            "layers": [],
        },
    )

    assert result is False
    assert fig._setup_status == "failed"
    assert isinstance(fig._setup_error, Exception)


def test_figure_set_alias_removed():
    assert not hasattr(Figure, "set")


def test_colorbar_contract_uses_frame_color_config():
    class DummyColorbarAxis:
        def __init__(self):
            self._cb = {
                "cmap": None,
                "vmin": None,
                "vmax": None,
                "norm": None,
                "levels": None,
                "used": False,
            }

    fig = SimpleNamespace(
        axes={"axc": DummyColorbarAxis()},
        axc=DummyColorbarAxis(),
        frame={"axc": {"color": {"scale": "log", "cmap": "viridis", "vmin": 1, "vmax": 10}}},
        logger=_logger(),
    )
    fig.axes["axc"] = fig.axc

    out = collect_and_attach_colorbar(
        fig,
        style={},
        coor={"c": {"expr": "c"}},
        method_key="scatter",
        df=pd.DataFrame({"c": [1, 2, 5]}),
    )

    assert out["cmap"] == "viridis"
    assert fig.axc._cb["used"] is True
    assert fig.axc._cb["mode"] == "log"
    assert fig.axc._cb["vmin"] == 1.0
    assert fig.axc._cb["vmax"] == 10.0


def test_colorbar_attachment_skips_plain_scatter_without_color_channel():
    class DummyColorbarAxis:
        def __init__(self):
            self._cb = {
                "cmap": "viridis",
                "vmin": 1.0,
                "vmax": 10.0,
                "norm": mcolors.Normalize(vmin=1.0, vmax=10.0),
                "levels": None,
                "used": True,
            }

    fig = SimpleNamespace(
        axes={"axc": DummyColorbarAxis()},
        frame={"axc": {"color": {"scale": "linear", "cmap": "viridis"}}},
        logger=_logger(),
    )

    out = collect_and_attach_colorbar(
        fig,
        style={"marker": "."},
        coor={"x": {"expr": "x"}, "y": {"expr": "y"}},
        method_key="scatter",
        df=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
    )

    assert out == {"marker": "."}


def test_colorbar_attachment_accepts_nullable_numeric_color_channel():
    class DummyColorbarAxis:
        def __init__(self):
            self._cb = {
                "cmap": None,
                "vmin": None,
                "vmax": None,
                "norm": None,
                "levels": None,
                "used": False,
            }

    fig = SimpleNamespace(
        axes={"axc": DummyColorbarAxis()},
        frame={"axc": {"color": {"scale": "linear", "cmap": "viridis"}}},
        logger=_logger(),
    )

    out = collect_and_attach_colorbar(
        fig,
        style={"marker": ".", "c": [1, None, 3]},
        coor={"x": {"expr": "x"}, "y": {"expr": "y"}},
        method_key="scatter",
        df=pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
    )

    assert out["cmap"] == "viridis"
    assert isinstance(out["norm"], mcolors.Normalize)
    assert fig.axes["axc"]._cb["used"] is True
    assert fig.axes["axc"]._cb["vmin"] == 1.0
    assert fig.axes["axc"]._cb["vmax"] == 3.0


def test_colorbar_log_scale_uses_positive_subset_for_limits():
    fig = Figure()
    fig.logger = _logger()
    fig.frame = {
        "figure": {"figsize": (2, 2)},
        "axc": {
            "label": {"ylabel": ""},
            "ticks": {},
            "color": {"scale": "log", "cmap": "viridis"},
        },
    }
    fig.fig = plt.figure(figsize=(2, 2))
    fig.axes = {}
    fig.axc = {"rect": [0.1, 0.1, 0.2, 0.8]}

    out = collect_and_attach_colorbar(
        fig,
        style={},
        coor={"z": {"expr": "z"}},
        method_key="voronoi",
        df=pd.DataFrame({"z": [0.0, 1.0, 10.0]}),
    )

    fig._finalize_axc("axc")

    assert isinstance(out["norm"], mcolors.LogNorm)
    assert fig.axc._cb["mode"] == "log"
    assert fig.axc._cb["vmin"] == 1.0
    assert fig.axc._cb["vmax"] == 10.0
    assert fig.axc.get_yscale() == "log"


def test_colorbar_legacy_axis_scale_is_ignored_without_color_scale():
    fig = Figure()
    fig.logger = _logger()
    fig.frame = {
        "figure": {"figsize": (2, 2)},
        "axc": {
            "label": {"ylabel": ""},
            "ticks": {},
            "yscale": "log",
            "color": {"scale": "linear", "cmap": "viridis"},
        },
    }
    fig.fig = plt.figure(figsize=(2, 2))
    fig.axes = {}
    fig.axc = {"rect": [0.1, 0.1, 0.2, 0.8]}

    out = collect_and_attach_colorbar(
        fig,
        style={},
        coor={"z": {"expr": "z"}},
        method_key="voronoi",
        df=pd.DataFrame({"z": [1.0, 5.0, 10.0]}),
    )
    fig._finalize_axc("axc")

    assert isinstance(out["norm"], mcolors.Normalize)
    assert not isinstance(out["norm"], mcolors.LogNorm)
    assert fig.axc._cb["mode"] == "norm"
    assert fig.axc.get_yscale() == "linear"


@pytest.mark.parametrize(
    "alias",
    [
        "line",
        "lines",
        "points",
        "point",
        "scatterplot",
        "tri_field",
        "tri_color",
        "tri_field_axes",
        "tripcolor_gouraud",
        "grid_profiling",
    ],
)
def test_removed_method_aliases_are_rejected(alias):
    with pytest.raises(KeyError):
        resolve_method(alias)


def test_colorbar_finalize_allows_missing_label_config():
    fig = Figure()
    fig.logger = _logger()
    fig.frame = {
        "figure": {"figsize": (2, 2)},
        "axc": {
            "ticks": {},
            "color": {"scale": "log", "cmap": "viridis"},
        },
    }
    fig.fig = plt.figure(figsize=(2, 2))
    fig.axes = {}
    fig.axc = {"rect": [0.1, 0.1, 0.2, 0.8]}

    out = collect_and_attach_colorbar(
        fig,
        style={},
        coor={"z": {"expr": "z"}},
        method_key="voronoi",
        df=pd.DataFrame({"z": [1.0, 5.0, 10.0]}),
    )
    fig._finalize_axc("axc")

    assert isinstance(out["norm"], mcolors.LogNorm)
    assert fig.axc._cb["mode"] == "log"
    assert fig.axc.get_yscale() == "log"


@pytest.mark.parametrize("ax_type", ["rect", "tri"])
def test_render_layer_does_not_mutate_layer_data(monkeypatch, ax_type):
    original_data = object()
    converted_data = object()

    fig = SimpleNamespace(
        logger=_logger(),
        style={},
        _ensure_pandas_data=lambda data, reason="render": converted_data,
        _eval_series=lambda data, spec: (data, spec["expr"]),
    )
    ax = SimpleNamespace(_type=ax_type)
    layer_info = {
        "name": "layer",
        "method": "scatter",
        "data": original_data,
        "coor": {"x": {"expr": "x"}, "y": {"expr": "y"}},
    }

    def dummy_method(**kwargs):
        return kwargs

    monkeypatch.setattr(layer_runtime_mod, "resolve_callable", lambda *args, **kwargs: (dummy_method, None))
    monkeypatch.setattr(layer_runtime_mod, "collect_and_attach_colorbar", lambda *args, **kwargs: args[1])

    out = layer_runtime_mod.render_layer(fig, ax, layer_info)

    assert layer_info["data"] is original_data
    assert out["x"] == (converted_data, "x")
    assert out["y"] == (converted_data, "y")


def test_render_releases_layer_data_even_when_render_fails(monkeypatch):
    fig = Figure()
    fig.logger = _logger()
    fig.axes = {}
    fig._render_queue = [
        (
            SimpleNamespace(_type="rect"),
            {
                "name": "layer",
                "data": pd.DataFrame({"x": [1.0]}),
                "data_loaded": False,
            },
        )
    ]

    monkeypatch.setattr(Figure, "_prescan_colorbar_ranges", lambda self: None)

    calls = []

    def fake_load(fig_obj, layer_info):
        calls.append("load")
        layer_info["data_loaded"] = True
        return layer_info["data"]

    def fake_render(fig_obj, ax_obj, layer_info):
        calls.append("render")
        raise RuntimeError("boom")

    def fake_release(fig_obj, layer_info):
        calls.append("release")
        layer_info["data"] = None
        layer_info["data_loaded"] = False

    monkeypatch.setattr(figure_mod, "runtime_load_layer_runtime_data", fake_load)
    monkeypatch.setattr(figure_mod, "runtime_render_layer", fake_render)
    monkeypatch.setattr(figure_mod, "runtime_release_layer_runtime_data", fake_release)

    with pytest.raises(RuntimeError):
        fig.render()

    assert calls == ["load", "render", "release"]
    assert fig._render_queue[0][1]["data_loaded"] is False
    assert fig._render_queue[0][1]["data"] is None


def test_prescan_release_does_not_consume_shared_sources():
    shared = SharedContent(logger=None)
    ctx = DataContext(shared)
    ctx.set_usage_plan({"porfXY": 1, "alias": 1})
    ctx.update("porfXY", pd.DataFrame({"x": [1.0]}))

    fig = SimpleNamespace(
        context=ctx,
        preprocessor=SimpleNamespace(should_release_between_uses=lambda *_args, **_kwargs: True, logger=_logger()),
        logger=_logger(),
    )
    layer_info = {
        "data": pd.DataFrame({"x": [1.0]}),
        "data_loaded": True,
        "share_name": "porfXY",
        "source_refs": ["alias"],
    }

    layer_runtime_mod.release_layer_runtime_data(fig, layer_info, consume_sources=False)

    assert layer_info["data"] is None
    assert ctx.remaining_uses("porfXY") == 1
    assert ctx.remaining_uses("alias") == 1
    assert isinstance(ctx.get("porfXY"), pd.DataFrame)


def test_grid_profile_mesh_reconstructs_from_grid_metadata():
    df = pd.DataFrame(
        {
            "__grid_ix__": [0, 0, 1, 0, 1],
            "__grid_iy__": [0, 0, 0, 1, 1],
            "__grid_bin__": [2, 2, 2, 2, 2],
            "__grid_xmin__": [0.0, 0.0, 0.0, 0.0, 0.0],
            "__grid_xmax__": [2.0, 2.0, 2.0, 2.0, 2.0],
            "__grid_ymin__": [0.0, 0.0, 0.0, 0.0, 0.0],
            "__grid_ymax__": [2.0, 2.0, 2.0, 2.0, 2.0],
            "__grid_xscale__": ["linear"] * 5,
            "__grid_yscale__": ["linear"] * 5,
            "__grid_objective__": ["max"] * 5,
        }
    )

    mesh = grid_profile_mesh(
        x=[0.25, 0.25, 1.25, 0.25, 1.25],
        y=[0.25, 0.25, 0.25, 1.25, 1.25],
        z=[1.0, 5.0, 3.0, 2.0, 4.0],
        df=df,
        xlim=[0.0, 2.0],
        ylim=[0.0, 2.0],
        xscale="linear",
        yscale="linear",
        objective="max",
        objective_from_style=False,
    )

    assert mesh is not None
    x_edges, y_edges, grid = mesh

    assert np.allclose(x_edges, [0.0, 1.0, 2.0])
    assert np.allclose(y_edges, [0.0, 1.0, 2.0])
    assert grid.shape == (2, 2)
    assert grid[0, 0] == 5.0
    assert grid[0, 1] == 3.0
    assert grid[1, 0] == 2.0
    assert grid[1, 1] == 4.0


def test_preprofile_identity_ignores_runtime_bin():
    dp = DataPreprocessor(context=None)
    cfg1 = {
        "coordinates": {"x": {"expr": "x"}, "y": {"expr": "y"}},
        "method": "bridson",
        "bin": 8,
        "objective": "max",
    }
    cfg2 = {
        "coordinates": {"x": {"expr": "x"}, "y": {"expr": "y"}},
        "method": "bridson",
        "bin": 32,
        "objective": "min",
    }

    assert dp._preprofile_profile_cfg(cfg1) == dp._preprofile_profile_cfg(cfg2)


def test_runtime_projection_includes_row_identity_and_demand():
    dp = DataPreprocessor(context=None)
    transform = [
        {"add_column": {"name": "c", "expr": "a + b"}},
        {"filter": "c > 0"},
    ]

    projection = dp._runtime_projection(transform, ["style_expr"])

    assert JP_ROW_IDX in projection
    assert "a" in projection
    assert "b" in projection
    assert "c" in projection
    assert "style_expr" in projection


def test_runtime_transform_to_csv_exports_and_bypasses_cache(tmp_path):
    class _Cache:
        def cache_key(self, payload):
            return "cache-key"

        def get_dataframe_meta(self, key):
            raise AssertionError("cache lookup should be bypassed for CSV exports")

        def get_dataframe(self, key):
            raise AssertionError("cache lookup should be bypassed for CSV exports")

        def put_dataframe(self, key, df, meta=None):
            raise AssertionError("cache write should be bypassed for CSV exports")

    source_df = pd.DataFrame({"x": [1, 2, 3]})
    dp = DataPreprocessor(
        context=SimpleNamespace(get=lambda source: source_df),
        cache=_Cache(),
        logger=_logger(),
        base_dir=str(tmp_path),
    )

    out, key, from_cache = dp.run_pipeline(
        "sample",
        [
            {"add_column": {"name": "y", "expr": "x * 2"}},
            {"to_csv": "./saved/runtime_export.csv"},
            {"add_column": {"name": "z", "expr": "y + 1"}},
        ],
        use_cache=True,
        mode="runtime",
    )

    out_csv = tmp_path / "saved" / "runtime_export.csv"
    saved = pd.read_csv(out_csv)

    assert out is not None
    assert key
    assert from_cache is False
    assert list(saved.columns) == ["x", "y"]
    assert list(out.columns) == ["y", "z"]


def test_prebuild_split_keeps_profile_identity_coordinates_only():
    dp = DataPreprocessor(context=None)
    transform = [
        {"add_column": {"name": "c", "expr": "a + b"}},
        {
            "profile": {
                "coordinates": {"x": {"expr": "x"}, "y": {"expr": "y"}},
                "method": "bridson",
                "bin": 16,
                "objective": "max",
            }
        },
        {"sortby": "c"},
    ]

    pre_transform, runtime_transform = dp._split_prebuild_transform(transform)

    assert pre_transform is not None
    assert runtime_transform is not None
    assert pre_transform[-1]["profile"] == {"coordinates": {"x": {"expr": "x"}, "y": {"expr": "y"}}}
    assert runtime_transform[0]["profile"]["method"] == "bridson"
    assert runtime_transform[0]["profile"]["bin"] == 16
    assert runtime_transform[1]["sortby"] == "c"


def test_pipeline_key_changes_when_projection_changes():
    dp = DataPreprocessor(context=None)
    transform = [{"filter": "x > 0"}]

    key1 = dp._pipeline_key("source", transform, combine="concat", mode="runtime", projection=["a"])
    key2 = dp._pipeline_key("source", transform, combine="concat", mode="runtime", projection=["b"])

    assert key1 != key2


def test_prebuild_profiles_rewrites_profile_source_to_alias():
    dp = DataPreprocessor(context=SimpleNamespace(register=lambda *args, **kwargs: None))
    config = {
        "Figures": [
            {
                "name": "fig",
                "layers": [
                    {
                        "name": "layer",
                        "data": [
                            {
                                "source": "dataset",
                                "transform": [
                                    {"addcolumn": {"name": "c", "expr": "x + 1"}},
                                    {
                                        "profile": {
                                            "coordinates": {"x": {"expr": "x"}, "y": {"expr": "y"}},
                                            "method": "bridson",
                                            "bin": 16,
                                            "objective": "max",
                                        }
                                    },
                                    {"sortby": "c"},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }

    stats = dp.prebuild_profiles(config)
    rewritten = config["Figures"][0]["layers"][0]["data"][0]

    assert stats["tasks"] == 1
    assert stats["hits"] == 0
    assert stats["miss"] == 1
    assert rewritten["source"].startswith("__jp_preprofile_")
    assert rewritten["transform"][0]["profile"] == {
        "coordinates": {"x": {"expr": "x"}, "y": {"expr": "y"}},
        "method": "bridson",
        "bin": 16,
        "objective": "max",
    }
    assert rewritten["transform"][1]["sortby"] == "c"

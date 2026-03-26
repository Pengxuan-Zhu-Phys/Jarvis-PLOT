from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from jarvisplot.Figure.colorbar_runtime import collect_and_attach_colorbar
from jarvisplot.Figure.config_runtime import apply_figure_config
from jarvisplot.Figure.figure import Figure
from jarvisplot.Figure.preprocessor import DataPreprocessor
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
    assert "\\<string\\>" in formatted


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

    ok = apply_figure_config(
        fig,
        {
            "name": "fig",
            "style": ["gambit_2x1", "rectcmap"],
            "frame": {"figure": {}, "axes": {}},
            "layers": [],
        },
    )

    assert ok is True
    assert fig.mode == "gambit"


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

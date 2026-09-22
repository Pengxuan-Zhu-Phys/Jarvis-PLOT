from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from jarvisplot.Figure import layer_runtime
from jarvisplot.Figure.preprocessor import DataPreprocessor
from jarvisplot.data_loader import DataSet
from jarvisplot.dryrun_runtime import dryrun_config, dryrun_file
from jarvisplot.generated_data import generate_dataframe
from jarvisplot.validation import validate_config


def _logger():
    return SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )


def test_generate_dataframe_evaluates_ordered_columns_and_broadcasts_scalars():
    frame = generate_dataframe(
        {
            "columns": {
                "x": {"linspace": [0.0, 2.0, 3]},
                "y": {"expr": "x ** 2"},
                "series": 7,
            }
        }
    )

    assert frame.to_dict(orient="list") == {
        "x": [0.0, 1.0, 2.0],
        "y": [0.0, 1.0, 4.0],
        "series": [7, 7, 7],
    }


def test_generated_dataset_is_named_and_runs_dataset_transforms(tmp_path):
    dataset = DataSet()
    dataset.logger = _logger()
    dataset.setinfo(
        {
            "name": "theory",
            "type": "generated",
            "generate": {
                "columns": {
                    "x": {"arange": [4]},
                    "y": {"expr": "x + 1"},
                }
            },
            "transform": [{"add_column": {"name": "twice_y", "expr": "2 * y"}}],
        },
        rootpath=str(tmp_path),
    )

    frame = dataset.get_data()

    assert list(frame["x"]) == [0, 1, 2, 3]
    assert list(frame["twice_y"]) == [2, 4, 6, 8]
    assert "__jp_row_idx__" in frame


def test_schema_accepts_generated_dataset_and_separate_layer_blocks():
    config = {
        "DataSet": [
            {
                "name": "theory",
                "type": "generated",
                "generate": {"columns": {"x": {"arange": [3]}, "y": {"expr": "x"}}},
            }
        ],
        "Figures": [
            {
                "name": "figure",
                "layers": [
                    {
                        "name": "curves",
                        "axes": "ax",
                        "method": "plot",
                        "combine": "separate",
                        "data": [
                            {
                                "label": "one",
                                "generate": {
                                    "columns": {"x": {"arange": [3]}, "y": {"expr": "x"}}
                                },
                                "style": {"color": "C0"},
                            },
                            {
                                "label": "two",
                                "generate": {
                                    "columns": {"x": {"arange": [3]}, "y": {"expr": "2 * x"}}
                                },
                            },
                        ],
                        "coordinates": {"x": {"expr": "x"}, "y": {"expr": "y"}},
                    }
                ],
            }
        ],
    }

    assert list(validate_config(config)) == []


def test_dryrun_materializes_generated_dataset_and_layer_block(tmp_path):
    config = tmp_path / "generated.yaml"
    config.write_text(
        """
DataSet:
  - name: base
    type: generated
    generate:
      columns:
        x: {arange: [5]}
        y: {expr: "x * x"}
Figures:
  - name: figure
    layers:
      - name: private_curve
        axes: ax
        method: plot
        data:
          - generate:
              columns:
                x: {arange: [4]}
                y: {expr: "x + 1"}
        coordinates:
          x: {expr: x}
          y: {expr: y}
""".lstrip(),
        encoding="utf-8",
    )

    report, bag = dryrun_file(str(config))

    assert bag.ok
    assert report["datasets"]["base"]["rows"] == 5
    layer = report["layers"][0]
    assert layer["source"] == "<generated>"
    assert layer["n_points"] == 4


def test_layer_generated_blocks_apply_transforms_and_stay_separate():
    fig = SimpleNamespace(logger=_logger(), preprocessor=None, context=None)
    layer = {
        "name": "curves",
        "combine": "separate",
        "data": [
            {
                "label": "first",
                "generate": {"columns": {"x": {"arange": [4]}, "y": {"expr": "x"}}},
                "transform": [{"add_column": {"name": "y", "expr": "y + 1"}}],
            },
            {
                "label": "second",
                "generate": {"columns": {"x": {"arange": [2]}, "y": {"full": [2, 5]}}},
            },
        ],
    }

    data, cache_ref = layer_runtime.load_layer_data(fig, layer)

    assert cache_ref is None
    assert set(data) == {"first", "second"}
    assert data["first"].to_dict(orient="list") == {"x": [0, 1, 2, 3], "y": [1, 2, 3, 4]}
    assert data["second"].to_dict(orient="list") == {"x": [0, 1], "y": [5, 5]}


def test_separate_generated_blocks_render_independently_with_style_overrides(monkeypatch):
    fig = SimpleNamespace(
        logger=_logger(),
        style={},
        _ensure_pandas_data=lambda data, reason="render": data,
        _eval_series=lambda data, spec: data[spec["expr"]].to_numpy(),
    )
    ax = SimpleNamespace(_type="rect")
    layer_info = {
        "name": "curves",
        "method": "plot",
        "data": {
            "exact": pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}),
            "approx": pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.5]}),
        },
        "coor": {"x": {"expr": "x"}, "y": {"expr": "y"}},
        "style": {"linewidth": 2.0},
        "layer_spec": {
            "name": "curves",
            "data": [
                {"label": "exact", "style": {"color": "C0"}},
                {"label": "approx", "style": {"color": "C1", "linestyle": "--"}},
            ],
        },
    }

    monkeypatch.setattr(layer_runtime, "resolve_callable", lambda *args, **kwargs: (lambda **kw: kw, None))
    monkeypatch.setattr(layer_runtime, "collect_and_attach_colorbar", lambda *args, **kwargs: args[1])

    rendered = layer_runtime.render_layer(fig, ax, layer_info)

    assert len(rendered) == 2
    assert rendered[0]["label"] == "exact"
    assert rendered[0]["color"] == "C0"
    assert rendered[0]["linewidth"] == 2.0
    assert rendered[1]["label"] == "approx"
    assert rendered[1]["color"] == "C1"
    assert rendered[1]["linestyle"] == "--"
    np.testing.assert_allclose(rendered[1]["y"], [0.0, 0.5])


def test_generated_declaration_participates_in_share_data_cache_signature():
    preprocessor = DataPreprocessor(context=None)
    common = {
        "name": "curves",
        "axes": "ax",
        "method": "plot",
        "combine": "separate",
        "share_data": "curve_data",
    }
    first = {
        **common,
        "data": [{"generate": {"columns": {"x": {"values": [1, 2]}}}}],
    }
    second = {
        **common,
        "data": [{"generate": {"columns": {"x": {"values": [9, 10]}}}}],
    }

    assert preprocessor._layer_signature(first) != preprocessor._layer_signature(second)


def test_dryrun_checks_every_generated_data_block():
    config = {
        "Figures": [
            {
                "name": "figure",
                "layers": [
                    {
                        "name": "curves",
                        "axes": "ax",
                        "method": "plot",
                        "combine": "separate",
                        "data": [
                            {
                                "generate": {
                                    "columns": {
                                        "x": {"values": [1, 2]},
                                        "y": {"values": [1, 2]},
                                    }
                                }
                            },
                            {
                                "generate": {
                                    "columns": {
                                        "x": {"expr": "undefined_name"},
                                        "y": {"values": [1, 2]},
                                    }
                                }
                            },
                        ],
                        "coordinates": {"x": {"expr": "x"}, "y": {"expr": "y"}},
                    }
                ],
            }
        ]
    }

    report, bag = dryrun_config(config)

    assert report["ok"] is False
    assert any("data[1].generate" in diagnostic.path for diagnostic in bag.errors)

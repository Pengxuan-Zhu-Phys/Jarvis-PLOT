"""E1/E2: JP-VIZ health rules on synthetic observations."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from jarvisplot.Figure import layer_runtime
from jarvisplot.render_health import (
    LayerObservation,
    TransformStepObs,
    evaluate_health,
)


def test_load_layer_runtime_data_prescan_does_not_double_observe(monkeypatch):
    """Colorbar prescan must not append LayerObservation (report duplicate bug)."""
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [1.0, 2.0]})
    monkeypatch.setattr(
        layer_runtime,
        "load_layer_data",
        lambda fig, layer: (df, None),
    )

    fig = SimpleNamespace(
        name="f1",
        health_observations=[],
        frame={"ax": {"xlim": [0, 1], "ylim": [0, 1]}},
        _yaml_frame={"ax": {"xlim": [0, 1], "ylim": [0, 1]}},
        logger=SimpleNamespace(debug=lambda *a, **k: None),
        _store_share_data_if_needed=lambda *a, **k: None,
    )
    layer_info = {
        "name": "_density",
        "layer_spec": {
            "name": "_density",
            "method": "pcolormesh",
            "data": [{"source": "s"}],
            "coordinates": {
                "x": {"expr": "x"},
                "y": {"expr": "y"},
                "z": {"expr": "z"},
            },
        },
        "data_loaded": False,
    }

    # Prescan path (figure._prescan_colorbar_ranges)
    layer_runtime.load_layer_runtime_data(fig, layer_info, observe=False)
    assert len(fig.health_observations) == 0
    layer_runtime.release_layer_runtime_data(fig, layer_info, consume_sources=False)
    assert layer_info.get("data_loaded") is False

    # Render loop path (default observe=True) — exactly one observation
    layer_runtime.load_layer_runtime_data(fig, layer_info)
    assert len(fig.health_observations) == 1
    assert fig.health_observations[0].layer == "_density"



def test_viz_001_empty_layer():
    bag = evaluate_health(
        [
            LayerObservation(
                figure="f1",
                layer="pts",
                method="scatter",
                n_points=0,
            )
        ]
    )
    assert any(d.code == "JP-VIZ-001" for d in bag.errors)


def test_viz_003_filter_empties_table():
    bag = evaluate_health(
        [
            LayerObservation(
                figure="f1",
                layer="pts",
                method="scatter",
                n_points=0,
                steps=[
                    TransformStepObs(
                        name="filter",
                        detail="LogL > 1e9",
                        rows_in=100,
                        rows_out=0,
                    )
                ],
            )
        ]
    )
    codes = {d.code for d in bag}
    assert "JP-VIZ-003" in codes
    assert "JP-VIZ-001" in codes


def test_viz_002_fully_outside_lim():
    bag = evaluate_health(
        [
            LayerObservation(
                figure="f1",
                layer="pts",
                method="scatter",
                n_points=50,
                data_bbox=(10.0, 20.0, 10.0, 20.0),
                axes_lim={"x": [0.0, 1.0], "y": [0.0, 1.0]},
            )
        ]
    )
    assert any(d.code == "JP-VIZ-002" and d.level == "error" for d in bag)


def test_viz_005_log_nonpositive():
    bag = evaluate_health(
        [
            LayerObservation(
                figure="f1",
                layer="pts",
                method="scatter",
                n_points=10,
                data_bbox=(-1.0, 5.0, 1.0, 2.0),
                axes_lim={"x": [0.1, 10.0], "y": [0.1, 10.0]},
                xscale="log",
            )
        ]
    )
    assert any(d.code == "JP-VIZ-005" for d in bag.warnings)


def test_viz_008_collapsed_cloud():
    bag = evaluate_health(
        [
            LayerObservation(
                figure="f1",
                layer="pts",
                method="scatter",
                n_points=20,
                data_bbox=(0.0, 0.01, 0.0, 0.01),
                axes_lim={"x": [0.0, 10.0], "y": [0.0, 10.0]},
            )
        ]
    )
    assert any(d.code == "JP-VIZ-008" for d in bag.warnings)


def test_viz_004_colorbar_saturation():
    bag = evaluate_health(
        [
            LayerObservation(
                figure="f1",
                layer="dens",
                method="pcolormesh",
                n_points=100,
                c_min=0.0,
                c_max=12.0,
                colorbar_vmin=0.0,
                colorbar_vmax=0.8,
            )
        ]
    )
    assert any(d.code == "JP-VIZ-004" for d in bag.errors)


def test_viz_006_occlusion_by_mesh():
    bag = evaluate_health(
        [
            LayerObservation(
                figure="f1",
                layer="pts",
                axes="ax",
                method="scatter",
                n_points=50,
                zorder=1,
                data_bbox=(1.0, 2.0, 1.0, 2.0),
            ),
            LayerObservation(
                figure="f1",
                layer="mesh",
                axes="ax",
                method="pcolormesh",
                n_points=200,
                zorder=10,
                data_bbox=(0.0, 5.0, 0.0, 5.0),
            ),
        ]
    )
    assert any(d.code == "JP-VIZ-006" for d in bag.warnings)


def test_viz_007_grid_nan():
    bag = evaluate_health(
        [
            LayerObservation(
                figure="f1",
                layer="field",
                method="jpfield",
                n_points=100,
                grid_nan_ratio=0.9,
            )
        ]
    )
    assert any(d.code == "JP-VIZ-007" for d in bag.errors)


def test_viz_009_legend_mismatch():
    bag = evaluate_health(
        [
            LayerObservation(
                figure="f1",
                layer="pts",
                method="scatter",
                n_points=10,
                style_label="signal",
                legend_labels=["background"],
            )
        ]
    )
    assert any(d.code == "JP-VIZ-009" for d in bag.warnings)

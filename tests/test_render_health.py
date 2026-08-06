"""E1/E2: JP-VIZ health rules on synthetic observations."""

from __future__ import annotations

from jarvisplot.render_health import (
    LayerObservation,
    TransformStepObs,
    evaluate_health,
)


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

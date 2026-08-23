from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from jarvisplot.Figure.figure import Figure


def _logger():
    return SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )


@pytest.mark.parametrize(
    ("position", "expected_position", "expected_ha"),
    [
        ("top", (0.005, 1.0), "left"),
        ("center", (0.5, 1.0), "center"),
        ("right", (0.995, 1.0), "right"),
    ],
)
def test_ax_title_position_maps_to_internal_axes_text_params(
    position, expected_position, expected_ha
):
    raw_fig = plt.figure()
    fig = Figure()
    fig.logger = _logger()
    fig.fig = raw_fig
    fig.frame = {
        "ax": {
            "title": "A title",
            "title_params": {
                "position": position,
                "fontsize": 15,
            },
            "labels": {},
            "ticks": {},
        }
    }

    fig.ax = {"rect": [0.1, 0.1, 0.8, 0.8]}

    raw_ax = fig.axes["ax"].ax
    assert len(raw_ax.texts) == 1
    title = raw_ax.texts[0]
    assert title.get_text() == "A title"
    assert title.get_position() == expected_position
    assert title.get_ha() == expected_ha
    assert title.get_va() == "bottom"
    assert title.get_fontsize() == 15
    assert title.get_transform() == raw_ax.transAxes
    plt.close(raw_fig)


def test_a4paper_2x1_rect_cards_define_default_title_params():
    cards_dir = Path(__file__).resolve().parents[1] / "jarvisplot" / "cards" / "a4paper" / "2x1"
    for filename in ("rect.json", "rect_cmap.json", "rect5x1.json"):
        card = json.loads((cards_dir / filename).read_text(encoding="utf-8"))
        axis_name = "ax0" if filename == "rect5x1.json" else "ax"
        title_params = card["Frame"][axis_name]["title_params"]
        assert title_params["position"] == "top"
        assert not {"x", "y", "ha", "va"}.intersection(title_params)


def test_a4paper_2x1_rect_default_axis_labels_are_centered():
    cards_dir = Path(__file__).resolve().parents[1] / "jarvisplot" / "cards" / "a4paper" / "2x1"
    card = json.loads((cards_dir / "rect.json").read_text(encoding="utf-8"))
    labels = card["Frame"]["ax"]["labels"]

    assert labels["xlabel"]["loc"] == "center"
    assert labels["ylabel"]["loc"] == "center"
    assert labels["ylabel_coords"]["y"] == 0.5


@pytest.mark.parametrize("axis_name", ["ax1", "ax2", "ax3", "ax4"])
def test_numbered_axis_titles_only_render_on_ax0(axis_name):
    warnings = []
    raw_fig = plt.figure()
    fig = Figure()
    fig.logger = SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        warning=lambda message: warnings.append(message),
    )
    fig.fig = raw_fig
    fig.frame = {
        "ax0": {
            "title": "Top panel",
            "title_params": {"position": "center"},
            "labels": {},
            "ticks": {},
        },
        axis_name: {
            "title": f"Ignored {axis_name}",
            "labels": {},
            "ticks": {},
        },
    }

    fig._ensure_numbered_rect_axes("ax0", {"rect": [0.1, 0.55, 0.8, 0.35]})
    fig._ensure_numbered_rect_axes(axis_name, {"rect": [0.1, 0.1, 0.8, 0.35]})

    assert len(fig.axes["ax0"].ax.texts) == 1
    assert len(fig.axes[axis_name].ax.texts) == 0
    assert warnings == [
        f"Ignoring title for axes '{axis_name}': title rendering is only supported on 'ax0'."
    ]
    plt.close(raw_fig)


def test_title_params_explicit_position_keys_are_ignored_with_warning():
    warnings = []
    raw_fig = plt.figure()
    fig = Figure()
    fig.logger = SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        warning=lambda message: warnings.append(message),
    )
    fig.fig = raw_fig
    fig.frame = {
        "ax": {
            "title": "A title",
            "title_params": {"position": "right", "x": 0.0, "y": 0.0, "ha": "left"},
            "labels": {},
            "ticks": {},
        }
    }

    fig.ax = {"rect": [0.1, 0.1, 0.8, 0.8]}

    title = fig.axes["ax"].ax.texts[0]
    assert title.get_position() == (0.995, 1.0)
    assert title.get_ha() == "right"
    assert warnings == [
        "Ignoring title_params ['x', 'y', 'ha'] for axes 'ax'; use position instead."
    ]
    plt.close(raw_fig)

"""Whole-figure regressions for defects met while writing the manual scenes.

Each test renders a small scene through ``JarvisPLOT`` the way ``jplot <yaml>``
does and looks at the figure that came out: the method that used to raise, the
badge controlled by ``--print``, the card list that used to be
appended to instead of replaced, and the two cards whose labels left the page
or sat on the tick labels.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml
from loguru import logger
from matplotlib.backends.backend_agg import FigureCanvasAgg

from jarvisplot.core import JarvisPLOT
from jarvisplot.Figure.figure import Figure


@pytest.fixture(autouse=True)
def _reset_logging_and_figures():
    plt.close("all")
    yield
    plt.close("all")
    logger.remove()


@pytest.fixture
def figures(monkeypatch):
    """The jarvisplot ``Figure`` objects that were saved, in order."""
    saved: list[Figure] = []
    original = Figure.savefig

    def spy(self, *args, **kwargs):
        saved.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", spy)
    return saved


def _write_data(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(0.1, 2.0, 120)
    y = rng.uniform(0.1, 2.0, 120)
    r2 = (x - 1.0) ** 2 + (y - 1.0) ** 2
    pd.DataFrame({"x": x, "y": y, "z": np.exp(-r2), "LogL": -0.5 * r2 / 0.04}).to_csv(
        tmp_path / "samples.csv", index=False
    )
    steps = np.arange(300)
    log_volume = -steps / 100.0
    log_like = -20.0 * np.exp(log_volume)
    log_weight = log_like + log_volume + np.log(1.0 - np.exp(-1.0 / 100.0))
    pd.DataFrame(
        {
            "log_PriorVolume": log_volume,
            "log_Like": log_like,
            "log_weight": log_weight,
            "log_Evidence": np.logaddexp.accumulate(log_weight),
            "log_Evidence_err": np.full(steps.size, 0.05),
            "samples_nlive": np.full(steps.size, 100),
            "samples_it": steps,
        }
    ).to_csv(tmp_path / "dynesty.csv", index=False)


def _render(tmp_path: Path, figs: list[dict], *options: str) -> int:
    _write_data(tmp_path)
    config = {
        "DataSet": [
            {"name": "samples", "type": "csv", "path": "samples.csv"},
            {"name": "dynesty", "type": "csv", "path": "dynesty.csv"},
        ],
        "Figures": figs,
        "output": {"dir": "./plots", "formats": ["png"], "dpi": 40},
    }
    path = tmp_path / "plot.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    try:
        JarvisPLOT(argv=[str(path), *options]).init()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def _layer(method: str, coordinates: dict, *, axes: str = "ax", **extra) -> dict:
    return {
        "name": f"{method}_layer",
        "axes": axes,
        "method": method,
        "data": [{"source": "samples"}],
        "coordinates": {k: {"expr": v} for k, v in coordinates.items()},
        **extra,
    }


def _figure(name: str, layers: list[dict], *, style=("a4paper_2x1", "rect"), **extra) -> dict:
    return {"name": name, "style": list(style), "layers": layers, **extra}


def _png(tmp_path: Path, name: str) -> Path:
    return tmp_path / "plots" / f"{name}.png"


# --------------------------------------------------------------------------- #
# Methods that could not render
# --------------------------------------------------------------------------- #


def test_voronoif_layer_renders(tmp_path):
    figs = [
        _figure(
            "vf",
            [
                _layer(
                    "voronoif",
                    {"x": "x", "y": "y", "where": "LogL > -2"},
                    style={"hatch": "///", "facecolor": "none", "edgecolor": "#cc3311"},
                )
            ],
            frame={"ax": {"xlim": [0, 2], "ylim": [0, 2]}},
        ),
        _figure(
            "vf_all",
            [_layer("voronoif", {"x": "x", "y": "y"})],
            frame={"ax": {"xlim": [0, 2], "ylim": [0, 2]}},
        ),
    ]

    assert _render(tmp_path, figs) == 0
    assert _png(tmp_path, "vf").is_file()
    assert _png(tmp_path, "vf_all").is_file()


def test_triplot_layer_renders_on_rect_axes(tmp_path):
    figs = [
        _figure(
            "tp",
            [_layer("triplot", {"x": "x", "y": "y"}, style={"linewidth": 0.3})],
            frame={"ax": {"xlim": [0, 2], "ylim": [0, 2]}},
        )
    ]

    assert _render(tmp_path, figs) == 0
    assert _png(tmp_path, "tp").is_file()


def test_triplot_layer_renders_on_ternary_axes(tmp_path):
    figs = [
        _figure(
            "tp_tri",
            [_layer("triplot", {"left": "x", "right": "y", "bottom": "z"}, axes="axtri")],
            style=("a4paper_2x1", "Ternary"),
            frame={"axtri": {"labels": {"left": "A", "bottom": "B", "right": "C"}}},
        )
    ]

    assert _render(tmp_path, figs) == 0
    assert _png(tmp_path, "tp_tri").is_file()


# --------------------------------------------------------------------------- #
# frame.* merged over the card
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", [False, True])
def test_frame_rejects_logo_switch(value):
    fig = Figure()
    with pytest.raises(ValueError, match='use --print'):
        fig.frame = {"axlogo": value}


def test_print_controls_the_badge(tmp_path, figures):
    fig = _figure("badge", [_layer("scatter", {"x": "x", "y": "y"})])
    assert _render(tmp_path, [fig]) == 0
    assert "axlogo" in figures[-1].axes
    assert _render(tmp_path, [fig], "--print") == 0
    assert "axlogo" not in figures[-1].axes


@pytest.mark.parametrize("family", ["a4paper_1x1", "gambit_1x1"])
def test_full_width_ternary_cards_render(tmp_path, figures, family):
    fig = _figure(
        family,
        [_layer("scatter", {"left": "x", "right": "y", "bottom": "z"}, axes="axtri")],
        style=(family, "Ternary"),
        frame={"axtri": {"labels": {"left": "A", "bottom": "B", "right": "C"}}},
    )
    assert _render(tmp_path, [fig]) == 0
    assert _png(tmp_path, family).is_file()
    assert {"axtri", "axlogo"} <= set(figures[-1].axes)
    assert figures[-1].fig.get_size_inches() == pytest.approx([6.7, 5.6])
    assert _render(tmp_path, [fig], "--print") == 0
    assert "axtri" in figures[-1].axes
    assert "axlogo" not in figures[-1].axes


def test_frame_list_replaces_the_cards_list(tmp_path, figures):
    rect = [0.2, 0.168, 0.74, 0.775]
    figs = [
        _figure(
            "rect",
            [_layer("scatter", {"x": "x", "y": "y"})],
            frame={"axes": {"ax": {"rect": rect}}},
        )
    ]

    assert _render(tmp_path, figs) == 0
    assert list(figures[0].axes["ax"].ax.get_position().bounds) == pytest.approx(rect)


def test_layer_without_method_is_announced_when_it_is_drawn():
    warnings: list[str] = []
    fig = Figure()
    fig.logger = SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        warning=lambda message, *args, **kwargs: warnings.append(str(message)),
    )
    fig.axes = {"ax": SimpleNamespace(layers=[])}

    fig.layers = [
        {"name": "implicit", "axes": "ax", "coordinates": {"x": {"expr": "x"}, "y": {"expr": "y"}}},
        {"name": "explicit", "axes": "ax", "method": "plot", "coordinates": {}},
    ]

    assert len(warnings) == 1
    assert "implicit" in warnings[0] and "scatter" in warnings[0]
    assert fig._render_queue[0][1]["method"] == "scatter"


# --------------------------------------------------------------------------- #
# Cards: labels inside the page, clear of the tick numbers
# --------------------------------------------------------------------------- #


def _drawn(fig: Figure):
    """A renderer for ``fig`` after a fresh draw.

    The saved figure was drawn at the output dpi, and a text's stored position
    is in pixels of that draw; measuring it at another dpi without drawing
    again compares two different scales.
    """
    canvas = FigureCanvasAgg(fig.fig)
    canvas.draw()
    return canvas.get_renderer()


def _extent(artist, renderer):
    return artist.get_window_extent(renderer)


def test_gambit_rectcmap_colorbar_label_clears_the_ticks_and_the_page(tmp_path, figures):
    figs = [
        _figure(
            "gambit",
            [_layer("scatter", {"x": "x", "y": "y", "c": "LogL"}, colorbar="axc")],
            style=("gambit_2x1", "rectcmap"),
            frame={"axc": {"label": {"ylabel": "$\\log\\mathcal{L}$"}}},
        )
    ]

    assert _render(tmp_path, figs) == 0
    fig = figures[0]
    renderer = _drawn(fig)
    axc = fig.axes["axc"]
    label = _extent(axc.yaxis.label, renderer)
    ticks = [
        _extent(tick.label2, renderer)
        for tick in axc.yaxis.get_major_ticks()
        if tick.label2.get_visible() and tick.label2.get_text()
    ]

    assert ticks
    assert label.x0 >= max(box.x1 for box in ticks)
    assert label.x1 <= fig.fig.bbox.x1


def test_dynesty_runplot_ylabels_stay_inside_the_page(tmp_path, figures):
    figs = [{"name": "run", "style": ["a4paper_2x1", "dynesty_runplot"]}]

    assert _render(tmp_path, figs) == 0
    fig = figures[0]
    renderer = _drawn(fig)
    labels = [
        _extent(fig.axes[name].yaxis.label, renderer)
        for name in ("ax0", "ax1", "ax2", "ax3", "ax4")
        if fig.axes[name].yaxis.label.get_text()
    ]
    # `set_labels: false` in the card leaves the runtime to write them.
    assert labels
    assert min(box.x0 for box in labels) >= 0.0

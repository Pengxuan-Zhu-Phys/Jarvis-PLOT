"""Per-element `show` switches: JSON decides which annotations get drawn.

The switch is a veto layered on top of the structural conditions in `_draw`
(which axes is primary, which is a colorbar, whether an axes has a frame), so
these tests assert two things at once: turning an element off removes exactly
its artists, and leaves every other element alone.
"""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.patches import FancyBboxPatch, Rectangle

from jarvisplot.Figure.design_runtime import draw_design_reference


def _wrapper(debug_config=None, *, numbered=False):
    raw = plt.figure(figsize=(3.3, 2.75))
    if numbered:
        axes = {
            "ax0": raw.add_axes([0.20, 0.70, 0.70, 0.20]),
            "ax1": raw.add_axes([0.20, 0.40, 0.70, 0.20]),
        }
    else:
        axes = {
            "axlogo": raw.add_axes([0.01, 0.01, 0.06, 0.072], frameon=False),
            "ax": raw.add_axes([0.142, 0.168, 0.680, 0.775]),
            "axc": raw.add_axes([0.827, 0.182, 0.036, 0.746]),
        }
    return SimpleNamespace(
        fig=raw,
        axes=axes,
        _debug_config=debug_config or {},
        logger=SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None),
    )


def _overlay(wrapper):
    draw_design_reference(wrapper)
    return wrapper.fig.axes[-1]


def _census(overlay):
    """Count the artist kinds the overlay produced.

    ``Annotation`` subclasses ``Text`` and lands in ``axes.texts`` too, so the
    arrowheads a dimension line draws would otherwise be counted as labels.
    """
    annotations = [a for a in overlay.texts if getattr(a, "arrow_patch", None) is not None]
    return {
        "labels": len(overlay.texts) - len(annotations),
        "arrows": len(annotations),
        "lines": len(overlay.lines),
        "rects": sum(1 for p in overlay.patches if type(p) is Rectangle),
        "panels": sum(1 for p in overlay.patches if isinstance(p, FancyBboxPatch)),
    }


@pytest.fixture
def baseline():
    w = _wrapper()
    try:
        yield _census(_overlay(w))
    finally:
        plt.close(w.fig)


def _census_with(config, **kw):
    w = _wrapper(config, **kw)
    try:
        return _census(_overlay(w))
    finally:
        plt.close(w.fig)


# --------------------------------------------------------------------------- #
# Each switch removes its own element
# --------------------------------------------------------------------------- #


def test_panel_off_removes_the_information_card(baseline):
    after = _census_with({"panel": {"show": False}})
    assert baseline["panels"] == 1
    assert after["panels"] == 0
    assert after["labels"] < baseline["labels"]


def test_figure_border_off_removes_one_rectangle(baseline):
    after = _census_with({"figure": {"border": {"show": False}}})
    assert after["rects"] == baseline["rects"] - 1


def test_axes_outline_off_removes_one_rectangle_per_axes(baseline):
    after = _census_with({"axes": {"outline": {"show": False}}})
    # three named axes, minus their three outlines; the figure border remains
    assert after["rects"] == baseline["rects"] - 3
    assert after["rects"] == 1


def test_caption_off_removes_exactly_one_text(baseline):
    after = _census_with({"figure": {"caption": {"show": False}}})
    assert after["labels"] == baseline["labels"] - 1
    assert after["rects"] == baseline["rects"]


def test_corner_ticks_off_only_affects_frameless_axes(baseline):
    after = _census_with({"axes": {"corner_ticks": {"show": False}}})
    # axlogo is the only frameless axes here: 8 hash marks
    assert after["lines"] == baseline["lines"] - 8
    assert after["rects"] == baseline["rects"]


def test_height_marker_off_removes_its_line_and_label(baseline):
    after = _census_with({"figure": {"height_marker": {"show": False}}})
    assert after["lines"] < baseline["lines"]
    assert after["labels"] == baseline["labels"] - 1


def test_colorbar_gap_off_removes_its_dimension(baseline):
    after = _census_with({"colorbar_gap": {"show": False}})
    assert after["labels"] == baseline["labels"] - 1
    assert after["panels"] == baseline["panels"]


@pytest.mark.parametrize("side, removed", [("left", 1), ("top", 2), ("bottom", 2)])
def test_each_margin_switch_removes_exactly_its_own_labels(baseline, side, removed):
    """`left` is the primary axes' inset alone; top/bottom cover every axes.

    The fixture has three -- axlogo, ax, axc -- and `margins.exclude` leaves
    the logo plate out, so each of those two switches takes two labels.
    """
    after = _census_with({"margins": {side: {"show": False}}})
    assert after["labels"] == baseline["labels"] - removed


def test_numbered_axes_off_removes_the_whole_column():
    with_column = _census_with({}, numbered=True)
    without = _census_with({"numbered_axes": {"show": False}}, numbered=True)
    assert without["labels"] < with_column["labels"]


def test_numbered_top_and_bottom_are_independent():
    both = _census_with({}, numbered=True)
    top_only = _census_with({"numbered_axes": {"bottom": {"show": False}}}, numbered=True)
    # two numbered axes lose their bottom dimension each
    assert top_only["labels"] == both["labels"] - 2
    assert top_only["arrows"] < both["arrows"]


# --------------------------------------------------------------------------- #
# Switching everything off must still leave a working figure
# --------------------------------------------------------------------------- #


def test_all_switches_off_draws_nothing_but_does_not_raise():
    after = _census_with(
        {
            "figure": {"border": {"show": False}, "caption": {"show": False},
                       "height_marker": {"show": False}},
            "axes": {"outline": {"show": False}, "corner_ticks": {"show": False}},
            "margins": {"left": {"show": False}, "top": {"show": False},
                        "bottom": {"show": False}},
            "colorbar_gap": {"show": False},
            "panel": {"show": False},
        }
    )
    assert after == {"labels": 0, "arrows": 0, "lines": 0, "rects": 0, "panels": 0}


def test_absent_show_key_means_on(baseline):
    """A card written before an element existed must keep drawing it."""
    after = _census_with({"panel": {"entry_gap": 5.0}})
    assert after == baseline

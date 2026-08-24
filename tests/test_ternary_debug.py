"""Ternary design reference: anchors, leaders, and where the leaders end.

The leader endpoints used to be literals `(0.84, 0.56)` / `(0.16, 0.56)`, which
match `ternary.json` but not `ternary_cmap.json`, whose labels sit at
`(0.8625, 0.575)` / `(0.1375, 0.575)`. On the two *Cmap cards the leaders
therefore stopped short of the label they point at. They are now derived from
the card, so the two can no longer disagree.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvisplot.Figure.design_runtime import draw_ternary_reference

CARDS = Path(__file__).resolve().parents[1] / "jarvisplot" / "cards"


class _Axtri:
    """Records what the ternary reference asked to draw."""

    def __init__(self):
        self.scatters = []
        self.plots = []

    def scatter(self, *, x, y, **kw):
        self.scatters.append((x, y, kw))

    def plot(self, *, x, y, **kw):
        self.plots.append((x, y, kw))


def _fig(frame, debug_config=None):
    return SimpleNamespace(
        axtri=_Axtri(),
        frame=frame,
        _debug_config=debug_config or {},
        logger=SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None),
    )


def _labels_from(card_relpath):
    card = json.loads((CARDS / card_relpath).read_text(encoding="utf-8"))
    return card["Frame"]["axtri"]["labels"]


ANCHORS = (([0.1], [0.2]), ([0.3], [0.4]), ([0.5], [0.6]))


# --------------------------------------------------------------------------- #
# Endpoints follow the card
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "card, side, expected",
    [
        ("a4paper/2x1/ternary.json", "right", (0.84, 0.56)),
        ("a4paper/2x1/ternary.json", "left", (0.16, 0.56)),
        ("a4paper/2x1/ternary_cmap.json", "right", (0.8625, 0.575)),
        ("a4paper/2x1/ternary_cmap.json", "left", (0.1375, 0.575)),
        ("gambit/2x1/ternary_cmap.json", "right", (0.8625, 0.575)),
        ("a4paper/2x1/ternary.json", "bottom", (0.5, -0.12)),
    ],
)
def test_leader_ends_at_the_cards_label_anchor(card, side, expected):
    fig = _fig({"axtri": {"labels": _labels_from(card)}})
    draw_ternary_reference(fig, ANCHORS)

    ends = {(round(x[-1], 6), round(y[-1], 6)) for x, y, _ in fig.axtri.plots}
    assert (round(expected[0], 6), round(expected[1], 6)) in ends


def test_cmap_and_plain_cards_get_different_leaders():
    """The regression this replaces: one hardcoded pair served both."""
    plain = _fig({"axtri": {"labels": _labels_from("a4paper/2x1/ternary.json")}})
    cmap = _fig({"axtri": {"labels": _labels_from("a4paper/2x1/ternary_cmap.json")}})
    draw_ternary_reference(plain, ANCHORS)
    draw_ternary_reference(cmap, ANCHORS)
    assert [p[0] for p in plain.axtri.plots] != [p[0] for p in cmap.axtri.plots]


def test_leader_starts_at_the_opposite_vertex_and_crosses_the_edge_midpoint():
    fig = _fig({"axtri": {"labels": _labels_from("a4paper/2x1/ternary.json")}})
    draw_ternary_reference(fig, ANCHORS)
    right = next(p for p in fig.axtri.plots if p[0][0] == 0.0 and p[1][0] == 0.0)
    assert right[0][:2] == [0.0, 0.75]
    assert right[1][:2] == [0.0, 0.5]


# --------------------------------------------------------------------------- #
# Switches and style come from JSON
# --------------------------------------------------------------------------- #


def test_anchors_are_marked_once_per_tick_group():
    fig = _fig({"axtri": {"labels": _labels_from("a4paper/2x1/ternary.json")}})
    draw_ternary_reference(fig, ANCHORS)
    assert len(fig.axtri.scatters) == 3
    assert fig.axtri.scatters[0][2]["c"] == "#FF42A1"
    assert fig.axtri.scatters[0][2]["marker"] == "."


def test_style_comes_from_json():
    fig = _fig(
        {"axtri": {"labels": _labels_from("a4paper/2x1/ternary.json")}},
        debug_config={"ternary": {"tick_anchors": {"style": {"c": "#00FF00", "s": 9.0}}}},
    )
    draw_ternary_reference(fig, ANCHORS)
    assert fig.axtri.scatters[0][2]["c"] == "#00FF00"
    assert fig.axtri.scatters[0][2]["s"] == 9.0


@pytest.mark.parametrize(
    "override, scatters, plots",
    [
        ({}, 3, 3),
        ({"ternary": {"tick_anchors": {"show": False}}}, 0, 3),
        ({"ternary": {"label_leaders": {"show": False}}}, 3, 0),
        (
            {"ternary": {"tick_anchors": {"show": False}, "label_leaders": {"show": False}}},
            0,
            0,
        ),
    ],
)
def test_show_switches(override, scatters, plots):
    fig = _fig({"axtri": {"labels": _labels_from("a4paper/2x1/ternary.json")}}, override)
    draw_ternary_reference(fig, ANCHORS)
    assert (len(fig.axtri.scatters), len(fig.axtri.plots)) == (scatters, plots)


# --------------------------------------------------------------------------- #
# It must never break a plot
# --------------------------------------------------------------------------- #


def test_a_card_without_label_anchors_draws_no_leaders_and_does_not_raise():
    fig = _fig({"axtri": {"labels": {}}})
    draw_ternary_reference(fig, ANCHORS)
    assert fig.axtri.plots == []
    assert len(fig.axtri.scatters) == 3


def test_a_broken_frame_is_swallowed():
    fig = SimpleNamespace(axtri=None, frame=None, _debug_config={}, logger=None)
    draw_ternary_reference(fig, ANCHORS)  # must not raise

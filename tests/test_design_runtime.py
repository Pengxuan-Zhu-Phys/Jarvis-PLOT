from __future__ import annotations

from math import isclose
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

from jarvisplot.Figure.design_runtime import draw_design_reference


def test_design_reference_collects_all_axes_info_inside_primary_ax():
    raw_fig = plt.figure(figsize=(3.3, 2.75))
    axes = {
        # Deliberately mirror style-card order, where axlogo commonly appears
        # before the main ax.  The info panel should still start with ax.
        "axlogo": raw_fig.add_axes([0.01, 0.01, 0.06, 0.072], frameon=False),
        "ax": raw_fig.add_axes([0.142, 0.168, 0.680, 0.775]),
        "axc": raw_fig.add_axes([0.827, 0.182, 0.036, 0.746]),
    }
    wrapper = SimpleNamespace(
        fig=raw_fig,
        axes=axes,
        logger=SimpleNamespace(debug=lambda *args, **kwargs: None),
    )

    try:
        draw_design_reference(wrapper)

        overlay = raw_fig.axes[-1]
        expected = [
            "axes layout",
            "ax",
            "  rect=[0.142, 0.168, 0.680, 0.775]",
            "  width: 5.700 cm  height: 5.413 cm",
            "axlogo",
            "  rect=[0.010, 0.010, 0.060, 0.072]",
            "  width: 0.503 cm  height: 0.503 cm",
            "axc",
            "  rect=[0.827, 0.182, 0.036, 0.746]",
            "  width: 0.302 cm  height: 5.211 cm",
        ]
        actual = [text.get_text() for text in overlay.texts if text.get_text() in expected]
        assert actual == expected

        info = next(text for text in overlay.texts if text.get_text() == "ax")
        detail = next(
            text for text in overlay.texts if text.get_text().startswith("  rect=[0.142")
        )
        assert info.get_fontsize() == 7.0
        assert info.get_fontweight() == "bold"
        assert detail.get_fontsize() == 4.6

        logo_outline = next(
            patch
            for patch in overlay.patches
            if isinstance(patch, Rectangle)
            and isclose(patch.get_x(), 0.01)
            and isclose(patch.get_y(), 0.01)
            and isclose(patch.get_width(), 0.06)
        )
        assert logo_outline.get_edgecolor() == to_rgba("#FF3FA4")

        info_panel = next(
            patch
            for patch in overlay.patches
            if patch.__class__.__name__ == "FancyBboxPatch"
        )
        assert info_panel.get_x() + info_panel.get_width() <= 0.70
        assert info_panel.get_facecolor() == to_rgba("#FFEC73", alpha=0.5)

        # axlogo is represented once, in the aggregate panel, instead of
        # receiving an annotation placed over its tiny image axes.
        assert sum("axlogo" in text.get_text() for text in overlay.texts) == 1

        ax_left, ax_bottom, ax_width, ax_height = axes["ax"].get_position().bounds
        info_x, info_y = info.get_position()
        assert ax_left < info_x < ax_left + ax_width
        assert ax_bottom < info_y < ax_bottom + ax_height
    finally:
        plt.close(raw_fig)


def test_numbered_axes_get_top_bottom_figure_edge_dimensions_in_order():
    raw_fig = plt.figure(figsize=(4.0, 4.0))
    axes = {
        "ax0": raw_fig.add_axes([0.20, 0.70, 0.70, 0.20]),
        "ax1": raw_fig.add_axes([0.20, 0.40, 0.70, 0.20]),
        "ax2": raw_fig.add_axes([0.20, 0.10, 0.70, 0.20]),
    }
    wrapper = SimpleNamespace(
        fig=raw_fig,
        axes=axes,
        logger=SimpleNamespace(debug=lambda *args, **kwargs: None),
    )

    try:
        draw_design_reference(wrapper)
        overlay = raw_fig.axes[-1]

        labels = {}
        for text in overlay.texts:
            if text.get_color() == "#1F21E9":
                labels.setdefault(text.get_text(), []).append(
                    round(text.get_position()[0], 3)
                )
        # 4 inches = 10.16 cm; ax0 is the rightmost column and ax1, ax2,
        # ... are shifted a further 0.040 figure fraction to the left.
        assert sorted(labels["1.016 cm"]) == [0.894, 0.974]
        assert sorted(labels["7.112 cm"]) == [0.894, 0.974]
        assert labels["4.064 cm"] == [0.934, 0.934]
    finally:
        plt.close(raw_fig)


def test_primary_top_and_bottom_margins_ride_the_axes_right_border():
    """They used to sit on the left spine, where the layout panel begins.

    They follow the primary axes' right border now -- the axes' border, not
    the page's -- so a narrower axes carries them inward with it.  The left
    inset is horizontal and keeps its own place.
    """
    def margin_label_x(rect):
        raw_fig = plt.figure(figsize=(3.3, 2.75))
        wrapper = SimpleNamespace(
            fig=raw_fig,
            axes={"ax": raw_fig.add_axes(rect)},
            logger=SimpleNamespace(debug=lambda *args, **kwargs: None),
        )
        try:
            draw_design_reference(wrapper)
            overlay = raw_fig.axes[-1]
            by_x = {}
            for text in overlay.texts:
                if text.get_color() == "#1F21E9" and text.get_rotation() == 90:
                    by_x.setdefault(round(text.get_position()[0], 3), []).append(text)
            return by_x
        finally:
            plt.close(raw_fig)

    by_x = margin_label_x([0.142, 0.168, 0.680, 0.775])
    # (left 0.142 + width 0.680) - corner_gap 0.018 + label_gap 0.008
    assert sorted(by_x) == [0.043, 0.812]
    margins = by_x[0.812]
    assert len(margins) == 2, "the top and bottom insets"
    assert {text.get_ha() for text in margins} == {"left"}, "label right of the line"
    assert len(by_x[0.043]) == 1, "the total-height marker keeps the far left"

    # Halve the axes and the pair comes with it; a page-anchored column would
    # have stayed put.
    narrow = margin_label_x([0.142, 0.168, 0.340, 0.775])
    assert sorted(narrow) == [0.043, 0.472]


def _vertical_dimension_labels(axes_rects, debug_config=None):
    """Map x -> the rotated dimension labels drawn there, for one layout."""
    raw_fig = plt.figure(figsize=(3.3, 2.75))
    wrapper = SimpleNamespace(
        fig=raw_fig,
        axes={name: raw_fig.add_axes(rect) for name, rect in axes_rects.items()},
        _debug_config=debug_config or {},
        logger=SimpleNamespace(
            debug=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
        ),
    )
    try:
        draw_design_reference(wrapper)
        by_x = {}
        for text in raw_fig.axes[-1].texts:
            if text.get_color() == "#1F21E9" and text.get_rotation() == 90:
                by_x.setdefault(round(text.get_position()[0], 3), []).append(text)
        return by_x
    finally:
        plt.close(raw_fig)


LAYOUT_WITH_LOGO = {
    "axlogo": [0.010, 0.010, 0.060, 0.072],
    "ax": [0.142, 0.168, 0.680, 0.775],
    "axc": [0.827, 0.182, 0.036, 0.746],
}


def test_every_axes_gets_its_own_top_and_bottom_margin_dimensions():
    """Not just the primary: a helper axes' offsets are part of the design."""
    by_x = _vertical_dimension_labels(LAYOUT_WITH_LOGO)

    # the total-height marker, then one column per axes on its right border
    # (right - corner_gap 0.018 + label_gap 0.008).  axlogo is excluded.
    assert sorted(by_x) == [0.043, 0.812, 0.853]
    assert [len(by_x[x]) for x in (0.812, 0.853)] == [2, 2]
    assert len(by_x[0.043]) == 1


def test_margins_exclude_keeps_the_logo_plate_out_of_the_columns():
    """A logo is placed by eye, not designed against the page edges.

    It is the one axes the shipped default leaves alone; emptying the list
    brings it back, so this is a card setting and not a rule in the code.
    """
    default = _vertical_dimension_labels(LAYOUT_WITH_LOGO)
    included = _vertical_dimension_labels(
        LAYOUT_WITH_LOGO, {"margins": {"exclude": []}}
    )

    assert sorted(default) == [0.043, 0.812, 0.853]
    # axlogo's own border is 0.017 from the height marker, so it steps clear
    # to 0.092 rather than landing on it.
    assert sorted(included) == [0.043, 0.100, 0.812, 0.853]
    assert len(included[0.100]) == 2


def test_axes_that_share_a_right_border_step_apart():
    """As ax and axr do on the rectRatio card -- one column would overlap."""
    by_x = _vertical_dimension_labels({
        "ax": [0.140, 0.333, 0.840, 0.620],
        "axr": [0.140, 0.168, 0.840, 0.155],
    })
    # both end at 0.980; the second steps in by marker_step 0.040
    assert sorted(by_x) == [0.043, 0.930, 0.970]
    assert len(by_x[0.970]) == 2 and len(by_x[0.930]) == 2

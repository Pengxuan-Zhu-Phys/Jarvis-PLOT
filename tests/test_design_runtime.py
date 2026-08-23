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

        # axlogo is represented once, in the aggregate panel, instead of
        # receiving an annotation placed over its tiny image axes.
        assert sum("axlogo" in text.get_text() for text in overlay.texts) == 1

        ax_left, ax_bottom, ax_width, ax_height = axes["ax"].get_position().bounds
        info_x, info_y = info.get_position()
        assert ax_left < info_x < ax_left + ax_width
        assert ax_bottom < info_y < ax_bottom + ax_height
    finally:
        plt.close(raw_fig)

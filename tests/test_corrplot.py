"""``method: corrplot`` and ``type: correlation_matrix``.

Three things are worth guarding here, and they are not the drawing:

1. **The config/render split.** Ordering is resolved before the figure exists,
   because the tick labels are. A regression that moved it back to draw time
   would still produce a plausible-looking matrix with every label on the
   wrong column, and no exception anywhere.
2. **The reserved card's contract.** The figure size is solved from one
   matrix, so a second layer -- or a ``scatter`` layer that accepts corrplot's
   formals and discards them -- has to be refused rather than drawn.
3. **The derivation.** Cell and font stay put; the figure moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jarvisplot.Figure.corr_layout import max_label_mm, solve_corr_geometry
from jarvisplot.Figure.corr_layout_diamond import diamond_extent, solve_diamond_geometry
from jarvisplot.core_runtime import _attach_corr_debug, _corr_debug_lines
from jarvisplot.Figure.corr_order import ORDERS, order_columns
from jarvisplot.Figure.correlation_runtime import (
    correlation,
    pearson_matrix,
    pearson_pvalues,
)
from jarvisplot.Figure.corrplot_runtime import GLYPHS, draw_corrplot
from jarvisplot.Figure.figure_types import KNOWN_FIGURE_TYPES, expand_figure_type
from jarvisplot.Figure.method_registry import METHOD_DISPATCH


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def _blocked_frame(rows: int = 200, seed: int = 7) -> pd.DataFrame:
    """Two planted blocks with an uncorrelated column wedged between them."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=rows)
    b = rng.normal(size=rows)
    noise = lambda: rng.normal(scale=0.25, size=rows)  # noqa: E731
    return pd.DataFrame(
        {
            "a1": a + noise(),
            "loner": rng.normal(size=rows),
            "b1": b + noise(),
            "a2": a + noise(),
            "b2": b + noise(),
        }
    )


def _long_table(columns=("x", "y", "z"), n_rows: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({name: rng.normal(size=n_rows) for name in columns})
    return correlation(frame, {"columns": list(columns)})


def _axes(n: int):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)          # row 0 on top, as the card sets it
    return ax


# --------------------------------------------------------------------------- #
# ordering: resolved from the matrix, before anything is drawn
# --------------------------------------------------------------------------- #


def test_hclust_brings_a_planted_block_together():
    frame = _blocked_frame()
    names = list(frame.columns)
    matrix = pearson_matrix(frame, names)
    ordered, _blocks = order_columns(matrix, names, "hclust")

    assert sorted(ordered) == sorted(names)
    positions = {name: index for index, name in enumerate(ordered)}
    assert abs(positions["a1"] - positions["a2"]) == 1
    assert abs(positions["b1"] - positions["b2"]) == 1


def test_addrect_blocks_are_contiguous_and_cover_every_position():
    frame = _blocked_frame()
    names = list(frame.columns)
    matrix = pearson_matrix(frame, names)
    ordered, blocks = order_columns(matrix, names, "hclust", addrect=3)

    assert len(blocks) == 3
    assert blocks[0][0] == 0
    assert blocks[-1][1] == len(ordered) - 1
    for (_, end), (start, _) in zip(blocks, blocks[1:]):
        assert start == end + 1, "a box has to enclose adjacent positions"


def test_addrect_without_a_tree_is_refused_rather_than_ignored():
    frame = _blocked_frame()
    names = list(frame.columns)
    matrix = pearson_matrix(frame, names)
    with pytest.raises(ValueError, match="order: hclust"):
        order_columns(matrix, names, "AOE", addrect=2)


def test_original_and_alphabet_need_no_matrix_at_all():
    """The two orders that read no data must not ask for any."""
    names = ["b", "a", "c"]
    assert order_columns(None, names, "original") == (["b", "a", "c"], None)
    assert order_columns(None, names, "alphabet") == (["a", "b", "c"], None)


def test_an_unknown_order_is_an_error_not_a_fallback():
    with pytest.raises(ValueError, match="corrplot order must be one of"):
        order_columns(None, ["a", "b"], "clustered")


def test_every_advertised_order_is_reachable():
    frame = _blocked_frame()
    names = list(frame.columns)
    matrix = pearson_matrix(frame, names)
    for order in ORDERS:
        ordered, _ = order_columns(matrix, names, order)
        assert sorted(ordered) == sorted(names), order


def test_eigenvector_orders_are_reproducible():
    """AOE/FPC depend on an eigenvector sign LAPACK is free to flip."""
    frame = _blocked_frame()
    names = list(frame.columns)
    matrix = pearson_matrix(frame, names)
    for order in ("AOE", "FPC"):
        first, _ = order_columns(matrix, names, order)
        again, _ = order_columns(matrix.loc[names, names].copy(), names, order)
        assert first == again, order


# --------------------------------------------------------------------------- #
# significance, without a p.mat
# --------------------------------------------------------------------------- #


def test_pvalues_come_from_rho_and_the_pair_count():
    # r = 0.5 on n = 100 is significant; the same r on n = 6 is not.
    p_large, p_small = pearson_pvalues([0.5, 0.5], [100, 6])
    assert p_large < 0.05
    assert p_small > 0.05


def test_a_pair_with_no_test_is_nan_not_a_confident_zero():
    assert np.isnan(pearson_pvalues([0.9], [2])[0])
    assert np.isnan(pearson_pvalues([np.nan], [100])[0])


# --------------------------------------------------------------------------- #
# geometry: the cell and the font stay put, the figure moves
# --------------------------------------------------------------------------- #


_GEOMETRY = {
    "units": "mm",
    "max_width": 170.0,
    "cell": 4.2,
    "cell_min": 1.6,
    "margin": {"top": 4.0, "right": 11.0, "bottom": 11.0, "left": 11.0},
    "colorbar": {"width": 2.6, "gap": 0.42, "inset": 5.0},
    "logo": {"width": 5.0, "height": 5.0, "offset": 0.838},
}


def test_the_figure_grows_with_n_while_the_cell_does_not():
    # n is the only input to the size now that the margins are authored, but
    # the labels are still passed so the measuring path is exercised.
    labels = ["v{:02d}".format(i) for i in range(40)]
    small = solve_corr_geometry(6, geometry=_GEOMETRY, x_labels=labels[:6], y_labels=labels[:6])
    large = solve_corr_geometry(12, geometry=_GEOMETRY, x_labels=labels[:12], y_labels=labels[:12])

    assert small.cell_mm == large.cell_mm == pytest.approx(4.2)
    assert large.width_mm - small.width_mm == pytest.approx(6 * 4.2, abs=1e-6)
    assert not small.clamped and not large.clamped


def test_the_solve_reaches_the_overlay_without_switching_it_on():
    # `Figures[].debug` is both the switch and the override, and the mapping
    # form defaults to on -- so the channel has to write `show` back explicitly
    # or every correlation figure would draw the design overlay.
    geom = solve_corr_geometry(9, geometry=_GEOMETRY, x_labels=["abc"] * 9,
                               y_labels=["abc"] * 9)
    lines = _corr_debug_lines(geom, 9)
    assert any("9 vars" in line for line in lines)
    assert any("cell 4.200 mm" in line for line in lines)

    off = {}
    _attach_corr_debug(off, lines)
    assert off["debug"]["show"] is False
    assert off["debug"]["solved"]["lines"] == lines

    on = {"debug": True}
    _attach_corr_debug(on, lines)
    assert on["debug"]["show"] is True

    kept = {"debug": {"show": True, "panel": {"show": False}}}
    _attach_corr_debug(kept, lines)
    assert kept["debug"]["panel"] == {"show": False}


def test_the_solve_lines_land_in_a_group_the_defaults_define():
    # Anything outside DEFAULT_DEBUG is reported as an unknown key and dropped,
    # which would lose the block in silence.
    from jarvisplot.Figure.debug_config import DEFAULT_DEBUG, merge_debug_config

    merged, problems = merge_debug_config(
        DEFAULT_DEBUG, {"solved": {"lines": ["a", "b"]}}
    )
    assert problems == []
    assert merged["solved"]["lines"] == ["a", "b"]
    assert DEFAULT_DEBUG["solved"]["lines"] == []


def test_the_panel_sits_the_same_distance_from_the_left_and_bottom_edges():
    # Both bands hold the same names, so an unequal corner reads as a mistake
    # in the figure.  `tl.pos: l` is the asymmetric case: y names to budget for,
    # no x names at all, and the bottom margin still matches the left one.
    labels = ["a_long_variable_name"] * 8
    both = solve_corr_geometry(8, geometry=_GEOMETRY, x_labels=labels, y_labels=labels)
    y_only = solve_corr_geometry(8, geometry=_GEOMETRY, y_labels=labels)
    for geom in (both, y_only):
        left_mm = geom.panel_rect[0] * geom.width_mm
        bottom_mm = geom.panel_rect[1] * geom.height_mm
        assert left_mm == pytest.approx(11.0, abs=1e-6)
        assert bottom_mm == pytest.approx(11.0, abs=1e-6)


def test_the_figure_is_the_same_shape_whatever_the_names_are():
    # The point of authored margins: the same card on two datasets is the same
    # card.  A name too wide for its band is reported and then printed past it.
    short = solve_corr_geometry(8, geometry=_GEOMETRY, x_labels=["a1"] * 8, y_labels=["a1"] * 8)
    long_ = solve_corr_geometry(8, geometry=_GEOMETRY,
                                x_labels=["a_very_long_variable_name"] * 8,
                                y_labels=["a_very_long_variable_name"] * 8)
    bare = solve_corr_geometry(8, geometry=_GEOMETRY)
    assert short.width_mm == pytest.approx(long_.width_mm, abs=1e-6)
    assert short.height_mm == pytest.approx(bare.height_mm, abs=1e-6)
    assert not short.notes
    assert any("print past it" in note for note in long_.notes)


def test_margin_fit_grows_the_corner_to_hold_the_widest_name():
    # The opt-in: `margin.fit` turns the overrun note into an action.  It needs
    # no trial render -- the names are measured exactly, with the tick font at
    # the tick size, so where the text ends is known before anything is drawn.
    fit = {**_GEOMETRY, "margin": {**_GEOMETRY["margin"], "fit": True, "slack": 1.0}}
    labels = ["a_long_variable_name"] * 8
    geom = solve_corr_geometry(8, geometry=fit, x_labels=labels, y_labels=labels,
                               label_pad_mm=0.6)
    # Where the text ends, plus the slack: fitting to the last glyph exactly
    # puts it on the edge of the paper, which reads as nearly cut off.
    need = 0.6 + max_label_mm(labels, size_pt=5.5) + 1.0
    left_mm = geom.panel_rect[0] * geom.width_mm
    bottom_mm = geom.panel_rect[1] * geom.height_mm
    assert left_mm == pytest.approx(need, abs=1e-6)
    assert bottom_mm == pytest.approx(need, abs=1e-6)   # still equal on both
    assert any("margin.fit" in note for note in geom.notes)

    # One corner for both bands, so hiding the x names cannot shrink it below
    # what the y names still need.
    y_only = solve_corr_geometry(8, geometry=fit, y_labels=labels, label_pad_mm=0.6)
    assert y_only.panel_rect[1] * y_only.height_mm == pytest.approx(need, abs=1e-6)


def test_margin_fit_only_ever_grows():
    # The card's margin is the floor: a short name does not pull the panel in
    # toward the edge, which would make a small matrix a different shape again.
    fit = {**_GEOMETRY, "margin": {**_GEOMETRY["margin"], "fit": True, "slack": 1.0}}
    short = solve_corr_geometry(8, geometry=fit, x_labels=["a1"] * 8,
                                y_labels=["a1"] * 8, label_pad_mm=0.6)
    plain = solve_corr_geometry(8, geometry=_GEOMETRY, x_labels=["a1"] * 8,
                                y_labels=["a1"] * 8, label_pad_mm=0.6)
    assert short.width_mm == pytest.approx(plain.width_mm, abs=1e-6)
    assert short.panel_rect[0] * short.width_mm == pytest.approx(11.0, abs=1e-6)
    assert not short.notes


def test_the_margin_never_prints_the_panel_on_top_of_the_badge():
    # The one thing the solver still enforces about the margins.
    narrow = {**_GEOMETRY, "margin": {**_GEOMETRY["margin"], "left": 1.0, "bottom": 1.0}}
    geom = solve_corr_geometry(8, geometry=narrow)
    floor = _GEOMETRY["logo"]["offset"] + _GEOMETRY["logo"]["height"]
    assert geom.panel_rect[0] * geom.width_mm == pytest.approx(floor, abs=1e-6)
    assert geom.panel_rect[1] * geom.height_mm == pytest.approx(floor, abs=1e-6)


def test_the_colorbar_is_a_key_not_a_column_of_the_matrix():
    # 0.42 mm off the panel, 1 cm shorter than it, and centred on it.
    geom = solve_corr_geometry(8, geometry=_GEOMETRY)
    panel_right = (geom.panel_rect[0] + geom.panel_rect[2]) * geom.width_mm
    assert geom.colorbar_rect[0] * geom.width_mm - panel_right == pytest.approx(0.42, abs=1e-6)
    assert geom.colorbar_rect[2] * geom.width_mm == pytest.approx(2.6, abs=1e-6)

    bar_mm = geom.colorbar_rect[3] * geom.height_mm
    assert geom.panel_mm - bar_mm == pytest.approx(10.0, abs=1e-6)
    below = geom.colorbar_rect[1] * geom.height_mm - geom.panel_rect[1] * geom.height_mm
    assert below == pytest.approx(5.0, abs=1e-6)


def test_the_top_margin_is_the_title_band_and_nothing_else():
    geom = solve_corr_geometry(8, geometry=_GEOMETRY)
    top_mm = geom.height_mm - (geom.panel_rect[1] + geom.panel_rect[3]) * geom.height_mm
    assert top_mm == pytest.approx(4.0, abs=1e-6)


def test_the_badge_keeps_the_house_offset_on_a_figure_of_any_shape():
    # 0.0838 cm from the left and bottom edges, on every card in the repo
    # (STYLE_SCHEMA.md, "The `axlogo` Discipline").  The figure is not square,
    # so the two rect fractions differ while the printed distances do not.
    geom = solve_corr_geometry(9, geometry=_GEOMETRY)
    left_mm = geom.logo_rect[0] * geom.width_mm
    bottom_mm = geom.logo_rect[1] * geom.height_mm
    assert left_mm == pytest.approx(0.838, abs=1e-6)
    assert bottom_mm == pytest.approx(0.838, abs=1e-6)
    assert geom.logo_rect[0] != pytest.approx(geom.logo_rect[1])


def test_the_panel_is_square_so_a_circle_glyph_is_round():
    geom = solve_corr_geometry(9, geometry=_GEOMETRY, x_labels=["abc"] * 9, y_labels=["abc"] * 9)
    width_mm = geom.panel_rect[2] * geom.width_mm
    height_mm = geom.panel_rect[3] * geom.height_mm
    assert width_mm == pytest.approx(height_mm, abs=1e-6)


def test_past_the_page_the_cell_shrinks_and_says_so_but_the_font_never_does():
    labels = ["a_rather_long_name_{}".format(i) for i in range(40)]
    geom = solve_corr_geometry(40, geometry=_GEOMETRY, x_labels=labels, y_labels=labels)

    assert geom.clamped
    assert geom.width_mm == pytest.approx(170.0, abs=1e-6)
    assert geom.cell_mm < 4.2
    assert any("Font sizes are unchanged" in note for note in geom.notes)


def test_a_colorbar_label_wider_than_its_margin_is_reported_not_budgeted():
    # The margin is the card's call, so an overrun buys a note, not a wider
    # figure -- otherwise `$\\rho$` would quietly move the whole panel.
    plain = solve_corr_geometry(8, geometry=_GEOMETRY, colorbar_labels=["-1", "1"])
    titled = solve_corr_geometry(
        8, geometry=_GEOMETRY, colorbar_labels=["-1", "1"], colorbar_title=r"$\rho$"
    )
    assert titled.width_mm == pytest.approx(plain.width_mm, abs=1e-6)

    crowded = solve_corr_geometry(
        8, geometry=_GEOMETRY,
        colorbar_labels=["-0.000000001"], colorbar_title="a very long colorbar title",
    )
    assert crowded.width_mm == pytest.approx(plain.width_mm, abs=1e-6)
    assert any("colorbar labels" in note for note in crowded.notes)


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #


def test_corrplot_is_a_registered_method_with_an_optional_coordinate_contract():
    from jarvisplot.method_contracts import contract_for

    assert METHOD_DISPATCH["corrplot"] == "corrplot"
    contract = contract_for("corrplot")
    assert tuple(contract["required"]) == ()
    assert set(contract["optional"]) == {"x", "y", "c"}


@pytest.mark.parametrize("glyph", GLYPHS)
def test_every_glyph_draws_something(glyph):
    table = _long_table()
    ax = _axes(3)
    result = draw_corrplot(ax, __df__=table, method=glyph)
    if glyph == "number":
        assert ax.texts, "method: number is the coefficient instead of a glyph"
    else:
        assert result is not None and len(result.get_paths()) == len(table)


def test_the_square_glyphs_are_built_as_bare_corners():
    # A Polygon patch per cell costs ~0.14 s at 100 variables and is thrown
    # away for its path; PolyCollection takes the corners directly.  Same four
    # corners, in the same order.
    from jarvisplot.Figure.corrplot_runtime import _glyph_vertices

    verts = _glyph_vertices("color", [0.0, 1.0], [0.0, 2.0], np.array([0.5, -0.25]), 0.9)
    assert verts.shape == (2, 4, 2)
    assert np.allclose(
        verts[0], [[-0.45, -0.45], [0.45, -0.45], [0.45, 0.45], [-0.45, 0.45]]
    )
    # `color` fills the cell whatever rho is; `square` carries |rho| in its
    # area, so its side is the square root of it.
    assert np.allclose(verts[1], verts[0] + [1.0, 2.0])
    square = _glyph_vertices("square", [0.0], [0.0], np.array([0.25]), 1.0)
    assert square[0][:, 0].max() * 2 == pytest.approx(np.sqrt(0.25))


def test_the_ellipse_fills_the_box_the_other_glyphs_do():
    # Rotated 45 degrees, the ellipse family's bounding box is
    # `glyph.scale * ellipse.scale / sqrt(2)` on a side *whatever rho is* --
    # the shape changes, the box does not.  So `ellipse.scale: 1.4` is what
    # makes the widest ellipse reach as far as a circle at |rho| = 1 does, and
    # the constant box is why it still cannot spill into the next cell.
    table = _long_table()
    drawn = draw_corrplot(_axes(3), __df__=table, method="ellipse")
    boxes = [p.get_extents() for p in drawn.get_paths()]
    side = 0.9 * 1.4 / np.sqrt(2.0)
    assert len({round(b.width, 2) for b in boxes}) == 1
    for box in boxes:
        assert box.width == pytest.approx(side, abs=2e-3)
        assert box.height == pytest.approx(side, abs=2e-3)
        assert box.width < 1.0

    bare = draw_corrplot(_axes(3), __df__=table, method="ellipse",
                         **{"ellipse.scale": 1.0})
    assert bare.get_paths()[0].get_extents().width == pytest.approx(
        0.9 / np.sqrt(2.0), abs=2e-3
    )


def test_the_glyph_edge_is_coloured_by_the_sign_of_the_cell():
    # A circle or square carries |rho| in its area, so a weak cell both shrinks
    # and fades to nearly white.  `outline: sign` draws its edge in the end of
    # the scale its sign points at, which is the only thing left on that cell
    # saying which way it went.
    import matplotlib as mpl

    table = _long_table()
    drawn = draw_corrplot(
        _axes(3), __df__=table, method="circle",
        outline="sign", cmap="RdBu", vmin=-1.0, vmax=1.0,
    )
    scale = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-1.0, 1.0), cmap="RdBu")
    expected = [scale.to_rgba(1.0 if value >= 0 else -1.0) for value in table["rho"]]
    assert np.allclose(drawn.get_edgecolor(), expected)
    assert np.allclose(drawn.get_linewidth(), 0.3)          # outline.lwd default
    assert np.allclose(
        draw_corrplot(_axes(3), __df__=table, method="circle", outline="sign",
                      **{"outline.lwd": 0.2}).get_linewidth(),
        0.2,
    )

    # R's own spellings still mean what they meant.
    plain = draw_corrplot(_axes(3), __df__=table, method="circle", outline=True)
    assert np.allclose(plain.get_edgecolor(), mpl.colors.to_rgba("#21171A"))
    named = draw_corrplot(_axes(3), __df__=table, method="circle", outline="#00FF00")
    assert np.allclose(named.get_edgecolor(), mpl.colors.to_rgba("#00FF00"))
    assert np.allclose(
        draw_corrplot(_axes(3), __df__=table, method="circle").get_linewidth(), 0.0
    )


def test_a_glyph_that_never_shrinks_gets_no_sign_edge():
    # `color` and `shade` cover the cell whatever rho is, so the sign edge
    # would be a second grid line in two loud colours rather than a mark.
    table = _long_table()
    for glyph in ("color", "shade"):
        filled = draw_corrplot(_axes(3), __df__=table, method=glyph, outline="sign")
        assert np.allclose(filled.get_linewidth(), 0.0), glyph
    # A colour asked for by name is still honoured there -- only `sign` opts out.
    edged = draw_corrplot(_axes(3), __df__=table, method="color", outline="#00FF00")
    assert np.allclose(edged.get_linewidth(), 0.3)


def test_the_grid_is_one_closed_loop_per_cell():
    # Four separate edges per cell is 40,000 paths at n = 100 for the same
    # picture.  Each loop closes, so no edge goes missing in the trade.
    table = _long_table()
    ax = _axes(3)
    draw_corrplot(ax, __df__=table, method="color", **{"addgrid.col": "#C2C2C2"})
    grid = min(ax.collections, key=lambda c: c.get_zorder())
    paths = grid.get_paths()
    assert len(paths) == len(table)
    for path in paths:
        assert len(path.vertices) == 5
        assert np.allclose(path.vertices[0], path.vertices[-1])


def test_the_printed_triangle_follows_the_picture_not_the_array():
    table = _long_table()
    ax = _axes(3)
    upper = draw_corrplot(ax, __df__=table, method="square", type="upper", diag=False)
    # column > row, with row 0 at the top: 3 cells of a 3x3.
    assert len(upper.get_paths()) == 3


def test_a_stray_matplotlib_argument_is_refused_rather_than_discarded():
    table = _long_table()
    with pytest.raises(TypeError, match="corrplot does not take linestyle"):
        draw_corrplot(_axes(3), __df__=table, linestyle="--")


def test_formals_resolved_at_config_time_are_accepted_and_not_forwarded():
    """They arrive because the author wrote them; the renderer must not choke."""
    table = _long_table()
    ax = _axes(3)
    draw_corrplot(ax, __df__=table, order="hclust", addrect=2, col="RdBu",
                  **{"hclust.method": "average", "tl.pos": "lt", "tl.cex": 1.0})


def test_no_table_says_which_transform_is_missing():
    with pytest.raises(ValueError, match="correlation"):
        draw_corrplot(_axes(3))


def test_significance_marks_only_when_a_level_is_given():
    from matplotlib.collections import LineCollection

    def marks(ax):
        return [
            c for c in ax.collections
            if isinstance(c, LineCollection) and len(c.get_segments()) % 2 == 0
            and c.get_zorder() > 20
        ]

    table = _long_table(n_rows=8)
    plain = _axes(3)
    draw_corrplot(plain, __df__=table, method="circle")
    assert not marks(plain)

    marked = _axes(3)
    draw_corrplot(marked, __df__=table, method="circle", **{"sig.level": 0.05})
    assert marks(marked), "sig.level should mark the insignificant cells"
    # drawn, not typed: nothing about the mark is text
    assert not marked.texts


def test_the_insignificance_mark_is_made_out_of_the_cell():
    # R's `pch` is a plotted symbol; set as a *character* it is sized in points,
    # so it neither fills the cell nor shrinks with it.  Two lines between
    # opposite corners are always exactly as big as what they mark -- a cross
    # on the square cell, a plus on the diamond, which is the same construction
    # seen from 45 degrees.
    from jarvisplot.Figure.corrplot_runtime import (
        _CORNERS, _DIAMOND_CORNERS, _insignificant_marks,
    )

    ix, iy = np.array([2.0]), np.array([5.0])
    square = _insignificant_marks(ix, iy, _CORNERS, 1.0)
    assert square.shape == (2, 2, 2)
    assert np.allclose(square[0], [[1.5, 4.5], [2.5, 5.5]])     # corner to corner
    assert np.allclose(square[1], [[2.5, 4.5], [1.5, 5.5]])
    # the rotated cell's corners are its own, so the mark turns with it
    plus = _insignificant_marks(ix, iy, _DIAMOND_CORNERS, 1.0)
    assert np.allclose(plus[0], [[2.0, 4.5], [2.0, 5.5]])
    assert np.allclose(plus[1], [[2.5, 5.0], [1.5, 5.0]])
    # and `pch.cex` scales it about the cell's centre
    assert np.allclose(_insignificant_marks(ix, iy, _CORNERS, 0.5)[0],
                       [[1.75, 4.75], [2.25, 5.25]])


# --------------------------------------------------------------------------- #
# the type: macro
# --------------------------------------------------------------------------- #


def test_correlation_matrix_is_a_known_type():
    assert "correlation_matrix" in KNOWN_FIGURE_TYPES


def test_the_macro_lowers_to_one_corrplot_layer_on_the_reserved_axes():
    out = expand_figure_type(
        {
            "name": "corr",
            "type": "correlation_matrix",
            "data": "samples",
            "variables": {"exclude": ["weight"]},
            "corrplot": {"method": "circle", "order": "hclust", "addrect": 3},
            "colorbar": {"label": r"$\rho$"},
        }
    )
    assert "type" not in out
    assert out["style"] == ["corrplot", "matrix"]
    assert out["frame"]["axccorr"]["label"]["ylabel"] == r"$\rho$"

    (layer,) = out["layers"]
    assert layer["axes"] == "axcorr"
    assert layer["colorbar"] == "axccorr"
    assert layer["method"] == "corrplot"
    assert layer["style"] == {"method": "circle", "order": "hclust", "addrect": 3}
    assert layer["data"][0]["transform"] == [{"correlation": {"exclude": ["weight"]}}]

    # None of the solved keys may be authored here: prebuild_correlations owns
    # them, and anything written would be silently overwritten.
    assert "figure" not in out["frame"]
    assert "axes" not in out["frame"]
    assert "coordinates" not in layer


def test_a_bare_list_of_variables_is_the_explicit_order():
    out = expand_figure_type(
        {"name": "c", "type": "correlation_matrix", "data": "s", "variables": ["b", "a"]}
    )
    assert out["layers"][0]["data"][0]["transform"] == [
        {"correlation": {"columns": ["b", "a"]}}
    ]


def test_the_macro_defaults_to_the_reserved_card_not_the_rect_one():
    out = expand_figure_type({"name": "c", "type": "correlation_matrix", "data": "s"})
    assert out["style"] == ["corrplot", "matrix"]


def test_column_selection_in_the_wrong_block_is_named():
    with pytest.raises(ValueError, match="Column selection goes in `variables`"):
        expand_figure_type(
            {
                "name": "c",
                "type": "correlation_matrix",
                "data": "s",
                "correlation": {"exclude": ["w"]},
            }
        )


def test_a_misspelled_selector_is_refused():
    with pytest.raises(ValueError, match="columns, regex or exclude"):
        expand_figure_type(
            {"name": "c", "type": "correlation_matrix", "data": "s", "variables": {"excluded": []}}
        )


def test_the_macro_needs_a_source():
    with pytest.raises(ValueError, match="requires a data source"):
        expand_figure_type({"name": "c", "type": "correlation_matrix"})


# --------------------------------------------------------------------------- #
# the diamond card: one triangle, turned 45 degrees
# --------------------------------------------------------------------------- #

_DIAMOND = {
    "units": "mm",
    "pitch": 4.2,
    "pitch_min": 2.6,
    "max_height": 247.0,
    "max_width": 170.0,
    "labels": {"gap": 1.4, "width": None},
    "margin": {"fit": True, "slack": 1.0, "top": 4.0, "right": 4.0,
               "bottom": 4.0, "left": 4.0},
    "colorbar": {"width": 2.6, "length": 0.22, "offset": 10.0},
    "logo": {"width": 5.0, "height": 5.0, "offset": 0.838},
}


def test_the_map_gathers_a_variables_pairs_into_one_v():
    # The property the whole card rests on: every cell variable k appears in
    # satisfies `v - |u| == k` or `v + |u| == k`, so its pairs are two rays
    # meeting at (0, k) -- which is exactly where its name is printed.
    from jarvisplot.Figure.corrplot_runtime import _diamond_uv

    n = 9
    ix, iy = np.meshgrid(np.arange(n), np.arange(n))
    ix, iy = ix.ravel(), iy.ravel()
    keep = ix > iy                                    # the upper triangle
    u, v = _diamond_uv(ix[keep], iy[keep])
    for k in range(n):
        mine = (ix[keep] == k) | (iy[keep] == k)
        # each of k's cells sits on one of the two rays out of (0, k)
        on_ray = np.isclose(v[mine] - u[mine], k) | np.isclose(v[mine] + u[mine], k)
        assert on_ray.all(), k
    # and nothing else is on those rays
    assert np.isclose(u, 0.0).sum() == 0              # no self-pairs drawn


def test_the_right_hand_figure_is_the_mirror_and_nothing_else():
    labels = ["v%02d" % i for i in range(12)]
    left = solve_diamond_geometry(12, geometry=_DIAMOND, labels=labels, side="left")
    right = solve_diamond_geometry(12, geometry=_DIAMOND, labels=labels, side="right")

    assert left.width_mm == pytest.approx(right.width_mm, abs=1e-9)
    assert left.height_mm == pytest.approx(right.height_mm, abs=1e-9)
    assert left.panel_rect[2:] == pytest.approx(right.panel_rect[2:], abs=1e-12)
    # u negated, and the two bands swap sides: the names are always the band
    # against the panel's own edge, and the bar is always the other one.
    assert left.xlim == (0.0, 6.0) and right.xlim == (-6.0, -0.0)
    assert left.panel_rect[0] * left.width_mm == pytest.approx(
        left.label_block_mm, abs=1e-9
    )
    right_gap = (1.0 - right.panel_rect[0] - right.panel_rect[2]) * right.width_mm
    assert right_gap == pytest.approx(right.label_block_mm, abs=1e-9)
    assert left.colorbar_rect[0] > right.colorbar_rect[0]


def test_the_panel_is_a_row_per_variable_and_half_as_wide():
    geom = solve_diamond_geometry(21, geometry=_DIAMOND, labels=["v"] * 21)
    lo_u, hi_u, lo_v, hi_v = diamond_extent(21)
    assert geom.panel_h_mm == pytest.approx((hi_v - lo_v) * geom.pitch_mm, abs=1e-9)
    assert geom.panel_w_mm == pytest.approx((hi_u - lo_u) * geom.pitch_mm, abs=1e-9)
    # (n-1) tall against n/2 wide: about twice, exactly 2(n-1)/n.
    assert geom.panel_h_mm / geom.panel_w_mm == pytest.approx(2 * 20 / 21, abs=1e-9)


def test_the_diamond_runs_out_of_page_height_not_width():
    # The opposite bound from the square card, and the clamp says which.
    tall = solve_diamond_geometry(70, geometry=_DIAMOND, labels=["v%02d" % i for i in range(70)])
    assert tall.clamped
    assert tall.height_mm == pytest.approx(247.0, abs=1e-6)
    assert tall.width_mm < 170.0
    assert tall.pitch_mm < 4.2
    assert any("height pinned" in note for note in tall.notes)


def test_below_the_pitch_floor_the_names_would_collide():
    huge = solve_diamond_geometry(140, geometry=_DIAMOND, labels=["v"] * 140)
    assert huge.pitch_mm == pytest.approx(2.6, abs=1e-9)
    assert huge.height_mm > 247.0
    assert any("the names collide" in note for note in huge.notes)


def test_the_label_column_is_measured_and_the_stripe_stops_at_the_text():
    short = solve_diamond_geometry(8, geometry=_DIAMOND, labels=["a1"] * 8)
    long_ = solve_diamond_geometry(8, geometry=_DIAMOND, labels=["a_long_name"] * 8)
    # At a 4 mm page margin even a two-character name asks for more, so the
    # floor is the margin and the column is whatever the names need above it.
    assert short.label_block_mm >= 4.0
    assert long_.label_block_mm > short.label_block_mm
    # The band the stripes use is the text, not the column: the page margin
    # under a short name stays white instead of bleeding to the paper edge.
    assert short.label_text_mm < short.label_block_mm
    assert long_.label_text_mm == pytest.approx(long_.label_block_mm - 1.0, abs=1e-9)


def test_a_block_of_variables_is_boxed_as_a_triangle():
    from jarvisplot.Figure.corrplot_runtime import _diamond_block

    box = _diamond_block(2, 6)
    assert box == [(0.0, 1.5), (2.5, 4.0), (0.0, 6.5)]
    # mirrored, and nothing else, on the right-hand figure
    assert _diamond_block(2, 6, "right") == [(-u, v) for u, v in box]


def test_the_diamond_refuses_the_whole_matrix():
    table = _long_table()
    with pytest.raises(ValueError, match="type: full on the diamond"):
        draw_corrplot(_axes(3), __df__=table, __corr_layout__="diamond", type="full")


def test_the_colorbar_stands_in_the_empty_triangle():
    # Turning the matrix leaves a right-angled hole above it as big as the
    # matrix itself.  The bar goes in the hole rather than beside the panel,
    # so the figure is the names plus the panel and nothing else -- and three
    # numbers place it: top aligned with the panel, `colorbar.length` of the
    # panel's height, and `colorbar.offset` in from the edge of the paper.
    labels = ["v%02d" % i for i in range(24)]
    # spelled out: the card's default is the other mirror, and every number
    # below is written for the names on the left
    geom = solve_diamond_geometry(24, geometry=_DIAMOND, labels=labels, side="left")

    panel_x0, panel_y0, panel_w, panel_h = geom.panel_rect
    bar_x0, bar_y0, bar_w, bar_h = geom.colorbar_rect
    assert bar_y0 + bar_h == pytest.approx(panel_y0 + panel_h, abs=1e-12)
    assert bar_h * geom.height_mm == pytest.approx(0.22 * geom.panel_h_mm, abs=1e-9)
    assert geom.width_mm - (bar_x0 + bar_w) * geom.width_mm == pytest.approx(
        10.0, abs=1e-9
    )
    assert not geom.notes

    # Above the matrix's own upper boundary, which is the line v = u - 1/2:
    # the bar's lowest point has to clear the diagonal under its near edge.
    near_u = (bar_x0 - panel_x0) * geom.width_mm / geom.pitch_mm
    diagonal_mm = (near_u - 0.5) * geom.pitch_mm
    assert bar_h * geom.height_mm < diagonal_mm

    # and the figure pays for the names, the panel and the page margin only
    assert geom.width_mm == pytest.approx(
        geom.label_block_mm + geom.panel_w_mm + 4.0, abs=1e-9
    )


def test_the_three_colorbar_numbers_are_never_overridden():
    # Top, length and offset are the author's three numbers, and the solve
    # realises them -- on every size of matrix and both mirrors, including the
    # tight ones.  An earlier version shortened the bar to fit a clearance
    # allowance of its own and so drew a figure that did not have the length it
    # was asked for, with nothing on the page saying so.
    for side in ("left", "right"):
        for n in (10, 13, 16, 24):
            geom = solve_diamond_geometry(
                n, geometry=_DIAMOND, labels=["v%02d" % i for i in range(n)],
                side=side, edge_numbers=True, colorbar_title=r"$\rho$",
                colorbar_labels=["-1.0", "-0.5", "0", "0.5", "1.0"],
            )
            bar_x0, bar_y0, bar_w, bar_h = geom.colorbar_rect
            panel_x0, panel_y0, panel_w, panel_h = geom.panel_rect
            assert bar_y0 + bar_h == pytest.approx(panel_y0 + panel_h, abs=1e-12)
            assert bar_h * geom.height_mm == pytest.approx(
                0.22 * geom.panel_h_mm, abs=1e-9
            )
            outer = (
                geom.width_mm - (bar_x0 + bar_w) * geom.width_mm
                if side == "left" else bar_x0 * geom.width_mm
            )
            assert outer == pytest.approx(10.0, abs=1e-9)


def test_a_tight_triangle_is_reported_not_resized():
    # The check is a measurement, not a limit.  At ten variables the text
    # beside the bar comes within a cell of the diagonal -- worth saying, and
    # worth naming the key that fixes it, but not worth silently redrawing the
    # figure the author asked for.
    geom = solve_diamond_geometry(
        10, geometry=_DIAMOND, labels=["v%02d" % i for i in range(10)],
        edge_numbers=True, colorbar_title=r"$\rho$",
        colorbar_labels=["-1.0", "-0.5", "0", "0.5", "1.0"],
    )
    assert geom.colorbar_rect[3] * geom.height_mm == pytest.approx(
        0.22 * geom.panel_h_mm, abs=1e-9
    )
    assert any("colorbar.offset" in note for note in geom.notes)

    # A bar that really would run over the cells says *that* instead, and is
    # still drawn at the length it was given.
    crossing = solve_diamond_geometry(
        10, geometry={**_DIAMOND, "colorbar": {**_DIAMOND["colorbar"],
                                               "length": 0.9}},
        labels=["v%02d" % i for i in range(10)],
    )
    assert crossing.colorbar_rect[3] * crossing.height_mm == pytest.approx(
        0.9 * crossing.panel_h_mm, abs=1e-9
    )
    assert any("across the cells" in note for note in crossing.notes)


def test_the_page_margins_are_symmetric_top_to_bottom():
    # Nothing prints above the matrix and nothing below it, so an uneven pair
    # would read as a figure sitting crookedly rather than as a title band.
    geom = solve_diamond_geometry(10, geometry=_DIAMOND, labels=["v"] * 10)
    top = (1.0 - geom.panel_rect[1] - geom.panel_rect[3]) * geom.height_mm
    bottom = geom.panel_rect[1] * geom.height_mm
    assert top == pytest.approx(bottom, abs=1e-9)
    assert top == pytest.approx(4.0, abs=1e-9)


def test_the_shading_follows_the_clusters_when_there_are_any():
    # The reference diagram tints groups of rows, not every other one, and the
    # clusters are the groups: with `addrect` the tint is what says where one
    # ends and the next begins, which is what the boxes were for.
    from jarvisplot.Figure.corrplot_runtime import _shaded_variables

    blocks = [[0, 2], [3, 5], [6, 9]]
    shaded = _shaded_variables(10, blocks)
    assert list(np.flatnonzero(shaded)) == [0, 1, 2, 6, 7, 8, 9]
    # and without clusters it falls back to every other name
    assert list(np.flatnonzero(_shaded_variables(6))) == [1, 3, 5]


def test_the_badge_takes_the_corner_the_names_are_not_in():
    # 4 mm margins leave no room under the last name for a 5.8 mm mark, and on
    # this layout the opposite corner is empty by construction -- the last
    # variable's only cell sits at u = 1/2.  The house offset is unchanged.
    left = solve_diamond_geometry(9, geometry=_DIAMOND, labels=["v"] * 9, side="left")
    right = solve_diamond_geometry(9, geometry=_DIAMOND, labels=["v"] * 9, side="right")
    assert left.logo_rect[0] * left.width_mm == pytest.approx(
        left.width_mm - 0.838 - 5.0, abs=1e-9
    )
    assert right.logo_rect[0] * right.width_mm == pytest.approx(0.838, abs=1e-9)
    for geom in (left, right):
        assert geom.logo_rect[1] * geom.height_mm == pytest.approx(0.838, abs=1e-9)

    # and a card that wants it back on the left says so
    pinned = solve_diamond_geometry(
        9, geometry={**_DIAMOND, "logo": {**_DIAMOND["logo"], "corner": "bottom-left"}},
        labels=["v"] * 9, side="left",
    )
    assert pinned.logo_rect[0] * pinned.width_mm == pytest.approx(0.838, abs=1e-9)


def test_the_notch_makes_the_edge_beside_the_names_straight():
    # A variable's two edge cells meet at a point, so the boundary is a
    # sawtooth; the notch fills it, and its two outer corners are exactly where
    # the closing rules run -- half a row above the first name and below the
    # last.
    from jarvisplot.Figure.corrplot_runtime import _notch_vertices

    notches = _notch_vertices(5)
    assert notches.shape == (5, 3, 2)
    assert np.allclose(notches[0], [[0.0, -0.5], [0.5, 0.0], [0.0, 0.5]])
    assert np.allclose(notches[4], [[0.0, 3.5], [0.5, 4.0], [0.0, 4.5]])
    # every notch touches u = 0 twice, which is what makes the edge straight
    assert np.allclose(notches[:, ::2, 0], 0.0)
    # and the mirror points the other way, like everything else about `side`
    assert np.allclose(_notch_vertices(5, "right")[:, 1, 0], -0.5)


def test_the_numbers_beside_the_names_are_a_column_of_their_own():
    # `edge.numbers` prints a variable's position three times: at the end of
    # each arm of its V, and once beside its name.  The third copy is not part
    # of the tick label -- it is set smaller and lighter, and a tick label is
    # one string in one colour -- so the band becomes two columns and the solve
    # has to pay for both.
    from jarvisplot.Figure.corr_layout import max_label_mm

    labels = ["mu", "m_gluino", "Omega_h2"] * 4
    plain = solve_diamond_geometry(12, geometry=_DIAMOND, labels=labels)
    numbered = solve_diamond_geometry(
        12, geometry=_DIAMOND, labels=labels, edge_numbers=True, number_size_pt=4.2,
    )
    # the names cost the same either way; the numbers are the difference
    assert numbered.name_pad_mm == pytest.approx(plain.name_pad_mm, abs=1e-9)
    digits = max_label_mm(["%02d" % k for k in range(1, 13)], size_pt=4.2)
    assert numbered.number_pad_mm - numbered.name_pad_mm == pytest.approx(
        digits + 1.0, abs=1e-9                      # labels.number_gap
    )
    assert numbered.label_block_mm - plain.label_block_mm == pytest.approx(
        digits + 1.0, abs=1e-9
    )
    # unnumbered there is no second column, so the two anchors are one anchor
    assert plain.number_pad_mm == pytest.approx(plain.name_pad_mm, abs=1e-9)


def test_the_tightness_is_measured_against_whichever_text_ends_up_inward():
    # The numbers print on the bar's left and the label on its right on both
    # mirrors, so *which* of them stands between the bar and the diagonal is
    # the mirror's doing -- and the two are obstructions of different shapes:
    # the numbers are a column as tall as the bar, the label one turned line at
    # its mid height.  Neither resizes anything; they decide what is reported.
    labels = ["v%02d" % i for i in range(9)]
    plain = dict(geometry=_DIAMOND, labels=labels)

    # On a right-hand diamond the label is inward and the numbers are outward,
    # where they have the page margin: only the label can make it tight.
    assert not solve_diamond_geometry(
        9, side="right", colorbar_labels=["-1.0", "1.0"], **plain
    ).notes
    assert solve_diamond_geometry(
        9, side="right", colorbar_title=r"$\rho$", **plain
    ).notes

    # On a left-hand diamond they swap.
    assert solve_diamond_geometry(
        9, side="left", colorbar_labels=["-1.0", "1.0"], **plain
    ).notes
    assert not solve_diamond_geometry(
        9, side="left", colorbar_title=r"$\rho$", **plain
    ).notes

    # A matrix with room to spare says nothing about either of them.
    wide = ["v%02d" % i for i in range(30)]
    for side in ("left", "right"):
        assert not solve_diamond_geometry(
            30, geometry=_DIAMOND, labels=wide, side=side,
            colorbar_title=r"$\rho$", colorbar_labels=["-1.0", "1.0"],
        ).notes


def test_the_edge_numbers_are_set_across_the_edge_a_settable_gap_off_it():
    # They name the arms of the V they stand at the end of, so they have to
    # read as part of the matrix rather than as a scale printed beside it --
    # set *across* the edge, like the leader on a dimension, because text lying
    # along a diagonal is text the reader has to tilt their head to take in.
    from jarvisplot.Figure.corr_layout import max_label_mm
    from jarvisplot.Figure.corrplot_runtime import _edge_numbers

    root = 1.0 / np.sqrt(2.0)
    pitch_pt = 12.0
    # `gap` is the clearance to the *near end of the text*, so the rest is
    # measured: the cell face at 0.5 / sqrt(2), half the text's width (which
    # lies along the normal now that it is turned), and one more sqrt(2) for
    # striking the whole offset along the diagonal.
    half = (max_label_mm("0", size_pt=4.2) / 25.4 * 72.0 / 2.0) / pitch_pt
    for gap in (0.06, 0.4):
        want = (0.5 * root + gap + half) * root
        ax = _axes(6)
        _edge_numbers(ax, 6, "left", 4.2, "#6E6E6E", False, 10.0,
                      gap=gap, pitch_pt=pitch_pt)
        first = ax.texts[0]
        assert first.get_position() == pytest.approx((want, -want), abs=1e-12)
        assert first.get_text() == "1"
        # across the edge, not along it: the upper edge runs at -45 degrees
        assert first.get_rotation() == pytest.approx(45.0)
        assert ax.texts[1].get_rotation() == pytest.approx(315.0)


def test_everything_that_follows_the_names_stops_at_the_names():
    # A position is a tag *on* the row, not part of it.  So the tint, the rules
    # between the names and the outline that closes the figure all reach the
    # names' own outer edge and no further, and the numbering falls outside all
    # three -- a box drawn round the tag would say it was part of the name.
    labels = ["mu", "m_gluino", "Omega_h2"] * 4
    geom = solve_diamond_geometry(
        12, geometry=_DIAMOND, labels=labels, edge_numbers=True, number_size_pt=4.2,
    )
    assert geom.name_pad_mm < geom.number_pad_mm <= geom.label_text_mm


def test_the_rules_between_the_names_are_the_cell_grid():
    # A row of the label column *is* a row of the matrix, so the line dividing
    # two names is the line dividing two cells -- the same colour and the same
    # weight, not a nearby pair, and one fewer than there are names.
    from matplotlib.collections import LineCollection

    from jarvisplot.Figure.corrplot_runtime import _GRID_LWD, _label_rules

    ax = _axes(7)
    _label_rules(ax, 7, (-0.4, 0.0), "#C2C2C2", _GRID_LWD, False, 1.0)
    drawn = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(drawn) == 1
    segments = drawn[0].get_segments()
    assert len(segments) == 6                       # between, not through
    assert [seg[0][1] for seg in segments] == [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    assert drawn[0].get_linewidths()[0] == pytest.approx(_GRID_LWD)
    # below the tick labels, which an axis draws at 2.5: a rule over a name is
    # a strike-through
    assert drawn[0].get_zorder() < 2.5

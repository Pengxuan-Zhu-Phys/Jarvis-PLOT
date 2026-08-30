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
    table = _long_table(n_rows=8)
    plain = _axes(3)
    draw_corrplot(plain, __df__=table, method="circle")
    assert not plain.texts

    marked = _axes(3)
    draw_corrplot(marked, __df__=table, method="circle", **{"sig.level": 0.05})
    assert marked.texts, "sig.level should mark the insignificant cells"


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

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jarvisplot.Figure.correlation_runtime import correlation
from jarvisplot.column_demand import transform_columns, transform_output_columns


def _frame():
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, np.nan],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],
            "c": [4.0, 3.0, 2.0, 1.0, 0.0],
        }
    )


def test_listwise_pearson_long_table_matches_expected_pairs():
    out = correlation(
        _frame(),
        {
            "columns": ["a", "b", "c"],
            "missing": "listwise",
            "triangle": "upper",
            "include_diagonal": False,
        },
    )
    assert list(out[["var_x", "var_y"]].itertuples(index=False, name=None)) == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]
    assert (out["n"] == 4).all()
    assert np.allclose(out["rho"], [1.0, -1.0, -1.0])
    assert np.allclose(out["abs_rho"], 1.0)


def test_pairwise_uses_its_own_valid_row_count():
    out = correlation(
        _frame(),
        {
            "columns": ["a", "b", "c"],
            "missing": "pairwise",
            "triangle": "upper",
            "include_diagonal": False,
        },
    )
    assert out.loc[(out.var_x == "a") & (out.var_y == "b"), "n"].item() == 4
    assert out.loc[(out.var_x == "b") & (out.var_y == "c"), "n"].item() == 5


def test_correlation_columns_are_visible_to_projection_and_validation():
    steps = [{"correlation": {"columns": ["a", "b", "c"]}}]
    assert transform_columns(steps) == {"a", "b", "c"}
    assert {"var_x", "var_y", "rho", "abs_rho", "n"} <= transform_output_columns(steps)


def test_grid_metadata_states_the_matrix_shape_a_triangle_hides():
    """A long table does not carry its own shape once cells are missing.

    Six variables in an upper triangle is 21 rows; anything inferring the
    side length from the row count lands on 4 and draws the matrix on the
    wrong mesh without complaining, so the step has to say 6 outright.
    """
    df = pd.DataFrame({name: np.arange(10.0) * (i + 1) for i, name in enumerate("abcdef")})
    out = correlation(df, {"columns": list("abcdef"), "triangle": "upper"})

    assert len(out) == 21
    assert int(np.sqrt(len(out))) == 4          # what guessing the shape would give
    assert out["__grid_nx__"].unique().tolist() == [6]
    assert out["__grid_ny__"].unique().tolist() == [6]
    assert np.array_equal(out["__grid_ix__"], out["x_index"])
    assert np.array_equal(out["__grid_iy__"], out["y_index"])
    # The extent is the index convention an image uses: cell k spans k +- 0.5.
    assert out["__grid_xmin__"].unique().tolist() == [-0.5]
    assert out["__grid_xmax__"].unique().tolist() == [5.5]


def test_grid_columns_survive_the_column_projection():
    steps = [{"correlation": {"columns": ["a", "b", "c"]}}]
    produced = transform_output_columns(steps)
    assert {"__grid_ix__", "__grid_iy__", "__grid_nx__", "__grid_ny__"} <= produced


def test_upper_triangle_is_the_half_above_the_drawn_diagonal():
    df = pd.DataFrame({name: np.arange(10.0) * (i + 1) for i, name in enumerate("abc")})
    out = correlation(df, {"columns": list("abc"), "triangle": "upper"})
    assert (out["y_index"] >= out["x_index"]).all()

    out = correlation(df, {"columns": list("abc"), "triangle": "lower"})
    assert (out["y_index"] <= out["x_index"]).all()


def test_matrix_is_rebuilt_from_the_published_grid():
    """The drawing side folds the long table back without guessing."""
    from jarvisplot.Figure.profile_runtime import grid_from_metadata

    df = pd.DataFrame({name: np.arange(20.0) ** (i + 1) for i, name in enumerate("abcdef")})
    out = correlation(df, {"columns": list("abcdef"), "triangle": "upper"})

    grid, extent = grid_from_metadata(out["rho"].to_numpy(), out)
    assert grid.shape == (6, 6)
    assert extent == (-0.5, 5.5, -0.5, 5.5)
    # Cells the triangle left out stay masked rather than borrowing a value.
    assert grid.mask.sum() == 36 - 21
    assert np.allclose(np.diagonal(grid.filled(np.nan)), 1.0)
    for row in out.itertuples(index=False):
        assert grid[row.y_index, row.x_index] == pytest.approx(row.rho)


def test_a_table_with_no_grid_behind_it_reports_that_rather_than_inventing_one():
    from jarvisplot.Figure.profile_runtime import grid_from_metadata

    assert grid_from_metadata(np.arange(9.0), pd.DataFrame({"z": np.arange(9.0)})) is None
    assert grid_from_metadata(np.arange(9.0), None) is None


def _wide():
    """A feature table shaped like a real one: features plus bookkeeping."""
    rng = np.random.default_rng(5)
    n = 60
    frame = {f"bdt_f{i}": rng.normal(size=n) for i in range(4)}
    frame["weight"] = rng.gamma(2.0, size=n)
    frame["label"] = rng.integers(0, 2, n)
    frame["plot_key"] = ["train"] * n          # not numeric
    frame["is_test"] = rng.integers(0, 2, n).astype(bool)   # a flag, not a feature
    frame["__grid_ix__"] = np.zeros(n, dtype=int)           # private bookkeeping
    return pd.DataFrame(frame)


def _selected(cfg):
    return correlation(_wide(), cfg)["var_x"].unique().tolist()


def test_naming_nothing_takes_every_numeric_column():
    assert _selected({}) == ["bdt_f0", "bdt_f1", "bdt_f2", "bdt_f3", "weight", "label"]


def test_strings_flags_and_private_columns_are_never_picked_up():
    picked = set(_selected({}))
    assert "plot_key" not in picked     # not numeric
    assert "is_test" not in picked      # bool: a label by construction
    assert "__grid_ix__" not in picked  # private bookkeeping


def test_exclude_is_the_short_list_to_write():
    assert _selected({"exclude": ["weight", "label"]}) == ["bdt_f0", "bdt_f1", "bdt_f2", "bdt_f3"]
    assert _selected({"exclude": "weight"}) == ["bdt_f0", "bdt_f1", "bdt_f2", "bdt_f3", "label"]


def test_regex_selects_and_composes_with_exclude():
    assert _selected({"regex": "^bdt_"}) == ["bdt_f0", "bdt_f1", "bdt_f2", "bdt_f3"]
    assert _selected({"regex": "^bdt_", "exclude": ["bdt_f2"]}) == ["bdt_f0", "bdt_f1", "bdt_f3"]


def test_order_is_stated_not_incidental():
    """x_index counts this order, so it has to be predictable."""
    written = _selected({"columns": ["label", "bdt_f2", "bdt_f0"]})
    assert written == ["label", "bdt_f2", "bdt_f0"]          # as written
    assert _selected({"regex": "^bdt_"}) == ["bdt_f0", "bdt_f1", "bdt_f2", "bdt_f3"]  # table order


def test_columns_and_regex_together_are_refused():
    with pytest.raises(ValueError, match="either 'columns' or 'regex'"):
        correlation(_wide(), {"columns": ["bdt_f0", "bdt_f1"], "regex": "^bdt_"})


def test_a_selection_that_leaves_too_few_columns_says_what_it_tried():
    with pytest.raises(ValueError, match=r"regex '\^nothing_' selected 0 column"):
        correlation(_wide(), {"regex": "^nothing_"})
    with pytest.raises(ValueError, match="removed by 'exclude'"):
        correlation(_wide(), {"regex": "^bdt_", "exclude": ["bdt_f0", "bdt_f1", "bdt_f2"]})


def test_an_invalid_regex_is_reported_as_one():
    with pytest.raises(ValueError, match="is not valid"):
        correlation(_wide(), {"regex": "["})


def test_only_an_explicit_column_list_can_be_projected():
    """Pruning a source to the names in the YAML would cut the features out."""
    from jarvisplot.column_demand import transform_needs_all_columns

    assert transform_needs_all_columns([{"correlation": {}}])
    assert transform_needs_all_columns([{"correlation": {"regex": "^bdt_"}}])
    assert transform_needs_all_columns([{"correlation": {"exclude": ["weight"]}}])
    assert transform_needs_all_columns([{"type": "correlation"}])
    assert not transform_needs_all_columns([{"correlation": {"columns": ["a", "b"]}}])
    assert not transform_needs_all_columns([{"filter": "x > 0"}])

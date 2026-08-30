"""The imshow layer, which until now validated fine and crashed at render.

``method_contracts`` declares ``z`` as imshow's one coordinate, so the layer
hands the adapter ``z=<column>`` -- while ``Axes.imshow`` takes its array
positionally as ``X``.  Every imshow layer therefore passed ``jplot validate``
and then died with "missing 1 required positional argument".  These tests
pin the route down to the drawn artist.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.image import AxesImage

from jarvisplot.Figure.adapters_rect import StdAxesAdapter
from jarvisplot.Figure.correlation_runtime import correlation


@pytest.fixture()
def adapter():
    fig, ax = plt.subplots()
    try:
        yield StdAxesAdapter(ax, {})
    finally:
        plt.close(fig)


def _events():
    rng = np.random.default_rng(11)
    a = rng.normal(size=400)
    b = 0.8 * a + 0.6 * rng.normal(size=400)
    c = -0.5 * a + rng.normal(size=400)
    return pd.DataFrame({"a": a, "b": b, "c": c})


def test_a_z_column_reaches_imshow_as_the_matrix_behind_it(adapter):
    table = correlation(_events(), {"columns": ["a", "b", "c"]})

    artist = adapter.imshow(z=table["rho"].to_numpy(), __df__=table)

    assert isinstance(artist, AxesImage)
    assert artist.get_array().shape == (3, 3)
    # Index convention: cell k is centred on k, so ticks at 0..n-1 land in
    # the middle of their cell.
    assert tuple(artist.get_extent()) == (-0.5, 2.5, -0.5, 2.5)
    assert artist.origin == "lower"
    # A card fixes the axes box; an image must not resize it to its own aspect.
    assert adapter.ax.get_aspect() == "auto"

    drawn = artist.get_array()
    for row in table.itertuples(index=False):
        assert drawn[row.y_index, row.x_index] == pytest.approx(row.rho)


def test_cells_a_triangle_left_out_stay_masked(adapter):
    table = correlation(_events(), {"columns": ["a", "b", "c"], "triangle": "upper"})

    artist = adapter.imshow(z=table["rho"].to_numpy(), __df__=table)

    drawn = artist.get_array()
    assert drawn.shape == (3, 3)
    assert drawn.mask.sum() == 9 - len(table)
    assert bool(drawn.mask[0, 1])          # y_index 0 < x_index 1: not emitted
    assert not bool(drawn.mask[1, 0])


def test_a_two_dimensional_z_is_drawn_as_given(adapter):
    values = np.arange(6.0).reshape(2, 3)
    artist = adapter.imshow(z=values)
    assert np.array_equal(artist.get_array(), values)


def test_an_array_passed_positionally_still_goes_straight_through(adapter):
    artist = adapter.imshow(np.zeros((4, 4)))
    assert isinstance(artist, AxesImage)


def test_x_and_y_coordinates_are_refused_rather_than_dropped(adapter):
    table = correlation(_events(), {"columns": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="only the 'z' coordinate"):
        adapter.imshow(
            x=table["x_index"].to_numpy(),
            y=table["y_index"].to_numpy(),
            z=table["rho"].to_numpy(),
            __df__=table,
        )


def test_a_column_with_no_grid_behind_it_is_an_error_not_a_guess(adapter):
    with pytest.raises(ValueError, match="shape is"):
        adapter.imshow(z=np.arange(9.0), __df__=pd.DataFrame({"z": np.arange(9.0)}))


def test_an_explicit_extent_in_the_style_wins_over_the_published_one(adapter):
    table = correlation(_events(), {"columns": ["a", "b", "c"]})
    artist = adapter.imshow(z=table["rho"].to_numpy(), __df__=table, extent=[0, 3, 0, 3])
    assert tuple(artist.get_extent()) == (0.0, 3.0, 0.0, 3.0)

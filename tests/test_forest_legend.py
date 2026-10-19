"""Footer collisions distinguish shared grid edges from occupied area."""
import numpy as np
import pytest
from matplotlib.transforms import Bbox

from jarvisplot.Figure.forest_legend import boxes_overlap


@pytest.mark.parametrize(
    'badge_right', [8.541666666666668, np.nextafter(8.541666666666668, np.inf)],
)
def test_legend_may_touch_the_badge_at_its_grid_boundary(badge_right):
    # Native 3.10.9 Helvetica layout: vertical ranges overlap by 0.064 mm,
    # but the legend starts exactly at the badge's right edge.
    item = Bbox.from_extents(8.541666666666668, 33.5, 23.4682, 37.0221)
    badge = Bbox.from_extents(2, 36.9583, badge_right, 43.5)
    assert not boxes_overlap(item, badge)
    assert not boxes_overlap(badge, item)


def test_positive_legend_badge_overlap_is_still_a_collision():
    item = Bbox.from_extents(8.541666666666668, 33.5, 23.4682, 37.0221)
    badge = Bbox.from_extents(2, 36.9583, 8.551666666666668, 43.5)
    assert boxes_overlap(item, badge)
    assert boxes_overlap(badge, item)


def test_empty_semantic_legend_column_does_not_occupy_area():
    empty = Bbox.from_extents(5, 5, 5, 5)
    badge = Bbox.from_extents(2, 2, 8, 8)
    assert not boxes_overlap(empty, badge)


def test_legend_may_touch_description_at_a_horizontal_edge():
    item = Bbox.from_extents(8, 30, 23, 37)
    description = Bbox.from_extents(15, 37, 45, 42)
    assert not boxes_overlap(item, description)
    description = Bbox.from_extents(15, 36.99, 45, 42)
    assert boxes_overlap(item, description)

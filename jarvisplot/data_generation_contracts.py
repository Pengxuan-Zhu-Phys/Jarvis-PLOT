"""Pure-data contracts shared by generated-data runtime and CLI discovery."""

from __future__ import annotations

__all__ = ["SEPARATE_DATA_METHODS"]

# Methods whose inputs are independent tabular series.  Matrix/grid methods
# intentionally stay out: their calculation is defined over one combined table.
SEPARATE_DATA_METHODS = frozenset(
    {
        "bar",
        "barh",
        "errorbar",
        "fill",
        "fill_between",
        "fill_betweenx",
        "plot",
        "scatter",
        "stairs",
        "step",
    }
)

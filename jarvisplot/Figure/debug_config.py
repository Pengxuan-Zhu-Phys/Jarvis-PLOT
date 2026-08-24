#!/usr/bin/env python3

"""Defaults and merge rules for the design-reference debug overlay.

Split out of :mod:`design_runtime` so the configuration surface has one owner
and can be tested without importing matplotlib. The drawing code reads only
what this module hands it -- there are no literals below the unpack.

Precedence, lowest to highest:

1. :data:`DEFAULT_DEBUG` -- ships with the package, always present.
2. A style card's root ``Debug`` block (``jarvisplot/cards/**/*.json``).
3. A YAML ``Figures[].debug`` mapping, for one figure only.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

__all__ = [
    "DEFAULT_DEBUG",
    "PALETTE_ROLES",
    "merge_debug_config",
    "resolve_debug_config",
]


DIM    = "#0A7FB6"   # blue dimension lines + labels
BOX    = "#111111"   # axes outline
NAME   = "#1A237E"   # axes name + rect text
MARGIN = "#FF3FA4"   # pink figure border / caption
PANEL  = "#FFEC73"   # axes-layout information card

#: Every colour role a ``*style`` block may inherit from ``palette``.
PALETTE_ROLES = ("dimension", "box", "name", "margin", "panel")

# order in which we look for the "primary" axes to fully dimension
PRIMARY_ORDER = ("ax", "ax0", "ax1", "axtri")


DEFAULT_DEBUG = {
    "palette": {
        "dimension": DIM,
        "box": BOX,
        "name": NAME,
        "margin": MARGIN,
        "panel": PANEL,
    },
    "figure": {
        "border_linewidth": 0.9,
        "caption_x": 0.5,
        "caption_y": 0.995,
        "caption_size": 6.0,
        "height_marker_x": 0.035,
    },
    "axes": {
        "outline_linewidth": 0.45,
        "frameless_extension_max": 0.018,
        "frameless_extension_min": 0.006,
        "frameless_extension_factor": 0.09,
    },
    "dimension": {
        "line_width": 0.7,
        "cap_half_pt": 5.0,
        "narrow_threshold_pt": 24.0,
        "narrow_length_pt": 8.0,
        "narrow_scale_factor": 0.45,
        "narrow_scale_min": 3.0,
        "narrow_scale_max": 6.0,
        "wide_scale": 6.0,
        "label_gap": 0.008,
        "label_box_alpha": 0.5,
        "corner_gap": 0.018,
        "colorbar_line_offset": 0.024,
    },
    "numbered_axes": {
        "right_edge": 1.0,
        "marker_step": 0.040,
        "label_side": "left",
    },
    "panel": {
        "alpha": 0.5,
        "right": 0.70,
        "pad_x_max": 0.018,
        "pad_x_min": 0.004,
        "pad_x_factor": 0.025,
        "pad_y_max": 0.018,
        "pad_y_min": 0.004,
        "pad_y_factor": 0.035,
        "boxstyle": "round,pad=0.004",
        "header_size": 6.0,
        "name_size": 7.0,
        "detail_size": 4.6,
        "entry_gap": 5.0,
        "header_leading": 1.20,
        "name_leading": 1.15,
        "detail_leading": 1.08,
        "min_header_size": 2.2,
        "min_name_size": 2.2,
        "min_detail_size": 1.8,
    },
}



def merge_debug_config(base: Mapping, override: Mapping | None) -> dict:
    """Merge a style-card ``Debug`` block over the runtime defaults."""
    merged = deepcopy(dict(base))
    if not isinstance(override, Mapping):
        return merged
    for key, value in override.items():
        if isinstance(merged.get(key), Mapping):
            if isinstance(value, Mapping):
                merged[key] = merge_debug_config(merged[key], value)
            continue
        merged[key] = value
    return merged


def resolve_debug_config(fig) -> dict:
    """Return the selected style card's debug settings with safe defaults."""
    return merge_debug_config(
        DEFAULT_DEBUG,
        getattr(fig, "debug_config", getattr(fig, "_debug_config", None)),
    )



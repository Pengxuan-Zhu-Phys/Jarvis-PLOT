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

from ..diagnostics import did_you_mean

__all__ = [
    "DEFAULT_DEBUG",
    "DELEGATED_LEAVES",
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
    # Which axes gets fully dimensioned and hosts the info panel. First match wins.
    "primary_order": ["ax", "ax0", "ax1", "axtri"],
    # The overlay's own stacking order, above every real artist.
    "zorder": {"overlay": 10000, "artist": 10001, "text": 10002},
    "units": {"cm_per_inch": 2.54, "pt_per_inch": 72.0},
    # Shared defaults for every dimension label; each call site may override.
    "labelstyle": {
        "fontsize": 5.0,
        "ha": "center",
        "va": "center",
        "rotation": 0,
        "fontweight": "normal",
        "family": "monospace",
        "clip_on": False,
    },
    "labelbox": {
        "boxstyle": "square,pad=0.12",
        "fc": "white",
        "ec": "none",
        "alpha": 0.82,
    },
    "figure": {
        "border_linewidth": 0.9,
        "caption_x": 0.5,
        "caption_y": 0.995,
        "caption_size": 6.0,
        "height_marker_x": 0.035,
        "border": {"show": True, "style": {"fill": False}},
        "caption": {
            "show": True,
            "template": "design reference  \u00b7  figure %.2f cm \u00d7 %.2f cm",
            "style": {"va": "top", "fontweight": "bold", "family": "DejaVu Sans"},
        },
        "height_marker": {"show": True, "template": "%.2f cm"},
    },
    "axes": {
        "outline_linewidth": 0.45,
        "frameless_extension_max": 0.018,
        "frameless_extension_min": 0.006,
        "frameless_extension_factor": 0.09,
        "outline": {"show": True, "style": {"fill": False}},
        "corner_ticks": {"show": True, "style": {"solid_capstyle": "projecting", "clip_on": False}},
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
        "linestyle": {"solid_capstyle": "projecting", "clip_on": False},
        "narrowarrowstyle": {"arrowstyle": "-|>", "shrinkA": 0, "shrinkB": 0},
        "widearrowstyle": {"arrowstyle": "<|-|>", "shrinkA": 0, "shrinkB": 0},
    },
    "numbered_axes": {
        "right_edge": 1.0,
        "marker_step": 0.040,
        "label_side": "left",
        "show": True,
        "template": "%.3f cm",
        "top": {"show": True},
        "bottom": {"show": True},
    },
    # Primary-axes margin dimensions.
    "margins": {
        "template": "%.3f cm",
        "left": {"show": True},
        "top": {"show": True},
        "bottom": {"show": True},
    },
    "colorbar_gap": {"show": True, "template": "%.3f cm"},
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
        "show": True,
        "header": "axes layout",
        "rect_template": "  rect=[%.3f, %.3f, %.3f, %.3f]",
        "size_template": "  width: %.3f cm  height: %.3f cm",
        "char_width_factor": 0.62,
        "detail_lines_per_entry": 2.0,
        "text_inset_x_max": 0.008,
        "text_inset_x_factor": 0.02,
        "text_inset_y_max": 0.006,
        "text_inset_y_factor": 0.03,
        "textstyle": {"ha": "left", "va": "top", "family": "monospace", "clip_on": True},
        "boxstyle_kwargs": {"edgecolor": "none", "linewidth": 0.0},
    },
    # Ternary tick anchors and label leader lines, drawn during axtri build.
    "ternary": {
        "tick_anchors": {
            "show": True,
            "style": {"s": 1.0, "marker": ".", "c": "#FF42A1", "clip_on": False},
        },
        "label_leaders": {
            "show": True,
            "style": {
                "marker": ".",
                "linestyle": "-",
                "lw": 0.3,
                "markersize": 1,
                "c": "#FF42A1",
                "clip_on": False,
            },
        },
    },
}



#: Leaf blocks whose keys are matplotlib kwargs, not Jarvis-PLOT vocabulary.
#: The merge stops validating inside these, the same way ``layers[].style`` is a
#: delegated zone in the YAML schema -- Jarvis-PLOT does not enumerate what
#: matplotlib accepts.
DELEGATED_LEAVES = ("bbox", "boxstyle_kwargs")


def _is_delegated(key: str) -> bool:
    return key.endswith("style") or key in DELEGATED_LEAVES


def merge_debug_config(
    base: Mapping,
    override: Mapping | None,
    *,
    path: str = "",
    problems: list[str] | None = None,
) -> tuple[dict, list[str]]:
    """Merge a ``Debug`` block over the defaults, reporting what it had to drop.

    Returns ``(merged, problems)``. Two things used to fail in silence here and
    now surface instead:

    - a key the defaults do not define was carried through as inert junk, so a
      typo looked exactly like a working override;
    - a scalar written where the defaults hold a mapping was discarded outright.

    Neither ever raises. A bad ``Debug`` block degrades to the defaults and says
    so once, because a debug overlay must never be able to break a plot.
    """
    problems = [] if problems is None else problems
    merged = deepcopy(dict(base))
    if not isinstance(override, Mapping):
        return merged, problems

    for key, value in override.items():
        where = f"{path}{key}"
        if key not in merged:
            hint = did_you_mean(str(key), [str(k) for k in merged])
            suffix = f" Did you mean {hint[0]!r}?" if hint else ""
            problems.append(f"{where}: unknown Debug key, ignored.{suffix}")
            continue

        if _is_delegated(str(key)):
            # matplotlib kwargs: take the override wholesale, validate nothing.
            if isinstance(merged[key], Mapping) and isinstance(value, Mapping):
                merged[key] = {**merged[key], **deepcopy(dict(value))}
            else:
                merged[key] = deepcopy(value)
            continue

        if isinstance(merged[key], Mapping):
            if isinstance(value, Mapping):
                merged[key], _ = merge_debug_config(
                    merged[key], value, path=f"{where}.", problems=problems
                )
            else:
                problems.append(
                    f"{where}: expected a mapping, got "
                    f"{type(value).__name__}; kept the default."
                )
            continue

        merged[key] = deepcopy(value)

    return merged, problems


def resolve_debug_config(fig) -> dict:
    """Return the merged debug settings, logging anything the merge dropped."""
    merged, problems = merge_debug_config(
        DEFAULT_DEBUG,
        getattr(fig, "debug_config", getattr(fig, "_debug_config", None)),
    )
    if problems:
        logger = getattr(fig, "logger", None)
        warn = getattr(logger, "warning", None)
        if callable(warn):
            try:
                warn("Debug config:\n\t" + "\n\t".join(problems))
            except Exception:
                pass
    return merged

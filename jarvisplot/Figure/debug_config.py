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

#!/usr/bin/env python3

"""Defaults and merge rules for the design-reference debug overlay.

Split out of :mod:`design_runtime` so the configuration surface has one owner
and can be tested without importing matplotlib. The drawing code reads only
what this module hands it -- there are no literals below the unpack.

**Every appearance value lives in a block named after the matplotlib callable
that consumes it, and is splatted straight into that call:**

.. code-block:: python

    ov.plot(xs, ys, **cfg["cap"]["plot"])
    ov.add_patch(mpl.patches.Rectangle((l, b), w, h, **cfg["outline"]["framed"]["Rectangle"]))

So a block's keys are matplotlib's vocabulary, not Jarvis-PLOT's, and one
drawn thing owns one complete kwargs table -- no shared palette to chase, no
partial style merged with hidden Python defaults. What stays outside those
blocks is geometry: positions, spans, thresholds and the autoscale constants
that Python computes with before it draws.

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
    "CALL_BLOCKS",
    "DEFAULT_DEBUG",
    "deep_update",
    "merge_debug_config",
    "resolve_debug_config",
]


# Colours are written out in full inside every call block below; these names
# exist only to keep that table readable and to make the roles obvious. They
# are not a runtime indirection -- a card overrides the block, not a palette.
_DIM = "#1F21E9"      # dimension lines, arrowheads and their labels
_BOX = "#111111"      # outline of a framed axes
_MARGIN = "#FF3FA4"   # figure border, caption, frameless-axes outline
_NAME = "#1A237E"     # text inside the axes-layout panel
_PANEL = "#FFEC73"    # the panel's own background
_TERNARY = "#FF42A1"  # ternary tick anchors and label leaders

# The overlay sits above every real artist; text above its own artists.
_Z_OVERLAY, _Z_ARTIST, _Z_TEXT = 10000, 10001, 10002

# Background box shared by the dimension labels and the caption. Written into
# each ``text`` block in full; the alpha differs between them.
_LABEL_BBOX = {"boxstyle": "square,pad=0.12", "fc": "white", "ec": "none"}

# Typography shared by the dimension labels, likewise written out in full.
_LABEL_FONT = {
    "fontsize": 5.0,
    "fontweight": "normal",
    "family": "monospace",
    "clip_on": False,
    "zorder": _Z_TEXT,
}


DEFAULT_DEBUG = {
    # Which axes gets fully dimensioned and hosts the info panel. First match wins.
    "primary_order": ["ax", "ax0", "ax1", "axtri"],
    "units": {"cm_per_inch": 2.54, "pt_per_inch": 72.0},
    # The transparent full-figure axes every annotation is drawn onto.
    "overlay": {"add_axes": {"zorder": _Z_OVERLAY}},
    "figure": {
        "border": {
            "show": True,
            "Rectangle": {"fill": False, "ec": _MARGIN, "lw": 0.9, "zorder": _Z_ARTIST},
        },
        "caption": {
            "show": True,
            "x": 0.5,
            "y": 0.995,
            "template": "design reference  ·  figure %.2f cm × %.2f cm",
            "text": {
                "color": _MARGIN,
                "fontsize": 6.0,
                "ha": "center",
                "va": "top",
                "rotation": 0,
                "fontweight": "bold",
                "family": "DejaVu Sans",
                "clip_on": False,
                "zorder": _Z_TEXT,
                "bbox": {**_LABEL_BBOX, "alpha": 0.82},
            },
        },
        # Drawn through the shared dimension machinery, so it has no call block
        # of its own -- only where it sits and how its number reads.
        "height_marker": {"show": True, "x": 0.035, "template": "%.2f cm"},
    },
    "axes": {
        "frameless_extension_max": 0.018,
        "frameless_extension_min": 0.006,
        "frameless_extension_factor": 0.09,
        "outline": {
            "show": True,
            # A framed axes keeps the neutral reference outline; a frameless
            # one (axlogo, say) is marked as a design box instead.
            "framed": {
                "Rectangle": {"fill": False, "ec": _BOX, "lw": 0.45, "zorder": _Z_ARTIST},
            },
            "frameless": {
                "Rectangle": {"fill": False, "ec": _MARGIN, "lw": 0.45, "zorder": _Z_ARTIST},
            },
        },
        # Hash marks past the corners, drawn for frameless axes only.
        "corner_ticks": {
            "show": True,
            "plot": {
                "color": _MARGIN,
                "lw": 0.45,
                "solid_capstyle": "projecting",
                "clip_on": False,
                "zorder": _Z_ARTIST,
            },
        },
    },
    # One dimension line: two terminal cap bars, an arrow, and a label. Every
    # measured distance in the overlay is drawn with these four calls.
    "dimension": {
        "cap_half_pt": 5.0,
        "narrow_threshold_pt": 24.0,
        "narrow_length_pt": 8.0,
        "narrow_scale_factor": 0.45,
        "narrow_scale_min": 3.0,
        "narrow_scale_max": 6.0,
        "label_gap": 0.008,
        "corner_gap": 0.018,
        "colorbar_line_offset": 0.024,
        "cap": {
            "plot": {
                "color": _DIM,
                "lw": 0.7,
                "solid_capstyle": "projecting",
                "clip_on": False,
                "zorder": _Z_ARTIST,
            },
        },
        # Short gaps get two inward-pointing arrows whose tails extend outside
        # the interval; ``mutation_scale`` is computed from the span, between
        # ``narrow_scale_min`` and ``narrow_scale_max``.
        "narrow_arrow": {
            "annotate": {
                "zorder": _Z_ARTIST,
                "arrowprops": {
                    "arrowstyle": "-|>",
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "color": _DIM,
                    "lw": 0.7,
                },
            },
        },
        "wide_arrow": {
            "annotate": {
                "zorder": _Z_ARTIST,
                "arrowprops": {
                    "arrowstyle": "<|-|>",
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "color": _DIM,
                    "lw": 0.7,
                    "mutation_scale": 6.0,
                },
            },
        },
        "hlabel": {
            "text": {
                **_LABEL_FONT,
                "color": _DIM,
                "ha": "center",
                "va": "top",
                "rotation": 0,
                "bbox": {**_LABEL_BBOX, "alpha": 0.5},
            },
        },
        "vlabel_left": {
            "text": {
                **_LABEL_FONT,
                "color": _DIM,
                "ha": "right",
                "va": "center",
                "rotation": 90,
                "bbox": {**_LABEL_BBOX, "alpha": 0.5},
            },
        },
        "vlabel_right": {
            "text": {
                **_LABEL_FONT,
                "color": _DIM,
                "ha": "left",
                "va": "center",
                "rotation": 90,
                "bbox": {**_LABEL_BBOX, "alpha": 0.5},
            },
        },
    },
    "numbered_axes": {
        "show": True,
        "right_edge": 1.0,
        "marker_step": 0.040,
        "label_side": "left",
        "template": "%.3f cm",
        "top": {"show": True},
        "bottom": {"show": True},
    },
    # Primary-axes margin dimensions. The top/bottom pair is anchored to the
    # right edge, like the numbered-axes column, so it stays clear of the
    # axes-layout panel; ``left`` measures along the top edge and stays put.
    "margins": {
        "template": "%.3f cm",
        "right_edge": 1.0,
        "label_side": "left",
        "left": {"show": True},
        "top": {"show": True},
        "bottom": {"show": True},
    },
    "colorbar_gap": {"show": True, "template": "%.3f cm"},
    "panel": {
        "show": True,
        "right": 0.70,
        "pad_x_max": 0.018,
        "pad_x_min": 0.004,
        "pad_x_factor": 0.025,
        "pad_y_max": 0.018,
        "pad_y_min": 0.004,
        "pad_y_factor": 0.035,
        "entry_gap": 5.0,
        "char_width_factor": 0.62,
        "detail_lines_per_entry": 2.0,
        "text_inset_x_max": 0.008,
        "text_inset_x_factor": 0.02,
        "text_inset_y_max": 0.006,
        "text_inset_y_factor": 0.03,
        "box": {
            "FancyBboxPatch": {
                "boxstyle": "round,pad=0.004",
                "facecolor": _PANEL,
                "alpha": 0.5,
                "edgecolor": "none",
                "linewidth": 0.0,
                "zorder": _Z_ARTIST,
            },
        },
        # ``size`` is the target; the whole group shrinks together, but never
        # below ``min_size``, when a short axes cannot fit every entry.
        # ``fontsize`` is therefore the one text kwarg Python supplies.
        "header": {
            "label": "axes layout",
            "size": 6.0,
            "min_size": 2.2,
            "leading": 1.20,
            "text": {
                "color": _NAME,
                "fontweight": "bold",
                "ha": "left",
                "va": "top",
                "family": "monospace",
                "clip_on": True,
                "zorder": _Z_TEXT,
            },
        },
        "name": {
            "size": 7.0,
            "min_size": 2.2,
            "leading": 1.15,
            "text": {
                "color": _NAME,
                "fontweight": "bold",
                "ha": "left",
                "va": "top",
                "family": "monospace",
                "clip_on": True,
                "zorder": _Z_TEXT,
            },
        },
        "detail": {
            "size": 4.6,
            "min_size": 1.8,
            "leading": 1.08,
            "rect_template": "  rect=[%.3f, %.3f, %.3f, %.3f]",
            "size_template": "  width: %.3f cm  height: %.3f cm",
            "text": {
                "color": _NAME,
                "fontweight": "normal",
                "ha": "left",
                "va": "top",
                "family": "monospace",
                "clip_on": True,
                "zorder": _Z_TEXT,
            },
        },
    },
    # Ternary tick anchors and label leader lines, drawn during axtri build.
    "ternary": {
        "tick_anchors": {
            "show": True,
            "scatter": {"s": 1.0, "marker": ".", "c": _TERNARY, "clip_on": False},
        },
        "label_leaders": {
            "show": True,
            "plot": {
                "marker": ".",
                "linestyle": "-",
                "lw": 0.3,
                "markersize": 1,
                "c": _TERNARY,
                "clip_on": False,
            },
        },
    },
}


#: Blocks named after the matplotlib callable that receives them. Their keys
#: are matplotlib's vocabulary, so the merge stops validating inside them --
#: the same way ``layers[].style`` is a delegated zone in the YAML schema.
#: Jarvis-PLOT does not enumerate what matplotlib accepts.
CALL_BLOCKS = (
    "add_axes",
    "annotate",
    "plot",
    "scatter",
    "text",
    "FancyBboxPatch",
    "Rectangle",
)


def deep_update(base: Mapping, override: Mapping | None) -> dict:
    """Layer one override mapping over another, without validating either.

    Used to combine a style card's ``Debug`` block with a figure's YAML
    ``debug:`` mapping. Validation is deliberately not done here: both are
    partial by design, so checking them against each other would report every
    key the card happens not to mention. :func:`merge_debug_config` validates
    the combined result against :data:`DEFAULT_DEBUG` once, later.

    Non-mapping values replace rather than merge -- notably lists, so a card
    overriding ``primary_order`` sets it instead of appending to it.
    """
    merged = deepcopy(dict(base))
    if not isinstance(override, Mapping):
        return merged
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _is_delegated(key: str) -> bool:
    return key in CALL_BLOCKS


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

    Inside a call block (see :data:`CALL_BLOCKS`) neither check applies: those
    keys belong to matplotlib. Overrides there are layered key by key, so a
    card may change one arrow property without restating the rest.

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
                merged[key] = deep_update(merged[key], value)
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

#!/usr/bin/env python3
"""Design-reference debug overlay for Jarvis-PLOT figures.

When a figure is rendered with ``debug: true`` in the YAML, this module draws a
dimension overlay on top of the finished figure: every axes defined by the style
card is outlined and annotated with

  - its name (``ax`` / ``axc`` / ``axtri`` / ``ax0`` ...),
  - its position rect ``[left, bottom, width, height]`` in figure fractions
    (exactly the ``Frame.axes.<name>.rect`` value from the style JSON card), and
  - its width / height converted to centimetres from ``figure.figsize``.

The primary axes additionally gets margin dimension lines (left / top / bottom
insets), numbered subplot axes get their top / bottom distances to the figure
edges in successive columns, and colorbar axes get a gap dimension to the
primary axes.  The per-axes geometry is collected into one compact, ordered
panel inside the primary ``ax`` (or the first available primary-style axes).
Keeping this information in one place prevents small helper axes such as
``axlogo`` from being covered by their own annotations.

The overlay never participates in data rendering; it is a read-only annotation
layer added just before the figure is saved.  Its visual parameters can be
provided by a style card's top-level ``Debug`` block.  All drawing is wrapped
so a failure here can never break a normal plot.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping

import matplotlib as mpl


# overlay palette
from .debug_config import (
    BOX as _BOX,
    DEFAULT_DEBUG as _DEFAULT_DEBUG,
    DIM as _DIM,
    MARGIN as _MARGIN,
    NAME as _NAME,
    PALETTE_ROLES,
    PANEL as _PANEL,
    PRIMARY_ORDER as _PRIMARY_ORDER,
    resolve_debug_config as _debug_config,
)


def _raw(ax):
    """Return the underlying matplotlib Axes for an adapter or raw Axes."""
    return getattr(ax, "ax", ax)


def _position(ax):
    try:
        bb = _raw(ax).get_position()
        return float(bb.x0), float(bb.y0), float(bb.width), float(bb.height)
    except Exception:
        return None


def _axis_info_lines(name, pos, w_cm, h_cm):
    """Return the three-line layout entry used in the information panel."""
    l, b, w, h = pos
    return (
        name,
        f"  rect=[{l:.3f}, {b:.3f}, {w:.3f}, {h:.3f}]",
        f"  width: {w * w_cm:.3f} cm  height: {h * h_cm:.3f} cm",
    )


def draw_design_reference(fig) -> None:
    """Draw the dimension overlay onto ``fig`` (a JarvisPLOT Figure wrapper)."""
    try:
        _draw(fig)
    except Exception as exc:  # never break a real plot because of the overlay
        logger = getattr(fig, "logger", None)
        if logger is not None:
            try:
                logger.debug(f"design-reference overlay skipped: {exc}")
            except Exception:
                pass


def _draw(fig) -> None:
    F = fig.fig
    w_in, h_in = (float(v) for v in F.get_size_inches())
    w_cm, h_cm = w_in * 2.54, h_in * 2.54
    debug = _debug_config(fig)
    palette = debug["palette"]
    figure_cfg = debug["figure"]
    axes_cfg = debug["axes"]
    dimension_cfg = debug["dimension"]
    numbered_cfg = debug["numbered_axes"]
    panel_cfg = debug["panel"]
    dimension_color = palette["dimension"]
    box_color = palette["box"]
    name_color = palette["name"]
    margin_color = palette["margin"]

    ov = F.add_axes([0.0, 0.0, 1.0, 1.0], zorder=10_000)
    ov.set_xlim(0.0, 1.0)
    ov.set_ylim(0.0, 1.0)
    ov.set_axis_off()
    ov.patch.set_alpha(0.0)

    def label(x, y, text, *, color=_DIM, size=5.0, ha="center", va="center",
              rotation=0, weight="normal", family="monospace", boxed=True,
              box_alpha=0.82):
        bbox = dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=box_alpha) if boxed else None
        ov.text(x, y, text, color=color, fontsize=size, ha=ha, va=va,
                rotation=rotation, fontweight=weight, family=family, bbox=bbox,
                zorder=10_002, clip_on=False)

    def dim(x0, y0, x1, y1, text, *, vertical=False, line_offset=0.0,
            vertical_label_side="right"):
        """Draw a dimension line offset from the measured edge.

        Labels sit just beside the line instead of being centered on top of
        the arrow, so the blue dimension line remains visible.
        """
        if vertical:
            x0 += line_offset
            x1 += line_offset
        else:
            y0 += line_offset
            y1 += line_offset

        # Add perpendicular terminal bars to make the measured endpoints
        # explicit.  Short gaps use inward-pointing arrows whose tails extend
        # outside the measured interval, matching ``->|  |<-``.
        line_width = float(dimension_cfg["line_width"])
        cap_half_pt = float(dimension_cfg["cap_half_pt"])
        if vertical:
            cap_half = cap_half_pt / (w_in * 72.0)
            ov.plot(
                [x0 - cap_half, x0 + cap_half], [y0, y0],
                color=dimension_color, lw=line_width, solid_capstyle="projecting",
                zorder=10_001, clip_on=False,
            )
            ov.plot(
                [x1 - cap_half, x1 + cap_half], [y1, y1],
                color=dimension_color, lw=line_width, solid_capstyle="projecting",
                zorder=10_001, clip_on=False,
            )
            span = abs(y1 - y0)
            span_pt = span * h_in * 72.0
            arrow_unit = float(dimension_cfg["narrow_length_pt"]) / (h_in * 72.0)
        else:
            cap_half = cap_half_pt / (h_in * 72.0)
            ov.plot(
                [x0, x0], [y0 - cap_half, y0 + cap_half],
                color=dimension_color, lw=line_width, solid_capstyle="projecting",
                zorder=10_001, clip_on=False,
            )
            ov.plot(
                [x1, x1], [y1 - cap_half, y1 + cap_half],
                color=dimension_color, lw=line_width, solid_capstyle="projecting",
                zorder=10_001, clip_on=False,
            )
            span = abs(x1 - x0)
            span_pt = span * w_in * 72.0
            arrow_unit = float(dimension_cfg["narrow_length_pt"]) / (w_in * 72.0)

        narrow_scale = min(
            float(dimension_cfg["narrow_scale_max"]),
            max(
                float(dimension_cfg["narrow_scale_min"]),
                span_pt * float(dimension_cfg["narrow_scale_factor"]),
            ),
        )

        def edge_safe_tail(value, endpoint):
            """Do not let an outside tail extend past the page edge."""
            if endpoint <= 0.0:
                return max(value, 0.0)
            if endpoint >= 1.0:
                return min(value, 1.0)
            return value

        arrow_props = dict(
            arrowstyle="-|>", color=dimension_color, lw=line_width,
            shrinkA=0, shrinkB=0, mutation_scale=narrow_scale,
        )
        if span_pt < float(dimension_cfg["narrow_threshold_pt"]):
            # Put the arrow tips exactly on the endpoint bars while extending
            # their line segments outward beyond those bars.
            outside_len = arrow_unit
            if vertical:
                direction = 1.0 if y1 >= y0 else -1.0
                tail0 = edge_safe_tail(y0 - direction * outside_len, y0)
                tail1 = edge_safe_tail(y1 + direction * outside_len, y1)
                ov.annotate(
                    "",
                    xy=(x0, y0), xytext=(x0, tail0),
                    arrowprops=arrow_props, zorder=10_001,
                )
                ov.annotate(
                    "",
                    xy=(x1, y1), xytext=(x1, tail1),
                    arrowprops=arrow_props, zorder=10_001,
                )
            else:
                direction = 1.0 if x1 >= x0 else -1.0
                tail0 = edge_safe_tail(x0 - direction * outside_len, x0)
                tail1 = edge_safe_tail(x1 + direction * outside_len, x1)
                ov.annotate(
                    "",
                    xy=(x0, y0), xytext=(tail0, y0),
                    arrowprops=arrow_props, zorder=10_001,
                )
                ov.annotate(
                    "",
                    xy=(x1, y1), xytext=(tail1, y1),
                    arrowprops=arrow_props, zorder=10_001,
                )
        else:
            ov.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="<|-|>", color=dimension_color, lw=line_width,
                    shrinkA=0, shrinkB=0,
                    mutation_scale=float(dimension_cfg["wide_scale"]),
                ),
                zorder=10_001,
            )
        label_gap = float(dimension_cfg["label_gap"])
        if vertical:
            if vertical_label_side == "left":
                label_x = (x0 + x1) / 2.0 - label_gap
                label_ha = "right"
            else:
                label_x = (x0 + x1) / 2.0 + label_gap
                label_ha = "left"
            label(
                label_x,
                (y0 + y1) / 2.0,
                text,
                rotation=90,
                ha=label_ha,
                boxed=True,
                color=dimension_color,
                box_alpha=float(dimension_cfg["label_box_alpha"]),
            )
        else:
            label(
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0 - label_gap,
                text,
                va="top",
                boxed=True,
                color=dimension_color,
                box_alpha=float(dimension_cfg["label_box_alpha"]),
            )

    # figure border + size caption
    ov.add_patch(mpl.patches.Rectangle(
        (0, 0), 1, 1, fill=False, ec=margin_color,
        lw=float(figure_cfg["border_linewidth"]), zorder=10_001,
    ))
    label(
        float(figure_cfg["caption_x"]),
        float(figure_cfg["caption_y"]),
        f"design reference  ·  figure {w_cm:.2f} cm × {h_cm:.2f} cm",
        color=margin_color,
        size=float(figure_cfg["caption_size"]),
        va="top",
        weight="bold",
        family="DejaVu Sans",
    )

    # total figure height marker on the far left
    height_marker_x = float(figure_cfg["height_marker_x"])
    dim(height_marker_x, 0.0, height_marker_x, 1.0,
        f"{h_cm:.2f} cm", vertical=True)

    # Resolve positions once.  ``fig.axes`` is the Jarvis-PLOT named-axis
    # mapping, not matplotlib's list of raw axes (which also gains ``ov``).
    axis_positions = []
    for name, ax in fig.axes.items():
        pos = _position(ax)
        if pos is not None:
            axis_positions.append((name, pos))

    # Pick the primary axes to fully dimension and to host the ordered info
    # panel.  The fallback keeps debug rendering useful for unusual cards that
    # do not define the conventional ``ax``/``ax0`` names.
    primary = next(
        (n for n in _PRIMARY_ORDER if any(name == n for name, _ in axis_positions)),
        axis_positions[0][0] if axis_positions else None,
    )
    positions_by_name = dict(axis_positions)
    primary_pos = positions_by_name.get(primary) if primary else None
    corner_gap = float(dimension_cfg["corner_gap"])

    for name, pos in axis_positions:
        l, b, w, h = pos

        # Outline only, so the real plot underneath keeps its true colors.
        # Axes without a real frame (for example axlogo) get a pink design
        # outline; framed axes retain the ordinary black reference outline.
        raw_axis = _raw(fig.axes[name])
        get_frame_on = getattr(raw_axis, "get_frame_on", None)
        frame_on = bool(get_frame_on()) if callable(get_frame_on) else True
        outline_color = box_color if frame_on else margin_color
        ov.add_patch(mpl.patches.Rectangle((l, b), w, h, fill=False,
                                           ec=outline_color,
                                           lw=float(axes_cfg["outline_linewidth"]),
                                           zorder=10_001))

        if not frame_on:
            # Only frameless axes get the hash-like endpoint treatment.
            # Framed axes already have their own complete border.
            extension = min(
                float(axes_cfg["frameless_extension_max"]),
                max(
                    float(axes_cfg["frameless_extension_min"]),
                    float(axes_cfg["frameless_extension_factor"])
                    * min(w, h),
                ),
            )
            for xa, xb, y in (
                (l - extension, l, b),
                (l + w, l + w + extension, b),
                (l - extension, l, b + h),
                (l + w, l + w + extension, b + h),
            ):
                ov.plot([xa, xb], [y, y], color=outline_color,
                        lw=float(axes_cfg["outline_linewidth"]),
                        solid_capstyle="projecting", zorder=10_001,
                        clip_on=False)
            for ya, yb, x in (
                (b - extension, b, l),
                (b + h, b + h + extension, l),
                (b - extension, b, l + w),
                (b + h, b + h + extension, l + w),
            ):
                ov.plot([x, x], [ya, yb], color=outline_color,
                        lw=float(axes_cfg["outline_linewidth"]),
                        solid_capstyle="projecting", zorder=10_001,
                        clip_on=False)

        if name == primary:
            yt = b + h
            # Keep the two margin dimensions off the shared top-left corner:
            # the horizontal one lands on the left spine below the corner,
            # while the vertical ones land on the top/bottom spines to its
            # right.
            dim(0.0, yt, l, yt, f"{l * w_cm:.3f} cm", line_offset=-corner_gap)
            # Numbered axes get their top/bottom edge dimensions in the
            # right-hand column group below.  Keep the original primary-axis
            # top/bottom dimensions for an unnumbered ``ax`` layout.
            if not (name.startswith("ax") and name[2:].isdigit()):
                dim(
                    l,
                    yt,
                    l,
                    1.0,
                    f"{(1.0 - yt) * h_cm:.3f} cm",
                    vertical=True,
                    line_offset=corner_gap,
                )
                dim(
                    l,
                    0.0,
                    l,
                    b,
                    f"{b * h_cm:.3f} cm",
                    vertical=True,
                    line_offset=corner_gap,
                )
        elif name.startswith("axc") and primary_pos is not None:
            # gap between the primary axes' right edge and this colorbar's left edge
            pl, pb, pw, ph = primary_pos
            ygap = b + h / 2.0
            if l > pl + pw:
                dim(
                    pl + pw,
                    ygap,
                    l,
                    ygap,
                    f"{(l - (pl + pw)) * w_cm:.3f} cm",
                    line_offset=float(dimension_cfg["colorbar_line_offset"]),
                )

    # Numbered subplot axes use the same top/bottom-to-figure-edge distances
    # as the primary ax0 marker.  Put ax0 at the rightmost column, then
    # stagger ax1, ax2, ... to the left so these labels stay clear of the
    # axes-layout information card.
    numbered_axes = sorted(
        (
            int(name[2:]),
            name,
            positions_by_name[name],
        )
        for name, _ in axis_positions
        if name.startswith("ax") and name[2:].isdigit()
    )
    if primary_pos is not None and numbered_axes:
        right_edge = float(numbered_cfg["right_edge"])
        marker_step = float(numbered_cfg["marker_step"])
        base_marker_x = right_edge - corner_gap
        label_side = str(numbered_cfg["label_side"]).lower()
        if label_side not in {"left", "right"}:
            label_side = "left"
        for number, _name, pos in numbered_axes:
            _left, bottom, _width, height = pos
            marker_x = base_marker_x - number * marker_step
            dim(
                marker_x,
                0.0,
                marker_x,
                bottom,
                f"{bottom * h_cm:.3f} cm",
                vertical=True,
                vertical_label_side=label_side,
            )
            dim(
                marker_x,
                bottom + height,
                marker_x,
                1.0,
                f"{(1.0 - bottom - height) * h_cm:.3f} cm",
                vertical=True,
                vertical_label_side=label_side,
            )

    # Put every axes' identifying information in the main plotting area.  The
    # primary axis is listed first, followed by the remaining axes in the
    # style-card order, so the panel is deterministic and easy to scan.
    if primary_pos is not None:
        pl, pb, pw, ph = primary_pos
        ordered_names = [primary] + [
            name for name, _ in axis_positions if name != primary
        ]
        header = "axes layout"
        entries = [
            _axis_info_lines(name, positions_by_name[name], w_cm, h_cm)
            for name in ordered_names
        ]

        pad_x = min(
            float(panel_cfg["pad_x_max"]),
            max(float(panel_cfg["pad_x_min"]), pw * float(panel_cfg["pad_x_factor"])),
        )
        pad_y = min(
            float(panel_cfg["pad_y_max"]),
            max(float(panel_cfg["pad_y_min"]), ph * float(panel_cfg["pad_y_factor"])),
        )
        panel_x = pl + pad_x
        panel_y = pb + ph - pad_y
        # Keep the layout information card compact by default.  This is the
        # overlay card's bound only; it does not change the real axes limits.
        panel_right = min(pl + pw - pad_x, float(panel_cfg["right"]))
        panel_w = panel_right - panel_x
        panel_h = ph - 2.0 * pad_y

        if panel_w > 0.0 and panel_h > 0.0:
            available_w_pt = panel_w * w_in * 72.0
            available_h_pt = panel_h * h_in * 72.0
            max_detail_chars = max(
                len(line)
                for entry in entries
                for line in entry[1:]
            )

            # Preserve the old visual hierarchy: the axis name was 7 pt and
            # bold, while its rect and cm dimensions were 4.6 pt.  Shrink the
            # complete group only when a short multi-panel axis cannot fit all
            # entries at those sizes.
            header_target = float(panel_cfg["header_size"])
            name_target = float(panel_cfg["name_size"])
            detail_target = float(panel_cfg["detail_size"])
            entry_gap_target = float(panel_cfg["entry_gap"])
            header_leading = float(panel_cfg["header_leading"])
            name_leading = float(panel_cfg["name_leading"])
            detail_leading = float(panel_cfg["detail_leading"])
            target_height = (
                header_target * header_leading
                + len(entries)
                * (name_target * name_leading + 2.0 * detail_target * detail_leading)
                + max(0, len(entries) - 1) * entry_gap_target
            )
            size_from_height = available_h_pt / max(1.0, target_height)
            size_from_width = available_w_pt / max(
                1.0, 0.62 * detail_target * max_detail_chars
            )
            scale = min(1.0, size_from_height, size_from_width)
            header_size = max(float(panel_cfg["min_header_size"]), header_target * scale)
            name_size = max(float(panel_cfg["min_name_size"]), name_target * scale)
            detail_size = max(float(panel_cfg["min_detail_size"]), detail_target * scale)
            entry_gap = entry_gap_target * scale

            ov.add_patch(
                mpl.patches.FancyBboxPatch(
                    (panel_x, panel_y - panel_h),
                    panel_w,
                    panel_h,
                    boxstyle=str(panel_cfg["boxstyle"]),
                    facecolor=palette["panel"],
                    edgecolor="none",
                    linewidth=0.0,
                    alpha=float(panel_cfg["alpha"]),
                    zorder=10_001,
                )
            )
            text_x = panel_x + min(0.008, panel_w * 0.02)
            cursor_y = panel_y - min(0.006, panel_h * 0.03)
            figure_height_pt = h_in * 72.0

            def panel_text(text, size, *, weight="normal"):
                ov.text(
                    text_x,
                    cursor_y,
                    text,
                    color=name_color,
                    fontsize=size,
                    ha="left",
                    va="top",
                    fontweight=weight,
                    family="monospace",
                    zorder=10_002,
                    clip_on=True,
                )

            panel_text(header, header_size, weight="bold")
            cursor_y -= header_size * header_leading / figure_height_pt
            for index, (name, rect_line, size_line) in enumerate(entries):
                panel_text(name, name_size, weight="bold")
                cursor_y -= name_size * name_leading / figure_height_pt
                panel_text(rect_line, detail_size)
                cursor_y -= detail_size * detail_leading / figure_height_pt
                panel_text(size_line, detail_size)
                cursor_y -= detail_size * detail_leading / figure_height_pt
                if index < len(entries) - 1:
                    cursor_y -= entry_gap / figure_height_pt

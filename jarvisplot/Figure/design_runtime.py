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
insets), and colorbar axes get a gap dimension to the primary axes.  The
per-axes geometry is collected into one compact, ordered panel inside the
primary ``ax`` (or the first available primary-style axes).  Keeping this
information in one place prevents small helper axes such as ``axlogo`` from
being covered by their own annotations.

The overlay never participates in data rendering; it is a read-only annotation
layer added just before the figure is saved.  All drawing is wrapped so a
failure here can never break a normal plot.
"""

from __future__ import annotations

import matplotlib as mpl


# overlay palette
_DIM    = "#0A7FB6"   # blue dimension lines + labels
_BOX    = "#111111"   # axes outline
_NAME   = "#1A237E"   # axes name + rect text
_MARGIN = "#FF3FA4"   # pink figure border / caption

# order in which we look for the "primary" axes to fully dimension
_PRIMARY_ORDER = ("ax", "ax0", "ax1", "axtri")


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

    def dim(x0, y0, x1, y1, text, *, vertical=False, line_offset=0.0):
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

        ov.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="<|-|>", color=_DIM, lw=0.7,
                                    shrinkA=0, shrinkB=0, mutation_scale=6),
                    zorder=10_001)
        label_gap = 0.008
        if vertical:
            label(
                (x0 + x1) / 2.0 + label_gap,
                (y0 + y1) / 2.0,
                text,
                rotation=90,
                ha="left",
                boxed=True,
                box_alpha=0.5,
            )
        else:
            label(
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0 - label_gap,
                text,
                va="top",
                boxed=True,
                box_alpha=0.5,
            )

    # figure border + size caption
    ov.add_patch(mpl.patches.Rectangle((0, 0), 1, 1, fill=False, ec=_MARGIN,
                                       lw=0.9, zorder=10_001))
    label(0.5, 0.995, f"design reference  ·  figure {w_cm:.2f} cm × {h_cm:.2f} cm",
          color=_MARGIN, size=6.0, va="top", weight="bold", family="DejaVu Sans")

    # total figure height marker on the far left
    dim(0.035, 0.0, 0.035, 1.0, f"{h_cm:.2f} cm", vertical=True)

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

    for name, pos in axis_positions:
        l, b, w, h = pos

        # Outline only, so the real plot underneath keeps its true colors.
        # Axes without a real frame (for example axlogo) get a pink design
        # outline; framed axes retain the ordinary black reference outline.
        raw_axis = _raw(fig.axes[name])
        get_frame_on = getattr(raw_axis, "get_frame_on", None)
        frame_on = bool(get_frame_on()) if callable(get_frame_on) else True
        outline_color = _BOX if frame_on else _MARGIN
        ov.add_patch(mpl.patches.Rectangle((l, b), w, h, fill=False,
                                           ec=outline_color, lw=0.45,
                                           zorder=10_001))

        if not frame_on:
            # Only frameless axes get the hash-like endpoint treatment.
            # Framed axes already have their own complete border.
            extension = min(0.018, max(0.006, 0.09 * min(w, h)))
            for xa, xb, y in (
                (l - extension, l, b),
                (l + w, l + w + extension, b),
                (l - extension, l, b + h),
                (l + w, l + w + extension, b + h),
            ):
                ov.plot([xa, xb], [y, y], color=outline_color, lw=0.45,
                        solid_capstyle="projecting", zorder=10_001,
                        clip_on=False)
            for ya, yb, x in (
                (b - extension, b, l),
                (b + h, b + h + extension, l),
                (b - extension, b, l + w),
                (b + h, b + h + extension, l + w),
            ):
                ov.plot([x, x], [ya, yb], color=outline_color, lw=0.45,
                        solid_capstyle="projecting", zorder=10_001,
                        clip_on=False)

        if name == primary:
            yt = b + h
            # Keep the two margin dimensions off the shared top-left corner:
            # the horizontal one lands on the left spine below the corner,
            # while the vertical ones land on the top/bottom spines to its
            # right.
            corner_gap = 0.018
            dim(0.0, yt, l, yt, f"{l * w_cm:.3f} cm", line_offset=-corner_gap)
            # top inset (axes top edge → figure top)
            dim(
                l,
                yt,
                l,
                1.0,
                f"{(1.0 - yt) * h_cm:.3f} cm",
                vertical=True,
                line_offset=corner_gap,
            )
            # bottom inset (figure bottom → axes bottom edge)
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
                    line_offset=0.024,
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

        pad_x = min(0.018, max(0.004, pw * 0.025))
        pad_y = min(0.018, max(0.004, ph * 0.035))
        panel_x = pl + pad_x
        panel_y = pb + ph - pad_y
        # Keep the layout information card compact by default.  This is the
        # overlay card's bound only; it does not change the real axes limits.
        panel_right = min(pl + pw - pad_x, 0.70)
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
            header_target = 6.0
            name_target = 7.0
            detail_target = 4.6
            entry_gap_target = 5.0
            header_leading = 1.20
            name_leading = 1.15
            detail_leading = 1.08
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
            header_size = max(2.2, header_target * scale)
            name_size = max(2.2, name_target * scale)
            detail_size = max(1.8, detail_target * scale)
            entry_gap = entry_gap_target * scale

            ov.add_patch(
                mpl.patches.FancyBboxPatch(
                    (panel_x, panel_y - panel_h),
                    panel_w,
                    panel_h,
                    boxstyle="round,pad=0.004",
                    facecolor="white",
                    edgecolor="none",
                    linewidth=0.0,
                    alpha=0.88,
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
                    color=_NAME,
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

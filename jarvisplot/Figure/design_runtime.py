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

Nothing here decides what an annotation looks like.  Geometry is computed
below; every appearance kwarg is splatted out of a block named after the
matplotlib callable it feeds -- ``cfg["cap"]["plot"]``, ``cfg["border"]
["Rectangle"]``, ``cfg["hlabel"]["text"]`` -- which a style card's ``Debug``
block, or one figure's YAML ``debug:`` mapping, may replace wholesale.  See
:mod:`jarvisplot.Figure.debug_config`.

The overlay never participates in data rendering; it is a read-only annotation
layer added just before the figure is saved.  All drawing is wrapped so a
failure here can never break a normal plot.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib as mpl


from .debug_config import resolve_debug_config as _debug_config


def _shown(cfg) -> bool:
    """Whether a JSON-declared annotation element is switched on.

    Absent means on, so a card written before an element existed keeps drawing
    it. The switch is a veto, not a command: the structural conditions in
    :func:`_draw` (which axes is primary, which is a colorbar, whether an axes
    has a frame) still decide whether the element applies at all.
    """
    if not isinstance(cfg, Mapping):
        return bool(cfg)
    return bool(cfg.get("show", True))


def _text(template, values, fig=None) -> str:
    """Render a JSON printf template, falling back to a readable string.

    Cards are not schema-validated, so a template with the wrong number of
    conversions would otherwise raise inside the overlay and -- because
    :func:`draw_design_reference` swallows everything -- silently erase the
    whole annotation layer instead of just this one label.
    """
    try:
        return str(template) % values
    except (TypeError, ValueError) as exc:
        logger = getattr(fig, "logger", None)
        warn = getattr(logger, "warning", None)
        if callable(warn):
            try:
                warn(f"Debug: bad template {template!r} ({exc}); using raw values.")
            except Exception:
                pass
        return str(values)


def _raw(ax):
    """Return the underlying matplotlib Axes for an adapter or raw Axes."""
    return getattr(ax, "ax", ax)


def _position(ax):
    try:
        bb = _raw(ax).get_position()
        return float(bb.x0), float(bb.y0), float(bb.width), float(bb.height)
    except Exception:
        return None


def _axis_info_lines(name, pos, w_cm, h_cm, *, detail_cfg, fig=None):
    """Return the three-line layout entry used in the information panel."""
    l, b, w, h = pos
    return (
        name,
        _text(detail_cfg["rect_template"], (l, b, w, h), fig),
        _text(detail_cfg["size_template"], (w * w_cm, h * h_cm), fig),
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
    debug = _debug_config(fig)
    figure_cfg = debug["figure"]
    axes_cfg = debug["axes"]
    dimension_cfg = debug["dimension"]
    numbered_cfg = debug["numbered_axes"]
    panel_cfg = debug["panel"]
    margins_cfg = debug["margins"]
    colorbar_cfg = debug["colorbar_gap"]
    units_cfg = debug["units"]
    pt_per_inch = float(units_cfg["pt_per_inch"])
    cm_per_inch = float(units_cfg["cm_per_inch"])
    w_cm, h_cm = w_in * cm_per_inch, h_in * cm_per_inch

    ov = F.add_axes([0.0, 0.0, 1.0, 1.0], **debug["overlay"]["add_axes"])
    ov.set_xlim(0.0, 1.0)
    ov.set_ylim(0.0, 1.0)
    ov.set_axis_off()
    ov.patch.set_alpha(0.0)

    def dim(x0, y0, x1, y1, text, *, vertical=False, line_offset=0.0,
            vertical_label_side="right"):
        """Draw a dimension line offset from the measured edge.

        Labels sit just beside the line instead of being centered on top of
        the arrow, so the dimension line remains visible.
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
        cap_style = dimension_cfg["cap"]["plot"]
        cap_half_pt = float(dimension_cfg["cap_half_pt"])
        if vertical:
            cap_half = cap_half_pt / (w_in * pt_per_inch)
            ov.plot([x0 - cap_half, x0 + cap_half], [y0, y0], **cap_style)
            ov.plot([x1 - cap_half, x1 + cap_half], [y1, y1], **cap_style)
            span = abs(y1 - y0)
            span_pt = span * h_in * pt_per_inch
            arrow_unit = float(dimension_cfg["narrow_length_pt"]) / (h_in * pt_per_inch)
        else:
            cap_half = cap_half_pt / (h_in * pt_per_inch)
            ov.plot([x0, x0], [y0 - cap_half, y0 + cap_half], **cap_style)
            ov.plot([x1, x1], [y1 - cap_half, y1 + cap_half], **cap_style)
            span = abs(x1 - x0)
            span_pt = span * w_in * pt_per_inch
            arrow_unit = float(dimension_cfg["narrow_length_pt"]) / (w_in * pt_per_inch)

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

        if span_pt < float(dimension_cfg["narrow_threshold_pt"]):
            # Put the arrow tips exactly on the endpoint bars while extending
            # their line segments outward beyond those bars.  The arrowhead is
            # the one kwarg the card cannot fix: it scales with the span.
            narrow = dict(dimension_cfg["narrow_arrow"]["annotate"])
            narrow["arrowprops"] = {
                **narrow.get("arrowprops", {}),
                "mutation_scale": narrow_scale,
            }
            outside_len = arrow_unit
            if vertical:
                direction = 1.0 if y1 >= y0 else -1.0
                tail0 = edge_safe_tail(y0 - direction * outside_len, y0)
                tail1 = edge_safe_tail(y1 + direction * outside_len, y1)
                ov.annotate("", xy=(x0, y0), xytext=(x0, tail0), **narrow)
                ov.annotate("", xy=(x1, y1), xytext=(x1, tail1), **narrow)
            else:
                direction = 1.0 if x1 >= x0 else -1.0
                tail0 = edge_safe_tail(x0 - direction * outside_len, x0)
                tail1 = edge_safe_tail(x1 + direction * outside_len, x1)
                ov.annotate("", xy=(x0, y0), xytext=(tail0, y0), **narrow)
                ov.annotate("", xy=(x1, y1), xytext=(tail1, y1), **narrow)
        else:
            ov.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                **dimension_cfg["wide_arrow"]["annotate"],
            )
        label_gap = float(dimension_cfg["label_gap"])
        if vertical:
            if vertical_label_side == "left":
                label_x = (x0 + x1) / 2.0 - label_gap
                label_style = dimension_cfg["vlabel_left"]["text"]
            else:
                label_x = (x0 + x1) / 2.0 + label_gap
                label_style = dimension_cfg["vlabel_right"]["text"]
            ov.text(label_x, (y0 + y1) / 2.0, text, **label_style)
        else:
            ov.text(
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0 - label_gap,
                text,
                **dimension_cfg["hlabel"]["text"],
            )

    # figure border + size caption
    border_cfg = figure_cfg["border"]
    if _shown(border_cfg):
        ov.add_patch(mpl.patches.Rectangle((0, 0), 1, 1, **border_cfg["Rectangle"]))

    caption_cfg = figure_cfg["caption"]
    if _shown(caption_cfg):
        ov.text(
            float(caption_cfg["x"]),
            float(caption_cfg["y"]),
            _text(caption_cfg["template"], (w_cm, h_cm), fig),
            **caption_cfg["text"],
        )

    # total figure height marker on the far left
    height_cfg = figure_cfg["height_marker"]
    if _shown(height_cfg):
        height_marker_x = float(height_cfg["x"])
        dim(height_marker_x, 0.0, height_marker_x, 1.0,
            _text(height_cfg["template"], h_cm, fig), vertical=True)

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
        (n for n in debug["primary_order"] if any(name == n for name, _ in axis_positions)),
        axis_positions[0][0] if axis_positions else None,
    )
    positions_by_name = dict(axis_positions)
    primary_pos = positions_by_name.get(primary) if primary else None
    corner_gap = float(dimension_cfg["corner_gap"])

    for name, pos in axis_positions:
        l, b, w, h = pos

        # Outline only, so the real plot underneath keeps its true colors.
        # Axes without a real frame (for example axlogo) get a design-coloured
        # outline; framed axes retain the ordinary reference outline.
        raw_axis = _raw(fig.axes[name])
        get_frame_on = getattr(raw_axis, "get_frame_on", None)
        frame_on = bool(get_frame_on()) if callable(get_frame_on) else True
        outline_cfg = axes_cfg["outline"]["framed" if frame_on else "frameless"]
        if _shown(axes_cfg["outline"]):
            ov.add_patch(
                mpl.patches.Rectangle((l, b), w, h, **outline_cfg["Rectangle"])
            )

        if not frame_on and _shown(axes_cfg["corner_ticks"]):
            # Only frameless axes get the hash-like endpoint treatment.
            # Framed axes already have their own complete border.
            tick_style = axes_cfg["corner_ticks"]["plot"]
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
                ov.plot([xa, xb], [y, y], **tick_style)
            for ya, yb, x in (
                (b - extension, b, l),
                (b + h, b + h + extension, l),
                (b - extension, b, l + w),
                (b + h, b + h + extension, l + w),
            ):
                ov.plot([x, x], [ya, yb], **tick_style)

        if name == primary:
            yt = b + h
            # Keep the two margin dimensions off the shared top-left corner:
            # the horizontal one lands on the left spine below the corner,
            # while the vertical ones land on the top/bottom spines to its
            # right.
            if _shown(margins_cfg["left"]):
                dim(0.0, yt, l, yt,
                    _text(margins_cfg["template"], l * w_cm, fig),
                    line_offset=-corner_gap)
            # Numbered axes get their top/bottom edge dimensions in the
            # right-hand column group below.  Keep the original primary-axis
            # top/bottom dimensions for an unnumbered ``ax`` layout.
            if not (name.startswith("ax") and name[2:].isdigit()):
                if _shown(margins_cfg["top"]):
                    dim(
                        l, yt, l, 1.0,
                        _text(margins_cfg["template"], (1.0 - yt) * h_cm, fig),
                        vertical=True,
                        line_offset=corner_gap,
                    )
                if _shown(margins_cfg["bottom"]):
                    dim(
                        l, 0.0, l, b,
                        _text(margins_cfg["template"], b * h_cm, fig),
                        vertical=True,
                        line_offset=corner_gap,
                    )
        elif name.startswith("axc") and primary_pos is not None:
            # gap between the primary axes' right edge and this colorbar's left edge
            pl, pb, pw, ph = primary_pos
            ygap = b + h / 2.0
            if l > pl + pw and _shown(colorbar_cfg):
                dim(
                    pl + pw,
                    ygap,
                    l,
                    ygap,
                    _text(colorbar_cfg["template"], (l - (pl + pw)) * w_cm, fig),
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
    if primary_pos is not None and numbered_axes and _shown(numbered_cfg):
        right_edge = float(numbered_cfg["right_edge"])
        marker_step = float(numbered_cfg["marker_step"])
        base_marker_x = right_edge - corner_gap
        label_side = str(numbered_cfg["label_side"]).lower()
        if label_side not in {"left", "right"}:
            label_side = "left"
        for number, _name, pos in numbered_axes:
            _left, bottom, _width, height = pos
            marker_x = base_marker_x - number * marker_step
            if _shown(numbered_cfg["bottom"]):
                dim(
                    marker_x, 0.0, marker_x, bottom,
                    _text(numbered_cfg["template"], bottom * h_cm, fig),
                    vertical=True,
                    vertical_label_side=label_side,
                )
            if _shown(numbered_cfg["top"]):
                dim(
                    marker_x, bottom + height, marker_x, 1.0,
                    _text(numbered_cfg["template"], (1.0 - bottom - height) * h_cm, fig),
                    vertical=True,
                    vertical_label_side=label_side,
                )

    # Put every axes' identifying information in the main plotting area.  The
    # primary axis is listed first, followed by the remaining axes in the
    # style-card order, so the panel is deterministic and easy to scan.
    if primary_pos is not None and _shown(panel_cfg):
        pl, pb, pw, ph = primary_pos
        header_cfg = panel_cfg["header"]
        name_cfg = panel_cfg["name"]
        detail_cfg = panel_cfg["detail"]
        ordered_names = [primary] + [
            name for name, _ in axis_positions if name != primary
        ]
        header = str(header_cfg["label"])
        entries = [
            _axis_info_lines(
                name, positions_by_name[name], w_cm, h_cm,
                detail_cfg=detail_cfg,
                fig=fig,
            )
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
            available_w_pt = panel_w * w_in * pt_per_inch
            available_h_pt = panel_h * h_in * pt_per_inch
            max_detail_chars = max(
                len(line)
                for entry in entries
                for line in entry[1:]
            )

            # Preserve the visual hierarchy: the axis name is larger and bold,
            # its rect and cm dimensions smaller.  Shrink the complete group
            # only when a short multi-panel axis cannot fit all entries at
            # those sizes.
            header_target = float(header_cfg["size"])
            name_target = float(name_cfg["size"])
            detail_target = float(detail_cfg["size"])
            entry_gap_target = float(panel_cfg["entry_gap"])
            header_leading = float(header_cfg["leading"])
            name_leading = float(name_cfg["leading"])
            detail_leading = float(detail_cfg["leading"])
            target_height = (
                header_target * header_leading
                + len(entries)
                * (
                    name_target * name_leading
                    + float(panel_cfg["detail_lines_per_entry"])
                    * detail_target
                    * detail_leading
                )
                + max(0, len(entries) - 1) * entry_gap_target
            )
            size_from_height = available_h_pt / max(1.0, target_height)
            size_from_width = available_w_pt / max(
                1.0,
                float(panel_cfg["char_width_factor"]) * detail_target * max_detail_chars,
            )
            scale = min(1.0, size_from_height, size_from_width)
            header_size = max(float(header_cfg["min_size"]), header_target * scale)
            name_size = max(float(name_cfg["min_size"]), name_target * scale)
            detail_size = max(float(detail_cfg["min_size"]), detail_target * scale)
            entry_gap = entry_gap_target * scale

            ov.add_patch(
                mpl.patches.FancyBboxPatch(
                    (panel_x, panel_y - panel_h),
                    panel_w,
                    panel_h,
                    **panel_cfg["box"]["FancyBboxPatch"],
                )
            )
            text_x = panel_x + min(
                float(panel_cfg["text_inset_x_max"]),
                panel_w * float(panel_cfg["text_inset_x_factor"]),
            )
            cursor_y = panel_y - min(
                float(panel_cfg["text_inset_y_max"]),
                panel_h * float(panel_cfg["text_inset_y_factor"]),
            )
            figure_height_pt = h_in * pt_per_inch

            def panel_text(text, size, style):
                ov.text(text_x, cursor_y, text, fontsize=size, **style)

            panel_text(header, header_size, header_cfg["text"])
            cursor_y -= header_size * header_leading / figure_height_pt
            for index, (name, rect_line, size_line) in enumerate(entries):
                panel_text(name, name_size, name_cfg["text"])
                cursor_y -= name_size * name_leading / figure_height_pt
                panel_text(rect_line, detail_size, detail_cfg["text"])
                cursor_y -= detail_size * detail_leading / figure_height_pt
                panel_text(size_line, detail_size, detail_cfg["text"])
                cursor_y -= detail_size * detail_leading / figure_height_pt
                if index < len(entries) - 1:
                    cursor_y -= entry_gap / figure_height_pt


# --------------------------------------------------------------------------- #
# Ternary design reference
# --------------------------------------------------------------------------- #


def draw_ternary_reference(fig, tick_anchors) -> None:
    """Mark the ternary tick anchors and draw leaders out to the axis labels.

    Called from the ``axtri`` setter, *during* axes construction rather than
    from the post-render overlay pass. That placement is deliberate: these
    artists sit underneath the data layers, and moving them to the overlay
    would silently put them on top.

    ``tick_anchors`` is the ``(x, y)`` pairs the caller already computed for the
    bottom / right / left tick labels.
    """
    try:
        _draw_ternary(fig, tick_anchors)
    except Exception as exc:  # never break a real plot because of the overlay
        logger = getattr(fig, "logger", None)
        if logger is not None:
            try:
                logger.debug(f"ternary design reference skipped: {exc}")
            except Exception:
                pass


#: Each leader runs from the opposite vertex, through the midpoint of the edge
#: it labels, out to the label anchor. The first two points are pure triangle
#: geometry; the third comes from the style card.
_TERNARY_LEADERS = {
    "right": ((0.0, 0.0), (0.75, 0.5)),
    "bottom": ((0.5, 1.0), (0.5, 0.0)),
    "left": ((1.0, 0.0), (0.25, 0.5)),
}


def _draw_ternary(fig, tick_anchors) -> None:
    cfg = _debug_config(fig).get("ternary", {})
    axtri = fig.axtri

    anchors_cfg = cfg.get("tick_anchors", {})
    if _shown(anchors_cfg):
        style = anchors_cfg.get("scatter", {})
        for xs, ys in tick_anchors:
            axtri.scatter(x=xs, y=ys, **style)

    leaders_cfg = cfg.get("label_leaders", {})
    if not _shown(leaders_cfg):
        return
    style = leaders_cfg.get("plot", {})
    labels = (fig.frame.get("axtri", {}) or {}).get("labels", {}) or {}
    for side, ((x0, y0), (x1, y1)) in _TERNARY_LEADERS.items():
        spec = labels.get(f"{side}style") or {}
        try:
            x2, y2 = float(spec["x"]), float(spec["y"])
        except (KeyError, TypeError, ValueError):
            continue
        axtri.plot(x=[x0, x1, x2], y=[y0, y1, y2], **style)

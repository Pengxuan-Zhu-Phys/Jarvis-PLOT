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
insets), and colorbar axes get a gap dimension to the primary axes.

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
              rotation=0, weight="normal", family="monospace", boxed=True):
        bbox = dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.82) if boxed else None
        ov.text(x, y, text, color=color, fontsize=size, ha=ha, va=va,
                rotation=rotation, fontweight=weight, family=family, bbox=bbox,
                zorder=10_002, clip_on=False)

    def dim(x0, y0, x1, y1, text, *, vertical=False):
        ov.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="<|-|>", color=_DIM, lw=0.7,
                                    shrinkA=0, shrinkB=0, mutation_scale=6),
                    zorder=10_001)
        label((x0 + x1) / 2.0, (y0 + y1) / 2.0, text,
              rotation=90 if vertical else 0)

    # figure border + size caption
    ov.add_patch(mpl.patches.Rectangle((0, 0), 1, 1, fill=False, ec=_MARGIN,
                                       lw=0.9, zorder=10_001))
    label(0.5, 0.995, f"design reference  ·  figure {w_cm:.2f} cm × {h_cm:.2f} cm",
          color=_MARGIN, size=6.0, va="top", weight="bold", family="DejaVu Sans")

    # total figure height marker on the far left
    dim(0.012, 0.0, 0.012, 1.0, f"{h_cm:.2f} cm", vertical=True)

    # pick the primary axes to fully dimension
    primary = next((n for n in _PRIMARY_ORDER if n in fig.axes), None)
    primary_pos = _position(fig.axes[primary]) if primary else None

    for name, ax in fig.axes.items():
        pos = _position(ax)
        if pos is None:
            continue
        l, b, w, h = pos

        # outline only, so the real plot underneath keeps its true colors
        ov.add_patch(mpl.patches.Rectangle((l, b), w, h, fill=False, ec=_BOX,
                                           lw=0.9, zorder=10_001))

        # name + rect + cm sizes — anchored so the text stays inside the figure
        if l + w / 2.0 > 0.6:            # axes sits in the right half
            tx, ha = min(l + w, 0.992), "right"
        else:
            tx, ha = max(l + 0.012, 0.008), "left"
        ty = min(b + h - 0.015, 0.93)
        label(tx, ty, name, color=_NAME, size=7.0, ha=ha, va="top",
              weight="bold", boxed=True)
        label(tx, ty - 0.057,
              f"[{l:.3f}, {b:.3f}, {w:.3f}, {h:.3f}]\n"
              f"width: {w * w_cm:.3f} cm   height: {h * h_cm:.3f} cm",
              color=_NAME, size=4.6, ha=ha, va="top", boxed=True)

        if name == primary:
            yt = b + h
            # left inset (figure left edge → axes left edge), drawn along the top edge
            dim(0.0, yt, l, yt, f"{l * w_cm:.3f} cm")
            # top inset (axes top edge → figure top)
            dim(l, yt, l, 1.0, f"{(1.0 - yt) * h_cm:.3f} cm", vertical=True)
            # bottom inset (figure bottom → axes bottom edge)
            dim(l, 0.0, l, b, f"{b * h_cm:.3f} cm", vertical=True)
        elif name.startswith("axc") and primary_pos is not None:
            # gap between the primary axes' right edge and this colorbar's left edge
            pl, pb, pw, ph = primary_pos
            ygap = b + h / 2.0
            if l > pl + pw:
                dim(pl + pw, ygap, l, ygap, f"{(l - (pl + pw)) * w_cm:.3f} cm")

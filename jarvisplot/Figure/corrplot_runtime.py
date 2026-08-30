#!/usr/bin/env python3

"""``method: corrplot`` -- R's corrplot glyphs, drawn on the reserved panel.

This is a drawing primitive, not a macro over ``scatter``.  The first port of
corrplot here *was* a scatter layer with ``s = abs_rho * <a number>``, and it
gave the right picture for exactly one of R's seven methods.  ``square`` needs
a marker whose side, not its area, tracks ``|rho|``; ``ellipse`` needs an
eccentricity and a tilt; ``pie`` needs a wedge; ``color`` needs to fill the
cell; ``number`` needs no marker at all.  None of those is a scatter argument,
and every one of them is what somebody means when they write
``method: circle``.

What this module owns is everything inside the cell.  What it does **not**
own is which variable sits where: ``order``, ``hclust.method`` and ``addrect``
are resolved before the figure exists (see :mod:`~jarvisplot.Figure.corr_order`
and ``core_runtime.prebuild_correlations``), because the tick labels are
written at config time and a render-time reordering would leave every label
naming the wrong column with nothing to say so.  The block boundaries arrive
here already computed, in ``__corr_blocks__``.

Two sizes are derived rather than configured, both for the same reason -- a
glyph that does not scale with the cell either escapes it or vanishes in it:

* the glyph, from the cell size in data units (a cell is 1.0 by construction);
* the coefficient text, from the cell size in **points**, measured off the
  axes at draw time so it is right even when the width clamp shrank the cell.

Anything outside the cell -- variable names, colorbar numbers -- is fixed in
points by the style card and is none of this module's business.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["GLYPHS", "draw_corrplot"]

#: R's ``method`` argument.  ``shade`` is ``color`` plus a sign hatch here,
#: which is what it amounts to once the underlying colour scale is diverging.
GLYPHS = ("circle", "square", "ellipse", "color", "shade", "pie", "number")

_TRIANGLES = ("full", "upper", "lower")

#: Formals resolved at config time.  They arrive because the user wrote them
#: in the layer, and are dropped here rather than forwarded to matplotlib.
_CONFIG_TIME = ("order", "hclust.method", "addrect", "col", "tl.pos", "tl.cex",
                "tl.col", "tl.srt", "tl.offset", "cl.cex")


def _pop(kwargs: dict, name: str, default=None):
    value = kwargs.pop(name, default)
    return default if value is None else value


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "t"}
    return bool(value)


def _matrix_size(df, ix: np.ndarray, iy: np.ndarray) -> int:
    """The side length of the matrix, stated by the transform where possible.

    A ``triangle`` selection leaves holes, so the largest index present is not
    the side length; ``correlation`` publishes ``__grid_nx__`` precisely so
    this does not have to be guessed.
    """
    for key in ("__grid_nx__", "__grid_ny__"):
        if key in getattr(df, "columns", ()):
            try:
                return int(np.asarray(df[key])[0])
            except Exception:
                pass
    return int(max(ix.max(), iy.max())) + 1


def _cell_points(ax, n: int) -> float:
    """One cell's side in points, measured off the axes.

    Read from the axes rather than from the style card because the card states
    an *intended* cell size and the geometry solver is allowed to shrink it to
    hold the 170 mm page width.  Measuring is the only way the coefficient
    text follows that shrink.
    """
    try:
        box = ax.get_window_extent()
        dpi = float(ax.figure.dpi)
        return min(abs(box.width), abs(box.height)) / dpi * 72.0 / max(n, 1)
    except Exception:
        return 12.0


def _screen_sign(ax) -> float:
    """``-1`` when the y axis points down, which the correlation panel does.

    Every angle below is written the way it is read off the page -- a positive
    correlation leans ``/`` -- and multiplied by this to land in data space.
    Hard-coding the inversion instead would silently mirror every ellipse and
    pie the day a card drew the matrix the other way up.
    """
    try:
        lo, hi = ax.get_ylim()
        return -1.0 if hi < lo else 1.0
    except Exception:
        return 1.0


def _visible(ix: np.ndarray, iy: np.ndarray, triangle: str, diag: bool) -> np.ndarray:
    """Cells this ``type``/``diag`` combination draws.

    ``upper`` is the wedge above the diagonal *as printed* -- column greater
    than row, with row 0 at the top.  The transform's own ``triangle`` option
    names the halves in array terms instead, which is why the two are kept
    apart: this one follows the picture.
    """
    if triangle == "upper":
        keep = ix > iy if not diag else ix >= iy
    elif triangle == "lower":
        keep = ix < iy if not diag else ix <= iy
    else:
        keep = np.ones(ix.shape, dtype=bool) if diag else ix != iy
    return keep


#: The four corners of a unit square, centred on the origin.  Broadcast against
#: the cell centres to build every square glyph at once.
_CORNERS = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

#: The same square, closed and at cell size: the outline of one cell.
_CELL_LOOP = 0.5 * np.vstack([_CORNERS, _CORNERS[:1]])

#: Square glyphs are a polygon per cell and nothing else, so they skip the
#: patch objects entirely -- see :func:`_glyph_vertices`.
_POLYGON_GLYPHS = ("square", "color", "shade")

#: Glyphs that cover their whole cell whatever rho is.  Everything else shrinks
#: with |rho|, which is what makes an edge worth drawing -- see
#: :func:`_outline_color`.
_FILLED_GLYPHS = ("color", "shade")


def _glyph_vertices(kind: str, cx, cy, rho, scale: float):
    """``(cells, 4, 2)`` corner array for the square-shaped glyphs.

    ``PatchCollection`` would build a ``Polygon`` object per cell and then throw
    it away for its path; at 100 variables that is 10,000 objects and about
    0.14 s, for four corners each of which is one add.  ``PolyCollection`` takes
    the corners directly, which is the same picture two orders of magnitude
    cheaper -- and ``color`` is the method large matrices are actually drawn
    with, so this is the case that scales.
    """
    magnitude = np.clip(np.abs(np.nan_to_num(rho, nan=0.0)), 0.0, 1.0)
    # `color` and `shade` fill the whole cell; `square` carries |rho| in its
    # area, which is the side length by its square root.
    half = 0.5 * scale * (np.sqrt(magnitude) if kind == "square" else 1.0)
    centres = np.stack([np.asarray(cx, dtype=float), np.asarray(cy, dtype=float)], axis=1)
    return centres[:, None, :] + np.asarray(half).reshape(-1, 1, 1) * _CORNERS[None, :, :]


def _glyph_patches(kind: str, cx, cy, rho, scale: float, sign: float,
                   ellipse_scale: float = 1.4):
    """One matplotlib patch per drawn cell, in data coordinates.

    Only the round glyphs come through here; the square ones are built as bare
    vertices by :func:`_glyph_vertices`.

    A cell is 1.0 data unit on both axes and the panel is square in print, so
    data units *are* the physical shape: a circle patch comes out round and a
    square comes out square with no aspect bookkeeping.  That squareness is
    guaranteed by the geometry solver (``panel = n * cell`` on both sides),
    not assumed here.
    """
    from matplotlib.patches import Circle, Ellipse, Wedge

    patches = []
    magnitude = np.clip(np.abs(np.nan_to_num(rho, nan=0.0)), 0.0, 1.0)
    for x, y, r, m in zip(cx, cy, rho, magnitude):
        if kind == "circle":
            patches.append(Circle((x, y), radius=0.5 * scale * np.sqrt(m)))
        elif kind == "ellipse":
            # The pictogram of a scatter plot with this correlation: round at
            # rho = 0, collapsed onto the diagonal at |rho| = 1, leaning `/`
            # when positive.
            #
            # `ellipse.scale` is why the family fills its cell.  Rotated 45
            # degrees, an ellipse of these two axes has a bounding box of
            # `k / sqrt(2)` on a side whatever rho is -- the shape changes,
            # the box does not -- so at k = glyph.scale the widest one covers
            # 0.64 of the cell while a circle covers 0.9.  sqrt(2) is the
            # factor that makes the two agree, which is where 1.4 comes from.
            k = scale * ellipse_scale
            r = 0.0 if not np.isfinite(r) else float(np.clip(r, -1.0, 1.0))
            width = k * np.sqrt(max(1.0 + r, 0.0)) / np.sqrt(2.0)
            # |rho| = 1 collapses the minor axis to zero, which fills no
            # pixels at all -- the diagonal of an `ellipse` matrix would read
            # as missing rather than as perfect. R draws the degenerate case
            # as a line, so the minor axis gets a hairline floor.
            height = max(k * np.sqrt(max(1.0 - r, 0.0)) / np.sqrt(2.0), 0.02 * k)
            patches.append(
                Ellipse((x, y), width=width, height=height, angle=sign * 45.0)
            )
        elif kind == "pie":
            # Clockwise from twelve o'clock, as read off the page.
            span = 360.0 * m
            start, end = 90.0 - span, 90.0
            if sign < 0:
                start, end = -end, -start
            patches.append(Wedge((x, y), 0.5 * scale, start, end))
        else:
            raise ValueError(f"corrplot has no glyph {kind!r}; expected one of {GLYPHS}")
    return patches


def _number_strings(rho: np.ndarray, digits: int, as_percent: bool) -> list[str]:
    if as_percent:
        return [
            "" if not np.isfinite(v) else "{:.0f}%".format(round(float(v) * 100.0))
            for v in rho
        ]
    return [
        "" if not np.isfinite(v) else "{:.{d}f}".format(float(v), d=digits) for v in rho
    ]


def _text_points(labels, cell_pt: float, fraction: float = 0.9) -> float:
    """Coefficient size, back-solved from the cell so the widest one fits.

    ``0.6`` is the usual advance-width-per-point rule of thumb.  It runs a few
    percent wide on real digit strings, which errs toward text that fits --
    the only direction worth erring in when the alternative is a number
    spilling into its neighbour.
    """
    widest = max((len(label) for label in labels if label), default=1)
    return max(cell_pt * fraction / (0.6 * widest), 1.0)


def draw_corrplot(ax, **kwargs):
    """Draw one correlation matrix.  Returns the glyph collection."""
    from matplotlib.collections import LineCollection, PatchCollection, PolyCollection
    from matplotlib.patches import Rectangle

    df = kwargs.pop("__df__", None)
    if df is None or not hasattr(df, "columns"):
        raise ValueError(
            "method: corrplot draws the long table emitted by the `correlation` "
            "transform and was handed no table. The layer needs "
            "`transform: [{correlation: {...}}]`."
        )

    # `coordinates` is optional here: the three columns below are part of the
    # `correlation` transform's published output, so restating them adds a
    # place to get it wrong without adding a choice.  A layer that does state
    # them wins, which is what makes a renamed column recoverable.
    columns = set(df.columns)
    missing = [
        column
        for axis, column in (("x", "x_index"), ("y", "y_index"), ("c", "rho"))
        if axis not in kwargs and column not in columns
    ]
    if missing:
        raise ValueError(
            "method: corrplot needs the `correlation` transform's output columns "
            "({}); this table has: {}.".format(
                ", ".join(missing),
                ", ".join(sorted(str(c) for c in columns if not str(c).startswith("__"))),
            )
        )

    ix = np.asarray(kwargs.pop("x", None) if "x" in kwargs else df["x_index"], dtype=float)
    iy = np.asarray(kwargs.pop("y", None) if "y" in kwargs else df["y_index"], dtype=float)
    rho = np.asarray(kwargs.pop("c", None) if "c" in kwargs else df["rho"], dtype=float)
    counts = np.asarray(df["n"], dtype=float) if "n" in columns else np.full(rho.shape, np.nan)

    n = _matrix_size(df, ix, iy)
    cell_pt = _cell_points(ax, n)
    sign = _screen_sign(ax)

    blocks = kwargs.pop("__corr_blocks__", None)
    for name in _CONFIG_TIME:
        kwargs.pop(name, None)

    kind = str(_pop(kwargs, "method", "circle")).strip().lower()
    if kind not in GLYPHS:
        raise ValueError(
            "corrplot method must be one of {}; got {!r}".format(", ".join(GLYPHS), kind)
        )
    triangle = str(_pop(kwargs, "type", "full")).strip().lower()
    if triangle not in _TRIANGLES:
        raise ValueError(
            "corrplot type must be one of {}; got {!r}".format(", ".join(_TRIANGLES), triangle)
        )
    diag = _as_bool(kwargs.pop("diag", True), True)
    scale = float(_pop(kwargs, "glyph.scale", 0.9))
    # Card-owned, and only `ellipse` reads it: the ellipse family is drawn from
    # two axes rather than from one radius, so `glyph.scale` alone leaves it
    # visibly smaller than the other glyphs at the same setting.  See
    # `_glyph_patches`.
    ellipse_scale = float(_pop(kwargs, "ellipse.scale", 1.4))
    # Also card-owned: R's `outline` is on or off, and this is how thin.
    outline_lwd = float(_pop(kwargs, "outline.lwd", 0.3))
    outline = kwargs.pop("outline", False)
    grid_color = kwargs.pop("addgrid.col", None)

    coef_color = kwargs.pop("addCoef.col", None)
    digits = int(_pop(kwargs, "number.digits", 2))
    number_cex = float(_pop(kwargs, "number.cex", 1.0))
    as_percent = _as_bool(kwargs.pop("addCoefasPercent", False), False)

    sig_level = kwargs.pop("sig.level", None)
    insig = str(_pop(kwargs, "insig", "pch")).strip().lower()
    pch = str(_pop(kwargs, "pch", "x"))
    pch_cex = float(_pop(kwargs, "pch.cex", 1.5))
    pch_color = _pop(kwargs, "pch.col", "#21171A")

    na_label = kwargs.pop("na.label", None)
    na_color = _pop(kwargs, "na.label.col", "#8A8A8A")

    rect_color = _pop(kwargs, "rect.col", "#21171A")
    rect_lwd = float(_pop(kwargs, "rect.lwd", 0.7))

    cmap = kwargs.pop("cmap", None)
    norm = kwargs.pop("norm", None)
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    alpha = kwargs.pop("alpha", None)
    zorder = float(_pop(kwargs, "zorder", 30))
    # Card-owned, like `zorder`, and not one of R's formals.  The card turns it
    # off: a cell in the first or last row sits *on* the axes edge, so a clipped
    # glyph loses the half of itself that is outside -- a circle drawn as a
    # half-moon reads as a different mark, not as a mark near the border -- and
    # the outermost gridlines come out at half the linewidth of every line
    # inside them.  Kept as a key rather than a constant so a card that does
    # frame its panel can ask for the clip back.
    clip = _as_bool(_pop(kwargs, "clip", True), True)
    kwargs.pop("label", None)

    keep = _visible(ix, iy, triangle, diag)
    if not np.any(keep):
        return None
    ix, iy, rho, counts = ix[keep], iy[keep], rho[keep], counts[keep]

    finite = np.isfinite(rho)
    insignificant = np.zeros(rho.shape, dtype=bool)
    if sig_level is not None:
        from .correlation_runtime import pearson_pvalues

        pvalues = pearson_pvalues(rho, counts)
        insignificant = np.isfinite(pvalues) & (pvalues > float(sig_level))

    drawn = finite.copy()
    if sig_level is not None and insig == "blank":
        drawn &= ~insignificant

    artists = None
    if kind == "number":
        # R's method="number" is the coefficient *instead of* a glyph, so the
        # colour lives on the text and nothing is filled.
        coef_color = coef_color if coef_color is not None else "__cmap__"
    else:
        if kind in _POLYGON_GLYPHS:
            artists = PolyCollection(
                _glyph_vertices(kind, ix[drawn], iy[drawn], rho[drawn], scale),
                closed=True,
            )
        else:
            artists = PatchCollection(
                _glyph_patches(
                    kind, ix[drawn], iy[drawn], rho[drawn], scale, sign,
                    ellipse_scale=ellipse_scale,
                ),
                match_original=False,
            )
        artists.set_array(rho[drawn])
        if cmap is not None:
            artists.set_cmap(cmap)
        if norm is not None:
            artists.set_norm(norm)
        else:
            artists.set_clim(
                -1.0 if vmin is None else float(vmin),
                1.0 if vmax is None else float(vmax),
            )
        edge = _outline_color(outline, kind, rho[drawn], cmap, norm, vmin, vmax)
        if edge is None:
            artists.set_linewidth(0.0)
        else:
            artists.set_edgecolor(edge)
            artists.set_linewidth(outline_lwd)
        if alpha is not None:
            artists.set_alpha(float(alpha))
        artists.set_zorder(zorder)
        artists.set_clip_on(clip)
        ax.add_collection(artists)

    if kind == "pie":
        # R draws the outline circle first and the wedge into it, and without
        # it the method loses its zero: a rho of 0.01 is a sliver too thin to
        # see, which reads as a cell that was never drawn rather than as a
        # cell with nothing in it.
        from matplotlib.patches import Circle

        rim = PatchCollection(
            [Circle((x, y), radius=0.5 * scale) for x, y in zip(ix[drawn], iy[drawn])],
            match_original=False,
        )
        rim.set_facecolor("none")
        rim.set_edgecolor(grid_color or "#C2C2C2")
        rim.set_linewidth(0.3)
        rim.set_zorder(zorder - 1)
        rim.set_clip_on(clip)
        ax.add_collection(rim)

    if kind == "shade":
        # The sign is already in the colour; the hatch is what survives a
        # greyscale print, which is the whole point of shade.
        hatch = LineCollection(
            [
                [
                    (x - 0.5 * scale, y - sign * 0.5 * scale * np.sign(r)),
                    (x + 0.5 * scale, y + sign * 0.5 * scale * np.sign(r)),
                ]
                for x, y, r in zip(ix[drawn], iy[drawn], rho[drawn])
                if r != 0.0
            ],
            colors="#FFFFFF",
            linewidths=0.4,
            zorder=zorder + 0.5,
        )
        hatch.set_clip_on(clip)
        ax.add_collection(hatch)

    if grid_color:
        # One closed loop per cell rather than four separate edges: same
        # picture, a quarter of the paths, and at 100 variables that is 40,000
        # of them.  The outline has to be per cell rather than n+1 ruled lines
        # because a `triangle` selection leaves holes for the ruling to cross.
        grid = LineCollection(
            np.stack([ix, iy], axis=1)[:, None, :] + _CELL_LOOP[None, :, :],
            colors=grid_color, linewidths=0.3, zorder=zorder - 10,
        )
        grid.set_clip_on(clip)
        ax.add_collection(grid)

    if coef_color is not None:
        labels = _number_strings(rho, digits, as_percent)
        size = _text_points([l for l, ok in zip(labels, drawn) if ok], cell_pt) * number_cex
        for x, y, label, value, ok in zip(ix, iy, labels, rho, drawn):
            if not ok or not label:
                continue
            color = coef_color
            if color == "__cmap__":
                color = _value_color(value, cmap, norm, vmin, vmax)
            ax.text(
                x, y, label,
                ha="center", va="center", fontsize=size, color=color,
                fontfamily="sans", clip_on=clip, zorder=zorder + 20,
            )

    if sig_level is not None and insig == "pch" and np.any(insignificant):
        for x, y in zip(ix[insignificant], iy[insignificant]):
            ax.text(
                x, y, pch,
                ha="center", va="center",
                fontsize=cell_pt * 0.5 * pch_cex / 1.5,
                color=pch_color, fontfamily="sans", clip_on=clip, zorder=zorder + 25,
            )

    if na_label:
        for x, y in zip(ix[~finite], iy[~finite]):
            ax.text(
                x, y, str(na_label),
                ha="center", va="center", fontsize=cell_pt * 0.45,
                color=na_color, fontfamily="sans", clip_on=clip, zorder=zorder + 25,
            )

    if blocks:
        for start, end in blocks:
            side = float(end) - float(start) + 1.0
            ax.add_patch(
                Rectangle(
                    (float(start) - 0.5, float(start) - 0.5), side, side,
                    fill=False, edgecolor=rect_color, linewidth=rect_lwd,
                    clip_on=clip, zorder=zorder + 30,
                )
            )

    kwargs.pop("levels", None)
    kwargs.pop("mode", None)
    if kwargs:
        # Anything still here is a key nobody read.  Saying so beats a figure
        # that quietly ignored half its style block -- the failure mode a
        # macro over `scatter` had, where `type: upper` was accepted and
        # discarded.
        raise TypeError(
            "corrplot does not take {}. It takes R's corrplot formals: method, "
            "type, diag, glyph.scale, ellipse.scale, outline, outline.lwd, "
            "addgrid.col, addCoef.col, "
            "number.digits, number.cex, addCoefasPercent, sig.level, insig, "
            "pch, pch.cex, pch.col, na.label, na.label.col, rect.col, rect.lwd "
            "(order / hclust.method / addrect / tl.* are resolved before the "
            "figure is built).".format(", ".join(sorted(map(str, kwargs))))
        )
    return artists


def _outline_color(outline, kind, rho, cmap, norm, vmin, vmax):
    """The glyph edge: ``None`` for no edge, one colour, or one per cell.

    R's ``outline`` is ``TRUE`` or a colour.  ``sign`` is this card's addition
    and the reason it exists is the small glyphs: ``circle`` and ``square``
    carry |rho| in their area, so a weak cell is a pale mark shrunk to nothing
    in the middle of an empty cell.  Drawing its edge in the *end* of the scale
    its sign points at -- ``rho = 1`` for a positive cell, ``rho = -1`` for a
    negative one -- keeps that cell readable as positive or negative when its
    fill has gone almost white.

    ``sign`` is not applied to ``color`` and ``shade``: those cover the whole
    cell whatever rho is, so their edge is not a mark on a glyph but a second
    grid line, drawn in two loud colours next to the one ``addgrid.col``
    already draws.  Nothing shrinks there, so there is nothing to rescue.
    """
    if outline is None or outline is False:
        return None
    if isinstance(outline, str) and outline.strip().lower() == "sign":
        if kind in _FILLED_GLYPHS:
            return None
        from matplotlib.colors import to_rgba

        ends = np.asarray([
            to_rgba(_value_color(-1.0 if vmin is None else vmin, cmap, norm, vmin, vmax)),
            to_rgba(_value_color(1.0 if vmax is None else vmax, cmap, norm, vmin, vmax)),
        ])
        return ends[(np.asarray(rho, dtype=float) >= 0.0).astype(int)]
    return outline if isinstance(outline, str) else "#21171A"


def _value_color(value, cmap, norm, vmin, vmax):
    """The colour a cell would have been filled with, for ``method: number``."""
    import matplotlib as mpl

    try:
        mappable = mpl.cm.ScalarMappable(
            norm=norm or mpl.colors.Normalize(
                -1.0 if vmin is None else float(vmin),
                1.0 if vmax is None else float(vmax),
            ),
            cmap=cmap,
        )
        return mappable.to_rgba(float(value))
    except Exception:
        return "#21171A"

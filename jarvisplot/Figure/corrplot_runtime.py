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
from matplotlib.transforms import blended_transform_factory

__all__ = ["GLYPHS", "draw_corrplot"]

#: R's ``method`` argument.  ``shade`` is ``color`` plus a sign hatch here,
#: which is what it amounts to once the underlying colour scale is diverging.
GLYPHS = ("circle", "square", "ellipse", "color", "shade", "pie", "number")

_TRIANGLES = ("full", "upper", "lower")

#: Formals resolved at config time.  They arrive because the user wrote them
#: in the layer, and are dropped here rather than forwarded to matplotlib.
_CONFIG_TIME = ("order", "hclust.method", "addrect", "col", "tl.pos", "tl.cex",
                "tl.col", "tl.srt", "tl.offset", "cl.cex", "edge.numbers.cex")


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


def _cell_points(ax, n: int, diamond: bool = False) -> float:
    """One cell's side in points, measured off the axes.

    Read from the axes rather than from the style card because the card states
    an *intended* cell size and the geometry solver is allowed to shrink it to
    hold the page.  Measuring is the only way the coefficient text follows
    that shrink.

    The two layouts are measured on different sides.  The square panel is n
    cells across and n down.  The diamond is ``n-1`` *rows* tall and half that
    wide, and its cell stands on its point, so the side of that cell is the
    row pitch over sqrt(2) -- measuring the width would report half a pitch and
    shrink every coefficient to nothing.
    """
    try:
        box = ax.get_window_extent()
        dpi = float(ax.figure.dpi)
        if diamond:
            pitch = abs(box.height) / dpi * 72.0 / max(n - 1, 1)
            return pitch / np.sqrt(2.0)
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

#: `_CORNERS` turned 45 degrees -- the corners of one cell of the diamond
#: layout, where a cell is a square standing on its point with both diagonals
#: one unit long.  Multiplied by the same `half` the square layout uses, so
#: `glyph.scale` keeps meaning the same thing: the glyph at |rho| = 1 as a
#: fraction of the cell.
_DIAMOND_CORNERS = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
_DIAMOND_LOOP = 0.5 * np.vstack([_DIAMOND_CORNERS, _DIAMOND_CORNERS[:1]])

#: A round glyph has to fit the cell's *inscribed* circle, and a diamond of
#: unit diagonals inscribes one of diameter 1/sqrt(2) where the square cell
#: inscribes one of diameter 1.
_DIAMOND_ROUND = 1.0 / np.sqrt(2.0)

#: The cell grid's line weight.  Shared with the rules between the names,
#: because a divider that is *nearly* the cell divider reads as a mistake.
_GRID_LWD = 0.3

#: Square glyphs are a polygon per cell and nothing else, so they skip the
#: patch objects entirely -- see :func:`_glyph_vertices`.
_POLYGON_GLYPHS = ("square", "color", "shade")

#: Glyphs that cover their whole cell whatever rho is.  Everything else shrinks
#: with |rho|, which is what makes an edge worth drawing -- see
#: :func:`_outline_color`.
_FILLED_GLYPHS = ("color", "shade")


def _notch_vertices(n: int, side: str = "left"):
    """The half cells that make the matrix's near edge straight.

    A variable's V has its vertex at ``(0, k)``, and the two cells meeting
    there touch the edge at a point, not along it -- so the boundary beside the
    names comes out as a row of notches.  The notch at ``k`` is the triangle
    ``(0, k-1/2) (1/2, k) (0, k+1/2)``: filling it with that variable's own
    tint carries the band from the name into the matrix without a gap, and
    turns the sawtooth into the straight edge the rules can close against.

    Returned for every variable; the caller picks the tinted ones.
    """
    k = np.arange(int(n), dtype=float)
    flip = -1.0 if side == "right" else 1.0
    zero = np.zeros_like(k)
    return np.stack(
        [
            np.stack([zero, k - 0.5], axis=1),
            np.stack([flip * 0.5 + zero, k], axis=1),
            np.stack([zero, k + 0.5], axis=1),
        ],
        axis=1,
    )


def _shaded_variables(n: int, blocks=None) -> np.ndarray:
    """Which variables carry the tint: alternate blocks, or alternate names.

    A block is a run of adjacent positions cut from the clustering tree, so
    tinting every other one makes the shading say where one group ends and the
    next begins -- which is what the boxes were for.
    """
    shaded = np.zeros(int(n), dtype=bool)
    if blocks:
        for position, (start, end) in enumerate(blocks):
            if position % 2 == 0:
                shaded[int(start):int(end) + 1] = True
        return shaded
    shaded[1::2] = True
    return shaded


def _edge_numbers(ax, n: int, side: str, size: float, color, clip: bool,
                  zorder: float, gap: float = 0.06, pitch_pt: float = 12.0):
    """The variable's position, repeated where each arm of its V ends.

    A variable's two arms run to the two outer edges of the triangle, and at
    forty names the far end of an arm is a long way from the name it belongs
    to.  R's matrix has the same problem and answers it by printing the names
    on both axes; this layout has only one band of names, so it answers it the
    way the architectural diagram does -- with the row number, repeated at both
    ends, which is why the names themselves are numbered too.

    Variable ``k``'s arms are the lines ``u + v = k`` (ending on the upper
    edge) and ``v - u = k`` (ending on the lower one), so the two ends are
    ``(k/2, k/2)`` and ``((n-1-k)/2, (n-1+k)/2)`` -- half a cell further out,
    and turned to lie along the edge they sit on.

    Each number is set **across** the edge rather than along it, so it reads
    out of the matrix like the leader on a dimension: the edge is a diagonal,
    and text lying on it is text the reader has to tilt their head to take in.

    ``gap`` is the clearance between the matrix's edge and the *near end of the
    text*, in cells -- what an author would actually want to set, and the only
    form of it that survives the text being turned.  The rest is measured: the
    edge under one of these is a cell *face*, not a corner, half a diagonal in
    at ``0.5 / sqrt(2)``; the text sticks out from its own centre by half its
    width, now that its width lies along the normal; and the whole offset is
    struck along the diagonal, which is the last factor of ``sqrt(2)``.
    """
    digits = len(str(int(n)))
    from .corr_layout import measure_text_mm

    text_pt = measure_text_mm("0" * digits, size_pt=size, family="sans")[0] / 25.4 * 72.0
    half = (text_pt / 2.0) / max(float(pitch_pt), 1e-9)
    reach = (0.5 * _DIAMOND_ROUND + float(gap) + half) * _DIAMOND_ROUND
    flip = -1.0 if side == "right" else 1.0
    for k in range(int(n)):
        label = str(k + 1).zfill(digits)
        for u, v, rotation in (
            (k / 2.0 + reach, k / 2.0 - reach, 45.0 * flip),
            ((n - 1 - k) / 2.0 + reach, (n - 1 + k) / 2.0 + reach, -45.0 * flip),
        ):
            ax.text(
                flip * u, v, label,
                ha="center", va="center", rotation=rotation,
                rotation_mode="anchor", fontsize=size, color=color,
                fontfamily="sans", clip_on=clip, zorder=zorder,
            )


def _label_numbers(ax, n: int, side: str, offset_mm: float, size: float,
                   color, clip: bool, zorder: float) -> None:
    """The same numbers again, as their own column beside the names.

    Not part of the tick label.  A tick label is one string in one colour, and
    these have to be set in the smaller, lighter face the numbers wear out on
    the diagonal -- otherwise the eye reads ``01`` as part of the name instead
    of as the tag that ties the name to the two arms it owns.

    Anchored on the outer edge of the band, so they line up in a column: they
    are zero-padded to a common width, which makes the inner edge line up too.
    ``offset_mm`` is the solver's ``number_pad_mm`` -- how far that outer edge
    stands from the panel.
    """
    target = ax.ax if hasattr(ax, "ax") else ax
    try:
        box = target.get_position()
        panel_mm = box.width * float(target.figure.get_size_inches()[0]) * 25.4
    except Exception:
        return
    if panel_mm <= 0:
        return
    reach = float(offset_mm) / panel_mm
    x = -reach if side == "left" else 1.0 + reach
    ha = "left" if side == "left" else "right"
    transform = blended_transform_factory(target.transAxes, target.transData)
    digits = len(str(int(n)))
    for k in range(int(n)):
        target.text(
            x, k, str(k + 1).zfill(digits),
            ha=ha, va="center", fontsize=size, color=color,
            fontfamily="sans", transform=transform, clip_on=clip, zorder=zorder,
        )


def _insignificant_marks(ix, iy, corners, scale: float) -> np.ndarray:
    """The cell's own two diagonals, as ``(2m, 2, 2)`` segments.

    R marks a pair its data cannot resolve with ``pch``, a plotted symbol.
    Drawn as a *character* it is a glyph sized in points, so it neither fills
    the cell it belongs to nor shrinks with it -- at fifty variables an ``x``
    set at half the cell rides over its neighbours, and at ten it sits in the
    middle of an empty square.  Two lines between opposite corners are the same
    mark made out of the cell itself: always exactly as big as what it marks.

    It reads as a cross on the square cell and as a plus on the diamond, which
    is the same construction seen from 45 degrees -- the corners are the
    corners either way.
    """
    corners = 0.5 * float(scale) * corners
    centres = np.stack([ix, iy], axis=1)[:, None, :]
    return np.concatenate([
        centres + corners[None, (0, 2), :],
        centres + corners[None, (1, 3), :],
    ])


def _label_rules(ax, n: int, span, color, lw: float, clip: bool,
                 zorder: float) -> None:
    """The cell grid, carried on between the names.

    A row of the label column *is* a row of the matrix -- the name at ``v = k``
    and the V rooted under it are the same variable -- so the line that divides
    two cells is the line that divides two names.  Drawn in the grid's own
    colour and weight rather than a nearby pair, because a divider that only
    nearly matches reads as a second, different thing.

    Below the tick labels, like the band: an axis draws its labels at zorder
    2.5, and a rule over one of them is a strike-through.
    """
    from matplotlib.collections import LineCollection

    target = ax.ax if hasattr(ax, "ax") else ax
    x0, x1 = span
    rules = LineCollection(
        [[(x0, k + 0.5), (x1, k + 0.5)] for k in range(int(n) - 1)],
        colors=color, linewidths=lw, zorder=0.6,
        transform=blended_transform_factory(target.transAxes, target.transData),
    )
    rules.set_clip_on(clip)
    target.add_collection(rules, autolim=False)


def _label_span(ax, side: str, column_mm: float):
    """The label column, in axes fractions, or ``None``.

    The column is a millimetre number the geometry solver measured, and only
    the solver could have: it is how wide the *names* came out.  Converting it
    here, against the panel's own width, is what lets the band and the edge
    rules stop exactly where the text does instead of bleeding to the paper.
    """
    try:
        box = ax.get_position()
        figure_w_in = float(ax.figure.get_size_inches()[0])
    except Exception:
        return None
    panel_mm = box.width * figure_w_in * 25.4
    if panel_mm <= 0 or column_mm <= 0:
        return None
    reach = float(column_mm) / panel_mm
    return (-reach, 0.0) if side == "left" else (1.0, 1.0 + reach)


def _edge_rules(ax, n: int, column_u: float, side: str, color, lw: float,
                zorder: float, clip: bool) -> None:
    """The outline that closes the figure, over the names and the matrix both.

    Turned 45 degrees the matrix has no top or bottom edge of its own: its
    boundary is two diagonals meeting the label column at a point, so the first
    and last names sit against nothing.  Each rule is that nothing -- out
    across the names, then down the matrix's own diagonal to the apex -- so
    what is enclosed is the names *and* the triangle, which is the figure.

    Half a row outside the first and last variable rather than through them: at
    ``v = 0`` a rule is struck through the first name, and it is the band the
    names occupy that wants closing.  Half a row out is also exactly where the
    filled notches end (see :func:`_notch_vertices`), so the horizontal meets
    the diagonal on the straight edge they make together.

    Drawn in data coordinates, in the panel's own axes, so the two halves of
    each rule are one polyline rather than two artists to keep in step.
    """
    target = ax.ax if hasattr(ax, "ax") else ax
    flip = -1.0 if side == "right" else 1.0
    apex = n / 2.0
    for outline in (
        [(-column_u, -0.5), (0.0, -0.5), (apex, apex - 0.5)],
        [(-column_u, n - 0.5), (0.0, n - 0.5), (apex, n - 0.5 - apex)],
    ):
        target.plot(
            [flip * u for u, _ in outline], [v for _, v in outline],
            color=color, linewidth=lw, solid_capstyle="butt",
            solid_joinstyle="miter", clip_on=clip, zorder=zorder,
        )


def _stripe_the_labels(ax, n: int, shaded, span, color, clip: bool) -> None:
    """Continue the band under the name it belongs to.

    Drawn under the tick labels rather than over them.  An axes' tick labels
    sit at the axis artist's zorder of 2.5, so a band at the glyphs' zorder
    would be a grey rectangle where every other name should be.
    """
    for k in np.flatnonzero(shaded):
        ax.axhspan(
            k - 0.5, k + 0.5, xmin=span[0], xmax=span[1],
            facecolor=color, edgecolor="none", linewidth=0.0,
            zorder=0.5, clip_on=clip,
        )


def _diamond_uv(ix, iy, side: str = "left"):
    """Index positions mapped to the rotated frame.

    ``u`` is half the distance between the two variables and ``v`` their
    midpoint, which is the whole of what makes this layout work: variable ``k``
    is in every cell with ``v - |u| == k`` or ``v + |u| == k``, so its pairs
    are one connected V rooted at ``(0, k)`` -- exactly where its name is
    printed.  See ``corr_layout_diamond``.

    ``side: right`` negates ``u`` and nothing else.
    """
    a = np.asarray(ix, dtype=float)
    b = np.asarray(iy, dtype=float)
    u = np.abs(a - b) / 2.0
    return (-u if side == "right" else u), (a + b) / 2.0


def _diamond_block(lo: float, hi: float, side: str = "left"):
    """The ``addrect`` box for a block of adjacent variables, rotated.

    A run ``[lo, hi]`` of positions is a *triangle* here, not a rectangle: its
    cells are exactly those with ``v - u`` and ``v + u`` both inside the run,
    so the boundary is the two diagonals of the run plus the panel's own edge
    at ``u = 0``.  Half a cell out on each diagonal, so it encloses the cells
    rather than cutting through their centres.
    """
    lo, hi = float(lo) - 0.5, float(hi) + 0.5
    corners = [(0.0, lo), ((hi - lo) / 2.0, (lo + hi) / 2.0), (0.0, hi)]
    return [(-u if side == "right" else u, v) for u, v in corners]


def _glyph_vertices(kind: str, cx, cy, rho, scale: float, corners=None):
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
    corners = _CORNERS if corners is None else corners
    return centres[:, None, :] + np.asarray(half).reshape(-1, 1, 1) * corners[None, :, :]


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
    from matplotlib.patches import Polygon, Rectangle

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

    # Written by `_prebuild_one` from the card's Contract, the same way the
    # addrect blocks are: the card chose the layout, at config time, and the
    # renderer should not have to resolve a card to find out which.
    diamond = str(kwargs.pop("__corr_layout__", "square")).strip().lower() == "diamond"
    # The width the solver gave the label column, in mm.  Only the stripes need
    # it, and only they could not work it out for themselves.
    label_column_mm = float(kwargs.pop("__corr_label_mm__", 0.0) or 0.0)
    # Where the number column is anchored, and how big the names are set: both
    # settled by the solve, both needed here because the numbers beside the
    # names are drawn as text of their own rather than as part of a tick label.
    number_column_mm = float(kwargs.pop("__corr_number_mm__", 0.0) or 0.0)
    number_pt = float(kwargs.pop("__corr_number_pt__", 4.2) or 4.2)
    side = str(_pop(kwargs, "side", "left")).strip().lower()
    if side not in ("left", "right"):
        raise ValueError(
            "corrplot side must be left or right; got {!r}. It mirrors the "
            "diamond layout: the names go on that side.".format(side)
        )

    n = _matrix_size(df, ix, iy)
    cell_pt = _cell_points(ax, n, diamond=diamond)
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
    # A round glyph fills its cell's inscribed circle, and the rotated cell
    # inscribes a smaller one than the square cell of the same diagonal.
    round_scale = scale * (_DIAMOND_ROUND if diamond else 1.0)
    # Diamond only.  See the stripe block below for why it is not decoration.
    stripe = _pop(kwargs, "stripe", "alternate")
    stripe = "none" if stripe is False else str(stripe).strip().lower()
    if stripe not in ("alternate", "none"):
        raise ValueError(
            "corrplot stripe must be alternate or none; got {!r}.".format(stripe)
        )
    stripe_color = _pop(kwargs, "stripe.col", "#EFEFEF")
    # Diamond only.  See `_edge_rules`: the rotated matrix has no top or bottom
    # edge of its own, so the first and last variable sit against nothing.
    edge_lwd = float(_pop(kwargs, "edge.lwd", 0.7))
    edge_color = _pop(kwargs, "edge.col", "#21171A")
    edge_numbers = _as_bool(_pop(kwargs, "edge.numbers", False), False)
    edge_numbers_gap = float(_pop(kwargs, "edge.numbers.gap", 0.21))
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
    # `pch` is R's symbol number and this draws exactly one symbol, the cross
    # through the cell's corners.  Named rather than numbered, and refused
    # rather than quietly substituted: a figure that marked its unresolved
    # pairs with something other than what the card asked for would be read as
    # if it had.
    pch = str(_pop(kwargs, "pch", "cross")).strip().lower()
    if pch not in ("cross", "x"):
        raise ValueError(
            "corrplot pch draws the cell's own diagonals and takes cross (or "
            "x, the same mark); got {!r}. insig: blank leaves the cell "
            "empty instead.".format(pch)
        )
    pch_cex = float(_pop(kwargs, "pch.cex", 1.0))
    pch_lwd = float(_pop(kwargs, "pch.lwd", 0.6))
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

    if diamond and triangle == "full":
        raise ValueError(
            "corrplot type: full on the diamond layout. Turning the matrix 45 "
            "degrees is how it stops drawing every pair twice, so the card "
            "keeps one triangle by construction: type is upper or lower, and "
            "on a symmetric matrix the two are the same picture."
        )

    keep = _visible(ix, iy, triangle, diag)
    if not np.any(keep):
        return None
    ix, iy, rho, counts = ix[keep], iy[keep], rho[keep], counts[keep]

    index_x, index_y = ix, iy
    if diamond:
        # The map the whole card is: how far apart the two variables are, and
        # their midpoint.  |i - j| rather than i - j so the triangle opens the
        # same way whichever half the table carries, and the sign of u is the
        # only difference between a left-hand and a right-hand figure.
        ix, iy = _diamond_uv(index_x, index_y, side)

    finite = np.isfinite(rho)
    insignificant = np.zeros(rho.shape, dtype=bool)
    if sig_level is not None:
        from .correlation_runtime import pearson_pvalues

        pvalues = pearson_pvalues(rho, counts)
        insignificant = np.isfinite(pvalues) & (pvalues > float(sig_level))

    drawn = finite.copy()
    if sig_level is not None and insig == "blank":
        drawn &= ~insignificant

    label_span = _label_span(ax, side, label_column_mm) if diamond else None

    if diamond and edge_lwd > 0.0 and label_span is not None:
        # The column is a width in axes fractions; the outline is drawn in cell
        # units, and the panel is `u_span` cells wide.
        column_u = -label_span[0] * (n / 2.0) if side == "left" else (
            (label_span[1] - 1.0) * (n / 2.0)
        )
        _edge_rules(ax, n, column_u, side, edge_color, edge_lwd, zorder + 40, clip)
    if diamond and grid_color and label_span is not None:
        _label_rules(ax, n, label_span, grid_color, _GRID_LWD, clip, zorder)
    if diamond and edge_numbers:
        _edge_numbers(
            ax, n, side, number_pt, edge_color, clip, zorder + 40,
            gap=edge_numbers_gap, pitch_pt=cell_pt / _DIAMOND_ROUND,
        )
        if number_column_mm > 0.0:
            _label_numbers(
                ax, n, side, number_column_mm, number_pt,
                edge_color, clip, zorder + 40,
            )

    if diamond and stripe == "alternate" and np.any(drawn):
        # Every other variable's cells, tinted.  Not decoration: a variable's
        # pairs run away from its name as a V, and at 40 names a reader
        # following one of those arms has nothing to hold on to.  The band is
        # what carries the name across, which is why it also runs back under
        # the label itself.
        #
        # *Which* variables are tinted is the interesting part.  With clusters
        # in hand it is alternate **blocks**, so the shading is what tells one
        # group from the next -- the boxes an `addrect` would draw are then
        # redundant, and the card turns them off.  Without clusters it falls
        # back to alternate names, which is the same device doing the same job
        # at a coarser grain.
        #
        # The union of those variables' cells rather than one band each, so a
        # cell where two bands cross is filled once and the tint stays even.
        shaded = _shaded_variables(n, blocks)
        band = shaded[index_x.astype(int)] | shaded[index_y.astype(int)]
        band &= drawn
        if np.any(band):
            cells = PolyCollection(
                np.stack([ix[band], iy[band]], axis=1)[:, None, :]
                + 0.5 * _DIAMOND_CORNERS[None, :, :],
                closed=True, facecolors=stripe_color, edgecolors="none",
                zorder=zorder - 20,
            )
            cells.set_clip_on(clip)
            ax.add_collection(cells)
        # The half cells beside the names, so the band reaches the matrix and
        # the edge it makes is straight enough for the rules to close against.
        notches = PolyCollection(
            _notch_vertices(n, side)[shaded], closed=True,
            facecolors=stripe_color, edgecolors="none", zorder=zorder - 20,
        )
        notches.set_clip_on(clip)
        ax.add_collection(notches)
        if label_span is not None:
            _stripe_the_labels(ax, n, shaded, label_span, stripe_color, clip)

    artists = None
    if kind == "number":
        # R's method="number" is the coefficient *instead of* a glyph, so the
        # colour lives on the text and nothing is filled.
        coef_color = coef_color if coef_color is not None else "__cmap__"
    else:
        if kind in _POLYGON_GLYPHS:
            artists = PolyCollection(
                _glyph_vertices(
                    kind, ix[drawn], iy[drawn], rho[drawn], scale,
                    corners=_DIAMOND_CORNERS if diamond else _CORNERS,
                ),
                closed=True,
            )
        else:
            artists = PatchCollection(
                _glyph_patches(
                    kind, ix[drawn], iy[drawn], rho[drawn], round_scale, sign,
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
            [Circle((x, y), radius=0.5 * round_scale) for x, y in zip(ix[drawn], iy[drawn])],
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
            np.stack([ix, iy], axis=1)[:, None, :]
            + (_DIAMOND_LOOP if diamond else _CELL_LOOP)[None, :, :],
            colors=grid_color, linewidths=_GRID_LWD, zorder=zorder - 10,
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
        marks = LineCollection(
            _insignificant_marks(
                ix[insignificant], iy[insignificant],
                _DIAMOND_CORNERS if diamond else _CORNERS, pch_cex,
            ),
            colors=pch_color, linewidths=pch_lwd, zorder=zorder + 25,
        )
        marks.set_clip_on(clip)
        ax.add_collection(marks)

    if na_label:
        for x, y in zip(ix[~finite], iy[~finite]):
            ax.text(
                x, y, str(na_label),
                ha="center", va="center", fontsize=cell_pt * 0.45,
                color=na_color, fontfamily="sans", clip_on=clip, zorder=zorder + 25,
            )

    # On the diamond the clusters are told by the shading (see the stripe
    # block above), so a box around each of them would be the same fact drawn
    # twice, in the heaviest ink on the figure.  `addrect` still decides *what*
    # the blocks are; it just stops drawing them.
    if blocks and not (diamond and stripe != "none"):
        for start, end in blocks:
            if diamond:
                ax.add_patch(
                    Polygon(
                        _diamond_block(start, end, side), closed=True,
                        fill=False, edgecolor=rect_color, linewidth=rect_lwd,
                        clip_on=clip, zorder=zorder + 30,
                    )
                )
                continue
            span = float(end) - float(start) + 1.0
            ax.add_patch(
                Rectangle(
                    (float(start) - 0.5, float(start) - 0.5), span, span,
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

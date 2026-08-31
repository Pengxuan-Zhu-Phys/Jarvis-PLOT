#!/usr/bin/env python3

"""Figure geometry for the rotated (diamond) correlation matrix.

``corr_layout`` solves the square card: one half of the matrix is redundant,
the names are rotated tick labels crammed under a 4.2 mm cell, and the figure
is bounded by the page *width*.  This module solves the other arrangement --
keep one triangle, turn it 45 degrees -- from the map::

    u = (j - i) / 2      how far apart the two variables are   ->  horizontal
    v = (i + j) / 2      the midpoint of the two               ->  vertical

Everything about this card is a consequence of that map:

* A cell's four neighbours land at ``(±1/2, ±1/2)``, so a cell is a square
  turned 45 degrees whose two diagonals are one unit each.
* Variable ``k`` appears in ``(k, j)`` for ``j > k``, running down-right from
  ``(0, k)``, and in ``(i, k)`` for ``i < k``, running up-right to the same
  point.  Its pairs are one connected **V** rooted at ``u = 0, v = k`` -- which
  is exactly where its name goes.  That is the whole reason for the card: on
  the square one a variable's pairs are a row and a column meeting at a corner.
* Because item ``k`` sits at ``v = k``, the names stay ordinary y tick labels.
  The ordering, the tick machinery and ``tl.*`` learn nothing new.

So the anchor moves.  The square card is anchored on the **cell** and bounded
by the page width; this one is anchored on the **row pitch** -- the label line
height -- and bounded by the page *height*, since the panel comes out about
twice as tall as it is wide.  Its floor is a different quantity too: below
``pitch_min`` it is the names that collide, not the marks that stop being
readable.  Two different anchors, two different bounds, two different floors,
which is why this is its own solver rather than a branch in the other one.

Nothing here imports the rest of the Figure runtime, and the measuring
primitives are shared with ``corr_layout`` rather than copied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .corr_layout import MM_PER_INCH, _geom, max_label_mm, measure_text_mm

__all__ = [
    "DiamondGeometry",
    "diamond_extent",
    "solve_diamond_geometry",
]

#: The two sides the label column may take.  ``right`` is the same map with
#: ``u`` negated; it must never become a second code path.
SIDES = ("left", "right")


def diamond_extent(n: int, *, diag: bool = False) -> tuple[float, float, float, float]:
    """``(u_lo, u_hi, v_lo, v_hi)`` of the panel, in cell units.

    Cell centres run ``u ∈ [1/2, (n-1)/2]`` and ``v ∈ [1/2, n-3/2]`` off the
    diagonal; the panel is those plus the half cell that hangs off each end.
    The v range is then exactly ``[0, n-1]``, which is where labels ``0`` and
    ``n-1`` sit -- the first and last name line up with the top and bottom of
    the diamond, as they must.

    With ``diag`` the self-pairs come in at ``u = 0`` and push both ranges out
    by half a cell.
    """
    n = max(int(n), 2)
    if diag:
        return -0.5, n / 2.0, -0.5, n - 0.5
    return 0.0, n / 2.0, 0.0, float(n - 1)


@dataclass(frozen=True)
class DiamondGeometry:
    """A solved diamond layout, in the units each consumer wants.

    ``pitch_mm`` is the distance from one variable to the next down the label
    column, and it is also the cell's diagonal -- the two are the same number
    by construction, which is what keeps the lattice square.

    ``label_block_mm`` is the band the column was given; ``label_text_mm`` is
    how much of it the names actually occupy.  The difference is page margin,
    and the stripes stop at the text rather than bleeding into it.

    ``name_pad_mm`` and ``number_pad_mm`` are how far from the panel's edge the
    two columns of the label band are anchored -- the names' outer edge and the
    numbers' outer edge.  Both are anchored on their *outer* edge, which is the
    whole point of the band: a number column is only a column if every number
    starts at the same place, and a ragged inner edge is the price.  See the
    ``edge_numbers`` branch below.

    The two ``colorbar_*`` numbers are carried out because the bar's axis label
    is placed in axes fractions of a bar this solve sized: the caller cannot
    turn ``colorbar.label_gap`` into a fraction without knowing the width it is
    a fraction of.
    """

    figsize: tuple[float, float]
    panel_rect: tuple[float, float, float, float]
    colorbar_rect: tuple[float, float, float, float]
    logo_rect: tuple[float, float, float, float]
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    pitch_mm: float
    width_mm: float
    height_mm: float
    label_block_mm: float
    label_text_mm: float
    name_pad_mm: float
    number_pad_mm: float
    number_size_pt: float
    colorbar_w_mm: float
    colorbar_label_gap_mm: float
    side: str
    clamped: bool
    n_cells: int
    notes: list[str] = field(default_factory=list)

    @property
    def panel_w_mm(self) -> float:
        return self.panel_rect[2] * self.figsize[0] * MM_PER_INCH

    @property
    def panel_h_mm(self) -> float:
        return self.panel_rect[3] * self.figsize[1] * MM_PER_INCH

    def as_frame(self) -> dict:
        """The ``Frame`` fragment this geometry implies.

        The same three keys the square solver writes, so the caller merges it
        the same way and neither card is a special case downstream.
        """
        return {
            "figure": {"figsize": list(self.figsize)},
            "axes": {
                "axcorr": {"rect": list(self.panel_rect)},
                "axccorr": {"rect": list(self.colorbar_rect), "xticks": []},
                "axlogo": {"rect": list(self.logo_rect)},
            },
        }


def solve_diamond_geometry(
    n: int,
    *,
    geometry: Mapping[str, Any],
    labels: Sequence[Any] = (),
    colorbar_labels: Sequence[Any] = (),
    colorbar_title: Any = "",
    side: str = "right",
    diag: bool = False,
    edge_numbers: bool = False,
    label_size_pt: float = 6.0,
    label_pad_mm: float = 0.0,
    number_size_pt: float = 4.2,
    colorbar_label_size_pt: float = 5.5,
    colorbar_tick_pad_mm: float = 0.6,
    colorbar_title_size_pt: float = 7.0,
    family: str = "sans",
) -> DiamondGeometry:
    """Solve the figure size and axes rects for an ``n`` variable diamond.

    ``labels`` are printed horizontally in a column of their own, so what they
    cost is their width -- once, in a band that has to exist anyway -- rather
    than a rotated tick label under every cell.
    """
    n = max(int(n), 2)
    side = str(side or "left").strip().lower()
    if side not in SIDES:
        raise ValueError(
            "corrplot side must be one of {}; got {!r}. It mirrors the figure: "
            "the names go on that side and the colorbar on the other.".format(
                ", ".join(SIDES), side
            )
        )

    pitch = _geom(geometry, "pitch", default=4.2)
    pitch_min = _geom(geometry, "pitch_min", default=2.6)
    max_width = _geom(geometry, "max_width", default=170.0)
    max_height = _geom(geometry, "max_height", default=247.0)
    m_left = _geom(geometry, "margin", "left", default=11.0)
    m_right = _geom(geometry, "margin", "right", default=11.0)
    m_top = _geom(geometry, "margin", "top", default=4.0)
    m_bottom = _geom(geometry, "margin", "bottom", default=11.0)
    margin = (geometry.get("margin") or {}) if isinstance(geometry, Mapping) else {}
    fit = bool(margin.get("fit", False))
    fit_slack = _geom(geometry, "margin", "slack", default=1.0)
    label_gap = _geom(geometry, "labels", "gap", default=1.4)
    number_gap = _geom(geometry, "labels", "number_gap", default=1.0)
    # `labels.width: null` means measured.  An author states it when two
    # figures have to line up down the page and the names differ.
    authored_column = _geom(geometry, "labels", "width", default=0.0)
    cb_w = _geom(geometry, "colorbar", "width", default=2.6)
    # A fraction of the panel's height, not a length in mm: the bar is a key to
    # the panel beside it, so it should keep the same visual weight against a
    # 13-variable matrix and a 40-variable one.
    cb_len = _geom(geometry, "colorbar", "length", default=0.22)
    # ... and how far its outer edge stands in from the edge of the paper.
    cb_offset = _geom(geometry, "colorbar", "offset", default=10.0)
    # ... and how far its own axis label stands off its inward edge.
    cb_label_gap = _geom(geometry, "colorbar", "label_gap", default=1.5)
    logo_w = _geom(geometry, "logo", "width", default=5.0)
    logo_h = _geom(geometry, "logo", "height", default=5.0)
    logo_off = _geom(geometry, "logo", "offset", default=0.838)
    logo_corner = str(
        (geometry.get("logo") or {}).get("corner", "far")
        if isinstance(geometry, Mapping) else "far"
    ).strip().lower()

    notes: list[str] = []

    # --- the two bands beside the panel ------------------------------------
    #
    # The names are the whole point of this layout, so the column that holds
    # them is measured, not guessed: `pad + widest name + gap`, which is where
    # the text actually ends.  `labels.width` overrides it for an author who
    # wants two figures to line up; `margin.fit` is what lets it grow past the
    # page margin, exactly as on the square card.
    name_width = max_label_mm(labels, size_pt=label_size_pt, family=family)
    # With `edge.numbers` the band is two columns, not one: the position, then
    # the name.  They are separate because they are set differently -- the
    # number in the smaller, lighter face the same number wears out on the
    # diagonal -- and a tick label is one string in one colour.  Splitting them
    # is also what lets the numbers line up: each column is anchored on its own
    # outer edge, so the numbers make a column and the names hang off it.
    number_width = (
        max_label_mm(
            [str(k + 1).zfill(len(str(n))) for k in range(n)],
            size_pt=number_size_pt, family=family,
        )
        if edge_numbers and labels
        else 0.0
    )
    number_block = (number_width + number_gap) if number_width else 0.0
    name_pad = label_pad_mm + name_width
    number_pad = name_pad + number_block
    label_need = number_pad + label_gap
    if authored_column:
        label_block = authored_column
        if label_need > label_block:
            notes.append(
                "labels.width is {:.1f} mm and the names need {:.1f} mm, so "
                "they print past it.".format(label_block, label_need)
            )
    elif fit:
        label_block = max(m_left if side == "left" else m_right, label_need + fit_slack)
    else:
        label_block = m_left if side == "left" else m_right
        if label_need > label_block:
            notes.append(
                "the names need {:.1f} mm and the margin is {:.1f} mm, so they "
                "print past it. margin.fit: true grows it instead.".format(
                    label_need, label_block
                )
            )

    # The colorbar does not stand beside the panel on this card.  Turning the
    # matrix leaves a right-angled triangle of empty page above it -- as big as
    # the matrix itself -- and a bar parked outside the panel would pay for a
    # band of width twice over, once for itself and once for the hole.  So it
    # goes in the hole, the way a ternary card puts its bar in the corner
    # beside the triangle.  The figure is then the names plus the panel, and
    # nothing else.
    cblab = max_label_mm(colorbar_labels, size_pt=colorbar_label_size_pt, family=family)
    cbtitle_w, cbtitle_h = (
        measure_text_mm(colorbar_title, size_pt=colorbar_title_size_pt, family=family)
        if colorbar_title
        else (0.0, 0.0)
    )

    # The square card floors its bottom-left margin at the badge, because there
    # a cell of the matrix reaches that corner.  Here nothing does: the last
    # variable's only cell sits at u = 1/2 and the whole lower corner of the
    # panel's box is empty by construction, so the mark can sit in a 4 mm
    # margin without anything to collide with.  The `axlogo` offset is still
    # the house one; it is only the *floor* that does not apply.
    #
    # Top and bottom are then the same band.  Nothing prints above the matrix
    # and nothing below it, so an asymmetric pair would read as a figure
    # sitting crookedly on the page rather than as a title band doing
    # something.
    bottom_block = m_bottom
    top_block = m_top
    far_block = m_right if side == "left" else m_left

    left_block, right_block = (
        (label_block, far_block) if side == "left" else (far_block, label_block)
    )

    # --- the pitch, and the two page bounds --------------------------------
    u_lo, u_hi, v_lo, v_hi = diamond_extent(n, diag=diag)
    u_span, v_span = u_hi - u_lo, v_hi - v_lo

    # Height binds first on any matrix worth drawing this way -- the panel is
    # about twice as tall as it is wide -- but both are checked, because a
    # figure that fits the page vertically and runs off it sideways is just as
    # useless.  The pitch is what gives; the font never does, for the same
    # reason as on the square card.
    clamped = False
    room_h = (max_height - bottom_block - top_block) / v_span
    room_w = (max_width - left_block - right_block) / u_span
    bound, room = ("height", room_h) if room_h <= room_w else ("width", room_w)
    if room < pitch:
        clamped = True
        if room < pitch_min:
            over = (
                bottom_block + v_span * pitch_min + top_block - max_height
                if bound == "height"
                else left_block + u_span * pitch_min + right_block - max_width
            )
            notes.append(
                "{} variables do not fit {:.0f} mm of {} at the minimum row "
                "pitch ({:.2f} mm needed, {:.2f} mm floor). The figure keeps "
                "the floor and overruns the page by {:.1f} mm. Below the floor "
                "the names collide, so the only real fixes are fewer variables "
                "or a taller page.".format(
                    n,
                    max_height if bound == "height" else max_width,
                    bound, room, pitch_min, over,
                )
            )
            pitch = pitch_min
        else:
            notes.append(
                "{} pinned to {:.0f} mm: row pitch {:.2f} -> {:.2f} mm ({} "
                "variables). Font sizes are unchanged.".format(
                    bound,
                    max_height if bound == "height" else max_width,
                    pitch, room, n,
                )
            )
            pitch = room

    panel_w = u_span * pitch
    panel_h = v_span * pitch
    width = left_block + panel_w + right_block
    height = bottom_block + panel_h + top_block

    # --- fractions, in the order add_axes wants them -----------------------
    panel_rect = (
        left_block / width,
        bottom_block / height,
        panel_w / width,
        panel_h / height,
    )

    # The bar, inside the empty triangle.  Two things fix it: its top is the
    # panel's top -- a key that starts where the thing it explains starts reads
    # as belonging to it, and a bar floating below the top edge reads as having
    # drifted -- and its outer edge stands `colorbar.offset` in from the paper.
    bar_len = cb_len * panel_h
    bar_x = (
        width - cb_offset - cb_w if side == "left" else cb_offset
    )
    bar_top = bottom_block + panel_h

    # The panel's upper boundary is the line `v = u - 1/2` (the first
    # variable's cells), so the headroom above the matrix grows with distance
    # from the label column.  Under the bar's *near* edge that headroom is what
    # the bar has to fit in -- less the numbers, when they are printed along
    # that diagonal.
    def _u_of(x_mm: float) -> float:
        return (
            (x_mm - left_block) if side == "left" else (left_block + panel_w - x_mm)
        ) / pitch

    near = bar_x if side == "left" else bar_x + cb_w
    inward = -1.0 if side == "left" else 1.0
    # Half a cell is the diagonal's own overhang.  The edge numbers add their
    # offset plus half their height, and a perpendicular offset costs sqrt(2)
    # as much measured straight down; 1.2 covers the card's own setting.
    clearance = (1.2 if edge_numbers else 0.5) * pitch

    def _head(reach_mm: float, allow: float) -> float:
        """Page above the matrix, ``reach_mm`` inside the bar's near edge."""
        return (_u_of(near + inward * reach_mm) - 0.5) * pitch - allow

    def _fits(reach_mm: float, half_extent_mm) -> float:
        """How long the bar may be with something standing ``reach_mm`` inside.

        ``half_extent_mm`` is half of what that something occupies *along* the
        bar, or ``None`` for text running its whole length.  A wall clears the
        diagonal only when the bar's own bottom does; a single line at the
        bar's mid height only needs half the bar plus half itself to.
        """
        head = _head(reach_mm, clearance)
        return head if half_extent_mm is None else 2.0 * (head - half_extent_mm)

    # The numbers print on the bar's left and the label on its right, whichever
    # way the figure is mirrored -- so *which* of them stands between the bar
    # and the diagonal depends on the mirror, and the two are obstructions of
    # different shapes.  The numbers are a column as tall as the bar: a wall.
    # The label is one line at its mid height, turned on its side, so what it
    # reaches inward by is its line height and what it takes up along the bar
    # is its text width.
    numbers = (colorbar_tick_pad_mm + cblab, None) if cblab else None
    title = (cb_label_gap + cbtitle_h, cbtitle_w / 2.0) if cbtitle_h else None
    inward_text, outward_text = (
        (numbers, title) if side == "left" else (title, numbers)
    )
    # These three are *measurements*, not limits.  The bar's top, length and
    # offset are the author's three numbers and the solve realises them; what
    # it does here is say when they are tight, not quietly pick others.  An
    # earlier version shortened the bar to fit its own clearance allowance and
    # so drew a figure that did not have the length it was asked for -- which
    # is a worse fault than a tight one, because nothing on the page says so.
    cells = _head(0.0, 0.0)
    if bar_len > cells:
        notes.append(
            "the colorbar is {:.1f} mm and only {:.1f} mm of page stands above "
            "the matrix under it, so it is drawn across the cells. A smaller "
            "colorbar.length, or a smaller colorbar.offset -- which moves the "
            "bar out toward the paper, where the triangle is taller -- clears "
            "them.".format(bar_len, max(cells, 0.0))
        )
    elif inward_text is not None and bar_len > _fits(*inward_text):
        notes.append(
            "the colorbar clears the matrix, but the text on its inward side "
            "comes within a cell of the diagonal. A smaller colorbar.offset "
            "moves the bar out toward the paper, where the triangle is taller."
        )
    # ... and whatever is on the other side has the page margin to live in.
    if outward_text is not None and outward_text[0] > cb_offset:
        notes.append(
            "the text on the outer side of the colorbar needs {:.1f} mm and "
            "only {:.1f} mm of paper stands there, so it prints past the page "
            "edge. A larger colorbar.offset moves the bar in.".format(
                outward_text[0], cb_offset
            )
        )
    colorbar_rect = (
        bar_x / width,
        (bar_top - bar_len) / height,
        cb_w / width,
        bar_len / height,
    )
    # The badge keeps the house offset -- 0.838 mm off two edges -- but not
    # necessarily the *left* two.  The corner beside the names is the one place
    # on this card where something already is: the last variable's row sits on
    # the panel's bottom edge and the lower run of numbers ends just inside it.
    # The far corner is empty by construction, because the last variable's only
    # cell is at u = 1/2.  `logo.corner: bottom-left` puts it back.
    logo_x = logo_off
    if (logo_corner == "far" and side == "left") or logo_corner == "bottom-right":
        logo_x = width - logo_off - logo_w
    logo_rect = (
        logo_x / width,
        logo_off / height,
        logo_w / width,
        logo_h / height,
    )

    # `u` is negated for a right-hand column, so the triangle opens the other
    # way and the names still sit against the panel's own edge.  This is the
    # entire difference between the two sides.
    xlim = (u_lo, u_hi) if side == "left" else (-u_hi, -u_lo)
    # Item 0 at the top, as on the square card and for the same reason: a
    # matrix is read from its first row down.
    ylim = (v_hi, v_lo)

    return DiamondGeometry(
        figsize=(width / MM_PER_INCH, height / MM_PER_INCH),
        panel_rect=panel_rect,
        colorbar_rect=colorbar_rect,
        logo_rect=logo_rect,
        xlim=xlim,
        ylim=ylim,
        pitch_mm=pitch,
        width_mm=width,
        height_mm=height,
        label_block_mm=label_block,
        label_text_mm=min(label_need, label_block),
        name_pad_mm=name_pad,
        number_pad_mm=number_pad,
        number_size_pt=number_size_pt,
        colorbar_w_mm=cb_w,
        colorbar_label_gap_mm=cb_label_gap,
        side=side,
        clamped=clamped,
        n_cells=n,
        notes=notes,
    )

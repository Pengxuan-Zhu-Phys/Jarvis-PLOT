#!/usr/bin/env python3

"""Figure geometry for the correlation matrix, solved in millimetres.

Every other card in this repository fixes ``figsize`` and writes its axes as
fractions of it.  That works because those figures carry a roughly constant
amount of information: one panel, two axes, a handful of curves.  A
correlation matrix does not.  Its information content is ``n**2`` cells and
``n`` variable names, so a fixed frame either wastes half the page at ``n=6``
or crushes the cells to noise at ``n=40``.

So this module inverts the relationship.  What stays fixed is the thing
legibility actually depends on -- **the cell size and the font size** -- and
the figure size is whatever those imply::

    panel = n * cell
    W = margin.left + panel + colorbar.gap + colorbar.width + margin.right
    H = margin.bottom + panel + margin.top

The margins are *authored*, not derived from the labels.  An earlier version
sized the left and bottom bands from the widest variable name, which made the
figure's proportions a property of the dataset: the same card printed twice,
with two different sets of names, came out two different shapes.  Fixed
margins cost the occasional clipped name -- the labels are still measured with
``TextToPath``, and an overrun is reported as a note -- and buy a card that
comes out the same shape every time.

The one hard constraint is the page: a figure wider than the text block is
useless however legible its cells are.  When the derived width exceeds
``max_width`` the solver pins the width and shrinks the cell to fit -- but
never the font, because a font that changes with ``n`` is what makes two
figures in one paper look like they came from different tools.

Nothing here imports the rest of the Figure runtime.  It takes numbers and
returns numbers, which is what makes it checkable without drawing anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "MM_PER_INCH",
    "CorrGeometry",
    "measure_text_mm",
    "max_label_mm",
    "solve_corr_geometry",
]

MM_PER_INCH = 25.4
_PT_PER_INCH = 72.0

#: Measuring a string means rasterising its glyph outlines, and the same
#: handful of labels is measured on every solve.  Keyed by (text, pt, family).
_WIDTH_CACHE: dict[tuple[str, float, str], tuple[float, float]] = {}


def _text_to_path():
    from matplotlib.textpath import TextToPath

    global _T2P
    try:
        return _T2P
    except NameError:
        _T2P = TextToPath()
        return _T2P


def measure_text_mm(text: str, *, size_pt: float, family: str = "sans") -> tuple[float, float]:
    """Width and height of ``text`` in millimetres, with no figure involved.

    Returns ``(0.0, 0.0)`` for anything that cannot be measured rather than
    raising: a label the font cannot render should cost the layout nothing,
    not abort the figure.
    """
    text = "" if text is None else str(text)
    if not text:
        return 0.0, 0.0
    key = (text, float(size_pt), str(family))
    hit = _WIDTH_CACHE.get(key)
    if hit is not None:
        return hit

    try:
        from matplotlib.font_manager import FontProperties

        prop = FontProperties(family=family, size=float(size_pt))
        # `$\rho$` measured as literal characters is three glyphs wide instead
        # of one, and a variable name written as mathtext is a normal thing to
        # want on this figure.  Matplotlib decides the same way.
        w_pt, h_pt, _descent = _text_to_path().get_text_width_height_descent(
            text, prop, ismath=text.count("$") >= 2
        )
        out = (w_pt / _PT_PER_INCH * MM_PER_INCH, h_pt / _PT_PER_INCH * MM_PER_INCH)
    except Exception:
        out = (0.0, 0.0)
    _WIDTH_CACHE[key] = out
    return out


def max_label_mm(labels: Iterable[Any], *, size_pt: float, family: str = "sans") -> float:
    """Width of the widest label, in millimetres."""
    widths = [measure_text_mm(label, size_pt=size_pt, family=family)[0] for label in labels or ()]
    return max(widths) if widths else 0.0


@dataclass(frozen=True)
class CorrGeometry:
    """A solved layout, in the units each consumer wants.

    ``figsize`` is inches because that is what ``plt.figure`` takes; the rects
    are figure fractions because that is what ``add_axes`` takes; ``cell_mm``
    is millimetres because that is what the glyph sizing needs.
    """

    figsize: tuple[float, float]
    panel_rect: tuple[float, float, float, float]
    colorbar_rect: tuple[float, float, float, float]
    logo_rect: tuple[float, float, float, float]
    cell_mm: float
    width_mm: float
    height_mm: float
    clamped: bool
    n_cells: int
    notes: list[str] = field(default_factory=list)

    @property
    def panel_mm(self) -> float:
        return self.cell_mm * self.n_cells

    def as_frame(self) -> dict:
        """The ``Frame`` fragment this geometry implies.

        Shaped so the caller can merge it straight into a figure's frame:
        the same three keys a hand-written card would have set, except that
        here they are derived rather than authored.
        """
        return {
            "figure": {"figsize": list(self.figsize)},
            "axes": {
                "axcorr": {"rect": list(self.panel_rect)},
                "axccorr": {"rect": list(self.colorbar_rect), "xticks": []},
                "axlogo": {"rect": list(self.logo_rect)},
            },
        }


def _geom(geometry: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    node: Any = geometry
    for key in keys:
        if not isinstance(node, Mapping):
            return float(default)
        node = node.get(key)
    try:
        return float(node)
    except (TypeError, ValueError):
        return float(default)


def solve_corr_geometry(
    n: int,
    *,
    geometry: Mapping[str, Any],
    x_labels: Sequence[Any] = (),
    y_labels: Sequence[Any] = (),
    colorbar_labels: Sequence[Any] = (),
    colorbar_title: Any = "",
    label_size_pt: float = 5.5,
    label_pad_mm: float = 0.0,
    colorbar_label_size_pt: float = 5.5,
    colorbar_title_size_pt: float = 7.0,
    family: str = "sans",
) -> CorrGeometry:
    """Solve the figure size and axes rects for an ``n x n`` matrix.

    ``x_labels`` are rotated 90 degrees, so what they cost the layout is their
    *width* hanging below the panel -- the same measurement as the y labels,
    not their line height.

    ``label_pad_mm`` is the gap the card's tick settings put between the panel
    and the first glyph of a name, so ``pad + widest label`` is where the text
    actually ends.  It only matters under ``margin.fit``; without it the fitted
    margin would end exactly where the text starts to be drawn.
    """
    n = max(int(n), 1)

    cell = _geom(geometry, "cell", default=4.2)
    cell_min = _geom(geometry, "cell_min", default=1.6)
    max_width = _geom(geometry, "max_width", default=170.0)
    m_left = _geom(geometry, "margin", "left", default=11.0)
    m_right = _geom(geometry, "margin", "right", default=11.0)
    m_top = _geom(geometry, "margin", "top", default=4.0)
    m_bottom = _geom(geometry, "margin", "bottom", default=11.0)
    # `margin.fit` is what to do when a name does not fit the margin it was
    # given: off, print past it and say so; on, grow the margin to hold it.
    # Off by default because the fixed margin is what makes the card come out
    # the same shape on any dataset -- growing is opting back into a figure
    # whose proportions depend on how long the variable names happen to be.
    fit = bool(((geometry.get("margin") or {}) if isinstance(geometry, Mapping) else {})
               .get("fit", False))
    # Breathing room on a band that had to grow.  Fitting to where the text
    # ends puts the last glyph exactly on the paper's edge, which reads as a
    # name that was nearly cut off rather than as one that fits.  Added only
    # when the margin grows: the card's own number is the card's business.
    fit_slack = _geom(geometry, "margin", "slack", default=1.0)
    cb_w = _geom(geometry, "colorbar", "width", default=2.6)
    cb_gap = _geom(geometry, "colorbar", "gap", default=0.42)
    # The bar is deliberately shorter than the panel it explains: it is a key,
    # not a second data axis, and one run to the full height reads as another
    # column of the matrix.  Taken off both ends, so the bar stays centred.
    cb_inset = _geom(geometry, "colorbar", "inset", default=5.0)
    logo_w = _geom(geometry, "logo", "width", default=5.0)
    logo_h = _geom(geometry, "logo", "height", default=5.0)
    # The badge sits 0.838 mm (0.0838 cm) off the left and bottom edges on
    # every card in the repository.  It is a house rule about the mark, not a
    # page margin, which is why it is its own key and not a margin: a margin is
    # what the *content* is inset by, and the badge sits outside it on purpose.
    logo_off = _geom(geometry, "logo", "offset", default=0.838)

    ylab = max_label_mm(y_labels, size_pt=label_size_pt, family=family)
    xlab = max_label_mm(x_labels, size_pt=label_size_pt, family=family)
    cblab = max_label_mm(colorbar_labels, size_pt=colorbar_label_size_pt, family=family)
    # The bar's axis label is rotated upright, so what it costs the width is
    # its line height, not its length.
    cbtitle = (
        measure_text_mm(colorbar_title, size_pt=colorbar_title_size_pt, family=family)[1]
        if colorbar_title
        else 0.0
    )

    notes: list[str] = []

    # The panel sits the same distance from the left edge as from the bottom
    # edge, because a correlation matrix is a square whose two label bands hold
    # the same names: an unequal corner reads as a mistake in the figure rather
    # than as a consequence of the text.  Both numbers come from the card; the
    # only thing enforced here is that the badge still fits underneath, since a
    # margin narrower than the mark would print the two on top of each other.
    corner_floor = logo_off + logo_h
    # What each band would have to be to hold its text: where the text ends,
    # which is the pad before the first glyph plus the widest name.  Measured
    # with the tick font at the tick size, so this is the real edge of the
    # printed text and not an estimate of it.
    name_need = label_pad_mm + max(ylab, xlab) + fit_slack
    bar_need = (cblab + 1.2 if cblab else 0.0) + (cbtitle + 0.8 if cbtitle else 0.0)

    if fit:
        bar_need += fit_slack
        # One corner for both bands, as below, and it holds the *widest* name
        # of either -- so fitting can never trade a clipped y name for a
        # clipped x one.  It only ever grows: the card's margin is the floor.
        corner = max(m_left, m_bottom, name_need, corner_floor)
        left_block = bottom_block = corner
        if corner > max(m_left, m_bottom, corner_floor):
            notes.append(
                "margin.fit: the names need {:.1f} mm plus {:.1f} mm of slack, "
                "so the corner grew from {:.1f} to {:.1f} mm.".format(
                    name_need - fit_slack, fit_slack, max(m_left, m_bottom), corner)
            )
        right_margin = max(m_right, bar_need)
        if right_margin > m_right:
            notes.append(
                "margin.fit: the colorbar labels need {:.1f} mm plus {:.1f} mm "
                "of slack, so the right margin grew from {:.1f} mm.".format(
                    bar_need - fit_slack, fit_slack, m_right)
            )
    else:
        # The panel sits the same distance from the left edge as from the
        # bottom edge, because a correlation matrix is a square whose two label
        # bands hold the same names: an unequal corner reads as a mistake in
        # the figure rather than as a consequence of the text.  Both numbers
        # come from the card; the only thing enforced here is that the badge
        # still fits underneath, since a margin narrower than the mark would
        # print the two on top of each other.
        left_block = max(m_left, corner_floor)
        # x names print *below* the panel, as they do on every other figure in
        # this repository -- corrplot's own default is the top, but a reader
        # who has to look up to find the column name of a cell is being asked
        # to read the figure backwards from every other one in the same paper.
        bottom_block = max(m_bottom, corner_floor)
        right_margin = m_right
        # Measured, reported, and then not acted on.
        for band, need, have in (
            ("y", label_pad_mm + ylab, left_block),
            ("x", label_pad_mm + xlab, bottom_block),
            ("colorbar", bar_need, m_right),  # no slack added in this branch
        ):
            if need > have:
                notes.append(
                    "the {} labels need {:.1f} mm and the margin is {:.1f} mm, "
                    "so they print past it. margin.fit: true grows it "
                    "instead.".format(band, need, have)
                )

    right_block = cb_gap + cb_w + right_margin
    top_block = m_top

    clamped = False
    width = left_block + n * cell + right_block
    if width > max_width:
        clamped = True
        room = max_width - left_block - right_block
        solved = room / n
        if solved < cell_min:
            notes.append(
                "{} variables do not fit {:.0f} mm at the minimum cell size "
                "({:.2f} mm needed, {:.2f} mm floor). The figure keeps the floor "
                "and overruns the page by {:.1f} mm. Below the floor a cell is "
                "no longer a readable mark, so the only real fixes are fewer "
                "variables, a narrower margin, or a figure printed wider than "
                "the text block.".format(
                    n, max_width, solved, cell_min,
                    left_block + n * cell_min + right_block - max_width,
                )
            )
            solved = cell_min
        else:
            notes.append(
                "width pinned to {:.0f} mm: cell {:.2f} -> {:.2f} mm "
                "({} variables). Font sizes are unchanged.".format(
                    max_width, cell, solved, n
                )
            )
        cell = solved
        width = left_block + n * cell + right_block

    panel = n * cell
    height = bottom_block + panel + top_block

    # Fractions, in the order add_axes wants them.
    panel_rect = (
        left_block / width,
        bottom_block / height,
        panel / width,
        panel / height,
    )
    # A small matrix can be shorter than twice the inset, where a centred bar
    # would come out with no height at all.  Give it back a third of the panel
    # rather than letting the figure fail to draw.
    inset = cb_inset if panel - 2 * cb_inset >= 0.3 * panel else 0.35 * panel
    colorbar_rect = (
        (left_block + panel + cb_gap) / width,
        (bottom_block + inset) / height,
        cb_w / width,
        (panel - 2 * inset) / height,
    )
    # The logo is the one thing on the page whose physical size must not move
    # with n: it is a mark, not data.  Fractions are computed from the solved
    # figure so the printed square stays the size the card asked for.
    logo_rect = (
        logo_off / width,
        logo_off / height,
        logo_w / width,
        logo_h / height,
    )

    return CorrGeometry(
        figsize=(width / MM_PER_INCH, height / MM_PER_INCH),
        panel_rect=panel_rect,
        colorbar_rect=colorbar_rect,
        logo_rect=logo_rect,
        cell_mm=cell,
        width_mm=width,
        height_mm=height,
        clamped=clamped,
        notes=notes,
        n_cells=n,
    )

# Correlation Matrix (`type: correlation_matrix`, `method: corrplot`)

Status: implemented
Last updated: 2026-08-29
Design authority: this document, plus `jarvisplot/cards/corrplot/matrix.json` (the card
is the defaults; this is the contract).

## 1. What this is

A port of R's [`corrplot`](https://github.com/taiyun/corrplot) into Jarvis-PLOT. It is the
only figure in the repository whose **size is derived rather than declared**, and the only
drawing method that is not a wrapper over a matplotlib primitive.

Both of those follow from the same fact: a correlation matrix carries `n²` cells and `n`
variable names, so a fixed frame either wastes half a page at `n = 6` or crushes the cells
to noise at `n = 40`. Every other card fixes `figsize` and writes its axes as fractions of
it. This one fixes **the cell size and the font size** — the two things legibility actually
depends on — and solves the figure from them.

## 2. Where each decision is made

This is the part worth reading before anything else, because it is the one thing that is
not obvious from either R or the rest of Jarvis-PLOT.

Jarvis-PLOT resolves tick labels **as the figure loads**, before any layer draws. The
labels of a correlation matrix are variable names sitting at integer positions, and
`x_index` counts positions in the transform's `columns`. So anything that decides *which
variable sits where* must be settled at config time, or the labels end up naming cells they
do not belong to — silently, with nothing downstream able to detect it.

| Stage | Owner | Decides | Formals |
|---|---|---|---|
| Config | `core_runtime.prebuild_correlations` | which variables, in what order, at what figure size | `order`, `hclust.method`, `addrect`, `tl.pos`, `col` |
| Render | `Figure/corrplot_runtime.draw_corrplot` | what one cell looks like | everything else |

The ordering pass writes its result into the transform's `columns:` — that is the whole of
applying it. Nothing reorders a matrix after its labels exist.

Ordering costs a data read **only when it is data-dependent**: `original` and `alphabet`
are answered from column names alone. When it is, the pass runs the data block's transforms
*up to* the correlation step, so rows a filter was going to remove never reach the
clustering.

## 3. Geometry

```
panel = n × cell
W = margin.left + panel + colorbar.gap + colorbar.width + margin.right
H = margin.bottom + panel + margin.top
```

The card authors those margins, in millimetres:

| key | mm | what sits in it |
|---|---|---|
| `margin.fit` | `true` | grow a margin that cannot hold its text — see below |
| `margin.slack` | 1.0 | breathing room added to a band that grew |
| `margin.left` | 11.0 | the y variable names, and the badge |
| `margin.bottom` | 11.0 | the x variable names, rotated |
| `margin.top` | 4.0 | the title |
| `margin.right` | 11.0 | the colorbar's numbers and its label |
| `colorbar.gap` | 0.42 | panel to bar |
| `colorbar.width` | 2.6 | the bar |
| `colorbar.inset` | 5.0 | off each end of the bar, so it is 10 mm shorter than the panel |

The variable names print **below** the panel, not above as R's corrplot does:
a reader who has to look *up* to find the column a cell belongs to is reading
this figure backwards from every other figure in the same paper.

The panel sits the **same distance from the left edge as from the bottom edge**.
Both bands hold the same names, so an unequal corner reads as a mistake in the
figure rather than as a consequence of the text. The badge shares the left/bottom
corner rather than being stacked under it.

The margins are **authored, not derived**. An earlier version sized the left and
bottom bands from the widest measured name, which made the figure's proportions a
property of the dataset: the same card printed twice, with two different sets of
names, came out two different shapes. The fixed margin trades that away. Names are
still measured — a band that cannot hold them is reported, in the log and on the
debug overlay —

```
correlation geometry: the y labels need 11.4 mm and the margin is 11.0 mm, so they print past it.
```

— and with `margin.fit: false` nothing moves in response: a long name prints
past the edge rather than pushing the panel over.

`margin.fit: true`, which the card sets, turns that measurement into an action:
a band too small for its text grows to `tick pad + widest name + margin.slack`,
and the note says so instead.

```
correlation geometry: margin.fit: the names need 12.0 mm plus 1.0 mm of slack, so the corner grew from 11.0 to 13.0 mm.
```

The slack is why the fit is not exact: growing to where the text *ends* puts its
last glyph on the edge of the paper, which reads as a name that was nearly cut
off rather than as one that fits. It is added only to a band that grew — the
card's own 11 mm is the card's business, and a name that already fits does not
get pushed outward.

It only ever **grows** — the card's margin is the floor, so a matrix of short
names keeps the shape the card asked for — and it grows *one* corner for both
bands, so fitting can never trade a clipped y name for a clipped x one. The
right margin grows the same way when the colorbar's numbers and label need more
than it was given.

No trial render is involved, and none is needed: the names are measured with
the tick font at the tick size, so where the text ends is known before anything
is drawn. Drawing a figure to find out would produce the same number, one
discarded figure later. What the fit does *not* know about is a font the
renderer substitutes for a missing one — measure and draw both ask for the same
family, but only the draw would notice a fallback.

The one thing enforced regardless is that `margin.left` and `margin.bottom` are
never smaller than the badge (`logo.offset + logo.height`), since a margin
narrower than the mark would print the two on top of each other. And note that
`tl.pos: n` frees no width unless `margin.fit` is on: without it the margin is
the card's number whether or not anything is printed in it.

The colorbar is deliberately **shorter than the panel** — `colorbar.inset` off each
end, so 1 cm shorter in total and centred on it. It is a key, not a second data
axis, and one run to the full height reads as another column of the matrix. Its
ticks are authored too: five of them, −1 … 1, in `Frame.axccorr.ticks.y`.

Solved by `Figure/corr_layout.py`, which imports nothing from the Figure runtime: numbers
in, numbers out. Label widths are **measured** with `TextToPath` (mathtext included), not
estimated — the usual `0.6 × len` rule runs ~11% wide on ordinary variable names and fails
in the other direction on CJK, and the overrun note is only useful if it is true.

The one hard constraint is the page: `max_width` is **170 mm**, A4's text block. Over it the
width is pinned and the cell shrinks — **never the font**, because a font size that moves
with `n` is what makes two figures in one paper look like they came from different tools.
The clamp says so:

```
width pinned to 170 mm: cell 4.20 -> 3.88 mm (34 variables). Font sizes are unchanged.
```

Height needs no cap of its own: the panel is square and the width is capped, so the height
lands inside A4's text block on its own.

**Nothing on the panel is clipped.** `Style.corrplot.clip: false` on the card
covers everything the method draws: the glyphs, the pie rims, the shade hatches,
the gridlines, the coefficient text and the `addrect` boxes. It is card-owned
rather than one of R's formals — the same footing as `zorder` — so a card that
does frame its panel can ask for the clip back. A cell in the first or last row sits *on* the axes edge, and a
clipped glyph loses the half of itself that is outside — a circle drawn as a
half-moon, which reads as a different mark rather than as a mark near the border;
the outermost gridlines would likewise come out at half the linewidth of every
line inside them. The panel is frameless and the cells are its frame, so there is
nothing for the clip to protect.

**Inside the cell scales with the cell; outside is fixed in points.** Variable names and
colorbar numbers are document-level typography and must match across figures; anything
inside a cell that did not scale would not fit. That line is why the coefficient text is
back-solved from the cell measured off the axes at draw time — so it follows the clamp —
and why the logo is fixed in millimetres rather than as a rect fraction, which would go
out of square the moment the figure height changed. The badge also keeps the house
offset — 0.838 mm (0.0838 cm) from the left and bottom edges, see the `axlogo`
discipline in `STYLE_SCHEMA.md` — through `Geometry.logo.offset`, which is why it sits
outside `margin` rather than inside it.

## 4. YAML

```yaml
Figures:
  - name: correlation
    type: correlation_matrix
    data: mssm
    variables: {exclude: [weight, label]}
    corrplot: {method: circle, type: upper, order: hclust, addrect: 3}
    colorbar: {label: '$\rho$'}
```

`variables` takes three spellings: a **list** (exactly these, in this order — the order the
matrix is drawn in), `{regex: ...}`, or `{exclude: [...]}`. `correlation:` carries the
transform's own options (`missing`, `min_periods`, `triangle`, `include_diagonal`); column
selection never goes there.

`jplot config expand` lowers the macro to its single layer:

```yaml
    style: [corrplot, matrix]
    layers:
      - name: correlation
        data: [{source: mssm, transform: [{correlation: {exclude: [weight, label]}}]}]
        axes: axcorr
        colorbar: axccorr
        method: corrplot
        style: {method: circle, type: upper, order: hclust, addrect: 3}
```

The layers form is fully supported; the macro exists so the four reserved names — `axcorr`,
`axccorr`, `method: corrplot`, `[corrplot, matrix]` — are one word instead of four things
to get right.

**No `coordinates`.** `x_index` / `y_index` / `rho` are the `correlation` transform's
published output columns, so restating them adds a place to be wrong without adding a
choice. The optional `x` / `y` / `c` exist only to point at a renamed column.

## 5. R → YAML crosswalk

`corrplot(M, ...)` formals, spelled as R spells them. Everything below goes in the
`corrplot:` block (or the layer's `style:`).

| R formal | Supported | Notes |
|---|---|---|
| `method` | ✅ `circle`, `square`, `ellipse`, `color`, `shade`, `pie`, `number` | `shade` is `color` plus a sign stripe that survives greyscale |
| `type` | ✅ `full`, `upper`, `lower` | The wedge **as printed**, row 0 at top. Distinct from the transform's `triangle`, which names halves in array terms — leave that at `full` and let corrplot mask |
| `diag` | ✅ | |
| `order` | ✅ `original`, `AOE`, `FPC`, `hclust`, `alphabet` | Config time. Eigenvector signs are pinned, so the same matrix always gives the same picture |
| `hclust.method` | ✅ | scipy linkage names plus R's (`mcquitty` → `weighted`, `ward.D`/`ward.D2` → `ward`) |
| `addrect` | ✅ | Config time; requires `order: hclust` — the boxes are cuts of that tree |
| `rect.col`, `rect.lwd` | ✅ | |
| `col` | ✅ | Translated to `frame.axccorr.color.cmap`; R's diverging presets carry over by name (`RdBu`, `BrBG`, `PiYG`, `PRGn`, `PuOr`, `RdYlBu`), as does any matplotlib or Jarvis colormap. Setting both spellings is an error |
| `addCoef.col` | ✅ | Size is back-solved from the cell, then scaled by `number.cex` |
| `number.digits`, `number.cex`, `addCoefasPercent` | ✅ | |
| `addgrid.col` | ✅ | The panel is frameless, as in R — `axcorr` carries `frameon: false` and no tick marks — so this grid is the drawn boundary, and it follows the mask: with `type: upper` nothing is drawn around the empty half. `null` leaves the matrix unbounded |
| `outline` | ✅ | `true`, a colour, or `sign` — this card's addition, see below. Width is `outline.lwd` |
| `sig.level`, `insig`, `pch`, `pch.cex`, `pch.col` | ✅ | **Off by default**, as in R. Unlike R no `p.mat` is needed: the long table carries `n` per pair, so the p-value comes from `(rho, n)`. `insig`: `pch` marks, `blank` omits the glyph, `n` does nothing |
| `na.label`, `na.label.col` | ✅ | |
| `tl.pos` | ⚠️ `lt`, `t`, `l`, `n` | Config time. It hides names; it does not change the figure size, because the margins are authored (§3). The x names print at the **bottom** (see §3), so `b` / `lb` are accepted as aliases of `t` / `lt` and mean the same band; R's spellings are kept so a pasted R call still runs. R's `d` is refused: the diagonal carries cells here |
| `tl.cex`, `tl.col`, `tl.srt`, `tl.offset` | ⚠️ card-level | Tick text belongs to `Frame.axcorr.ticks` on the style card, which is what keeps it consistent across figures. `tl.cex` is read by the geometry solver |
| `cl.*` (`cl.pos`, `cl.ratio`, `cl.length`, …) | ⚠️ card-level | The colorbar is `axccorr`, an axes of the card. Its label is `colorbar: {label: ...}` |
| `is.corr`, `p.mat` | ❌ | `p.mat` is unnecessary (see `sig.level`); `is.corr = FALSE` would mean drawing a non-correlation matrix, which is a different figure |
| `plotCI`, `lowCI.mat`, `uppCI.mat` | ❌ | Confidence-interval glyphs. Not ported |
| `mar`, `title`, `bg`, `win.asp` | ❌ | Owned by the card and the geometry solver, which is the point of the card |

Five keys in `Style.corrplot` are the card's own rather than R's:

| key | default | what it does |
|---|---|---|
| `glyph.scale` | 0.9 | the glyph at \|rho\| = 1, as a fraction of the cell |
| `ellipse.scale` | 1.4 | `ellipse` only, on top of `glyph.scale` — see below |
| `outline.lwd` | 0.2 | width of the glyph edge R's `outline` switches on |
| `clip` | `false` | nothing on the panel is clipped — see §3 |
| `zorder` | 30 | where the matrix sits among the layers |

`ellipse.scale` exists because the ellipse is the one glyph drawn from two axes
instead of one radius. Rotated 45°, an ellipse of those axes has a bounding box
of `glyph.scale / √2` on a side *whatever rho is* — the shape changes, the box
does not — so at `glyph.scale: 0.9` the widest ellipse covers 0.64 of the cell
while a circle covers 0.9. √2 is the factor that makes the two agree, which is
where 1.4 comes from: it fills the same box the other glyphs do, and still
cannot spill into the neighbouring cell at any rho.

`outline: sign` draws each glyph's own edge in the **end of the scale its sign
points at** — the `rho = 1` colour on a positive cell, `rho = -1` on a negative
one, read off the card's `vmin`/`vmax`. It is what keeps a weak cell readable.
A `circle`, `square`, `pie` or `ellipse` carries \|rho\| in its area, so as the
coefficient falls the glyph shrinks *and* its fill fades toward white at the
same time: at rho = 0.05 there is nearly nothing left on the cell, and the two
loud edge colours are then the only thing saying which way it went.

It is not applied to `color` and `shade`. Those cover the whole cell whatever
rho is, so their edge is not a mark on a glyph but a second grid line, in two
loud colours, next to the one `addgrid.col` already draws — and nothing shrinks
there, so there is nothing to rescue.

## 6. Reading the solve on the figure

`debug: true` on the figure draws the design-reference overlay, and on this card
it carries an extra block under the caption:

```
solved geometry · 13 vars · cell 4.200 mm
corner 12.982 mm · panel 54.60 mm sq
figure 81.60 × 71.58 mm
```

The caption reports the size that came *out*; on a solved card that is the least
interesting number, so this reports what it came out *of* — and any clamp note
verbatim, which is the fastest way to see why a figure is the size it is. The
lines are written by `core_runtime._attach_corr_debug` into the figure's own
`debug:` mapping (see `STYLE_SCHEMA.md`, "Handing the overlay a value from
Python"); the overlay only places them.

Worth knowing when reading those numbers: `plt.figure(figsize=...)` under an
interactive backend is rounded to whole window pixels — the macOS one turns
3.377953 in into 3.37 — which is invisible on an authored card whose `figsize` is
already round, and costs this one 0.2 mm of width, enough to stop the panel being
exactly square. `Figure/config_runtime.py` restates the size after the figure is
built, so the overlay and the solver now agree to the last digit.

## 7. What the card refuses

`Contract.exclusive` is enforced at config time, not left to the renderer, because the card
is solved *from* the matrix:

- exactly one layer, on `axcorr`;
- its `method` must be `corrplot` — `scatter` accepts `type` / `order` / `diag` and
  discards them, which is how the first port of this looked correct and was not;
- rendering the card without the solve is an error, not a fallback: it carries no `figsize`
  and no rects, and `jplot cap styles` reports it as `geometry: solved`,
  `requires_type: correlation_matrix`.

## 8. The other card

`[corrplot, diamond]` draws the same matrix as one triangle turned 45 degrees,
with the names as a plain column of horizontal text and each variable's pairs
as a V rooted at its own name. Same transform, same ordering, same formals;
its own solver, because it is anchored on the row pitch and bounded by the page
height where this one is anchored on the cell and bounded by the width. See
`CORRELATION_DIAMOND.md` and `Example/corrplot_diamond.yaml`.

## 8. Files

| File | Role |
|---|---|
| `jarvisplot/cards/corrplot/matrix.json` | The reserved card: `Contract`, `Geometry` (mm), `Frame`, `Debug`, `Style.corrplot` defaults |
| `jarvisplot/Figure/corr_layout.py` | Geometry solver. No Figure-runtime imports |
| `jarvisplot/Figure/corr_order.py` | `corrMatOrder` + `addrect` blocks. No matplotlib |
| `jarvisplot/Figure/correlation_runtime.py` | The `correlation` transform, `pearson_matrix`, `pearson_pvalues` |
| `jarvisplot/Figure/corrplot_runtime.py` | `draw_corrplot` — everything inside a cell |
| `jarvisplot/Figure/figure_types.py` | `expand_correlation_matrix` |
| `jarvisplot/core_runtime.py` | `prebuild_correlations` — order, geometry, frame, contract |
| `Example/corrplot_matrix.yaml` | Worked example over `Example/correlation_demo.csv` |
| `jarvisplot/cards/corrplot/diamond.json`, `Figure/corr_layout_diamond.py` | The rotated card and its solver (`CORRELATION_DIAMOND.md`) |

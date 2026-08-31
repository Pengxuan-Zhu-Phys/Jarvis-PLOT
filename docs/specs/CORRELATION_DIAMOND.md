# Correlation diamond — design plan

Status: implemented
Companion to
`CORRELATION_MATRIX.md`, which describes the square card. Sections 1-6 describe
what the card does; §7 records what changed against the plan while building it,
§8 records assumptions, and §9 records remaining planned work.

## 1. What it is

The square matrix prints every pair twice and spends half its area on a
redundant triangle. The architectural "programmatic relationship" diagram
solves that by keeping one half and turning it 45°: the names become a plain
vertical list, read like any other list, and each name's pairs fan out from it
as a **V** opening away from the column.

```
  01 Entrance      ┐
  02 Reception     ├──◆     the cell for (02,05) sits at the
  03 Retails       │  ◆ ◆   *average* height of 02 and 05, and
  04 Public toilet │◆ ◆ ◆   as far out as they are far apart
  05 Dining        ┘  ◆ ◆
```

Two things follow, and they are the whole reason to build it:

- A name is one line of horizontal text in a column, so **long names cost
  nothing** but width in a column that was always going to be there. On the
  square card a name is a rotated tick label crammed under a 4.2 mm cell.
- Everything about one variable — all `n-1` of its pairs — is a single
  connected V rooted at its own label, so **one variable can be traced**. On
  the square card its pairs are a row and a column that meet at a corner.

What it costs is aspect: the panel is about twice as tall as it is wide (§2),
so this card runs out of *page height* where the square one runs out of width.

## 2. The coordinate map

The whole card is this map, applied to the upper triangle `i < j`:

```
u = (j - i) / 2        how far apart the two variables are  →  horizontal
v = (i + j) / 2        the midpoint of the two              →  vertical
```

Read off what it implies, because every later section is a consequence:

- **Neighbours.** `(i, j+1)` is at `(u+½, v+½)` and `(i+1, j)` at `(u-½, v+½)`.
  So a cell's four neighbours sit at `(±½, ±½)`: the cell is a **square rotated
  45°** whose two diagonals are 1 unit each.
- **The V.** Variable `k` appears in `(k, j)` for `j > k`, which runs down-right
  from `(0, k)`, and in `(i, k)` for `i < k`, which runs up-right to the same
  point. Its vertex is at `u = 0, v = k` — exactly where its label goes.
- **The label axis is the y axis.** Item `k` sits at `v = k`. So the names stay
  ordinary y tick labels on an axes whose `ylim` is `[n-1, 0]`, and none of the
  tick machinery, the ordering, or `tl.*` has to learn anything new.
- **Extent.** Centres run `v ∈ [½, n-1½]` and `u ∈ [½, (n-1)/2]`; with the
  half-cell the panel is `u ∈ [0, n/2]`, `v ∈ [0, n-1]`. So the panel is
  `n-1` rows tall and `n/2` wide — **height / width = 2(n-1)/n**, which is
  1.85 at n = 13 and approaches 2. Near enough to "twice as tall" to plan
  with, and the exact ratio is what the tests pin.
- **`diag`.** A self-pair lands at `u = 0`, half of it outside the panel. The
  reference diagram omits it and so should the default: `diag: false` on this
  card. With `diag: true` the panel simply starts at `u = -½`.

`side` names the side the *labels* go on, and the card's default is
**`side: right`** — names on the right, colorbar on the left, the triangle
sitting on the left of the page. `side: left` is the same map with `u` negated.
It is a mirror and nothing else — it must not be a second code path.

## 3. Geometry: solved from the row pitch

The square card is anchored on **cell size** and bounded by **width**. This one
is anchored on **row pitch** — the label line height — and bounded by **height**.
That is not a variation, it is the reason it deserves its own solver.

```
panel height = (n - 1) × pitch
panel width  = (n / 2)  × pitch          # about half the height
cell         = a diamond, both diagonals = pitch
glyph        = inscribed, so its diameter is pitch / √2
```

Draft `Geometry` block for `cards/corrplot/diamond.json`:

```json
{
    "units": "mm",
    "pitch": 4.2,
    "pitch_min": 2.6,
    "max_height": 247.0,
    "max_width": 170.0,
    "labels": { "gap": 1.4, "number_gap": 0.6, "width": null },
    "margin": { "fit": true, "slack": 1.0, "top": 4.0, "right": 4.0,
                "bottom": 4.0, "left": 4.0 },
    "colorbar": { "width": 2.6, "length": 0.22, "offset": 10.0,
                  "label_gap": 1.5 },
    "logo": { "width": 5.0, "height": 5.0, "offset": 0.838 }
}
```

The width is the names plus the panel plus a page margin, and nothing else:
**the colorbar does not stand beside the panel.** Turning the matrix leaves a
right-angled triangle of empty page above it as big as the matrix itself, and a
bar parked outside would pay for a band of width twice over — once for itself,
once for the hole. So it goes in the hole, the way a ternary card puts its bar
in the corner beside the triangle. Three numbers place it:

| | |
|---|---|
| top | the panel's own top — a key that starts where the thing it explains starts reads as belonging to it, and one floating below the top edge reads as having drifted |
| `colorbar.length` | **0.22 of the panel's height**, a fraction rather than a length in mm, so the bar keeps the same weight against a 13-variable matrix and a 40-variable one |
| `colorbar.offset` | 10 mm, from the bar's outer edge to the edge of the paper |

Its two pieces of text go on *opposite* sides, and the arrangement is fixed:
**numbers on the left, label on the right, on both mirrors.** It does not
follow `side`. That costs something real and it is worth naming: on a
right-hand diamond the label is the inward text and on a left-hand one the
numbers are, and a column of numbers as tall as the bar is a much larger
obstruction than one turned line — see the clearance rule below.

The label stands `colorbar.label_gap` (1.5 mm) off the bar's inward edge,
turned on its side as a y label is, and **anchored on the edge of the text away
from the bar**. Which edge that is comes from `va`, not `ha`: under
`rotation_mode: anchor` the alignment is applied before the rotation, so
`va: bottom` puts the anchor on the text's right edge and `va: top` on its
left. Matplotlib takes `va` from the *tick* side, which on both mirrors of this
card is the opposite side to the label — left alone it anchors the near edge,
and the label grows toward the bar and prints on the scale. That was worth 1.5
mm of clearance turning into 0.04 mm.

**Nothing overrides the three numbers.** The solve measures the triangle and
*reports*; it does not pick different numbers. An earlier version shortened the
bar to fit a clearance allowance of its own, and so drew a figure that did not
have the length it was asked for with nothing on the page saying so — a worse
fault than a tight figure, and the reason the check is a note now.

What is measured: the panel's upper boundary is the line `v = u − ½`, so the
page above it at the bar's near edge is `(u_near − ½) × pitch`. A bar longer
than that really would run over the cells, and says so. Separately, whichever
of the bar's two texts ends up between it and the diagonal is measured against
the band the edge numbers stand in, and the two are different shapes:

| | reaches inward by | occupies along the bar |
|---|---|---|
| the numbers (inward on a **left**-hand diamond) | tick pad + their width | the whole bar — a wall |
| the label (inward on a **right**-hand one) | `label_gap` + its line *height*, since it is turned on its side | its text *width*, one line at the bar's mid height |

A wall clears that band only when the bar's own bottom does; a single line at
mid height only needs half the bar plus half itself to. Coming inside it is
not a collision — at thirteen variables the bar still clears the cells by 5 mm
— so it is reported and drawn as asked. The key it names is `colorbar.offset`,
and **smaller** is what helps: it moves the bar out toward the paper, where the
triangle is taller.

**All four margins are the same 4 mm.** Nothing prints above the matrix or
below it, so an uneven pair would only make the figure sit crookedly; and with
the colorbar inside the panel there is nothing left for a wide band to hold.

**The badge takes the corner the names are not in.** 4 mm leaves no room under
the last name for a 5.8 mm mark, and the square card's floor — raise the margin
until the badge fits — would undo the 4 mm. It does not need to: the corner
beside the names is the one place on this layout where something already is
(the last variable's row sits on the panel's bottom edge, and the lower run of
numbers ends just inside it), while the opposite corner is empty *by
construction*, because the last variable's only cell is at `u = ½`. So the
badge crosses to that corner and the wordmark mirrors with it
(`Frame.axlogo.anchor`). The house offset — 0.838 mm off two edges — is
unchanged; `logo.corner: bottom-left` puts it back.

- `pitch` 4.2 mm keeps the same anchor number as the square card's cell, which
  makes two figures in one paper look related. Like for like at
  `glyph.scale: 0.9`, the widest glyph is then 2.67 mm across against the
  square card's 3.78 mm — the diamond cell inscribes a circle of `pitch / √2`
  where the square cell inscribes one of `pitch`. Smaller, and acceptable
  because this diagram is read by *rows* rather than by individual marks; a
  card that wants them equal sets `pitch: 5.94` and pays 40 % more height.
- `pitch_min` 2.6 mm is **derived, not chosen**: 6 pt text at matplotlib's 1.2
  leading is 2.54 mm, and below that the names collide. That is a different
  failure from the square card's floor — there the cell stops being a readable
  mark, here the *labels* stop being readable — and it is why the number is
  its own key rather than a copy of `cell_min`.
- **The clamp is on height.** `(n-1) × pitch + margins > max_height` pins the
  height and shrinks the pitch, never the font, exactly as `max_width` does
  today. Width is checked too, but it does not bind first: at the floor the
  page runs out of height at about **n ≈ 90**, where the width is still around
  155 mm with a 30 mm label column.
  Same practical ceiling as the square card, reached for the opposite reason —
  worth saying in the clamp note, because the fix is different (fewer
  variables, or a taller page).
- `labels.width: null` means measured; `margin.fit` from the square card
  carries over unchanged and is what sizes the column. With `edge.numbers` the
  column is two: `pad + name + number_gap + number + gap`, and both halves are
  measured at the size they are actually set in.
- The badge floor, the `logo` block and the `margin.slack` rule are inherited
  verbatim. The `axlogo` discipline is a house rule, not a per-card one.

## 4. What the method learns

One switch, `layout: square | diamond`, stated **once** in the card's
`Contract` and injected into the layer's style by `_prebuild_one` — the channel
`__corr_blocks__` already uses. The style block is not where it belongs: the
author does not choose it, the card does.

| step | square | diamond |
|---|---|---|
| coefficient size | the shorter side of the panel over n | the row pitch over √2, which is the rotated cell's side |
| position | `(x_index, y_index)` | `u = (j-i)/2`, `v = (i+j)/2`, `u ← -u` when `side: right` |
| square glyphs | `_CORNERS` | `_CORNERS` rotated 45° and scaled `1/√2` — one constant |
| grid | `_CELL_LOOP` | the same constant, rotated: the lattice is diamonds |
| `addrect` | a rectangle over `[a, b]²` | **no box at all** — the blocks are told by the shading (below); the sub-triangle `(0,a) (0,b) ((b-a)/2, (a+b)/2)` is still what would be drawn with `stripe: none` |
| `type` | `full`/`upper`/`lower` | one half exists by construction — `full` is refused, not ignored |
| colorbar | a bar beside the panel, in its own band of width | inside the empty triangle above the matrix; ticks and axis label on opposite sides of it |
| `outline: sign` | unchanged | unchanged |
| `pch` | the square cell's diagonals — a cross | the rotated cell's diagonals — a plus, the same construction seen from 45° |
| ellipse angle | 45° in screen space | **unchanged, deliberately** — the pictogram is about the scatter plot, not about the matrix's orientation |

New and specific to this layout:

- **`stripe: alternate`** — the V-shaped band behind every other variable's
  cells, plus the matching band behind its label. This is not decoration: with
  41 names and a diagonal lattice it is the only thing that lets a reader carry
  a name across to a cell. Drawn as the *union* of the tinted variables' cells
  in one `PolyCollection` — one band per variable would double-tint every
  crossing — plus one `axhspan` per tinted name.

  The tint also fills the **notch** beside each name it belongs to. A
  variable's two cells at the edge meet at a point rather than along it, so the
  boundary beside the names comes out as a sawtooth; the notch at `k` is the
  triangle `(0, k−½) (½, k) (0, k+½)`, and filling it carries the band from the
  name into the matrix without a gap *and* turns that sawtooth into the
  straight edge the rules close against.

  **Which** variables are tinted is the interesting half. With `addrect` in
  hand it is alternate **blocks**, so the shading is what tells one cluster
  from the next — which is what the boxes were for, and why they are not drawn.
  Without clusters it falls back to alternate names, the same device at a
  coarser grain.
  `stripe: none` turns it off, `stripe.col` recolours it.

  The names are also ruled apart, at `addgrid.col` and the grid's own weight
  (`_GRID_LWD`, shared rather than repeated). A row of the label column *is* a
  row of the matrix — the name at `v = k` and the V rooted under it are one
  variable — so the line that divides two cells is the line that divides two
  names, and a divider that only nearly matches the grid reads as a second,
  different thing. Below the tick labels, like the band: an axis draws its
  labels at zorder 2.5, and a rule over one of them is a strike-through.
- **`edge.numbers`** — each variable's position, printed three times: at the end
  of both arms of its V, and once more beside its name. A variable's arms are
  the lines `u + v = k` and `v − u = k`, so the ends are `(k/2, k/2)` on the
  upper edge and `((n−1−k)/2, (n−1+k)/2)` on the lower one, turned to lie along
  the edge they sit on, set **across** it rather than along it — text lying on
  a diagonal is text the reader has to tilt their head to take in, and turned
  across, each number reads out of the matrix like the leader on a dimension.
  `edge.numbers.gap` (0.06 cells) is the clearance from the matrix's edge to
  the *near end of the text*, which is what an author would want to set and the
  only form of it that survives the text being turned. The rest is measured:
  the edge under one of these is a cell *face*, half a diagonal in at `0.5/√2`,
  not a corner; the text sticks out from its own centre by half its width, now
  that its width lies along the normal; and the whole offset is struck along
  the diagonal, which is the last factor of √2. Close, on purpose: these name
  the arms they stand at the end of, so they have to read as part of the matrix
  rather than as a scale printed beside it. At
  forty names the far end of an arm is a long way from the name it belongs to;
  R's square matrix answers that by printing the names on both axes, and this
  layout has only one band of names. The numbering is one feature, not two: the
  edge numbers mean nothing unless the names carry them too. The colorbar's
  clearance grows to match — the numbers stand on the diagonal the bar is
  cornered against.

  The third copy is **not** glued onto the name. Numbered, the label band is a
  table of two columns — the position, then the name — because the two are set
  differently: the number in the same smaller, lighter face it wears out on the
  diagonal (`edge.numbers.cex`, 0.7 of the tick label's size, in `edge.col`),
  and a tick label is one string in one colour. Splitting them is also what
  lets the numbers line up. **Each column is anchored on its own outer edge**:
  the numbers make a column because they are zero-padded to a common width and
  all start in the same place, and the names hang off the number column rather
  than off the panel. The cost is a ragged edge against the matrix, which is
  the right side to spend it on — a column of numbers only reads as one if
  every number starts in the same place. Unnumbered there is no second column
  and the names go back to sitting flush against the panel, which is the better
  arrangement when there is nothing to line up with.

  The turn is a *pad*, not just an alignment: matplotlib anchors a tick label a
  pad in from the spine, so `ha: left` alone would set every name running back
  over its own cells. The solve writes the pad — the full width it gave the
  names — into `axcorr.ticks.major.pad`, and `labels.number_gap` (0.6 mm) is
  what separates the two columns. Both gaps are set tight on purpose: a
  position is a tag, not a name, and it should cost the band as little width
  as it can.

  For the same reason **everything that follows the names stops at the
  names**: the tint, the rules between them and the outline that closes the
  figure all reach `name_pad_mm`, so the numbering falls outside all three. A
  box drawn round the tag would say it was part of the name. That one number
  travels as `__corr_label_mm__` and is the reach of all three.
- **`edge.lwd` / `edge.col`** — the outline that closes the figure, at the cell
  grid's own weight (0.3): it bounds the matrix, so it should not read heavier
  than the matrix's own lines. Turned 45°,
  the matrix has no top or bottom edge of its own: its boundary is two
  diagonals meeting the label column at a point, so the first and last names
  sit against nothing. Each rule is that nothing — out across the names, then
  **down the matrix's own diagonal to the apex** — so what it encloses is the
  names and the triangle together, which is the figure. Drawn half a row
  outside the first and last variable (a rule at `v = 0` strikes through the
  first name), which is also exactly where the filled notches end, so the
  horizontal meets the diagonal on the straight edge they make together. Both
  halves are one polyline in data coordinates on `axcorr`, not two artists to
  keep in step.

## 5. Author surface

```yaml
- name: adjacency
  type: correlation_matrix          # the macro is unchanged
  style: [corrplot, diamond]        # the only new word
  data: rooms
  variables: {exclude: [weight]}
  corrplot:
    method: circle
    side: right                     # optional; left is the default
    stripe: alternate               # the default; `none` turns it off
    order: hclust
    addrect: 3
```

`side` lives in `Style.corrplot` because both halves need it — the solver at
config time (which side the label column is on) and the renderer (the sign of
`u`) — and `Style` is already the one place both of them read. `layout` does
**not**: it comes from the Contract, because it is a property of the card.

## 6. Open/closed, concretely

Untouched: `correlation_runtime.py`, `corr_order.py`, the transform, the
schema, the macro in `figure_types.py`, `[corrplot, matrix]` and every figure
that uses it.

Added: `cards/corrplot/diamond.json`, `corr_layout_diamond.py` (or a second
solver function beside `solve_corr_geometry` — decide when writing it), one
`style_preference.json` entry, and the branches in §4.

Changed: `_prebuild_one` gains a two-line dispatch on `Contract.layout`.
`draw_corrplot` gains the position map and the rotated constants. That is the
whole seam, and it is a seam the square card cannot notice.

## 7. What the building changed

The plan held. Four things moved:

- **`stripe` needed the label column's width**, which the renderer cannot
  measure: the band has to run back *under* the name, and how far out the name
  goes is a number only the solver knows. It travels as `__corr_label_mm__`
  beside `__corr_layout__`. And it is the measured *text*, not the column, so a
  short name leaves the page margin white instead of bleeding to the paper edge.
  `__corr_number_mm__` and `__corr_number_pt__` travel the same way and for the
  same reason: the number column is *drawn* by the renderer but placed and
  sized by the solve, which is the only place that knows what the names cost.
- **The label band had to be drawn under the tick labels.** An axis draws its
  labels at zorder 2.5; a band at the glyphs' zorder is a grey rectangle where
  every other name should be. Found by looking at the figure, which is what §8
  said the checkpoints were for.
- **`side: right` needed two more flips than "negate u"**: the names anchor
  `ha: left` instead of `right` (otherwise a right-hand column prints back over
  its own cells), and the colorbar's ticks and label move to its left. Both are
  one line each in the solved frame, and both are things only a rendered figure
  shows.
- **`_cell_points` had to learn the layout.** It measured the shorter side of
  the panel, which on a diamond is the width — half a pitch per variable — and
  every coefficient would have been sized for a cell half the real one. It now
  measures the row pitch and divides by √2, which is the rotated cell's side.

## 8. Assumptions this card makes

Stated so they are cheap to overturn:

1. **The alternating band is on by default** (`stripe: alternate`, `stripe.col`
   `#EFEFEF`). Without it the diagram is pretty and unreadable at n = 41. The
   reference's pale orange is one `stripe.col` away.
2. **The cells keep corrplot's glyphs** — circle/square/ellipse/pie sized by
   |rho|, coloured by rho. The reference's filled dot / open circle / plus is
   its own three-level ordinal scale and is a different figure; that would be a
   `method: pch` over a categorical scale, and separate work.
3. **`type: full` is an error on this card**, not a silently halved figure.
4. **The colorbar stays `axccorr`**, on the side opposite the names.

## 9. Still open

- **Aspect.** Height about 2 × width is fixed by the map. At n = 41 and pitch
  4.2 that is 168 × 86 mm plus the label column — a tall portrait figure that
  will not sit in a two-column paper without shrinking the pitch.
- **`tl.pos`** takes `l` and `n` here and refuses `t`/`lt` with a message: there
  is one band of names, and which side it is on is `side`.

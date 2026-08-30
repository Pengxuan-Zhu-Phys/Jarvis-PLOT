# Jarvis-PLOT — design-reference debug examples

One example per style card. The layout-reference figures set `debug: true`, which draws the
**design-reference overlay**: every axis is outlined, while all axes'
`rect = [left, bottom, width, height]` (figure fractions) and width/height in
centimetres are listed sequentially in one panel inside the primary `ax`.
The figure size and margin / colorbar-gap dimension lines remain on the outer
overlay.

Most figures have **no data layers** on purpose — they are pure layout references
for each style card. `corrplot_matrix.yaml` is the exception: a correlation
matrix has nothing to show without data, and its card solves the figure size
*from* the data rather than fixing it.

## Run

```bash
jplot Example/a4paper_2x1.yaml
```

(`plots/` and `.cache/` are created on the fly and are git-ignored.)

## Styles covered

Examples are grouped by geometry family and paper ratio:

| YAML | Style `[family, variant]` |
|------|---------------------------|
| `a4paper_2x1.yaml` | All `a4paper_2x1` cards: `rect`, `rectRatio`, `rectMarginal`, `rectcmap`, `Ternary`, `TernaryCmap`, `rect_5x1`, `dynesty_runplot` |
| `a4paper_4x1_rectcmap.yaml` | `[a4paper_4x1, rectcmap]` |
| `gambit_2x1.yaml` | All `gambit_2x1` cards: `rectcmap`, `Ternary`, `TernaryCmap` |
| `corrplot_matrix.yaml` | `[corrplot, matrix]` via `type: correlation_matrix` — a real correlation matrix, not a layout reference |
| `corrplot_scales.yaml` | The same card at n = 6 … 100, one glyph method each — where the derivation stops working |

These are every style registered in `jarvisplot/cards/style_preference.json`,
except `a4paper_1x1/Ternary` and `gambit_1x1/Ternary`, whose card JSONs are still
placeholder stubs (no `Frame`) and cannot render yet.

## The `corrplot` family is different

`[corrplot, matrix]` is the only card whose **figure size is derived, not
declared**. It carries a `Geometry` block in millimetres — cell size, padding,
colorbar width, a 170 mm cap — and the figure size falls out of the matrix:
`n` variables and the measured width of their names. Cell size and font size
stay fixed as `n` grows; the figure grows instead, and only past the cap does
the cell shrink (never the font).

That is why the card has no `figsize` and no axes `rect`, why its panel and
colorbar are called `axcorr` / `axccorr` rather than `ax` / `axc`, and why
`jplot cap styles` reports it as `geometry: solved, requires_type:
correlation_matrix`. Naming it in `style:` alone is an error, not a fallback.

`corrplot_matrix.yaml` therefore states no `figsize`, no axes `rect`, no tick
positions and no coordinates: `prebuild_correlations` measures the variable
names and writes all of it into the config before the figure is built. Add a
variable to the CSV and the figure grows — the cells and the type do not move.

It is also the one example written as a **`type:` macro** rather than as
layers, because the four reserved names here (`axcorr`, `axccorr`,
`method: corrplot`, `[corrplot, matrix]`) are not choices — they are one thing
spelled four times. `jplot config expand Example/corrplot_matrix.yaml --figure
corrplot_matrix` shows the single layer it lowers to.

Under it is **`method: corrplot`**, a drawing primitive of its own rather than a
`scatter` layer with a computed marker size. That distinction is what makes
`method: circle | square | ellipse | color | shade | pie | number` mean
anything: only `circle` is expressible as a scatter marker. The `corrplot:`
block takes R's own formals, spelled as R spells them.

The one split worth knowing is where each formal is applied. `order`,
`addrect`, `hclust.method`, `tl.pos` and `col` decide **which variable sits
where** (or what the whole scale is), and the tick labels are written at config
time — so those are resolved by `prebuild_correlations`, which writes the
resolved order into the transform's `columns`. Everything else describes **one
cell** and is applied at draw time. Nothing reorders a matrix after its labels
exist. `jplot man type-correlation-matrix` and
`docs/specs/CORRELATION_MATRIX.md` carry the full crosswalk.

### Where it stops working

`corrplot_scales.yaml` is the experiment behind the claim above: one 100-column
table, one card, seven values of `n`, a different glyph method each time. Up to
`n = 33` the cell stays 4.20 mm and the figure grows to fill 170 mm. From
`n = 40` the width is pinned and the **cell** shrinks — never the font, which
is what lets two of these sit in one paper. At `n = 100` even the 1.60 mm cell
floor does not fit: the solver keeps the floor, overruns the page by 13.8 mm,
and says so. The two `corrplot_100_*` figures are included precisely because
they are past the limit — the labelled one has 100 names colliding in 160 mm,
which is what "past the limit" looks like.

## Notes

- Variant tokens are case-sensitive: `rect`, `rectcmap` (lower-case) and
  `Ternary`, `TernaryCmap` (capitalized).
- To turn the overlay off, set `debug: false` (or remove it).
- To see the overlay on a real plot, add normal `layers` to any of these files.

See the docs: **Jarvis-PLOT → Style Cards and Layout** for a full explanation of
the card JSON structure and the overlay.

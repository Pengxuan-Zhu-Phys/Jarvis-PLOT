# STYLE_SCHEMA

Status: partial

## Purpose

This document defines the intended style-card contract for Jarvis-PLOT.

Current implementation reads style bundles from JSON files under `jarvisplot/cards/**` and combines them with figure and layer overrides at runtime.

There is no schema validator for style cards yet.

## Current Shape

The current bundle shape is effectively:

```json
{
  "Frame": {},
  "Debug": {},
  "Style": {},
  "Layers": []
}
```

Where:

- `Frame` describes figure and axes configuration
- `Debug` describes the design-reference overlay drawn when a figure sets
  `debug:` (see below)
- `Style` describes method defaults and render-time style values
- `Layers` is optional and provides default figure layers when the YAML figure
  does not define `layers`

`Layers` is intended for complete reusable plot formats such as
`dynesty_runplot`, where the style card owns the standard axes, labels, and
method layer. YAML-level `layers` still takes precedence when present.

## The `Debug` Block

`Debug` configures the design-reference overlay: the annotation layer that
labels every axes box with its `rect`, its size in centimetres, the figure
margins and the colorbar gap. It draws only when a figure asks for it.

Two things come from separate places:

- **Whether the overlay is drawn** comes from YAML: `Figures[].debug`, either
  `true` / `false` or a mapping (which turns it on and overrides this block for
  that one figure). A card cannot suppress a `debug: true` the user wrote.
- **What it looks like, and which annotations appear at all**, comes from this
  block.

Precedence is packaged defaults, then the card's `Debug`, then a YAML `debug:`
mapping. Every key is optional; anything absent falls back.

### Groups

| Group | Owns |
|---|---|
| `primary_order` | which axes gets fully dimensioned and hosts the info panel |
| `units` | `cm_per_inch`, `pt_per_inch` |
| `overlay` | the transparent full-figure axes every annotation is drawn onto |
| `figure` | border, size caption, total-height marker |
| `axes` | per-axes outline (framed / frameless) and the frameless corner ticks |
| `dimension` | cap bars, arrowheads and labels -- every measured distance uses these |
| `margins` | **every** axes' top / bottom insets, each riding that axes' own right border (`marker_step` separates crowded columns, `exclude` names axes to leave alone -- `axlogo` by default); plus the primary axes' left inset |
| `colorbar_gap` | gap between the primary axes and a colorbar |
| `solved` | on cards whose size is solved rather than authored, what it was solved *from* — printed under the caption. Its `lines` is the one part of `Debug` written by Python: the solver fills it through the figure's own `debug:` mapping, and a card that solves nothing leaves it empty and draws nothing |
| `colorbar_preview` | fills a colorbar no layer fed with the card's own colormap, so a `layers: []` reference still shows the bar; only ships on cards that have an `axc` axes |
| `numbered_axes` | the staggered edge dimensions for `ax0`, `ax1`, … |
| `panel` | the "axes layout" information card: one type scale on every figure, box measured from its content, free to run past the axes |
| `ternary` | tick anchors and label leaders; only on cards with an `axtri` axes |

`jplot cap styles --json` reports each card's `debug_groups`.

### Two shapes inside a group

**Appearance lives in a block named after the matplotlib callable that
consumes it, and is splatted straight into that call. Geometry stays a named
scalar beside it.**

```json
"axes": {
  "frameless_extension_max": 0.018,
  "outline": {
    "show": true,
    "framed":    { "Rectangle": { "fill": false, "ec": "#111111", "lw": 0.45, "zorder": 10001 } },
    "frameless": { "Rectangle": { "fill": false, "ec": "#FF3FA4", "lw": 0.45, "zorder": 10001 } }
  },
  "corner_ticks": {
    "show": true,
    "plot": { "color": "#FF3FA4", "lw": 0.45, "solid_capstyle": "projecting", "zorder": 10001 }
  }
}
```

The drawing code is then literally `ov.plot(xs, ys, **cfg["corner_ticks"]["plot"])`.
One drawn thing owns one complete kwargs table: colour, width and stacking order
are written where they are used, not inherited from a palette the card would
have to know about.

The call blocks are `add_axes`, `annotate`, `plot`, `scatter`, `text`,
`Rectangle` and `FancyBboxPatch`. Their keys -- including nested `bbox` and
`arrowprops` -- are matplotlib's vocabulary and are passed through unchecked,
the same way `layers[].style` is in the YAML schema. An override inside a call
block is layered key by key, so a card may change one arrow property without
restating the rest.

Two kwargs are Python's to supply because they depend on what is being
measured: the narrow arrowhead's `mutation_scale`, from the span, bounded by
`dimension.narrow_scale_min` / `_max`. The panel text's `fontsize` is also
supplied by Python, but it is `panel.<part>.size` verbatim -- the panel no
longer rescales itself to fit, so the same numbers read at the same size on
every card.

### Switching annotations off

Every annotation element takes `show`. Absent means on, so a card written
before an element existed keeps drawing it.

```json
"panel": { "show": false }
```

`show` is a veto, not a command: the structural conditions still apply. Setting
`colorbar_gap.show: true` on a card with no colorbar draws nothing.

### Handing the overlay a value from Python

`Figures[].debug` is both the switch and the per-figure override, which makes it
the delivery channel for anything only the config stage knows — the correlation
matrix's solve is the one case today (`core_runtime._attach_corr_debug`). Two
rules keep it a channel rather than a second way to turn the overlay on:

- carry `show` through explicitly. The mapping form defaults to *on*, so a
  figure that never asked for the overlay must be written back as
  `{"show": False, ...}`;
- put the value in a group the defaults define, or `merge_debug_config` reports
  it as an unknown key and drops it.

### Validation

Card JSON has no JSON-schema. Instead the merge checks each `Debug` key against
the packaged defaults at runtime and reports unknown keys with a did-you-mean
through `logger.warning`, and `tests/test_card_debug_shape.py` checks the
shipped cards at CI time. A bad block degrades to the defaults; the overlay
never raises, because it must not be able to break a plot.

Owner: `jarvisplot/Figure/debug_config.py` (defaults and merge),
`jarvisplot/Figure/design_runtime.py` (drawing).

## The `axlogo` Discipline

Every card that carries the Jarvis-HEP badge places it the same way, and the
numbers are not negotiable per card:

| Quantity | Value |
|---|---|
| offset from the **left** figure edge | **0.0838 cm** (0.838 mm, 0.033 in) |
| offset from the **bottom** figure edge | **0.0838 cm** |
| badge size | 0.503 cm square (0.198 in) |

`axlogo` is written as a rect in figure fractions, so the two offsets are
*different* fractions on any figure that is not square, and a card that copies
one number into both corners is wrong on the page even though it looks
symmetric in JSON. Compute them from the card's own `figsize`:

```
rect = [0.0838 / (2.54 * width_in), 0.0838 / (2.54 * height_in),
        0.503  / (2.54 * width_in), 0.503  / (2.54 * height_in)]
```

Nothing on `axlogo` is clipped, either — not the icon and not the wordmark.
Every card carries `Frame.axlogo.clip: false`, which the setter applies to the
image and to each `text` entry (an entry may still override it with its own
`clip_on`). The axes is sized to the icon, so a clip can only ever shave a pixel
row off an edge, and the "Powered by Jarvis-HEP" text is written at `x = 1.0` in
axes coordinates — outside the axes, so unclipped is the only way it exists at
all. It is a card key rather than a constant in `Figure.axlogo` so the rule sits
where the rest of the badge is described; omit it and matplotlib's default
(clipped) applies.

The rule is about *printed* distance because that is what a reader sees. The
badge is a mark of provenance, not a plot element: it must sit in the same
place, at the same size, on a 3.3 x 2.75 in panel and on a 6.7 x 6.5 in
correlation matrix, or a paper with two Jarvis-PLOT figures in it looks like a
paper made with two tools. This is also why the badge sits *outside* the page
pad rather than inside it — the pad insets the content, and the mark is not
content.

Cards whose size is solved rather than authored cannot write a rect at all, so
they carry the discipline as millimetres in `Geometry` and divide by the solved
figure at the end; `corrplot/matrix.json` spells it `Geometry.logo.offset:
0.838`, consumed by `Figure/corr_layout.py`.

Two deliberate departures, so they are not read as drift:

- `a4paper/4x1/rect_cmap.json` is a 1.7 in card and scales the whole badge
  block by ~0.5 (0.0432 cm offset, 0.259 cm badge). A full-size badge on it
  would be a fifth of the figure width.
- The `gambit/**` family is a different house style with its own header band,
  and positions the badge inside that band.

## Current Owner

- `jarvisplot/core.py` loads the bundle map
- `jarvisplot/Figure/style_runtime.py` resolves the bundle payload
- `jarvisplot/Figure/figure.py` applies frame/style defaults, stores optional default layers, and owns the `debug` / `debug_config` properties
- `jarvisplot/Figure/debug_config.py` holds the overlay defaults and the merge
- `jarvisplot/Figure/config_runtime.py` applies YAML `layers` or falls back to style-card `Layers`
- `jarvisplot/cards/**` stores the actual JSON files

## Boundary Rule

Keep colorbar defaults explicit. `frame.axc.color` is the preferred source of truth.

Layer-level style keys are layer kwargs; `frame.axc.color` is the colorbar contract.

Figure blocks and style cards do not own file-output policy. Output directory,
DPI, and formats are project/YAML-level settings and should be configured through
the top-level `output` block, for example `output.formats`. Do not add
figure-local `output: {formats: ...}` overrides.

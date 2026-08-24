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

### Validation

Card JSON has no JSON-schema. Instead the merge checks each `Debug` key against
the packaged defaults at runtime and reports unknown keys with a did-you-mean
through `logger.warning`, and `tests/test_card_debug_shape.py` checks the
shipped cards at CI time. A bad block degrades to the defaults; the overlay
never raises, because it must not be able to break a plot.

Owner: `jarvisplot/Figure/debug_config.py` (defaults and merge),
`jarvisplot/Figure/design_runtime.py` (drawing).

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

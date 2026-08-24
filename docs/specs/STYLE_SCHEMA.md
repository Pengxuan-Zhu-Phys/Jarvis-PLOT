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
| `palette` | the five overlay colours (`dimension`, `box`, `name`, `margin`, `panel`) |
| `zorder` | the overlay / artist / text stacking order |
| `units` | `cm_per_inch`, `pt_per_inch` |
| `labelstyle`, `labelbox` | dimension-label typography and its background box |
| `figure` | border, size caption, total-height marker |
| `axes` | per-axes outline and the frameless corner ticks |
| `dimension` | dimension-line geometry, cap size, arrowheads |
| `margins` | the primary axes' left / top / bottom insets |
| `colorbar_gap` | gap between the primary axes and a colorbar |
| `numbered_axes` | the staggered edge dimensions for `ax0`, `ax1`, … |
| `panel` | the "axes layout" information card |
| `ternary` | tick anchors and label leaders; only on cards with an `axtri` axes |

`jplot cap styles --json` reports each card's `debug_groups`.

### Two shapes inside a group

Following the ternary renderer's convention: **geometry is a named scalar,
appearance is a `*style` block splatted straight into the matplotlib call.**

```json
"axes": {
  "outline_linewidth": 0.45,
  "outline": { "show": true, "style": { "fill": false } },
  "corner_ticks": { "show": true, "style": { "solid_capstyle": "projecting" } }
}
```

Keys inside `*style`, `bbox` and `boxstyle_kwargs` are matplotlib's vocabulary
and are passed through unchecked, the same way `layers[].style` is in the YAML
schema.

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

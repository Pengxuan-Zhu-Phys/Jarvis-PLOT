# Jarvis-PLOT Style System Design

Status: partial

## Purpose

This document defines the style boundary for Jarvis-PLOT.

The style system should own:

- style bundle selection
- style card loading
- override merge order
- theme defaults
- backend-neutral style normalization

It should not own data loading or layout placement.

## Current Reality

Style behavior is currently spread across:

- `jarvisplot/core.py` for wiring the loaded style bundle into the runtime
- `jarvisplot/core_assets.py` for bootstrap helpers that return the loaded style bundle
- `jarvisplot/Figure/figure.py` for figure/frame merge and per-layer style merge
- `jarvisplot/Figure/style_runtime.py` for style family / variant resolution
- `jarvisplot/cards/**` for bundle assets
- `jarvisplot/utils/cmaps.py` for colormap registration

The current source of truth for shared colorbar settings is `frame.axc.color`.

Style cards may also provide optional default `Layers`. This is used for
complete reusable formats where the card owns both axes style and the standard
render layer, for example the `a4paper_2x1/dynesty_runplot` card.

## Merge Order

The current implementation effectively uses this order:

1. bundle defaults
2. default layers from `Layers` when YAML has no `layers`
3. frame overrides
4. YAML layer overrides when provided
5. render-time style resolution

That order should stay explicit.

YAML figure `layers` override card `Layers` entirely. This keeps reusable cards
compact while preserving an escape hatch for bespoke figures.

## Boundary Rule

Do not turn the renderer into the main style-definition site.

If a style choice is truly global, it belongs in the style assets or a future style owner module.

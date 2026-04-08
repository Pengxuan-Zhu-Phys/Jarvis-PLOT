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

## Merge Order

The current implementation effectively uses this order:

1. bundle defaults
2. frame overrides
3. layer overrides
4. render-time style resolution

That order should stay explicit.

## Boundary Rule

Do not turn the renderer into the main style-definition site.

If a style choice is truly global, it belongs in the style assets or a future style owner module.

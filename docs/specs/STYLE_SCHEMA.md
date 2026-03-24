# STYLE_SCHEMA

Status: spec only

## Purpose

This document defines the intended style-card contract for Jarvis-PLOT.

Current implementation reads style bundles from JSON files under `jarvisplot/cards/**` and combines them with figure and layer overrides at runtime.

There is no schema validator for style cards yet.

## Current Shape

The current bundle shape is effectively:

```json
{
  "Frame": {},
  "Style": {}
}
```

Where:

- `Frame` describes figure and axes configuration
- `Style` describes method defaults and render-time style values

## Current Owner

- `jarvisplot/core.py` loads the bundle map
- `jarvisplot/Figure/figure.py` applies the figure and layer merge
- `jarvisplot/cards/**` stores the actual JSON files

## Boundary Rule

Keep colorbar defaults explicit. `frame.axc.color` is the preferred source of truth.

Layer-level style keys are compatibility fallback, not the primary contract.

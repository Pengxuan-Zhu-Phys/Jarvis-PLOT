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
  "Style": {},
  "Layers": []
}
```

Where:

- `Frame` describes figure and axes configuration
- `Style` describes method defaults and render-time style values
- `Layers` is optional and provides default figure layers when the YAML figure
  does not define `layers`

`Layers` is intended for complete reusable plot formats such as
`dynesty_runplot`, where the style card owns the standard axes, labels, and
method layer. YAML-level `layers` still takes precedence when present.

## Current Owner

- `jarvisplot/core.py` loads the bundle map
- `jarvisplot/Figure/style_runtime.py` resolves the bundle payload
- `jarvisplot/Figure/figure.py` applies frame/style defaults and stores optional default layers
- `jarvisplot/Figure/config_runtime.py` applies YAML `layers` or falls back to style-card `Layers`
- `jarvisplot/cards/**` stores the actual JSON files

## Boundary Rule

Keep colorbar defaults explicit. `frame.axc.color` is the preferred source of truth.

Layer-level style keys are layer kwargs; `frame.axc.color` is the colorbar contract.

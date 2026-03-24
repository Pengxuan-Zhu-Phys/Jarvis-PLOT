# PROFILE_SCHEMA

Status: spec only

## Purpose

This document defines the intended profile-step contract for Jarvis-PLOT.

Current implementation uses `profile` and `grid_profile` transform steps inside YAML layer transforms.

## Current Shape

The current profile payload is effectively a transform config object with fields such as:

```json
{
  "coordinates": {
    "x": { "expr": "..." },
    "y": { "expr": "..." },
    "z": { "expr": "..." }
  },
  "method": "bridson",
  "bin": 40,
  "objective": "max"
}
```

For `grid_profile`, the method is usually treated as grid-style reduction.

## Current Owner

- `jarvisplot/Figure/load_data.py` contains the reduction algorithms
- `jarvisplot/Figure/preprocessor.py` owns projection, cache identity, and prebuild/runtime behavior
- `jarvisplot/data_loader.py` applies dataset-level profile transforms

## Boundary Rule

Profile steps are data-reduction transforms, not rendering instructions.

If a field changes what rows survive or how they are aggregated, it belongs in the profile contract.

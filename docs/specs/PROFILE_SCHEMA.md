# PROFILE_SCHEMA

Status: spec only

## Purpose

This document defines the intended profile-step contract for Jarvis-PLOT.

Current implementation uses `profile` transform steps inside YAML layer transforms.

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

For grid-style reduction, use `profile` with `method: grid`.

## Current Owner

- `jarvisplot/Figure/profile_runtime.py` contains the reduction algorithms
- `jarvisplot/Figure/preprocessor.py` and `jarvisplot/Figure/preprocessor_runtime.py` own projection, cache identity, and prebuild/runtime behavior
- `jarvisplot/data_loader.py` applies dataset-level profile transforms

## Boundary Rule

Profile steps are data-reduction transforms, not rendering instructions.

If a field changes what rows survive or how they are aggregated, it belongs in the profile contract.

## Dataset Transform Contract

Dataset-level `transform` is an ordered YAML list and is executed strictly top-to-bottom.
The runtime does not reorder steps.
Each step sees the dataframe produced by the previous step.

Supported dataset transform steps include:

- `filter`
- `profile`
- `make_density_core`
- `make_interp_2d`
- `add_column`
- `sortby`
- `keep_columns`
- `drop_columns`
- `to_csv`
- `to_parquet`

There is no standalone `expression` transform step. Computed columns are added through `add_column`.

Execution rules:

- `keep_columns` and `drop_columns` are the only explicit column-pruning steps.
- If a transform list does not contain one of those pruning steps, no implicit column pruning is applied.
- `to_csv` and `to_parquet` execute in list order and export the dataframe state at that point.
- If an export step is the last transform step, the runtime may reuse the final frame without extra post-processing.
- If an export step appears before later transforms, those later transforms still run and the export captures the earlier state.
- `profile` remains a reduction step and may shrink row counts or reshape the table.

Example:

```yaml
transform:
- filter: (mH1 >= 115) & (mH1 <= 135)
- add_column:
    name: z
    expr: np.exp(LogL1)
- keep_columns:
  - pM1
  - pMu
  - mC1
  - mN1
  - LogL1
  - z
- to_csv: ./data/MSSM7_light.csv
```

# Jarvis-PLOT Profile System Design

Status: implemented but mixed

## Purpose

This document defines the profile boundary for Jarvis-PLOT.

The profile system currently acts as a data-reduction stage for the plotting pipeline.

It should own:

- `profile` transform semantics, including `method: grid`
- prebuild/runtime split behavior
- cache identity for profile results
- narrow selection-table reduction

It should not own final rendering.

## Current Reality

The profile system is implemented across:

- `jarvisplot/Figure/profile_runtime.py`
- `jarvisplot/Figure/preprocessor.py`
- `jarvisplot/Figure/preprocessor_runtime.py`
- `jarvisplot/data_loader.py`
- `jarvisplot/data_loader_runtime.py`
- `jarvisplot/data_loader_hdf5.py`

Current behavior:

- `filter`, `add_column`, `sortby`, `keep_columns`, `drop_columns`, `to_csv`, and `to_parquet` remain in the transform primitive layer
- computed columns are created through `add_column`; there is no standalone `expression` transform type
- `profile` lives in `profile_runtime.py` and is called through the transform pipeline
- `make_density_core` and `make_interp_2d` are field-preparation transforms, not profile methods
- prebuild can rewrite the first profile step into a reusable alias
- runtime reuses compact cached profile tables when possible
- the pipeline is designed to stay narrow

## Boundary Rule

Profiles are data transforms, not view primitives.

If a change affects binning, reduction, demand projection, or cache identity, it belongs here.

If a change affects colors, legends, or draw order, it belongs in the renderer.

# Jarvis-PLOT Profile System Design

Status: implemented but mixed

## Purpose

This document defines the profile boundary for Jarvis-PLOT.

The profile system currently acts as a data-reduction stage for the plotting pipeline.

It should own:

- `profile` and `grid_profile` transform semantics
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

- `filter`, `add_column`, and `sortby` remain in the transform primitive layer
- `profile` and `grid_profile` live in `profile_runtime.py` and are called through the transform pipeline
- prebuild can rewrite the first profile step into a reusable alias
- runtime reuses compact cached profile tables when possible
- the pipeline is designed to stay narrow

## Boundary Rule

Profiles are data transforms, not view primitives.

If a change affects binning, reduction, demand projection, or cache identity, it belongs here.

If a change affects colors, legends, or draw order, it belongs in the renderer.

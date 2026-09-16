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

## Finite extrema and preprofile cache identity

Pre-binning retains one finite extremum per occupied cell, selected by
`objective: max` (default) or `objective: min`. It preserves the selected source
row, including its coordinates and other columns. It must not emit both extrema
into a one-sided profile. Binning remains a spatial approximation: changing the
pregrid resolution can change the support supplied to runtime reduction.

The prebuild split preserves `objective`, `pregrid`, and `pregrid_bin` in addition
to `coordinates`. These settings participate in the preprofile cache identity;
runtime `method` and `bin` may still reuse a compatible preprofile. In particular,
`pregrid: false` must survive the split and disable the prebuild reduction.

Bridson drops non-finite x/y/z source rows before computing normalization ranges.
A NaN objective previously poisoned the whole z range and disabled z-aware
exclusion, allowing near-zero likelihood points to survive inside high-likelihood
regions. Constant linear z fields use a finite normalization denominator. Empty
finite input produces an empty support table, with named output columns intact.

After greedy Bridson thinning, assign all finite input samples to their nearest
retained source seed in normalized x/y space. Select the requested extremum in
each resulting Voronoi cell. A candidate suppressed during thinning still
contributes to that reduction; otherwise an inferior seed can survive just
outside an earlier seed's exclusion radius. Return the entire winning source
row, with its own x/y and other columns. Consequently, final extrema can be
closer together than the original seeds. Synthetic empty-grid support remains
NaN and does not compete with source rows for cell membership.

Pipeline and profile-layer signatures are versioned to invalidate the old
preprofile, runtime support, and downstream named interpolation caches. This fix
retains irregular Bridson-derived cells and does not change the interpolation
algorithm.

## Boundary Rule

Profiles are data transforms, not view primitives.

If a change affects binning, reduction, demand projection, or cache identity, it belongs here.

If a change affects colors, legends, or draw order, it belongs in the renderer.

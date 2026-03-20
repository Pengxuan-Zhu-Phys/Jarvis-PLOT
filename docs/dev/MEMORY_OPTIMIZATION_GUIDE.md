# Memory Optimization Guide

JarvisPLOT 1.3.0 changed the data pipeline to lower peak memory without changing the plotting model. This document records the reason for that change and the mechanisms that now protect the pipeline.

## Previous Architecture Problem

The old failure mode was a wide-table pipeline:

```text
collect -> 66-column pandas table
        -> profile/preprofile on the full table
        -> cache full or near-full tables
```

Result:

- cold peak RSS around `3.05 GB`
- warm peak RSS around `0.75 GB`

Those numbers are release-validation benchmarks from the 1.3.0 optimization cycle. They describe why the current architecture exists.

## New Architecture

The 1.3.0 pipeline moves memory pressure out of the hot path by narrowing tables as early as possible.

### 1. Compact Dataset Collect

Implemented mainly in `jarvisplot/data_loader.py`.

- `JarvisPLOT.plan_dataset_required_columns()` computes `required_columns` and `retained_columns` before datasets are materialized.
- HDF5 sources use `_build_hdf5_whitelist()` to avoid loading unrelated leaf datasets.
- `_load_hdf5_materialized()` writes selected columns to cached parquet parts instead of assembling a giant in-memory dataframe first.
- `_apply_dataset_transform_polars()` keeps filter/sort/add-column work in polars when possible, then collects only the kept columns.

Impact:

- fewer columns cross the collect boundary
- fewer bytes reach pandas
- less duplicate dataframe copying later

### 2. Retained Column Planning

Implemented in `jarvisplot/core.py`.

- Layer coordinates, style expressions, and transform expressions are scanned up front.
- Dataset-level transform inputs and outputs are folded into the retained set.
- `JP_ROW_IDX` is forced in so later enrichment can recover specific rows.

Impact:

- datasets retain the smallest safe working set
- wide-table regressions become visible at planning time

### 3. Selection-Table Profiling

Implemented in `jarvisplot/Figure/preprocessor.py` and `jarvisplot/Figure/load_data.py`.

- `_runtime_projection()` keeps only transform inputs, transform outputs, `__jp_row_idx__`, and layer demand.
- `_runtime_cache_columns()` trims the post-transform dataframe before cache storage.
- `_preprofile_base_projection()` narrows prebuild work to the first profile step's true coordinate demand.
- `_preprofiling()` reduces the candidate set before runtime profiling continues.

Impact:

- `profile`, `preprofile`, and `grid_profile` operate on selection tables instead of full datasets
- runtime caches store compact payloads

### 4. Narrow Cache Payloads

Implemented in `ProjectCache` plus `DataPreprocessor.run_pipeline()`.

- `.cache/data` stores pipeline outputs after projection, not raw source tables
- cache metadata includes a demand fingerprint and runtime profile signature so stale wide payloads are not reused
- `share_data` prefers `put_named_reference()` when a named layer can point at an existing compact pipeline cache entry

Impact:

- warm runs stay narrow
- cache reuse no longer implies wide-table reuse

### 5. Demand-Based Enrichment

Implemented in `DataPreprocessor._enrich_for_demand()` and `DataSet.fetch_rows_columns()`.

- the layer render path asks only for columns referenced by layer coordinates or style expressions
- missing columns are fetched by `__jp_row_idx__`
- the fetch is merged into the selection table only at the render boundary

Impact:

- display-only columns do not bloat profiling or cache storage
- the runtime path can stay compact while still supporting rich layer expressions

## Benchmark Example

| Metric | Before | After |
| --- | ---: | ---: |
| Cold peak RSS | `~3.05 GB` | `~1.66 GB` |
| Warm peak RSS | `~0.75 GB` | `~0.41 GB` |

These numbers are the representative 1.3.0 benchmark targets that the current design is meant to preserve.

## Remaining Hotspot: polars -> pandas Conversion

The largest remaining boundary is the conversion from `polars` to `pandas`.

Where it happens:

- `jarvisplot/data_loader.py:polars_to_pandas_compat()`
- `jarvisplot/Figure/preprocessor.py:_polars_to_pandas_compat()`
- `jarvisplot/Figure/figure.py:_polars_to_pandas_compat()`

Why it still exists:

- transform primitives in `jarvisplot/Figure/load_data.py` are written against pandas and numpy semantics
- coordinate evaluation uses pandas dataframe locals for Python expression evaluation
- render adapters expect in-memory numpy arrays derived from pandas-backed columns
- some cache payloads are stored with `DataFrame.to_pickle()`, which is pandas-specific

Current architectural position:

- keep work lazy in polars as long as possible
- cross into pandas only after column pruning
- do not introduce earlier pandas materialization in new code unless a stage truly requires it

## Operational Notes

- `columns.load_whitelist` and `columns.isvalid_policy` are part of the memory strategy, not just source parsing options.
- `JP_MEM_TRACE=1` is the supported way to validate memory behavior while changing the pipeline.
- `--rebuild-cache` is useful when checking whether a change affects cold-cache behavior or demand fingerprints.

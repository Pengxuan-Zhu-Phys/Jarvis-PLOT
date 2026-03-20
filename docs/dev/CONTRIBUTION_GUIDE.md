# Contribution Guide

This guide explains how to extend JarvisPLOT without breaking the 1.3.0 dataflow architecture.

Read these documents first:

- `docs/dev/ARCHITECTURE_OVERVIEW.md`
- `docs/dev/DATAFLOW_ARCHITECTURE.md`
- `docs/dev/DEVELOPER_RULES.md`

## General Workflow

1. Identify whether the change affects:
   - source loading
   - transform execution
   - layer rendering
   - cache payload shape
2. Preserve the three-table model:
   - dataset table
   - selection table
   - enriched table
3. Verify that the change does not widen the runtime pipeline accidentally.
4. If the stage is memory-relevant, validate it with `JP_MEM_TRACE=1`.

## Adding a New Layer

There are two different layer changes.

### A. Add a new YAML method alias

If the layer is only a new name for an existing drawing primitive:

- update `jarvisplot/Figure/method_registry.py`
- register the YAML key, aliases, and allowed axes types

### B. Add a new drawing primitive

If the layer needs new draw logic:

- implement the method on:
  - `StdAxesAdapter` for rectangular axes
  - `TernaryAxesAdapter` for ternary axes
- keep `Figure.render_layer()` as the routing point that prepares coordinates and style payloads

Notes:

- `Figure.render_layer()` already evaluates coordinate expressions into numpy arrays
- `grid_profile` is the current example of a custom adapter method that receives both arrays and the backing dataframe via `__df__`

### How a Layer Requests Columns

Render-time column demand is discovered from expressions, not from adapter code.

Today the planners inspect:

- layer coordinate expressions
- layer style expressions
- transform expressions

That means a new layer should request columns by exposing them through config expressions, for example:

```yaml
coordinates:
  x: {expr: "mN1"}
  y: {expr: "mC1"}
style:
  c: {expr: "LogL"}
```

If you introduce a new config field that can contain expressions, update both:

- `JarvisPLOT._collect_expr_columns()` in `jarvisplot/core.py`
- `DataPreprocessor._layer_expr_columns()` or related projection helpers in `jarvisplot/Figure/preprocessor.py`

If you do not update both sides, the layer may render only by accident on warm caches and will regress later.

## Adding a New Transform

Transform behavior currently lives in two places:

- execution primitives in `jarvisplot/Figure/load_data.py`
- orchestration and projection logic in `jarvisplot/Figure/preprocessor.py`

Dataset-level transforms also pass through `DataSet._apply_dataset_transform()` and, when possible, `DataSet._apply_dataset_transform_polars()`.

Required checklist for a new transform:

1. Implement the transform primitive.
2. Add runtime handling in `DataPreprocessor._apply_transforms()`.
3. If it is safe for lazy pushdown, add it to `DataSet._apply_dataset_transform_polars()`.
4. Update input-column discovery.
5. Update output-column discovery.
6. Confirm that cached payloads still project correctly.

The input/output discovery step matters because selection tables are built from those helper functions. A transform that silently depends on undeclared columns will force developers to widen the pipeline later.

### Do Not Break the Selection-Table Contract

A new transform should follow one of two patterns:

- work entirely on the existing selection table
- declare the extra columns it needs so they become part of the projection

Do not:

- fetch the full dataset inside the transform
- treat layer enrichment as a transform input mechanism
- bypass cache fingerprinting when the transform changes row or column shape

## Adding a New Data Source

New source backends belong in `jarvisplot/data_loader.py`.

A source implementation should provide the same operational contract as `DataSet` does today:

- source fingerprinting for cache invalidation
- compact collect or lazy scan where possible
- stable `__jp_row_idx__`
- dataset summary emission
- `fetch_rows_columns()` support if the backend is expected to participate in late enrichment

Preferred pattern:

```text
source-specific lazy planning -> compact collect -> retained dataset table
```

Avoid:

- eager full-table materialization before projection
- backend-specific side channels that bypass `ProjectCache`
- source implementations that cannot support row-id based enrichment but still rely on render-time extra columns

## Existing Repo Examples

- `bin/SUSYRun2_EWMSSM.yaml` uses `load_whitelist: only_in_list`, `isvalid_policy: clean`, and repeated `share_data` names. That file demonstrates why source planning and named cache reuse matter.
- `bin/EggBox_Dynesty_06.yaml` uses `share_data: gridprofXY`, which is a concrete example of cross-layer runtime reuse.
- `grid_profile` in `jarvisplot/Figure/adapters.py` is the best reference for a custom rendering primitive that consumes a compact profiled table instead of the full source dataframe.

## Recommended Verification

This repository does not currently provide an automated test suite, so manual verification matters.

For dataflow changes, run at least:

```bash
jplot path/to/config.yaml --rebuild-cache
JP_MEM_TRACE=1 jplot path/to/config.yaml --rebuild-cache
JP_MEM_TRACE=1 jplot path/to/config.yaml
```

Check:

- output is unchanged or intentionally changed
- cold and warm traces still show narrow pipeline stages
- runtime cache metadata still matches demand fingerprints

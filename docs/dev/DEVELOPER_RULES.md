# Developer Rules

These rules are mandatory for any change that touches data loading, profiling, caching, or layer rendering.

## Rule 1: Do Not Reintroduce Wide-Table Propagation

Full dataset tables must not flow through the runtime pipeline by default.

Required behavior:

- dataset load retains only planned columns plus `__jp_row_idx__`
- runtime transforms run on projected selection tables
- profiling operates on selection tables, not on full source tables

If you add a new transform or layer field that depends on extra columns:

- update column-demand planning in `jarvisplot/core.py`
- update runtime/preprofile projection logic in `jarvisplot/Figure/preprocessor.py`

Do not solve missing-column bugs by passing the entire source dataframe deeper into the pipeline.

## Rule 2: Cache Payloads Must Stay Compact

Cache entries should contain:

- selection-table results
- compact post-transform tables
- named references to existing compact cache entries

Cache entries must not contain:

- raw HDF5 group expansions
- unrelated source columns
- blanket source-column carry-through that is wider than the current layer demand

When extending cache behavior, preserve the compatibility checks based on:

- demand fingerprint
- runtime profile signature
- transform signature

## Rule 3: Layer Enrichment Must Be Lazy

Missing render-only columns must be fetched on demand by `__jp_row_idx__`.

Required behavior:

- rendering asks for columns through layer coordinate/style expressions
- current-layer demand may already be present in the projected payload
- `DataPreprocessor._enrich_for_demand()` resolves the missing set
- `DataSet.fetch_rows_columns()` backfills only those columns

Do not preload full source tables for labels, markers, colors, or style expressions.

## Rule 4: Preserve the pandas Conversion Boundary

Avoid converting to pandas earlier than necessary.

Preferred flow:

```text
polars lazy -> compact collect -> pandas -> render
```

Allowed pushdown today:

- `filter`
- `sortby`
- `add_column`

Current non-pushdown stages:

- `profile`
- `grid_profile`
- render-time expression evaluation

If you add a new stage, first ask whether it can stay in lazy polars. If not, make the pandas boundary later, not earlier.

## Rule 5: `__jp_row_idx__` Is a Contract, Not an Implementation Detail

`__jp_row_idx__` is the join key between:

- compact dataset tables
- selection tables
- enriched render tables

Rules:

- do not drop it from retained or projected payloads
- do not repurpose it for user-facing semantics
- if a new source backend is added, it must provide stable row ids before late enrichment is expected to work

## Rule 6: Preprofile Must Stay Reusable

`prebuild_profiles()` exists to reduce repeated runtime work.

Do not break the current split:

- prebuild work depends on source plus profile coordinates
- runtime work keeps the remaining transform tail

In particular:

- do not key preprofile caches on runtime-only knobs such as render `bin` when the base candidate set is unchanged
- do not replace the alias rewrite (`__jp_preprofile_<hash>`) with eager figure-local copies

## Rule 7: New Heavy Stages Need Memory Observability

If a change adds a heavy collect, transform, cache write, or conversion step:

- add or move `memtrace_checkpoint()` calls around it
- add `memtrace_object_inventory()` if the step can hold a large dataframe
- keep log stage names specific enough to identify the boundary later

If memory behavior cannot be explained with `JP_MEM_TRACE=1`, the stage is under-instrumented.

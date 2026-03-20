# Memtrace System

JarvisPLOT includes an opt-in memory diagnostic layer in `jarvisplot/memtrace.py`.

Enable it with:

```bash
JP_MEM_TRACE=1 jplot path/to/config.yaml
```

Accepted truthy values are `1`, `true`, `yes`, `on`, and `debug`.

## What It Records

### RSS and RSS Delta

`memtrace_checkpoint()` records:

- sequence number
- timestamp
- stage name
- current RSS in MB
- RSS delta from the previous checkpoint in the same process

RSS is read from:

- `psutil` when available
- otherwise `resource.getrusage()`

### Object Shape and Backend

For supported objects, checkpoints add shape tokens such as:

- pandas rows/cols
- polars dataframe rows/cols
- polars lazy frame schema width
- list/dict item counts

### Estimated Object Size

`memtrace_object_inventory()` inspects large objects and emits checkpoints for them.

It currently estimates bytes for:

- pandas dataframes via `memory_usage()`
- polars dataframes via `estimated_size()`
- numpy arrays via `nbytes`

By default it only logs objects estimated at `>= 64 MiB`.

### Cache File Size

`memtrace_file_checkpoint()` records file size and file path after cache writes.

## Where Probes Exist

### Dataset Collect and Source Materialization

Main probes:

- `jarvisplot/data_loader.py:polars_to_pandas_compat()`
- `jarvisplot/data_loader.py:_apply_dataset_transform()`
- `jarvisplot/data_loader.py:_apply_dataset_transform_polars()`
- `jarvisplot/data_loader.py:_activate_materialized_manifest()`

Stages include:

- `dataset:<name>.polars_collect.before`
- `dataset:<name>.polars_collect.after`
- `dataset:<name>.pandas_convert.before`
- `dataset:<name>.pandas_convert.after`
- `hdf5.materialized.ready`
- `hdf5.materialized.cache_hit`
- `dataset.transform.before`
- `dataset.transform.after`

### pandas Conversion Boundaries

There are three conversion sites:

- dataset load: `jarvisplot/data_loader.py`
- preprocessing pipeline: `jarvisplot/Figure/preprocessor.py`
- figure/render path: `jarvisplot/Figure/figure.py`

Typical stage names:

- `pipeline.polars_collect.before`
- `pipeline.pandas_convert.before`
- `figure.polars_collect.before`
- `figure.pandas_convert.before`

### Pipeline Stages

`DataPreprocessor.run_pipeline()` and transform helpers emit checkpoints around:

- source resolution
- profile and grid-profile execution
- transform completion
- pipeline return

Examples:

- `pipeline.source_resolved`
- `pipeline.profile.before`
- `pipeline.profile.after`
- `pipeline.grid_profile.before`
- `pipeline.grid_profile.after`
- `pipeline.transform_done`
- `pipeline.return`

`jarvisplot/Figure/load_data.py` also emits profiling-specific checkpoints such as:

- `profile.before`
- `profile.concat_ready`
- `profile.after`
- `grid_profile.before`
- `grid_profile.groupby_ready`
- `grid_profile.after`

### Cache Writes

`jarvisplot/cache_store.py` records memory around:

- dataframe cache writes
- named cache writes
- cache file creation

Examples:

- `cache.put_dataframe.before`
- `cache.put_dataframe.inventory`
- `cache.put_dataframe.after`
- `cache.put_named.before`
- `cache.put_named.after`

## Log Shape

A typical line looks like:

```text
[MEMTRACE] seq=17 ts=... stage=pipeline.transform_done rss_mb=842.13 delta_mb=+61.22 backend=pandas rows=120000 cols=6 source=scanA mode=runtime
```

Interpretation:

- `stage` tells you which boundary moved memory
- `delta_mb` tells you whether the last step expanded or released memory
- backend/rows/cols tell you what object shape crossed that boundary

## How to Use It During Development

1. Run a cold-cache job with `JP_MEM_TRACE=1`.
2. Run the same job again warm-cache.
3. Compare where the large RSS deltas happen.
4. If a new change increases memory, check:
   - whether projection widened
   - whether pandas conversion moved earlier
   - whether a cache payload became wider
   - whether unrelated source columns leaked into the selection table or cache payload

If a new heavy stage does not show up clearly in the trace, add instrumentation before merging the change.

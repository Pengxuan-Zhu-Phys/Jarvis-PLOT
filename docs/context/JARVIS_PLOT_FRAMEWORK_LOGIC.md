# Jarvis-PLOT Framework Logic (Execution Contract)

Status: implemented
Last updated: 2026-07-16

## 1) Execution Layers

Jarvis-PLOT runtime can be viewed as five stacked layers for the YAML figure path,
plus a standalone flowchart path:

1. Orchestration layer (`core.py` + `core_runtime.py` + `core_assets.py` + `config.py` + `cli.py`)
2. Data pipeline layer (`data_loader.py` + `data_loader_runtime.py` + `data_loader_hdf5.py` + `data_loader_summary.py` + `cache_store.py` + `Figure/preprocessor.py` + `Figure/preprocessor_runtime.py` + `Figure/profile_runtime.py` + `Figure/data_pipelines.py`)
3. Figure expansion layer (`Figure/figure_types.py` — expands high-level figure macros before planning)
4. Render layer (`Figure/figure.py` + `Figure/config_runtime.py` + `Figure/layer_runtime.py` + `Figure/adapters_rect.py` + `Figure/adapters_ternary.py` + `Figure/method_registry.py` + `Figure/style_runtime.py` + `Figure/layout_runtime.py` + `Figure/colorbar_runtime.py` + `Figure/dynesty_runtime.py` + `Figure/design_runtime.py` + `Figure/helper.py`)
5. Asset/config layer (`cards/*.json` + `Figure/cards/*.json` + `utils/cmaps.py` + `utils/pathing.py` + `utils/expression.py` + `utils/interpolator.py` + `inner_func.py` + user YAML in `bin/`)

Standalone flowchart path (does not use the YAML figure stack):

- CLI: `jplot flowchart <scene.json>`
- owner: `jarvisplot/flowchart.py`
- card: `jarvisplot/cards/flowchart/default.json`


## 2) Figure Lifecycle Contract

For each figure item in YAML:

1. expand high-level figure types when present (`figure_types.expand_figure_types_in_config`)
2. load style card defaults (`Frame` + `Style`)
3. deep-merge user `frame` overrides
4. create declared axes from `frame.axes`
5. resolve each layer dataset:
  - source read from `DataContext`
  - transform chain via `DataPreprocessor.run_pipeline`
6. evaluate coordinates (`expr` or direct data)
7. dispatch drawing method by registry
8. attach/resolve shared colorbar (`axc`) lazily
9. optional design-reference overlay when enabled
10. finalize + save


## 3) Data Pipeline Contract

### Input Sources

- `source: <dataset_name>`
- `source: [dataset_1, dataset_2, ...]`

### Transform Step Types

- `filter`
- `add_column`
- `sortby`
- `profile`
- `make_density_core`
- `make_interp_2d`
- `keep_columns`
- `drop_columns`
- `to_csv`
- `to_parquet`

Computed columns are expressed with `add_column`; `expression` is not a standalone transform type.

Order is authoritative; transforms run sequentially. `keep_columns` and `drop_columns` are the only explicit column-pruning steps. `to_csv` and `to_parquet` export the dataframe state at their position in the ordered list. The canonical transform reference is `docs/specs/TRANSFORMS.md`.

### Prebuild vs Runtime

- prebuild phase rewrites expensive profile inputs to alias data (`__jp_preprofile_*`)
- runtime phase reuses alias/cache and applies remaining transforms


## 4) Render Dispatch Contract

Method resolution path:

- YAML `layer.method`
- `method_registry.resolve_callable(...)`
- adapter method invocation

Axes-type-sensitive behavior:

- rect: expects `x,y` (+ optional `z/c`)
- ternary: accepts `left,right,bottom` (or projected `x,y`)


## 5) Colorbar Contract

### Preferred

```yaml
frame:
  axc:
    color:
      scale: linear|log
      cmap: ...
      vmin: ...
      vmax: ...
```

### Runtime Rules

- colorbar is created only when at least one colored layer is detected
- `frame.axc.color` has higher priority than layer style color keys
- if `scale: log`, norm resolves to `LogNorm` only when limits are positive
- orientation decides whether ticks/labels live on x-axis or y-axis of `axc`


## 6) Caching Contract

### Storage

- `.cache/data/*.pkl` + `*.json` meta
- `.cache/named/*.pkl` for `share_data`
- `.cache/summary/*.txt`

### Keying Inputs

- source fingerprint (file stat/hash + dataset attributes)
- transform payload
- mode (`runtime` / `preprofile` / `preprofile-base`)
- runtime profile signatures

### Invalidation

- `--rebuild-cache` wipes `.cache`
- metadata mismatch invalidates target cache


## 7) Extension Points

### Add New Draw Method

1. implement method on `StdAxesAdapter`/`TernaryAxesAdapter`
2. register key in `method_registry.py`
3. provide default style entry in relevant card if needed
4. validate rect/tri axes type compatibility

### Add New Expression Function

1. static function injection: `inner_func.update_funcs`
2. dynamic/lazy function: YAML `Functions` + `InterpolatorManager`

### Add New Style Family

1. add card json under `jarvisplot/cards/...`
2. map in `jarvisplot/cards/style_preference.json`
3. ensure `Frame.axes` and `Frame.ax*` schema stay coherent


## 8) Flowchart Lifecycle Contract

CLI form:

```bash
jplot flowchart path/to/scene.json
# optional: -o /path/to/output.png
```

Library form:

```python
from jarvisplot import render_flowchart, render_flowchart_file
```

Runtime steps:

1. load scene JSON
2. validate `schema == jarvisplot.scene/v1` and `scene_type == flowchart`
3. require `layers`, `nodes`, and `edges`
4. merge flowchart style card defaults with optional overrides
5. run classic layout (`_ClassicGraph.layout`)
6. draw modules, files, bridges, variables/selections, and edges
7. save image next to the scene file or to `-o` / explicit output path

This path intentionally bypasses dataset loading, transform prebuild, and `Figure` assembly.


## 9) Practical Performance Notes

- expensive hotspots:
  - HDF5 flatten/merge
  - runtime `profile`
  - dense Voronoi operations (`scipy` + `shapely`)
- safest optimization order:
  1. prefilter data early
  2. cache preprofile outputs
  3. reduce `bin` or triangle density
  4. avoid repeated expression eval for identical columns

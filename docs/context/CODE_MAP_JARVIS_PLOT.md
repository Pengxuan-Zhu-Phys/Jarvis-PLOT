# CODE_MAP_JARVIS_PLOT

## Purpose

This document is the code-oriented companion to `JARVIS_PLOT_CONTEXT.md`.

Use it when you need to answer practical questions such as:

- which module owns a change,
- whether a subsystem is implemented or only specified,
- where to add a new feature without breaking current boundaries.

Read this after `docs/context/JARVIS_PLOT_CONTEXT.md`.

## Current Ownership Model

Jarvis-PLOT currently has four real runtime ownership zones:

1. orchestration and startup
2. data loading and transform/cache pipeline
3. figure runtime and rendering
4. assets, cards, and helper utilities

There is no separate implemented scene parser or layout engine yet. Those are still spec-level boundaries.

## Implemented Owners

### Entry and orchestration [implemented]

- `jarvisplot/client.py`: `main()` entry point that boots `JarvisPLOT`
- `jarvisplot/cli.py`: argparse bootstrap from `jarvisplot/cards/args.json`
- `jarvisplot/core.py`: runtime init, YAML load, dataset registration, prebuild pass, figure loop
- `jarvisplot/core_runtime.py`: project layout, dataset demand planning, usage plan, and YAML rewrite helpers
- `jarvisplot/core_assets.py`: colormap, interpolator, and style bootstrap helpers used by `core.py`
- `jarvisplot/config.py`: YAML path bookkeeping and dataset update helper; not a schema validator

### Source loading and dataset shaping [implemented]

- `jarvisplot/data_loader.py`: CSV loading, dataset lifecycle, summary emission, late row/column fetch, HDF5 call-through
- `jarvisplot/data_loader_summary.py`: dataframe summary formatting and HDF5 tree diagnostics
- `jarvisplot/data_loader_runtime.py`: dataset-level transform execution, HDF5 runtime loading/materialization, dataset transform wrappers
- `jarvisplot/data_loader_hdf5.py`: HDF5 whitelist/rename policy, materialization keys/manifests, HDF5 summary helpers
- `jarvisplot/cache_store.py`: workdir-local cache root, dataframe cache, named cache, materialized HDF5 manifest, summaries
- `jarvisplot/Figure/data_pipelines.py`: `SharedContent` and `DataContext`, lazy shared values, usage counts, invalidation

### Transform and profiling pipeline [implemented]

- `jarvisplot/Figure/preprocessor.py`: demand projection, cache identity, preprofile split, named `share_data` persistence
- `jarvisplot/Figure/preprocessor_runtime.py`: source resolution, runtime transform execution, pipeline cache flow
- `jarvisplot/Figure/load_data.py`: primitive transform functions (`filter`, `addcolumn`, `sortby`)
- `jarvisplot/Figure/profile_runtime.py`: `profile` / `grid_profile` implementations and profile prebuild helpers
- `jarvisplot/inner_func.py`: eval namespace injection for expression helpers
- `jarvisplot/utils/interpolator.py`: lazy YAML `Functions` loader and callable registry

### Figure runtime and rendering [implemented]

- `jarvisplot/Figure/figure.py`: axis construction, layer binding, `savefig()`
- `jarvisplot/Figure/config_runtime.py`: figure config ingestion from YAML dictionaries
- `jarvisplot/Figure/layer_runtime.py`: layer data loading, runtime data retention, and render dispatch
- `jarvisplot/Figure/adapters.py`: thin compatibility re-export for axis adapters
- `jarvisplot/Figure/adapters_rect.py`: rectangular-axes drawing primitives, custom `grid_profile` / Voronoi / tripcolor behavior
- `jarvisplot/Figure/adapters_ternary.py`: ternary-axes drawing primitives and ternary render behavior
- `jarvisplot/Figure/method_registry.py`: YAML `method` key to adapter callable resolution
- `jarvisplot/Figure/style_runtime.py`: style family / variant resolution and frame/style bundle selection
- `jarvisplot/Figure/helper.py`: clipping and geometry helpers used by adapters
- `jarvisplot/Figure/layout_runtime.py`: axis-geometry helpers for numbered axes, ticks, and endpoint application
- `jarvisplot/Figure/colorbar_runtime.py`: colorbar assembly helpers and frame-driven colorbar config lookup

### Style, assets, and shared utilities [implemented]

- `jarvisplot/cards/**`: style bundles, CLI arg metadata, color maps, icons
- `jarvisplot/Figure/cards/**`: adapter-specific config
- `jarvisplot/utils/pathing.py`: repo-root and workdir-relative path resolution helper
- `jarvisplot/utils/cmaps.py`: colormap registration and lookup
- `jarvisplot/utils/expression.py`: shared dataframe-expression evaluation helper

## Partial Or Mixed Ownership

These modules are real owners, but they still mix concerns that should stay separated over time.

- `jarvisplot/core.py`: orchestration plus column-demand planning and usage planning
- `jarvisplot/core_assets.py`: colormap, interpolator, and style bootstrap helpers used by `core.py`
- `jarvisplot/data_loader.py`: CSV loading, dataset lifecycle, summary emission, late row/column fetch, HDF5 call-through
- `jarvisplot/data_loader_summary.py`: summary formatting and HDF5 tree diagnostics helper
- `jarvisplot/data_loader_runtime.py`: runtime HDF5 loading/materialization plus dataset transform execution
- `jarvisplot/data_loader_hdf5.py`: HDF5 policy helpers plus materialized manifest helpers
- `jarvisplot/Figure/figure.py`: config ingestion, axes building, layer runtime, colorbar coordination, and backend dispatch in one class
- `jarvisplot/Figure/preprocessor.py`: transform projection, cache compatibility, and preprofile rewriting
- `jarvisplot/Figure/preprocessor_runtime.py`: runtime execution and source resolution helpers
- `jarvisplot/Figure/load_data.py`: transform primitives only
- `jarvisplot/Figure/profile_runtime.py`: profiling algorithms and preprofiling helpers
- `jarvisplot/Figure/adapters.py`: compatibility layer that still groups the rect/ternary adapter entry points
- `jarvisplot/Figure/adapters_rect.py` and `jarvisplot/Figure/adapters_ternary.py`: adapter-family owners, but still share helper utilities
- `jarvisplot/config.py`: config state holder, not a validator or schema owner

Use these modules carefully. They are the current implementation, but they are not ideal long-term boundaries.

## Spec Only / Missing Code Owner

These concepts exist in docs, but they do not yet have a dedicated runtime owner in `jarvisplot/`.

- semantic scene parsing / normalization from `docs/specs/SCENE_JSON_SCHEMA.md`
- explicit layout engine ownership
- explicit style schema ownership
- explicit profile schema ownership
- layer type registry as a first-class runtime contract
- flowchart node / edge / port runtime model

Track the remaining implementation work for these missing owners in `docs/roadmap/IMPLEMENTATION_ROADMAP.md`.

Do not hide these concerns inside `figure.py` or `adapters.py` when implementing future flowchart support.

## Where To Put Common Changes

- new CLI flag or argument parsing change -> `jarvisplot/cli.py` and `jarvisplot/cards/args.json`
- new data source backend -> `jarvisplot/data_loader.py` and `jarvisplot/cache_store.py`
- new summary formatting or HDF5 tree diagnostic helper -> `jarvisplot/data_loader_summary.py`
- new HDF5 policy helper -> `jarvisplot/data_loader_hdf5.py` and `jarvisplot/data_loader_runtime.py`
- new dataset transform/runtime helper -> `jarvisplot/data_loader_runtime.py` and `jarvisplot/data_loader.py`
- new transform primitive -> `jarvisplot/Figure/load_data.py` and `jarvisplot/Figure/preprocessor_runtime.py`
- new profile helper -> `jarvisplot/Figure/profile_runtime.py` and `jarvisplot/Figure/preprocessor_runtime.py`
- new pipeline/runtime helper -> `jarvisplot/Figure/preprocessor_runtime.py` and `jarvisplot/Figure/preprocessor.py`
- new render primitive -> `jarvisplot/Figure/adapters_rect.py`, `jarvisplot/Figure/adapters_ternary.py`, and `jarvisplot/Figure/method_registry.py`
- new style bundle or asset -> `jarvisplot/cards/**`, `jarvisplot/utils/cmaps.py`, and if needed `jarvisplot/core.py`
- new shared-data behavior -> `jarvisplot/Figure/data_pipelines.py` and `jarvisplot/Figure/preprocessor.py`
- new expression helper -> `jarvisplot/inner_func.py`, `jarvisplot/utils/interpolator.py`, and `jarvisplot/utils/expression.py`
- new path-resolution helper -> `jarvisplot/utils/pathing.py`
- future semantic scene / flowchart runtime -> new owner module, not `figure.py`

## Boundary Warnings

- `figure.py` is the render/runtime owner, not a scene parser.
- `data_loader.py` is a source loader, not a layout engine.
- `config.py` is a config holder, not a schema validator.
- `core.py` is orchestration, not the long-term owner of semantic scene contracts.

## Flowchart Migration Note

The upcoming Jarvis-HEP flowchart export should land in a new semantic-scene owner before it reaches rendering.

Current code can consume figure/layer configs and data pipeline inputs.
It does not yet own a real semantic graph-to-layout pipeline.

That means:

- Jarvis-HEP should export semantic scene JSON only
- Jarvis-PLOT should eventually own parsing, layout, sizing, routing, styling, and rendering
- the missing scene/layout owner should be added before the feature grows further

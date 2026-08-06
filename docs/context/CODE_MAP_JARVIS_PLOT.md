# CODE_MAP_JARVIS_PLOT

Last updated: 2026-07-16

## Purpose

This document is the code-oriented companion to `JARVIS_PLOT_CONTEXT.md`.

Use it when you need to answer practical questions such as:

- which module owns a change,
- whether a subsystem is implemented or only specified,
- where to add a new feature without breaking current boundaries.

Read this after `docs/context/JARVIS_PLOT_CONTEXT.md`.

## Current Ownership Model

Jarvis-PLOT currently has five real runtime ownership zones:

1. orchestration and startup
2. data loading and transform/cache pipeline
3. figure runtime and rendering
4. flowchart scene rendering (standalone path)
5. assets, cards, and helper utilities

Figure YAML layout is still implicit in the figure/adapters stack.
Flowchart layout and rendering live in `jarvisplot/flowchart.py` and do not go through `Figure/figure.py`.

## Implemented Owners

### Entry and orchestration [implemented]

- `jarvisplot/client.py`: `main()` entry point that boots `JarvisPLOT`
- `jarvisplot/cli.py`: argparse bootstrap from `jarvisplot/cards/args.json`
- `jarvisplot/core.py`: runtime init, YAML load, dataset registration, prebuild pass, figure loop; also short-circuits to flowchart mode via `jplot flowchart <scene.json>`
- `jarvisplot/core_runtime.py`: project layout, expression analysis helpers (`_expr_symbols`, `_collect_expr_columns`, `_transform_columns`, etc.), dataset demand planning, usage plan, YAML rewrite helpers, and figure-type expansion hook
- `jarvisplot/core_assets.py`: colormap, interpolator, and style bootstrap helpers used by `core.py`
- `jarvisplot/config.py`: YAML path bookkeeping and dataset update helper; not a schema validator
- `jarvisplot/memtrace.py`: opt-in memory tracing (`JP_MEM_TRACE`), RSS checkpoints, dataframe inventories

### Source loading and dataset shaping [implemented]

- `jarvisplot/data_loader.py`: CSV loading, dataset lifecycle, summary emission, late row/column fetch, HDF5 call-through
- `jarvisplot/data_loader_summary.py`: dataframe summary formatting and HDF5 tree diagnostics
- `jarvisplot/data_loader_runtime.py`: dataset-level transform execution, HDF5 runtime loading/materialization, dataset transform wrappers
- `jarvisplot/data_loader_hdf5.py`: HDF5 whitelist/rename policy, materialization keys/manifests, HDF5 summary helpers
- `jarvisplot/cache_store.py`: workdir-local cache root, dataframe cache, named cache, materialized HDF5 manifest, summaries
- `jarvisplot/Figure/data_pipelines.py`: support-layer `SharedContent` / `DataContext` only — lazy shared values, usage counts, invalidation; not a transform or render owner

### Transform and profiling pipeline [implemented]

- `jarvisplot/Figure/preprocessor.py`: demand projection, cache identity, preprofile split, named `share_data` persistence
- `jarvisplot/Figure/preprocessor_runtime.py`: source resolution, ordered transform pipeline execution, and primitive transforms (`filter_df`, `add_column`, `sort_by`, `keep_columns`, `drop_columns`, CSV/Parquet export)
- `jarvisplot/Figure/profile_runtime.py`: `profile` implementations and profile prebuild helpers
- `jarvisplot/Figure/density_cell_runtime.py`: `make_density_core` support/core construction for posterior mass tables
- `jarvisplot/Figure/interp_2d_runtime.py`: `make_interp_2d` support/core-to-grid interpolation
- `jarvisplot/Figure/posterior_density_runtime.py`: posterior density reconstruction helpers used by density transforms
- `jarvisplot/Figure/posterior_mesh.py`: mesh construction for posterior / density geometry
- `jarvisplot/Figure/posterior_hpd.py`: integrated-mass HPD contour threshold computation and contour style preparation
- `jarvisplot/Figure/interp_natural_neighbor.py` / `interp_natural_neighbor_exact.py`: natural-neighbor interpolation backends
- `jarvisplot/inner_func.py`: eval namespace injection for expression helpers
- `jarvisplot/utils/interpolator.py`: lazy YAML `Functions` loader and callable registry

### Figure runtime and rendering [implemented]

- `jarvisplot/Figure/figure.py`: axis construction, layer binding, coordinate evaluation, design-reference overlay hook, `savefig()`
- `jarvisplot/Figure/figure_types.py`: expands high-level figure types (for example profile/posterior 2D macros) into concrete layer configs before planning
- `jarvisplot/Figure/config_runtime.py`: figure config ingestion from YAML dictionaries, style bundle resolution, `rcParams` setup
- `jarvisplot/Figure/layer_runtime.py`: layer data loading, style merge, coordinate validation, expression evaluation, clip_expr support, and render dispatch to adapters
- `jarvisplot/Figure/dynesty_runtime.py`: specialized dynesty `runplot` renderer and default panel semantics
- `jarvisplot/Figure/adapters_rect.py`: rectangular-axes drawing primitives, custom `pcolormesh` / Voronoi / tripcolor behavior
- `jarvisplot/Figure/adapters_ternary.py`: ternary-axes drawing primitives and ternary render behavior
- `jarvisplot/Figure/method_registry.py`: YAML `method` key to adapter callable resolution
- `jarvisplot/Figure/style_runtime.py`: style family / variant resolution and frame/style bundle selection
- `jarvisplot/Figure/helper.py`: clipping and geometry helpers used by adapters
- `jarvisplot/Figure/layout_runtime.py`: axis-geometry helpers for numbered axes, ticks, and endpoint application
- `jarvisplot/Figure/colorbar_runtime.py`: colorbar assembly and frame-driven colorbar config (`frame.axc.color`); accumulates shared state across layers via `fig.axc._cb`
- `jarvisplot/Figure/design_runtime.py`: optional design-reference debug overlay for style cards

### Flowchart scene path [implemented / partial]

- `jarvisplot/flowchart.py`: semantic flowchart scene validation, classic layout (`_ClassicGraph`), and Matplotlib rendering
- public API: `render_flowchart`, `render_flowchart_file` (also re-exported from `jarvisplot/__init__.py`)
- CLI: `jplot flowchart <scene.json>` short-circuits in `core.py` before the YAML figure pipeline
- style card: `jarvisplot/cards/flowchart/default.json`
- contract doc: `docs/specs/SCENE_JSON_SCHEMA.md` (partial; flowchart subset is consumed in code)

Current scope is the Jarvis-HEP classic flowchart grammar (`schema: jarvisplot.scene/v1`, `scene_type: flowchart`).
It is **not** a general-purpose scene/layout engine for all future diagram types.

### Style, assets, and shared utilities [implemented]

- `jarvisplot/cards/**`: style bundles, CLI arg metadata, color maps, icons, flowchart card
- `jarvisplot/Figure/cards/**`: adapter-specific config
- `jarvisplot/utils/pathing.py`: repo-root and workdir-relative path resolution helper
- `jarvisplot/utils/cmaps.py`: colormap registration and lookup
- `jarvisplot/utils/expression.py`: shared dataframe-expression evaluation helper
- `jarvisplot/utils/dataframes.py`: dataframe conversion helpers (including polars paths)

## Partial Or Mixed Ownership

These modules are real owners, but they still mix concerns that should stay separated over time.

- `jarvisplot/core.py`: orchestration plus flowchart short-circuit and figure loop glue
- `jarvisplot/Figure/figure.py`: config ingestion, axes building, layer runtime coordination, colorbar coordination, and backend dispatch in one class
- `jarvisplot/Figure/preprocessor.py`: transform projection, cache compatibility, and preprofile rewriting
- `jarvisplot/Figure/preprocessor_runtime.py`: runtime execution plus transform primitives
- `jarvisplot/Figure/data_pipelines.py`: support-layer shared storage; planning still lives in `core_runtime`, persistence in preprocessor/layer runtime
- `jarvisplot/flowchart.py`: scene validation, layout, and rendering still live in one module
- `jarvisplot/config.py`: config state holder, not a validator or schema owner

Use these modules carefully. They are the current implementation, but they are not ideal long-term boundaries.

## Spec Only / Missing Code Owner

These concepts exist in docs, but they do not yet have a dedicated runtime owner beyond the partial implementations above.

- general semantic scene parsing / normalization for non-flowchart scene types
- an explicit shared layout engine for figure panels and future diagram types (flowchart layout is local to `flowchart.py`)
- explicit style schema ownership as a first-class validator
- explicit profile schema ownership as a first-class validator
- layer type registry as a first-class runtime contract (registry code exists; schema doc is still aspirational)
- Agent Data API verbs (`docs/specs/AGENT_DATA_API.md`)

Track remaining work in `docs/roadmap/IMPLEMENTATION_ROADMAP.md`.

Do not hide new general scene/layout ownership inside `figure.py` or adapter modules.

## Where To Put Common Changes

- new CLI flag or argument parsing change -> `jarvisplot/cli.py` and `jarvisplot/cards/args.json`
- new data source backend -> `jarvisplot/data_loader.py` and `jarvisplot/cache_store.py`
- new summary formatting or HDF5 tree diagnostic helper -> `jarvisplot/data_loader_summary.py`
- new HDF5 policy helper -> `jarvisplot/data_loader_hdf5.py` and `jarvisplot/data_loader_runtime.py`
- new dataset transform/runtime helper -> `jarvisplot/data_loader_runtime.py` and `jarvisplot/data_loader.py`
- new transform primitive -> focused runtime helper under `jarvisplot/Figure/`, plus dispatch/projection updates in `jarvisplot/Figure/preprocessor_runtime.py`, `jarvisplot/Figure/preprocessor.py`, `jarvisplot/data_loader_runtime.py`, and `jarvisplot/core_runtime.py`
- new posterior-density or HPD behavior -> keep the measure estimator independent from rendering; document it in `docs/specs/POSTERIOR_DENSITY.md`
- new profile helper -> `jarvisplot/Figure/profile_runtime.py` and `jarvisplot/Figure/preprocessor_runtime.py`
- new pipeline/runtime helper -> `jarvisplot/Figure/preprocessor_runtime.py` and `jarvisplot/Figure/preprocessor.py`
- new render primitive -> `jarvisplot/Figure/adapters_rect.py`, `jarvisplot/Figure/adapters_ternary.py`, and `jarvisplot/Figure/method_registry.py`
- new multi-axes domain renderer -> a focused runtime helper under `jarvisplot/Figure/`, `jarvisplot/Figure/method_registry.py`, and a style card under `jarvisplot/cards/**`
- new figure-type macro / expander -> `jarvisplot/Figure/figure_types.py` and `jarvisplot/core_runtime.py`
- new style bundle or asset -> `jarvisplot/cards/**`, `jarvisplot/utils/cmaps.py`, and if needed `jarvisplot/core_assets.py`
- new shared-data behavior -> `jarvisplot/Figure/data_pipelines.py` and `jarvisplot/Figure/preprocessor.py`
- new expression helper -> `jarvisplot/inner_func.py`, `jarvisplot/utils/interpolator.py`, and `jarvisplot/utils/expression.py`
- new path-resolution helper -> `jarvisplot/utils/pathing.py`
- flowchart scene grammar / classic layout / rendering -> `jarvisplot/flowchart.py` and `jarvisplot/cards/flowchart/`
- Agent Data API verbs -> new thin CLI/envelope layer on top of existing loaders and transform runtimes; do not fork the pipeline

## Known Boundary Issues

These are documented architectural issues that should be addressed over time:

1. **Grid metadata coupling**: `profile_runtime.py` writes `__grid_*` columns that `adapters_rect.py` reads and reconstructs for `pcolormesh`. If profile column names change, adapters must change in sync.

2. **Colorbar state accumulation**: `colorbar_runtime.collect_and_attach_colorbar()` mutates shared `fig.axc._cb` across layers to track the union of all color ranges. This is intentional but should stay documented.

3. **SharedContent usage plan**: `consume()` invalidates cached values at zero remaining uses but keeps the registry entry, so a later `get()` recomputes. This is intentional support-layer behavior documented in `data_pipelines.py`.

## Boundary Warnings

- `figure.py` is the YAML figure render/runtime owner, not the flowchart scene parser.
- `flowchart.py` owns flowchart scene intake, classic layout, and rendering for that path only.
- `data_loader.py` is a source loader, not a layout engine.
- `config.py` is a config holder, not a schema validator.
- `core.py` is orchestration; it may short-circuit to flowchart mode, but it should not absorb scene grammar.

## Flowchart Status Note

Jarvis-HEP flowchart export is already consumable:

- semantic scene JSON with `schema: jarvisplot.scene/v1` and `scene_type: flowchart`
- API: `render_flowchart` / `render_flowchart_file`
- CLI: `jplot flowchart path/to/scene.json`

Jarvis-PLOT owns:

- scene validation for the supported flowchart subset
- classic layout (coordinates, sizes, routing for the current visual grammar)
- style card application
- PNG/PDF-style file output via Matplotlib

Jarvis-HEP should still export semantic graph information only, not final geometry.
Future general diagram types should extend a clearer scene/layout split rather than growing ad hoc paths inside `figure.py`.

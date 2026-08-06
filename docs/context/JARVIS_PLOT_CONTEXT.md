# Project Context For Codex

Last updated: 2026-07-16
Audience: Codex, maintainers, and contributors making implementation changes.
Status: implemented
Document role: primary pre-read for Jarvis-PLOT changes.

## 1. Mandatory Read Order

Read these files before changing Jarvis-PLOT:

1. `docs/context/JARVIS_PLOT_CONTEXT.md`
2. `docs/context/CODE_MAP_JARVIS_PLOT.md`
3. `docs/context/JARVIS_PLOT_FRAMEWORK_LOGIC.md`
4. `docs/design/ARCHITECTURE_OVERVIEW.md`
5. `docs/design/DATAFLOW_ARCHITECTURE.md`
6. `docs/design/LAYOUT_ENGINE_DESIGN.md`
7. `docs/design/STYLE_SYSTEM_DESIGN.md`
8. `docs/design/PROFILE_SYSTEM_DESIGN.md`
9. `docs/specs/SCENE_JSON_SCHEMA.md`
10. `docs/roadmap/IMPLEMENTATION_ROADMAP.md`

Use this file first for current boundaries and ownership.
Use the design and schema docs only after the current boundary is clear.

## 2. Status Legend

- implemented: supported by current code
- partial: real code exists, but the boundary is still mixed
- spec only: described in docs, but no stable code owner yet

## 3. Current Project Shape

Jarvis-PLOT is a plotting, scene, and layout framework.

Current stable product facts:

- package / product version: **1.4.2** (`pyproject.toml`; distribution name `JarvisPLOT`, import package `jarvisplot`)
- CLI entry point: `jplot`
- primary package path: `jarvisplot/`
- primary orchestrator: `jarvisplot/core.py`
- current main renderer: Matplotlib-based figure rendering
- input surfaces:
  - YAML figure configs, style cards, dataset configs (**implemented**)
  - flowchart scene JSON via `jplot flowchart <scene.json>` or `render_flowchart*` (**implemented**, classic flowchart subset)
  - general semantic scene types beyond flowchart (**partial / mostly spec**)
- current data pipeline transforms: `filter`, `add_column`, `sortby`, `profile`, `make_density_core`, `make_interp_2d`, `keep_columns`, `drop_columns`, `to_csv`, `to_parquet`
- computed columns use `add_column`; there is no standalone `type: expression` transform
- expensive profile work is split across prebuild and runtime phases
- high-level figure macros expand through `jarvisplot/Figure/figure_types.py` before planning
- transform primitives live in `jarvisplot/Figure/preprocessor_runtime.py`; profiling helpers live in `jarvisplot/Figure/profile_runtime.py`
- dataset summary helpers live in `jarvisplot/data_loader_summary.py`
- dataset-level runtime helpers live in `jarvisplot/data_loader_runtime.py`; pipeline runtime helpers live in `jarvisplot/Figure/preprocessor_runtime.py`
- flowchart runtime lives in `jarvisplot/flowchart.py` (standalone path; not the YAML figure stack)
- runtime artifacts: output images plus workdir-local cache under `.cache/` for the YAML figure pipeline

Treat the project as a framework that converts semantic plotting or diagram input into final rendered output.
Do not reduce it to a bag of plotting helpers.

## 4. Explicit Project Boundary

Jarvis-PLOT owns:

- scene parsing and normalization for supported scene types (currently flowchart)
- layout computation for supported paths (figure axes geometry + flowchart classic layout)
- style and profile application
- renderer dispatch and final output generation
- cache behavior needed to support the YAML figure pipeline

Jarvis-PLOT does not own:

- Jarvis-HEP workflow semantics
- physics-domain graph construction logic
- calculator or sampler runtime concerns
- business logic that decides what semantic entities exist upstream

Boundary with Jarvis-HEP:

- Jarvis-HEP should export semantic graph or scene information only.
- Jarvis-PLOT should consume that semantic description and produce the visual result.
- If a change is about workflow meaning, node identity, or dependency semantics, it belongs in Jarvis-HEP.
- If a change is about coordinates, sizes, routing, style, layout, or rendering, it belongs in Jarvis-PLOT.

## 5. Internal Conceptual Stack

Think in this order when reading or changing the code:

1. semantic scene input
2. layout engine
3. style and profile system
4. renderer and output backends

Current ownership snapshot:

- semantic scene / config input:
  - YAML path: `jarvisplot/config.py`, `jarvisplot/data_loader.py`, `jarvisplot/data_loader_summary.py`, `jarvisplot/data_loader_runtime.py`, `jarvisplot/Figure/preprocessor_runtime.py`, `jarvisplot/Figure/figure_types.py`
  - flowchart path: `jarvisplot/flowchart.py`
  - contract: `docs/specs/SCENE_JSON_SCHEMA.md` (partial)
- layout:
  - figure axes / panel geometry: `jarvisplot/Figure/figure.py`, `jarvisplot/Figure/layout_runtime.py`, `jarvisplot/Figure/helper.py`
  - flowchart classic layout: `jarvisplot/flowchart.py` (`_ClassicGraph`)
  - design note: `docs/design/LAYOUT_ENGINE_DESIGN.md` (partial)
- style and profile system:
  - `jarvisplot/cards/**`
  - `jarvisplot/core_assets.py`
  - `jarvisplot/Figure/preprocessor.py`
  - `jarvisplot/Figure/preprocessor_runtime.py`
  - `jarvisplot/Figure/profile_runtime.py`
  - `jarvisplot/Figure/style_runtime.py`
  - `jarvisplot/utils/cmaps.py`
  - `docs/design/STYLE_SYSTEM_DESIGN.md`
  - `docs/design/PROFILE_SYSTEM_DESIGN.md`
- renderer and output backends:
  - YAML figures: `jarvisplot/Figure/figure.py`, `jarvisplot/Figure/config_runtime.py`, `jarvisplot/Figure/layer_runtime.py`, `jarvisplot/Figure/adapters_rect.py`, `jarvisplot/Figure/adapters_ternary.py`, `jarvisplot/Figure/method_registry.py`, `jarvisplot/Figure/colorbar_runtime.py`, `jarvisplot/Figure/dynesty_runtime.py`, `jarvisplot/Figure/design_runtime.py`
  - flowchart: `jarvisplot/flowchart.py`

Default debugging order:

1. validate semantic input
2. validate layout decisions
3. validate style/profile resolution
4. validate renderer behavior

## 6. Runtime Ownership Snapshot

### YAML figure path

1. `jarvisplot/client.py` and `jarvisplot/cli.py` parse CLI input
2. `jarvisplot/core.py` initializes project state, workdir, cache, datasets, and style assets
3. `jarvisplot/Figure/figure_types.py` expands high-level figure macros when present
4. `jarvisplot/data_loader.py` registers or loads source datasets
5. `jarvisplot/data_loader_summary.py` formats summary text and HDF5 tree diagnostics
6. `jarvisplot/data_loader_runtime.py` and `jarvisplot/Figure/preprocessor_runtime.py` execute runtime transform and pipeline logic
7. `jarvisplot/Figure/preprocessor.py` prepares transform and profile pipelines
8. `jarvisplot/Figure/figure.py` builds axes and scene state for each figure
9. `jarvisplot/Figure/method_registry.py` resolves draw methods
10. `jarvisplot/Figure/adapters_rect.py` and `jarvisplot/Figure/adapters_ternary.py` execute backend-specific drawing
11. output is written through figure save/render paths

### Flowchart path

1. CLI form: `jplot flowchart path/to/scene.json`
2. `core.py` detects the flowchart command and skips the YAML figure pipeline
3. `jarvisplot/flowchart.py` validates the scene, runs classic layout, and renders the image
4. optional library use: `from jarvisplot import render_flowchart, render_flowchart_file`

## 7. Flowchart Context

Jarvis-PLOT is the rendering target for Jarvis-HEP flowchart export.

**Current status: implemented for the classic flowchart grammar.**

Supported contract:

- Jarvis-HEP exports semantic flowchart JSON (`schema: jarvisplot.scene/v1`, `scene_type: flowchart`).
- Jarvis-PLOT consumes that JSON as scene input.
- Jarvis-PLOT computes node coordinates, box sizes, edge routing, and final rendering in `flowchart.py`.

Jarvis-HEP should not emit:

- final x/y coordinates
- final node widths or heights from text measurement
- renderer-specific image paths or style tokens
- final edge curve geometry

Jarvis-PLOT owns for this path:

- coordinate placement
- size calculation
- routing strategy
- theme/style card selection (`cards/flowchart/`)
- backend-specific rendering to PNG (and related Matplotlib outputs)

For flowcharts, treat the upstream payload as semantic scene data, not pre-rendered geometry.
A general multi-diagram layout engine is still future work; do not force every diagram type through `figure.py`.

## 8. Active Engineering Rules

- Keep ownership explicit across the semantic input -> layout -> style -> render stack.
- Prefer extending existing runtime owners before adding new framework layers.
- Preserve the current prebuild/runtime pipeline split when changing `profile` behavior.
- Cache changes must keep `.cache/data`, `.cache/named`, and `.cache/summary` semantics coherent.
- `frame.axc.color` is the preferred colorbar source of truth; layer-level color keys remain layer kwargs, not colorbar configuration.
- Style keys should be explicit and backed by card definitions; do not invent implicit defaults.
- Do not add new unsafe eval surfaces; existing eval paths are already a technical debt area.
- If transform semantics change, update cache identity assumptions in the same change.
- Update docs in the same change cycle when a boundary or runtime contract moves.
- Keep flowchart changes in `flowchart.py` / flowchart cards unless a change is intentionally promoting a shared scene/layout owner.

## 9. Practical Patch Guidance

When deciding where a patch belongs:

- data import, semantic normalization, or scene schema intake -> semantic input layer
- box placement, panel arrangement, axis geometry, or graph routing -> layout layer
- theme selection, profile behavior, or card defaults -> style/profile layer
- draw primitive behavior or file output differences -> renderer/backend layer
- agent-facing JSON verbs / digests -> Agent Data API layer on top of existing pipeline owners

If a patch crosses multiple layers, define the ownership split first and keep the interfaces narrow.

## 10. Open Work

The active project backlog lives in `docs/roadmap/IMPLEMENTATION_ROADMAP.md`.

Use that file for:

- remaining boundary cleanup (`data_pipelines` lifecycle, general scene/layout split)
- Agent Data API implementation
- longer-horizon v2.0 architecture work (`docs/roadmap/soft-cooking-wilkinson.md`)
- validation and test work that should not be mixed into this boundary doc

Keep this context doc focused on current state, ownership, and boundaries.

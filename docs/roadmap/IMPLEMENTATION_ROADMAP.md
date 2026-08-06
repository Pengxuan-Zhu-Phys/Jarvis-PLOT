# Jarvis-PLOT Implementation Roadmap

Status: active backlog
Last updated: 2026-07-16

This document is the project-wide task list after the docs-alignment pass for **v1.4.2**.

- Release notes / historical release tasks live under `docs/release/releases/`.
- Longer-horizon v2.0 architecture work lives in `docs/roadmap/soft-cooking-wilkinson.md`.
- This roadmap tracks remaining 1.x implementation work that should not be hidden inside boundary docs.

## 1. Current Code Review Summary

The current codebase is functional. The main engineering risks are mixed ownership and unfinished agent-facing surfaces, not missing core plotting features.

Key observations aligned with the current tree:

- package version is **1.4.2** (`pyproject.toml` distribution name `JarvisPLOT`; import package `jarvisplot`)
- `jarvisplot/Figure/figure.py` remains a large mixed-responsibility module, but many helpers already live in focused runtimes
- `jarvisplot/Figure/preprocessor.py` owns projection, cache identity, and prebuild rewrite policy; runtime execution lives in `preprocessor_runtime.py`
- `jarvisplot/data_loader.py` is mostly source-loading and lifecycle wiring; HDF5/runtime/summary helpers are split out
- `jarvisplot/core.py` delegates planning/layout policy to `core_runtime.py` and short-circuits flowchart mode
- `jarvisplot/flowchart.py` is an implemented classic flowchart scene path (not spec-only)
- path resolution is centralized in `utils/pathing.py`
- expression evaluation is centralized in `utils/expression.py`, with a remaining local `clip_expr` eval surface in `layer_runtime.py`
- adapter-family logic lives in `adapters_rect.py` and `adapters_ternary.py`
- Agent Data API remains **spec only**

Refactor priority is therefore:

1. keep ownership docs truthful,
2. finish agent channel work without forking the pipeline,
3. narrow remaining mixed owners before a v2.0 restructure.

## 2. Closed Documentation Alignment Work

These tasks are already done and should stay closed unless a future change reopens them.

- updated the root `README.md` and `docs/README.md` entry points
- rewrote `docs/context/JARVIS_PLOT_CONTEXT.md` as the Codex-facing boundary doc
- replaced the placeholder code map with a concrete owner map
- filled the design/spec/template docs with honest status labels
- archived the historical 2026-03-04 code review snapshot
- added release/archive/roadmap index files
- validated the example JSON templates and removed stale doc references
- **2026-07-16**: re-aligned context/code-map/roadmap/specs with flowchart implementation, figure-type expansion, and v1.4.2 product state

## 3. Code Review Backlog

### P0 - correctness and safety

- [x] `jarvisplot/data_loader.py`: fix `DataSet.file` and `DataSet.type` so `None` returns early instead of falling through.
- [x] `jarvisplot/data_loader.py`: finish the HDF5 fallback branch so it either materializes a dataframe or raises an explicit error.
- [x] `jarvisplot/Figure/figure.py` and `jarvisplot/config.py`: remove runtime `print()` calls and replace bare `except:` blocks with logged, bounded fallbacks.
- [x] `jarvisplot/Figure/figure.py`: make style fallback explicit; do not assume a non-existent `default` style card.
- [x] `jarvisplot/Figure/figure.py`, `jarvisplot/Figure/preprocessor_runtime.py`, and `jarvisplot/utils/interpolator.py`: reduce `eval()` surfaces and centralize expression evaluation.
- [x] `jarvisplot/utils/cmaps.py` and `jarvisplot/Figure/figure.py`: keep colormap registration single-sourced and observable.
- [x] `jarvisplot/Figure/layer_runtime.py`: fix the invalid-transform error path in `load_bool_df()` so the failure message itself cannot crash.
- [x] `jarvisplot/Figure/figure.py`, `jarvisplot/data_loader.py`, and `jarvisplot/core.py`: reduce path-resolution duplication and pick one owner for workdir-relative resolution.
- [x] `jarvisplot/utils/expression.py`: constrain the centralized `eval()` surface further, or document the exact trusted-input assumption in the code owner map.
- [x] `jarvisplot/Figure/layer_runtime.py`: route `clip_expr` through shared `eval_scalar_expression`.
- [x] packaging metadata: align product docs, `VERSION`, CLI version lookup, and `jprel` with distribution name `JarvisPLOT` (import `jarvisplot`; product brand Jarvis-PLOT).

### P1 - boundary cleanup

- [x] `jarvisplot/Figure/figure.py`: split the monolith into config ingestion, layout assembly, layer runtime, colorbar manager, and render dispatch.
- [x] `jarvisplot/Figure/preprocessor.py` and `jarvisplot/Figure/preprocessor_runtime.py`: keep runtime transform execution in one owner; remove fallback duplication from `figure.py`.
- [x] `jarvisplot/core.py`: separate orchestration from dataset planning and YAML rewrite policy.
- [x] `jarvisplot/data_loader.py`: narrow the remaining CSV/source lifecycle and summary-emission glue; HDF5 policy/runtime now live in helper modules, and summary formatting now lives in `jarvisplot/data_loader_summary.py`.
- [x] `jarvisplot/Figure/data_pipelines.py`: document and narrow the `share_data` / usage-plan lifecycle so it stays a support layer, not a hidden runtime owner.
- [x] `jarvisplot/Figure/preprocessor_runtime.py`: split transform primitives from profiling helpers into `jarvisplot/Figure/profile_runtime.py`.
- [x] split render primitives by family into `jarvisplot/Figure/adapters_rect.py` and `jarvisplot/Figure/adapters_ternary.py`.

### P1 - flowchart readiness

- [x] Add a first flowchart runtime owner for Jarvis-HEP classic scenes (`jarvisplot/flowchart.py`).
- [x] Keep Jarvis-HEP responsible for semantic graph emission only; Jarvis-PLOT owns classic geometry and rendering for that path.
- [x] Expose library API (`render_flowchart`, `render_flowchart_file`) and CLI short-circuit (`jplot flowchart ...`).
- [ ] Split flowchart scene validation / layout / render ownership if the module continues to grow.
- [ ] Add a general semantic scene parser / normalizer beyond the flowchart subset.
- [ ] Add a shared layout owner for non-flowchart diagram types; do not put those rules in `figure.py` or adapters.

### P2 - validation and tests

- [x] Add smoke tests for style fallback, colorbar scale/limits, and profile cache reuse.
- [x] Add schema validation or a docs-lint step for scene/style/profile templates.
- [x] Add a consistency check that verifies docs status labels (`implemented` / `partial` / `spec only` / `historical`) stay truthful.
- [x] Add a focused regression test for the `load_bool_df()` invalid-transform error path.
- [x] Add a path-resolution regression test that covers `core.py`, `figure.py`, and `data_loader.py` behavior on the same YAML input.
- [x] Add flowchart renderer coverage (`tests/test_flowchart_renderer.py`).
- [x] Add usage-plan / `share_data` lifecycle regression coverage around `data_pipelines.py`.

### Notes on completed code work

The following implementation items are now in place in the current tree:

- `jarvisplot/data_loader.py` returns early on `None` setters and can materialize a single HDF5 dataset when no named group is provided.
- colormap registration is handled through `jarvisplot/utils/cmaps.py` / `core_assets.py`.
- figure config ingestion, layer runtime, layout, colorbar, dynesty, and design-reference helpers live under focused `Figure/*_runtime.py` modules.
- path resolution is centralized in `jarvisplot/utils/pathing.py`.
- expression evaluation is centralized through `jarvisplot/utils/expression.py`.
- profiling, density, interp, and posterior helpers live in dedicated modules.
- adapter-family implementations live in `adapters_rect.py` and `adapters_ternary.py`.
- flowchart classic rendering is implemented and tested.
- figure-type expansion is implemented in `Figure/figure_types.py`.
- smoke tests cover style fallback, colorbar wiring, profile cache reuse, flowchart rendering, and template JSON parsing.
- docs status labels are checked for consistency, including archive placement for historical notes.

## 3.1 Agent Data API Backlog

**Status: frozen** — do not implement until unfrozen by an explicit product decision.

Contract: `docs/specs/AGENT_DATA_API.md` (spec only). Consumer: Jarvis-Agent milestone M4.6
(`Jarvis-Agent/docs/PLOT_TOOLS.md`). All verbs are additive flags on `jplot`; the human YAML
surface does not change.

### P1 - agent channel core (frozen)

- [ ] JP-A1: JSON envelope helper + `--version-json` + `--validate --json` (jsonschema is already a dependency; diagnostics `{level, path, message}`).
- [ ] JP-A2: `--describe --json` wrapping `jarvisplot/data_loader_summary.py` (columns, dtypes, ranges/quantiles, HDF5 tree).
- [ ] JP-A3: `--analyze` headless channel + `likelihood_report` reducer — profile cells via `profile_runtime`, posterior mass via `make_density_core`, HPD thresholds via `posterior_hpd`; NEW: connected-region extraction over cell adjacency; artifacts `cells.parquet` / 1D curves / optional `regions.geojson`; digest budget rules per spec §4.2. Includes `summary_stats`, `top_points`, `interval_report`.

### P2 - agent channel extensions (frozen)

- [ ] JP-A4: `--template` catalog + slot schemas (`source_hint` contract) shared with packaged cards and `docs/templates/`.
- [ ] JP-A5: `--render-json` machine-readable render outcome + `--with-data` per-layer parquet sidecars (reuse `to_parquet`).
- [ ] `compare_report` reducer (two-run region overlap / interval deltas) — extend spec before implementation.

## 4. Refactor Rules of Thumb

Use these constraints when turning the backlog into code:

- do not grow a second flowchart implementation inside `figure.py`
- do not split a module unless the new owner can be named in the code map
- do not keep duplicate path-resolution or transform-fallback logic in multiple owners
- keep `figure.py` as a YAML-figure runtime owner, not a general scene parser
- keep `flowchart.py` as the current flowchart owner until a shared scene/layout package exists
- keep `preprocessor.py` as the transform policy owner, not a sidecar to `figure.py`
- keep `data_loader.py` as a source-loading owner, not a layout or render owner
- keep `utils/pathing.py` as the single source of path resolution semantics
- keep Agent Data API as a thin skin over existing loaders/transforms/cache, never a pipeline fork

## 5. Integrity Check

This is the final step for any future docs pass.

Before closing a docs-alignment change, verify:

- entry docs point to real files
- no active README/index file is empty
- context docs only describe current ownership
- spec docs are labeled `spec only` when no code owner exists, or `partial` / `implemented` when code exists
- historical notes are archived or clearly marked historical
- roadmap items are either open, done, or moved to a release-specific / v2.0 plan
- templates match the spec language they claim to represent

## 6. Next Task Slice

Completed in the post-docs 1.x cleanup pass:

1. ~~`clip_expr` shared expression path~~ done
2. ~~`data_pipelines` lifecycle docs + regression tests~~ done
3. ~~packaging metadata alignment (`JarvisPLOT` / `jprel` / `VERSION`)~~ done

Remaining optional 1.x work (Agent API stays frozen):

1. split flowchart validation / layout / render only if `flowchart.py` keeps growing
2. general semantic scene / shared layout ownership for non-flowchart diagram types
3. longer-horizon v2.0 restructure remains in `soft-cooking-wilkinson.md`

## 7. Frozen Tracks

- **Agent Data API (JP-A1…A5)**: frozen; keep the spec, do not implement until unfrozen.


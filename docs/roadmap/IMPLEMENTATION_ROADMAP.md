# Jarvis-PLOT Implementation Roadmap

Status: active backlog
Last updated: 2026-03-25

This document is the project-wide task list after the docs-alignment pass.

- Release-specific planning still lives in `docs/release/releases/v1.2.6.md`.
- This roadmap tracks the broader implementation work that should not be hidden inside boundary docs.

## 1. Current Code Review Summary

The current codebase is functional, but the main engineering risk is still mixed ownership, not missing features.

Key observations from the latest review:

- `jarvisplot/Figure/figure.py` is still the largest mixed-responsibility module.
- `jarvisplot/Figure/preprocessor.py` now owns projection, cache identity, and prebuild rewrite policy; runtime execution lives in `jarvisplot/Figure/preprocessor_runtime.py`.
- `jarvisplot/data_loader.py` is now mostly source-loading and lifecycle wiring; runtime HDF5 loading/materialization lives in `jarvisplot/data_loader_runtime.py`, and policy helpers live in `jarvisplot/data_loader_hdf5.py`.
- `jarvisplot/data_loader_summary.py` now owns dataframe summary formatting and HDF5 tree diagnostics.
- `jarvisplot/core.py` now delegates most planning/layout policy to `jarvisplot/core_runtime.py`, but it still owns orchestration.
- path resolution is now centralized, but the owners still need more boundary cleanup.
- `eval()` has been centralized, but it is still a technical debt surface.
- adapter-family logic now lives in `adapters_rect.py` and `adapters_ternary.py`.
- `jarvisplot/Figure/preprocessor_runtime.py` now carries the transform primitives; profiling helpers live in `jarvisplot/Figure/profile_runtime.py`.
- `jarvisplot/data_loader_hdf5.py` now owns HDF5 whitelist/rename policy helpers, while `jarvisplot/data_loader_runtime.py` owns runtime materialization/loading and dataset transform execution.
- `jarvisplot/Figure/preprocessor_runtime.py` now owns runtime source resolution and transform execution.
- flowchart / semantic-scene support is still spec-only and should not be treated as implemented runtime code yet.

Refactor priority is therefore:

1. make ownership boundaries explicit,
2. remove duplicated policy,
3. then split large modules only where the split has a clear owner.

## 2. Closed Documentation Alignment Work

These tasks are already done and should stay closed unless a future change reopens them.

- updated the root `README.md` and `docs/README.md` entry points
- rewrote `docs/context/JARVIS_PLOT_CONTEXT.md` as the Codex-facing boundary doc
- replaced the placeholder code map with a concrete owner map
- filled the design/spec/template docs with honest status labels
- archived the historical 2026-03-04 code review snapshot
- added release/archive/roadmap index files
- validated the example JSON templates and removed stale doc references

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

### P1 - boundary cleanup

- [x] `jarvisplot/Figure/figure.py`: split the monolith into config ingestion, layout assembly, layer runtime, colorbar manager, and render dispatch.
- [x] `jarvisplot/Figure/preprocessor.py` and `jarvisplot/Figure/preprocessor_runtime.py`: keep runtime transform execution in one owner; remove fallback duplication from `figure.py`.
- [x] `jarvisplot/core.py`: separate orchestration from dataset planning and YAML rewrite policy.
- [x] `jarvisplot/data_loader.py`: narrow the remaining CSV/source lifecycle and summary-emission glue; HDF5 policy/runtime now live in helper modules, and summary formatting now lives in `jarvisplot/data_loader_summary.py`.
- [ ] `jarvisplot/Figure/data_pipelines.py`: document and narrow the `share_data` / usage-plan lifecycle so it stays a support layer, not a hidden runtime owner.
- [x] `jarvisplot/Figure/preprocessor_runtime.py`: split transform primitives from profiling helpers into `jarvisplot/Figure/profile_runtime.py`.
- [x] split render primitives by family into `jarvisplot/Figure/adapters_rect.py` and `jarvisplot/Figure/adapters_ternary.py`.

### P1 - flowchart readiness

- [ ] Add a semantic scene parser / normalizer module for the Jarvis-HEP JSON contract.
- [ ] Add a layout owner for coordinates, sizes, and routing; do not put these rules in `figure.py` or adapters.
- [ ] Add a first-class scene runtime model for nodes, edges, and ports.
- [x] Keep Jarvis-HEP responsible for semantic graph emission only; Jarvis-PLOT should own geometry and rendering.

### P2 - validation and tests

- [x] Add smoke tests for style fallback, colorbar scale/limits, and profile cache reuse.
- [x] Add schema validation or a docs-lint step for scene/style/profile templates.
- [x] Add a consistency check that verifies docs status labels (`implemented` / `partial` / `spec only` / `historical`) stay truthful.
- [x] Add a focused regression test for the `load_bool_df()` invalid-transform error path.
- [x] Add a path-resolution regression test that covers `core.py`, `figure.py`, and `data_loader.py` behavior on the same YAML input.

### Notes on completed code work

The following implementation items are now in place in the current tree:

- `jarvisplot/data_loader.py` now returns early on `None` setters and can materialize a single HDF5 dataset when no named group is provided.
- `jarvisplot/Figure/figure.py` and `jarvisplot/config.py` no longer rely on runtime `print()` for normal control flow.
- colormap registration is now handled in `core.py` through `jarvisplot/utils/cmaps.py`; `figure.py` no longer repeats the registration path.
- core asset bootstrap now lives behind `jarvisplot/core_assets.py`, and style bundle resolution now lives behind `jarvisplot/Figure/style_runtime.py`.
- figure config ingestion now lives behind `jarvisplot/Figure/config_runtime.py`.
- figure layer/runtime dispatch now lives behind `jarvisplot/Figure/layer_runtime.py`.
- core orchestration planning now lives behind `jarvisplot/core_runtime.py`.
- path resolution is now centralized in `jarvisplot/utils/pathing.py`.
- expression evaluation is centralized through `jarvisplot/utils/expression.py`.
- profiling algorithms and preprofiling helpers now live in `jarvisplot/Figure/profile_runtime.py`.
- figure runtime helpers for layout and colorbar handling now live in `jarvisplot/Figure/layout_runtime.py` and `jarvisplot/Figure/colorbar_runtime.py`.
- adapter-family implementations now live in `jarvisplot/Figure/adapters_rect.py` and `jarvisplot/Figure/adapters_ternary.py`.
- smoke tests cover style fallback, colorbar wiring, profile cache reuse, and template JSON parsing.
- docs status labels are checked for consistency, including archive placement for historical notes.

## 4. Refactor Rules of Thumb

Use these constraints when turning the backlog into code:

- do not add new flowchart runtime ownership until the current figure/core/data-loader split is clearer
- do not split a module unless the new owner can be named in the code map
- do not keep duplicate path-resolution or transform-fallback logic in multiple owners
- keep `figure.py` as a runtime owner, not a hidden scene parser
- keep `preprocessor.py` as the transform policy owner, not a sidecar to `figure.py`
- keep `data_loader.py` as a source-loading owner, not a layout or render owner
- keep `utils/pathing.py` as the single source of path resolution semantics

## 5. Integrity Check

This is the final step for any future docs pass.

Before closing a docs-alignment change, verify:

- entry docs point to real files
- no active README/index file is empty
- context docs only describe current ownership
- spec docs are labeled `spec only` when no code owner exists
- historical notes are archived or clearly marked historical
- roadmap items are either `open`, `done`, or moved to a release-specific task list
- templates match the spec language they claim to represent

## 6. Next Task Slice

After the current code/doc alignment batch, the next implementation slice should focus on the remaining deeper boundaries:

1. narrow `jarvisplot/Figure/data_pipelines.py` so `share_data` / usage-plan lifecycles stay support-only
2. add the semantic scene / flowchart owner only after the current figure/core/data-loader split is stable
3. add regression tests for the remaining split points above before turning them into release work

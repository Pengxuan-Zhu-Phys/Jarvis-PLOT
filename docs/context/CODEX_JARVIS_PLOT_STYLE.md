# Codex Style for Jarvis-PLOT (Self Policy)

## 0) Goal

This file is my own hard policy for editing Jarvis-PLOT quickly but safely.

Priority order:

1. correctness
2. backward compatibility
3. debuggability
4. performance
5. elegance


## 1) 明令禁止 (MUST NOT)

### 1.1 Bare `except:`

- Forbidden in new code.
- Must use `except Exception as e:` at minimum, with clear fallback behavior.

### 1.2 Silent swallow on core path

- Forbidden to swallow exceptions in:
  - data loading
  - cache compatibility checks
  - render dispatch
  - file writes
- If fallback is intentional, must log reason and fallback target.

### 1.3 `print()` in runtime code path

- Forbidden for production flow.
- Use logger (`debug/info/warning/error`) only.

### 1.4 Breaking YAML interface without compatibility layer

- Forbidden to remove old keys directly.
- Must provide compatibility window and explicit precedence rules.

### 1.5 Implicit mutation of caller-owned objects

- Forbidden to mutate incoming config/data objects unless function contract states so.
- Default policy: copy before mutation in transform/render helpers.

### 1.6 New unsafe eval surface

- Forbidden to add new `eval` entry points.
- Existing eval paths must stay restricted to controlled globals and dataframe locals.

### 1.7 Ambiguous cache identity

- Forbidden to change transform semantics without updating cache fingerprint/schema signal.

### 1.8 Hardcoded absolute local paths

- Forbidden in package code and tracked config.
- Only allowed in local docs/examples explicitly marked local.


## 2) 不提倡 (SHOULD AVOID)

### 2.1 Monolithic methods

- Avoid expanding giant multi-responsibility functions.
- Prefer extracting coherent helpers first.

### 2.2 Dict contract without validation

- Avoid deep nested `dict.get(...).get(...)` in many places.
- Prefer normalized config object per stage.

### 2.3 Repeated logic across `Figure` and `load_data`

- Avoid copying same expression-eval and transform routines.
- Reuse centralized helpers.

### 2.4 Logging at wrong severity

- Avoid `warning` for normal lifecycle noise.
- keep:
  - `debug`: diagnostics
  - `info`: meaningful runtime milestones
  - `warning`: degraded mode / fallback
  - `error`: failure

### 2.5 Hidden global coupling

- Avoid importing runtime globals from another module for constants when it can form circular dependencies.


## 3) Required Change Checklist (Every PR/patch)

1. Confirm YAML compatibility impact.
2. Confirm cache identity impact.
3. Verify at least one real YAML from `bin/` still runs/loads.
4. Run syntax checks for touched modules.
5. For rendering changes:
  - one rect case
  - one ternary case (if affected)
  - one colorbar case (if affected)
6. Update local docs if behavior/contract changed.


## 4) Project-Specific Rules

### 4.1 Colorbar rules

- Single source of truth is `frame.axc.color`.
- Layer `style.cmap/vmin/vmax/norm` is compatibility fallback only.

### 4.2 Style card rules

- Style keys must exist in `style_preference.json`; no implicit `"default"` unless defined.

### 4.3 Dataset loader rules

- all `None`-guard branches in setters must return early.
- HDF5 fallback branch must end with concrete dataframe assignment or explicit error.

### 4.4 Expression rules

- Any added expression helper must be registered through `inner_func.update_funcs`.
- Do not create hidden ad-hoc eval namespaces in random modules.


## 5) Refactor Priority Queue (When Time Allows)

1. reduce duplicated path-resolution and dataset-policy logic across `core.py`, `figure.py`, `data_loader.py`, `data_loader_runtime.py`, and `data_loader_hdf5.py`
2. split `Figure` monolith into:
  - config ingestion
  - layout assembly
  - layer runtime
  - colorbar manager
  - render dispatch
3. split `data_loader.py` into source loading and summary emission helpers; keep transform/runtime handling in `data_loader_runtime.py` and HDF5 policy in `data_loader_hdf5.py`
4. replace ad-hoc dict contracts with typed schema/dataclasses for frame/layer color config
5. add minimal automated regression tests for:
  - profile/grid_profile cache reuse
  - colorbar scale/limits
  - style fallback resolution
  - path-resolution behavior on a real YAML input


## 6) Definition of Done (for Me)

A change is done only if:

- runtime behavior is clear and documented
- legacy YAML does not silently break
- log messages explain fallback behavior
- cache correctness is preserved
- local context docs remain up to date

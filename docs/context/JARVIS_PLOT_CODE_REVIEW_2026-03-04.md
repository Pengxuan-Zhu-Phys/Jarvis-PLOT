# Jarvis-PLOT Code Review Snapshot (2026-03-04)

Scope reviewed:

- `jarvisplot/*.py`
- `jarvisplot/Figure/*.py`
- `jarvisplot/utils/*.py`
- style/config assets required to understand runtime behavior


## Findings (Ordered by Severity)

## [P1] HDF5 fallback branch is unfinished

- file: `jarvisplot/data_loader.py:665`
- detail:
  - In `load_hdf5`, when configured `group` is missing/not-group, branch only does:
    - `path, arr = _pick_dataset(f1)`
  - No conversion to dataframe, no `self.data/self.keys` assignment, no return/error.
- impact:
  - dataset may remain unloaded/silent-fail depending on caller path.


## [P1] Style default fallback likely invalid

- file: `jarvisplot/Figure/figure.py:1261`
- detail:
  - missing figure `style` falls back to `["a4paper_2x1", "default"]`
  - but style cards (`jarvisplot/cards/style_preference.json`) do not define `"default"` for `a4paper_2x1`.
- impact:
  - figure setup may fail when style omitted.


## [P1] `DataSet` setters mishandle `None`

- file: `jarvisplot/data_loader.py:193`
- file: `jarvisplot/data_loader.py:205`
- detail:
  - `file`/`type` setters set fields on `None` case but do not return.
  - code continues into `Path(value)` or `str(value).lower()`.
- impact:
  - latent runtime error and wrong state when `None` is passed.


## [P2] Multiple runtime `eval` entry points

- files:
  - `jarvisplot/Figure/load_data.py:29`
  - `jarvisplot/Figure/load_data.py:615`
  - `jarvisplot/Figure/load_data.py:637`
  - `jarvisplot/Figure/figure.py:1377`
  - `jarvisplot/utils/interpolator.py:300`
- detail:
  - expression flexibility is high, but attack surface and debug complexity are also high.
- impact:
  - security and maintainability risk; especially when YAML input is untrusted.


## [P2] Runtime debug `print()` remains

- files:
  - `jarvisplot/config.py:64`
  - `jarvisplot/config.py:66`
  - `jarvisplot/core.py:358`
- detail:
  - print statements bypass logger policy.
- impact:
  - noisy output and inconsistent diagnostics.


## [P2] Logging format bug in `grid_profile` runtime log

- file: `jarvisplot/Figure/preprocessor.py:449`
- detail:
  - format string includes `delta` argument but no placeholder in string tail.
- impact:
  - loses useful debug signal and indicates message drift.


## [P3] Colormap registration duplicated and partially no-op

- files:
  - `jarvisplot/core.py:94`
  - `jarvisplot/Figure/figure.py:1276`
- detail:
  - core loads colormap JSON with explicit path.
  - figure setup calls `cmaps.setup(force=True)` without json path; current API returns empty registration if path omitted.
- impact:
  - redundant call path; confusing logs.


## [P3] Version metadata inconsistent

- files:
  - `pyproject.toml` (`version = "1.2.5"`)
  - `VERSION` (`1.1.1`)
- impact:
  - release/version confusion in local tooling and docs.


## [P3] Test coverage gap

- observation:
  - no `tests/` or automated regression suite in repo.
- impact:
  - behavior regressions likely in plotting and cache paths.


## Strengths Observed

- clear module boundaries between orchestration/data/render.
- thoughtful caching model with demand fingerprints and metadata checks.
- adapter architecture supports custom plotting primitives.
- recent colorbar interface unification (`frame.axc.color`) moves in the right direction.


## Suggested Next Fix Order

1. finish HDF5 fallback path and add explicit error behavior
2. fix style fallback and setter `None` handling
3. clean print/debug artifacts and logging format issue
4. reduce/centralize eval paths
5. add minimum smoke tests for:
  - colorbar scale/limits
  - profile cache reuse
  - style selection fallback

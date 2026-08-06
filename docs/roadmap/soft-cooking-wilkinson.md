# Jarvis-PLOT v2.0 Architectural Refactoring Plan

Status: active backlog
Last updated: 2026-07-16

## Context

Jarvis-PLOT is a YAML-driven HEP plotting engine (~20k LOC in `jarvisplot/`, currently **v1.4.2**) that has grown organically with incremental "patch-on-patch" development. The goal is a **v2.0 release** with a clean architecture designed for a 5+ year lifecycle. Since this is a major version bump, we can break backward compatibility with a migration guide — no need for re-export shims or facade classes.

This document is a longer-horizon plan. Near-term 1.x work (Agent Data API, remaining boundary cleanup) lives in `IMPLEMENTATION_ROADMAP.md`.

### Key Problems in v1.x

1. **God Objects** — `JarvisPLOT` (core.py) and `Figure` (figure.py, 1165 lines) own everything
2. **Property-setter-as-constructor** — `self.ax = kwgs` triggers 100+ lines of axis setup
3. **No interfaces** — no Protocol/ABC for DataSet, adapters, transforms
4. **Grab-bag `Figure/`** — rendering, data preprocessing, interpolation algorithms, profiling all in one package
5. **Tight YAML coupling** — runtime code is married to specific YAML key names
6. **Duplicated helpers** — `_expr_symbols()` in 2 places, `_coerce_positive_int()` in 2 places
7. **No dependency direction** — core ↔ Figure imports tangle
8. **Weak typing** — `dict[str, Any]` everywhere, no schema validation
9. **Low test coverage** — 15 test files for 23k lines

### User Decisions

- **Priority:** Clean architecture first (package restructure)
- **Compatibility:** Major version bump (v2.0) — YAML schema cleanup allowed, no backward-compat shims
- **Tooling:** Full modernization — hatch, mypy, ruff, pre-commit, import-linter, hypothesis, coverage, structured logging

---

## Phase 1: Build System & Tooling Modernization (Week 1)

**Goal:** Modern Python project foundation before any code changes.

### 1.1 Switch to Hatch build system

Replace `setuptools` in `pyproject.toml` with `hatchling`:
- Move version to `jarvisplot/__about__.py` (single source of truth)
- Remove `JarvisPLOT.egg-info/`, `dist/`, `setup.cfg` if present
- Configure `[tool.hatch.build]` for package-data (cards/)

### 1.2 Add development tooling

Add to `pyproject.toml`:
- `[tool.ruff]` — formatting and linting rules
- `[tool.mypy]` — strict mode, start with `--ignore-missing-imports`
- `[tool.pytest.ini_options]` — test config, coverage plugin
- `[tool.import-linter]` — enforce dependency layers (defined in Phase 2)

### 1.3 Pre-commit hooks

Create `.pre-commit-config.yaml`:
- ruff (format + lint)
- mypy
- check-yaml, check-toml, trailing-whitespace

### 1.4 CI pipeline update

Update `.github/workflows/python-package.yml`:
- Matrix: Python 3.10, 3.11, 3.12, 3.13
- Steps: lint → type-check → test → coverage report

**Files:** `pyproject.toml`, `.pre-commit-config.yaml`, `.github/workflows/`, `jarvisplot/__about__.py`

---

## Phase 2: Package Restructure (Week 2-4)

**Goal:** Enforce a strict dependency DAG. This is the highest-priority phase.

### Target Layout

```
jarvisplot/
├── __about__.py              # version
├── __init__.py               # public API
├── cli.py                    # argparse (slimmed)
├── client.py                 # entry: main()
│
├── schema/                   # [NEW] Typed YAML config
│   ├── __init__.py
│   ├── project.py            # ProjectConfig, OutputConfig
│   ├── dataset.py            # DataSetConfig
│   ├── figure.py             # FigureConfig, LayerConfig
│   ├── transforms.py         # TransformStep, CoordinateSpec
│   └── validation.py         # parse + validate at boundary
│
├── loaders/                  # [NEW] Data loading
│   ├── __init__.py
│   ├── base.py               # DataLoader Protocol
│   ├── csv.py
│   ├── hdf5.py
│   ├── parquet.py
│   └── dataset.py            # DataSet class (slimmed from data_loader.py)
│
├── transforms/               # [NEW] Data transforms
│   ├── __init__.py
│   ├── base.py               # Transform Protocol + TransformRegistry
│   ├── add_column.py
│   ├── filter.py
│   ├── sort.py
│   ├── profile.py            # from Figure/profile_runtime.py
│   ├── density.py            # from Figure/density_cell_runtime.py
│   ├── interpolation.py      # from Figure/interp_2d_runtime.py
│   └── posterior.py           # from Figure/posterior_density_runtime.py
│
├── algorithms/               # [NEW] Pure math, zero matplotlib dependency
│   ├── __init__.py
│   ├── natural_neighbor.py   # from Figure/interp_natural_neighbor*.py
│   ├── posterior_mesh.py     # from Figure/posterior_mesh.py (1416 lines)
│   ├── posterior_hpd.py      # from Figure/posterior_hpd.py
│   └── density_cell.py       # density estimation core
│
├── rendering/                # [REPLACES Figure/]
│   ├── __init__.py
│   ├── figure.py             # slimmed Figure (orchestrator only, <300 lines)
│   ├── axes_factory.py       # [NEW] explicit axis creation
│   ├── renderer.py           # [NEW] render loop + colorbar prescan
│   ├── saver.py              # [NEW] savefig + metadata
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py           # AxesAdapter Protocol
│   │   ├── rect.py           # from adapters_rect.py
│   │   └── ternary.py        # from adapters_ternary.py
│   ├── colorbar.py           # from colorbar_runtime.py
│   ├── layout.py             # from layout_runtime.py
│   ├── style.py              # from style_runtime.py
│   ├── config.py             # from config_runtime.py
│   └── methods/
│       ├── __init__.py
│       ├── registry.py       # from method_registry.py
│       └── dynesty.py        # from dynesty_runtime.py
│
├── pipeline/                 # [NEW] Orchestration layer
│   ├── __init__.py
│   ├── session.py            # Session: workdir, cache, logger
│   ├── planner.py            # column planning, usage plan
│   ├── preprocessor.py       # from Figure/preprocessor.py
│   ├── engine.py             # figure loop
│   └── data_context.py       # from Figure/data_pipelines.py
│
├── cache/                    # [EXTRACTED]
│   ├── __init__.py
│   └── store.py              # from cache_store.py
│
├── utils/                    # (expanded, deduplicated)
│   ├── __init__.py
│   ├── expression.py         # + consolidated _expr_symbols
│   ├── pathing.py
│   ├── dataframes.py
│   ├── hashing.py            # [NEW] from preprocessor._stable_hash
│   ├── numeric.py            # [NEW] from _coerce_positive_int etc.
│   ├── cmaps.py
│   ├── interpolator.py
│   └── logging.py            # [NEW] structured logging setup
│
├── flowchart.py              # (standalone, keep as-is)
└── cards/                    # (unchanged)
```

### Dependency Layers (enforced by import-linter)

```
Layer 1 (bottom):  utils, schema, algorithms
Layer 2:           loaders, transforms, cache
Layer 3:           rendering, pipeline
Layer 4 (top):     cli, client
```

Each layer may only import from layers below it. No lateral imports within the same layer except within the same subpackage.

### import-linter contract in `pyproject.toml`:

```toml
[tool.importlinter]
root_packages = ["jarvisplot"]

[[tool.importlinter.contracts]]
name = "Layered architecture"
type = "layers"
layers = [
    "jarvisplot.cli | jarvisplot.client",
    "jarvisplot.pipeline | jarvisplot.rendering",
    "jarvisplot.loaders | jarvisplot.transforms | jarvisplot.cache",
    "jarvisplot.utils | jarvisplot.schema | jarvisplot.algorithms",
]
```

### Migration strategy

Move files in small batches, one subpackage at a time:
1. `utils/` expansion + `schema/` (no existing code depends on these)
2. `algorithms/` (extract pure math from Figure/)
3. `loaders/` (extract from data_loader*.py)
4. `transforms/` (extract from Figure/*_runtime.py)
5. `cache/` (rename cache_store.py)
6. `pipeline/` (extract from core.py, core_runtime.py, Figure/data_pipelines.py, Figure/preprocessor.py)
7. `rendering/` (what remains in Figure/)

After each step, run tests + import-linter to verify no layer violations.

---

## Phase 3: Break the God Objects (Week 4-6)

### 3A: Decompose `JarvisPLOT` → `pipeline/`

`JarvisPLOT.init()` currently does ~15 things in sequence. Decompose into:

- **`Session`** (`pipeline/session.py`): CLI args, logger, workdir, cache lifecycle
- **`Planner`** (`pipeline/planner.py`): dataset required columns, usage plan
- **`Engine`** (`pipeline/engine.py`): the figure loop — creates Figure, wires dependencies, calls plot()

New `client.py`:
```python
def main():
    session = Session.from_cli()
    config = session.load_config()  # returns typed ProjectConfig
    planner = Planner(session, config)
    planner.prepare()
    Engine(session, planner).run()
```

### 3B: Decompose `Figure` → `rendering/`

Split 1165-line Figure into:
- **`Figure`** (rendering/figure.py, <300 lines): data container, delegates to factory/renderer/saver
- **`AxesFactory`** (rendering/axes_factory.py): `create_rect()`, `create_ternary()`, `create_colorbar()`, `create_logo()` — explicit method calls, not property setters
- **`Renderer`** (rendering/renderer.py): `render(figure)` — colorbar prescan, layer loop, legend, finalize
- **`FigureSaver`** (rendering/saver.py): `save(figure)` — savefig with metadata, PNG post-processing

### 3C: Clean up property-setter anti-pattern

Replace property-setter-as-constructor with explicit builder methods:

```python
# BEFORE (v1.x)
fig.style = ["gambit", "2DPL"]    # triggers 15 lines of deep-copy + merge
fig.ax = {"rect": [0.1, 0.1, ...]} # triggers 100 lines of axis setup

# AFTER (v2.0)
fig.apply_style(style_bundle)
axes_factory.create("ax", rect=[0.1, 0.1, ...], config=frame_config)
```

---

## Phase 4: Protocol-Based Extension Points (Week 6-8)

### 4A: Transform Protocol

```python
# transforms/base.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class Transform(Protocol):
    key: str
    def input_columns(self, cfg: Mapping) -> set[str]: ...
    def output_columns(self, cfg: Mapping) -> set[str]: ...
    def execute(self, df: pd.DataFrame, cfg: Mapping, ctx: TransformContext) -> pd.DataFrame: ...
```

Each existing transform becomes a class implementing this protocol. Register in `TransformRegistry`:
```python
registry = TransformRegistry()
registry.register(ProfileTransform())
registry.register(DensityCellTransform())
registry.register(Interp2DTransform())
```

This replaces the cascading if/elif chains in `preprocessor_runtime.py` and `core_runtime.py:_collect_expr_columns`.

### 4B: DataLoader Protocol

```python
# loaders/base.py
@runtime_checkable
class DataLoader(Protocol):
    type_key: str
    def load(self, config: DataSetConfig, session: Session) -> pd.DataFrame: ...
    def lazy_metadata(self, config: DataSetConfig) -> list[str]: ...
```

### 4C: AxesAdapter Protocol

Formalize the adapter interface that `StdAxesAdapter` and `TernaryAxesAdapter` already implicitly implement:
```python
# rendering/adapters/base.py
class AxesAdapter(Protocol):
    ax: Axes
    layers: list
    status: str
    def plot(self, **kwargs): ...
    def scatter(self, **kwargs): ...
    def finalize(self): ...
```

---

## Phase 5: YAML Schema v2.0 & Validation (Week 8-9)

Since this is a major version bump, clean up the YAML schema:

1. **Validate on load** — `jplot validate <file.yaml>` and automatic validation in `session.load_config()`
2. **Clear error messages** — "Figure 'MSSM7_2D' layer 0: 'method' must be one of: scatter, contour, ... Got: 'scater'"
3. **Schema cleanup opportunities:**
   - Normalize coordinate specification (currently accepts both flat and nested forms)
   - Standardize transform step format (currently some use `{type: X}`, others use `{X: config}`)
   - Add `version: 2` field to YAML for format detection
4. **Migration tool:** `jplot migrate <old.yaml> -o <new.yaml>` converts v1 YAML to v2 format

---

## Phase 6: Testing Infrastructure (Week 9-11)

### Unit tests (target: >70% coverage on non-rendering code)
- Each Transform implementation: synthetic DataFrame in → assert columns + values out
- Each DataLoader: fixture files in `tests/fixtures/` (tiny CSV, HDF5, Parquet)
- Schema validation: valid/invalid YAML samples
- Utils: pathing, expression, hashing

### Integration tests
- Full pipeline: small YAML + tiny dataset → assert PNG exists + correct metadata
- Transform chain: multi-step transform pipeline → assert final DataFrame shape

### Property-based tests (hypothesis)
- `algorithms/natural_neighbor.py` — random point sets, verify interpolation contracts
- `algorithms/posterior_mesh.py` — random density fields, verify mesh properties
- `algorithms/density_cell.py` — random 2D samples, verify density normalization

### Golden-file regression
- Run reference YAML files from `bin/`, store output hashes
- CI compares against stored hashes, flags visual regressions

### Coverage reporting
- `pytest-cov` with `--cov-report=html`
- CI fails if coverage drops below threshold

---

## Phase 7: Structured Logging & Developer Experience (Week 11-12)

### Structured logging
- Add `utils/logging.py` — configure loguru with structured JSON output option
- Replace bare `self.logger.warning(f"...")` with structured fields where useful
- Keep loguru (already a dependency), add JSON sink option for machine parsing

### Developer docs
- `CONTRIBUTING.md` — how to add a new transform, plot method, or data loader
- `MIGRATING.md` — v1.x → v2.0 migration guide
- `docs/design/` — update architecture docs to reflect new structure

---

## Verification Strategy

For each phase:
1. `pytest tests/` — all existing tests pass
2. `ruff check .` — no lint violations
3. `mypy jarvisplot/` — type checks pass on changed modules
4. `import-linter` — no layer violations
5. **Visual regression:** render reference YAML files, compare output PNGs
6. `jplot MSSM7.yaml` and other reference files produce identical output to v1.x

## Risk Mitigation

| Phase | Risk | Mitigation |
|---|---|---|
| 1 (tooling) | Low | Purely additive |
| 2 (restructure) | High | Move one subpackage at a time, run tests after each move |
| 3 (god objects) | Medium | Keep old entry point working throughout, refactor internals |
| 4 (protocols) | Medium | Wrap existing functions in protocol classes first, then refactor internals |
| 5 (YAML v2) | Medium | Provide `jplot migrate` tool, keep v1 parser as fallback initially |
| 6 (testing) | Low | Purely additive |
| 7 (DX) | Low | Purely additive |

## Timeline

- **Weeks 1-4:** Phase 1 + 2 (foundation + restructure) — this delivers the biggest architectural win
- **Weeks 4-6:** Phase 3 (decompose god objects)
- **Weeks 6-9:** Phase 4 + 5 (protocols + schema)
- **Weeks 9-12:** Phase 6 + 7 (testing + DX)
- **Week 12:** v2.0-rc1 release

# Transform Contract

Status: implemented

This document is the canonical transform contract for the current Jarvis-PLOT
runtime. It reflects the dispatch paths in:

- `jarvisplot/data_loader_runtime.py`
- `jarvisplot/Figure/preprocessor_runtime.py`
- `jarvisplot/Figure/preprocessor.py`
- `jarvisplot/core_runtime.py`

## Canonical Transform Steps

The ordered `transform` list supports these steps:

- `filter`
- `profile`
- `make_density_core`
- `make_interp_2d`
- `posterior_density`
- `sortby`
- `add_column`
- `keep_columns`
- `drop_columns`
- `to_csv`
- `to_parquet`

Order is authoritative. Each step consumes the DataFrame produced by the
previous step.

`expression` is not a transform step. Computed columns must use `add_column`,
with the expression stored in `add_column.expr`.

## Primitive Steps

### `filter`

Filters rows by a boolean expression or boolean-like scalar.

```yaml
transform:
  - filter: x > 0
```

### `add_column`

Adds or replaces a column from an expression evaluated against the current
DataFrame.

```yaml
transform:
  - add_column:
      name: posterior_weight
      expr: exp(logL)
```

### `sortby`

Sorts rows by an existing column or expression.

```yaml
transform:
  - sortby: logL
```

### `keep_columns` / `drop_columns`

Explicit column-pruning steps. Without these, Jarvis-PLOT does not implicitly
drop columns from a dataset transform list.

```yaml
transform:
  - keep_columns: [x, y, logL]
  - drop_columns: [temporary]
```

### `to_csv` / `to_parquet`

Exports the current DataFrame state at that point in the ordered transform
list.

```yaml
transform:
  - to_csv: ./data/out.csv
  - to_parquet: ./data/out.parquet
```

## Reduction And Field Steps

### `profile`

Data-reduction transform owned by `jarvisplot/Figure/profile_runtime.py`.
`method: grid` is part of `profile`; there is no separate `grid_profile`
transform.

```yaml
transform:
  - profile:
      method: grid
      coordinates:
        x: {expr: x, name: x}
        y: {expr: y, name: y}
        z: {expr: logL, name: logL}
      objective: max
```

### `make_density_core`

Raw samples to minimal posterior mass-support table. Supported methods are:

- `grid`
- `bridson`
- `kde`

The output table contains only the configured x/y/weight columns. See
`POSTERIOR_DENSITY.md` for the statistical contract.

Canonical form:

```yaml
transform:
  - make_density_core:
      x: {expr: x, lim: [0, 5]}
      y: {expr: y, lim: [0, 5]}
      weight: {expr: exp(logL)}
      bins: 100
```

The default method is `bridson`. The runtime also accepts the old
`coordinates:` block plus `type: make_density_core` for compatibility.

### `make_interp_2d`

Support/core table to regular 2D field grid. Supported methods are:

- `natural_neighbor`
- `triangulation`
- `griddata`

The output table contains only the configured x/y/z columns. See `INTERP_2D.md`
for grid syntax, `as_density`, and `normalize`.

Canonical form:

```yaml
transform:
  - make_interp_2d:
      method: natural_neighbor
      coordinates:
        x: {expr: x, name: x, lim: [0, 5], scale: linear}
        y: {expr: y, name: y, lim: [0, 5], scale: linear}
        z: {expr: mass, name: posterior_pdf}
      grid: 500
```

The runtime also accepts `type: make_interp_2d` for compatibility, but new YAML
should use the nested form above.

### `posterior_density`

Raw samples to regular posterior density grid. This is the compact path for the
common `make_density_core -> make_interp_2d` workflow.

Supported methods:

- `voronoi`: Bridson support reconstruction followed by natural-neighbor
  interpolation.
- `adaptive`: `voronoi` plus posterior mesh refinement.
- `grid`: regular histogram mass assignment converted directly to density.
- `kde`: weighted Gaussian KDE converted directly to density.

Canonical form:

```yaml
transform:
  - posterior_density:
      x: {expr: x, lim: [0, 5]}
      y: {expr: y, lim: [0, 5]}
      weight: {expr: exp(logL)}
      bins: 100
      grid: 300
```

The output table contains `x`, `y`, and `density` by default. Set
`output: posterior_pdf` to rename the density column. The runtime also accepts
`type: posterior_density`.

## Runtime Scope Notes

- Dataset-level transforms run in `data_loader_runtime.py`.
- Figure/layer pipeline transforms run in `Figure/preprocessor_runtime.py`.
- Layer-local fallback transforms in `Figure/layer_runtime.py:load_bool_df()` are
  older compatibility code and support only `filter`, `profile`, `sortby`,
  `add_column`, and CSV export. New transform work should target the main
  preprocessor runtime path.
- Polars pushdown currently covers `filter`, `sortby`, `add_column`,
  `keep_columns`, `drop_columns`, and export steps where the backend supports
  them. `profile`, `make_density_core`, `make_interp_2d`, and
  `posterior_density` force the pandas transform path.

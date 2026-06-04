# Posterior Density Reconstruction

Status: implemented

Jarvis-PLOT separates posterior measure reconstruction from interpolation and
rendering:

1. `make_density_core` maps raw samples to a minimal support/core table.
2. `make_interp_2d` maps support/core values to a regular field grid.
3. `posterior_density` combines those steps for the common "raw samples to
   regular density grid" workflow.
4. Plot layers render the already prepared grid with `pcolormesh` or `contour`.

## Statistical Contract

Posterior reconstruction is a measure problem.

- Raw posterior weights must not be silently discarded.
- `make_density_core` normalizes support mass when `normalize: true`.
- Interpolation happens after mass assignment.
- HPD contour levels are integrated-probability thresholds, not fixed fractions
  of the maximum density.

## Density Core Transform

`make_density_core` consumes the full raw sample DataFrame and returns only:

```text
x
y
weight
```

Column names come from `coordinates.<key>.name` when provided. No density,
area, cell id, polygon, grid metadata, or plotting state is emitted.

Supported methods:

- `grid`: aggregate raw sample mass into regular grid cells.
- `bridson`: generate Bridson / Poisson-disk support points and assign each raw
  sample mass to the nearest support point.
- `kde`: evaluate weighted Gaussian KDE from raw samples and return grid-cell
  probability masses.

Compact form:

```yaml
transform:
  - make_density_core:
      x: {expr: xx, lim: [0, 5]}
      y: {expr: yy, lim: [0, 5]}
      weight: {expr: exp(LogL)}
      bins: 100
```

Legacy explicit form:

```yaml
transform:
  - make_density_core:
      method: bridson
      coordinates:
        x: {expr: xx, name: x, lim: [0, 5], scale: linear}
        y: {expr: yy, name: y, lim: [0, 5], scale: linear}
        weight: {expr: exp(LogL), name: mass}
      bridson:
        bin: 100
        seed: 123
      normalize: true
```

The default method is `bridson`, default `bins` is `64`, and `normalize`
defaults to `true`. Legacy `coordinates.*.name`, `grid: {bins: ...}`, and
`bridson: {bin: ...}` remain supported.

## Combined Posterior Density Transform

Use `posterior_density` when the desired output is already a regular density
grid:

```yaml
transform:
  - posterior_density:
      x: {expr: xx, lim: [0, 5]}
      y: {expr: yy, lim: [0, 5]}
      weight: {expr: exp(LogL)}
      bins: 120
      grid: 300
```

Output columns are:

```text
x
y
density
```

Supported methods:

- `voronoi` default: Bridson support reconstruction followed by
  natural-neighbor interpolation with `as_density: true`.
- `adaptive`: same as `voronoi`, with posterior mesh refinement enabled.
- `grid`: regular histogram mass assignment, converted directly to density.
- `kde`: weighted Gaussian KDE on a regular grid, converted directly to density.

Common options:

```yaml
method: voronoi          # voronoi | adaptive | grid | kde
bins: 64                 # support/grid resolution
grid: 256                # output interpolation grid for voronoi/adaptive
normalize: true
seed: 123
output: posterior_pdf    # z column name, defaults to density
```

## Interpolation To PDF

Use `make_interp_2d` for support/core-to-grid interpolation. When the input
`z` column is conserved mass, set `as_density: true` so support cell areas are
computed internally and the interpolated field is a density:

```yaml
transform:
  - make_interp_2d:
      method: natural_neighbor
      as_density: true
      normalize: true
      coordinates:
        x: {expr: x, name: x, lim: [0, 5], scale: linear}
        y: {expr: y, name: y, lim: [0, 5], scale: linear}
        z: {expr: mass, name: posterior_pdf}
      grid: 500
      nan_policy: strict
```

`as_density: true` converts support mass to local density before interpolation.
`normalize: true` rescales the final regular grid so that
`sum(posterior_pdf) * dx * dy` is approximately one.

## HPD Posterior Contours

Posterior credible-region contours use highest-posterior-density thresholds
computed from integrated posterior mass. For a regular density grid,
Jarvis-PLOT normalizes the grid internally, sorts density values from high to
low, accumulates `density * cell_area`, and finds thresholds for the requested
credible masses.

Example:

```yaml
- name: posterior_pdf_hpd_contours
  data:
  - source: posterior_pdf_grid
  axes: ax
  method: contour
  coordinates:
    x: {expr: x}
    y: {expr: y}
    z: {expr: posterior_pdf}
  style:
    contour_mode: posterior_hpd
    masses: [0.6827, 0.9545]
    labels: ["1σ / 68%", "2σ / 95%"]
    colors: ["black", "white"]
    linestyles: ["solid", "solid"]
    linewidths: [0.2, 0.2]
```

## Output Format Policy

File formats belong in the YAML top-level `output.formats` list, not inside
individual posterior figure blocks.

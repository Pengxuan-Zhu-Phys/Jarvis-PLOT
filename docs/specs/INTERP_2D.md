# 2D Interpolation Transform

Status: implemented

`make_interp_2d` converts a support/core table into a regular 2D scalar-field grid.
It is deliberately independent of density reconstruction and plotting layers.

## Contract

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

Input is the current raw/support DataFrame. Output is a new DataFrame with only
the three configured coordinate columns:

```text
x
y
posterior_pdf
```

Column names come from `coordinates.<key>.name`; defaults are `x`, `y`, and `z`.

## Coordinates

- `coordinates.x.expr`, `coordinates.y.expr`, and `coordinates.z.expr` select
  or compute input arrays.
- `coordinates.x.lim` and `coordinates.y.lim` define the interpolation range.
  If omitted, limits are inferred from finite input values.
- `coordinates.x.scale` and `coordinates.y.scale` support `linear` and `log`.
  Interpolation is performed in the scaled coordinate space, while output `x`
  and `y` are written in the original physical coordinate space.

There is no separate `domain` block for this transform.

## Grid

Preferred compact syntax:

```yaml
grid: 500        # 500 x 500
grid: [500, 300] # 500 x 300
```

If omitted, the default is `256 x 256`. The older verbose forms remain valid:

```yaml
grid: {bins: 500}
grid: {nx: 500, ny: 300}
```

## Methods

- `natural_neighbor`: uses the registered Natural Neighbor backend.
- `triangulation`: Delaunay-based interpolation.
  ```yaml
  triangulation:
    kind: linear  # linear / cubic
  ```
- `griddata`: SciPy-style griddata interpolation.
  ```yaml
  griddata:
    kind: nearest  # nearest / linear / cubic
  ```

`nan_policy: strict` preserves `NaN` outside the interpolation hull where the
backend supports hull-aware behavior.

## Density And Normalization

`as_density: false` directly interpolates `coordinates.z` and is appropriate
for generic scalar fields, profile likelihood, or already-computed density.

`as_density: true` treats `coordinates.z` as conserved support/core mass. The
transform computes support-cell areas internally, converts `mass / area`, and
interpolates the resulting density. Regular support grids use inferred cell
areas; irregular support uses clipped Voronoi areas inside the x/y limits.

`normalize: true` rescales the final output grid so finite values satisfy:

```text
sum(z) * dx * dy ~= 1
```

NaN values remain NaN.

## Boundary

`make_interp_2d` does not assign posterior mass or choose plotting styles. Mass
assignment belongs to `make_density_core`; plotting belongs to the layer.

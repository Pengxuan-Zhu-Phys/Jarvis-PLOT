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

## NaN Policy

`nan_policy` decides what the interpolator does with a *core* whose `z` is not
finite. An empty profile-likelihood cell is the usual source of one: the bin
holds no sample, so `profile` writes `NaN` there.

| value | meaning |
|-------|---------|
| `strict` (default) | the core is kept and its missing value propagates: every query point whose natural-neighbor stencil touches it returns `NaN`. One empty cell therefore blanks the cells around it too. |
| `ignore` (`omit`, `drop`) | the core is deleted before the triangulation is built. No sample landed there, so there is nothing to interpolate from, and the surrounding cores close over the gap. |
| `fill` | the core is kept with `backend_options.fill_value` (default: the smallest finite core value), which floors the gap instead of smoothing it. |

With `backend_options.boundary: nan` (the transform default), query points
outside the convex hull return `NaN` under every NaN policy.

Under `ignore` the hull is rebuilt from the surviving cores, so empty cells on
the rim shrink the drawn region rather than being invented.

`ignore` will close over a large void as happily as a one-cell gap. To keep
regions the scan never visited blank, cap how far a query point may sit from a
real core:

```yaml
backend_options:
  max_fill_spacing: 2.0     # multiples of the median core spacing
  # max_fill_distance: 0.05 # or an absolute distance, in interpolation coords
```

`make_interp_2d` validates coordinates separately from values and applies the
NaN policy **before duplicate merging**. `strict` retains missing cores,
`ignore` drops them, and `fill` assigns their configured value. This also
applies to the triangulation and griddata methods, with propagation determined
by the chosen interpolator. Non-finite coordinates and out-of-domain rows are
always discarded. Diagnostics report missing, dropped, and filled core counts.

For `profile: {method: bridson, grid_points: rect}`, background candidates
follow each coordinate's scale: linear spacing for linear axes, geometric
spacing for log axes. Bridson removes candidates near earlier seeds and emits
surviving background points with `z: NaN`. A preceding `filter` acts on the
source samples, not these subsequently generated points. `share_data` retains
them; interpolation then follows its explicit NaN policy. For a likelihood
background intended to equal zero, use `nan_policy: fill` with
`backend_options: {fill_value: 0}`; missing data are not implicitly zero.

## Past The Hull

Cores stand at the *centre* of the cell they summarise, so the convex hull of
the cores stops half a cell short of the domain on every side — a blank frame
around the picture that is an artifact of the discretization, not a statement
about the scan. At a corner it is worse: the hull cuts the diagonal, and any
empty cell on the rim moves that cut further in.

`backend_options.boundary` decides how far the surface is carried past the
hull:

| value | meaning |
|-------|---------|
| `nan` | nothing is drawn outside the hull. The backend default, and what `make_interp_2d` and `posterior_density` use — extrapolating a normalized density past its support would move the integral. |
| `clamp` | the surface is read at the closest point of the hull and carried outwards from there: constant normal to the boundary, continuous with the interior, no seam. |
| `nearest` | the closest core's value is carried outwards. Cheaper, and blockier along the rim. |

The reach is always bounded — `max_boundary_spacing` in multiples of the core
spacing (default `2.0`), or `max_boundary_distance` as an absolute distance. A
`max_fill_spacing` / `max_fill_distance` set for the interior is inherited when
no boundary-specific reach is given, so one number can govern both.

**Drawing calls default to `clamp`**: `style.interp` on a contour/contourf
layer, the `jpcontour` / `jpcontourf` / `jpfield` methods, and the
`type: profile_2d` macro. They exist to make a picture, and the half-cell frame
is not information. Set `boundary: nan` to draw only the hull the cores span:

```yaml
interp:
  method: natural_neighbor
  backend_options:
    boundary: nan
```

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

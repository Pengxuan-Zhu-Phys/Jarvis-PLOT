# Raw-sample PDF and CDF transforms

Status: implemented

`PDF1D` constructs a continuous CDF from raw sample values and evaluates its
analytic derivative as a PDF. `CDF1D` constructs only the CDF. Construction and
repeat statistics run entirely in the data transform; rendering uses ordinary
`plot` and `fill_between` layers. No histogram bins participate.

## YAML contract

```yaml
data:
  - source: score_events
    transform:
      - filter: '(sample == "signal") & (split == "train")'
      - PDF1D:
          coordinates:
            x: {expr: score, name: x, lim: [0, 1], grid: 600}
            weight: {expr: weight}
          repeat: repeat
          anchors: {method: adaptive, tolerance: 0.005, min_mass: 0.01}
          interpolation: monotone_c2
```

Replace `PDF1D` with `CDF1D` to omit PDF evaluation. Names are case sensitive in
YAML; `jplot man PDF1D` and `jplot man transform.CDF1D` describe the same live
contracts advertised by `jplot cap transforms --json`.

| Field | Meaning / default |
| --- | --- |
| `coordinates.x.expr` | Raw sample column or shared expression; required unless `name` supplies the input column |
| `coordinates.x.name` | Output grid column, default `x`; input column fallback without `expr` |
| `coordinates.x.lim` | Normalization support `[lo, hi]`; inferred when omitted |
| `coordinates.x.grid` | Integer number of evaluation points, default 600, minimum 2 |
| `coordinates.x.scale` | `linear` (linspace) or `log` (geomspace); default `linear` |
| `coordinates.weight.expr` | Nonnegative weight column/expression; omitted means unit weights |
| `repeat` | Optional column identifying independent batches / train-test repeats |
| `anchors` | `adaptive` (default), `all`, or the settings mapping below |
| `anchors.method` | `adaptive` or `all`; default `adaptive` |
| `anchors.tolerance` | Target maximum absolute CDF error at empirical nodes; default 0.005 |
| `anchors.min_mass` | Minimum observed cumulative probability on **each side** of a new anchor; default 0.01 |
| `anchors.max_points` | Anchor budget including support endpoints; default 256, minimum 4 |
| `interpolation` | `monotone_c2` (default) or `pchip` |

`coordinates.x: score` and `coordinates.weight: weight` are supported shorthand.
Weight mappings also allow `name` as the input-column fallback. Expressions
support scalar broadcasting. The input table must be nonempty with finite
samples and weights. Negative probability weights, missing repeat IDs, and any
repeat with zero total weight are errors. Filter invalid rows explicitly first.

Use a long event table: one row is one sample in one repeat. `repeat` identifies
the batch; `sample` and `split` can select the four signal/background and
train/test populations with `filter`. Other fields can be present but are not
retained after the transform. There is no `groupby` option.

## Cumulative reconstruction and normalization

Within each repeat, sort positive-weight samples, combine duplicate values, and
normalize their weights to total one. At each distinct interior value `s`, the
CDF knot is the inclusive weighted empirical probability
`sum(w[x <= s]) / sum(w)`. The adaptive selector retains a subset of these true
nodes plus support endpoints, with `F(lo)=0` and `F(hi)=1`. The interpolant
passes exactly through the retained anchors; probabilities at unselected
samples are approximated. Each interval between retained anchors preserves
its observed probability mass. The analytic PDF is nonnegative and integrates
to one over `[lo, hi]`.

An explicit `lim` must contain every positive-weight sample. It does not silently
truncate or condition the distribution; use `filter` first if that is intended.
Without `lim`, all repeats share pooled support: extend the smallest/largest
distinct sample by half its gap to the next/previous distinct sample. One unique
pooled value needs explicit limits. Log grids require positive support; inferred
lower padding is bounded below by half the smallest sample. Zero-weight rows
do not define support or CDF knots.

A continuous CDF cannot retain an empirical jump exactly at `lo` while also
having `F(lo)=0`. If positive mass lies there, its mass is spread into the first
interpolation interval and the runtime emits a warning. Selected interior knots
retain their empirical cumulative probability. Prefer support below the minimum
sample when preserving every empirical knot is important.

## Adaptive sampling and interpolation

The default now selects adaptive anchors and uses `monotone_c2`. Existing YAML
without these options therefore gets the smoother reconstruction. To reproduce
the original all-node PCHIP result, explicitly set:

```yaml
anchors: all
interpolation: pchip
```

Adaptive selection starts with a small set of true empirical nodes near sparse
cumulative-probability targets. It also retains the first sample and the first
empirical node reaching probability one, to preserve the observed onset and
upper plateau. After fitting, evaluate the CDF error at **all candidate nodes**,
independently of the output grid. Scan disjoint intervals and use a priority
queue to add anchors in up to eight worst eligible intervals per pass. A new
node needs at least `min_mass` observed probability between it and each existing
neighbor. This prevents refinement from following every individual event's
CDF step. Sorting and cumulative sums are done once per repeat; interpolation
operates on the bounded anchor set, without a nonlinear optimizer or a raw
samples-by-grid matrix.

`tolerance` is a target, not an unconditional bound. Refinement stops when it is
met, when `max_points` is reached, or when `min_mass` prevents further eligible
splits. Lower tolerance requests more local detail; larger `min_mass` generally
suppresses more fine-scale structure. The two constraints can conflict. The
runtime warns when the measured node error exceeds the target. This is a CDF
approximation criterion, **not a bound on PDF error or a confidence interval**.
The node error does not claim a uniform bound between empirical jumps. A
significant empirical atom is reported separately: a continuous CDF spreads
its mass and cannot exactly reproduce that jump.

`monotone_c2` constructs quintic Hermite segments. A clamped cubic spline
estimates smooth first/second derivatives in linear time; that unconstrained
cubic is never used as the final CDF. Curvatures and slopes are limited so the
quartic PDF's Bernstein control coefficients are nonnegative. This certifies
PDF nonnegativity over entire intervals, rather than only at grid points. Each
anchor shares one slope and one curvature between its neighbors: CDF is C2 and
PDF is C1. The endpoints have zero slope/curvature, compatible with constant
CDF and zero PDF extension. Evaluation outside the support returns NaN, as for
the original interpolator. This is a conservative Bernstein constraint method,
not a port of MQSI. No additional dependency is required.

`pchip` constructs a monotone cubic C1 CDF and a continuous PDF whose slope can
jump at anchors. It works with either adaptive or all anchors. Neither backend
guarantees a featureless PDF: retained interval probabilities can require real
peaks, and all-node reconstruction can retain the original sampling spikes.

The grid only samples the fitted function. Increasing `grid` does not change
the adaptive anchors or smooth the CDF/PDF. A plotted polyline or numerical
integration of a coarse output grid can miss them, even though the analytic PDF
integral is one. Output is not renormalized on the sampled grid. `scale: log`
changes query spacing only; fitting and density units remain in physical `x`.

## Reconstruction diagnostics

The output column contract is unchanged. Compact diagnostics are retained in
`DataFrame.attrs['distribution_1d']` and emitted in debug logs. The metadata
contains algorithm revision, interpolation, anchor method, common support, and
one record per repeat with:

- `repeat`: batch identifier (None for an ungrouped input).
- `n_anchors`, `n_candidates`: retained and original node counts, including support endpoints.
- `max_cdf_error`: maximum absolute error at candidate nodes.
- `stop_reason`: `tolerance`, `min_mass`, `max_points`, or `all`.
- `largest_empirical_jump`: largest normalized mass at one distinct raw value.

Metadata does not retain full raw anchor arrays. CSV exports do not encode
DataFrame attributes; inspect the logs or the in-memory transformed table for
diagnostics. Warnings identify unmet tolerance, substantial empirical jumps,
and mass on the lower support boundary.

## Repeat statistics and output

Each repeat is independently normalized and reconstructed on the same support
and grid. Compute PDF derivatives before averaging. Repeat means are equally
weighted, even when the batches have different sample counts or total weights;
the implementation accumulates statistics without retaining a repeats-by-grid
matrix. It does not pool repeat samples or differentiate CDF uncertainty bands.

| Output column | Meaning |
| --- | --- |
| `x` (or configured name) | Common evaluation grid |
| `cdf` | Mean reconstructed CDF |
| `cdf_std` | Sample standard deviation of repeat CDFs, `ddof=1` |
| `pdf` | Mean repeat PDF; `PDF1D` only |
| `pdf_std` | Sample standard deviation of repeat PDFs, `ddof=1`; `PDF1D` only |
| `n_repeats` | Number of batches at every grid point |

Without `repeat`, the whole input is one batch. For one batch, standard
deviations are `NaN` because a sample standard deviation is undefined. These
columns are **standard deviations**, not errors on the mean. For a standard
error under an independent-repeat assumption, use the ordinary expression
`pdf_std / sqrt(n_repeats)` (or its CDF equivalent).

Draw `x` against `pdf` or `cdf` with `plot`; use `fill_between` with
`y1: {expr: pdf - pdf_std}` and `y2: {expr: pdf + pdf_std}` for a band. The
mean ± std band is descriptive and can extend below zero. The transform does
not clip it. A train/test ratio of mean PDFs differs from the mean of ratios
computed per repeat; this transform emits summary curves, not paired ratios.

Schema validation checks the closed configuration. Column demand includes
sample/weight expressions and the repeat column, and registers all output
columns. Both dataset and layer dispatch use one runtime implementation; the
algorithm revision participates in dataset fingerprints and layer cache signatures. Dryrun follows the
existing heavy-step policy and skips reconstruction; render or a full data
transform run checks numerical constraints.

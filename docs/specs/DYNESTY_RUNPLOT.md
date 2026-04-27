# Dynesty Runplot Format

Status: implemented

## Purpose

Jarvis-PLOT provides a reusable `dynesty_runplot` format for Jarvis-HEP
dynesty result CSV files.

The goal is that a normal YAML figure only names the data source and selects
the format. The style card owns the standard five axes, labels, render layer,
KDE smoothing, scatter overlays, evidence band, and summary annotation.

## Minimal YAML

The data source must be named `dynesty` unless the card layer is overridden.

```yaml
DataSet:
- name: dynesty
  path: path/to/dynesty_result.csv
  type: csv

Figures:
- name: dynesty_logL_vs_logX
  enable: true
  style:
  - a4paper_2x1
  - dynesty_runplot
```

The style card lives at:

```text
jarvisplot/cards/a4paper/2x1/dynesty_runplot.json
```

It is registered through:

```text
jarvisplot/cards/style_preference.json
```

## Data Columns

The runtime accepts the Jarvis-HEP dynesty CSV column names by default:

- `log_PriorVolume`
- `log_Like`
- `log_weight`
- `log_Evidence`
- `log_Evidence_err`
- `samples_nlive`
- `samples_it`

The implementation also accepts dynesty-like aliases such as `logvol`, `logl`,
`logwt`, `logz`, `logzerr`, and `samples_n`.

Custom column names can be provided in the layer style under `columns` when a
YAML figure overrides the default layer.

## Panels

The default card draws five stacked axes:

- `ax0`: live points, scatter only
- `ax1`: likelihood normalized to its maximum value, runplot curve plus scatter overlay
- `ax2`: importance weight PDF, dynesty-style KDE curve plus scatter overlay
- `ax3`: evidence, curve plus one error band and final-value annotation
- `ax4`: iteration index, scatter only

`ax0` through `ax3` keep x-axis tick marks but hide x tick labels. `ax4` shows
the shared x tick labels.

## Runplot Semantics

The importance-weight curve follows dynesty `runplot` semantics:

- weights are computed as `exp(log_weight - final_logZ)`
- samples are equal-weight resampled with systematic resampling
- `scipy.stats.gaussian_kde` evaluates the probability density on an `nkde` grid
- the KDE curve remains area-normalized as a PDF

The scatter overlay for the importance panel is only a visual companion. The
default `importance_scatter_normalize: pdf_peak` keeps its relative shape and
scales it to the KDE peak height so it remains visible on the same panel.

## Default Style Controls

The default `Style.dynesty_runplot` includes:

```json
{
  "axes": ["ax0", "ax1", "ax2", "ax3", "ax4"],
  "kde": true,
  "nkde": 1000,
  "lnz_error": true,
  "lnz_error_levels": [1],
  "evidence_summary": true,
  "ylim_factor": 1.3,
  "scatter_panels": ["nlive", "iters"],
  "overlay_scatter_panels": ["likelihood", "importance", "evidence"],
  "importance_scatter_normalize": "pdf_peak",
  "set_labels": false
}
```

`ylim_factor` sets the default y range to `[0, ylim_factor * ymax]`.
For evidence, `ymax` includes the configured evidence error band.

`evidence_summary` draws a horizontal line at the final evidence value and
places a left-side annotation formatted as `Z ± sigma_Z`. For linear evidence
plots, the error is converted from `log_Evidence_err`.

## Override Rules

The `dynesty_runplot` card provides a default `Layers` block. A YAML figure that
defines its own `layers` replaces the card layers completely. A YAML `frame`
still merges into the card `Frame`, so individual axes can be adjusted without
rewriting the layer.

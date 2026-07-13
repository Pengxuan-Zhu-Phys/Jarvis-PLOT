# Agent Data API

Status: spec only
Last updated: 2026-07-03
Design authority: this document (wire contract for the Jarvis-Agent ↔ Jarvis-PLOT bridge).
Agent-side tool spec: `Jarvis-Agent/docs/PLOT_TOOLS.md`.
Related workshop precedent: `Jarvis-Books/Docs/DESIGN_AGENT_BRIDGE_2.0.md` (the Jarvis-HEP V2
bridge; this API reuses its envelope and handshake conventions).

## 1. Purpose

Jarvis-Agent drives a **non-multimodal local model**: it can never read a rendered figure.
This API gives the agent a numeric channel into the same statistics the figures show, plus a
machine-readable way to validate configs, generate templates, and trigger renders for the
human.

Two entries, one body:

- **Human entry (unchanged)**: `jplot config.yaml` — YAML in, figures out.
- **Agent entry (this spec)**: additive `jplot` verbs with a JSON envelope — structured
  specs in, structured data + bounded digests out.

Both entries must converge on the same loader / transform / cache engine
(`data_loader*`, `preprocessor_runtime`, `profile_runtime`, `.cache/` fingerprints). The
agent channel is a *skin*, never a fork of the pipeline.

## 2. Envelope and Handshake

Every agent verb prints exactly one JSON object on stdout (humans/logs on stderr):

```json
{"api_version": 1, "kind": "<verb>", "ok": true, "data": {...}, "error": null}
```

`ok=false` ⇒ `error={"type","message"}` with actionable text. Exit codes: 0 ok, 1 failed,
2 usage. `jplot --version-json` returns `{package_version, api_version}`; the agent caches it
per session and refuses on `api_version` mismatch (no silent text-parsing fallback).

## 3. Verbs

| Verb | Form | Purpose |
|------|------|---------|
| Version | `jplot --version-json` | handshake |
| Describe | `jplot --describe <data-file> --json [--type hdf5\|csv] [--group G]` | dataset summary: columns, dtypes, row count, per-column range/quantiles, HDF5 tree. Wraps `data_loader_summary.py` |
| Validate | `jplot <config.yaml> --validate --json` | config check without rendering: schema (jsonschema is already a dependency), dataset paths exist, referenced columns resolvable, style cards found. Diagnostics list `{level, path, message}` |
| Template | `jplot --template [<kind>] --json` | template standard interface (§5): no kind = list catalog; with kind = emit template YAML + slot schema |
| Analyze | `jplot --analyze <spec.json\|yaml> --json` | headless analysis channel (§4): transforms + report reducers, **no figure** |
| Render | `jplot <config.yaml> --render-json [--with-data]` | normal render with machine-readable outcome: figures written, per-figure status, warnings, cache hits. `--with-data` also exports each figure's prepared plot-ready table (§6) |

## 4. Analysis Channel

### 4.1 Analysis spec

JSON or YAML; deliberately the same shape as a dataset + transform block so it normalizes
into the existing pipeline spec:

```yaml
data: {path: outputs/scan/DATABASE/samples.hdf5, type: hdf5}
workdir: .                       # cache + artifact anchor
transform: []                    # optional pre-steps: filter / add_column / ...
report:
  kind: likelihood_report        # reducer (§4.2); one report per analyze call
  x: {expr: m0,  lim: [0, 5000]}
  y: {expr: m12, lim: [0, 3000]}
  objective: LogL                # profile side
  weight: exp(LogL)              # posterior side; omit to skip posterior
  bins: 64                       # support cells (bridson default, as make_density_core)
  levels: [0.68, 0.95]
  max_regions: 8                 # digest cap; remainder aggregated into "other"
out_dir: null                    # default <workdir>/.cache/agent/<spec-fingerprint>/
```

Expressions go through the existing centralized surface (`utils/expression.py`) and inherit
its trusted-input assumption — the agent is a trusted local caller; no new eval surface is
added.

### 4.2 Reducer: `likelihood_report` (flagship)

Voronoi-cellularized likelihood analysis. Composition of existing owners plus one new
algorithmic piece:

1. **Profile side** — `profile` transform (bridson/grid, objective max) → per-cell
   `logl_max` → PL ratio vs global best → CL membership per level (χ², 2 dof for 2D).
2. **Posterior side** — `make_density_core` (same support cells) → per-cell mass →
   HPD membership via the integrated-probability thresholds in `posterior_hpd.py`.
3. **Region extraction (NEW)** — connected components over cell adjacency (Delaunay
   neighbors of support points; grid neighbors for `method: grid`) per CL/HPD level →
   per-region: id, n_cells, area fraction, bbox, best/mode point, local `logl_max` / mass
   share.
4. **1D sections** — per axis: profile curve (binned max), PL intervals per level;
   weighted marginal quantiles + HPD intervals; mode count.

**Artifacts** (full data, file-based):

- `cells.parquet` — one row per support cell:
  `cell_id, x, y, area, n_samples, logl_max, pl_ratio, in_cl68, in_cl95, region68_id,
  region95_id, mass, in_hpd68, in_hpd95, hpd_region_id`
- `profile_1d.csv` / `marginal_1d.csv` — per-axis curves
- `regions.geojson` — optional cell-union polygons per region (shapely is a dependency)

**Digest** (`report.json`, returned in `data`; budget ≤ ~4 KB):

```json
{
  "n_samples": 182340, "data_fingerprint": "…",
  "axes": {"x": {"expr": "m0", "lim": [0, 5000]}, "y": {"expr": "m12", "lim": [0, 3000]}},
  "profile": {
    "best_fit": {"x": 812.3, "y": 402.1, "logl": -12.02},
    "cl_regions": {"0.95": [{"id": 1, "n_cells": 118, "area_frac": 0.031,
                              "bbox": [700, 950, 350, 520],
                              "peak": {"x": 812.3, "y": 402.1, "logl": -12.02}}]},
    "n_disconnected": {"0.68": 1, "0.95": 2},
    "profile_1d": {"x": {"intervals": {"0.68": [[712, 1005]], "0.95": [[640, 1213]]},
                          "n_modes": 1}}
  },
  "posterior": {
    "mode": {"x": 805.0, "y": 410.5}, "mean": {"x": 990.2, "y": 471.8},
    "hpd_regions": {"0.95": [{"id": 1, "mass": 0.93, "bbox": [640, 1300, 300, 700]}]},
    "marginal_1d": {"x": {"quantiles": {"0.05": 700.1, "0.5": 950.0, "0.95": 1400.2},
                            "hpd": {"0.68": [[712, 1080]]}}}
  },
  "notes": ["profile 95% CL splits into 2 regions; posterior keeps 93% mass in region 1"],
  "artifacts": {"cells": "…/cells.parquet", "regions_geojson": "…/regions.geojson"}
}
```

Digest discipline: regions capped at `max_regions` (rest aggregated with combined
area/mass); 1D **arrays never enter the digest** — only interval endpoints and quantiles;
full curves live in artifacts.

### 4.3 Additional reducers (same channel, smaller)

| kind | Output |
|------|--------|
| `summary_stats` | per-column count/min/max/mean/quantiles + top-k |corr| pairs |
| `top_points` | best N rows by an objective, selected columns |
| `interval_report` | 1D-only PL + posterior intervals for a parameter list (no 2D cells) |
| `compare_report` | two datasets: best-fit shift, per-level region-overlap fraction, interval deltas. **Phase 2** — spec to be extended before implementation |

## 5. Template Standard Interface

The template generator becomes a first-class contract shared by both entries; the agent
fills slots, the human edits YAML — same catalog, same defaults (packaged cards +
`docs/templates/`).

- `jplot --template --json` → catalog: `[{kind, title, family: rect|ternary|runplot,
  requires: [x, y, weight?], description}]`. First catalog: `profile2d_rect`,
  `posterior2d_rect`, `profile_posterior_2x1`, `ternary_cmap`, `dynesty_runplot`.
- `jplot --template <kind> --json` → `{yaml_text, slots}` where each slot is
  `{path, type, required, enum?, default?, description, source_hint}`.
  `source_hint ∈ {dataset_column, axis_limits, output_path, style_choice, free_text}` tells
  the agent *where an answer should come from* (e.g. a `dataset_column` slot is filled from
  a prior `--describe` call, never invented) — the anti-hallucination hook.
- Filled templates are ordinary `jplot` YAML: the agent writes them to its `proposals/`
  area, validates with `--validate --json`, renders with `--render-json`. Human entry is
  untouched.

## 6. Figure Numeric Twins (`--render-json --with-data`)

The cell report alone is not the whole answer: whatever a figure shows, the agent must be
able to read numerically. With `--with-data`, every rendered figure/layer exports its
**prepared plot-ready table** (the DataFrame state after the layer's transform chain —
exactly what `pcolormesh`/`contour`/`scatter` consumed) as parquet next to the figure,
listed in the JSON outcome:

```json
{"figures": [{"name": "mass_plane", "file": "plots/mass_plane.pdf",
               "status": "ok", "cache": "hit",
               "data_sidecars": [{"layer": "pl_ratio_mesh", "rows": 90000,
                                    "path": "plots/.data/mass_plane.pl_ratio_mesh.parquet",
                                    "columns": ["x", "y", "z"]}]}],
 "warnings": []}
```

Sidecars are bounded by row count in the outcome JSON (paths + shapes only, never inline
data). This reuses the existing `to_parquet` transform as the export primitive.

## 7. Artifact and Cache Layout

```
<workdir>/.cache/agent/<spec-fingerprint>/   # analyze artifacts (report.json, cells.parquet, …)
<output.dir>/.data/                          # render --with-data sidecars
```

Analyze reuses the standard `.cache` fingerprint rules: same data fingerprint + same spec ⇒
cache hit, artifacts returned without recomputation.

## 8. Work Breakdown (roadmap backlog `JP-A*`)

| WP | Delivers | Depends |
|----|----------|---------|
| JP-A1 | envelope + `--version-json` + `--validate --json` | — |
| JP-A2 | `--describe --json` (wrap `data_loader_summary`) | JP-A1 |
| JP-A3 | `--analyze` channel + `likelihood_report` (+ `summary_stats`, `top_points`, `interval_report`); region extraction is the main new algorithm | JP-A1 |
| JP-A4 | `--template` catalog + slot schemas | JP-A1 |
| JP-A5 | `--render-json` + `--with-data` sidecars | JP-A1 |

Agent-side consumption = Jarvis-Agent milestone **M4.6 "Analysis Loop"**
(`Jarvis-Agent/docs/PLOT_TOOLS.md`); it depends on JP-A1–A3, with A4/A5 upgrading the loop.
`compare_report` is Phase 2.

## 9. Non-Goals

- No image understanding, no GUI, no long-lived server; verbs are one-shot processes.
- No new expression-evaluation surface beyond `utils/expression.py`.
- The human YAML surface and rendered outputs do not change shape under this spec.

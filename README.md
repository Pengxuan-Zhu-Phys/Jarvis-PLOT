# Jarvis-PLOT

Jarvis-PLOT is a lightweight, Python/Matplotlib-based plotting framework developed for **Jarvis-HEP**,  
but it can also be used as a **standalone scientific plotting tool**.

It provides a simple command-line interface (CLI) to generate publication-quality figures from YAML configuration files, with most layout and style decisions handled by predefined profiles and style cards.

Current source candidate: **2.1.11** (`jplot -v` reports the installed version).
Release history: [CHANGELOG.md](CHANGELOG.md). Design documents and specifications: [Jarvis-Books `Docs/Jarvis-PLOT/`](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-PLOT/README.md).

---

## Installation

```bash
pip install JarvisPLOT
```

The PyPI / distribution name is `JarvisPLOT` (normalized wheel name: `jarvisplot`).
The product name remains **Jarvis-PLOT**.
The Python import package and entrypoint remain unchanged:

```python
import jarvisplot
```

```bash
jplot path/to/config.yaml
```

If you have an older environment that still uses a historical package name, replace:

```bash
pip uninstall jarvisplot Jarvis-PLOT JarvisPLOT
pip install JarvisPLOT
```

## Command-Line Usage

Human help uses the same fixed-column Rich panels as Jarvis-HEP V2 / Portal;
`jplot -h` prints the verb map, and `jplot <verb> -h` opens one verb.

Render with a bare path — there is no `jplot run`:

```bash
jplot path/to/config.yaml
jplot path/to/config.yaml --rebuild-cache
jplot path/to/config.yaml --print
```

Everything else is discovery and judgement, in three groups.

**Discover** — find out what is legal before writing anything:

```bash
jplot data describe samples.csv     # columns, dtypes, ranges (HDF5 tree too)
jplot data head samples.csv -n 5    # real sample rows
jplot data eval 'exp(LogL)' --data samples.csv    # sandbox an expr
jplot data suggest-axes samples.csv # scale / lim hints for frame.ax
jplot cap                           # section index
jplot cap methods                   # all | methods | transforms | types
jplot cap styles                    #   | styles | cmaps | funcs | cli
jplot man                           # manual index
jplot man workflow                  # topic page
jplot man scatter                   # one method's contract
jplot explain JP-COL-001            # a diagnostic code
```

**Draft & edit** — scaffold or address a YAML by name:

```bash
jplot template list                 # template kinds
jplot template show posterior_2d    # one template + its slots
jplot suggest --data samples.csv --kind posterior_2d
jplot config paths plot.yaml        # named addresses in this file
jplot config get plot.yaml 'Figures[p].style'
jplot config set plot.yaml 'Figures[p].frame.ax.xlim' '[0, 10]' --write
jplot config expand plot.yaml --figure p --write   # type: → layers
```

**Judge** — check without rendering:

```bash
jplot validate plot.yaml            # schema/contracts + source paths/column headers; no transforms
jplot dryrun plot.yaml              # load data + transforms, row ledger, JP-VIZ health
jplot doctor plot.yaml              # validate + dryrun in one pass
```

Every one of these takes `--json`.

### Default fonts

The bundled font card recommends **Helvetica** for sans text and **STIXGeneral**
for serif text, with **STIX** MathText formulas. Normal YAML rendering applies
the card automatically; explicit YAML fonts and sizes take precedence.
When Helvetica is absent, Jarvis-PLOT selects the first installed similar family:
Arial, Liberation Sans, Nimbus Sans, then DejaVu Sans. `check` and `resolve`
report the actual font and substitution reason.

```bash
jplot fonts                         # role card: title, labels, ticks, legend, text, annotations, caption
jplot fonts show title --json       # current defaults for one role
jplot fonts show --recommended --json
jplot fonts list --filter Helvetica --json
jplot fonts list --refresh --json   # rescan after installing fonts
jplot fonts resolve Helvetica --weight bold --json
jplot fonts check --json            # actual installed faces used by the card
jplot fonts set roles.title.family sans
jplot fonts set roles.legend.family serif
jplot fonts set roles.title.size_pt 9
jplot fonts set roles.title.size_pt null  # inherit the plot style's size
jplot fonts reset roles.title.size_pt
jplot fonts reset                   # remove all personal overrides
jplot man fonts --json
```

Personal overrides persist in `~/.jarvis/plot-fonts.json` (`JARVIS_HOME` overrides
the directory). Queries never create that file. `size_pt: null` preserves the
plot style's canvas-specific size; `recommended_size_pt` is drafting advice.
Font discovery uses Matplotlib's cached system and bundled font index, while
TTC face selection preserves the requested weight and slant. `$...$` uses
MathText; an external LaTeX compiler has its own font configuration.

### Design-Reference Overlay

Set `debug: true` on a figure and Jarvis-PLOT draws a dimension overlay on top
of it: every axes box labelled with its rect and its size in centimetres, page
margins, the colorbar gap, and an "axes layout" panel. It is meant for checking
a layout, so a figure with `layers: []` is a legitimate use.

```yaml
Figures:
  - name: layout_check
    debug: true
    style: [a4paper_2x1, rectcmap]
    layers: []
```

The style card's `Debug` block decides what the annotations look like and which
of them appear; a per-figure `debug:` mapping overrides the card for that figure
only. See `jplot man debug-overlay` and [STYLE_SCHEMA.md](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-PLOT/specs/STYLE_SCHEMA.md).

> `Figures[].debug` is not the `jplot --debug` flag — the flag only raises the
> log level.

## For Coding Agents

Jarvis-PLOT is built so a coding agent can drive it without ever seeing a
rendered figure. The rule is **the agent writes the YAML itself**; the CLI's job
is to say what is legal and to judge what was written. There is no verb that
generates a finished config for you.

Every discovery and judging verb prints **one JSON envelope on stdout**, with
humans and logs on stderr:

```json
{"api_version": 1, "kind": "cap.methods", "ok": true,
 "data": {...}, "diagnostics": [], "error": null}
```

`--json` turns it on, and it is **already the default when stdout is not a
TTY** — piping is enough.

The loop `jplot man` prints is:

Before drafting, inspect `jplot fonts show --json` and `jplot fonts check --json`
for the typography defaults and actual installed font choices.

1. `jplot data describe <file> --json` — column names from the real file. Never
   invent one; a wrong name is a hard error, not a warning.
2. `jplot cap methods|styles|cmaps|funcs|transforms|types --json` — the legal
   strings, read live from the registries rather than from documentation.
3. Edit the YAML in an editor (or `jplot config set … --write`). Reach for a
   `type:` macro first, drop to explicit `layers` when the macro cannot say it.
4. `jplot doctor <file> --json` — validate + dryrun in one pass. Diagnostics
   carry `JP-*` codes; `jplot explain JP-XXX-NNN` expands any of them.
5. `jplot <file>` — render, only when the figure itself is needed.

`jplot man --json` is the machine-readable manual: a topic index, plus
live pages for every registered layer method and transform step
(`jplot man scatter`, `jplot man transform.profile`, `jplot man PDF1D`). Method contracts come from
the same registry `jplot cap methods` reads, so a man page cannot drift from the
code.

For compact legends and keys composed from real layers, use
`jplot man legend --json`. Declare `legend: {object, label, role}` on each drawing layer; components
with the same object regroup automatically using card templates such as
`band_line`, `line_errorbar` and `error_bounds`. The role is optional for a
single-component object. `frame.<axes>.legend` owns card selection, `order`,
`ncols`, position and spacing. JSON cards provide
reusable defaults, and YAML can override them. See
[the legend contract](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-PLOT/specs/LEGEND.md) and the
[self-contained example](Example/compact_legend.yaml).

For continuous one-dimensional distributions from raw samples, use `PDF1D`
(CDF plus PDF) or `CDF1D` in a data transform. `coordinates.x` owns `expr`,
`lim`, and `grid`; optional `repeat` computes independent batch means and
standard deviations. See [the contract](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-PLOT/specs/PDF_CDF_1D.md) and the
[runnable example](Example/pdf_cdf_1d.yaml).
Reconstruction defaults to adaptive true CDF anchors and `monotone_c2`
interpolation (C2 CDF / C1 PDF). Set `anchors: all` and `interpolation: pchip`
to reproduce the original all-node reconstruction.

Two things worth knowing before trusting a config:

- `jplot validate` checks shape and contracts and never touches the data;
  `jplot dryrun` loads the data and reports the row ledger and JP-VIZ render
  health. Passing the first says nothing about the second.
- [AGENT_DATA_API.md](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-PLOT/specs/AGENT_DATA_API.md) describes a *different*, still-frozen bridge
  (a numeric digest channel for Jarvis-Agent). The surface above is what ships
  today.

---

## Flowchart Rendering

Render a Jarvis-HEP flowchart scene JSON (produced by HEP scan tooling — this is
not general plot YAML):

```bash
jplot flowchart path/to/scene.json
# optional output path:
jplot flowchart path/to/scene.json -o path/to/out.png
```

Library form:

```python
from jarvisplot import render_flowchart, render_flowchart_file
```

### Project Workdir and Cache

- You can set `project.workdir` in YAML.
- If `output.dir` is omitted, Jarvis-PLOT defaults to `<workdir>/plots/`.
- Data cache is stored in `<workdir>/.cache/`.
- `--print` accepts the current settings and renders final figures without the logo. After all enabled figures are saved successfully, it requests deletion of this run's cache payloads. Cleanup waits until every reader of the workdir has finished. Rendering failures or a run with no enabled figures do not request immediate cleanup; ordinary age and capacity limits still apply. Unrelated entries and source fingerprints are retained unless the whole idle cache is evicted.
- Profiling pipelines are prebuilt once and reused from cache when source fingerprint and profile settings are unchanged.
- Profiling uses a fast two-stage grid reduction (`pregrid` + render `bin`) for large datasets.

Plot caches are automatically registered in `~/.jarvis/plot.json` (or
`$JARVIS_HOME/plot.json`). The registry groups caches by workdir and records
associated YAML paths and last use; payloads stay in their original directories.
This includes caches created by `data describe` and default `dryrun --with-data`
exports. Sizes are calculated when listing, rather than stored in the registry.

```bash
jplot cache list
jplot cache scan ~/Jarvis-Workshop               # register existing plot caches
jplot cache clean --workdir /path/to/project    # this directory and all nested projects; repeatable
jplot cache clean --all --older-than 30         # unused for at least 30 days
jplot cache clean --all --dry-run               # preview; no deletion/index changes
jplot cache clean --all                         # clean all registered inactive caches
jplot cache -h
jplot man cache
```

Automatic maintenance runs when a cache session starts and closes. The default
global budget is **5 GiB** of registered cache file sizes; idle workdirs unused
for **30 days** expire. Above the budget, the least recently used idle workdirs
are removed until usage falls to 80% of the limit (4 GiB by default). Active
workdirs may temporarily exceed the budget and are reconsidered after closing.
Eviction works on whole workdirs; it does not track each file's access time.
There is no background timer, and unregistered caches require `cache scan` first.

Configure positive finite limits in `~/.jarvis/plot-cache.json`
(`$JARVIS_HOME/plot-cache.json` when overridden):

```json
{"max_size_gib": 5, "max_age_days": 30}
```

`JP_CACHE_MAX_GIB` and `JP_CACHE_MAX_AGE_DAYS` override the corresponding file
settings. For example, `JP_CACHE_MAX_GIB=10 JP_CACHE_MAX_AGE_DAYS=14 jplot plot.yaml`.
`jplot cache list` shows the effective policy; its JSON adds `data.policy`.
Invalid limits fail before rendering or automatic deletion.

`--rebuild-cache` requires exclusive access for the entire session and reports
an error if the workdir is active. A normal session also reports an error if a
rebuild holds the exclusive lease. Rebuild and cleanup remove only owned cache
components, preserving source files, figures, and unrelated `.cache` content.
Interrupted temporary writes and superseded HDF5 generations are reclaimed
when the workdir becomes idle. Failure to acquire a lifetime lock stops cache
use; index-registration failures with a valid lock remain warnings.

`--workdir PATH` selects registered caches in PATH itself and all of its
subdirectories, even when PATH has no cache of its own. Repeat it to select
multiple directory trees; overlapping selections are deduplicated. Use
`cache scan PATH` first for old caches that have not been registered.

All cache subcommands accept `--json` and use the existing CLI envelope and
exit codes (0 success/partial, 1 failure, 2 usage). Non-TTY stdout defaults to
JSON; human results go to stderr. Active caches are skipped (`ok: null` when
the remaining operation succeeds), and records for missing paths or unmarked
`.cache` directories containing only empty Jarvis-PLOT component directories
are pruned during cleanup. Only marked
Jarvis-PLOT components are removed; source data,
plots, and unrelated `.cache` content are retained. Failed deletions retain
their registry entries; nonempty unmarked directories remain invalid and are
not deleted. Scanning adopts legacy caches only when the four
pipeline directories and a compatible `manifest.json` identify them; old
unmarked twin-only directories need a new `dryrun --with-data` run to register.
An explicitly supplied `--out-dir` is an export directory and is not registered.
Concurrent registry writes use a lock and atomic replacement. Registration
failures warn without preventing rendering.

### Example: Ternary Layouts

```bash
jplot ./Example/gambit_2x1.yaml
```

This layout reference uses bundled style cards and renders without external data.

### Example: Dynesty Runplot

Jarvis-PLOT includes a reusable dynesty runplot format. With a dataset named
`dynesty`, the figure can use the built-in card without writing axes or layer
details:

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

See [DYNESTY_RUNPLOT.md](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-PLOT/specs/DYNESTY_RUNPLOT.md) for the default axes, KDE, scatter overlay,
and evidence summary behavior.

---

## Notes

- Figures are saved automatically to the output paths defined in the YAML configuration.
- Common output formats include PNG and PDF (backend-dependent).
- Saved figures include file metadata such as `Creator: Jarvis-PLOT, powered by Jarvis-HEP` and `Jarvis-PLOT version: X.Y.Z`; the PNG `Description` also includes both fields for macOS Finder compatibility.
- Jarvis-PLOT works in headless environments (SSH, batch jobs) without any GUI backend.

---

## Requirements

### Python
- **Python ≥ 3.10** (tested on 3.10–3.13)

### Required Packages
- `numpy`
- `pandas`
- `polars`
- `matplotlib`
- `pyyaml`
- `jsonschema`
- `scipy` — numerical utilities
- `h5py` — required for loading HDF5 data files
- `shapely`
- `sympy`
- `loguru`
- `deepmerge`
- `ruamel.yaml>=0.18` — round-trip YAML editing for `jplot config set`
- `rich` — the fixed-column help and manual panels
- `Jarvis-Operas>=1.1.4`

### Github Page
[https://github.com/Pengxuan-Zhu-Phys/Jarvis-PLOT](https://github.com/Pengxuan-Zhu-Phys/Jarvis-PLOT)

### Documentation
[https://pengxuan-zhu-phys.github.io/Jarvis-Docs/](https://pengxuan-zhu-phys.github.io/Jarvis-Docs/)

### Project Docs

This repository keeps only `README.md` and `CHANGELOG.md`. Design documents,
specifications and the roadmap live in Jarvis-Books:

- [Doc index](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-PLOT/README.md)
- [JARVIS_PLOT_CONTEXT.md](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-PLOT/context/JARVIS_PLOT_CONTEXT.md) - project boundary; read before changing code
- [CODE_MAP_JARVIS_PLOT.md](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-PLOT/context/CODE_MAP_JARVIS_PLOT.md) - code owner map
- [JARVIS_PLOT_FRAMEWORK_LOGIC.md](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-PLOT/context/JARVIS_PLOT_FRAMEWORK_LOGIC.md) - runtime execution contract
- [IMPLEMENTATION_ROADMAP.md](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-PLOT/roadmap/IMPLEMENTATION_ROADMAP.md) - open work
- [RELEASING.md](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-PLOT/release/RELEASING.md) - versioning, `tools/check_release.py` and CI verification
- [STYLE_SCHEMA.md](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-PLOT/specs/STYLE_SCHEMA.md) - style card contract, including the `Debug` overlay block
- [DEVELOPER_RULES.md](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-PLOT/dev/DEVELOPER_RULES.md) - pipeline and cache rules

Read the context docs and roadmap before changing parsing, transforms, rendering, or layout-related behavior.

---

## License

MIT License

Create a continuous forest figure with `type: forest`, a generic row table, optional points and multiple supplied intervals per row. Native YAML supports class-specific interval lanes, CSV value annotations, explicit `x.ticks`, opt-in MathText, header notes and a shared-format legend. The default Swiss-column card uses 82.5 mm width and 5.5 mm row pitch; every row shares one numerical x domain. See [the forest contract](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-PLOT/specs/FOREST.md), [native examples](Example/forest/README.md), or `jplot man forest --json`.

AI agents: start with `jplot man agent-readme --json` for the official-package YAML workflow and missing-feature reporting instructions. Feature requests include the installed version from `jplot --version`; the CLI does not send email automatically.

## Releasing

`jprel` is a maintainer-local script, excluded from Git and distribution archives.

Record changes in `CHANGELOG.md`, then run from this checkout (replace the
example version with the intended release):

```bash
./jprel 2.1.11 --dry
./jprel 2.1.11
```

`--dry` performs read-only source and release-note checks. The normal command
requires a clean Git checkout, builds and checks the wheel and sdist, uploads
to PyPI, refreshes the local version with `python3 -m pip install -e .`, and
verifies the installed version. Only then does it commit the version, push the
`vX.Y.Z` tag, and create a GitHub release with changelog notes. Configure Twine
and `gh` authentication before publishing.

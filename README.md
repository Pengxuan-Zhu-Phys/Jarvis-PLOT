# Jarvis-PLOT

Jarvis-PLOT is a lightweight, Python/Matplotlib-based plotting framework developed for **Jarvis-HEP**,  
but it can also be used as a **standalone scientific plotting tool**.

It provides a simple command-line interface (CLI) to generate publication-quality figures from YAML configuration files, with most layout and style decisions handled by predefined profiles and style cards.

Current release: **2.0.6** (`jplot -v` reports the installed version).

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
jplot validate plot.yaml            # schema + contracts, no data touched
jplot dryrun plot.yaml              # load data + transforms, row ledger, JP-VIZ health
jplot doctor plot.yaml              # validate + dryrun in one pass
```

Every one of these takes `--json`.

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
only. See `jplot man debug-overlay` and `docs/specs/STYLE_SCHEMA.md`.

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

1. `jplot data describe <file> --json` — column names from the real file. Never
   invent one; a wrong name is a hard error, not a warning.
2. `jplot cap methods|styles|cmaps|funcs|transforms|types --json` — the legal
   strings, read live from the registries rather than from documentation.
3. Edit the YAML in an editor (or `jplot config set … --write`). Reach for a
   `type:` macro first, drop to explicit `layers` when the macro cannot say it.
4. `jplot doctor <file> --json` — validate + dryrun in one pass. Diagnostics
   carry `JP-*` codes; `jplot explain JP-XXX-NNN` expands any of them.
5. `jplot <file>` — render, only when the figure itself is needed.

`jplot man --json` is the machine-readable manual: 15 topics, plus a
live page for every one of the 28 layer methods and 11 transform steps
(`jplot man scatter`, `jplot man transform.profile`). Method contracts come from
the same registry `jplot cap methods` reads, so a man page cannot drift from the
code.

Two things worth knowing before trusting a config:

- `jplot validate` checks shape and contracts and never touches the data;
  `jplot dryrun` loads the data and reports the row ledger and JP-VIZ render
  health. Passing the first says nothing about the second.
- `docs/specs/AGENT_DATA_API.md` describes a *different*, still-frozen bridge
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
- Profiling pipelines are prebuilt once and reused from cache when source fingerprint and profile settings are unchanged.
- Profiling uses a fast two-stage grid reduction (`pregrid` + render `bin`) for large datasets.

### Example: SUSYRun2 Ternary Plots

```bash
jplot ./bin/SUSYRun2_EWMSSM.yaml
jplot ./bin/SUSYRun2_GEWMSSM.yaml
```

> **Note:** The data file paths inside the YAML files must be updated to match your local setup.

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

See `docs/specs/DYNESTY_RUNPLOT.md` for the default axes, KDE, scatter overlay,
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

### Repository Docs

Tracked project docs live in `docs/`.

- `docs/README.md` - repo doc index
- `docs/context/JARVIS_PLOT_CONTEXT.md` - primary Codex-facing boundary doc
- `docs/context/CODE_MAP_JARVIS_PLOT.md` - concrete code owner map
- `docs/context/JARVIS_PLOT_FRAMEWORK_LOGIC.md` - runtime execution contract
- `docs/roadmap/IMPLEMENTATION_ROADMAP.md` - active backlog and future work list
- `docs/specs/AGENT_DATA_API.md` - planned numeric agent bridge (spec only, frozen — not the shipped CLI agent surface above)
- `docs/specs/STYLE_SCHEMA.md` - style card contract, including the `Debug` overlay block
- `docs/dev/DEVELOPER_RULES.md` - current pipeline and cache rules
- `docs/dev/MEMORY_OPTIMIZATION_GUIDE.md` - narrow-table memory notes

Read the context docs and roadmap before changing parsing, transforms, rendering, or layout-related behavior.

---

## License

MIT License

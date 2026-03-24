# Jarvis-PLOT

Jarvis-PLOT is a lightweight, Python/Matplotlib-based plotting framework developed for **Jarvis-HEP**,  
but it can also be used as a **standalone scientific plotting tool**.

It provides a simple command-line interface (CLI) to generate publication-quality figures from YAML configuration files, with most layout and style decisions handled by predefined profiles and style cards.

---

## Installation

```bash
pip install Jarvis-PLOT
```

The PyPI distribution name is now `Jarvis-PLOT`.
The Python import package and entrypoint remain unchanged:

```python
import jarvisplot
```

```bash
jplot path/to/config.yaml
```

If you have an older environment that still uses the historical package name, replace:

```bash
pip uninstall jarvisplot
pip install Jarvis-PLOT
```

## Command-Line Usage

Display help information:

```bash
jplot -h
```

Run Jarvis-PLOT with one or more YAML configuration files:

```bash
jplot path/to/config.yaml
```

Rebuild local cache for the current project workdir:

```bash
jplot path/to/config.yaml --rebuild-cache
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

---

## Notes

- Figures are saved automatically to the output paths defined in the YAML configuration.
- Common output formats include PNG and PDF (backend-dependent).
- Jarvis-PLOT works in headless environments (SSH, batch jobs) without any GUI backend.

---

## Requirements

### Python
- **Python ≥ 3.10** (tested on 3.10–3.13)

### Required Packages
- `numpy`
- `pandas`
- `matplotlib`
- `pyyaml`
- `jsonschema`
- `scipy` — numerical utilities
- `h5py` — required for loading HDF5 data files
- `shapely`
- `scipy`
- `sympy`

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
- `docs/dev/DEVELOPER_RULES.md` - current pipeline and cache rules
- `docs/dev/MEMORY_OPTIMIZATION_GUIDE.md` - narrow-table memory notes

Read the context docs and roadmap before changing parsing, transforms, rendering, or layout-related behavior.

---

## License

MIT License

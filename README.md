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

### Developer Docs

Tracked architecture and maintenance docs now live in `docs/dev/`.

- `docs/dev/ARCHITECTURE_OVERVIEW.md`
- `docs/dev/DATAFLOW_ARCHITECTURE.md`
- `docs/dev/MEMORY_OPTIMIZATION_GUIDE.md`
- `docs/dev/DEVELOPER_RULES.md`
- `docs/dev/MEMTRACE_SYSTEM.md`
- `docs/dev/CONTRIBUTION_GUIDE.md`

These documents describe the 1.3.0 selection-table pipeline architecture that 1.3.1 consolidates. Future work should preserve the narrow-table dataflow and must not reintroduce wide-table profiling or cache payloads.

---

## License

MIT License

# JarvisPLOT

JarvisPLOT is a lightweight, Python/Matplotlib-based plotting framework developed for **Jarvis-HEP**,  
but it can also be used as a **standalone scientific plotting tool**.

It provides a simple command-line interface (CLI) to generate publication-quality figures from YAML configuration files, with most layout and style decisions handled by predefined profiles and style cards.

---

## Installation 

```bash 
pip install jarvisplot
```

## Command-Line Usage

Display help information:

```bash
jplot -h
```

Run JarvisPLOT with one or more YAML configuration files:

```bash
jplot path/to/config.yaml
```

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
- JarvisPLOT works in headless environments (SSH, batch jobs) without any GUI backend.

---

## Requirements

### Python
- **Python ≥ 3.9** (tested on 3.9–3.12)

### Required Packages
- `numpy`
- `pandas`
- `matplotlib`
- `pyyaml`
- `jsonschema`
- `scipy` — numerical utilities
- `h5py` — required for loading HDF5 data files

### Github Page
[https://github.com/Pengxuan-Zhu-Phys/Jarvis-PLOT](https://github.com/Pengxuan-Zhu-Phys/Jarvis-PLOT)

### Documentation
[https://pengxuan-zhu-phys.github.io/Jarvis-Docs/](https://pengxuan-zhu-phys.github.io/Jarvis-Docs/)

---

## License

MIT License
# JarvisPLOT

JarvisPLOT is a lightweight, Python/Matplotlib-based plotting framework developed for **Jarvis-HEP**,  
but it can also be used as a **standalone scientific plotting tool**.

It provides a simple command-line interface (CLI) to generate publication-quality figures from YAML configuration files, with most layout and style decisions handled by predefined profiles and style cards.

---

## Command-Line Usage

Display help information:

```bash
./jarvisplot -h
```

Run JarvisPLOT with one or more YAML configuration files:

```bash
./jarvisplot path/to/config.yaml
```

### Example: SUSYRun2 Ternary Plots

```bash
./jarvisplot ./bin/SUSYRun2_EWMSSM.yaml
./jarvisplot ./bin/SUSYRun2_GEWMSSM.yaml
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

### Relationship to Jarvis-HEP
- JarvisPLOT is **fully decoupled** from Jarvis-HEP/GAMBIT

---

## Installation

Editable install for development:

```bash
pip install -e .
```

Or install minimal dependencies manually:

```bash
pip install numpy pandas matplotlib pyyaml jsonschema scipy h5py 
```

---

## License

MIT License
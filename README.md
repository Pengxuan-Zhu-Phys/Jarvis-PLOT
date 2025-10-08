# JarvisPLOT

JarvisPLOT is a Python/Matplotlib-based plotting framework for **Jarvis-HEP**.
It provides a simple command-line interface (CLI) to generate scientific plots from configuration files.

---

## Command-Line Usage

Run JarvisPLOT by pointing it at one or more YAML configuration files:

```bash
./jarvisplot -h 
```

### Usage for SUSYRun2 Teranry plots

  ```bash
  ./jarvisplot ./bin/susyrun2.yaml
  ```
Requirements: the data file path need to be CHANGED. 

---

## Notes

- Figures are saved automatically to the specified output directory.
- PNG and PDF are commonly supported; other formats depend on your Matplotlib backend.

---

## Requirements

- Python 3.10+
- NumPy, Pandas, Matplotlib

---

## License

MIT License
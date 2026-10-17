# Changelog

Release history of the `JarvisPLOT` distribution (import package `jarvisplot`,
CLI `jplot`). Design documents, specifications and the roadmap live in
[Jarvis-Books `Docs/Jarvis-PLOT/`](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-PLOT/README.md).

## Unreleased

- Display `Jarvis-PLOT` in text and JSON version reports; the PyPI distribution
  remains `JarvisPLOT`.
- Handle `-v` / `--version` without initializing the plotting runtime and
  support structured version output with `--json`, also through `Jarvis plot`.

## 2.1.11 — 2026-10-18

- Exclude maintainer release helpers, local configuration, credentials, and
  generated files from distribution archives.

- `./jprel X.Y.Z` now completes PyPI upload, local editable install, Git tag,
  and GitHub release. `--dry` checks inputs without changing the checkout.
- Remove private research cards and scripts under `bin/` from version control
  and distribution archives; public examples remain under `Example/`.
- Cache cleanup removes empty registered cache directories whose marker is
  missing, while retaining unmarked directories containing files.
- Font card: `jplot fonts show|list|resolve|check|set|reset` and
  `jplot man fonts`. Normal YAML rendering applies the bundled role card
  (Helvetica sans, STIXGeneral serif, STIX MathText); explicit YAML fonts and
  sizes take precedence. Personal overrides live in
  `~/.jarvis/plot-fonts.json` (`JARVIS_HOME` overrides the directory).
- When Helvetica is absent, the first installed of Arial, Liberation Sans,
  Nimbus Sans and DejaVu Sans is used; `check` and `resolve` report the actual
  font and the substitution reason.
- Cut descriptions take their MathText font from the font card by default.
- Documentation moved to Jarvis-Books; the repository keeps only `README.md`,
  `CHANGELOG.md` and the example READMEs. The source distribution ships
  `CHANGELOG.md` instead of a `docs/` tree, and `tools/check_release.py` checks
  the latest `CHANGELOG.md` entry instead of the old release-note files.

## 2.1.10 — 2026-10-13

- Native forest figures read multi-interval CSV sources and mappings,
  intervals-only rows, box/scatter coexistence and literal CSV value
  annotations. Values are plotted as supplied, never computed.
- Explicit shared x ticks, independent of grid divisions.
- MathText needs `math: true`; formula fonts are STIX by default with DejaVu
  Sans as an alternative, selected separately for legends and cut
  descriptions. Header notes, paragraphs and formulas are preserved, and
  layout uses the actual font, glyph and annotation bounds.
- TTC faces keep their weight and slant; an explicit font size governs
  measurement.
- CLI, manuals, render reports and image metadata read the version from
  `jarvisplot/version.py`. SVG output carries Dublin Core metadata; export
  retries only on a backend's unsupported-metadata error and reports drawing
  and I/O failures.
- Packaged AI README (`jplot man agent-readme --json`), schema, manual and
  example updates.
- `tools/check_release.py` source, tag and wheel gates; the sdist includes the
  native forest examples; CI runs Linux Python 3.10–3.13 and macOS 3.13.

Existing forest YAML needs no migration. Custom export metadata must use keys
the selected backend supports; invalid SVG keys are now reported.

## Earlier releases

See the [PyPI release history](https://pypi.org/project/JarvisPLOT/#history)
and [GitHub Releases](https://github.com/Pengxuan-Zhu-Phys/Jarvis-PLOT/releases).
Earlier release notes are archived in Jarvis-Books
`Docs/archive/Jarvis-PLOT/2026-10-17/docs/release/`.

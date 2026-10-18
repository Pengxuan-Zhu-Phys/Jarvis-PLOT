# Changelog

Release history of the `JarvisPLOT` distribution (import package `jarvisplot`,
CLI `jplot`). User guides and reference documentation are available in
[Jarvis-Books `Docs/Jarvis-PLOT/`](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-PLOT/README.md).

## Unreleased

### Added

- Automatically manage registered plot caches with a default 5 GiB total budget
  and a 30-day idle limit. Configure `~/.jarvis/plot-cache.json` (under `JARVIS_HOME`
  when set), `JP_CACHE_MAX_GIB` or `JP_CACHE_MAX_AGE_DAYS`. `jplot cache list`
  reports the effective policy.
- Run cache maintenance when sessions open and close. Idle caches are removed
  toward 80% of the budget; active caches remain protected and may temporarily
  exceed the limit.

### Fixed

- Keep caches used by another plotting session available during final rendering
  and cleanup. `--print` defers deletion until readers finish; rebuilding a busy
  cache reports an error without deleting its contents.
- Prevent concurrent cache writes from exposing incomplete files or losing
  registered entries. If safe locking is unavailable, cache use stops with an
  error. Interrupted temporary files are reclaimed during idle maintenance.

### Changed

- Display `Jarvis-PLOT` in text and JSON version reports; its PyPI name remains
  `JarvisPLOT`.
- Show `-v` / `--version` without starting the plotting runtime. Structured
  `--json` output is also available through `Jarvis plot`.

## 2.1.11 — 2026-10-18

### Added

- Manage plot fonts with `jplot fonts show|list|resolve|check|set|reset` and
  `jplot man fonts`. The default font card uses Helvetica sans, STIXGeneral serif
  and STIX MathText. Explicit YAML fonts and sizes take precedence.
- Save personal font overrides in `~/.jarvis/plot-fonts.json`; `JARVIS_HOME`
  changes the configuration directory.
- When Helvetica is unavailable, use the first installed font among Arial,
  Liberation Sans, Nimbus Sans and DejaVu Sans. `check` and `resolve` report the
  selected font and the reason for substitution.

### Fixed

- Remove empty registered cache directories whose marker is missing while
  preserving unmarked directories that contain files.

### Changed

- Use the font card's MathText font for cut descriptions by default.
- Find the plotting guides and reference documentation in
  [Jarvis-Books](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-PLOT/README.md).
  Public examples remain available in `Example/`.

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
- Show consistent versions in the CLI, manuals, render reports and image
  metadata. SVG output includes Dublin Core metadata. Export reports drawing
  and I/O failures and retries only when a backend rejects metadata support.
- Packaged AI README (`jplot man agent-readme --json`), schema, manual and
  example updates.
- Include native forest examples in the source distribution.

Existing forest YAML needs no migration. Custom export metadata must use keys
the selected backend supports; invalid SVG keys are now reported.

## Earlier releases

See the [PyPI release history](https://pypi.org/project/JarvisPLOT/#history)
and [GitHub Releases](https://github.com/Pengxuan-Zhu-Phys/Jarvis-PLOT/releases).

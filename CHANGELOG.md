# Changelog

Release history of the `JarvisPLOT` distribution (import package `jarvisplot`,
CLI `jplot`). User guides and reference documentation are available in
[Jarvis-Books `Docs/Jarvis-PLOT/`](https://github.com/Pengxuan-Zhu-Phys/Jarvis-Books/blob/main/Docs/Jarvis-PLOT/README.md).

## Unreleased

### Fixed

- Correct A4 ternary badge image paths on case-sensitive filesystems, including
  Linux, for the full-width, standard and colorbar cards.
- Keep the GAMBIT `rectcmap` colorbar label inside the page with Matplotlib 3.10
  while preserving its separation from tick labels.

## 2.1.12 — 2026-10-19

### Added

- Automatically manage registered plot caches with a default 5 GiB total budget
  and a 30-day idle limit. Configure `~/.jarvis/plot-cache.json` (under
  `JARVIS_HOME` when set), `JP_CACHE_MAX_GIB` or `JP_CACHE_MAX_AGE_DAYS`.
  `jplot cache list` reports the effective policy. Idle caches are reclaimed
  toward 80% of the budget; active caches remain protected.
- Show `-v` / `--version` without loading the plotting runtime. Structured
  `--json` version output is also available through `Jarvis plot`.

### Fixed

- Draw `voronoif` and `triplot` layers. `voronoif` takes `x`, `y` and an
  optional `where` mask for the selected region's hatched boundary; it has no
  `z` or colorbar. `triplot` works on rectangular and ternary axes.
- Evaluate built-in mathematical functions, including hyperbolic and inverse
  trigonometric functions, `Gauss`, `Normal` and `Heaviside`, on whole columns.
  Preserve the `Heaviside(x, H0=...)` keyword. Define `Pi`, `pi`, `E` and `Inf`,
  and publish them through `jplot cap funcs`.
- Avoid false missing-column errors for built-in functions, keyword argument
  labels, YAML `Functions`, Operas namespaces and columns produced by
  `significance` transforms.
- Accept `voronoi` and `adaptive` in `make_density_core`, and recognize the
  existing `profile_2d` reduction and grid options during YAML checks.
- Restore usable full-width `[a4paper_1x1, Ternary]` and
  `[gambit_1x1, Ternary]` cards. See the
  [runnable example](https://github.com/Pengxuan-Zhu-Phys/Jarvis-PLOT/blob/main/Example/ternary_1x1.yaml).
- Print mappings containing quoted YAML strings with `jplot config get`.
- Keep `dynesty_runplot` y labels inside the page and the GAMBIT `rectcmap`
  colorbar label clear of its tick numbers.
- Protect caches used by concurrent plotting sessions. `--print` defers
  cleanup until readers finish, and rebuilding a busy cache reports an error
  without deleting it. Atomic writes prevent incomplete payloads and lost
  registry updates; interrupted temporary files are reclaimed when idle.
  Cache use reports an error if reliable locking is unavailable.
- Report missing layer axes or data before rendering. A layer without a method
  reports a warning and continues to use `scatter`.

### Compatibility and upgrade notes

- Use `jplot doctor <file>` in place of the removed `jplot validate <file>`
  command. Automatic `--fix` / `--fix-unsafe` YAML rewriting has been removed.
  `doctor` performs internal YAML checks and a light dryrun; configuration
  editing and suggestions also retain their internal checks.
- Remove boolean `frame.axlogo` entries from YAML. The field accepts a settings
  mapping only, including for forest figures. Use `jplot <file> --print` for
  final output without the badge.
- Lists in `frame` now replace the style card's lists in full. Review text,
  tick and other list overrides; include every entry you want to keep. This
  also fixes four-element axes rectangles being appended to the card's rectangle.
- `Min` / `Max` and their lowercase aliases are elementwise with multiple
  arguments; a single column argument reduces the whole column to one value.
  Lowercase `e` is a column name, not a built-in constant; use `E` for Euler's number.
- Text and JSON version reports display `Jarvis-PLOT`. The PyPI distribution
  name remains `JarvisPLOT`, and the import package remains `jarvisplot`.

### Documentation

- Include the public YAML examples and their supplied CSV data in the source
  distribution, including the repaired full-width ternary example.

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

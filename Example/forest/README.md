# Native forest examples

- `interval.yaml`: generic supplied estimates/bounds, string IDs with leading zeros.
- `box.yaml`: generic supplied quartiles/whiskers and observations, including a point-only row.
- `multi-interval.yaml`: two supplied boxes, ten original points per row, three interval classes,
  field-based values, lane offsets, a black common mean ± SD legend key, explicit ticks,
  MathText row/cut labels and a caller-supplied header note.
- `intervals-only.yaml`: row IDs/labels plus three intervals per row; no primary glyph or scatter source.
- `real-demo.yaml`: producer-exported generic tables, 97 rows/241 points. This fixture preserves supplied box coordinates, classification and annotations. Producer-specific metadata in CSV is inspectable and never used by the renderer.

Run `jplot validate Example/forest/real-demo.yaml --json`, `jplot doctor Example/forest/real-demo.yaml --json`, then `jplot Example/forest/real-demo.yaml --report`. Outputs go to `Workshop/forest_native/`; each native PNG/SVG has a `.layout.json` receipt with shared limits/ticks, physical positions, annotations, styles and verification. Generated images/reports are not package assets.

The two multi-interval examples write PNG/PDF and receipts only to
`/tmp/jarvisplot-forest-examples`. They select the commonly installed DejaVu Sans font;
the original examples retain the Swiss card's Helvetica default. Run with the official package:

```sh
jplot validate Example/forest/multi-interval.yaml --json
jplot doctor Example/forest/multi-interval.yaml --json
jplot Example/forest/multi-interval.yaml --report
jplot validate Example/forest/intervals-only.yaml --json
jplot doctor Example/forest/intervals-only.yaml --json
jplot Example/forest/intervals-only.yaml --report
```

`summary-rows.csv` supplies the already formatted box values; `summary-intervals.csv`
supplies interval endpoints and already formatted mean ± SD values. The examples are
illustrative supplied data, not a statistical reconstruction from the scatter sample.
Class names are arbitrary. `interval_fields` can map noncanonical CSV headers without
renaming files. `glyph_offset_mm` shifts the box and its points together; overlay `offset_mm`
sets independent interval lanes. A positive `value_column_mm` reserves a right-hand values
strip; zero keeps the full numerical width and requires room beside every interval.
Long annotations or marks near a boundary fail explicitly when their measured ink does not fit.

`multi-interval.yaml` gives the primary box/points and the three summary intervals equal
5 mm halves of the 10 mm mark band. Its plot centre is 7.5 mm, primary offset -2.5 mm,
and summary offsets 0.85/2.5/4.15 mm. This puts the primary distribution higher and the
summary group lower with equal allocated vertical space. Legend formulas default to
STIX; `frame.ax.legend.prop.math_fontfamily: dejavusans` restores the previous look.
`jplot man legend --json` documents formula and ordinary legend font controls.
The cut description explicitly sets `description.math: true` and
`description.math_fontfamily: stix`; legend font settings do not change cut fonts.

Use `jplot man forest --json`, `jplot template show forest --json`, and `jplot config expand Example/forest/real-demo.yaml --figure forest-all-97 --json` for discovery/inspection. Config expansion preserves the native high-level type and returns its solved scene in `data.forests`.

The real demo declares Bin/W/P/E/n/CV columns. Column names and units appear only at the top; body cells contain values, with n formatted .0f and CV .1f. title/description are standard text blocks. The 20 mm footer has three sections: the glyph sample starts at reference column 2, point classes at column 5, and the unchanged YAML description at column 8. Description wraps in the last five reference columns; the header is now 20 mm. The standard Jarvis-HEP icon stays at bottom left, sharing height with the point column; the full figure is 82.5 x 573.5 mm.

Columns use role: row_name/left/right. Bin is the row name; W/P/E form the compact left group; n/CV form the right group. Widths follow actual header/body font metrics, with no manual width declarations. Text sits near the top of each row; the full-width common numerical band below it contains the distributions and observations, extending horizontally beneath both text groups.

Swiss style: installed Helvetica regular/bold; 9 pt title; 12 reference columns; compact left-aligned left cells and right-aligned right cells; black, white, Jarvis blue and red #DA291C. Widths follow actual text metrics without grid rounding; default gaps are 0.8 mm. `real-demo-debug.yaml` enables the separate grid/baseline/band overlay. `meta` controls the optional number/year/place.

The normal chart shows a shared top numerical axis and continuous vertical grid; row rules/per-row axes are removed. `x.ticks` supplies explicit short marks and top/bottom labels. `forest.grid` independently controls numerical lines and deduplicates coincident coordinates; divisions do not add ticks. `debug.forest_grid` controls the separate design overlay.

The default canvas is white, with no row shading, numbered metadata or legend heading. Point legend markers and their labels are left aligned using compact keys.

For a study previously using `jarvis_forest_preview`, install the formal project in the same
Python environment (`python -m pip install -e .` from the Jarvis-PLOT repository root),
remove its preview path from the shell/launcher PYTHONPATH, and call the formal `jplot` on
the existing YAML. The native interface accepts the established `intervals`, annotation,
MathText, header_note and geometry keys; no task-specific names or data conversion are needed.
For a one-off check use `env -u PYTHONPATH jplot validate <existing.yaml> --json`, followed
by doctor and rendering in that clean environment. Verify `jplot --version` in that environment.
To preserve existing study figures during verification, use a temporary YAML with only
`project.workdir` and `output.dir` redirected to a temporary directory; retain the absolute
source paths and all Figures/coordinates/statistical strings. Normal image output is governed
by YAML `output.dir`, not the CLI `-o` flag.

AI agents should read `jplot man agent-readme --json` before authoring YAML or reporting a
missing feature; it includes the installed-version and feature-request feedback requirements.

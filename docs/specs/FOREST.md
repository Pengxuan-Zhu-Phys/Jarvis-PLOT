# Native forest format

`type: forest` consumes `data: <DataSet.name>`, optional `points: <DataSet.name>` and optional `intervals: <DataSet.name>` through the standard dataset loader. It solves one continuous physical canvas. It does not expand into user-authored layers or one axes per row. No producer schema, target names or statistical rules live in this type.

## Tables

Rows are ordered by table order and have unique nonempty string `row`. Optional literal `label`, `detail`, `stats` annotations default to blank (label defaults to row ID). The default `glyph: interval` requires `estimate/lower/upper` with lower <= estimate <= upper when a primary glyph is present. `glyph: box` requires lower <= q1 <= median <= q3 <= upper. Entirely blank primary coordinates are allowed when that row has points or supplied overlay intervals; when an `intervals` source is declared, all primary coordinate columns may also be absent. Partially supplied glyphs, NaN/Inf and rows with nothing drawable fail. Bounds, quartiles and whiskers are supplied; their meanings belong to the producer.

Points require string `row` and `id`, finite `x`, and optional string `class` (default `normal`). `(row,id)` must be unique and row must exist. Every supplied point becomes one mark; no statistical computation, jitter in x, thinning, removal or automatic shrinking is performed. Native CSV loading preserves IDs and literal text, including leading zeros and the literal annotation `NA`; empty numeric cells remain deliberate blanks and numeric NaN/Inf remain invalid. Typed table producers must export identifiers as strings and missing coordinates as nulls. Extra source columns are permitted; the runtime selection is projected to the declared fields, preserving the existing narrow-table architecture.

Intervals require nonempty string `row`, `id` and `class`, with finite `estimate/lower/upper` ordered as lower <= estimate <= upper. Every interval's row must exist, `(row,id)` must be unique within the interval table, and each class must be declared in forest.overlays.classes. Every interval is rendered; interval classes and endpoints are producer supplied. Multiple intervals can share a row without any primary glyph or points. Their input order and the configured offsets determine their displayed layers; the library does not calculate means, SDs, CIs or folds.

`fields`, `point_fields` and `interval_fields` map canonical roles to literal column names. `interval_fields` supports `row`, `id`, `estimate`, `lower`, `upper` and `class`. `annotations: {label: <column>, detail: <column>, stats: <column>}` selects legacy row text; false hides a slot. Explicit mappings must exist. Extra annotation fields are projected on demand; no unrestricted expression evaluation or full-table propagation is added.

## Multiple interval layers and value annotations

`forest.overlays.classes` supplies generic class styles: `label`, `color`, `alpha`, `linewidth_pt`, `linestyle`, `marker`, `diameter_mm`, `cap_height_mm` and `offset_mm`. Offsets are millimetres relative to the row's plot centre, with positive values further down the page. The base interval style is inherited from `forest.interval`; there are no hard-coded task or sampling-method names. `forest.geometry.glyph_offset_mm` shifts the primary glyph and its points together, independently of overlay offsets. Increase row pitch/band and choose offsets that keep all layers and their text inside the row.

`forest.box.annotation` references a row CSV field, for example an already formatted `median [Q1,Q3]`. `forest.overlays.annotation` references an interval CSV field, for example an already formatted `mean ± SD`. These are literal supplied strings, not computations from the glyph coordinates. Both support `fontsize`, `gap_mm` and opt-in `math`; box annotations additionally support `color` and `position: box|row_text` (default `box`). `row_text` puts the box value on the upper row text line. Other box annotations sit on the primary glyph lane. Overlay annotations inherit each interval class's color and sit on that interval's own lane. The optional overlay annotation `label` names the value area, not an extra per-row legend entry.

`forest.geometry.value_column_mm` is a nonnegative physical width. A positive value reserves a right-hand values strip; `0` retains the full numerical column width and uses the available room to the right of each interval, with `gap_mm` clearance. Values are right aligned at the shared value boundary. The shared numerical mapping still applies to every glyph, point and interval. Annotation placement uses measured text/mark bounds: long values or near-boundary intervals that leave insufficient room produce an actionable overflow/collision error. The library does not silently move the value to another lane or shrink text. Explicit geometry/font adjustments are allowed.

```yaml
Figures:
- name: supplied-summary
  type: forest
  data: rows
  points: points
  intervals: intervals
  glyph: box
  x: {lim: [0.60, 0.66], ticks: [0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66]}
  forest:
    geometry:
      point_band_mm: 10
      row_band_mm: 13
      row_pitch_mm: 13.3
      plot_y_mm: 8
      cell_y_mm: 1.35
      glyph_offset_mm: -2.5
      value_column_mm: 0
    box:
      annotation: {field: stats, position: row_text, fontsize: 4.3}
    overlays:
      legend: {label: '$\mathrm{mean}\pm\mathrm{SD}$', color: '#000000'}
      annotation: {field: stats, fontsize: 4.3, gap_mm: 0.5}
      classes:
        first: {label: First cohort, color: '#0072B2', offset_mm: 0}
        second: {label: Second cohort, color: '#D55E00', offset_mm: 1.6}
        repeat: {label: Repeated fits, color: '#009E73', offset_mm: 3.2}
```

For intervals-only rows, omit `points` and provide row IDs/labels plus the `intervals` source. Absent primary coordinate fields are deliberate blanks when an overlay exists. [The minimal examples](../../Example/forest/README.md) cover both combinations.

## Shared axis and physical appearance

All panels use one global domain/tick set/x mapping. Tick labels appear only at top and bottom. Inference includes every primary glyph, point, overlay interval endpoint/estimate and optional `x.reference`. Stable rounded enclosing limits include physical padding. `x.minimum_span` is positive, default 1 input unit, and handles degenerate inputs. Explicit `x.lim` is ascending, finite, shared and must retain all values/reference. `x.ticks` optionally supplies at least two finite strictly ascending tick values inside the domain; otherwise the established default shared tick selection applies. Grid divisions never create axis ticks. Per-row scaling and nonlinear transforms are unsupported.

Omitted style selects `[forest, swiss_column]`. The card owns canvas width 82.5 mm, row pitch 5.5 mm, band 5.2 mm, and header/footer 20/20 mm. The numerical region spans the canvas between its 2 mm outer margins (78.5 mm by default), independent of measured text widths. Height = header + row_count * pitch + footer. The 97-row fixture is 573.5 mm tall. Each row has two vertical bands: compact text near the top (cell centre 1 mm), and glyphs/points below (plot centre 3.6 mm, point band 3 mm). Row names and left information form the left text group; right information forms the right text group. The lower numerical region extends horizontally beneath both text groups, identically for every row. The groups overlap in x, while text ink and marks remain separated in y. No pagination or newspaper columns.

Sparse `forest` overrides merge box/interval/scatter/overlays/rows/axis/reference appearance. Two explicit appearance groups, `forest.geometry` and `forest.typography`, expose physical mm layout and pt fonts for personalisation and larger row bands. Canvas width remains 82.5 mm. The complete closed vocabulary is in `schema/types/forest.json`; native `cap styles` reports the card and `man forest` documents the ownership. Column widths follow actual text metrics without grid rounding. Adjacent columns have a 0.8 mm gap; at least 2 mm separates the left/right text groups on the upper line. Text column widths do not reduce the lower numerical span; only a positive value_column_mm reserves numerical width for annotations. Longer annotations can select/hide columns or explicitly change typography/gaps; fonts and marks are never shrunk silently.

Points are packed in deterministic x/id order using physical tangent candidates, marker diameter, stroke and gap. Numerical x stays exact; offsets stay within the row's point band. `o` and `x` are supported with conservative collision bounds. Overflow names the row and required span; increase point_band_mm plus row_band_mm/row_pitch_mm/plot_y_mm.

Present classes produce internal single-component `legend: {object: <class>, label: <card label>}` declarations from their actual scatter artists. They are resolved by the existing canonical legend implementation. `frame.ax.legend` controls only overall order/card/columns/location and item formats, as elsewhere. The Swiss card defines normal/outlier/unrated/singleton; additional classes require explicit styles (color/marker/fill), and all classifications are producer supplied. No parallel legend API or legacy artist-label discovery exists.

Forest owns its frame; layers, method, arbitrary frame rect/figsize, and other type slots fail. `debug` uses the existing overlay against the solved frame.

## Validation and inspection

`validate` checks the closed schema, appearance merging, source names and required/explicit columns without importing Matplotlib or loading all rows. `doctor` uses the production loader and shared solve for table values, keys, primary/overlay glyphs, classes, common limits/ticks, plain-text/MathText column metrics and point packing, without creating a figure or canvas. Heavy transforms remain skipped in doctor, preserving partial coverage. Final annotation space, rendered ink, collision and legend bounds are explicitly pending until native render.

`config expand --figure <name> --json` preserves native forest YAML and exposes the solved scene under `data.forests`; it does not fabricate a primitive layer stack. Native rendering measures actual text ink, including MathText, against reserved physical slots (allowing half a raster pixel for font hinting), detects annotation collisions, measures legend/header/footer bounds and verifies every glyph/point/overlay lane. Successful render supports the standard PNG/PDF/SVG outputs and a JSON layout receipt; `--report` also includes that receipt in the existing render-health report. Inspect actual short tick artists and interval/point coordinates, not just label counts, when validating an axis or data-preservation fix.

Errors: JP-FOR-001 is a source/appearance contract error; JP-FOR-002 is a data/packing error discovered by doctor. Native rendering reports row/field context for physical overflow.

## Implementation ownership

`forest_contract.py` contains the contract, source demand and deterministic physical solve. `forest_text.py` lazily loads plain-text/MathText metrics and formula-preserving wrapping for content-sized columns and text blocks; schema validation and discovery remain independent of Matplotlib. `Figure/forest_runtime.py` is a native Figure subclass using standard output metadata/save, context and debug. `Figure/forest_legend.py` handles symbolic glyph keys and physically measured legend layout through the canonical legend machinery. Core chooses the native renderer and plans all declared narrow row/point/interval inputs, including referenced annotation fields. Schema, card, manual and template ship in the wheel; no new dependency is introduced.

## Table columns, structured text and footer

`columns` declares literal fields with required `field` and `label`, optional `role` (`row_name`, `left`, `right`; default `left`), numeric `format` (.0f/.1f/etc), `missing` (default —), `fontsize` and `fontweight`. At most one `row_name` is allowed; it is placed first on the left, left aligned and bold by default. Other `left` columns follow it compactly and are left aligned; `right` columns form a compact right-aligned group anchored at the rightmost reference-grid boundary. Declaration order is preserved within the right group, so a declared n/CV pair puts CV at the far right and n immediately to its left. Declaration order is preserved within each side, even if the list interleaves the sides. A right group's column widths are individual: short N cells never reserve the width of CV cells.

Widths use the maximum actual font advance of the header and every formatted body value, plus measured ink overhang, padding and column clearance. No column is widened to fit a grid module. Left text anchors follow measured column starts; right text anchors follow measured ends. Header and body font size/weight both contribute. Cells are single-line literal text; `columns[].math: true` explicitly enables MathText for that column's header and values. The same measured grid applies to all rows. `width_mm` and `align` are not YAML options; `role` owns side placement and text alignment: row_name/left are left aligned; right is right aligned. Solved `columns[].width_mm`, `left_mm`, `content_width_mm`, `measured_width_mm` and `align` remain inspectable in the layout receipt. `forest.geometry.column_gap_mm` separates measured columns; `group_gap_mm` sets the minimum gap between left and right text groups. If the groups do not fit on their upper line, solve fails with an actionable error. `margin_left_mm` and `margin_right_mm` determine the lower numerical span. Both outer margins are minima: solve enlarges them when needed to contain the outward half of centred endpoint labels, measured with the actual font and unit suffix. The label remains anchored at its exact tick; the resulting grid and x mapping are common to all rows. Marks may extend beneath the text columns in x, but must stay inside the canvas and the reserved lower band. Final rendered text ink must remain above that band; no text is silently shrunk.

Column names appear once in the header; rows display values only. Units belong in labels, e.g. CV [%]. Formatting changes display precision without changing data or classification. Explicit field references must exist. Omitted columns derives active Label/Detail/Stats slots with roles row_name/left/right.

```yaml
columns:
  - {field: label, label: Bin, role: row_name}
  - {field: W, label: W, role: left}
  - {field: P, label: P, role: left}
  - {field: E, label: E, role: left}
  - {field: n, label: N, role: right, format: .0f}
  - {field: cv, label: 'CV [%]', role: right, format: .1f}
```

`title`, `header_note` and `description` share the closed standard shape `{text, fontsize?, color?, fontweight?, math?}`; multiline text uses YAML `text: |`. These replace the initial prototype's scalar title/caption. All content is YAML-owned: header_note can display supplied signal/background MC-entry counts without counting input samples. Title and header_note occupy distinct measured header slots above the column headings; `forest.geometry.header_note_y_mm` chooses the note baseline. Structured `meta` at a conflicting baseline is rejected. Description stays in the third footer section at reference column 8 and defaults to 5 pt. Its first-line centre is footer-relative description_y_mm; description_columns defaults to 5. Paragraphs and blank lines are preserved. With `math: true`, wrapping treats every `$...$` formula as an indivisible unit; a formula wider than its slot fails explicitly. There is no automatic count or technical explanatory text. Physical baselines/fonts belong to the card.

MathText is opt-in and uses Matplotlib's supported math syntax, not an external TeX process. Enable `columns[].math`, `title.math`, `header_note.math`, `description.math`, annotation `math`, or `forest.legend_math` at the needed boundary. Plain-text defaults retain the previous literal behavior. Actual MathText ink participates in width/height/collision checks, including scripts and fractions. Measurement caches include the active MathText font settings, so another figure's style cannot reuse stale widths. Title, header note and description placement accounts for left ink overhang; wrapping uses the complete ink width. Footer line-to-line and description-to-badge collisions are checked even when the legend is disabled. Nonfinite appearance values and unrepresentable axis spans fail before rendering.

The footer's canonical legend combines the present primary glyph, present point classes, and overlay declarations. With overlays, `forest.overlays.legend` supplies one common estimate/bounds format key, e.g. a black mean ± SD sample; each present overlay class contributes its configured color label. There is no redundant format key per interval and no per-row numeric value in the legend. Intervals-only figures omit absent primary glyph/point keys. The standard legend resolver owns grouping/order and uses a dedicated native container handler for symbolic samples with no fabricated data coordinates. `glyph_label` optionally declares producer-defined primary glyph semantics. The glyph object is named glyph (prefixed with underscores if a point class already uses that name). The default two-column layout puts format/glyph keys on the left and class labels on the right. Canonical order is preserved within each group. Keys start at reference columns 2 and 5 respectively, configured by `forest.geometry.legend_start_columns: [2, 5]`. The logo stays in the lower-left corner, sharing the class column's height; absent groups keep their semantic column empty. Other ncols values use the standard sequential arrangement. The third footer section is the YAML description, starting at forest.geometry.description_start_column (default 8). frame.ax.legend.ncols stays 2 because the description is explanatory text, not an artist item. Native render checks legend items against the icon, wordmark and description for overlaps. Long labels wrap within their physical column without shrinking; with forest.legend_math: true whole formulas remain intact.

When a shared overlay-format key is configured, present point keys join the left format/primary-glyph group and the right group contains overlay class colors. Without that shared-format key, the established primary-glyph/point-class grouping remains. Class legend objects retain distinct namespaces even when a point class and an interval class share the same source name.

Legend formulas default to STIX, inherited from the JSON legend card. Keep
`forest.legend_math: true`, and select `frame.ax.legend.prop.math_fontfamily: stix`
or `dejavusans` (previous appearance). Ordinary legend text continues to use the forest
font unless `prop.family` or `prop.fname` explicitly overrides it. `prop.size` overrides
`fontsize`; selecting only a math font preserves the configured size. Wrapping and
actual ink checks use the resulting legend font properties. Layout receipts expose
the effective family, size and formula family in `legend_fonts`; both `man forest`
and `man legend` document the settings.

Cut/footer formulas have their own explicit font selector, independent of legend fonts:

```yaml
description:
  math: true
  math_fontfamily: stix  # or dejavusans
  text: |
    Parton cuts:
    $m_{t\bar t}^{\mathrm{gen}} < 355\,\mathrm{GeV}$
```

Omitting `description.math_fontfamily` preserves the active MathText default for backward
compatibility. `math: true` enables LaTeX-style MathText syntax, not an external LaTeX
compiler. Use single-pair `$...$` delimiters; MathText does not support LaTeX's
display-math `$$...$$` delimiters. YAML literal blocks (`text: |`) or single-quoted
strings preserve command backslashes. Wrapping, overhang and actual rendered bounds use the selected description
font consistently; ordinary text keeps the forest family. `man forest --json` exposes
the selector and supported choices, and render receipts record `description_fonts`.

For a primary distribution plus three summary intervals, the shipped multi-interval
example splits its 10 mm plot band into two equal 5 mm halves: `plot_y_mm: 7.5`
places the band at row-relative 2.5–12.5 mm; `glyph_offset_mm: -2.5` centres the box
and points at 5 mm in the upper half. Interval offsets `0.85`, `2.5`, `4.15` centre
the three summaries at 8.35, 10, 11.65 mm in the lower half. This is an explicit
generic YAML layout; ordinary forest defaults and caller-supplied data are preserved.

The Swiss axlogo uses the shared `cards/icons/JarvisHEP.png` icon and Jarvis-HEP wordmark through the standard image badge renderer, aligned to the first grid module; `frame.axlogo: false` and standard --print policy disable it. A mapping overrides text settings or selects an image file. This helper adds a badge axes while the numerical plot remains on one shared canvas; no per-row axes are created. The compact footer is 20 mm; pitch stays 5.5 mm, and the full 97-row canvas is 82.5 x 573.5 mm. Measured table header/cell/title/description/legend/wordmark bounds are checked.


## Swiss International Typographic Style

The card retains a 12-column reference grid for debug guides, metadata, badge and legend. Table columns follow actual text width rather than grid spans: W/P/E sit close together, and n/CV are right aligned from the far-right boundary. Title uses the available header width, with a 9 pt title and roughly 5–6 pt body. Description uses the last five reference columns in the footer. The header is 20 mm including the column names and shared top numerical axis. Editorial text wraps at literal spaces without changing cell values or shrinking text. Excess title/description height fails the measured slot check. Data and numerical labels retain their exact x coordinates.

The palette is black #000000, white #FFFFFF, Jarvis blue #2F7FD8 for box borders and Jarvis light blue #73B8F4 for box fills and accent red #DA291C for outliers. The canvas is plain white and row fills are absent by default; box borders, whisker end caps and horizontal whisker lines use Jarvis blue, with Jarvis light blue interiors and black median lines (forest.box.median_color); the actual-artist legend inherits both colours; row separators and per-row horizontal axes are absent by default; ordinary observations are black, and potential outliers are red. Unrated/singleton meaning remains encoded by hollow circles/crosses. The standard Jarvis-HEP icon retains its brand colours, occupies one horizontal grid module, and uses logo_size_mm as its physical height. Data styling remains flat, without shadows, gradients or invented classifications.

Typography defaults to installed Helvetica, with regular/bold table text. A missing requested single family fails explicitly; `forest.typography.family` may select an installed Inter or DejaVu Sans family for portable machines. Legend `prop.weight` and `prop.style` also select numeric weight and italic/oblique faces. On macOS, Matplotlib normally renders the first TTC face even when another weight/slant is requested: the font helper selects the matching installed face and extracts it into a process-local temporary directory. Native `fname` overrides and ordered fallback-family lists retain native resolution. Fonts are never bundled or redistributed; fontTools is already a Matplotlib dependency. Portable CLI examples use DejaVu Sans; font-specific Swiss regression fixtures require Helvetica and run on the macOS CI job, rather than silently substituting another font on Linux.

Optional structured metadata is wholly YAML-owned:

```yaml
meta: {number: '01', year: '2026', place: ADELAIDE}
debug: true
```

The existing debug switch now dispatches to a native forest design-reference overlay. It shows the 13 boundaries of the 12 columns, 01–12 module labels, header/row baselines and upper text/lower mark band outlines. References live on a separate axes and do not modify scene data or packing. `debug: false` and `debug: {show: false}` leave them off. The idempotent overlay honours the standard defaults/card/YAML merge:

```yaml
debug:
  forest_grid:
    columns: {show: true}
    baselines: {show: true}
    bands: {show: true}
    labels: {show: true}
```

Each section can be disabled or its Matplotlib call block overridden (`plot`, `Rectangle`, `text` respectively). The forest path uses this grid overlay instead of the generic dimensions panel, so reference labels do not cover the title. The two-column legend starts at reference columns 2 and 5 by default, ending before the description at column 8; other column counts use equal spans across the grid; ncols must divide 12 and each key/label must fit its span. Glyph labels start one module after their keys; point keys are compact (legend_point_key_mm, default 1.2 mm), with all marker centres and label starts left aligned; font size/labels/column count remain configurable and overflow is explicit.


## Shared numerical grid

The normal figure shows one top horizontal numerical axis at `forest.geometry.top_axis_y_mm`, plus continuous vertical numerical grid lines down the data region. Top-axis short marks and both sets of labels are drawn only at `shared_ticks`. `forest.grid` independently controls `show`, `divisions` (default 4 equal intervals), colour, linewidth_pt, alpha and linestyle; changing or disabling it never changes the axis ticks. Visible grid coordinates are the union of equal-interval divisions and shared ticks, merged in normalized domain space so floating-point round-off cannot produce coincident duplicate lines. A reference is a separate single continuous line. Column headings and body text do not act as numerical ticks. Matching flat text backgrounds keep grid lines from cutting through annotations.

`forest.rows.separators` defaults to false; explicit true enables row rules. Normal numerical grid lines are distinct from the 12-column design-reference overlay controlled by `debug.forest_grid`. Both share the solved geometry and neither changes data coordinates or packing.

No numbered metadata or legend heading is inserted automatically. The shipped examples omit meta and the style card has an empty legend_heading. Explicit YAML meta and a nonempty forest.rows.legend_heading remain optional. Point legend keys use actual scatter styles, with their markers placed toward the left edge and adjacent labels aligned consistently.

Legend column widths are stored in physical units and converted by each active renderer, so PNG/SVG/PDF save order and DPI do not change reference-column positions. Regression checks inspect exported SVG coordinates as well as the raster scene at 96/220/600 DPI. Forest uses unhinted text when measuring and exporting so small Helvetica glyphs retain consistent physical metrics across raster/vector backends.

X tick labels are centred on their exact numerical coordinates and display a literal percent suffix by default (-40%, 0%, 40%). Style.forest.axis.tick_align and tick_suffix in the JSON card control display; YAML forest.axis overrides them. The suffix does not rescale data or tick values. Both top and bottom labels use the same settings.

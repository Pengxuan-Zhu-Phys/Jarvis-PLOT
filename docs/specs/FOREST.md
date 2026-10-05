# Native forest format

`type: forest` consumes `data: <DataSet.name>` and optional `points: <DataSet.name>` through the standard dataset loader. It solves one continuous physical canvas. It does not expand into user-authored layers or one axes per row. No producer schema, target names or statistical rules live in this type.

## Tables

Rows are ordered by table order and have unique nonempty string `row`. Optional literal `label`, `detail`, `stats` annotations default to blank (label defaults to row ID). The default `glyph: interval` requires `estimate/lower/upper` with lower <= estimate <= upper. `glyph: box` requires lower <= q1 <= median <= q3 <= upper. Entirely blank glyphs require observations; partial glyphs, NaN/Inf and panels with nothing drawable fail. Bounds, quartiles and whiskers are supplied; their meanings belong to the producer.

Points require string `row` and `id`, finite `x`, and optional string `class` (default `normal`). `(row,id)` must be unique and row must exist. Every supplied point becomes one mark; no statistical computation, jitter in x, thinning, removal or automatic shrinking is performed. Native CSV loading preserves IDs and literal text, including leading zeros and the literal annotation `NA`; empty numeric cells remain deliberate blanks and numeric NaN/Inf remain invalid. Typed table producers must export identifiers as strings and missing coordinates as nulls. Extra source columns are permitted; the runtime selection is projected to the declared fields, preserving the existing narrow-table architecture.

`fields` and `point_fields` map canonical roles to literal column names. `annotations: {label: <column>, detail: <column>, stats: <column>}` selects row text; false hides a slot. Explicit column mappings must exist. No expression or formatting language is added.

## Shared axis and physical appearance

All panels use one global domain/tick set/x mapping. Tick labels appear only at top and bottom. Inference includes all drawn coordinates/observations (including outliers) and optional `x.reference`. Stable rounded enclosing limits include physical padding. `x.minimum_span` is positive, default 1 input unit, and handles degenerate inputs. Explicit `x.lim` is ascending, finite, shared and must retain all values/reference. Per-row scaling and nonlinear transforms are unsupported.

Omitted style selects `[forest, swiss_column]`. The card owns canvas width 82.5 mm, row pitch 5.5 mm, band 5.2 mm, and header/footer 20/20 mm. The numerical region spans the canvas between its 2 mm outer margins (78.5 mm by default), independent of measured text widths. Height = header + row_count * pitch + footer. The 97-row fixture is 573.5 mm tall. Each row has two vertical bands: compact text near the top (cell centre 1 mm), and glyphs/points below (plot centre 3.6 mm, point band 3 mm). Row names and left information form the left text group; right information forms the right text group. The lower numerical region extends horizontally beneath both text groups, identically for every row. The groups overlap in x, while text ink and marks remain separated in y. No pagination or newspaper columns.

Sparse `forest` overrides merge box/interval/scatter/rows/axis/reference appearance. Two explicit appearance groups, `forest.geometry` and `forest.typography`, expose physical mm layout and pt fonts for personalisation and larger row bands. Canvas width remains 82.5 mm. The complete closed vocabulary is in `schema/types/forest.json`; native `cap styles` reports the card and `man forest` documents the ownership. Column widths follow actual text metrics without grid rounding. Adjacent columns have a 0.8 mm gap; at least 2 mm separates the left/right text groups on the upper line. Text column widths do not reduce the lower numerical span. Longer annotations can select/hide columns or explicitly change typography/gaps; fonts and marks are never shrunk silently.

Points are packed in deterministic x/id order using physical tangent candidates, marker diameter, stroke and gap. Numerical x stays exact; offsets stay within the row's point band. `o` and `x` are supported with conservative collision bounds. Overflow names the row and required span; increase point_band_mm plus row_band_mm/row_pitch_mm/plot_y_mm.

Present classes produce internal single-component `legend: {object: <class>, label: <card label>}` declarations from their actual scatter artists. They are resolved by the existing canonical legend implementation. `frame.ax.legend` controls only overall order/card/columns/location and item formats, as elsewhere. The Swiss card defines normal/outlier/unrated/singleton; additional classes require explicit styles (color/marker/fill), and all classifications are producer supplied. No parallel legend API or legacy artist-label discovery exists.

Forest owns its frame; layers, method, arbitrary frame rect/figsize, and other type slots fail. `debug` uses the existing overlay against the solved frame.

## Validation and inspection

`validate` checks the closed schema, appearance merging, source names and required/explicit columns without importing Matplotlib or loading all rows. `doctor` uses the production loader and shared solve for table values, keys, glyphs, classes, common limits/ticks, literal font metrics, measured columns and point packing, without creating a figure or canvas. Heavy transforms remain skipped in doctor, preserving partial coverage. Final rendered ink/legend bounds are explicitly pending until native render.

`config expand --figure <name> --json` preserves native forest YAML and exposes the solved scene under `data.forests`; it does not fabricate a primitive layer stack. Native rendering measures actual literal glyph ink against reserved physical slots (allowing half a raster pixel for font hinting), detects annotation collisions, measures legend bounds and verifies glyph/point bands. Successful render emits PNG/SVG and a JSON layout receipt; `--report` also includes that receipt in the existing render-health report.

Errors: JP-FOR-001 is a source/appearance contract error; JP-FOR-002 is a data/packing error discovered by doctor. Native rendering reports row/field context for physical overflow.

## Implementation ownership

`forest_contract.py` contains the contract, source demand and deterministic physical solve. `forest_text.py` lazily loads Matplotlib font metrics for content-sized columns; schema validation and discovery remain independent of Matplotlib. `Figure/forest_runtime.py` is a native Figure subclass using standard output metadata/save, context and debug. Core chooses that native renderer and plans both narrow inputs. Schema, card, manual and template ship in the wheel; no new dependency is introduced.

## Table columns, structured text and footer

`columns` declares literal fields with required `field` and `label`, optional `role` (`row_name`, `left`, `right`; default `left`), numeric `format` (.0f/.1f/etc), `missing` (default —), `fontsize` and `fontweight`. At most one `row_name` is allowed; it is placed first on the left, left aligned and bold by default. Other `left` columns follow it compactly and are left aligned; `right` columns form a compact right-aligned group anchored at the rightmost reference-grid boundary. Declaration order is preserved within the right group, so a declared n/CV pair puts CV at the far right and n immediately to its left. Declaration order is preserved within each side, even if the list interleaves the sides. A right group's column widths are individual: short N cells never reserve the width of CV cells.

Widths use the maximum actual font advance of the header and every formatted body value, plus measured ink overhang, padding and column clearance. No column is widened to fit a grid module. Left text anchors follow measured column starts; right text anchors follow measured ends. Header and body font size/weight both contribute. Cells are single-line literal text (no tabs or math parsing). The same measured grid applies to all rows. `width_mm` and `align` are not YAML options; `role` owns side placement and text alignment: row_name/left are left aligned; right is right aligned. Solved `columns[].width_mm`, `left_mm`, `content_width_mm`, `measured_width_mm` and `align` remain inspectable in the layout receipt. `forest.geometry.column_gap_mm` separates measured columns; `group_gap_mm` sets the minimum gap between left and right text groups. If the groups do not fit on their upper line, solve fails with an actionable error. `margin_left_mm` and `margin_right_mm` determine the lower numerical span. Both outer margins are minima: solve enlarges them when needed to contain the outward half of centred endpoint labels, measured with the actual font and unit suffix. The label remains anchored at its exact tick; the resulting grid and x mapping are common to all rows. Marks may extend beneath the text columns in x, but must stay inside the canvas and the reserved lower band. Final rendered text ink must remain above that band; no text is silently shrunk.

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

`title` and `description` share the closed standard shape `{text, fontsize?, color?, fontweight?}`; multiline text uses YAML `text: |`. These replace the initial prototype's scalar title/caption. Title and description content is wholly YAML-owned. Title stays in the header; description moves to the third footer section at reference column 8, wraps at literal spaces, and defaults to 5 pt. Its first-line centre is footer-relative description_y_mm; description_columns defaults to 5. There is no automatic count or technical explanatory text. Physical baselines/fonts belong to the card.

The footer's canonical legend combines one native glyph container from actual drawn box/whisker/median or estimate/bounds artists with the present point class artists. The standard legend resolver owns grouping/order and uses a dedicated native container handler for a readable symbolic sample, with no fabricated data coordinates. `glyph_label` optionally declares producer-defined semantics. The glyph object is named glyph (prefixed with underscores if a point class already uses that name). The default two-column layout puts the glyph alone on the left and all present point classes on the right. Canonical order is preserved within each group. The glyph and point keys start at reference columns 2 and 5 respectively, configured by `forest.geometry.legend_start_columns: [2, 5]`. The logo stays in the lower-left corner, sharing the point column's height; absent groups keep their semantic column empty. Other ncols values use the standard sequential arrangement. The third footer section is the YAML description, starting at forest.geometry.description_start_column (default 8). frame.ax.legend.ncols stays 2 because the description is explanatory text, not a fabricated artist item. Native render checks legend items against the icon, wordmark and description for overlaps. Long legend labels wrap within their physical column without shrinking the font.

The Swiss axlogo uses the shared `cards/icons/JarvisHEP.png` icon and Jarvis-HEP wordmark through the standard image badge renderer, aligned to the first grid module; `frame.axlogo: false` and standard --print policy disable it. A mapping overrides text settings or selects an image file. This helper adds a badge axes while the numerical plot remains on one shared canvas; no per-row axes are created. The compact footer is 20 mm; pitch stays 5.5 mm, and the full 97-row canvas is 82.5 x 573.5 mm. Measured table header/cell/title/description/legend/wordmark bounds are checked.


## Swiss International Typographic Style

The card retains a 12-column reference grid for debug guides, metadata, badge and legend. Table columns follow actual text width rather than grid spans: W/P/E sit close together, and n/CV are right aligned from the far-right boundary. Title uses the available header width, with a 9 pt title and roughly 5–6 pt body. Description uses the last five reference columns in the footer. The header is 20 mm including the column names and shared top numerical axis. Editorial text wraps at literal spaces without changing cell values or shrinking text. Excess title/description height fails the measured slot check. Data and numerical labels retain their exact x coordinates.

The palette is black #000000, white #FFFFFF, Jarvis blue #2F7FD8 for box fills and accent red #DA291C for outliers. The canvas is plain white and row fills are absent by default; box interiors use the project-defined Jarvis blue, inherited by the actual-artist legend; row separators and per-row horizontal axes are absent by default; ordinary observations are black, and potential outliers are red. Unrated/singleton meaning remains encoded by hollow circles/crosses. The standard Jarvis-HEP icon retains its brand colours, occupies one horizontal grid module, and uses logo_size_mm as its physical height. Data styling remains flat, without shadows, gradients or invented classifications.

Typography defaults to installed Helvetica, with only regular and bold weights. A missing requested family fails explicitly; `forest.typography.family` may select an installed Inter or other chosen family. On macOS, Matplotlib normally renders the first TTC face even when bold is requested: the font helper selects the installed regular/bold face and extracts it into a process-local temporary directory. Fonts are never bundled or redistributed; fontTools is already a Matplotlib dependency.

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

The normal figure shows one top horizontal numerical axis at `forest.geometry.top_axis_y_mm`, plus continuous vertical numerical grid lines down the data region. `forest.grid` controls `show`, `divisions` (default 4 equal intervals), colour, linewidth_pt, alpha and linestyle. Shared labelled ticks are included even when not on an interval boundary; reference is a single continuous line. Column headings and body text do not act as numerical ticks. Matching flat text backgrounds keep grid lines from cutting through annotations.

`forest.rows.separators` defaults to false; explicit true enables row rules. Normal numerical grid lines are distinct from the 12-column design-reference overlay controlled by `debug.forest_grid`. Both share the solved geometry and neither changes data coordinates or packing.

No numbered metadata or legend heading is inserted automatically. The shipped examples omit meta and the style card has an empty legend_heading. Explicit YAML meta and a nonempty forest.rows.legend_heading remain optional. Point legend keys use actual scatter styles, with their markers placed toward the left edge and adjacent labels aligned consistently.

Legend column widths are stored in physical units and converted by each active renderer, so PNG/SVG/PDF save order and DPI do not change reference-column positions. Regression checks inspect exported SVG coordinates as well as the raster scene at 96/220/600 DPI. Forest uses unhinted text when measuring and exporting so small Helvetica glyphs retain consistent physical metrics across raster/vector backends.

X tick labels are centred on their exact numerical coordinates and display a literal percent suffix by default (-40%, 0%, 40%). Style.forest.axis.tick_align and tick_suffix in the JSON card control display; YAML forest.axis overrides them. The suffix does not rescale data or tick values. Both top and bottom labels use the same settings.

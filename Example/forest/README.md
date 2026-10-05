# Native forest examples

- `interval.yaml`: generic supplied estimates/bounds, string IDs with leading zeros.
- `box.yaml`: generic supplied quartiles/whiskers and observations, including a point-only row.
- `real-demo.yaml`: producer-exported generic tables, 97 rows/241 points. This fixture preserves supplied box coordinates, classification and annotations. Producer-specific metadata in CSV is inspectable and never used by the renderer.

Run `jplot validate Example/forest/real-demo.yaml --json`, `jplot doctor Example/forest/real-demo.yaml --json`, then `jplot Example/forest/real-demo.yaml --report`. Outputs go to `Workshop/forest_native/`; each native PNG/SVG has a `.layout.json` receipt with shared limits/ticks, physical positions, annotations, styles and verification. Generated images/reports are not package assets.

Use `jplot man forest --json`, `jplot template show forest --json`, and `jplot config expand Example/forest/real-demo.yaml --figure forest-all-97 --json` for discovery/inspection. Config expansion preserves the native high-level type and returns its solved scene in `data.forests`.

The real demo declares Bin/W/P/E/n/CV columns. Column names and units appear only at the top; body cells contain values, with n formatted .0f and CV .1f. title/description are standard text blocks. The 20 mm footer has three sections: the glyph sample starts at reference column 2, point classes at column 5, and the unchanged YAML description at column 8. Description wraps in the last five reference columns; the header is now 20 mm. The standard Jarvis-HEP icon stays at bottom left, sharing height with the point column; the full figure is 82.5 x 573.5 mm.

Columns use role: row_name/left/right. Bin is the row name; W/P/E form the compact left group; n/CV form the right group. Widths follow actual header/body font metrics, with no manual width declarations. Text sits near the top of each row; the full-width common numerical band below it contains the distributions and observations, extending horizontally beneath both text groups.

Swiss style: installed Helvetica regular/bold; 9 pt title; 12 reference columns; compact left-aligned left cells and right-aligned right cells; black, ivory #F5F2E9 and red #DA291C. Widths follow actual text metrics without grid rounding; default gaps are 0.8 mm. `real-demo-debug.yaml` enables the separate grid/baseline/band overlay. `meta` controls the optional number/year/place.

The normal chart shows a shared top numerical axis and continuous vertical grid; row rules/per-row axes are removed. `forest.grid` controls numerical lines, while `debug.forest_grid` controls the separate design overlay.

The default canvas is white, with no row shading, numbered metadata or legend heading. Point legend markers and their labels are left aligned using compact keys.

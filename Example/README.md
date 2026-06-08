# Jarvis-PLOT — design-reference debug examples

One example per style card. Each figure sets `debug: true`, which draws the
**design-reference overlay**: every axis is outlined and annotated with its
`rect = [left, bottom, width, height]` (figure fractions), its width/height in
centimetres, the figure size, and the margin / colorbar-gap dimension lines.

The figures have **no data layers** on purpose — they are pure layout references
for each style card.

## Run

```bash
jplot a4paper_2x1_rectcmap.yaml      # → plots/a4paper_2x1_rectcmap.png
```

(`plots/` and `.cache/` are created on the fly and are git-ignored.)

## Styles covered

| File | Style `[family, variant]` |
|------|---------------------------|
| `a4paper_2x1_rect.yaml`            | `[a4paper_2x1, rect]` |
| `a4paper_2x1_rectcmap.yaml`        | `[a4paper_2x1, rectcmap]` |
| `a4paper_2x1_Ternary.yaml`         | `[a4paper_2x1, Ternary]` |
| `a4paper_2x1_TernaryCmap.yaml`     | `[a4paper_2x1, TernaryCmap]` |
| `a4paper_2x1_rect_5x1.yaml`        | `[a4paper_2x1, rect_5x1]` (5 panels) |
| `a4paper_2x1_dynesty_runplot.yaml` | `[a4paper_2x1, dynesty_runplot]` (5 panels) |
| `a4paper_4x1_rectcmap.yaml`        | `[a4paper_4x1, rectcmap]` |
| `gambit_2x1_rectcmap.yaml`         | `[gambit_2x1, rectcmap]` |
| `gambit_2x1_Ternary.yaml`          | `[gambit_2x1, Ternary]` |
| `gambit_2x1_TernaryCmap.yaml`      | `[gambit_2x1, TernaryCmap]` |

These are every style registered in `jarvisplot/cards/style_preference.json`,
except `a4paper_1x1/Ternary` and `gambit_1x1/Ternary`, whose card JSONs are still
placeholder stubs (no `Frame`) and cannot render yet.

## Notes

- Variant tokens are case-sensitive: `rect`, `rectcmap` (lower-case) and
  `Ternary`, `TernaryCmap` (capitalized).
- To turn the overlay off, set `debug: false` (or remove it).
- To see the overlay on a real plot, add normal `layers` to any of these files.

See the docs: **Jarvis-PLOT → Style Cards and Layout** for a full explanation of
the card JSON structure and the overlay.

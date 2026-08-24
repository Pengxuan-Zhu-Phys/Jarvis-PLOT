# Jarvis-PLOT — design-reference debug examples

One example per style card. Each figure sets `debug: true`, which draws the
**design-reference overlay**: every axis is outlined, while all axes'
`rect = [left, bottom, width, height]` (figure fractions) and width/height in
centimetres are listed sequentially in one panel inside the primary `ax`.
The figure size and margin / colorbar-gap dimension lines remain on the outer
overlay.

The figures have **no data layers** on purpose — they are pure layout references
for each style card.

## Run

```bash
jplot Example/a4paper_2x1.yaml
```

(`plots/` and `.cache/` are created on the fly and are git-ignored.)

## Styles covered

Examples are grouped by geometry family and paper ratio:

| YAML | Style `[family, variant]` |
|------|---------------------------|
| `a4paper_2x1.yaml` | All `a4paper_2x1` cards: `rect`, `rectRatio`, `rectcmap`, `Ternary`, `TernaryCmap`, `rect_5x1`, `dynesty_runplot` |
| `a4paper_4x1_rectcmap.yaml` | `[a4paper_4x1, rectcmap]` |
| `gambit_2x1.yaml` | All `gambit_2x1` cards: `rectcmap`, `Ternary`, `TernaryCmap` |

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

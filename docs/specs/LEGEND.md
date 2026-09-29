# Layer-declared legend objects and JSON cards

`Figures[].layers[].legend` declares membership in a legend item. The renderer
regroups components after drawing the real layers, and inherits their colors,
alpha, widths, markers and line styles. `frame.<axes>.legend` owns overall
layout: card selection, object order, columns, position, fonts and spacing.

Use `jplot man legend --json` for the manual and the live installed card catalog.
The data-only resolver in `jarvisplot/legend_cards.py` is shared by validation
and rendering. Native rendering lives in `jarvisplot/Figure/legend_runtime.py`.

## Forward declaration

```yaml
frame:
  ax:
    legend:
      card: paper
      order: [measurement, theory]
      ncols: 2
      loc: upper left
      handleheight: 1.2
layers:
  - name: measured_curve
    axes: ax
    method: plot
    coordinates: {x: [0, 1], y: [0.4, 0.8]}
    style: {color: '#A23E00', linewidth: 0.7}
    legend: {object: measurement, label: Measurement, role: line}
  - name: measured_errors
    axes: ax
    method: errorbar
    coordinates: {x: [0, 1], y: [0.4, 0.8], yerr: [0.1, 0.1]}
    style: {fmt: none, ecolor: '#A23E00', capsize: 2}
    legend: {object: measurement, label: Measurement, role: errorbar}
  - name: theory
    axes: ax
    method: plot
    coordinates: {x: [0, 1], y: [0.5, 0.7]}
    style: {color: '#005A8D'}
    legend: {object: theory, label: Theory}
```

The layer legend mapping is closed: `object` and `label` are required non-empty
strings, and `role` is optional. `object` identifies the legend item; it is
independent of the layer's `name` and of the card format name. Layer names may
be omitted. Different objects may have identical display labels.

Objects are grouped independently on each axes. Components need not be adjacent
in the layer list. Every component of an object must declare the same label.
A role can appear only once in an object.

For an object containing one layer, `role` can be omitted: that layer's native
artist/container supplies the entire key. This also works for a native
ErrorbarContainer that already includes its line, markers, errors and caps.
For a multi-layer object, every component must supply a role.

The complete role set automatically selects a matching item template in the
resolved card. Template `roles` determine paint/stack order, regardless of layer
order. `errorbar + line` matches `line_errorbar`, whose errorbars are painted
before the center line. `upper + line + lower` matches `error_bounds` and stacks
three curves vertically. Missing roles are errors; the renderer never guesses
an incomplete composite.

Multiple templates with the same role set are allowed if their ordered roles
and key layout agree. Different layouts for the same role set are ambiguous
and rejected. To distinguish custom formats, give them different role sets.
No layer-level `format` or `key` option is required or accepted.

## Overall layout and ordering

Objects appear in first-appearance order by default. `order` contains object
names, not layer names or display labels. Listed objects come first in that
order; unlisted objects follow in their original first-appearance order.
Duplicates and unknown object names are errors. `order: []` keeps the default.
Native multi-column legends fill columns first. Use `ncols`, `loc`,
`bbox_to_anchor`, `fontsize`/`prop`, `handleheight`, `labelspacing`, etc. to
configure the whole legend.

With layer declarations, only declared objects appear: unrelated `style.label`
values are not collected. A layer declaration activates the axes' inherited
legend card even if its default has `enabled: false`. Explicit YAML
`frame.<axes>.legend: false` or `enabled: false` disables it.

An enabled legend requires at least one declared object. `legend: true`
turns on the declared objects; it does not collect artist labels. `style.label`
and `data[].label` are artist/series metadata and never create legend items.
Invalid declarations are checked even when drawing is disabled.

## JSON legend cards

Each installed card has a file at
`jarvisplot/cards/legends/<name>/legend-card.json`.

| Card | Defaults | Inherits |
| --- | --- | --- |
| `default` | Native Matplotlib layout, common item templates | — |
| `compact` | Small unframed layout | `default` |
| `paper` | Longer keys and more vertical room | `compact` |
| `presentation` | Framed layout, font size 10, larger spacing | `default` |

The JSON `legend` block holds overall native Matplotlib kwargs. The `items`
block defines templates with ordered `roles` and `key: overlay` or `stack`.
Cards contain no scientific object names, layer names or labels.

| Item format | Roles in drawing order | Layout |
| --- | --- | --- |
| `line` | `line` | single native handle |
| `marker` | `marker` | single native handle |
| `band` | `band` | single native handle |
| `errorbar` | `errorbar` | native errorbar, including caps/markers |
| `band_line` | `band`, `line` | overlay |
| `line_marker` | `line`, `marker` | overlay |
| `line_errorbar` | `errorbar`, `line` | overlay |
| `error_bounds` | `upper`, `line`, `lower` | stack |

A local card can inherit an installed card or another local JSON file:

```json
{
  "schema_version": 1,
  "extends": "paper",
  "legend": {"handleheight": 1.2, "labelspacing": 0.4},
  "items": {
    "ribbon_curve": {"roles": ["ribbon", "central"], "key": "overlay"}
  }
}
```

Select it with `card: ./my-legend-card.json`. Card paths are relative to the
YAML file; relative `extends` paths are relative to the containing JSON file.
`&JP/` paths work as well. Cycles, missing files and malformed cards are errors.
Mappings merge recursively; arrays and scalars replace inherited values.
YAML `item_formats` can override or add item definitions using the same rules.
For example, change all boundary composites to overlay without editing layers:

```yaml
frame:
  ax:
    legend:
      card: paper
      item_formats:
        error_bounds: {key: overlay}
```

Geometry/style JSON declares defaults at `Frame.<axes>.legend.card`; installed
drawing axes select `compact` with `enabled: false`. Colorbar/logo axes have no
legend card. `jplot cap styles --json` reports the declarations. YAML card
selection and native kwargs override these defaults.
`card` is the only card selector. `preset` is not supported.

## Native keys, spacing and limitations

`overlay` paints handles in template order. `stack` positions equal-height
slots top to bottom, centered inside one native key. Increase `handleheight`
for more separation between upper/central/lower curves; `labelspacing` controls
the gap between item rows. Dimensions use Matplotlib's font-size units.
The legend adds no data artists and does not change axes limits.

A declared layer must return exactly one supported artist/container. Multiple
series are ambiguous and cause a render error; split them into distinct layers
with explicit object/role declarations. Unsupported handles such as images
also produce an error. Native errorbar/bar containers remain intact.
Legend render errors always fail the render; none are downgraded to warnings.

The compact defaults are `frameon: false`, `borderpad: 0.25`,
`labelspacing: 0.25`, `handlelength: 1.6`, `handleheight: 0.7`,
`handletextpad: 0.5`, `columnspacing: 1.0`, `borderaxespad: 0.4`.

## Validation and migration

`jplot validate` checks shapes, object grouping, role completeness, duplicate
roles, conflicting labels, template ambiguity and ordering without importing
Matplotlib. Layer declaration failures report `JP-LEG-005` at the relevant
layer's legend; ordering failures point to axes legend order.
`JP-LEG-003` reports card/item-definition errors.

There is one declaration path. Axes `entries`, `preset`, `handles`, `labels`
and `axes` are unsupported and rejected by schema, validation and runtime.
Figure-level `legend` is also rejected. `JP-LEG-001` reports unsupported
configuration or an enabled legend without declared objects. The former
reverse-reference diagnostics JP-LEG-002/004 and label-mismatch health check
JP-VIZ-009 are removed. Doctor observations expose `legend_object`,
`legend_label` and `legend_role`; grouping validation uses the same resolver
as rendering.

To migrate, move each entry's object identity, label and component roles onto
the corresponding real layers, replace `entries` with optional `order`, and
retain axes layout kwargs. Delete hand-built `legend_key_*` layers and text.
`Example/compact_legend.yaml` demonstrates four train/test objects;
`Example/legend_cards.yaml` demonstrates band/line and line/errorbar objects.
Both generate PNG and PDF without external data.



# SCENE_JSON_SCHEMA

Status: spec only

## Purpose

This document defines the **scene-level JSON contract** that Jarvis-PLOT can consume as a stable plotting/layout input.

For the upcoming Jarvis-HEP flowchart migration, this schema is the key interface boundary:

- **Jarvis-HEP exports semantic flowchart JSON**.
- **Jarvis-PLOT parses that JSON, computes layout, applies style/profile defaults, and renders the final figure**.

This file defines the **scene/schema side** only.
It does **not** define the layout algorithm, renderer internals, or theme implementation.

---

## Design Goal

The scene JSON should be:

- **semantic**, not renderer-specific,
- **typed**, not an ad hoc nested dictionary,
- **stable**, so producers and consumers can evolve safely,
- **lightweight**, so it can be exported from other projects,
- **general enough** to support multiple diagram-like figure types,
- **specific enough** to support deterministic layout/rendering in Jarvis-PLOT.

The first major producer is expected to be Jarvis-HEP flowchart export.

---

## Core Model

A scene is modeled as a **typed graph plus non-geometric metadata**.

Primary entities:

- `scene` = full document
- `layers` = ordered stage/group structure
- `nodes` = visible logical objects
- `edges` = directed relationships between nodes/ports
- `ports` = named connection anchors defined on nodes
- `hints` = optional non-binding layout/style guidance
- `metadata` = document-level provenance / producer information

Important:

- final coordinates do **not** belong in the semantic scene schema,
- final text-measured widths/heights do **not** belong here,
- renderer-specific path coordinates do **not** belong here.

Those belong to Jarvis-PLOT runtime/layout output, not scene input.

---

## Top-Level Structure

A scene document should conceptually follow this shape:

```json
{
  "schema": "jarvisplot.scene/v1",
  "scene_type": "flowchart",
  "scene_id": "workflow_main",
  "title": "Workflow Overview",
  "metadata": {},
  "layers": [],
  "nodes": [],
  "edges": [],
  "hints": {}
}
```

### Required top-level fields

- `schema`
- `scene_type`
- `layers`
- `nodes`
- `edges`

### Recommended top-level fields

- `scene_id`
- `title`
- `metadata`
- `hints`

---

## Top-Level Field Definitions

### `schema`

String schema identifier.

Example:

```json
"schema": "jarvisplot.scene/v1"
```

This must version the contract explicitly.

---

### `scene_type`

String describing the semantic scene family.

Examples:

- `flowchart`
- `dependency_graph`
- `pipeline_overview`
- `diagram`

For the Jarvis-HEP migration, the initial value will likely be `flowchart`.

---

### `scene_id`

Producer-defined stable identifier for the scene.

Examples:

- workflow name
- run name
- document slug

Should be unique enough for logging/debugging/artifact naming.

---

### `title`

Optional human-readable scene title.

This is semantic document metadata, not a guarantee that the renderer must display a title.

---

### `metadata`

Optional object carrying provenance and producer information.

Typical fields may include:

- `producer`
- `producer_version`
- `source_project`
- `source_file`
- `created_at`
- `notes`

Example:

```json
"metadata": {
  "producer": "Jarvis-HEP",
  "producer_version": "1.6.10",
  "source_project": "workflow export"
}
```

Consumers should ignore unknown metadata fields.

---

## Layers

### Purpose

`layers` define ordered grouping/staging information for scene layout.

A layer is a semantic ordering/grouping concept and, for flowchart scenes, it also defines a logical column in the layout. The layer itself is still not a final renderer coordinate system, but its existence is mandatory because each layer corresponds to one column.

### Layer object shape

```json
{
  "id": "layer_1",
  "index": 1,
  "label": "Parameters",
  "nodes": ["Parameters"],
  "role": "source_layer",
  "hints": {}
}
```

### Required layer fields

- `id`
- either `index` or meaningful order in list position

### Recommended layer fields

- `label`
- `nodes`
- `role`
- `hints`

### Notes

- `nodes` should reference node ids.
- A node may also carry its own `layer` field.
- If both are present, they must be consistent.
- Layout engines may use `index`, list order, or both.

---

### Flowchart constraint

For `scene_type = "flowchart"`, `layers` are mandatory and must be interpreted as ordered columns.

This means:

- every flowchart scene must provide a `layers` array,
- each layer represents one logical column,
- layout engines may still choose exact spacing and geometry,
- but they must preserve the column order defined by the layer sequence and/or layer index.

`layers` are therefore part of the required semantic contract for flowchart scenes, not an optional grouping convenience.

---

## Nodes

### Purpose

Nodes are first-class visible semantic objects in the scene.

The schema intentionally promotes more than just “main modules” to node status.
Depending on the producer, nodes may include:

- modules
- files
- variables
- sources/sinks
- bridge/inter-layer relay nodes
- future domain-specific diagram objects

### Node object shape

```json
{
  "id": "CalcA",
  "kind": "module",
  "role": "calculator",
  "label": "CalcA",
  "layer": "layer_2",
  "in_ports": [],
  "out_ports": [],
  "hints": {},
  "metadata": {}
}
```

### Required node fields

- `id`
- `kind`

### Recommended node fields

- `role`
- `label`
- `layer`
- `in_ports`
- `out_ports`
- `hints`
- `metadata`

### Node field semantics

#### `id`
Stable scene-local node identifier.
Must be unique within the scene.

#### `kind`
Broad object category for layout/render logic.

Examples:

- `module`
- `file`
- `variable`
- `source`
- `sink`
- `bridge`
- `group`

#### `role`
More specific semantic role layered on top of `kind`.

Examples:

- `calculator`
- `operas`
- `parameter_source`
- `input_file`
- `output_file`
- `observable`
- `nuisance`
- `parameter`

`kind` is the broader structural class.
`role` is the finer semantic class.

#### `label`
Human-readable display label.

Renderer may choose whether/how to display it.

#### `layer`
References a layer id when the node belongs to a semantic layer.

#### `metadata`
Producer-specific semantic metadata.
Must not contain renderer-only geometry.

---

## Ports

### Purpose

Ports are named connection anchors defined on nodes.

Ports are **not** first-class nodes in the initial schema.
They are properties of nodes.

This keeps the graph expressive without exploding graph size.

### Port object shape

```json
{
  "id": "m0",
  "role": "parameter",
  "label": "m0",
  "side": "left",
  "metadata": {}
}
```

### Required port fields

- `id`

### Recommended port fields

- `role`
- `label`
- `side`
- `metadata`

### Notes

- Port ids only need to be unique within their parent node.
- The producer may define separate `in_ports` and `out_ports` arrays.
- Consumers should interpret port identity in the context of the parent node.

### `side`
Optional non-binding hint.

Examples:

- `left`
- `right`
- `top`
- `bottom`

This is not a final coordinate instruction.
It is only a semantic/layout hint.

---

## Edges

### Purpose

Edges define directed relationships between nodes/ports.

Edges are semantic connections, not renderer paths.

### Edge object shape

```json
{
  "id": "e1",
  "source": {"node": "Parameters", "port": "m0"},
  "target": {"node": "CalcA", "port": "m0"},
  "role": "parameterflow",
  "hints": {},
  "metadata": {}
}
```

### Required edge fields

- `source`
- `target`

### Recommended edge fields

- `id`
- `role`
- `hints`
- `metadata`

### Endpoint shape

```json
{"node": "NodeId", "port": "PortId"}
```

The `port` field is optional if the edge is defined at node level only.

Examples:

```json
{"node": "CalcA"}
```

### Edge role examples

- `dataflow`
- `parameterflow`
- `fileflow`
- `bridgeflow`
- `dependency`
- `controlflow`

For Jarvis-HEP flowcharts, the first relevant roles are likely:

- `parameterflow`
- `fileflow`
- `dataflow`
- `bridgeflow`

---

## Hints

### Purpose

`hints` provide optional, non-binding information that may help layout/style decisions.

Hints must not be treated as mandatory renderer instructions unless a specific consumer explicitly chooses to do so.

### Allowed hint categories

Examples:

- preferred ordering inside a layer,
- attachment preference,
- side preference,
- compactness group,
- grouping or clustering intent,
- visibility preference,
- emphasis priority.

### Examples

Node-level example:

```json
"hints": {
  "placement": "attached",
  "belongs_to": "CalcA",
  "preferred_side": "right"
}
```

Layer-level example:

```json
"hints": {
  "preferred_order": ["CalcA", "CalcB", "CalcC"]
}
```

Top-level example:

```json
"hints": {
  "layout_family": "flowchart"
}
```

### Non-goal

Hints must not turn into a hidden coordinate system.

Do not encode:

- final x/y coordinates,
- final box width/height,
- renderer path control points,
- text measurement output.

Those belong elsewhere.

---

## Producer Rules

A producer emitting scene JSON should follow these rules.

### Producer must

- emit a valid `schema` string,
- keep node ids unique within the scene,
- keep edge endpoint references valid,
- keep layer/node references internally consistent,
- keep the scene semantic rather than renderer-specific.

### Producer should

- emit stable ids when practical,
- emit `role` information where meaningful,
- use `metadata` for provenance rather than overloading `label`,
- use `hints` only for soft guidance.

### Producer must not

- emit final coordinates as part of the semantic scene contract,
- emit text-bbox-derived geometry,
- embed matplotlib-specific objects or concepts,
- hard-code style implementation details into semantic fields.

---

## Consumer Rules

A Jarvis-PLOT-side consumer should follow these rules.

### Consumer must

- validate required top-level structure,
- validate node/edge references,
- tolerate unknown metadata fields,
- tolerate additional optional fields where safe.

### Consumer should

- use `kind` and `role` together for layout/render decisions,
- interpret hints as advisory rather than absolute,
- preserve schema-version awareness.

### Consumer must not

- assume non-flowchart scenes can omit all structural grouping forever without checking `scene_type`; for `scene_type = "flowchart"`, `layers` are required,
- assume every edge names ports,
- assume every node kind/role is already known forever,
- assume missing hints imply invalid input.

---

## Initial Jarvis-HEP Flowchart Mapping

This is an initial conceptual mapping, not a final locked contract.

### Likely node kinds

- `source`
- `module`
- `file`
- `variable`
- `bridge`

### Likely roles

- `parameter_source`
- `sampler_source`
- `calculator`
- `operas`
- `input_file`
- `output_file`
- `parameter`
- `observable`
- `nuisance`

### Likely edge roles

- `parameterflow`
- `dataflow`
- `fileflow`
- `bridgeflow`

### Important mapping principle

Jarvis-HEP should export **diagram semantics**, not reuse old draw-time internal geometry fields.

Do not carry over fields whose only meaning came from the old matplotlib implementation.

---

## Minimal Example

```json
{
  "schema": "jarvisplot.scene/v1",
  "scene_type": "flowchart",
  "scene_id": "demo_workflow",
  "metadata": {
    "producer": "Jarvis-HEP",
    "producer_version": "1.6.10"
  },
  "layers": [
    {
      "id": "layer_1",
      "index": 1,
      "label": "Parameters",
      "nodes": ["Parameters"]
    },
    {
      "id": "layer_2",
      "index": 2,
      "label": "Calculators",
      "nodes": ["CalcA"]
    }
  ],
  "nodes": [
    {
      "id": "Parameters",
      "kind": "source",
      "role": "parameter_source",
      "label": "Parameters",
      "layer": "layer_1",
      "out_ports": [
        {"id": "m0", "role": "parameter", "label": "m0"}
      ]
    },
    {
      "id": "CalcA",
      "kind": "module",
      "role": "calculator",
      "label": "CalcA",
      "layer": "layer_2",
      "in_ports": [
        {"id": "m0", "role": "parameter", "label": "m0"}
      ],
      "out_ports": [
        {"id": "result", "role": "observable", "label": "result"}
      ]
    }
  ],
  "edges": [
    {
      "id": "e1",
      "source": {"node": "Parameters", "port": "m0"},
      "target": {"node": "CalcA", "port": "m0"},
      "role": "parameterflow"
    }
  ]
}
```

---

## Explicit Non-Goals

This schema file does **not** define:

- layout algorithm details,
- edge routing mathematics,
- final renderer draw order,
- style bundle internals,
- profile merge rules,
- scene-to-runtime object conversion internals,
- interactive editor semantics.

Those belong to other Jarvis-PLOT design/spec documents.

---

## Future Evolution

Likely future extensions may include:

- group/cluster nodes,
- scene-level annotations,
- explicit legend metadata,
- visibility policies,
- subscene composition,
- layout-family-specific hint registries,
- runtime scene format distinct from semantic input scene format.

Any future extension should preserve the main invariant:

**semantic scene JSON is input to Jarvis-PLOT layout/rendering, not a dump of renderer state.**

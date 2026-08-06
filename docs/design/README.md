# Design

This directory holds architecture notes and subsystem design docs.

## Purpose

Use these docs to understand the intended split between input parsing, dataflow, layout, style/profile handling, and rendering.

## Current Files

- `ARCHITECTURE_OVERVIEW.md`: implemented narrow-table runtime architecture and ownership snapshot (v1.4.2)
- `DATAFLOW_ARCHITECTURE.md`: implemented three-table dataflow model
- `LAYOUT_ENGINE_DESIGN.md`: partial layout boundary — flowchart classic layout is implemented; shared multi-diagram layout engine is not
- `STYLE_SYSTEM_DESIGN.md`: partial style system boundary and future owner contract
- `PROFILE_SYSTEM_DESIGN.md`: implemented-but-mixed profile system boundary
- `YAML_DESIGN.md`: YAML schema reference — transforms, methods, coordinates, style, complexity analysis

## Reading Order

1. `ARCHITECTURE_OVERVIEW.md`
2. `DATAFLOW_ARCHITECTURE.md`
3. the subsystem design doc relevant to the change

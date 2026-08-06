# Jarvis-PLOT Layout Engine Design

Status: partial
Last updated: 2026-07-16

## Purpose

This document defines the intended layout boundary for Jarvis-PLOT.

The layout engine should own:

- node sizing
- node placement
- edge routing
- panel and column arrangement
- placement decisions derived from already-normalized semantic scene input

It should not own rendering, source loading, or style selection.

## Current Reality

Layout behavior is currently split across two paths:

### YAML figure path

Implicit layout / axes geometry lives in:

- `jarvisplot/Figure/figure.py`
- `jarvisplot/Figure/layout_runtime.py`
- `jarvisplot/Figure/adapters_rect.py`
- `jarvisplot/Figure/adapters_ternary.py`
- `jarvisplot/Figure/helper.py`

There is still no dedicated shared layout engine for figure panels.

### Flowchart path

Classic flowchart layout is implemented inside:

- `jarvisplot/flowchart.py` (`_ClassicGraph.layout` and related geometry helpers)

This is a real owner for the classic Jarvis-HEP flowchart grammar, but it is not yet a reusable multi-diagram layout engine.

## Intended Boundary

Future layout code should consume normalized semantic scene input and produce layout decisions only.

The output of the layout stage should be data, not matplotlib artists.

## Non-Goals

- loading CSV/HDF5 sources
- applying style cards
- choosing output filenames
- creating final matplotlib artists

## Flowchart Migration Note

Flowchart layout already lives in `flowchart.py`. Future general diagram types should grow a shared layout owner rather than landing new geometry rules in `figure.py` or adapters.

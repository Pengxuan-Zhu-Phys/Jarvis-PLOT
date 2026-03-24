# Jarvis-PLOT Layout Engine Design

Status: spec only

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

Layout behavior is currently implicit in:

- `jarvisplot/Figure/figure.py`
- `jarvisplot/Figure/adapters.py`
- `jarvisplot/Figure/adapters_rect.py`
- `jarvisplot/Figure/adapters_ternary.py`
- `jarvisplot/Figure/helper.py`

There is no dedicated layout owner yet. That is the gap this document is tracking.

## Intended Boundary

Future layout code should consume normalized semantic scene input and produce layout decisions only.

The output of the layout stage should be data, not matplotlib artists.

## Non-Goals

- loading CSV/HDF5 sources
- applying style cards
- choosing output filenames
- creating final matplotlib artists

## Flowchart Migration Note

Future Jarvis-HEP flowchart support should land here first, not in the renderer.

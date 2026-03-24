# Developer Docs

This directory holds the active implementation rules for Jarvis-PLOT.

## Purpose

Use these docs when changing data loading, transforms, caching, rendering, or memory-sensitive paths.

## Current Files

- `CONTRIBUTION_GUIDE.md`: how to extend the pipeline without widening it accidentally
- `DEVELOPER_RULES.md`: mandatory rules for data loading, profiling, caching, and layer rendering
- `MEMORY_OPTIMIZATION_GUIDE.md`: why the narrow-table architecture exists and how to preserve it
- `MEMTRACE_SYSTEM.md`: how to use memory tracing and how to interpret its output

## Reading Order

1. `DEVELOPER_RULES.md`
2. `CONTRIBUTION_GUIDE.md`
3. `MEMORY_OPTIMIZATION_GUIDE.md`
4. `MEMTRACE_SYSTEM.md` when memory behavior matters

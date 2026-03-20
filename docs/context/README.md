# Context

This directory contains the short, high-signal context documents that contributors and coding agents should read before making implementation changes in Jarvis-PLOT.

## Purpose

Files here provide the fastest route to understanding:

- what Jarvis-PLOT currently is
- which document defines the active implementation boundary
- where to look next for framework logic, design context, and schema details

## Primary Entry Point

Start with:

1. `docs/context/JARVIS_PLOT_CONTEXT.md`

That file is the canonical Codex-facing pre-read. It defines the current project shape, ownership boundary, conceptual stack, runtime ownership, flowchart migration boundary, and active engineering rules.

## Current Files

- `JARVIS_PLOT_CONTEXT.md`: primary context document and boundary guide
- `JARVIS_PLOT_FRAMEWORK_LOGIC.md`: runtime execution contract for figure build, data pipeline, cache, and render dispatch
- `CODEX_JARVIS_PLOT_STYLE.md`: local editing policy and project-specific change checklist
- `JARVIS_PLOT_CODE_REVIEW_2026-03-04.md`: review snapshot with concrete findings and debt markers

## How To Use This Directory

Recommended reading order:

1. `JARVIS_PLOT_CONTEXT.md`
2. `JARVIS_PLOT_FRAMEWORK_LOGIC.md`
3. other files here only if the change needs their narrower guidance

Use this directory first when you need to:

- orient a new contributor or coding agent
- decide which layer owns a change
- recover the Jarvis-PLOT versus Jarvis-HEP boundary before refactoring
- find the right deeper design or schema doc to read next

## Maintenance Rule

Keep files in this directory short, current, and implementation-oriented.
Do not keep stale historical notes here as active guidance.

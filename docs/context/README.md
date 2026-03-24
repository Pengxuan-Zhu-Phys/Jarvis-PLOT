# Context

This directory contains the shortest, highest-signal docs for Jarvis-PLOT implementation work.

## Purpose

Files here answer three questions quickly:

- what Jarvis-PLOT currently owns
- which code paths own which responsibilities
- where the flowchart migration boundary starts and stops

## Primary Entry Point

Start with:

1. `docs/context/JARVIS_PLOT_CONTEXT.md`

That file is the canonical Codex-facing pre-read. It defines the current project shape, boundary with Jarvis-HEP, conceptual stack, runtime ownership, and the flowchart migration contract.

## Current Files

- `JARVIS_PLOT_CONTEXT.md`: primary boundary and ownership doc
- `CODE_MAP_JARVIS_PLOT.md`: concrete code owner map for common changes
- `JARVIS_PLOT_FRAMEWORK_LOGIC.md`: runtime execution contract for figure build, data pipeline, cache, and render dispatch
- `CODEX_JARVIS_PLOT_STYLE.md`: local editing policy and project-specific change checklist

## Recommended Reading Order

1. `JARVIS_PLOT_CONTEXT.md`
2. `CODE_MAP_JARVIS_PLOT.md`
3. `JARVIS_PLOT_FRAMEWORK_LOGIC.md`
4. `CODEX_JARVIS_PLOT_STYLE.md` if you are editing code
5. `docs/roadmap/IMPLEMENTATION_ROADMAP.md` if you need the remaining task list

Use this directory first when you need to:

- orient a new contributor or coding agent
- decide which layer owns a change
- recover the Jarvis-PLOT versus Jarvis-HEP boundary before refactoring
- find the right deeper design or schema doc to read next
- hand off open implementation work to `docs/roadmap/IMPLEMENTATION_ROADMAP.md`
- find historical review material in `docs/archive/`

## Maintenance Rule

Keep files in this directory short, current, and implementation-oriented.
Do not keep stale historical notes here as active guidance.

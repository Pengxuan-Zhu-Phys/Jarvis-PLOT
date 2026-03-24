# Jarvis-PLOT Docs

This directory is the tracked documentation entry point for Jarvis-PLOT.

Use it to find the current implementation boundary, subsystem design notes, schema contracts, and developer rules.

## Structure

- `context/`: primary Codex-facing boundary docs and code owner map
- `design/`: architecture and subsystem design notes
- `specs/`: schema and contract docs
- `dev/`: contributor rules, memory notes, and instrumentation guidance
- `roadmap/`: active implementation backlog and future work list
- `templates/`: example scene, style, and profile payloads
- `release/`: release playbooks and version notes
- `archive/`: historical or retired notes

## Primary Entry

Start with:

- `context/JARVIS_PLOT_CONTEXT.md`

Then read:

- `context/CODE_MAP_JARVIS_PLOT.md`
- `context/JARVIS_PLOT_FRAMEWORK_LOGIC.md`
- `roadmap/IMPLEMENTATION_ROADMAP.md` when you need the remaining task list

## Navigation Notes

- `context/` explains project boundaries and current ownership.
- `design/` explains current architecture and the intended split between layers.
- `specs/` defines stable input contracts, especially semantic scene input.
- `dev/` defines rules for safe implementation changes.
- `templates/` provides example payloads that should stay aligned with the spec docs.

# Jarvis-PLOT Docs

This directory is the tracked documentation entry point for Jarvis-PLOT.

Use it to find the current implementation boundary, subsystem design notes, schema contracts, and developer rules.

Current package version: **1.4.2** (`pyproject.toml`).

## Structure

- `context/`: primary Codex-facing boundary docs and code owner map
- `design/`: architecture and subsystem design notes
- `specs/`: schema and contract docs
- `dev/`: contributor rules, memory notes, instrumentation, and CLI design notes
  - `dev/MAN_CLI.md`: **`jplot man` design** (Portal-style dual human/agent manuals)
  - `dev/YAML_HUMAN_AI_OCCAM_REVIEW.md`: YAML Occam review (human+AI)
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
- `specs/POSTERIOR_DENSITY.md` for posterior reconstruction, KDE/query-Voronoi transforms, and HPD contour behavior
- `specs/SCENE_JSON_SCHEMA.md` for flowchart scene JSON (partial; classic flowchart is implemented)
- `specs/AGENT_DATA_API.md` for the planned agent bridge (spec only)
- `roadmap/IMPLEMENTATION_ROADMAP.md` when you need the remaining task list
- `roadmap/soft-cooking-wilkinson.md` for the longer-horizon v2.0 restructure plan

## Navigation Notes

- `context/` explains project boundaries and current ownership.
- `design/` explains current architecture and the intended split between layers.
- `specs/` defines stable input contracts; status labels mark `implemented` / `partial` / `spec only`.
- `dev/` defines rules for safe implementation changes.
- `templates/` provides example payloads that should stay aligned with the spec docs.

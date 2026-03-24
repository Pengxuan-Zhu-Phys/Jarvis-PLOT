# Release Playbook

Status: active

This doc describes the release workflow for Jarvis-PLOT.

## Scope

- Use this playbook for release-specific notes, tags, and publication steps.
- Keep project-wide implementation backlog items in `docs/roadmap/IMPLEMENTATION_ROADMAP.md`.

## Typical Release Flow

1. confirm the release note exists under `docs/release/releases/`
2. update the version note with the current acceptance criteria or task list
3. run the documentation integrity check from `docs/roadmap/IMPLEMENTATION_ROADMAP.md`
4. verify the referenced YAML/template files still match the docs
5. publish or tag the release artifact

## Guardrail

Do not mix active backlog work into the release playbook if it is not release-specific.

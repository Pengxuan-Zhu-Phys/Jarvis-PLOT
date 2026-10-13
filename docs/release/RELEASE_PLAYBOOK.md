# Release Playbook

Status: active

This doc describes the release workflow for Jarvis-PLOT.

## Scope

- Use this playbook for release-specific notes, tags, and publication steps.
- Keep project-wide implementation backlog items in `docs/roadmap/IMPLEMENTATION_ROADMAP.md`.

## Typical Release Flow

1. align `VERSION`, `[project].version` in `pyproject.toml`, live-version docs and `docs/release/releases/vVERSION.md`
2. run `python tools/check_release.py`, focused regressions, the full suite and fatal-error lint
3. build sdist/wheel in a temporary directory and run `python -m twine check <dist files>`
4. run `python tools/check_release.py --wheel <wheel>` and install that wheel non-editably into an isolated temporary target; verify CLI/man/validate/doctor/render from outside the checkout
5. record exact acceptance results and platform limitations in the version note
6. only when explicitly authorized, commit/tag/publish; the tag must be VERSION or vVERSION and the publish workflow repeats the source/wheel gates

CLI and image metadata use `jarvisplot/version.py`, not independent version
parsers. Source fallback is only for an uninstalled checkout. Keep a local
candidate clearly distinguished from a published package.

## Guardrail

Do not mix active backlog work into the release playbook if it is not release-specific.

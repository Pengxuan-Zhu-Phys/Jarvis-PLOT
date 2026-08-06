#!/usr/bin/env python3

"""Config validation that runs without rendering.

Two hard constraints:

- **No matplotlib.** ``jplot validate`` must answer before any renderer is
  imported, otherwise a ten-figure config still takes ten rounds to converge.
- **Never raise on the first problem.** Every check appends to a
  :class:`~jarvisplot.diagnostics.DiagnosticBag` and keeps walking, so one run
  reports everything a caller has to fix.

Ownership split, so one problem never produces two diagnostics:

===========================  ==========================================
the schema catalog owns      shape -- vocabulary, types, enums, required
this module owns             everything a schema cannot see: the
                             filesystem, name uniqueness, cross-block
                             references, and keys the runtime silently
                             ignores
===========================  ==========================================
"""

from __future__ import annotations

import os
from typing import Any

import yaml

from .diagnostics import Diagnostic, DiagnosticBag, Fix, did_you_mean, join_path
from .schema_catalog import config_validator, ignored_properties
from .schema_diagnostics import diagnostics_for_errors
from .utils.pathing import resolve_project_path

__all__ = [
    "FIGURE_SCHEMA",
    "LAYER_SCHEMA",
    "validate_config",
    "validate_file",
]

FIGURE_SCHEMA = "https://jarvis-plot.org/schema/v2/core/figure.json"
LAYER_SCHEMA = "https://jarvis-plot.org/schema/v2/core/layer.json"


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def validate_file(
    path: str,
    *,
    check_columns: bool = True,
) -> tuple[dict[str, Any] | None, DiagnosticBag]:
    """Load and validate one YAML config.

    Returns the parsed config (``None`` when it could not be parsed) alongside
    every diagnostic found. A parse failure short-circuits: there is nothing
    downstream to inspect.
    """
    bag = DiagnosticBag()
    resolved = os.path.abspath(os.path.expanduser(str(path)))

    if not os.path.exists(resolved):
        bag.error(
            "JP-YML-000",
            "$",
            f"config file not found: {resolved}",
            suggestion="Check the path; jplot resolves it relative to the current directory.",
        )
        return None, bag

    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        bag.add(_yaml_parse_diagnostic(exc))
        return None, bag
    except OSError as exc:
        bag.error("JP-YML-000", "$", f"could not read config: {exc}")
        return None, bag

    validate_config(
        config,
        base_dir=os.path.dirname(resolved),
        bag=bag,
        check_columns=check_columns,
    )
    return config, bag


def validate_config(
    config: Any,
    *,
    base_dir: str | None = None,
    bag: DiagnosticBag | None = None,
    check_columns: bool = True,
) -> DiagnosticBag:
    """Validate an already-parsed config mapping.

    ``check_columns=False`` keeps the pass free of any file read at all, for
    callers that only want the shape verdict.
    """
    bag = bag if bag is not None else DiagnosticBag()

    if config is None:
        bag.error(
            "JP-YML-002",
            "$",
            "config is empty",
            suggestion="A config needs at least DataSet and Figures.",
            example="DataSet: []\nFigures:\n  - name: f1\n    layers: []",
        )
        return bag

    if not isinstance(config, dict):
        bag.error(
            "JP-YML-003",
            "$",
            f"config root must be a mapping, got {type(config).__name__}",
            suggestion="The top level of a jplot config is key: value pairs, not a list or scalar.",
        )
        return bag

    _check_schema(config, bag)
    resolved_paths = _check_dataset_files(config, base_dir, bag)
    _check_unique_names(config, bag)
    _check_layer_sources(config, set(resolved_paths) | _collect_shared_names(config), bag)
    _check_ignored_keys(config, bag)
    if check_columns:
        _check_columns_exist(config, resolved_paths, bag)
    return bag


def _yaml_parse_diagnostic(exc: yaml.YAMLError) -> Diagnostic:
    mark = getattr(exc, "problem_mark", None)
    where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark is not None else ""
    problem = getattr(exc, "problem", None) or str(exc)
    return Diagnostic(
        code="JP-YML-001",
        level="error",
        path="$",
        message=f"YAML could not be parsed{where}: {problem}",
        suggestion=(
            "Fix the YAML syntax first; nothing else can be checked until the file parses. "
            "Most often this is inconsistent indentation or a missing quote."
        ),
    )


# --------------------------------------------------------------------------- #
# Shape: delegated to the schema catalog
# --------------------------------------------------------------------------- #


def _check_schema(config: dict[str, Any], bag: DiagnosticBag) -> None:
    """The closed-vocabulary pass: every unknown key becomes a did-you-mean."""
    bag.extend(diagnostics_for_errors(config_validator().iter_errors(config)))


# --------------------------------------------------------------------------- #
# Filesystem
# --------------------------------------------------------------------------- #


def _check_dataset_files(
    config: dict[str, Any],
    base_dir: str | None,
    bag: DiagnosticBag,
) -> dict[str, str]:
    """Resolve every declared data path.

    Returns ``dataset name -> first readable path``, which is what the column
    probe needs next. Names with no readable file are still present (mapped to
    ``""``) so an unrelated source typo is not misreported as a missing dataset.
    """
    readable: dict[str, str] = {}
    for index, entry in enumerate(config.get("DataSet") or ()):
        if not isinstance(entry, dict):
            continue

        name = entry.get("name")
        key = name.strip() if isinstance(name, str) and name.strip() else None
        if key is not None:
            readable.setdefault(key, "")

        raw = entry.get("path")
        if raw is None:
            continue
        sources = raw if isinstance(raw, list) else [raw]
        for offset, source in enumerate(sources):
            if not isinstance(source, str):
                continue
            resolved = resolve_project_path(source, base_dir)
            if resolved.exists():
                if key is not None and not readable[key]:
                    readable[key] = str(resolved)
                continue
            where = join_path("DataSet", index, "path")
            if isinstance(raw, list):
                where = join_path(where, offset)
            bag.error(
                "JP-DAT-004",
                where,
                f"data file not found: {resolved}",
                suggestion=(
                    "Paths resolve relative to the config file's directory "
                    "(or use the &JP/ prefix for repo-relative paths)."
                ),
                context={"declared": source, "resolved": str(resolved)},
            )
    return readable


# --------------------------------------------------------------------------- #
# Name uniqueness
# --------------------------------------------------------------------------- #


def _check_unique_names(config: dict[str, Any], bag: DiagnosticBag) -> None:
    """Names are addresses. A duplicate silently redirects or overwrites."""
    _report_duplicates(
        config.get("DataSet") or (),
        "DataSet",
        "JP-DAT-005",
        "Dataset names are how layers address data; make them unique.",
        bag,
    )
    _report_duplicates(
        config.get("Figures") or (),
        "Figures",
        "JP-FIG-003",
        "Figure names become output filenames; a later figure overwrites the earlier one.",
        bag,
    )


def _report_duplicates(
    entries: Any,
    container: str,
    code: str,
    suggestion: str,
    bag: DiagnosticBag,
) -> None:
    seen: dict[str, int] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        if name in seen:
            bag.error(
                code,
                join_path(container, index, "name"),
                f"duplicate name {name!r} (also at {container}[{seen[name]}])",
                suggestion=suggestion,
            )
        seen[name] = index


# --------------------------------------------------------------------------- #
# Cross-block references
# --------------------------------------------------------------------------- #


def _collect_shared_names(config: dict[str, Any]) -> set[str]:
    """Names published by ``layers[].share_data``, usable as a later ``source``.

    Collected across every figure without regard to order: enforcing
    produce-before-consume needs the usage plan (``Figure/data_pipelines.py``),
    and guessing here would only manufacture false positives.
    """
    names: set[str] = set()
    for figure in config.get("Figures") or ():
        if not isinstance(figure, dict):
            continue
        for layer in figure.get("layers") or ():
            if not isinstance(layer, dict):
                continue
            shared = layer.get("share_data")
            if isinstance(shared, str) and shared.strip():
                names.add(shared.strip())
    return names


def _check_layer_sources(
    config: dict[str, Any],
    known_sources: set[str],
    bag: DiagnosticBag,
) -> None:
    """Catch ``source: df_smaples`` before it becomes a render-time KeyError."""
    if not known_sources:
        return

    for figure_index, figure in enumerate(config.get("Figures") or ()):
        if not isinstance(figure, dict) or "type" in figure:
            # A `type:` figure has not been through figure_types expansion yet,
            # and that expansion injects its own share_data names.
            continue
        for layer_index, layer in enumerate(figure.get("layers") or ()):
            if not isinstance(layer, dict):
                continue
            base = join_path("Figures", figure_index, "layers", layer_index)
            for block_index, block in enumerate(layer.get("data") or ()):
                if not isinstance(block, dict):
                    continue
                source = block.get("source")
                sources = source if isinstance(source, list) else [source]
                for offset, item in enumerate(sources):
                    if not isinstance(item, str) or item in known_sources:
                        continue
                    where = join_path(base, "data", block_index, "source")
                    if isinstance(source, list):
                        where = join_path(where, offset)
                    near = did_you_mean(item, known_sources)
                    hint = f" Did you mean {near[0]!r}?" if near else ""
                    bag.error(
                        "JP-REF-001",
                        where,
                        f"unknown dataset source {item!r}.{hint}",
                        suggestion=(
                            "data[].source must name a root DataSet entry, or a name "
                            "published by another layer's share_data."
                        ),
                        context={
                            "available_sources": sorted(known_sources),
                            "did_you_mean": near,
                        },
                    )


# --------------------------------------------------------------------------- #
# Column existence
# --------------------------------------------------------------------------- #


def _check_columns_exist(
    config: dict[str, Any],
    resolved_paths: dict[str, str],
    bag: DiagnosticBag,
) -> None:
    """Answer "does column 'aa' exist?" before the renderer raises `name 'aa' is not defined`.

    Header reads only -- no rows are materialized. The demand map comes from
    :func:`jarvisplot.column_demand.plan_source_demand`, which attributes columns
    to the source a layer actually names instead of the union used for pruning.
    """
    from .column_demand import plan_source_demand
    from .column_probe import probe_dataset_columns

    entries = {
        str(entry["name"]).strip(): entry
        for entry in config.get("DataSet") or ()
        if isinstance(entry, dict) and isinstance(entry.get("name"), str) and entry["name"].strip()
    }
    demand = plan_source_demand(config)

    for name, entry in entries.items():
        wanted = demand.get(name)
        path = resolved_paths.get(name) or ""
        if wanted is None or not path:
            continue
        candidates = wanted.missing_candidates
        if not candidates:
            continue

        probe = probe_dataset_columns(entry, path)
        if probe.error:
            bag.warning(
                "JP-COL-900",
                join_path("DataSet", _index_of(config, name), "path"),
                f"column names could not be checked for {name!r}: {probe.error}",
                suggestion="Fix the file or pass --no-columns to skip the column check.",
            )
            continue
        if not probe.supported or not probe.names:
            continue

        missing = sorted(c for c in candidates if not probe.resolves(c))
        if not missing:
            continue

        available = sorted(probe.names)
        fallback = join_path("DataSet", _index_of(config, name), "name")
        for column in missing:
            near = did_you_mean(column, available)
            hint = f" Did you mean {near[0]!r}?" if near else ""
            origins = wanted.where(column)
            for where in origins or [fallback]:
                bag.error(
                    "JP-COL-001",
                    where,
                    f"dataset {name!r} has no column {column!r}.{hint}",
                    suggestion=(
                        "Run `jplot data describe <file>` for the real column names; "
                        "that is the only reliable source."
                    ),
                    fix=(
                        Fix(op="set_value", path=where, old=column, to=near[0], confidence="heuristic")
                        if near and len(origins) == 1
                        else None
                    ),
                    context={
                        "dataset": name,
                        "column": column,
                        "did_you_mean": near,
                        "available_columns": available[:60],
                        "available_count": len(available),
                    },
                )


def _index_of(config: dict[str, Any], name: str) -> int:
    for index, entry in enumerate(config.get("DataSet") or ()):
        if isinstance(entry, dict) and str(entry.get("name", "")).strip() == name:
            return index
    return 0


# --------------------------------------------------------------------------- #
# Silently-ignored keys
# --------------------------------------------------------------------------- #


def _check_ignored_keys(config: dict[str, Any], bag: DiagnosticBag) -> None:
    """Report keys the runtime accepts into the dict but never reads.

    These are worse than errors: the figure still renders, just not the way the
    config says. The forwarding address comes from the schema's
    ``x-jarvis-ignored`` annotation, so that knowledge lives in exactly one place.

    Reported as warnings because they are already ignored today -- promoting them
    to errors would fail configs that have looked fine for months. G2/R3 upgrades
    them once V2 is allowed to break.
    """
    figure_ignored = ignored_properties(FIGURE_SCHEMA)
    coord_ignored = ignored_properties(LAYER_SCHEMA, "$defs", "coordinateSpec")

    for index, figure in enumerate(config.get("Figures") or ()):
        if not isinstance(figure, dict):
            continue
        figure_path = join_path("Figures", index)
        _report_ignored(figure, figure_path, figure_ignored, bag)

        for layer_index, layer in enumerate(figure.get("layers") or ()):
            if not isinstance(layer, dict):
                continue
            coordinates = layer.get("coordinates")
            if not isinstance(coordinates, dict):
                continue
            layer_path = join_path(figure_path, "layers", layer_index, "coordinates")
            for axis, spec in coordinates.items():
                if isinstance(spec, dict):
                    _report_ignored(spec, join_path(layer_path, axis), coord_ignored, bag)


def _report_ignored(
    node: dict[str, Any],
    path: str,
    ignored: dict[str, str],
    bag: DiagnosticBag,
) -> None:
    for key, belongs_to in ignored.items():
        if key not in node:
            continue
        bag.warning(
            "JP-OWN-001",
            join_path(path, key),
            f"{key!r} is accepted here but never read; it has no effect",
            suggestion=f"Move it to {belongs_to}.",
            context={"belongs_to": belongs_to},
        )

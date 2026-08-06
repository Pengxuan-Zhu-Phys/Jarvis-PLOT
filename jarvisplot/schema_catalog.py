#!/usr/bin/env python3

"""Loader for the file-composed JSON Schema catalog under ``jarvisplot/schema/``.

Generic on purpose. Adding a method, transform, figure type or dataset format
means adding a schema file and one manifest line -- **this module never
changes**. Jarvis-PLOT dispatches on five axes (``Figures[].type``,
``layers[].method``, transform type, ``DataSet[].type``, style tokens), so an
index that stays data-only is worth more here than it is upstream in
Jarvis-HEP v2, where the same pattern covers two.

The registry is local-only: every ``$ref`` resolves against a file named in the
manifest. Nothing is fetched, ever.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Mapping

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

__all__ = [
    "MANIFEST_PATH",
    "SCHEMA_DIR",
    "ZONES",
    "catalog_lint_errors",
    "config_validator",
    "ignored_properties",
    "iter_schema_files",
    "load_manifest",
    "schema_catalog",
    "subschema",
]


SCHEMA_DIR = Path(__file__).with_name("schema")
MANIFEST_PATH = SCHEMA_DIR / "manifest.json"

#: ``closed``    an unknown key is an error, and gets a did-you-mean.
#: ``delegated`` a downstream owner (matplotlib, a style card) owns the keywords.
#:
#: Jarvis-HEP v2 also has an ``open`` zone for un-migrated surfaces. Jarvis-PLOT
#: deliberately does not: V2 is allowed to break, so there is no reason to build
#: the back door on day one.
ZONES = frozenset({"closed", "delegated"})

#: Keywords whose subschemas only *test* an instance; they define no surface.
_ASSERTION_KEYWORDS = frozenset({"if", "not"})


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def iter_schema_files() -> Iterator[tuple[str, Path]]:
    """Yield ``(relative_name, path)`` for every manifest-listed schema."""
    for relative_name in load_manifest()["schema_files"]:
        yield relative_name, SCHEMA_DIR / relative_name


# --------------------------------------------------------------------------- #
# Catalog self-check
# --------------------------------------------------------------------------- #


def catalog_lint_errors() -> list[str]:
    """Authoring errors in the bundled catalog itself.

    Living next to the loader is the point: a missing ``x-jarvis-zone`` becomes
    a hard failure at load time rather than a silent hole in the closed surface.
    """
    errors: list[str] = []

    def walk(node: Any, location: str, *, asserting: bool = False) -> None:
        if isinstance(node, Mapping):
            zone = node.get("x-jarvis-zone")
            if node.get("type") == "object" and zone is None and not asserting:
                errors.append(f"{location}: object schema has no x-jarvis-zone")
            if zone is not None and zone not in ZONES:
                errors.append(
                    f"{location}: invalid x-jarvis-zone {zone!r} (allowed: {sorted(ZONES)})"
                )
            if zone == "closed" and node.get("type") == "object":
                has_bound = "additionalProperties" in node or "patternProperties" in node
                if not has_bound:
                    errors.append(
                        f"{location}: closed zone must pin additionalProperties"
                    )
            ignored = node.get("x-jarvis-ignored")
            if ignored is not None and not isinstance(ignored, str):
                errors.append(f"{location}: x-jarvis-ignored must name where the key belongs")
            for key, child in node.items():
                walk(
                    child,
                    f"{location}/{key}",
                    # `if` / `not` subschemas are type assertions, not surface
                    # definitions, so there is no ownership question to answer.
                    asserting=asserting or key in _ASSERTION_KEYWORDS,
                )
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{location}/{index}", asserting=asserting)

    for relative_name, path in iter_schema_files():
        if not path.exists():
            errors.append(f"{relative_name}: listed in manifest but missing on disk")
            continue
        with path.open(encoding="utf-8") as handle:
            try:
                schema = json.load(handle)
            except json.JSONDecodeError as exc:
                errors.append(f"{relative_name}: not valid JSON ({exc})")
                continue
        if not isinstance(schema, Mapping) or "$id" not in schema:
            errors.append(f"{relative_name}: schema has no $id")
            continue
        walk(schema, relative_name)

    return errors


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def schema_catalog() -> tuple[dict[str, Any], Registry]:
    """Load the manifest-selected schemas into a local-only registry."""
    manifest = load_manifest()

    lint_errors = catalog_lint_errors()
    if lint_errors:
        raise RuntimeError(
            "Invalid Jarvis-PLOT schema catalog:\n  " + "\n  ".join(lint_errors)
        )

    schemas: list[dict[str, Any]] = []
    for _, path in iter_schema_files():
        with path.open(encoding="utf-8") as handle:
            schema = json.load(handle)
        Draft202012Validator.check_schema(schema)
        schemas.append(schema)

    registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema, default_specification=DRAFT202012))
        for schema in schemas
    )
    return manifest, registry


@lru_cache(maxsize=1)
def config_validator() -> Draft202012Validator:
    """The composed root validator for a Jarvis-PLOT YAML config."""
    manifest, registry = schema_catalog()
    root = registry.contents(manifest["root"])
    return Draft202012Validator(root, registry=registry)


# --------------------------------------------------------------------------- #
# Annotation lookups
# --------------------------------------------------------------------------- #


def subschema(schema_id: str, *path: str) -> dict[str, Any]:
    """Fetch a nested piece of a catalog schema, e.g. ``("$defs", "dataBlock")``."""
    _, registry = schema_catalog()
    node: Any = registry.contents(schema_id)
    for step in path:
        node = node[step]
    return node


@lru_cache(maxsize=16)
def ignored_properties(schema_id: str, *path: str) -> dict[str, str]:
    """Map ``property name -> where the key actually belongs``.

    Reads the ``x-jarvis-ignored`` annotation. This exists because Jarvis-PLOT's
    worst failure mode is not a confusing error but *silence*: a key the runtime
    never reads looks exactly like a key that worked. Declaring those keys in the
    schema turns them into a warning with a forwarding address.
    """
    node = subschema(schema_id, *path)
    return {
        name: str(spec["x-jarvis-ignored"])
        for name, spec in node.get("properties", {}).items()
        if isinstance(spec, Mapping) and "x-jarvis-ignored" in spec
    }

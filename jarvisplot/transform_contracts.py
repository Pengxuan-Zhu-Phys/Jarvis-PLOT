#!/usr/bin/env python3

"""Agent-facing contracts for every transform step name.

``schema/core/transform.json`` only pins the vocabulary; several heavy steps are
``x-jarvis-zone: delegated`` and previously advertised as
"see the runtime module". This module is the closed contract surface for:

- ``jplot cap transforms``
- ``jplot man transforms`` / ``jplot man transform.<name>``

Keys and defaults are taken from the runtime owners
(``preprocessor_runtime``, ``profile_runtime``, ``density_cell_runtime``,
``posterior_density_runtime``, ``interp_2d_runtime``). Prefer updating this
module when a runtime key is added.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "TRANSFORM_NAMES",
    "RUNTIME_TOP_LEVEL_KEYS",
    "contract_for",
    "contract_top_level_keys",
    "list_contracts",
]

# --------------------------------------------------------------------------- #
# Shared field fragments
# --------------------------------------------------------------------------- #

_COORD_AXIS = {
    "type": "object|string",
    "description": (
        "Axis field: string column name, or mapping with expr / name / lim / scale."
    ),
    "properties": {
        "expr": {"type": "expression", "description": "column expression"},
        "name": {"type": "string", "description": "output column name for this axis"},
        "lim": {"type": "array[2]", "description": "[lo, hi] domain"},
        "scale": {"type": "enum", "enum": ["linear", "log"], "default": "linear"},
    },
}

_COORD_BLOCK = {
    "type": "object",
    "description": "Named axes (x/y/z or ternary left/right/bottom).",
    "properties": {
        "x": _COORD_AXIS,
        "y": _COORD_AXIS,
        "z": _COORD_AXIS,
        "left": _COORD_AXIS,
        "right": _COORD_AXIS,
        "bottom": _COORD_AXIS,
    },
}


def _c(
    *,
    description: str,
    form: str,
    value: dict[str, Any] | None = None,
    required: dict[str, Any] | None = None,
    optional: dict[str, Any] | None = None,
    defaults: dict[str, Any] | None = None,
    enums: dict[str, list[Any]] | None = None,
    input_kind: str = "table",
    output_kind: str = "table",
    owner: str = "",
    examples: list[dict[str, Any]] | None = None,
    notes: list[str] | None = None,
    form_extra: str = "",
) -> dict[str, Any]:
    return {
        "description": description,
        "form": form,
        "form_note": form_extra,
        "value": value or {},
        "required": required or {},
        "optional": optional or {},
        "defaults": defaults or {},
        "enums": enums or {},
        "input": input_kind,
        "output": output_kind,
        "owner": owner,
        "examples": examples or [],
        "notes": notes or [],
    }


TRANSFORM_CONTRACTS: dict[str, dict[str, Any]] = {
    "filter": _c(
        description=(
            "Boolean expression over columns; rows evaluating false are dropped. "
            "Literal true/false (or 1/0) keeps or empties the whole table."
        ),
        form="scalar",
        value={
            "type": "string|bool|number",
            "description": 'e.g. "LogL > -100" or true',
        },
        owner="Figure/preprocessor_runtime.py::filter_df",
        examples=[
            {"title": "cut", "yaml": 'transform:\n  - filter: "LogL > -100"\n'},
            {"title": "keep all", "yaml": "transform:\n  - filter: true\n"},
        ],
        notes=["Uses the same expression language as coordinates / data eval."],
    ),
    "sortby": _c(
        description="Sort rows by an expression or column name (ascending).",
        form="scalar",
        value={
            "type": "string|list[string]",
            "description": "column / expression, or list of them",
        },
        owner="Figure/preprocessor_runtime.py::sort_by",
        examples=[{"title": "by LogL", "yaml": "transform:\n  - sortby: LogL\n"}],
    ),
    "add_column": _c(
        description="Append one derived column from an expression.",
        form="object",
        required={
            "name": {"type": "identifier", "description": "new column name"},
            "expr": {"type": "expression", "description": "expression over existing columns"},
        },
        optional={
            "fillna": {"type": "any", "description": "fill NaN after evaluation"},
        },
        owner="Figure/preprocessor_runtime.py::add_column",
        examples=[
            {
                "title": "ratio",
                "yaml": (
                    "transform:\n"
                    "  - add_column: {name: ratio, expr: \"m_A / tanb\"}\n"
                ),
            }
        ],
    ),
    "keep_columns": _c(
        description="Project to a subset of columns (others dropped).",
        form="scalar",
        value={"type": "string|list[string]", "description": "columns to keep"},
        owner="Figure/preprocessor_runtime.py::keep_columns",
        examples=[
            {
                "title": "list",
                "yaml": "transform:\n  - keep_columns: [m_A, tanb, LogL]\n",
            }
        ],
    ),
    "drop_columns": _c(
        description="Drop named columns.",
        form="scalar",
        value={"type": "string|list[string]", "description": "columns to drop"},
        owner="Figure/preprocessor_runtime.py::drop_columns",
        examples=[
            {"title": "drop", "yaml": "transform:\n  - drop_columns: [tmp, flag]\n"}
        ],
    ),
    "duplicate": _c(
        description=(
            "Detach from a shared table before editing it. Use it as the first "
            "step of a block whose source is a table another block published."
        ),
        form="scalar",
        value={"type": "boolean", "description": "true to copy; false is a no-op"},
        owner="Figure/preprocessor_runtime.py",
        examples=[
            {
                "title": "work on a private copy",
                "yaml": "transform:\n  - duplicate: true\n",
            }
        ],
    ),
    "to_df": _c(
        description=(
            "Publish this block's finished table under a name a later data[] "
            "block can use as source. Must be the last step of its block."
        ),
        form="scalar|object",
        value={"type": "string|object", "description": "name string or {name: ..., keep: ...}"},
        optional={
            "name": {"type": "identifier", "description": "table name"},
            "keep": {
                "type": "boolean",
                "description": "also hand the table to the layer (default false)",
            },
        },
        owner="Figure/preprocessor_runtime.py",
        examples=[
            {
                "title": "produce a table for a later block",
                "yaml": "transform:\n  - filter: 'split == \"test\"'\n  - to_df: sig_rows\n",
            }
        ],
        notes=[
            "By default the producing block draws nothing: it hands the table to "
            "the name instead of to the layer.",
            "The name carries a chain signature, so changing anything upstream "
            "invalidates it in the cache without touching unrelated tables.",
        ],
    ),
    "to_ds": _c(
        description=(
            "Store the finished table in this block's own in-memory DataSet entry "
            "and release the scratch tables to_df published in this layer."
        ),
        form="scalar",
        value={"type": "boolean", "description": "true to store and clean up"},
        owner="Figure/preprocessor_runtime.py",
        examples=[
            {
                "title": "settle the result into an env dataset",
                "yaml": "transform:\n  - duplicate: true\n  - to_ds: true\n",
            }
        ],
        notes=["Needs the block to name a single source (a `pd.DataFrame` entry)."],
    ),
    "to_csv": _c(
        description=(
            "Write the table at this pipeline point to CSV (debug aid). "
            "Honoured at dataset load and layer preprocess."
        ),
        form="scalar|object",
        value={"type": "string|object", "description": "path string or {path: ...}"},
        optional={
            "path": {"type": "path", "description": "output path"},
        },
        owner="Figure/preprocessor_runtime.py + data_loader_runtime",
        examples=[{"title": "debug dump", "yaml": "transform:\n  - to_csv: ./debug/step.csv\n"}],
        notes=["Not a render step; does not change the figure pipeline shape."],
    ),
    "to_parquet": _c(
        description="Write the table at this pipeline point to Parquet (dataset-level today).",
        form="scalar|object",
        value={"type": "string|object", "description": "path string or {path: ...}"},
        optional={"path": {"type": "path"}},
        owner="data_loader_runtime",
        examples=[
            {"title": "debug dump", "yaml": "transform:\n  - to_parquet: ./debug/step.parquet\n"}
        ],
    ),
    "profile": _c(
        description=(
            "Profile / reduce an objective over 2D support cells "
            "(bridson mesh or regular grid)."
        ),
        form="object",
        form_extra="single-key mapping only (not {type: profile, ...})",
        required={
            "coordinates": {
                **_COORD_BLOCK,
                "description": "Must provide x, y, z (or ternary left/right/bottom + z).",
            },
        },
        optional={
            "method": {
                "type": "enum",
                "description": "reduction mesh strategy",
                "default": "bridson",
            },
            "bin": {"type": "int", "description": "mesh density parameter", "default": 100},
            "objective": {
                "type": "enum",
                "description": "how z is reduced inside a cell",
                "default": "max",
            },
            "grid_points": {
                "type": "enum",
                "description": "cell geometry hint",
                "default": "rect",
            },
            "fill_empty": {
                "type": "bool",
                "description": "fill empty cells (grid method)",
                "default": False,
            },
            "empty_value": {"type": "number", "description": "value for empty cells when fill_empty"},
            "pregrid": {
                "type": "object|bool",
                "description": (
                    "Optional coarse pre-binning before Bridson/grid profile "
                    "(large tables). Mapping: {bin, enable}; false disables; "
                    "omit for auto-prebin from row count."
                ),
                "properties": {
                    "bin": {
                        "type": "int",
                        "description": "pre-bin count (overrides auto rule)",
                    },
                    "enable": {
                        "type": "bool",
                        "default": True,
                        "description": "set false to skip pregrid while keeping other keys",
                    },
                },
            },
            "pregrid_bin": {
                "type": "int",
                "description": "Shorthand for pregrid.bin (same effect as pregrid: {bin: N}).",
            },
        },
        defaults={
            "method": "bridson",
            "bin": 100,
            "objective": "max",
            "grid_points": "rect",
            "fill_empty": False,
        },
        enums={
            "method": ["bridson", "grid"],
            "objective": ["max", "min", "mean", "sum"],
            "grid_points": ["rect", "hex"],
        },
        owner="Figure/profile_runtime.py::profiling / grid_profiling",
        examples=[
            {
                "title": "bridson profile",
                "yaml": (
                    "transform:\n"
                    "  - profile:\n"
                    "      method: bridson\n"
                    "      bin: 100\n"
                    "      objective: max\n"
                    "      coordinates:\n"
                    "        x: {expr: m_A, lim: [0.1, 5000], scale: log}\n"
                    "        y: {expr: tanb, lim: [1, 60]}\n"
                    "        z: {expr: LogL, name: z}\n"
                ),
            },
            {
                "title": "large table with explicit pregrid",
                "yaml": (
                    "transform:\n"
                    "  - profile:\n"
                    "      method: bridson\n"
                    "      bin: 100\n"
                    "      pregrid: {bin: 300, enable: true}\n"
                    "      coordinates:\n"
                    "        x: {expr: m_A}\n"
                    "        y: {expr: tanb}\n"
                    "        z: {expr: LogL}\n"
                ),
            },
        ],
        notes=[
            "Heavy step: dryrun skips it (doctor status=partial is expected).",
            "Prefer type: profile_2d unless you need custom layer stacks.",
            "pregrid / pregrid_bin are user-writable (profile_runtime); no bins/seed on profile "
            "(those belong to make_density_core / posterior_density).",
        ],
    ),
    "make_density_core": _c(
        description="Build posterior mass support cells (core of density reconstruction).",
        form="object",
        form_extra="single-key or {type: make_density_core, ...}",
        required={
            "x": _COORD_AXIS,
            "y": _COORD_AXIS,
            "weight": {
                "type": "object|string",
                "description": "sample weight (often exp(LogL))",
                "properties": {"expr": {"type": "expression"}},
            },
        },
        optional={
            "method": {"type": "enum", "default": "voronoi"},
            "bins": {"type": "int", "default": 64},
            "bin": {"type": "int", "description": "alias of bins"},
            "normalize": {"type": "bool", "default": True},
            "diagnostics": {"type": "bool", "default": True},
            "seed": {"type": "int"},
            "output": {
                "type": "object|string",
                "description": "output column names for x/y/z (or z name string)",
            },
            "voronoi": {"type": "object", "description": "voronoi backend options (k, …)"},
            "adaptive": {"type": "object", "description": "adaptive refinement options"},
            "kde": {"type": "object", "description": "kde options (bw_method, …)"},
            "bw_method": {"type": "string|number", "description": "kde bandwidth shortcut"},
            "coordinates": _COORD_BLOCK,
            "domain": {"type": "object", "description": "optional xlim/ylim/scales"},
        },
        defaults={
            "method": "voronoi",
            "bins": 64,
            "normalize": True,
            "diagnostics": True,
        },
        enums={"method": ["voronoi", "adaptive", "kde", "grid"]},
        owner="Figure/density_cell_runtime.py",
        examples=[
            {
                "title": "voronoi core",
                "yaml": (
                    "transform:\n"
                    "  - make_density_core:\n"
                    "      method: voronoi\n"
                    "      bins: 64\n"
                    "      x: {expr: m_A}\n"
                    "      y: {expr: tanb}\n"
                    "      weight: {expr: exp(LogL)}\n"
                ),
            }
        ],
        notes=["Usually followed by make_interp_2d; or use type: posterior_2d / posterior_density."],
    ),
    "posterior_density": _c(
        description=(
            "Merged posterior-density pipeline (density core + optional grid interp). "
            "Preferred single step vs chaining make_density_core + make_interp_2d."
        ),
        form="object",
        form_extra="single-key or {type: posterior_density, ...}",
        required={
            "x": _COORD_AXIS,
            "y": _COORD_AXIS,
            "weight": {
                "type": "object|string",
                "description": "sample weight expression",
                "properties": {"expr": {"type": "expression"}},
            },
        },
        optional={
            "method": {"type": "enum", "default": "voronoi"},
            "bins": {"type": "int", "default": 64},
            "bin": {"type": "int"},
            "grid": {
                "type": "int|array[2]|object",
                "description": "interp grid size (int, [nx,ny], or {nx,ny})",
                "default": 256,
            },
            "normalize": {"type": "bool", "default": True},
            "diagnostics": {"type": "bool", "default": True},
            "seed": {"type": "int"},
            "output": {
                "type": "object|string",
                "description": "density output name (string) or {x,y,z}",
                "default": "density",
            },
            "nan_policy": {"type": "enum", "default": "strict"},
            "voronoi": {"type": "object"},
            "adaptive": {"type": "object"},
            "kde": {"type": "object"},
            "bw_method": {"type": "string|number"},
            "coordinates": _COORD_BLOCK,
        },
        defaults={
            "method": "voronoi",
            "bins": 64,
            "grid": 256,
            "normalize": True,
            "diagnostics": True,
            "output": "density",
            "nan_policy": "strict",
        },
        enums={
            "method": ["voronoi", "adaptive", "kde", "grid"],
            "nan_policy": ["strict", "ignore", "fill"],
        },
        owner="Figure/posterior_density_runtime.py",
        examples=[
            {
                "title": "voronoi posterior",
                "yaml": (
                    "transform:\n"
                    "  - posterior_density:\n"
                    "      method: voronoi\n"
                    "      bins: 128\n"
                    "      grid: 256\n"
                    "      x: {expr: m_A, lim: [0, 5000]}\n"
                    "      y: {expr: tanb, lim: [1, 60]}\n"
                    "      weight: {expr: exp(LogL)}\n"
                    "      output: density\n"
                ),
            }
        ],
        notes=[
            "Heavy step: dryrun skips it.",
            "type: posterior_2d expands to this + pcolormesh/contour layers.",
        ],
    ),
    "make_interp_2d": _c(
        description="Interpolate scattered (x,y,z) support onto a regular 2D grid.",
        form="object",
        form_extra="single-key or {type: make_interp_2d, ...}",
        required={
            "coordinates": {
                **_COORD_BLOCK,
                "description": "x, y, z of the support samples (expr/name/lim/scale).",
            },
        },
        optional={
            "method": {
                "type": "enum",
                "description": "interpolator backend",
                "default": "natural_neighbor",
            },
            "grid": {
                "type": "int|array[2]|object",
                "description": "int, [nx,ny], or {nx,ny} / {bins}",
                "default": 256,
            },
            "bins": {"type": "int", "description": "alias for square grid size"},
            "bin": {"type": "int"},
            "nx": {"type": "int"},
            "ny": {"type": "int"},
            "nan_policy": {"type": "enum", "default": "strict"},
            "as_density": {
                "type": "bool",
                "description": "treat z as density and re-normalize on the grid",
                "default": False,
            },
            "normalize": {"type": "bool", "default": False},
            "diagnostics": {"type": "bool", "default": True},
            "output": {"type": "object", "description": "{x,y,z} output column names"},
            "output_z": {"type": "string", "description": "shortcut for output z name"},
            "backend_options": {"type": "object"},
            "triangulation": {"type": "object", "description": "for triangulation backends"},
            "griddata": {
                "type": "object",
                "description": "scipy.griddata options when method=griddata",
                "properties": {
                    "kind": {"type": "enum", "enum": ["nearest", "linear", "cubic"]}
                },
            },
            "kind": {
                "type": "string",
                "description": "shortcut for triangulation/griddata kind",
            },
        },
        defaults={
            "method": "natural_neighbor",
            "grid": 256,
            "nan_policy": "strict",
            "as_density": False,
            "normalize": False,
            "diagnostics": True,
        },
        enums={
            "method": [
                "natural_neighbor",
                "linear",
                "cubic",
                "nearest",
                "griddata",
                "rbf",
            ],
            "nan_policy": ["strict", "ignore", "fill"],
            "griddata.kind": ["nearest", "linear", "cubic"],
        },
        owner="Figure/interp_2d_runtime.py",
        examples=[
            {
                "title": "natural neighbor grid",
                "yaml": (
                    "transform:\n"
                    "  - make_interp_2d:\n"
                    "      method: natural_neighbor\n"
                    "      grid: 500\n"
                    "      nan_policy: strict\n"
                    "      coordinates:\n"
                    "        x: {expr: x, scale: linear}\n"
                    "        y: {expr: y, scale: linear}\n"
                    "        z: {expr: z}\n"
                ),
            }
        ],
        notes=[
            "Heavy step: dryrun skips it.",
            "Common after profile or make_density_core before pcolormesh/contour.",
        ],
    ),
}


TRANSFORM_NAMES: tuple[str, ...] = tuple(sorted(TRANSFORM_CONTRACTS))


#: Top-level config keys each heavy runtime is allowed to advertise.
#: Nested axis fields live under ``coordinates.*`` / axis mappings (see ``_COORD_AXIS``).
#: CI asserts ``contract_top_level_keys(name) == RUNTIME_TOP_LEVEL_KEYS[name]``.
#: For ``profile`` the set was grepped from ``Figure/profile_runtime.py`` (user-facing
#: ``prof.get(...)`` / ``"pregrid_bin" in prof``) — no ghost ``bins``/``seed``.
RUNTIME_TOP_LEVEL_KEYS: dict[str, frozenset[str]] = {
    "profile": frozenset(
        {
            "method",
            "bin",
            "coordinates",
            "objective",
            "grid_points",
            "fill_empty",
            "empty_value",
            "pregrid",
            "pregrid_bin",
        }
    ),
    "make_density_core": frozenset(
        {
            "x",
            "y",
            "weight",
            "method",
            "bins",
            "bin",
            "normalize",
            "diagnostics",
            "seed",
            "output",
            "voronoi",
            "adaptive",
            "kde",
            "bw_method",
            "coordinates",
            "domain",
        }
    ),
    "posterior_density": frozenset(
        {
            "x",
            "y",
            "weight",
            "method",
            "bins",
            "bin",
            "grid",
            "normalize",
            "diagnostics",
            "seed",
            "output",
            "nan_policy",
            "voronoi",
            "adaptive",
            "kde",
            "bw_method",
            "coordinates",
        }
    ),
    "make_interp_2d": frozenset(
        {
            "coordinates",
            "method",
            "grid",
            "bins",
            "bin",
            "nx",
            "ny",
            "nan_policy",
            "as_density",
            "normalize",
            "diagnostics",
            "output",
            "output_z",
            "backend_options",
            "triangulation",
            "griddata",
            "kind",
        }
    ),
}


def contract_for(name: str) -> dict[str, Any] | None:
    key = str(name).strip()
    base = TRANSFORM_CONTRACTS.get(key)
    if base is None:
        return None
    out = dict(base)
    out["name"] = key
    return out


def contract_top_level_keys(name: str) -> set[str]:
    """Union of required + optional top-level keys for one contract."""
    c = contract_for(name)
    if c is None:
        return set()
    keys: set[str] = set()
    for block in ("required", "optional"):
        block_map = c.get(block) or {}
        if isinstance(block_map, dict):
            keys.update(block_map.keys())
    return keys


def list_contracts() -> list[dict[str, Any]]:
    return [contract_for(n) for n in TRANSFORM_NAMES if contract_for(n) is not None]

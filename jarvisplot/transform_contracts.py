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


_BACKEND_OPTIONS = {
    "type": "object",
    "description": (
        "Interpolation backend knobs. The three that decide where the picture "
        "is blank are boundary, max_fill_spacing and (with nan_policy) "
        "fill_value; see `jplot man field-interp`."
    ),
    "properties": {
        "boundary": {
            "type": "enum",
            "enum": ["nan", "clamp", "nearest"],
            "description": (
                "How far past the convex hull of the cores the surface is drawn. "
                "Cores sit at the centre of the bin they summarise, so the hull "
                "stops half a cell short of the domain and leaves a blank frame. "
                "nan: draw only the hull (default here; keeps a normalized "
                "density honest). clamp: read the surface at the closest hull "
                "point and carry it out, continuous with the interior. nearest: "
                "carry the closest core's value out."
            ),
            "default": "nan",
        },
        "max_boundary_spacing": {
            "type": "number",
            "description": (
                "Reach of the boundary extension, in multiples of the median "
                "core spacing. Inherits max_fill_spacing when unset."
            ),
            "default": 2.0,
        },
        "max_boundary_distance": {
            "type": "number",
            "description": "Same bound as an absolute distance in interpolation coordinates.",
        },
        "max_fill_spacing": {
            "type": "number",
            "description": (
                "Blank any query point more than this many core spacings from a "
                "real core, so nan_policy: ignore closes one-cell gaps without "
                "closing a region the scan never visited. Off by default."
            ),
        },
        "max_fill_distance": {
            "type": "number",
            "description": "Same coverage bound as an absolute distance.",
        },
        "fill_value": {
            "type": "number",
            "description": (
                "Value given to valueless cores under nan_policy: fill. "
                "Defaults to the smallest finite core value."
            ),
        },
        "vertex_tol": {
            "type": "number",
            "description": "Distance below which two cores are one core (merged by mean).",
        },
        "nominal_point_spacing": {
            "type": "number",
            "description": "Override the measured core spacing the reaches are scaled by.",
        },
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
    see_also: list[str] | None = None,
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
        "see_also": see_also or [],
    }


def _distribution_contract(kind: str) -> dict[str, Any]:
    axis = {
        **_COORD_AXIS,
        "description": "Raw sample expression and common evaluation grid/support.",
        "properties": {
            **_COORD_AXIS["properties"],
            "grid": {"type": "integer", "minimum": 2, "default": 600,
                     "description": "Number of evaluation points; not histogram bins or smoothing strength."},
        },
    }
    pdf = kind == "PDF1D"
    return _c(
        description="Adaptively select true weighted empirical CDF anchors and construct a monotone C2 interpolant"
        + (" and its analytic PDF derivative." if pdf else "."),
        form="mapping",
        required={"coordinates": {
            "type": "object", "description": "x is required; weight is optional (unit weights by default).",
            "properties": {"x": axis, "weight": {
                "type": "object|string", "description": "Nonnegative sample-weight expression.",
                "properties": {"expr": _COORD_AXIS["properties"]["expr"],
                               "name": {"type": "string", "description": "Input column fallback when expr is omitted."}},
            }},
        }},
        optional={
            "repeat": {"type": "string", "description": "Batch-ID column: reconstruct each batch independently, then equal-weight mean and sample std."},
            "anchors": {"type": "object|enum", "description": "adaptive (default), all, or adaptive-selection settings.",
                        "properties": {
                            "method": {"enum": ["adaptive", "all"], "default": "adaptive"},
                            "tolerance": {"type": "number", "default": 0.005, "description": "Target max absolute CDF error at empirical nodes (0 < value < 1)."},
                            "min_mass": {"type": "number", "default": 0.01, "description": "Observed probability required on each side of a new anchor (0 < value < 1); prevents chasing sample noise."},
                            "max_points": {"type": "integer", "minimum": 4, "default": 256, "description": "Anchor budget including support endpoints."},
                        }},
            "interpolation": {"type": "enum", "enum": ["monotone_c2", "pchip"], "default": "monotone_c2",
                              "description": "Local monotone C2 quintic CDF / C1 PDF, or PCHIP C1 cubic CDF / continuous PDF."},
        },
        defaults={"coordinates.x.name": "x", "coordinates.x.grid": 600, "coordinates.x.scale": "linear",
                  "anchors.method": "adaptive", "anchors.tolerance": 0.005, "anchors.min_mass": 0.01,
                  "anchors.max_points": 256, "interpolation": "monotone_c2"},
        input_kind="raw sample table",
        output_kind="grid table: x (or coordinates.x.name), cdf, cdf_std, n_repeats"
        + (", pdf, pdf_std" if pdf else ""),
        owner="Figure/distribution_1d_runtime.py",
        examples=[{"description": "Select one population, then reconstruct its repeat mean/std.",
                   "yaml": f'transform:\n  - filter: \'(sample == "signal") & (split == "train")\'\n  - {kind}:\n      coordinates:\n        x: {{expr: score, lim: [0, 1], grid: 600}}\n        weight: {{expr: weight}}\n      repeat: repeat\n      anchors: {{method: adaptive, tolerance: 0.005, min_mass: 0.01}}\n      interpolation: monotone_c2\n'}],
        notes=[
            "No bins: aggregate duplicate sample values, normalize weights per repeat, interpolate cumulative probabilities, then differentiate for PDF1D.",
            "repeat is optional. cdf/pdf are equal-weight repeat means; *_std is sample standard deviation (ddof=1), not standard error. A single repeat has NaN std.",
            "Use filter before the transform to select populations; no groupby option. Input rows and unrelated columns are replaced by the grid table.",
            "lim is the normalization support and must contain every positive-weight sample. Omit it to pad the pooled sample extrema by half the adjacent distinct-value gap; one unique value requires explicit lim.",
            "F(lo)=0 and F(hi)=1; the analytic PDF integrates to 1 on lim. Selected interior anchors retain their exact empirical probabilities; unselected nodes are approximated. Lower-boundary mass is spread into the first interval with a warning.",
            "log changes grid spacing only: interpolation and PDF units remain in physical x.",
            "Inputs must be finite, weights nonnegative, and each repeat must have positive total weight. Zero-weight samples do not define support.",
            "Adaptive anchors start at sparse probability seeds, then refine up to eight worst CDF-error intervals per pass. min_mass and max_points bound refinement; tolerance is a target, not an unconditional guarantee.",
            "monotone_c2 uses quintic Bernstein/Hermite segments with shared first/second derivatives and nonnegative quartic derivative control coefficients on whole intervals; no nonlinear solver or MQSI dependency.",
            "Reconstruction diagnostics live in DataFrame.attrs['distribution_1d']: per-repeat anchor/candidate counts, max CDF node error, stop reason, and largest empirical jump. Unmet tolerance and significant discrete masses emit warnings.",
            "For the original exact all-node PCHIP reconstruction set anchors: all and interpolation: pchip. Close scores can still force spikes in that mode.",
            "Grid changes evaluation density only. A coarse plotted polyline can miss sharp features even though the analytic PDF integral is one.",
            "Heavy step: dryrun checks structure/columns but skips reconstruction and numerical validation.",
        ],
        see_also=["transform.CDF1D" if pdf else "transform.PDF1D", "transform.filter", "plot", "fill_between"],
    )


TRANSFORM_CONTRACTS: dict[str, dict[str, Any]] = {
    "PDF1D": _distribution_contract("PDF1D"),
    "CDF1D": _distribution_contract("CDF1D"),
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
    "significance": _c(
        description=(
            "Align a binned signal table and a binned background table on the "
            "bin key and evaluate a figure of merit per bin."
        ),
        form="object",
        form_extra="single-key mapping or {type: significance, ...}",
        required={
            "dfS": {
                "type": "object",
                "description": "{source: <table>, Sn: <expr>, sigmaSn: <expr>}",
            },
            "dfB": {
                "type": "object",
                "description": "{source: <table>, Bn: <expr>, sigmaBn: <expr>}",
            },
        },
        optional={
            "key": {"type": "identifier", "description": "column both tables align on"},
            "formula": {"type": "string", "description": "figure of merit"},
            "cumulative": {
                "type": "boolean",
                "description": "Z above each threshold instead of inside each bin",
            },
            "drop_invalid": {
                "type": "boolean",
                "description": "drop bins where Z is undefined instead of leaving NaN",
            },
        },
        defaults={
            "key": "bin_index",
            "formula": "s_over_sqrt_sb",
            "cumulative": False,
            "drop_invalid": False,
        },
        enums={
            "formula": [
                "s_over_sqrt_sb",
                "s_over_sqrt_b",
                "s_over_sqrt_b_syst",
                "asimov",
                "asimov_syst",
            ]
        },
        input_kind="two published tables",
        output_kind="table (one row per shared bin)",
        owner="Figure/significance_runtime.py",
        examples=[
            {
                "title": "signal and background from separate files",
                "yaml": (
                    "transform:\n"
                    "  - significance:\n"
                    "      dfS: {source: sig_bins, Sn: Sn}\n"
                    "      dfB: {source: bkg_bins, Bn: Bn, sigmaBn: '0.2 * Bn'}\n"
                    "      formula: s_over_sqrt_b_syst\n"
                ),
            }
        ],
        notes=[
            "The two tables come from bin_stat and must share the same bins; the "
            "step aligns on the integer bin index, not on a float bin centre.",
            "Undefined bins stay NaN by default, so a step plot breaks the line "
            "rather than bridging over them.",
            "The _syst formulas need dfB.sigmaBn; asimov_syst reduces to asimov "
            "as the uncertainty goes to zero.",
        ],
    ),
    "bin_stat": _c(
        description=(
            "Collapse the table into one row per bin of x (weighted, 1D). "
            "Emits bin_index / x_lo / x_hi / x_center plus the binned column."
        ),
        form="object",
        form_extra="single-key mapping or {type: bin_stat, ...}",
        required={
            "x": {"type": "string", "description": "column name or expression to bin along"},
            "bins": {"type": "int|list", "description": "bin count, or explicit increasing edges"},
        },
        optional={
            "weights": {
                "type": "string",
                "description": "column name or expression; omit to count each row as 1",
            },
            "range": {"type": "list", "description": "[lo, hi]; required when bins is a count"},
            "out": {"type": "identifier", "description": "name of the binned column"},
            "normalise": {"type": "string", "description": "sum | integral | none"},
            "scale_to": {
                "type": "object",
                "description": "{name, value} -> add a column name = value * <out>",
            },
        },
        defaults={"out": "density", "normalise": "sum"},
        enums={"normalise": ["sum", "integral", "none"]},
        output_kind="table (one row per bin)",
        owner="Figure/bin_stat_runtime.py",
        examples=[
            {
                "title": "weighted shape, rescaled to a total",
                "yaml": (
                    "transform:\n"
                    "  - bin_stat:\n"
                    "      x: score\n"
                    "      weights: weight\n"
                    "      bins: 25\n"
                    "      range: [0.0, 1.0]\n"
                    "      scale_to: {name: Sn, value: 10000}\n"
                ),
            }
        ],
        notes=[
            "normalise: sum makes the bins add to 1, so scale_to.value * density is "
            "the expected count in that bin. normalise: integral is matplotlib's "
            "density=True and differs by a bin width.",
            "Only rows landing inside the bin range take part in the normalisation.",
            "scale_to.value may be a column, but it has to be constant over the "
            "block -- filter first if it is not.",
        ],
    ),
    "correlation": _c(
        description=(
            "Compute unweighted Pearson correlations between selected event-level "
            "columns and emit one long-table row per requested variable pair."
        ),
        form="object",
        form_extra="single-key mapping or {type: correlation, ...}",
        required={},
        optional={
            "columns": {"type": "list[string]", "description": "explicit source columns, in matrix order"},
            "regex": {"type": "string", "description": "take numeric columns matching this pattern"},
            "exclude": {"type": "string|list[string]", "description": "names to drop from the selection"},
            "missing": {"type": "enum", "description": "listwise | pairwise"},
            "min_periods": {"type": "int", "description": "minimum valid entries per pair"},
            "triangle": {"type": "enum", "description": "full | upper | lower"},
            "include_diagonal": {"type": "boolean", "description": "emit self-correlations"},
        },
        defaults={
            "columns": "every numeric column",
            "missing": "listwise",
            "min_periods": 2,
            "triangle": "full",
            "include_diagonal": True,
        },
        enums={
            "missing": ["listwise", "pairwise"],
            "triangle": ["full", "upper", "lower"],
        },
        output_kind="long table (var_x, var_y, x_index, y_index, rho, abs_rho, n)",
        owner="Figure/correlation_runtime.py",
        examples=[
            {
                "title": "BDT input correlations, transform to matrix",
                "yaml": (
                    "frame:\n"
                    "  ax:\n"
                    "    xlim: [-0.5, 2.5]\n"
                    "    ylim: [-0.5, 2.5]\n"
                    "    ticks:\n"
                    "      x: {positions: [0, 1, 2], labels: [m_ttbar, p_star, delta_y]}\n"
                    "      y: {positions: [0, 1, 2], labels: [m_ttbar, p_star, delta_y]}\n"
                    "  axc:\n"
                    "    color: {cmap: coolwarm, vmin: -1.0, vmax: 1.0}\n"
                    "layers:\n"
                    "  - name: rho\n"
                    "    data:\n"
                    "      - source: events\n"
                    "        transform:\n"
                    "          - correlation:\n"
                    "              exclude: [weight, label, event_id]\n"
                    "              triangle: upper\n"
                    "    axes: ax\n"
                    "    # imshow takes z alone: the cells come from the grid\n"
                    "    # this step publishes, not from x/y columns.\n"
                    "    method: imshow\n"
                    "    colorbar: axc\n"
                    "    coordinates: {z: {expr: rho}}\n"
                    "    style: {interpolation: nearest}\n"
                ),
            },
            {
                "title": "Other ways to pick the features",
                "yaml": (
                    "- correlation: {}                              # every numeric column\n"
                    "- correlation: {exclude: [weight, label]}      # ... minus the bookkeeping ones\n"
                    "- correlation: {regex: '^bdt_'}                # ... matching a prefix\n"
                    "- correlation: {columns: [m_ttbar, p_star]}    # ... or named outright\n"
                ),
            },
        ],
        notes=[
            "Unweighted by design: this is the standard Pearson redundancy diagnostic, not a yield-weighted observable.",
            "Naming N features by hand costs N strings, so the default names none of them: "
            "with no columns and no regex the step takes every numeric column. In a feature "
            "table the bookkeeping columns are the short list, so `exclude` is usually what "
            "you want to write.",
            "Order is what x_index counts: an explicit `columns` keeps the order you wrote, "
            "while `regex` and the default keep the table's own column order.",
            "Bool columns, non-numeric columns and the private __* ones are never picked up "
            "automatically -- a flag is a label by construction, not a feature.",
            "A step that selects turns off column projection for its source, so the table "
            "loads whole. That is the cost of not naming the columns: with an explicit "
            "`columns` list the loader still reads only what it needs.",
            "listwise reproduces a dataframe after finite-value cleaning of all selected BDT inputs.",
            "x_index and y_index are positions in `columns`, so `triangle: upper` keeps "
            "y_index >= x_index -- the half above the diagonal once drawn with y upward.",
            "The step also publishes the private __grid_* columns that say how big the "
            "matrix is. Without them a long table does not carry its own shape, and "
            "pcolormesh would infer one from the row count -- right only when every "
            "cell is present, so a triangle would silently draw on the wrong mesh.",
            "The colour range belongs on frame.axc.color.vmin/vmax, not on the layer: a "
            "bound colorbar owns the norm, and a correlation plot wants a fixed "
            "symmetric [-1, 1] rather than one that follows the data.",
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
                "description": (
                    "grid method: floor cells that caught no sample instead of "
                    "leaving them NaN. The floor is min(finite z) - 0.1 unless "
                    "empty_value says otherwise. Affects the stored grid, so a "
                    "raw pcolormesh sees it too; to smooth over the gaps at "
                    "draw time instead, leave this off and let the "
                    "interpolation drop them (nan_policy: ignore)."
                ),
                "default": False,
            },
            "empty_value": {"type": "number", "description": "value for empty cells when fill_empty"},
            "pregrid": {
                "type": "object|bool",
                "description": (
                    "Optional coarse pre-binning before Bridson/grid profile "
                    "(large tables). Mapping: {bin, enable}; false disables; "
                    "omit for auto-prebin from row count. Keeps one finite "
                    "extremum per cell according to objective (max/min)."
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
            "method: grid emits one row per cell, the core placed at the cell "
            "CENTRE, and z = NaN for every cell that caught no sample. Raising "
            "bin empties more of them. What the picture then does with those "
            "NaN cells is the interpolation's nan_policy -- see "
            "`jplot man field-interp`.",
        ],
        see_also=["field-interp", "type-profile-2d"],
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
            "nan_policy": {
                "type": "enum",
                "description": "cores with no value: strict keeps them (and blanks their neighbourhood), ignore drops them, fill replaces them",
                "default": "strict",
            },
            "backend_options": _BACKEND_OPTIONS,
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
            "backend_options.boundary": ["nan", "clamp", "nearest"],
        },
        owner="Figure/posterior_density_runtime.py",
        see_also=["field-interp", "type-posterior-2d"],
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
            "Keep backend_options.boundary at nan while normalize is true: the "
            "integral is taken over the drawn grid, so extrapolating past the "
            "support of the cores moves it.",
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
            "nan_policy": {
                "type": "enum",
                "description": "cores with no value: strict keeps them (and blanks their neighbourhood), ignore drops them, fill replaces them",
                "default": "strict",
            },
            "as_density": {
                "type": "bool",
                "description": "treat z as density and re-normalize on the grid",
                "default": False,
            },
            "normalize": {"type": "bool", "default": False},
            "diagnostics": {"type": "bool", "default": True},
            "output": {"type": "object", "description": "{x,y,z} output column names"},
            "output_z": {"type": "string", "description": "shortcut for output z name"},
            "backend_options": _BACKEND_OPTIONS,
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
            "backend_options.boundary": ["nan", "clamp", "nearest"],
            "griddata.kind": ["nearest", "linear", "cubic"],
        },
        owner="Figure/interp_2d_runtime.py",
        see_also=["field-interp", "type-profile-2d"],
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
            },
            {
                "title": "profile grid with empty bins (the usual gap recipe)",
                "yaml": (
                    "transform:\n"
                    "  - profile: {method: grid, bin: 60, objective: max, coordinates: {...}}\n"
                    "  - make_interp_2d:\n"
                    "      method: natural_neighbor\n"
                    "      grid: 400\n"
                    "      nan_policy: ignore      # empty bins are not cores\n"
                    "      backend_options:\n"
                    "        boundary: clamp       # close the half-cell frame\n"
                    "        max_fill_spacing: 2.5 # but keep real voids blank\n"
                    "      coordinates:\n"
                    "        x: {expr: xx}\n"
                    "        y: {expr: yy}\n"
                    "        z: {expr: zz}\n"
                ),
            },
        ],
        notes=[
            "Heavy step: dryrun skips it.",
            "Common after profile or make_density_core before pcolormesh/contour.",
            "Non-finite z rows are dropped while the input is validated, so on "
            "this transform nan_policy strict and ignore behave the same; the "
            "policy separates them on a drawing layer's style.interp.",
            "Defaults here are the plain ones (nan_policy: strict, boundary: "
            "nan). Drawing paths -- style.interp, jpfield/jpcontour/jpcontourf, "
            "and the type: profile_2d macro -- default to ignore + clamp "
            "instead. See `jplot man field-interp`.",
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
    "PDF1D": frozenset({"coordinates", "repeat", "anchors", "interpolation"}),
    "CDF1D": frozenset({"coordinates", "repeat", "anchors", "interpolation"}),
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
            "backend_options",
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

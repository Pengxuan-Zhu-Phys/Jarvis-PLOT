#!/usr/bin/env python3

"""``jplot suggest`` -- data-aware config synthesis (F2).

Agent supplies intent (kind + column roles); PLOT supplies numbers (scale/lim)
and a decisions[] log with reasons.
"""

from __future__ import annotations

import argparse

from ..cli_help import RichArgumentParser
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from ..agent_io import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, envelope, system_exit_code, error_payload
from ..diagnostics import Diagnostic, DiagnosticBag
from ..templates_catalog import get_template, list_templates, render_template_yaml
from ..validation import validate_config
from .data import describe_file, suggest_axes

__all__ = ["SuggestError", "build_parser", "run", "suggest_config"]


class SuggestError(Exception):
    """Structured suggest failure with a stable JP-TPL-* code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        suggestion: str = "",
        path: str = "$.suggest",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.suggestion = suggestion
        self.path = path
        self.context = context or {}

    def to_diagnostic(self) -> Diagnostic:
        return Diagnostic(
            code=self.code,
            level="error",
            path=self.path,
            message=self.message,
            suggestion=self.suggestion,
            context=dict(self.context),
        )


def build_parser(prog: str = "jplot suggest") -> argparse.ArgumentParser:
    parser = RichArgumentParser(
        prog=prog,
        description=(
            "Synthesize a type-first YAML from a data file. "
            "PLOT chooses scales/lims; agent chooses columns and kind."
        ),
        rich_title="suggest",
        rich_usage=f"{prog} --data <file> --kind <kind> [--json]",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="path to csv / parquet / hdf5",
    )
    parser.add_argument(
        "--kind",
        default="posterior_2d",
        help="template kind (default: posterior_2d); see jplot template list",
    )
    parser.add_argument("--x", default=None, help="x column (default: first parameter-like)")
    parser.add_argument("--y", default=None, help="y column")
    parser.add_argument(
        "--weight",
        default=None,
        help='weight expression for posterior_2d (default: exp(<LogL col>) or first weight col)',
    )
    parser.add_argument(
        "--z",
        default=None,
        help="objective column for profile_2d (default: LogL-like)",
    )
    parser.add_argument(
        "--c",
        default=None,
        help="colour column for scatter_2d",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="figure name (default: from kind)",
    )
    parser.add_argument(
        "--style",
        default=None,
        help="comma-separated style tokens (default: a4paper_2x1,rectcmap)",
    )
    parser.add_argument(
        "--dataset-name",
        default="samples",
        help="DataSet.name in the emitted YAML",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--write",
        default=None,
        metavar="PATH",
        help="write the YAML to PATH after internal validate",
    )
    return parser


def run(argv: Sequence[str], *, prog: str = "jplot suggest") -> int:
    parser = build_parser(prog)
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        return system_exit_code(exc)

    as_json = bool(args.json) or not sys.stdout.isatty()
    try:
        result = suggest_config(
            data_path=args.data,
            kind=args.kind,
            x=args.x,
            y=args.y,
            weight=args.weight,
            z=args.z,
            c=args.c,
            name=args.name,
            style=args.style,
            dataset_name=args.dataset_name,
        )
    except SuggestError as exc:
        bag = DiagnosticBag()
        bag.add(exc.to_diagnostic())
        env = envelope(
            "suggest",
            False,
            data={
                "available_kinds": [t["kind"] for t in list_templates()],
                **exc.context,
            },
            diagnostics=bag,
            error=error_payload(exc.code, exc.message),
        )
        return emit(env) if as_json else _fail(env)
    except Exception as exc:
        bag = DiagnosticBag()
        bag.add(
            Diagnostic(
                code="JP-TPL-000",
                level="error",
                path="$.suggest",
                message=str(exc),
                suggestion="Check --data path, --kind (jplot template list), and column flags.",
            )
        )
        env = envelope(
            "suggest",
            False,
            error=error_payload("JP-TPL-000", str(exc)),
            data={"available_kinds": [t["kind"] for t in list_templates()]},
            diagnostics=bag,
        )
        return emit(env) if as_json else _fail(env)

    # Always validate the synthesized YAML (including column existence against
    # the source file). Agents must not treat a bad --x / weight as success.
    parsed = yaml.safe_load(result["yaml_text"])
    base_dir = str(Path(args.data).expanduser().resolve().parent)
    bag = validate_config(
        parsed,
        base_dir=base_dir,
        check_columns=True,
    )
    if not bag.ok:
        env = envelope(
            "suggest",
            False,
            data={
                **result,
                "wrote": None,
                "validate_error_count": len(bag.errors),
            },
            diagnostics=bag,
            error=error_payload(
                "ValidationError",
                "suggested YAML failed validate"
                + ("; not written" if args.write else ""),
            ),
        )
        return emit(env) if as_json else _fail(env)

    if args.write:
        path = Path(args.write)
        path.write_text(result["yaml_text"], encoding="utf-8")
        result["wrote"] = str(path.resolve())
    else:
        result["wrote"] = None

    env = envelope("suggest", True, data=result)
    if as_json:
        return emit(env)
    print(result["yaml_text"], end="")
    print("", file=sys.stderr)
    for decision in result.get("decisions") or []:
        print(
            f"  decision {decision.get('field')}: {decision.get('value')!r}  "
            f"— {decision.get('reason')}",
            file=sys.stderr,
        )
    return EXIT_OK


def suggest_config(
    *,
    data_path: str,
    kind: str = "posterior_2d",
    x: str | None = None,
    y: str | None = None,
    weight: str | None = None,
    z: str | None = None,
    c: str | None = None,
    name: str | None = None,
    style: str | None = None,
    dataset_name: str = "samples",
) -> dict[str, Any]:
    path = os.path.abspath(os.path.expanduser(str(data_path)))
    if not os.path.isfile(path):
        raise SuggestError(
            "JP-TPL-001",
            f"data file not found: {path}",
            suggestion="Pass an existing --data path (csv / parquet / hdf5).",
            path="$.suggest.data",
            context={"path": path},
        )

    # Ensure kind exists
    try:
        get_template(kind)
    except Exception as exc:
        kinds = [t["kind"] for t in list_templates()]
        near = _did_you_mean(str(kind), kinds)
        hint = f" Did you mean {near[0]!r}?" if near else ""
        raise SuggestError(
            "JP-TPL-002",
            f"unknown template kind {kind!r}.{hint}",
            suggestion="Use a kind from `jplot template list`.",
            path="$.suggest.kind",
            context={"kind": kind, "available_kinds": kinds, "did_you_mean": near},
        ) from exc

    desc = describe_file(path, use_cache=True)
    axes = suggest_axes(path)
    columns = desc.get("columns") or []
    by_name = {col["name"]: col for col in columns}
    col_names = [col["name"] for col in columns]
    if not col_names:
        raise SuggestError(
            "JP-TPL-003",
            f"no columns found in {path}",
            suggestion="Check the file type and path; run `jplot data describe`.",
            path="$.suggest.data",
            context={"path": path},
        )

    decisions: list[dict[str, Any]] = []

    def decide(field: str, value: Any, reason: str) -> Any:
        decisions.append({"field": field, "value": value, "reason": reason})
        return value

    def require_column(name: str, *, field: str) -> str:
        if name not in by_name:
            near = _did_you_mean(name, col_names)
            hint = f" Did you mean {near[0]!r}?" if near else ""
            raise SuggestError(
                "JP-TPL-004",
                f"--{field} column {name!r} not in data file{hint}. Available: {col_names}",
                suggestion="Run `jplot data describe <file>` for legal column names.",
                path=f"$.suggest.{field}",
                context={
                    "field": field,
                    "column": name,
                    "available_columns": col_names,
                    "did_you_mean": near,
                },
            )
        return name

    if x is not None:
        x_col = require_column(str(x), field="x")
        x_col = decide("x", x_col, "user-provided")
    else:
        x_col = _pick_column(columns, role="parameter", exclude=())
        x_col = decide("x", x_col, f"first parameter-like column ({x_col!r})")

    if y is not None:
        y_col = require_column(str(y), field="y")
        y_col = decide("y", y_col, "user-provided")
    else:
        y_col = _pick_column(columns, role="parameter", exclude={x_col})
        y_col = decide("y", y_col, f"next parameter-like column ({y_col!r})")

    # Role-only lookups (no silent fallback that pretends a parameter is LogL).
    logl = _find_role(columns, "log_likelihood")
    weight_col = _find_role(columns, "weight")

    if kind == "posterior_2d":
        if weight:
            weight_expr = str(weight).strip()
            _assert_expr_columns(weight_expr, col_names, field="weight")
            weight_expr = decide("weight", weight_expr, "user-provided weight expression")
        elif logl:
            weight_expr = decide(
                "weight",
                f"exp({logl})",
                f"column {logl!r} has role_hint=log_likelihood",
            )
        elif weight_col:
            weight_expr = decide(
                "weight",
                weight_col,
                f"column {weight_col!r} has role_hint=weight",
            )
        else:
            raise SuggestError(
                "JP-TPL-005",
                "posterior_2d needs a weight: pass --weight 'exp(LogL)' (or a weight "
                "column), or include a LogL/weight column in the data. "
                f"Columns seen: {col_names} (no role_hint=log_likelihood|weight).",
                suggestion="Add --weight 'exp(<LogL_col>)' or use a file with a LogL column.",
                path="$.suggest.weight",
                context={"columns_seen": col_names, "kind": kind},
            )
        z_col = None
        c_col = None
    elif kind == "profile_2d":
        weight_expr = None
        if z is not None:
            z_col = require_column(str(z), field="z")
            z_col = decide("z", z_col, "user-provided")
        elif logl:
            z_col = decide("z", logl, f"role_hint=log_likelihood ({logl!r})")
        else:
            raise SuggestError(
                "JP-TPL-006",
                "profile_2d needs an objective column: pass --z LogL (or similar), "
                f"or include a LogL-like column. Columns seen: {col_names}.",
                suggestion="Pass --z <column> after `jplot data describe`.",
                path="$.suggest.z",
                context={"columns_seen": col_names, "kind": kind},
            )
        c_col = None
    else:
        weight_expr = None
        z_col = None
        if c is not None:
            c_col = require_column(str(c), field="c")
            decide("c", c_col, "user-provided")
        else:
            c_col = logl
            if c_col:
                decide("c", c_col, f"colour from {c_col!r}")

    axis_by_col = {a["col"]: a for a in (axes.get("axes") or [])}
    x_axis = axis_by_col.get(x_col) or {}
    y_axis = axis_by_col.get(y_col) or {}
    xscale = decide(
        "xscale",
        x_axis.get("scale") or "linear",
        x_axis.get("reason") or "default linear",
    )
    yscale = decide(
        "yscale",
        y_axis.get("scale") or "linear",
        y_axis.get("reason") or "default linear",
    )
    xlim = x_axis.get("lim")
    ylim = y_axis.get("lim")
    if xlim:
        decide("xlim", xlim, x_axis.get("reason") or "from data quantiles")
    if ylim:
        decide("ylim", ylim, y_axis.get("reason") or "from data quantiles")

    style_tokens = (
        [t.strip() for t in str(style).split(",") if t.strip()]
        if style
        else ["a4paper_2x1", "rectcmap"]
    )
    decide("style", style_tokens, "default a4paper_2x1 + rectcmap" if not style else "user-provided")

    fig_name = name or kind
    decide("name", fig_name, "user-provided" if name else f"from kind {kind}")

    # detect type from extension
    suffix = Path(path).suffix.lower()
    dtype = "hdf5" if suffix in {".h5", ".hdf5", ".hdf"} else (
        "parquet" if suffix in {".parquet", ".pq"} else "csv"
    )
    decide("dtype", dtype, f"from file extension {suffix or '(none)'}")

    values: dict[str, Any] = {
        "name": fig_name,
        "data": dataset_name,
        "path": path,
        "dtype": dtype,
        "x": x_col,
        "y": y_col,
        "style": style_tokens,
        "xscale": xscale,
        "yscale": yscale,
    }
    if weight_expr is not None:
        values["weight"] = weight_expr
    if z_col is not None:
        values["z"] = z_col
    if c_col is not None:
        values["c"] = c_col

    yaml_text = render_template_yaml(kind, values=values)
    # inject frame lims if we have them (template skeleton only has scales)
    config = yaml.safe_load(yaml_text)
    fig0 = (config.get("Figures") or [{}])[0]
    frame_ax = ((fig0.get("frame") or {}).get("ax")) or {}
    if xlim:
        frame_ax["xlim"] = list(xlim)
    if ylim:
        frame_ax["ylim"] = list(ylim)
    fig0.setdefault("frame", {})["ax"] = frame_ax
    yaml_text = yaml.safe_dump(
        config, sort_keys=False, allow_unicode=True, default_flow_style=False
    )

    return {
        "kind": kind,
        "data_path": path,
        "yaml_text": yaml_text,
        "decisions": decisions,
        "columns_seen": col_names,
        "role_hints": {
            col["name"]: col.get("role_hint")
            for col in columns
            if col.get("role_hint")
        },
    }


def _find_role(columns: list[dict[str, Any]], role: str) -> str | None:
    for col in columns:
        if col.get("role_hint") == role:
            return str(col["name"])
    return None


def _pick_column(
    columns: list[dict[str, Any]],
    *,
    role: str,
    exclude: set[str] | tuple[str, ...],
) -> str:
    excluded = set(exclude)
    # prefer matching role_hint
    for col in columns:
        name = col["name"]
        if name in excluded:
            continue
        if col.get("role_hint") == role:
            return name
    # fallback: numeric parameter-like (not logl/weight)
    for col in columns:
        name = col["name"]
        if name in excluded:
            continue
        if col.get("role_hint") in {"log_likelihood", "weight", "flag", "chi2"}:
            continue
        if "min" in col:  # numeric
            return name
    for col in columns:
        if col["name"] not in excluded:
            return col["name"]
    raise ValueError("no columns available to pick")


def _assert_expr_columns(expr: str, columns: list[str], *, field: str) -> None:
    from ..column_demand import _expr_symbols
    from ..expr_names import EXPR_IDENTIFIER_IGNORE

    symbols = _expr_symbols(expr)
    missing = sorted(
        s for s in symbols if s not in columns and s not in EXPR_IDENTIFIER_IGNORE
    )
    if missing:
        near = _did_you_mean(missing[0], columns)
        hint = f" Did you mean {near[0]!r}?" if near else ""
        raise SuggestError(
            "JP-TPL-007",
            f"--{field} expression references unknown column(s) {missing}.{hint} "
            f"Available: {columns}",
            suggestion="Fix the expression to use real columns from `jplot data describe`.",
            path=f"$.suggest.{field}",
            context={
                "field": field,
                "expr": expr,
                "missing": missing,
                "available_columns": columns,
                "did_you_mean": near,
            },
        )


def _did_you_mean(word: str, candidates: list[str]) -> list[str]:
    from ..diagnostics import did_you_mean

    return did_you_mean(word, candidates)


def _fail(env: dict) -> int:
    print(env.get("error", {}).get("message", "failed"), file=sys.stderr)
    return EXIT_FAILED

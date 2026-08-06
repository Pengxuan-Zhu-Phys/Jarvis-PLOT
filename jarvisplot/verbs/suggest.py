#!/usr/bin/env python3

"""``jplot suggest`` -- data-aware config synthesis (F2).

Agent supplies intent (kind + column roles); PLOT supplies numbers (scale/lim)
and a decisions[] log with reasons.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from ..agent_io import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, envelope, error_payload
from ..templates_catalog import get_template, list_templates, render_template_yaml
from ..validation import validate_config
from .data import describe_file, suggest_axes

__all__ = ["build_parser", "run", "suggest_config"]


def build_parser(prog: str = "jplot suggest") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Synthesize a type-first YAML from a data file. "
            "PLOT chooses scales/lims; agent chooses columns and kind."
        ),
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
        return int(exc.code or EXIT_USAGE)

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
    except Exception as exc:
        env = envelope(
            "suggest",
            False,
            error=error_payload(exc),
            data={"available_kinds": [t["kind"] for t in list_templates()]},
        )
        return emit(env) if as_json else _fail(env)

    if args.write:
        path = Path(args.write)
        # write-validate-rollback
        parsed = yaml.safe_load(result["yaml_text"])
        bag = validate_config(
            parsed,
            base_dir=str(path.parent.resolve()),
            check_columns=False,
        )
        if not bag.ok:
            env = envelope(
                "suggest",
                False,
                data={**result, "validate_error_count": len(bag.errors)},
                diagnostics=bag,
                error=error_payload(
                    "ValidationError",
                    "suggested YAML failed validate; not written",
                ),
            )
            return emit(env) if as_json else _fail(env)
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
        raise FileNotFoundError(path)

    # Ensure kind exists
    get_template(kind)

    desc = describe_file(path, use_cache=True)
    axes = suggest_axes(path)
    columns = desc.get("columns") or []
    by_name = {col["name"]: col for col in columns}
    col_names = [col["name"] for col in columns]
    if not col_names:
        raise ValueError(f"no columns found in {path}")

    decisions: list[dict[str, Any]] = []

    def decide(field: str, value: Any, reason: str) -> Any:
        decisions.append({"field": field, "value": value, "reason": reason})
        return value

    x_col = x or _pick_column(columns, role="parameter", exclude=())
    x_col = decide(
        "x",
        x_col,
        "user-provided" if x else f"first parameter-like column ({x_col!r})",
    )
    y_col = y or _pick_column(columns, role="parameter", exclude={x_col})
    y_col = decide(
        "y",
        y_col,
        "user-provided" if y else f"next parameter-like column ({y_col!r})",
    )

    logl = _pick_column(columns, role="log_likelihood", exclude=())
    weight_col = _pick_column(columns, role="weight", exclude=())

    if kind == "posterior_2d":
        if weight:
            weight_expr = decide("weight", weight, "user-provided weight expression")
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
            weight_expr = decide(
                "weight",
                f"exp({col_names[-1]})",
                "fallback: exp(last column); prefer a real LogL column",
            )
        z_col = None
        c_col = None
    elif kind == "profile_2d":
        weight_expr = None
        z_col = z or logl or col_names[-1]
        z_col = decide(
            "z",
            z_col,
            "user-provided" if z else (
                f"role_hint=log_likelihood ({z_col!r})" if z_col == logl else f"fallback {z_col!r}"
            ),
        )
        c_col = None
    else:
        weight_expr = None
        z_col = None
        c_col = c or logl
        if c_col:
            decide("c", c_col, "user-provided" if c else f"colour from {c_col!r}")

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


def _fail(env: dict) -> int:
    print(env.get("error", {}).get("message", "failed"), file=sys.stderr)
    return EXIT_FAILED

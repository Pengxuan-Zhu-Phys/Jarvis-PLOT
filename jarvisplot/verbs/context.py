#!/usr/bin/env python3

"""``jplot context`` — one-shot agent context pack for data + figure kind.

Aggregates describe / axes / template slots / cap vocab / related transforms /
next CLI / optional YAML skeleton so agents need not hop man↔cap↔template.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from ..agent_io import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_USAGE,
    emit,
    envelope,
    error_payload,
    system_exit_code,
)
from ..cli_help import RichArgumentParser
from ..transform_contracts import contract_for

__all__ = ["build_parser", "run", "build_context"]

# kind → transforms an agent is most likely to need when customizing beyond type:
_KIND_TRANSFORMS: dict[str, list[str]] = {
    "posterior_2d": ["posterior_density", "make_density_core", "make_interp_2d", "filter"],
    "profile_2d": ["profile", "make_interp_2d", "filter", "add_column"],
    "scatter_2d": ["filter", "sortby", "add_column", "keep_columns"],
}


def build_parser(prog: str = "jplot context") -> argparse.ArgumentParser:
    parser = RichArgumentParser(
        prog=prog,
        description=(
            "One-shot agent context: columns, type slots, defaults, vocab, "
            "related transforms, next CLI, and a data-aware YAML skeleton."
        ),
        rich_title="context",
        rich_usage=f"{prog} --data <file> [--kind profile_2d] [--json]",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="path to csv / parquet / hdf5",
    )
    parser.add_argument(
        "--kind",
        default="posterior_2d",
        help="figure type / template kind (default: posterior_2d)",
    )
    parser.add_argument("--x", default=None, help="optional x column override")
    parser.add_argument("--y", default=None, help="optional y column override")
    parser.add_argument("--z", default=None, help="optional z / objective column")
    parser.add_argument("--weight", default=None, help="optional weight expression")
    parser.add_argument("--c", default=None, help="optional colour column (scatter)")
    parser.add_argument(
        "--dataset-name",
        default="samples",
        help="DataSet.name in the skeleton (default: samples)",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def run(argv: Sequence[str], *, prog: str = "jplot context") -> int:
    parser = build_parser(prog)
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        return system_exit_code(exc)

    as_json = bool(args.json) or not sys.stdout.isatty()
    try:
        data = build_context(
            data_path=args.data,
            kind=args.kind,
            x=args.x,
            y=args.y,
            z=args.z,
            weight=args.weight,
            c=args.c,
            dataset_name=args.dataset_name,
        )
    except Exception as exc:
        env = envelope("context", False, error=error_payload(exc))
        return emit(env) if as_json else _fail(env)

    # ok=true if we produced usable context; gaps are listed, not hard-fail
    # (missing LogL is a gap, not a context command failure).
    env = envelope("context", True, data=data)
    if as_json:
        return emit(env)
    _print_human(data)
    return EXIT_OK


def build_context(
    *,
    data_path: str,
    kind: str = "posterior_2d",
    x: str | None = None,
    y: str | None = None,
    z: str | None = None,
    weight: str | None = None,
    c: str | None = None,
    dataset_name: str = "samples",
) -> dict[str, Any]:
    from ..capabilities import section
    from ..templates_catalog import get_template, list_templates, render_template_yaml
    from ..data_access import describe_file, suggest_axes
    from .suggest import SuggestError, suggest_config

    path = os.path.abspath(os.path.expanduser(str(data_path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    kind_key = str(kind or "").strip()
    available_kinds = [t["kind"] for t in list_templates()]
    try:
        tmpl = get_template(kind_key)
    except Exception as exc:
        raise KeyError(
            f"unknown kind {kind_key!r}; choose one of: {', '.join(available_kinds)}"
        ) from exc

    desc = describe_file(path, use_cache=True)
    axes = suggest_axes(path)
    columns = desc.get("columns") or []
    col_names = [col["name"] for col in columns]
    roles = {
        col["name"]: col.get("role_hint")
        for col in columns
        if col.get("role_hint")
    }

    # Cap vocab (compact digests)
    styles = section("styles")
    usable_styles = [
        {"bundle": s.get("bundle"), "token": s.get("token"), "style": s.get("style"), "axes": s.get("axes")}
        for s in styles
        if s.get("usable")
    ][:40]
    cmaps = section("cmaps")
    methods = [
        {
            "name": m["name"],
            "required": (m.get("coordinates") or {}).get("required") or [],
            "optional": (m.get("coordinates") or {}).get("optional") or [],
        }
        for m in section("methods")
    ]
    funcs = section("funcs")

    related_names = list(_KIND_TRANSFORMS.get(kind_key, ["filter", "add_column"]))
    transforms = []
    for tname in related_names:
        tcontract = contract_for(tname)
        if tcontract is None:
            continue
        transforms.append(
            {
                "name": tname,
                "form": tcontract.get("form"),
                "required": tcontract.get("required") or {},
                "optional": tcontract.get("optional") or {},
                "defaults": tcontract.get("defaults") or {},
                "enums": tcontract.get("enums") or {},
                "input": tcontract.get("input"),
                "output": tcontract.get("output"),
                "examples": tcontract.get("examples") or [],
                "man": f"jplot man transform.{tname}",
            }
        )

    gaps: list[dict[str, Any]] = []
    recommendations: dict[str, Any] = {}
    decisions: list[dict[str, Any]] = []
    yaml_text: str | None = None
    suggest_ok = False

    try:
        suggested = suggest_config(
            data_path=path,
            kind=kind_key,
            x=x,
            y=y,
            z=z,
            weight=weight,
            c=c,
            dataset_name=dataset_name,
        )
        yaml_text = suggested.get("yaml_text")
        decisions = list(suggested.get("decisions") or [])
        for d in decisions:
            field = d.get("field")
            if field in {"x", "y", "z", "weight", "c", "xscale", "yscale", "xlim", "ylim", "style"}:
                recommendations[str(field)] = d.get("value")
        suggest_ok = True
    except SuggestError as exc:
        gaps.append(
            {
                "code": exc.code,
                "message": exc.message,
                "suggestion": exc.suggestion,
                "path": exc.path,
                "context": exc.context,
            }
        )
        # Soft recommendations from roles / order when suggest cannot finish.
        recommendations = _soft_recommendations(
            columns, kind=kind_key, x=x, y=y, z=z, weight=weight, c=c
        )
        # Still emit a slot-default skeleton so the agent has something to edit.
        try:
            soft_values = {
                "path": path,
                "data": dataset_name,
                "dtype": _dtype_from_path(path),
                **{k: v for k, v in recommendations.items() if k in {"x", "y", "z", "c"}},
            }
            if "weight" in recommendations:
                soft_values["weight"] = recommendations["weight"]
            yaml_text = render_template_yaml(kind_key, values=soft_values)
        except Exception:
            yaml_text = None
    except Exception as exc:
        gaps.append(
            {
                "code": "JP-TPL-000",
                "message": str(exc),
                "suggestion": "Check --data and --kind.",
            }
        )
        recommendations = _soft_recommendations(
            columns, kind=kind_key, x=x, y=y, z=z, weight=weight, c=c
        )

    slots = tmpl.get("slots") or []
    slot_defaults = {s["name"]: s.get("default") for s in slots if isinstance(s, dict)}

    next_cli = _next_cli(
        path=path,
        kind=kind_key,
        suggest_ok=suggest_ok,
        gaps=gaps,
        recommendations=recommendations,
    )

    return {
        "data": {
            "path": path,
            "type": desc.get("type"),
            "rows": desc.get("rows"),
            "columns": columns,
            "column_names": col_names,
            "role_hints": roles,
            "axes": axes.get("axes") if isinstance(axes, dict) else axes,
        },
        "kind": kind_key,
        "type": {
            "title": tmpl.get("title"),
            "family": tmpl.get("family"),
            "requires": list(tmpl.get("requires") or []),
            "description": tmpl.get("description") or "",
            "slots": slots,
            "defaults": slot_defaults,
            "explain": f"jplot explain {kind_key}",
            "man": _type_man(kind_key),
        },
        "recommendations": recommendations,
        "decisions": decisions,
        "styles": {
            "usable": usable_styles,
            "note": "Prefer tokens with usable=true; full list: jplot cap styles --json",
        },
        "cmaps": {
            "jarvis": (cmaps.get("jarvis") or [])[:30],
            "note": "Full + matplotlib: jplot cap cmaps --json",
        },
        "methods": {
            "items": methods,
            "man": "jplot man methods",
        },
        "funcs": {
            "names": (funcs.get("names") or [])[:40],
            "note": funcs.get("note") or "",
        },
        "transforms": {
            "related": transforms,
            "catalog": "jplot man transforms",
            "note": (
                "Heavy transforms (profile / density / interp) are skipped in dryrun; "
                "doctor may return ok=null coverage=partial — not a config failure."
            ),
        },
        "yaml_skeleton": yaml_text,
        "suggest_ok": suggest_ok,
        "gaps": gaps,
        "next_cli": next_cli,
        "available_kinds": available_kinds,
        "write_yaml": False,
    }


def _type_man(kind: str) -> str:
    if kind == "posterior_2d":
        return "jplot man type-posterior-2d"
    if kind == "profile_2d":
        return "jplot man type-profile-2d"
    return f"jplot explain {kind}"


def _dtype_from_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".h5", ".hdf5", ".hdf"}:
        return "hdf5"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    return "csv"


def _soft_recommendations(
    columns: list[dict[str, Any]],
    *,
    kind: str,
    x: str | None,
    y: str | None,
    z: str | None,
    weight: str | None,
    c: str | None,
) -> dict[str, Any]:
    names = [col["name"] for col in columns]
    by_role = {
        col["name"]: col.get("role_hint")
        for col in columns
    }
    params = [
        n
        for n in names
        if by_role.get(n) not in {"log_likelihood", "weight", "flag", "chi2"}
    ]
    logl = next((n for n, r in by_role.items() if r == "log_likelihood"), None)
    wcol = next((n for n, r in by_role.items() if r == "weight"), None)

    out: dict[str, Any] = {}
    out["x"] = x or (params[0] if params else (names[0] if names else None))
    out["y"] = y or (
        params[1] if len(params) > 1 else (names[1] if len(names) > 1 else out["x"])
    )
    if kind == "posterior_2d":
        if weight:
            out["weight"] = weight
        elif logl:
            out["weight"] = f"exp({logl})"
        elif wcol:
            out["weight"] = wcol
    elif kind == "profile_2d":
        out["z"] = z or logl
    elif kind == "scatter_2d":
        if c:
            out["c"] = c
        elif logl:
            out["c"] = logl
    out["style"] = ["a4paper_2x1", "rectcmap" if kind != "scatter_2d" else "rect"]
    return {k: v for k, v in out.items() if v is not None}


def _next_cli(
    *,
    path: str,
    kind: str,
    suggest_ok: bool,
    gaps: list[dict[str, Any]],
    recommendations: dict[str, Any],
) -> list[dict[str, str]]:
    steps: list[dict[str, str]] = []
    if gaps:
        for g in gaps:
            steps.append(
                {
                    "argv": f"jplot explain {g.get('code', 'JP-TPL-000')} --json",
                    "why": g.get("suggestion") or g.get("message") or "resolve gap",
                }
            )
        if kind == "posterior_2d" and any(g.get("code") == "JP-TPL-005" for g in gaps):
            steps.append(
                {
                    "argv": (
                        f"jplot suggest --data {path} --kind posterior_2d "
                        "--weight 'exp(<LogL_col>)' --json"
                    ),
                    "why": "retry suggest with an explicit weight",
                }
            )
        if kind == "profile_2d" and any(g.get("code") == "JP-TPL-006" for g in gaps):
            steps.append(
                {
                    "argv": f"jplot suggest --data {path} --kind profile_2d --z <col> --json",
                    "why": "retry suggest with objective column",
                }
            )
    elif suggest_ok:
        steps.append(
            {
                "argv": (
                    f"jplot suggest --data {path} --kind {kind} --write plot.yaml --json"
                ),
                "why": "materialize the skeleton YAML",
            }
        )
        steps.append(
            {
                "argv": "jplot doctor plot.yaml --json",
                "why": "validate + dryrun (partial ok for heavy types)",
            }
        )
        steps.append(
            {
                "argv": "jplot plot.yaml",
                "why": "render when you need the figure (bare path; no jplot run)",
            }
        )

    steps.extend(
        [
            {
                "argv": f"jplot man transform.{t} --json",
                "why": "deep-dive a related transform contract",
            }
            for t in _KIND_TRANSFORMS.get(kind, [])[:2]
        ]
    )
    steps.append(
        {
            "argv": f"jplot man methods --json",
            "why": "drawing method catalog if hand-writing layers",
        }
    )
    return steps


def _print_human(data: dict[str, Any]) -> None:
    d = data.get("data") or {}
    print(
        f"context  kind={data.get('kind')}  data={d.get('path')}  "
        f"rows={d.get('rows')}  cols={len(d.get('column_names') or [])}",
        file=sys.stderr,
    )
    rec = data.get("recommendations") or {}
    if rec:
        print("  recommendations:", file=sys.stderr)
        for k, v in rec.items():
            print(f"    {k}: {v!r}", file=sys.stderr)
    gaps = data.get("gaps") or []
    if gaps:
        print("  gaps:", file=sys.stderr)
        for g in gaps:
            print(f"    {g.get('code')}: {g.get('message')}", file=sys.stderr)
    print("  next_cli:", file=sys.stderr)
    for step in data.get("next_cli") or []:
        print(f"    {step.get('argv')}  # {step.get('why')}", file=sys.stderr)
    skel = data.get("yaml_skeleton")
    if skel:
        print("", file=sys.stderr)
        print(skel, end="")


def _fail(env: dict) -> int:
    print(env.get("error", {}).get("message", "failed"), file=sys.stderr)
    return EXIT_FAILED

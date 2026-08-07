#!/usr/bin/env python3

"""``jplot data …`` — CLI skin over :mod:`jarvisplot.data_access`.

Column names only come from the data file. Handlers here parse argv and emit
envelopes; loading / describe / eval live in ``data_access`` so dryrun and
render share one engine (AGENT_DATA_API: agent channel is a skin).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

from ..agent_io import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, envelope, system_exit_code, error_payload
from ..cli_help import RichArgumentParser
from ..data_access import (
    EvalFailed,
    describe_file,
    eval_on_file,
    head_file,
    suggest_axes,
)

# Re-export for callers that historically imported from verbs.data
__all__ = [
    "SUBCOMMANDS",
    "EvalFailed",
    "build_parser",
    "describe_file",
    "eval_on_file",
    "head_file",
    "run",
    "suggest_axes",
]

SUBCOMMANDS = ("describe", "head", "eval", "suggest-axes")

def build_parser(prog: str = "jplot data") -> argparse.ArgumentParser:
    parser = RichArgumentParser(
        prog=prog,
        description="Inspect a data file the way Jarvis-PLOT will load it.",
        rich_title="data",
        rich_usage=(
            f"{prog} describe <file> [--json]\n"
            f"{prog} head <file> [-n N] [--json]\n"
            f"{prog} eval <expr> --data <file> [--json]\n"
            f"{prog} suggest-axes <file> [--json]"
        ),
    )
    sub = parser.add_subparsers(dest="action", required=True, parser_class=RichArgumentParser)

    def _common_source(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--type",
            dest="file_type",
            choices=["csv", "parquet", "hdf5", "auto"],
            default="auto",
            help="force a loader (default: guess from extension)",
        )
        p.add_argument(
            "--group",
            default=None,
            help="HDF5 group to open (default: file root)",
        )
        p.add_argument(
            "--json",
            action="store_true",
            help="emit one JSON envelope on stdout",
        )

    describe = sub.add_parser(
        "describe",
        help="column names, dtypes, ranges (and HDF5 tree when applicable)",
        rich_title="data describe",
        rich_usage=f"{prog} describe <file> [--type auto] [--json]",
    )
    describe.add_argument("file", help="path to csv / parquet / hdf5")
    _common_source(describe)
    describe.add_argument(
        "--no-rows",
        action="store_true",
        help="header-only: skip min/max/quantile stats (still lists columns)",
    )
    describe.add_argument(
        "--no-cache",
        action="store_true",
        help="bypass the workdir .cache/summary for this describe",
    )

    head = sub.add_parser(
        "head",
        help="first N real sample rows",
        rich_title="data head",
        rich_usage=f"{prog} head <file> [-n N] [--cols a,b] [--json]",
    )
    head.add_argument("file", help="path to csv / parquet / hdf5")
    _common_source(head)
    head.add_argument(
        "-n",
        "--n",
        dest="n_rows",
        type=int,
        default=5,
        metavar="N",
        help="number of rows (default: 5, hard cap 100)",
    )
    head.add_argument(
        "--cols",
        default=None,
        help="comma-separated column subset (default: all)",
    )

    evaluate = sub.add_parser(
        "eval",
        help="evaluate an expression against the file (sandbox before YAML)",
        rich_title="data eval",
        rich_usage=f'{prog} eval <expr> --data <file> [--sample N] [--json]',
    )
    evaluate.add_argument("expr", help='expression, e.g. "exp(LogL)"')
    evaluate.add_argument(
        "--data",
        required=True,
        dest="file",
        help="path to csv / parquet / hdf5",
    )
    _common_source(evaluate)
    evaluate.add_argument(
        "--sample",
        type=int,
        default=5,
        metavar="N",
        help="how many sample values to return (default: 5)",
    )

    axes = sub.add_parser(
        "suggest-axes",
        help="per-column scale/lim suggestions for frame.ax",
        rich_title="data suggest-axes",
        rich_usage=f"{prog} suggest-axes <file> [--cols a,b] [--json]",
    )
    axes.add_argument("file", help="path to csv / parquet / hdf5")
    _common_source(axes)
    axes.add_argument(
        "--cols",
        default=None,
        help="comma-separated column subset (default: all numeric)",
    )
    return parser


def run(argv: Sequence[str], *, prog: str = "jplot data") -> int:
    parser = build_parser(prog)
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        return system_exit_code(exc)

    action = str(getattr(args, "action", "") or "")
    as_json = bool(getattr(args, "json", False)) or not sys.stdout.isatty()

    handlers = {
        "describe": _run_describe,
        "head": _run_head,
        "eval": _run_eval,
        "suggest-axes": _run_suggest_axes,
    }
    handler = handlers.get(action)
    if handler is None:
        env = envelope(
            f"data.{action or 'unknown'}",
            False,
            error=error_payload(
                "UsageError",
                f"data action {action!r} is not implemented yet "
                f"(available: describe, head, eval, suggest-axes)",
            ),
        )
        return emit(env) if as_json else _usage_err(env)

    return handler(args, as_json=as_json, prog=prog)


# --------------------------------------------------------------------------- #
# Subcommand runners
# --------------------------------------------------------------------------- #


def _run_describe(args, *, as_json: bool, prog: str) -> int:
    try:
        data = describe_file(
            args.file,
            file_type=args.file_type,
            group=args.group,
            stats=not args.no_rows,
            use_cache=not args.no_cache,
        )
    except Exception as exc:
        return _emit_failure("data.describe", args.file, exc, as_json=as_json)

    env = envelope("data.describe", True, data=data)
    if as_json:
        return emit(env)
    _print_describe(data)
    return EXIT_OK


def _run_head(args, *, as_json: bool, prog: str) -> int:
    try:
        data = head_file(
            args.file,
            n=int(args.n_rows),
            cols=args.cols,
            file_type=args.file_type,
            group=args.group,
        )
    except Exception as exc:
        return _emit_failure("data.head", args.file, exc, as_json=as_json)

    env = envelope("data.head", True, data=data)
    if as_json:
        return emit(env)
    _print_head(data)
    return EXIT_OK


def _run_suggest_axes(args, *, as_json: bool, prog: str) -> int:
    try:
        data = suggest_axes(
            args.file,
            cols=args.cols,
            file_type=args.file_type,
            group=args.group,
        )
    except Exception as exc:
        return _emit_failure("data.suggest_axes", args.file, exc, as_json=as_json)

    env = envelope("data.suggest_axes", True, data=data)
    if as_json:
        return emit(env)
    _print_suggest_axes(data)
    return EXIT_OK


def _run_eval(args, *, as_json: bool, prog: str) -> int:
    try:
        data = eval_on_file(
            args.expr,
            args.file,
            file_type=args.file_type,
            group=args.group,
            sample=int(args.sample),
        )
    except EvalFailed as exc:
        env = envelope(
            "data.eval",
            False,
            data=exc.data,
            error=error_payload(exc.code, exc.message),
            diagnostics=[
                {
                    "code": exc.code,
                    "level": "error",
                    "path": "$.expr",
                    "message": exc.message,
                    "suggestion": exc.suggestion,
                    "context": {
                        k: v
                        for k, v in (exc.data or {}).items()
                        if k
                        in {
                            "available_columns",
                            "did_you_mean",
                            "available_functions",
                            "symbols_unresolved",
                        }
                    },
                }
            ],
        )
        if as_json:
            return emit(env)
        print(exc.message, file=sys.stderr)
        if exc.data.get("did_you_mean"):
            print(f"  did_you_mean: {exc.data['did_you_mean']}", file=sys.stderr)
        return EXIT_FAILED
    except Exception as exc:
        return _emit_failure("data.eval", getattr(args, "file", ""), exc, as_json=as_json)

    env = envelope("data.eval", True, data=data)
    if as_json:
        return emit(env)
    _print_eval(data)
    return EXIT_OK


def _emit_failure(kind: str, path: str, exc: BaseException, *, as_json: bool) -> int:
    env = envelope(kind, False, data={"path": str(path)}, error=error_payload(exc))
    if as_json:
        return emit(env)
    return _fail(env)



def _print_describe(data: dict[str, Any]) -> None:
    print(
        f"{data.get('type')}  {data.get('path')}  rows={data.get('rows')}  "
        f"cache={data.get('cache', '-')}",
        file=sys.stderr,
    )
    for col in data.get("columns") or []:
        bits = [col["name"], col.get("dtype", "")]
        if "min" in col:
            bits.append(f"[{col['min']}, {col['max']}]")
        if col.get("role_hint"):
            bits.append(f"role={col['role_hint']}")
        print("  " + "  ".join(str(b) for b in bits if b), file=sys.stderr)
    if data.get("tree"):
        print(data["tree"], file=sys.stderr)


def _print_head(data: dict[str, Any]) -> None:
    print(
        f"{data.get('type')}  {data.get('path')}  n={data.get('n')}  "
        f"columns={data.get('columns')}",
        file=sys.stderr,
    )
    for row in data.get("rows") or []:
        print(f"  {row}", file=sys.stderr)


def _print_eval(data: dict[str, Any]) -> None:
    print(
        f"expr={data.get('expr')!r}  n={data.get('n')}  "
        f"finite={data.get('n_finite')}  "
        f"range=[{data.get('min')}, {data.get('max')}]",
        file=sys.stderr,
    )
    print(f"  sample: {data.get('sample')}", file=sys.stderr)


def _print_suggest_axes(data: dict[str, Any]) -> None:
    print(f"{data.get('type')}  {data.get('path')}", file=sys.stderr)
    for axis in data.get("axes") or []:
        print(
            f"  {axis['col']}: scale={axis.get('scale')}  lim={axis.get('lim')}  "
            f"({axis.get('reason')})",
            file=sys.stderr,
        )


def _usage_err(env: dict) -> int:
    print(env.get("error", {}).get("message", "usage error"), file=sys.stderr)
    return EXIT_USAGE


def _fail(env: dict) -> int:
    print(env.get("error", {}).get("message", "failed"), file=sys.stderr)
    return EXIT_FAILED

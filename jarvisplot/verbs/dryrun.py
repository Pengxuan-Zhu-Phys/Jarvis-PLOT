#!/usr/bin/env python3

"""``jplot dryrun`` -- load data and light transforms, no matplotlib figure.

Check phase only: never re-runs heavy profile/density/interp. Execute with
``jplot <yaml>``. Emits a row ledger and light ``JP-VIZ-*`` findings.
"""

from __future__ import annotations

import argparse

from ..cli_help import RichArgumentParser
import sys
from typing import Sequence

from ..agent_io import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, envelope, system_exit_code
from ..dryrun_runtime import dryrun_file

__all__ = ["build_parser", "run"]


def build_parser(prog: str = "jplot dryrun") -> argparse.ArgumentParser:
    parser = RichArgumentParser(
        prog=prog,
        description=(
            "Load datasets + light layer transforms without rendering; "
            "emit a row ledger and light JP-VIZ diagnostics. "
            "Does not re-run heavy profile/density/interp."
        ),
        rich_title="dryrun",
        rich_usage=f"{prog} <file> [--json] [--with-data]",
    )
    parser.add_argument("file", help="path to a YAML plotting configuration")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON envelope on stdout",
    )
    parser.add_argument(
        "--with-data",
        action="store_true",
        help="write per-layer parquet twins under .cache/agent_twins/ (or --out-dir)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory for --with-data twins (default: <yaml-dir>/.cache/agent_twins)",
    )
    return parser


def run(argv: Sequence[str], *, prog: str = "jplot dryrun") -> int:
    parser = build_parser(prog)
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        return system_exit_code(exc)

    as_json = bool(args.json) or not sys.stdout.isatty()
    report, bag = dryrun_file(
        args.file,
        with_data=bool(args.with_data),
        out_dir=args.out_dir,
    )
    # Prefer the tri-state verdict from dryrun_config (None = partial coverage).
    verdict = report.get("ok")
    if "ok" not in report:
        verdict = bag.ok
    coverage = report.get("coverage") or ("full" if bag.ok else "failed")
    status = report.get("status") or (
        "ok"
        if verdict is True
        else ("partial_renderable" if verdict is None else "failed")
    )
    data = {
        "file": report.get("file", args.file),
        "status": status,
        "coverage": coverage,
        "renderable": report.get("renderable", status in {"ok", "partial_renderable"}),
        "status_note": report.get("status_note"),
        "type_expanded": report.get("type_expanded") or [],
        "heavy_skipped": report.get("heavy_skipped") or [],
        "datasets": report.get("datasets") or {},
        "layers": report.get("layers") or [],
        "twins": report.get("twins") or {},
        "error_count": len(bag.errors),
        "warning_count": len(bag.warnings),
    }
    env = envelope("dryrun", verdict, data=data, diagnostics=bag)
    if as_json:
        return emit(env)

    _print_human(args.file, report, bag)
    # partial (ok is null) exits 0; only real errors fail
    return EXIT_OK if verdict is not False else EXIT_FAILED


def _print_human(path: str, report: dict, bag) -> None:
    status = report.get("status") or "?"
    coverage = report.get("coverage") or "?"
    print(f"{path}: dryrun  status={status}  coverage={coverage}", file=sys.stderr)
    for name, meta in (report.get("datasets") or {}).items():
        rows = meta.get("rows")
        print(f"  DataSet {name:<16} {rows if rows is not None else '?':>10} rows", file=sys.stderr)
    for layer in report.get("layers") or []:
        fig = layer.get("figure")
        lname = layer.get("layer")
        print(
            f"  Figure {fig} / layer {lname}  n_points={layer.get('n_points')}  "
            f"method={layer.get('method')}",
            file=sys.stderr,
        )
        for step in layer.get("steps") or []:
            mark = "  ⚠" if step.get("rows_in", 0) > 0 and step.get("rows_out", 0) == 0 else ""
            print(
                f"    → {step.get('name'):<16} "
                f"{step.get('rows_in')} → {step.get('rows_out')}{mark}"
                f"  {step.get('detail') or ''}",
                file=sys.stderr,
            )
    if bag:
        print("", file=sys.stderr)
        print(bag.render_human(), file=sys.stderr)

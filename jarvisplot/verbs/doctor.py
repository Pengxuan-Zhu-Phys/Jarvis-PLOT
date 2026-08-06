#!/usr/bin/env python3

"""``jplot doctor`` -- validate + dryrun in one agent round-trip."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from ..agent_io import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, envelope
from ..dryrun_runtime import dryrun_file
from ..validation import validate_file

__all__ = ["build_parser", "run"]


def build_parser(prog: str = "jplot doctor") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Validate the YAML and dryrun transforms/health in one pass.",
    )
    parser.add_argument("file", help="path to a YAML plotting configuration")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON envelope on stdout",
    )
    parser.add_argument(
        "--no-columns",
        dest="check_columns",
        action="store_false",
        help="skip column-existence check during validate",
    )
    return parser


def run(argv: Sequence[str], *, prog: str = "jplot doctor") -> int:
    parser = build_parser(prog)
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        return int(exc.code or EXIT_USAGE)

    as_json = bool(args.json) or not sys.stdout.isatty()
    _config, vbag = validate_file(args.file, check_columns=args.check_columns)
    report, dbag = dryrun_file(args.file)

    # merge bags: validate first, then dryrun health
    from ..diagnostics import DiagnosticBag

    merged = DiagnosticBag()
    for d in vbag:
        merged.add(d)
    for d in dbag:
        merged.add(d)

    data = {
        "file": args.file,
        "validate": {
            "error_count": len(vbag.errors),
            "warning_count": len(vbag.warnings),
            "ok": vbag.ok,
        },
        "dryrun": {
            "error_count": len(dbag.errors),
            "warning_count": len(dbag.warnings),
            "ok": dbag.ok,
            "datasets": report.get("datasets") or {},
            "layers": report.get("layers") or [],
        },
        "error_count": len(merged.errors),
        "warning_count": len(merged.warnings),
    }
    env = envelope("doctor", merged.ok, data=data, diagnostics=merged)
    if as_json:
        return emit(env)

    print(f"{args.file}: doctor", file=sys.stderr)
    print(
        f"  validate: {len(vbag.errors)} error(s), {len(vbag.warnings)} warning(s)",
        file=sys.stderr,
    )
    print(
        f"  dryrun:   {len(dbag.errors)} error(s), {len(dbag.warnings)} warning(s)",
        file=sys.stderr,
    )
    if merged:
        print("", file=sys.stderr)
        print(merged.render_human(), file=sys.stderr)
    elif merged.ok:
        print("  OK", file=sys.stderr)
    return EXIT_OK if merged.ok else EXIT_FAILED

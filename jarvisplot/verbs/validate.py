#!/usr/bin/env python3

"""``jplot validate`` -- check a config without rendering it.

Deliberately importing nothing heavier than PyYAML: the value of this verb is
that it answers in one round, before any renderer exists.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from ..agent_io import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, envelope
from ..validation import validate_file

__all__ = ["build_parser", "run"]


def build_parser(prog: str = "jplot validate") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Validate a Jarvis-PLOT YAML configuration without rendering.",
    )
    parser.add_argument("file", help="path to a YAML plotting configuration")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON envelope on stdout instead of a human report",
    )
    parser.add_argument(
        "--no-columns",
        dest="check_columns",
        action="store_false",
        help="skip the column-existence check (pure shape verdict, reads no data file)",
    )
    return parser


def run(argv: Sequence[str], *, prog: str = "jplot validate") -> int:
    parser = build_parser(prog)
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        return int(exc.code or EXIT_USAGE)

    config, bag = validate_file(args.file, check_columns=args.check_columns)

    data = {
        "file": args.file,
        "parsed": config is not None,
        "columns_checked": args.check_columns,
        "error_count": len(bag.errors),
        "warning_count": len(bag.warnings),
    }
    env = envelope("validate", bag.ok, data=data, diagnostics=bag)

    if args.json:
        return emit(env)

    _print_human(args.file, bag)
    return EXIT_OK if bag.ok else EXIT_FAILED


def _print_human(path: str, bag) -> None:
    """Human report on stderr; stdout stays reserved for machine output."""
    if bag.ok and not len(bag):
        print(f"{path}: OK", file=sys.stderr)
        return

    rows = bag.summary_rows()
    width_code = max((len(code) for code, _, _ in rows), default=4)
    width_path = max((len(loc) for _, loc, _ in rows), default=4)
    print(f"{path}: {len(bag.errors)} error(s), {len(bag.warnings)} warning(s)\n", file=sys.stderr)
    for code, loc, message in rows:
        print(f"  {code:<{width_code}}  {loc:<{width_path}}  {message}", file=sys.stderr)
    print("", file=sys.stderr)
    print(bag.render_human(), file=sys.stderr)

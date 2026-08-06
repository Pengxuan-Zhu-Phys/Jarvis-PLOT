#!/usr/bin/env python3

"""``jplot validate`` -- check a config without rendering it.

Deliberately importing nothing heavier than PyYAML: the value of this verb is
that it answers in one round, before any renderer exists.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from ..agent_io import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, envelope
from ..fix_apply import apply_fixes, planned_fixes
from ..validation import validate_config, validate_file

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
    parser.add_argument(
        "--fix",
        action="store_true",
        help="apply mechanical Fix ops (certain confidence by default)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="with --fix: write the repaired YAML back to the input path",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        default=None,
        help="with --fix: print a unified diff (default when not --write)",
    )
    parser.add_argument(
        "--fix-unsafe",
        action="store_true",
        help="with --fix: also apply heuristic confidence fixes",
    )
    return parser


def run(argv: Sequence[str], *, prog: str = "jplot validate") -> int:
    parser = build_parser(prog)
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        return int(exc.code or EXIT_USAGE)

    if (args.write or args.diff is True or args.fix_unsafe) and not args.fix:
        print(f"{prog}: --write / --diff / --fix-unsafe require --fix", file=sys.stderr)
        return EXIT_USAGE

    config, bag = validate_file(args.file, check_columns=args.check_columns)

    data: dict[str, Any] = {
        "file": args.file,
        "parsed": config is not None,
        "columns_checked": args.check_columns,
        "error_count": len(bag.errors),
        "warning_count": len(bag.warnings),
    }

    if args.fix:
        result = _run_fix(
            path=args.file,
            config=config,
            bag=bag,
            write=bool(args.write),
            # default to diff when not writing
            show_diff=bool(args.diff) if args.diff is not None else not bool(args.write),
            include_heuristic=bool(args.fix_unsafe),
            check_columns=bool(args.check_columns),
        )
        data.update(result["data"])
        bag = result["bag"]
        data["error_count"] = len(bag.errors)
        data["warning_count"] = len(bag.warnings)

    env = envelope("validate", bag.ok, data=data, diagnostics=bag)

    if args.json:
        return emit(env)

    _print_human(args.file, bag)
    if args.fix and data.get("diff"):
        print(data["diff"], file=sys.stderr)
    if args.fix:
        applied = data.get("fixes_applied") or []
        failed = [f for f in applied if f.get("error")]
        ok_n = len(applied) - len(failed)
        mode = "wrote" if data.get("wrote") else "planned"
        print(
            f"{args.file}: --fix {mode} {ok_n} fix(es)"
            + (f", {len(failed)} failed" if failed else ""),
            file=sys.stderr,
        )
    return EXIT_OK if bag.ok else EXIT_FAILED


def _run_fix(
    *,
    path: str,
    config: dict[str, Any] | None,
    bag,
    write: bool,
    show_diff: bool,
    include_heuristic: bool,
    check_columns: bool,
) -> dict[str, Any]:
    if config is None:
        return {
            "data": {
                "fix": True,
                "fixes_planned": 0,
                "fixes_applied": [],
                "wrote": False,
                "diff": None,
            },
            "bag": bag,
        }

    fixes = planned_fixes(bag, include_heuristic=include_heuristic)
    original_text = Path(path).read_text(encoding="utf-8")
    repaired, applied = apply_fixes(config, fixes)
    new_text = _dump_yaml(repaired)
    diff_text = None
    if show_diff or not write:
        diff_text = "".join(
            difflib.unified_diff(
                original_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=path,
                tofile=path + " (fixed)",
            )
        )

    wrote = False
    if write and any(not f.get("error") for f in applied):
        Path(path).write_text(new_text, encoding="utf-8")
        wrote = True
        # re-validate the on-disk result
        _, bag = validate_file(path, check_columns=check_columns)
    elif applied:
        # re-validate the in-memory repair so the envelope reflects post-fix state
        bag = validate_config(
            repaired,
            base_dir=str(Path(path).resolve().parent),
            check_columns=check_columns,
        )

    return {
        "data": {
            "fix": True,
            "fixes_planned": len(fixes),
            "fixes_applied": applied,
            "wrote": wrote,
            "diff": diff_text or None,
            "comments_preserved": False,
        },
        "bag": bag,
    }


def _dump_yaml(config: Any) -> str:
    return yaml.safe_dump(
        config,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def _print_human(path: str, bag) -> None:
    """Human report on stderr; stdout stays reserved for machine output."""
    if bag.ok and not len(bag):
        print(f"{path}: OK", file=sys.stderr)
        return

    rows = bag.summary_rows()
    width_code = max((len(code) for code, _, _ in rows), default=4)
    width_path = max((len(loc) for _, loc, _ in rows), default=4)
    print(
        f"{path}: {len(bag.errors)} error(s), {len(bag.warnings)} warning(s)\n",
        file=sys.stderr,
    )
    for code, loc, message in rows:
        print(f"  {code:<{width_code}}  {loc:<{width_path}}  {message}", file=sys.stderr)
    print("", file=sys.stderr)
    print(bag.render_human(), file=sys.stderr)

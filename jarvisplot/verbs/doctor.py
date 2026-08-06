#!/usr/bin/env python3

"""``jplot doctor`` -- validate + dryrun in one agent round-trip."""

from __future__ import annotations

import argparse

from ..cli_help import RichArgumentParser
import sys
from typing import Sequence

from ..agent_io import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, envelope, system_exit_code
from ..dryrun_runtime import dryrun_file
from ..validation import validate_file

__all__ = ["build_parser", "run"]


def build_parser(prog: str = "jplot doctor") -> argparse.ArgumentParser:
    parser = RichArgumentParser(
        prog=prog,
        description="Validate the YAML and dryrun transforms/health in one pass.",
        rich_title='doctor',
        rich_usage=f"{prog} <file> [--json]",
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
        return system_exit_code(exc)

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

    dryrun_verdict = report.get("ok")
    coverage = str(report.get("coverage") or ("full" if dbag.ok else "failed"))
    dryrun_status = str(report.get("status") or "")
    # validate hard-fails; dryrun may be True / False / None (partial_renderable).
    if not vbag.ok or dryrun_verdict is False or not dbag.ok:
        overall: bool | None = False
        status = "failed"
        renderable = False
    elif dryrun_verdict is None or coverage == "partial" or dryrun_status == "partial_renderable":
        overall = None
        # Prefer explicit partial_renderable so agents do not treat this as failure.
        status = "partial_renderable"
        coverage = "partial"
        renderable = True
    else:
        overall = True
        status = "ok"
        coverage = "full"
        renderable = True

    from pathlib import Path

    from ..agent_digest import plan_agent_exports

    base_dir = str(Path(args.file).expanduser().resolve().parent)
    # Use validated config when available; else re-load is fine for planning.
    cfg = _config if isinstance(_config, dict) else {}
    if not cfg:
        try:
            import yaml

            cfg = yaml.safe_load(Path(args.file).read_text(encoding="utf-8")) or {}
        except Exception:
            cfg = {}
    exports = plan_agent_exports(cfg, base_dir=base_dir, bag=merged)

    # Invalid exports fail the overall doctor only when they are hard errors.
    if any(ex.get("status") == "invalid" for ex in exports):
        overall = False
        status = "failed"

    data = {
        "file": args.file,
        "status": status,
        "coverage": coverage,
        "renderable": renderable,
        "status_note": (
            "heavy transforms skipped in dryrun; config is expected to render"
            if status == "partial_renderable"
            else None
        ),
        "type_expanded": report.get("type_expanded") or [],
        "heavy_skipped": report.get("heavy_skipped") or [],
        "exports": exports,
        "validate": {
            "error_count": len(vbag.errors),
            "warning_count": len(vbag.warnings),
            "ok": vbag.ok,
        },
        "dryrun": {
            "error_count": len(dbag.errors),
            "warning_count": len(dbag.warnings),
            "ok": dryrun_verdict if dryrun_verdict is not None else None,
            "coverage": coverage,
            "status": report.get("status"),
            "renderable": report.get("renderable"),
            "datasets": report.get("datasets") or {},
            "layers": report.get("layers") or [],
        },
        "error_count": len(merged.errors),
        "warning_count": len(merged.warnings),
    }
    env = envelope("doctor", overall, data=data, diagnostics=merged)
    if as_json:
        return emit(env)

    print(
        f"{args.file}: doctor  status={status}  coverage={coverage}  "
        f"renderable={renderable}",
        file=sys.stderr,
    )
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
    elif overall is True:
        print("  OK", file=sys.stderr)
    elif overall is None:
        print(
            "  PARTIAL_RENDERABLE (structure ok; heavy transforms not simulated — safe to jplot <file>)",
            file=sys.stderr,
        )
    if exports:
        print("  exports:", file=sys.stderr)
        for ex in exports:
            print(
                f"    {ex.get('figure')}: {ex.get('status')} "
                f"{ex.get('method')} max_cells={ex.get('max_cells')} -> {ex.get('path')}",
                file=sys.stderr,
            )
    return EXIT_OK if overall is not False else EXIT_FAILED

#!/usr/bin/env python3

"""``jplot cap …`` -- the closed string vocabulary, as data.

Agents must not invent method names, style tokens, cmaps, or expression
functions. Every list here is derived from the same registries the runtime
consults (see :mod:`jarvisplot.capabilities`).
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from ..agent_io import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, envelope, error_payload
from ..capabilities import CAPABILITY_SECTIONS, capabilities, section

__all__ = ["SECTIONS", "build_parser", "run"]

SECTIONS = CAPABILITY_SECTIONS  # methods … cli


def build_parser(prog: str = "jplot cap") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "List every string Jarvis-PLOT will accept "
            "(methods, transforms, types, styles, cmaps, funcs, cli)."
        ),
    )
    parser.add_argument(
        "section",
        nargs="?",
        default="all",
        help=(
            "which catalogue to print: all | "
            + " | ".join(SECTIONS)
            + "  (default: all)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON envelope on stdout (default when stdout is not a TTY)",
    )
    return parser


def run(argv: Sequence[str], *, prog: str = "jplot cap") -> int:
    parser = build_parser(prog)
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        return int(exc.code or EXIT_USAGE)

    name = str(args.section or "all").strip().lower()
    as_json = bool(args.json) or not sys.stdout.isatty()

    try:
        if name in {"all", "*"}:
            data = capabilities()
            kind = "cap.all"
        elif name in SECTIONS:
            data = {name: section(name)}
            if name != "cli":
                # single-section payloads stay under a stable key; digest is
                # only meaningful for the full catalogue.
                pass
            kind = f"cap.{name}"
        else:
            env = envelope(
                "cap",
                False,
                data={"section": name, "available": ["all", *SECTIONS]},
                error=error_payload(
                    "UsageError",
                    f"unknown cap section {name!r}; "
                    f"choose all | {' | '.join(SECTIONS)}",
                ),
            )
            if as_json:
                return emit(env)
            print(env["error"]["message"], file=sys.stderr)
            return EXIT_USAGE
    except Exception as exc:
        env = envelope("cap", False, error=exc)
        if as_json:
            return emit(env)
        print(f"{prog}: {exc}", file=sys.stderr)
        return EXIT_FAILED

    env = envelope(kind, True, data=data)
    if as_json:
        return emit(env)

    _print_human(name, data)
    return EXIT_OK


def _print_human(name: str, data: dict) -> None:
    """Compact human listing on stderr; stdout reserved for --json."""
    if name in {"all", "*"}:
        digest = data.get("digest", "")
        print(f"cap all  digest={digest}", file=sys.stderr)
        for key in SECTIONS:
            payload = data.get(key)
            count = _count(payload)
            print(f"  {key:<12} {count}", file=sys.stderr)
        return

    key = name
    payload = data.get(key, data)
    if key == "methods":
        for entry in payload:
            axes = ",".join(entry.get("axes_types") or []) or "-"
            print(f"  {entry['name']:<20} mpl={entry.get('mpl_method', '')}  axes={axes}", file=sys.stderr)
    elif key == "transforms":
        for entry in payload:
            print(f"  {entry['name']:<20} {entry.get('description', '')[:60]}", file=sys.stderr)
    elif key == "types":
        for entry in payload:
            print(f"  {entry['name']}", file=sys.stderr)
    elif key == "styles":
        for entry in payload:
            usable = "ok" if entry.get("usable", True) else "BROKEN"
            axes = ",".join(entry.get("axes") or []) or "-"
            print(
                f"  {entry['bundle']}/{entry['token']:<16} axes=[{axes}]  {usable}",
                file=sys.stderr,
            )
            if not entry.get("usable", True) and entry.get("error"):
                print(f"      {entry['error']}", file=sys.stderr)
    elif key == "cmaps":
        jarvis = payload.get("jarvis") or []
        print(f"  jarvis ({len(jarvis)}): {', '.join(jarvis)}", file=sys.stderr)
        print(f"  (+ every matplotlib colormap; each jarvis name has _r)", file=sys.stderr)
    elif key == "funcs":
        names = payload.get("names") or []
        print(f"  {len(names)} callables", file=sys.stderr)
        print(f"  sample: {', '.join(names[:20])}…", file=sys.stderr)
    elif key == "cli":
        import json

        print(json.dumps(payload, indent=2, ensure_ascii=False), file=sys.stderr)
    else:
        import json

        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str), file=sys.stderr)


def _count(payload) -> str:
    if isinstance(payload, list):
        return str(len(payload))
    if isinstance(payload, dict):
        if "jarvis" in payload:
            return f"{len(payload.get('jarvis') or [])} jarvis (+mpl)"
        if "names" in payload:
            return f"{len(payload.get('names') or [])} names"
        if "commands" in payload:
            return f"{len(payload.get('commands') or [])} commands"
        return f"{len(payload)} keys"
    return "-"

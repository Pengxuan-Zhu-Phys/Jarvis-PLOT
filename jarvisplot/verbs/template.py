#!/usr/bin/env python3

"""``jplot template list|show`` -- type-first YAML templates (F1)."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from ..agent_io import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, envelope, error_payload
from ..templates_catalog import get_template, list_templates, render_template_yaml

__all__ = ["build_parser", "run"]


def build_parser(prog: str = "jplot template") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="List or emit type-first YAML templates with slot schemas.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    list_p = sub.add_parser("list", help="catalog of template kinds")
    list_p.add_argument("--json", action="store_true")

    show_p = sub.add_parser("show", help="emit one template YAML + slots")
    show_p.add_argument("kind", help="template kind (e.g. posterior_2d)")
    show_p.add_argument("--json", action="store_true")
    show_p.add_argument(
        "--yaml-only",
        action="store_true",
        help="print only the YAML body on stdout (human path)",
    )
    return parser


def run(argv: Sequence[str], *, prog: str = "jplot template") -> int:
    parser = build_parser(prog)
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        return int(exc.code or EXIT_USAGE)

    as_json = bool(getattr(args, "json", False)) or not sys.stdout.isatty()
    action = args.action

    if action == "list":
        catalog = list_templates()
        env = envelope("template.list", True, data={"templates": catalog})
        if as_json:
            return emit(env)
        for item in catalog:
            print(
                f"  {item['kind']:<16} {item['family']:<8} {item['title']}",
                file=sys.stderr,
            )
        return EXIT_OK

    if action == "show":
        try:
            spec = get_template(args.kind)
            yaml_text = render_template_yaml(args.kind)
        except KeyError as exc:
            env = envelope(
                "template.show",
                False,
                data={"available": [t["kind"] for t in list_templates()]},
                error=error_payload("UsageError", str(exc)),
            )
            return emit(env) if as_json else _usage(env)

        data = {
            "kind": spec["kind"],
            "title": spec["title"],
            "family": spec["family"],
            "requires": spec["requires"],
            "description": spec["description"],
            "slots": spec["slots"],
            "yaml_text": yaml_text,
        }
        env = envelope("template.show", True, data=data)
        if as_json:
            return emit(env)
        if args.yaml_only:
            print(yaml_text, end="")
            return EXIT_OK
        print(f"# {spec['title']}", file=sys.stderr)
        print(f"# slots: {', '.join(s['name'] for s in spec['slots'])}", file=sys.stderr)
        print(yaml_text, end="")
        return EXIT_OK

    return EXIT_USAGE


def _usage(env: dict) -> int:
    print(env.get("error", {}).get("message", "usage error"), file=sys.stderr)
    return EXIT_USAGE

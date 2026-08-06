#!/usr/bin/env python3

"""``jplot man [topic] [--json]`` — callable manual (human Rich / agent JSON)."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from ..agent_io import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, envelope, system_exit_code, error_payload
from ..cli import render_man_help
from ..cli_help import RichArgumentParser
from ..man_catalog import ManCatalogError, resolve_topic
from ..man_render_agent import agent_index_data, agent_topic_data
from ..man_render_human import render_index, render_topic

__all__ = ["build_parser", "run"]


def build_parser(prog: str = "jplot man") -> argparse.ArgumentParser:
    parser = RichArgumentParser(
        prog=prog,
        description=(
            "Show the Jarvis-PLOT manual (index, topic, method, or transform). "
            "Methods: man methods | man scatter. "
            "Transforms: man transforms | man transform.profile | man filter."
        ),
        rich_title="man",
        rich_usage=(
            f"{prog}\n"
            f"{prog} <topic>\n"
            f"{prog} methods | transforms\n"
            f"{prog} <method>                 # e.g. scatter\n"
            f"{prog} transform.<name>         # e.g. transform.profile\n"
            f"{prog} filter | profile | …     # bare transform names\n"
            f"{prog} <topic> --json\n"
            f"{prog} --json"
        ),
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default=None,
        help="topic id, methods/transforms, method name, or transform name",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit structured agent payload on stdout",
    )
    return parser


def run(argv: Sequence[str], *, prog: str = "jplot man") -> int:
    # Route bare -h before argparse so we match root help geometry even if
    # RichArgumentParser is bypassed by custom clients.
    tokens = list(argv)
    if tokens and tokens[0] in {"-h", "--help"} and len(tokens) == 1:
        print(render_man_help(prog=prog.rsplit(" ", 1)[0] if " " in prog else "jplot"), end="")
        return EXIT_OK

    parser = build_parser(prog)
    try:
        args = parser.parse_args(tokens)
    except SystemExit as exc:
        return system_exit_code(exc)

    # Manual is human Rich by default (even when piped). Agents must pass --json.
    as_json = bool(args.json)
    root_prog = prog.rsplit(" ", 1)[0] if prog.endswith(" man") else "jplot"

    try:
        topic = resolve_topic(args.topic)
    except ManCatalogError as exc:
        env = envelope(
            "man",
            False,
            data={"topic": args.topic},
            error=error_payload(exc),
        )
        if as_json:
            return emit(env)
        print(str(exc), file=sys.stderr)
        return EXIT_FAILED

    try:
        if topic is None:
            if as_json:
                data = agent_index_data()
                return emit(envelope("man", True, data=data))
            print(render_index(prog=root_prog), end="")
            return EXIT_OK

        if as_json:
            data = agent_topic_data(topic)
            return emit(envelope(f"man.{topic}", True, data=data))
        print(render_topic(topic, prog=root_prog), end="")
        return EXIT_OK
    except ManCatalogError as exc:
        env = envelope(
            "man" if topic is None else f"man.{topic}",
            False,
            data={"topic": topic},
            error=error_payload(exc),
        )
        if as_json:
            return emit(env)
        print(str(exc), file=sys.stderr)
        return EXIT_FAILED

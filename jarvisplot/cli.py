#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from importlib.metadata import version as _pkg_version
from io import StringIO
from typing import Any, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _resolve_version() -> str:
    """Resolve the installed distribution version without making imports fragile."""
    for dist_name in ("JarvisPLOT", "jarvisplot", "Jarvis-PLOT"):
        try:
            return _pkg_version(dist_name)
        except Exception:
            continue
    return "0.0.0"


JPLOT_VERSION = _resolve_version()

_HELP_PRIMARY_COLUMN_WIDTH = 24
_HELP_ALIAS_COLUMN_WIDTH = 6


def _terminal_width() -> int:
    return max(80, shutil.get_terminal_size().columns)


def help_panel(
    title: str,
    rows: Sequence[tuple[str, str, str]],
    *,
    width: int,
    primary_style: str = "bold cyan",
) -> Panel:
    """Build one fixed-column panel shared by root and subcommand help."""
    table = Table(
        show_header=False,
        expand=True,
        box=None,
        pad_edge=False,
        padding=(0, 1),
    )
    table.add_column(
        width=_HELP_PRIMARY_COLUMN_WIDTH,
        no_wrap=True,
        style=primary_style,
    )
    table.add_column(
        width=_HELP_ALIAS_COLUMN_WIDTH,
        no_wrap=True,
        style="bold cyan",
    )
    table.add_column(ratio=1, overflow="fold")
    for primary, alias, description in rows:
        table.add_row(Text(primary), Text(alias), Text(description))
    return Panel(
        table,
        title=Text(title, style="bold magenta"),
        title_align="left",
        border_style="dim",
        width=width,
    )


def _usage_panel(title: str, usage: str, *, width: int) -> Panel:
    usage_text = Text("Usage:\n", style="yellow")
    usage_text.append(usage, style="bold yellow")
    return Panel(
        usage_text,
        title=Text(title, style="bold magenta"),
        title_align="left",
        border_style="dim",
        width=width,
    )


def _render_help_page(
    *,
    title: str,
    usage: str,
    sections: Sequence[tuple[str, Sequence[tuple[str, str, str]], str]],
) -> str:
    """Render a help page with the fixed Jarvis2 CLI geometry."""
    width = _terminal_width()
    is_tty = sys.stdout.isatty()
    buffer = StringIO()
    console = Console(
        file=buffer,
        width=width,
        force_terminal=is_tty,
        color_system="standard" if is_tty else None,
        highlight=False,
    )
    console.print(_usage_panel(title, usage, width=width))
    for section_title, rows, primary_style in sections:
        console.print(
            help_panel(
                section_title,
                rows,
                width=width,
                primary_style=primary_style,
            )
        )
    return buffer.getvalue()


def _load_cli_spec() -> dict[str, Any]:
    spec_path = os.path.join(os.path.dirname(__file__), "cards", "args.json")
    with open(spec_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _option_rows(spec: dict[str, Any], *, flowchart: bool = False) -> list[tuple[str, str, str]]:
    """Convert the package CLI data into the stable two-label help rows."""
    options = spec.get("options", [])
    by_long = {
        option.get("long"): option
        for option in options
        if option.get("long")
    }

    rows: list[tuple[str, str, str]] = [
        ("--help", "-h", "show this help message and exit"),
    ]
    if not flowchart:
        rows.append(("--version", "-v", "show Jarvis-PLOT version"))

    selected = (
        ("--debug", "--out")
        if flowchart
        else tuple(option.get("long") for option in options if option.get("long"))
    )
    for long_name in selected:
        if not long_name or long_name in {"--version"}:
            continue
        option = by_long.get(long_name, {})
        alias = str(option.get("short", ""))
        description = str(option.get("help", "")).replace("$n", " ").strip()
        rows.append((str(long_name), alias, description))
    return rows


def _command_rows(spec: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Commands come from ``cards/args.json`` so the CLI stays self-describing."""
    return [
        (str(command.get("name", "")), "", str(command.get("help", "")))
        for command in spec.get("commands", [])
        if command.get("name")
    ]


def render_help(*, prog: str = "jplot") -> str:
    """Render the root ``jplot -h`` page."""
    spec = _load_cli_spec()
    sections = [
        ("Commands", _command_rows(spec), "bold cyan"),
        (
            "Inputs",
            [
                ("file", "", "YAML to render (bare path; not a verb)"),
                ("flowchart_file", "", "path to a flowchart scene JSON"),
            ],
            "bold cyan",
        ),
        ("Options", _option_rows(spec), "bold cyan"),
    ]
    return _render_help_page(
        title="Jarvis-PLOT",
        usage=(
            f"{prog} <file>\n"
            f"{prog} validate <file> [--json]\n"
            f"{prog} cap [section] [--json]\n"
            f"{prog} data describe|head|eval … [--json]\n"
            f"{prog} flowchart <flowchart_file>\n"
            f"{prog} -h\n"
            f"{prog} -v"
        ),
        sections=sections,
    )



def render_flowchart_help(*, prog: str = "jplot") -> str:
    """Render the ``jplot flowchart -h`` page."""
    spec = _load_cli_spec()
    sections = [
        (
            "Arguments",
            [("flowchart_file", "", "path to the input flowchart scene JSON")],
            "bold cyan",
        ),
        ("Options", _option_rows(spec, flowchart=True), "bold cyan"),
    ]
    return _render_help_page(
        title="flowchart",
        usage=f"{prog} flowchart <flowchart_file>",
        sections=sections,
    )


class _JarvisArgumentParser(argparse.ArgumentParser):
    """Keep argparse parsing while routing help through the Rich renderer."""

    def __init__(self, *args: Any, help_prog: str = "jplot", **kwargs: Any) -> None:
        self.help_prog = help_prog
        super().__init__(*args, **kwargs)

    def format_help(self) -> str:
        return render_help(prog=self.help_prog)

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        argv = list(sys.argv[1:] if args is None else args)
        if "-h" in argv or "--help" in argv:
            if argv and argv[0] == "flowchart":
                print(render_flowchart_help(prog=self.help_prog), end="")
            else:
                print(render_help(prog=self.help_prog), end="")
            raise SystemExit(0)
        return super().parse_args(argv, namespace)


class CLI:
    def __init__(self, *, prog: str = "jplot") -> None:
        self.pwd = os.path.abspath(os.path.dirname(__file__))
        self.args = _JarvisArgumentParser(
            prog=prog,
            help_prog=prog,
            add_help=False,
            description="Jarvis-PLOT command-line interface",
        )
        self.args.add_argument("-h", "--help", action="store_true", help=argparse.SUPPRESS)

        spec = _load_cli_spec()
        for positional in spec.get("positionals", []):
            help_text = str(positional.get("help", "")).replace("$n", "\n")
            kwargs: dict[str, Any] = {"help": help_text}
            if "nargs" in positional:
                kwargs["nargs"] = positional["nargs"]
            elif positional.get("name") == "file":
                kwargs["nargs"] = "?"
            self.args.add_argument(positional["name"], **kwargs)

        for option in spec.get("options", []):
            action = option.get("action", "store")
            kwargs: dict[str, Any] = {
                "help": option.get("help", ""),
                "dest": option.get("dest"),
            }
            if action == "version":
                kwargs.update(action="version", version=f"JarvisPLOT {JPLOT_VERSION}")
            else:
                kwargs["action"] = action
                if "metavar" in option:
                    kwargs["metavar"] = option["metavar"]
            if "default" in option:
                kwargs["default"] = option["default"]
            if option.get("type") == "int":
                kwargs["type"] = int
            elif option.get("type") == "float":
                kwargs["type"] = float
            elif "type" in option:
                kwargs["type"] = str

            flags = [flag for flag in (option.get("short"), option.get("long")) if flag]
            if flags:
                self.args.add_argument(*flags, **kwargs)

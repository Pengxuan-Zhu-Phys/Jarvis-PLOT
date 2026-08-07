#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import version as _pkg_version
from typing import Any, Sequence

from .cli_help import (
    HELP_ALIAS_COLUMN_WIDTH,
    HELP_PRIMARY_COLUMN_WIDTH,
    RichArgumentParser,
    help_panel,
    render_help_page,
    terminal_width,
)

# Re-export geometry helpers so existing tests / importers keep working.
__all__ = [
    "CLI",
    "JPLOT_VERSION",
    "RichArgumentParser",
    "help_panel",
    "render_flowchart_help",
    "render_help",
    "render_help_page",
    "terminal_width",
]


def _resolve_version() -> str:
    """Resolve the installed distribution version without making imports fragile."""
    for dist_name in ("JarvisPLOT", "jarvisplot", "Jarvis-PLOT"):
        try:
            return _pkg_version(dist_name)
        except Exception:
            continue
    return "0.0.0"


JPLOT_VERSION = _resolve_version()

# Back-compat private names used by tests that patch module attrs.
_HELP_PRIMARY_COLUMN_WIDTH = HELP_PRIMARY_COLUMN_WIDTH
_HELP_ALIAS_COLUMN_WIDTH = HELP_ALIAS_COLUMN_WIDTH


def _terminal_width() -> int:
    return terminal_width()


def _usage_panel(title: str, usage: str, *, width: int):
    from .cli_help import usage_panel

    return usage_panel(title, usage, width=width)


def _render_help_page(
    *,
    title: str,
    usage: str,
    sections: Sequence[tuple[str, Sequence[tuple[str, str, str]], str]],
) -> str:
    return render_help_page(title=title, usage=usage, sections=sections)


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


def _command_by_name(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(command.get("name", "")): command
        for command in spec.get("commands", [])
        if command.get("name")
    }


def _grouped_command_sections(
    spec: dict[str, Any],
) -> list[tuple[str, list[tuple[str, str, str]], str]]:
    """Build Jarvis2-style operation panels from ``help_groups`` + command meta.

    Order follows ``help_groups`` (Discover → Draft & edit → Judge → Render).
    Commands missing from the map still appear under the group that names them;
    unknown leftovers fall into a final "Other" panel so the CLI never hides a verb.
    """
    by_name = _command_by_name(spec)
    # Synthetic render entry: bare path is not a verb (DR-08).
    synthetic = {
        "file": {
            "name": "file",
            "help": "YAML to render (bare path; not a verb — DR-08)",
        }
    }
    claimed: set[str] = set()
    sections: list[tuple[str, list[tuple[str, str, str]], str]] = []

    groups = spec.get("help_groups")
    if not isinstance(groups, list) or not groups:
        # Fallback: single panel (old shape).
        rows = [
            (str(c.get("name", "")), "", str(c.get("help", "")))
            for c in spec.get("commands", [])
            if c.get("name")
        ]
        rows.insert(0, ("file", "", synthetic["file"]["help"]))
        return [("Commands", rows, "bold cyan")]

    for group in groups:
        if not isinstance(group, dict):
            continue
        title = str(group.get("title") or "Commands").strip() or "Commands"
        names = group.get("commands") or []
        rows: list[tuple[str, str, str]] = []
        for raw in names:
            name = str(raw)
            if name in claimed:
                continue
            meta = by_name.get(name) or synthetic.get(name)
            if meta is None:
                continue
            rows.append((name, "", str(meta.get("help", ""))))
            claimed.add(name)
        if rows:
            sections.append((title, rows, "bold cyan"))

    leftovers = [
        (str(c.get("name", "")), "", str(c.get("help", "")))
        for c in spec.get("commands", [])
        if c.get("name") and str(c.get("name")) not in claimed
    ]
    if leftovers:
        sections.append(("Other", leftovers, "bold cyan"))
    return sections


def render_help(*, prog: str = "jplot") -> str:
    """Render the root ``jplot -h`` page (Jarvis2-style operation groups)."""
    spec = _load_cli_spec()
    sections = _grouped_command_sections(spec)
    sections.append(("Options", _option_rows(spec), "bold cyan"))
    return _render_help_page(
        title="Jarvis-PLOT",
        usage=(
            f"{prog} <file>\n"
            f"{prog} <command> [args]\n"
            f"{prog} -h\n"
            f"{prog} -v\n"
            f"\n"
            f"Command help: {prog} COMMAND -h\n"
            f"Write YAML yourself; CLI discovers and judges (see `{prog} man`)."
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
        usage=(
            f"{prog} flowchart <flowchart_file>\n"
            f"\n"
            f"Jarvis-HEP project-scan flowchart only "
            f"(scene JSON from HEP scan tooling — not plot YAML)."
        ),
        sections=sections,
    )


def render_man_help(*, prog: str = "jplot") -> str:
    """Render ``jplot man -h`` (same geometry as root help)."""
    sections = [
        (
            "Arguments",
            [
                ("topic", "", "manual topic id (omit for index); see jplot man"),
            ],
            "bold cyan",
        ),
        (
            "Options",
            [
                ("--help", "-h", "show this help message and exit"),
                ("--json", "", "emit structured agent payload on stdout"),
            ],
            "bold cyan",
        ),
    ]
    return _render_help_page(
        title="man",
        usage=(
            f"{prog} man\n"
            f"{prog} man <topic>\n"
            f"{prog} man <topic> --json\n"
            f"{prog} man --json"
        ),
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

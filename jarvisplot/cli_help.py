#!/usr/bin/env python3

"""Shared Jarvis CLI help geometry (HEP V2 / Portal / PLOT fixed columns).

Human default help uses Rich panels with a 24+6 fixed grid so every
``jplot … -h`` page matches the family look. Agent paths use ``--json``
elsewhere; help itself stays human-facing Rich.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from io import StringIO
from typing import Any, Callable, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

__all__ = [
    "HELP_ALIAS_COLUMN_WIDTH",
    "HELP_PRIMARY_COLUMN_WIDTH",
    "RichArgumentParser",
    "help_panel",
    "render_help_page",
    "sections_from_parser",
    "terminal_width",
    "usage_panel",
]

HELP_PRIMARY_COLUMN_WIDTH = 24
HELP_ALIAS_COLUMN_WIDTH = 6

# section = (title, rows, primary_style)
HelpSection = tuple[str, Sequence[tuple[str, str, str]], str]
HelpSections = Sequence[HelpSection]


def terminal_width() -> int:
    return max(80, shutil.get_terminal_size().columns)


def help_panel(
    title: str,
    rows: Sequence[tuple[str, str, str]],
    *,
    width: int,
    primary_style: str = "bold cyan",
) -> Panel:
    """One fixed-column panel (primary | alias | description)."""
    table = Table(
        show_header=False,
        expand=True,
        box=None,
        pad_edge=False,
        padding=(0, 1),
    )
    table.add_column(
        width=HELP_PRIMARY_COLUMN_WIDTH,
        no_wrap=True,
        style=primary_style,
    )
    table.add_column(
        width=HELP_ALIAS_COLUMN_WIDTH,
        no_wrap=True,
        style="bold cyan",
    )
    table.add_column(ratio=1, overflow="fold")
    for primary, alias, description in rows:
        table.add_row(Text(str(primary)), Text(str(alias or "")), Text(str(description or "")))
    return Panel(
        table,
        title=Text(title, style="bold magenta"),
        title_align="left",
        border_style="dim",
        width=width,
    )


def usage_panel(title: str, usage: str, *, width: int) -> Panel:
    usage_text = Text("Usage:\n", style="yellow")
    usage_text.append(usage, style="bold yellow")
    return Panel(
        usage_text,
        title=Text(title, style="bold magenta"),
        title_align="left",
        border_style="dim",
        width=width,
    )


def render_help_page(
    *,
    title: str,
    usage: str,
    sections: HelpSections,
    force_terminal: bool | None = None,
) -> str:
    """Render a help page with the fixed Jarvis CLI geometry."""
    width = terminal_width()
    is_tty = sys.stdout.isatty() if force_terminal is None else force_terminal
    buffer = StringIO()
    console = Console(
        file=buffer,
        width=width,
        force_terminal=is_tty,
        color_system="standard" if is_tty else None,
        highlight=False,
    )
    console.print(usage_panel(title, usage, width=width))
    for section_title, rows, primary_style in sections:
        if not rows:
            continue
        console.print(
            help_panel(
                section_title,
                rows,
                width=width,
                primary_style=primary_style,
            )
        )
    return buffer.getvalue()


def _option_flags(action: argparse.Action) -> tuple[str, str]:
    """Return (primary long-or-short, short alias) for an optional action."""
    longs = [o for o in action.option_strings if o.startswith("--")]
    shorts = [o for o in action.option_strings if not o.startswith("--")]
    flag_actions = (
        argparse._StoreTrueAction,
        argparse._StoreFalseAction,
        argparse._StoreConstAction,
        argparse._CountAction,
        argparse._AppendConstAction,
        argparse._HelpAction,
    )
    if longs:
        primary = longs[0]
        # Append metavar for value-taking options (--data PATH).
        if not isinstance(action, flag_actions) and action.nargs != 0:
            meta = action.metavar
            if meta is None and action.dest:
                meta = str(action.dest).upper()
            if meta:
                primary = f"{primary} {meta}"
        return primary, shorts[0] if shorts else ""
    if shorts:
        primary = shorts[0]
        if not isinstance(action, flag_actions) and action.nargs != 0:
            meta = action.metavar or (str(action.dest).upper() if action.dest else "")
            if meta:
                primary = f"{primary} {meta}"
        return primary, ""
    return action.dest or "", ""


def sections_from_parser(
    parser: argparse.ArgumentParser,
    *,
    commands_title: str = "Commands",
    arguments_title: str = "Arguments",
    options_title: str = "Options",
) -> list[HelpSection]:
    """Derive fixed-column sections from an argparse parser's actions."""
    command_rows: list[tuple[str, str, str]] = []
    argument_rows: list[tuple[str, str, str]] = []
    option_rows: list[tuple[str, str, str]] = [("--help", "-h", "show this help message and exit")]

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            # choices can be dict of name -> parser
            choices = getattr(action, "choices", None) or {}
            for name, sub in choices.items():
                help_text = ""
                # pull help from the action's choice help map when present
                if hasattr(action, "_choices_actions"):
                    for ca in action._choices_actions:
                        if ca.dest == name or getattr(ca, "metavar", None) == name:
                            help_text = ca.help or ""
                            break
                if not help_text and isinstance(sub, argparse.ArgumentParser):
                    help_text = (sub.description or "").split("\n")[0]
                command_rows.append((str(name), "", str(help_text or "")))
            continue

        if not action.option_strings:
            # positional
            if action.dest in (argparse.SUPPRESS, None):
                continue
            if isinstance(action, argparse._HelpAction):
                continue
            name = action.metavar or action.dest
            if name in (None, ""):
                continue
            argument_rows.append((str(name), "", str(action.help or "")))
            continue

        if isinstance(action, argparse._HelpAction):
            continue
        if action.help is argparse.SUPPRESS:
            continue
        primary, alias = _option_flags(action)
        if primary in {"--help", "-h"}:
            continue
        option_rows.append((primary, alias, str(action.help or "")))

    sections: list[HelpSection] = []
    if command_rows:
        sections.append((commands_title, command_rows, "bold cyan"))
    if argument_rows:
        sections.append((arguments_title, argument_rows, "bold cyan"))
    if option_rows:
        sections.append((options_title, option_rows, "bold cyan"))
    return sections


class RichArgumentParser(argparse.ArgumentParser):
    """Argparse parser that prints Jarvis fixed-column Rich help.

    Parsing stays argparse; ``-h``, usage errors, and ``format_help`` all use
    the same Rich card geometry as ``jplot -h`` / HEP V2.
    """

    def __init__(
        self,
        *args: Any,
        rich_title: str | None = None,
        rich_usage: str | None = None,
        rich_sections: HelpSections | Callable[[], HelpSections] | None = None,
        **kwargs: Any,
    ) -> None:
        self.rich_title = rich_title
        self.rich_usage = rich_usage
        self.rich_sections = rich_sections
        # We still let argparse register -h so parse_args handles it, but
        # print_help routes through Rich.
        kwargs.setdefault("add_help", True)
        super().__init__(*args, **kwargs)

    def _rich_title(self) -> str:
        return self.rich_title or self.prog or "jplot"

    def _rich_usage_text(self) -> str:
        if self.rich_usage is not None:
            return self.rich_usage
        usage = super().format_usage().strip()
        if usage.lower().startswith("usage:"):
            usage = usage[6:].strip()
        return usage

    def _rich_sections_list(self) -> list[HelpSection]:
        if callable(self.rich_sections):
            return list(self.rich_sections())
        if self.rich_sections is not None:
            return list(self.rich_sections)
        return sections_from_parser(self)

    def format_help(self) -> str:
        return render_help_page(
            title=self._rich_title(),
            usage=self._rich_usage_text(),
            sections=self._rich_sections_list(),
        )

    def format_usage(self) -> str:
        # Keep a plain one-liner for libraries that call format_usage();
        # print_usage uses the card form instead.
        return f"Usage: {self._rich_usage_text().splitlines()[0]}\n"

    def print_help(self, file: Any = None) -> None:
        text = self.format_help()
        stream = file if file is not None else sys.stdout
        stream.write(text)
        if not text.endswith("\n"):
            stream.write("\n")

    def print_usage(self, file: Any = None) -> None:
        """Usage-only card (same geometry as full help's Usage panel)."""
        width = terminal_width()
        is_tty = (file if file is not None else sys.stderr).isatty()
        buffer = StringIO()
        console = Console(
            file=buffer,
            width=width,
            force_terminal=is_tty,
            color_system="standard" if is_tty else None,
            highlight=False,
        )
        console.print(
            usage_panel(self._rich_title(), self._rich_usage_text(), width=width)
        )
        text = buffer.getvalue()
        stream = file if file is not None else sys.stderr
        stream.write(text)
        if not text.endswith("\n"):
            stream.write("\n")

    def error(self, message: str) -> None:
        """Bad argv: full Rich help card on stderr + one error line."""
        # Full card so agents/humans see options without a second -h round-trip.
        help_text = self.format_help()
        sys.stderr.write(help_text)
        if not help_text.endswith("\n"):
            sys.stderr.write("\n")
        sys.stderr.write(f"{self.prog}: error: {message}\n")
        self.exit(2)

#!/usr/bin/env python3

"""Human Rich manuals for ``jplot man`` (Jarvis fixed-column geometry)."""

from __future__ import annotations

import sys
from io import StringIO
from typing import Any, Sequence

from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from .cli_help import help_panel, terminal_width
from .man_catalog import index_payload, load_card

__all__ = ["render_index", "render_topic"]


def _console() -> Console:
    is_tty = sys.stdout.isatty()
    return Console(
        file=StringIO(),
        width=terminal_width(),
        force_terminal=is_tty,
        color_system="standard" if is_tty else None,
        highlight=False,
    )


def _panel(console: Console, *, title: str, body: Any) -> None:
    console.print(
        Panel(
            body,
            title=Text(title, style="bold magenta"),
            title_align="left",
            border_style="dim",
            width=console.width,
        )
    )


def _notes_body(items: Sequence[str]) -> Text:
    return Text("\n".join(f"• {item}" for item in items if str(item).strip()))


def render_index(*, prog: str = "jplot") -> str:
    console = _console()
    width = console.width
    overview = (
        "CLI discovers and judges; you write YAML in an editor.\n"
        "Humans: skim panels below. Agents: jplot man <topic> --json."
    )
    _panel(console, title="Jarvis-PLOT manual", body=Text(overview))

    loop_rows = [
        ("1. data describe", "", "column names from the real file"),
        ("2. cap …", "", "legal methods / styles / cmaps / funcs"),
        ("3. edit YAML", "", "type: macros first; layers when needed"),
        ("4. doctor", "", "validate + dryrun in one pass"),
        ("5. jplot <file>", "", "render only when you need the figure"),
    ]
    console.print(
        help_panel("Agent loop (write YAML yourself)", loop_rows, width=width)
    )

    idx = index_payload()
    topics = idx["topics"]
    topic_rows = [(t["id"], "", t["summary"]) for t in topics]
    console.print(help_panel("Topics", topic_rows, width=width))

    method_rows = [
        (m["name"], "", f"{prog} man {m['name']}")
        for m in (idx.get("methods") or [])
    ]
    if method_rows:
        console.print(
            help_panel(
                f"Drawing methods ({len(method_rows)}) — live from cap methods",
                method_rows,
                width=width,
            )
        )

    transform_rows = [
        (t["name"], "", f"{prog} man transform.{t['name']}")
        for t in (idx.get("transforms") or [])
    ]
    if transform_rows:
        console.print(
            help_panel(
                f"Transforms ({len(transform_rows)}) — live contracts",
                transform_rows,
                width=width,
            )
        )

    usage_rows = [
        (f"{prog} man", "", "this index"),
        (f"{prog} man <topic>", "", "human Rich manual for one topic"),
        (f"{prog} man methods", "", "all layer drawing methods"),
        (f"{prog} man scatter", "", "one method (any name from cap methods)"),
        (f"{prog} man transforms", "", "all pipeline transforms"),
        (f"{prog} man transform.profile", "", "one transform contract + examples"),
        (f"{prog} man <topic> --json", "", "structured agent payload"),
        (f"{prog} man --json", "", "index as JSON"),
    ]
    console.print(help_panel("Usage", usage_rows, width=width))
    return console.file.getvalue()


def render_topic(topic: str, *, prog: str = "jplot") -> str:
    card = load_card(topic)
    console = _console()
    width = console.width

    header = f"{card['summary']}\nrole: {card.get('role', '')}  ·  id: {card['id']}"
    _panel(console, title=str(card["title"]), body=Text(header))

    related = card.get("related_cli") or []
    if related:
        rows = []
        for item in related:
            if not isinstance(item, dict):
                continue
            argv = item.get("argv") or []
            cmd = " ".join(str(x) for x in argv) if isinstance(argv, list) else str(argv)
            rows.append((cmd or "—", "", str(item.get("why") or "")))
        if rows:
            console.print(help_panel("What to run", rows, width=width))

    human = card.get("human") or {}
    panels = human.get("panels") if isinstance(human, dict) else None
    if isinstance(panels, list):
        for block in panels:
            if not isinstance(block, dict):
                continue
            kind = str(block.get("kind") or "").lower()
            title = str(block.get("title") or kind or "Section")
            if kind == "overview":
                body = str(block.get("body") or "").rstrip()
                if body:
                    _panel(console, title=title if title != "overview" else "Overview", body=Text(body))
            elif kind == "steps":
                items = block.get("items") or []
                if items:
                    _panel(console, title=title, body=_notes_body([str(i) for i in items]))
            elif kind == "yaml":
                body = str(block.get("body") or "").rstrip()
                lexer = str(block.get("lexer") or "yaml")
                if body:
                    _panel(
                        console,
                        title=title,
                        body=Syntax(body, lexer, theme="ansi_dark", word_wrap=True),
                    )
            elif kind in {"notes", "traps", "list"}:
                items = block.get("items") or []
                if items:
                    _panel(console, title=title, body=_notes_body([str(i) for i in items]))
            elif kind == "text":
                body = str(block.get("body") or "").rstrip()
                if body:
                    _panel(console, title=title, body=Text(body))

    # Fallback if no human panels: show agent yaml example if present
    if not panels:
        agent = card.get("agent") or {}
        examples = agent.get("examples") if isinstance(agent, dict) else None
        if isinstance(examples, list) and examples:
            ex0 = examples[0]
            if isinstance(ex0, dict) and ex0.get("yaml"):
                _panel(
                    console,
                    title=str(ex0.get("title") or "Example"),
                    body=Syntax(str(ex0["yaml"]).rstrip(), "yaml", theme="ansi_dark", word_wrap=True),
                )

    see = card.get("see_also") or []
    if see:
        rows = [(str(t), "", f"{prog} man {t}") for t in see]
        console.print(help_panel("See also", rows, width=width))

    return console.file.getvalue()

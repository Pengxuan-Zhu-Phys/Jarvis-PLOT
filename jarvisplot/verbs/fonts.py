"""Discover fonts and maintain the default typography card for agents and humans."""

from __future__ import annotations

import argparse
import json
import sys
from io import StringIO
from typing import Any, Sequence

from ..agent_io import EXIT_USAGE, emit, envelope, error_payload, exit_code_for, system_exit_code
from ..cli_help import RichArgumentParser, terminal_width

__all__ = ["build_parser", "run"]

ROLES = ("title", "axis_label", "tick", "legend", "text", "annotation", "caption")


class _UsageError(ValueError):
    pass


class _FontArgumentParser(RichArgumentParser):
    """Defer errors to run() so machine invocations never print human help."""

    def error(self, message: str) -> None:
        raise _UsageError(message)


def _json_option(parser: argparse.ArgumentParser, *, root: bool = False) -> None:
    parser.add_argument(
        "--json", action="store_true", default=False if root else argparse.SUPPRESS,
        help="emit one JSON envelope (default when stdout is not a TTY)",
    )


def build_parser(prog: str = "jplot fonts") -> argparse.ArgumentParser:
    parser = _FontArgumentParser(
        prog=prog, rich_title="fonts",
        description="Discover available fonts and edit the persistent default typography card.",
        rich_usage=(
            f"{prog} [--json]\n"
            f"{prog} show [ROLE] [--recommended] [--json]\n"
            f"{prog} list [--filter TEXT] [--refresh] [--json]\n"
            f"{prog} resolve FAMILY [--weight WEIGHT] [--style STYLE] [--json]\n"
            f"{prog} check [--json]\n"
            f"{prog} set PATH VALUE [--json]\n"
            f"{prog} reset [PATH] [--json]"
        ),
    )
    _json_option(parser, root=True)
    sub = parser.add_subparsers(dest="action", parser_class=_FontArgumentParser)
    show = sub.add_parser(
        "show", help="show the effective or recommended card, optionally for one role",
        rich_title="fonts show", rich_usage=f"{prog} show [ROLE] [--recommended] [--json]",
    )
    show.add_argument("role", nargs="?", choices=ROLES, metavar="ROLE", help="title / axis_label / tick / legend / text / annotation / caption")
    show.add_argument("--recommended", action="store_true", help="read the bundled recommendation without user overrides")
    listing = sub.add_parser(
        "list", help="list fonts indexed by Matplotlib, with paths and faces",
        rich_title="fonts list", rich_usage=f"{prog} list [--filter TEXT] [--refresh] [--json]",
    )
    listing.add_argument("--filter", dest="query", metavar="TEXT", help="case-insensitive font family filter")
    listing.add_argument("--refresh", action="store_true", help="rescan fonts in this process before listing")
    resolve = sub.add_parser(
        "resolve", help="resolve a family and face; Helvetica uses similar sans fonts if missing",
        rich_title="fonts resolve", rich_usage=f"{prog} resolve FAMILY [--weight WEIGHT] [--style STYLE] [--json]",
    )
    resolve.add_argument("family", metavar="FAMILY", help="a literal font family, e.g. Helvetica or STIXGeneral")
    resolve.add_argument("--weight", default="normal", help="font weight, e.g. normal, bold, or 700")
    resolve.add_argument("--style", default="normal", choices=("normal", "italic", "oblique"), help="font face style")
    check = sub.add_parser(
        "check", help="check card fonts and report actual Helvetica substitutions",
        rich_title="fonts check", rich_usage=f"{prog} check [--json]",
    )
    setting = sub.add_parser(
        "set", help="validate and persist one default card field",
        rich_title="fonts set", rich_usage=f"{prog} set PATH VALUE [--json]",
    )
    setting.add_argument("path", metavar="PATH", help="e.g. families.sans, math.fontset, or roles.title.size_pt")
    setting.add_argument("value", metavar="VALUE", help="a JSON scalar or plain string; null inherits the style card size")
    reset = sub.add_parser(
        "reset", help="restore a field or the whole card to bundled recommendations",
        rich_title="fonts reset", rich_usage=f"{prog} reset [PATH] [--json]",
    )
    reset.add_argument("path", nargs="?", metavar="PATH", help="one editable field; omit to remove all user overrides")
    for child in (show, listing, resolve, check, setting, reset):
        _json_option(child)
    return parser


def _scalar(value: str) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(parsed, (dict, list)):
        raise _UsageError("VALUE must be a JSON scalar or plain string, not an object or array")
    return parsed


def run(argv: Sequence[str], *, prog: str = "jplot fonts") -> int:
    tokens = list(argv)
    as_json = "--json" in tokens or not sys.stdout.isatty()
    parser = build_parser(prog)
    try:
        args = parser.parse_args(tokens)
    except _UsageError as exc:
        env = envelope("fonts", False, error=error_payload("UsageError", str(exc)))
        if as_json:
            return emit(env)
        sys.stderr.write(parser.format_help())
        sys.stderr.write(f"{prog}: error: {exc}\n")
        return EXIT_USAGE
    except SystemExit as exc:
        return system_exit_code(exc)

    # Import only after argv is known to be valid; help does not load Matplotlib.
    from ..fonts import FontCardUsageError, check_fonts, font_card_payload, font_inventory, reset_font_card, resolve_font, update_font_card

    action = args.action or "show"
    kind = f"fonts.{action}"
    try:
        if action == "show":
            data = font_card_payload(role=getattr(args, "role", None), recommended=getattr(args, "recommended", False))
        elif action == "list":
            data = font_inventory(query=args.query, refresh=args.refresh)
        elif action == "resolve":
            weight = _scalar(args.weight)
            data = resolve_font(args.family, weight=weight, style=args.style)
        elif action == "check":
            data = check_fonts()
        elif action == "set":
            data = update_font_card(args.path, _scalar(args.value))
        else:
            data = reset_font_card(args.path)
        env = envelope(kind, bool(data.get("ok", True)), data=data)
    except (_UsageError, FontCardUsageError) as exc:
        env = envelope(kind, False, error=error_payload("UsageError", str(exc)))
    except (OSError, ValueError) as exc:
        env = envelope(kind, False, error=exc)
    if as_json:
        return emit(env)
    sys.stderr.write(_render_human(action, env, prog=prog))
    return exit_code_for(env)


def _size(value: Any) -> str:
    return "inherit" if value is None else f"{value:g} pt" if isinstance(value, (float, int)) else str(value)


def _family(role: dict, card: dict) -> str:
    family = role.get("family", "—")
    return str(card.get("families", {}).get(family, family))


def _render_human(action: str, env: dict, *, prog: str = "jplot fonts") -> str:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    is_tty = sys.stdout.isatty()
    console = Console(file=StringIO(), width=terminal_width(), force_terminal=is_tty,
                      color_system="standard" if is_tty else None, highlight=False)

    def panel(title: str, body: Any) -> None:
        console.print(Panel(body, title=Text(title, style="bold magenta"), title_align="left",
                            border_style="dim", width=console.width))

    if env.get("error"):
        panel(f"fonts {action} · failed", Text(f"{env['error']['type']}: {env['error']['message']}", style="red"))
        return console.file.getvalue()
    data = env["data"]
    if action in {"show", "set", "reset"}:
        card = data["card"]
        recommended = data["recommended"]
        label = "Recommended" if data.get("source") == "recommended" else "Default"
        intro = Text(f"{label} typography card\n", style="bold cyan")
        intro.append(f"Sans: {card['families']['sans']} · Serif: {card['families']['serif']}\n")
        intro.append(f"User overrides: {data['user_path']}", style="dim")
        panel(f"fonts {action}", intro)
        table = Table(expand=True, box=box.SIMPLE, padding=(0, 1), show_edge=False)
        table.add_column("Role", style="bold cyan", width=11)
        table.add_column("Default", ratio=1, overflow="fold")
        table.add_column("Size", width=8)
        table.add_column("Recommended", ratio=1, overflow="fold")
        table.add_column("Face", ratio=1, overflow="fold")
        for name, role in data["roles"].items():
            rec = recommended["roles"][name]
            table.add_row(name, _family(role, card), _size(role.get("size_pt")),
                          f"{_family(rec, recommended)}\n{_size(rec.get('recommended_size_pt'))} reference",
                          f"{role.get('weight', 'normal')} / {role.get('style', 'normal')}")
        panel("Role defaults", table)
        notes = Text(f"MathText: {card['math']['fontset']} · Recommended: {recommended['math']['fontset']}\n")
        notes.append("Size inherit keeps the selected style card's size. Reference sizes help draft a new style.\n", style="dim")
        notes.append("Priority: explicit YAML/layer → user card → recommended card → style card.\n")
        notes.append("$…$ formulas use MathText; external LaTeX has its own font configuration.", style="dim")
        notes.append("\nHelvetica substitutes the first available: Arial → Liberation Sans → Nimbus Sans → DejaVu Sans.\n")
        notes.append("Use fonts check to inspect the actual fonts selected.", style="dim")
        panel("Typography policy", notes)
        panel("Open next", Text(f"{prog} check\n{prog} set roles.title.size_pt 14\n{prog} reset roles.title.size_pt\njplot man fonts"))
    elif action == "list":
        panel("fonts list", Text(f"{data['count']} font faces · Matplotlib index\nRefreshed: {'yes' if data.get('refreshed') else 'no'}\nCache: {data.get('cache_dir', '—')}"))
        table = Table(expand=True, box=box.SIMPLE, padding=(0, 1), show_edge=False)
        table.add_column("Family", style="bold cyan", ratio=1, overflow="fold")
        table.add_column("Face", ratio=1, overflow="fold")
        table.add_column("File", ratio=2, overflow="fold")
        for font in data["fonts"]:
            table.add_row(str(font["family"]), f"{font['weight']} / {font['style']}", str(font["path"]))
        panel("Available fonts", table if data["fonts"] else Text("No fonts match this filter."))
    elif action == "resolve":
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style="bold cyan", width=15)
        table.add_column(ratio=1, overflow="fold")
        for key in ("requested_family", "family", "face", "path", "weight", "style", "fallback", "fallback_reason"):
            if key in data:
                table.add_row(key.replace("_", " "), Text(str(data[key])))
        panel("fonts resolve · substituted" if data.get("fallback") else "fonts resolve · available", table)
    else:
        panel("fonts check", Text("All default fonts are available." if data["ok"] else "Some default fonts are missing.", style="green" if data["ok"] else "red"))
        table = Table(expand=True, box=box.SIMPLE, padding=(0, 1), show_edge=False)
        table.add_column("Status", width=10)
        table.add_column("Roles", style="bold cyan", ratio=1, overflow="fold")
        table.add_column("Font / file", ratio=2, overflow="fold")
        for result in data["results"]:
            available = result["available"]
            font = str(result.get("requested_family", result.get("family", "—")))
            detail = result.get("path", "") if available else result.get("error", "Missing font")
            if result.get("fallback"):
                font += f" → {result['family']}"
                detail = f"{result.get('fallback_reason', 'Helvetica unavailable; similar sans selected')}\n{detail}"
            status = "Substitute" if result.get("fallback") else "Available" if available else "Missing"
            table.add_row(Text(status, style="yellow" if result.get("fallback") else "green" if available else "red"),
                          ", ".join(result["roles"]), Text(f"{font}\n{detail}"))
        panel("Required fonts", table)
        if not data["ok"]:
            panel("Next", Text(f"Install the requested font, or select an available family:\n{prog} list --filter STIX\n{prog} set families.sans STIXGeneral\n{prog} check"))
    return console.file.getvalue()

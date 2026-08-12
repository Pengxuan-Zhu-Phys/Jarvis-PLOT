#!/usr/bin/env python3

"""``jplot cap …`` -- the closed string vocabulary, as data.

Agents must not invent method names, style tokens, cmaps, or expression
functions. Every list here is derived from the same registries the runtime
consults (see :mod:`jarvisplot.capabilities`).
"""

from __future__ import annotations

import argparse
import sys
from io import StringIO
from typing import Any

from rich import box
from rich.box import Box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from typing import Sequence

from ..cli_help import RichArgumentParser, terminal_width

from ..agent_io import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, envelope, system_exit_code, error_payload
from ..capabilities import CAPABILITY_SECTIONS, capabilities, section

__all__ = ["SECTIONS", "build_parser", "run"]

SECTIONS = CAPABILITY_SECTIONS  # methods … cli


def build_parser(prog: str = "jplot cap") -> argparse.ArgumentParser:
    section_list = "all | " + " | ".join(SECTIONS)
    parser = RichArgumentParser(
        prog=prog,
        description=(
            "List every string Jarvis-PLOT will accept "
            "(methods, transforms, types, styles, cmaps, funcs, cli)."
        ),
        rich_title="cap",
        rich_usage=(
            f"{prog}                      # section index (human card)\n"
            f"{prog} all [--json]         # full catalogue\n"
            f"{prog} <section> [--json]   # one of: {section_list}"
        ),
    )
    parser.add_argument(
        "section",
        nargs="?",
        default=None,
        help=(
            "which catalogue to print: all | "
            + " | ".join(SECTIONS)
            + "  (omit for the section index card; all is explicit)"
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
        return system_exit_code(exc)

    # Bare `jplot cap` → human section card (like data/template), not 35KB cap.all.
    # Explicit `jplot cap all` remains the full-catalogue path.
    if args.section is None:
        if args.json:
            env = envelope(
                "cap.index",
                True,
                data={
                    "sections": ["all", *list(SECTIONS)],
                    "note": "Pass a section name (or all) for catalogue data; bare cap is the index.",
                },
            )
            return emit(env)
        _print_index_human(prog)
        return EXIT_OK

    name = str(args.section).strip().lower()
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
            from . import CLI_USAGE_HINT

            env = envelope(
                "cap",
                False,
                data={"section": name, "available": ["all", *SECTIONS]},
                error=error_payload(
                    "UsageError",
                    f"unknown cap section {name!r}. {CLI_USAGE_HINT}",
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

    _print_human(name, data, prog=prog)
    return EXIT_OK


def _print_human(name: str, data: dict, *, prog: str = "jplot cap") -> None:
    """Render one capability page as a Jarvis-style Rich card on stderr."""
    console = _human_console()
    if name in {"all", "*"}:
        _render_all(console, data, prog=prog)
    else:
        key = name
        payload = data.get(key, data)
        _render_section(console, key, payload, prog=prog)
    _write_console(console)


_NAV_EXPAND = "▸"
_NAV_LEAF = "·"
_NAV_EXPAND_STYLE = "bold cyan"
_NAV_LEAF_STYLE = "dim"
_NAV_LEGEND = (
    f"{_NAV_EXPAND} cyan = deeper jplot page (see Open next / --json); "
    f"{_NAV_LEAF} dim = leaf value"
)

_SECTION_DESCRIPTIONS = {
    "methods": "Drawing methods and their coordinate contracts.",
    "transforms": "Preprocessor names, forms, inputs, and outputs.",
    "types": "Type-first YAML macros that expand to layers.",
    "styles": "Style bundle/token cards and renderer compatibility.",
    "cmaps": "Jarvis colormaps plus the matplotlib fallback.",
    "funcs": "Expression callables and namespaces.",
    "cli": "Top-level commands, arguments, and options.",
}

_CMAP_DESCRIPTIONS = {
    "qual22": "22-color qualitative palette for categorical series.",
    "tab5": "5-color qualitative palette for short categorical series.",
    "gambit_cmap": "Continuous black → blue → cyan → yellow map.",
    "jarvis_rainbow": "Continuous purple → blue → cyan → green → orange → red map.",
    "jarvis_rainbow2": "Continuous red → yellow → green → blue map.",
    "SpectralB": "Spectral-style continuous map from red through green to blue.",
    "RdBuB": "Diverging red → white → blue map.",
    "chrisB": "Diverging black → red → white → blue → navy map.",
}

# ``box.SIMPLE_HEAVY`` intentionally leaves section rows blank. Add a heavy
# row character so the Jarvis and Matplotlib registries are visibly separated
# while remaining one table.
_CMAP_TABLE_BOX = Box(
    "\n".join(
        (
            "    ",
            "    ",
            " ━━ ",
            "    ",
            " ━━ ",
            " ━━ ",
            "    ",
            "    ",
        )
    )
)


def _human_console() -> Console:
    """Build a buffered Rich console with the same geometry as ``jplot man``."""
    is_tty = sys.stdout.isatty()
    return Console(
        file=StringIO(),
        width=terminal_width(),
        force_terminal=is_tty,
        color_system="standard" if is_tty else None,
        highlight=False,
    )


def _write_console(console: Console) -> None:
    """Keep the existing cap convention: human output is on stderr."""
    sys.stderr.write(console.file.getvalue())


def _bullet_text(items: Sequence[Any]) -> Text:
    lines = []
    for item in items:
        value = str(item).strip()
        if value:
            lines.append(value if value.startswith("•") else f"• {value}")
    return Text("\n".join(lines))


def _panel(console: Console, title: str, items: Sequence[Any], *, border_style: str = "cyan") -> None:
    console.print(
        Panel(
            _bullet_text(items),
            title=title,
            box=box.ROUNDED,
            border_style=border_style,
        )
    )


def _man_table(title: str, *, table_box: Box = box.SIMPLE_HEAVY) -> Table:
    """Create a capability table with Jarvis-HEP man geometry."""
    return Table(
        title=title,
        title_justify="center",
        title_style="bold",
        box=table_box,
        show_header=True,
        expand=True,
    )


def _nav_cell(navigable: bool) -> Text:
    return Text(
        _NAV_EXPAND if navigable else _NAV_LEAF,
        style=_NAV_EXPAND_STYLE if navigable else _NAV_LEAF_STYLE,
    )


def _name_cell(value: Any, navigable: bool) -> Text:
    return Text(str(value or "—"), style=_NAV_EXPAND_STYLE if navigable else "bold")


def _short(value: Any, limit: int = 100) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text or "—"
    return text[: max(1, limit - 1)].rstrip() + "…"


def _join(value: Any, *, empty: str = "—") -> str:
    if isinstance(value, (list, tuple)):
        values = [str(item) for item in value if str(item).strip()]
        return ", ".join(values) if values else empty
    return str(value) if value not in (None, "") else empty


def _payload_type(payload: Any) -> str:
    if isinstance(payload, list):
        return "list"
    if isinstance(payload, dict):
        return "object"
    return type(payload).__name__


def _root_prog(prog: str) -> str:
    """Return the parent CLI command for links opened from a cap page."""
    parts = str(prog).rsplit(" ", 1)
    return parts[0] if len(parts) == 2 and parts[1] == "cap" else "jplot"


def _index_rows(data: dict[str, Any]) -> list[tuple[bool, str, str, str, str]]:
    rows = [(True, "all", "object", "7 sections", "Full catalogue; use --json for every entry.")]
    for key in SECTIONS:
        payload = data.get(key)
        rows.append(
            (
                True,
                key,
                _payload_type(payload),
                _count(payload),
                _SECTION_DESCRIPTIONS[key],
            )
        )
    return rows


def _render_index(console: Console, data: dict[str, Any], *, prog: str) -> None:
    _panel(
        console,
        prog,
        [
            "Purpose: inspect every closed string vocabulary accepted by Jarvis-PLOT.",
            "Humans: use a section below. Coding agents: add --json for structured data.",
            f"Source: live runtime registries; {prog} all --json is the complete payload.",
        ],
    )

    table = _man_table(f"Sections · {prog}")
    table.add_column("", justify="center", width=1, no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Type")
    table.add_column("Count")
    table.add_column("Description", overflow="fold")
    for navigable, name, kind, count, description in _index_rows(data):
        table.add_row(
            _nav_cell(navigable),
            _name_cell(name, navigable),
            kind,
            count,
            description,
        )
    console.print(table)
    console.print(Text(_NAV_LEGEND, style="dim"))
    _panel(
        console,
        f"{_NAV_EXPAND} Open next",
        [f"{prog} {name}" for name in ("all", *SECTIONS)],
    )
    _panel(
        console,
        "See also",
        [
            f"{prog} methods --json  ·  method and coordinate contracts",
            f"{prog} styles --json   ·  usable style cards",
            f"{prog} funcs --json    ·  expression callables",
        ],
    )


def _print_index_human(prog: str) -> None:
    """Render bare ``jplot cap`` as a section index, like ``Jarvis man``."""
    data = capabilities()
    console = _human_console()
    _render_index(console, data, prog=prog)
    _write_console(console)


def _render_all(console: Console, data: dict[str, Any], *, prog: str) -> None:
    digest = str(data.get("digest") or "—")
    _panel(
        console,
        f"{prog} all",
        [
            "Complete live capability catalogue for YAML authors and agents.",
            f"Digest: {digest}",
            f"Human view is a summary; use {prog} all --json for every registry entry.",
        ],
    )
    table = _man_table(f"Capabilities · {prog} all")
    table.add_column("", justify="center", width=1, no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Type")
    table.add_column("Count")
    table.add_column("Description", overflow="fold")
    for navigable, name, kind, count, description in _index_rows(data)[1:]:
        table.add_row(
            _nav_cell(navigable),
            _name_cell(name, navigable),
            kind,
            count,
            description,
        )
    console.print(table)
    console.print(Text(_NAV_LEGEND, style="dim"))
    _panel(console, f"{_NAV_EXPAND} Open next", [f"{prog} {key}" for key in SECTIONS])
    _panel(console, "See also", [f"{prog}  · section index", f"{prog} --json  · machine-readable index"])


def _render_section(console: Console, key: str, payload: Any, *, prog: str) -> None:
    _panel(
        console,
        f"{prog} {key}",
        [
            _SECTION_DESCRIPTIONS.get(key, "Live capability payload."),
            f"Agents: {prog} {key} --json",
        ],
    )
    renderer = {
        "methods": _render_methods,
        "transforms": _render_transforms,
        "types": _render_types,
        "styles": _render_styles,
        "cmaps": _render_cmaps,
        "funcs": _render_funcs,
        "cli": _render_cli,
    }.get(key)
    if renderer is None:
        _render_mapping(console, payload, title=f"Payload · {prog} {key}")
        return
    renderer(console, payload, prog=prog)


def _render_methods(console: Console, payload: list[dict[str, Any]], *, prog: str) -> None:
    table = _man_table(f"Methods · {prog} methods")
    table.add_column("", justify="center", width=1, no_wrap=True)
    table.add_column("Name")
    table.add_column("MPL")
    table.add_column("Axes")
    table.add_column("Required", overflow="fold")
    table.add_column("Optional", overflow="fold")
    for entry in payload:
        coords = entry.get("coordinates") or {}
        name = str(entry.get("name") or "")
        table.add_row(
            _nav_cell(True),
            _name_cell(name, True),
            str(entry.get("mpl_method") or "—"),
            _join(entry.get("axes_types")),
            _join(coords.get("required")),
            _join(coords.get("optional")),
        )
    console.print(table)
    console.print(Text(_NAV_LEGEND, style="dim"))
    root = _root_prog(prog)
    _panel(console, f"{_NAV_EXPAND} Open next", [f"{root} man {entry.get('name')}" for entry in payload])


def _render_transforms(console: Console, payload: list[dict[str, Any]], *, prog: str) -> None:
    table = _man_table(f"Transforms · {prog} transforms")
    table.add_column("", justify="center", width=1, no_wrap=True)
    table.add_column("Name")
    table.add_column("Form")
    table.add_column("Input → output")
    table.add_column("Description", overflow="fold")
    for entry in payload:
        name = str(entry.get("name") or "")
        table.add_row(
            _nav_cell(bool(entry.get("man"))),
            _name_cell(name, bool(entry.get("man"))),
            str(entry.get("form") or "—"),
            f"{entry.get('input') or '—'} → {entry.get('output') or '—'}",
            _short(entry.get("description")),
        )
    console.print(table)
    console.print(Text(_NAV_LEGEND, style="dim"))
    deeper = [str(entry.get("man")) for entry in payload if entry.get("man")]
    if deeper:
        _panel(console, f"{_NAV_EXPAND} Open next", deeper)


def _render_types(console: Console, payload: list[dict[str, Any]], *, prog: str) -> None:
    table = _man_table(f"Types · {prog} types")
    table.add_column("", justify="center", width=1, no_wrap=True)
    table.add_column("Name")
    table.add_column("Expands to")
    table.add_column("Description", overflow="fold")
    for entry in payload:
        navigable = bool(entry.get("man") or entry.get("explain"))
        table.add_row(
            _nav_cell(navigable),
            _name_cell(entry.get("name"), navigable),
            str(entry.get("expands_to") or "—"),
            _short(entry.get("explain") or entry.get("description")),
        )
    console.print(table)
    console.print(Text(_NAV_LEGEND, style="dim"))
    deeper = []
    for entry in payload:
        for target in (entry.get("explain"), entry.get("man")):
            if target and str(target) not in deeper:
                deeper.append(str(target))
    if deeper:
        _panel(console, f"{_NAV_EXPAND} Open next", deeper)


def _render_styles(console: Console, payload: list[dict[str, Any]], *, prog: str) -> None:
    table = _man_table(f"Styles · {prog} styles")
    table.add_column("", justify="center", width=1, no_wrap=True)
    table.add_column("Bundle / token")
    table.add_column("Axes")
    table.add_column("Usable")
    table.add_column("Methods")
    table.add_column("Error", overflow="fold")
    for entry in payload:
        usable = bool(entry.get("usable", True))
        methods = entry.get("styled_methods") or []
        table.add_row(
            _nav_cell(False),
            _name_cell(f"{entry.get('bundle', '—')} / {entry.get('token', '—')}", False),
            _join(entry.get("axes")),
            Text("OK" if usable else "BROKEN", style="green" if usable else "bold red"),
            _join(methods),
            "—" if usable else _short(entry.get("error"), 120),
        )
    console.print(table)
    console.print(Text(_NAV_LEGEND, style="dim"))


def _render_cmaps(console: Console, payload: dict[str, Any], *, prog: str) -> None:
    jarvis = list(payload.get("jarvis") or [])
    matplotlib_entries = [
        entry for entry in (payload.get("matplotlib") or []) if isinstance(entry, dict)
    ]

    table = _man_table(f"Colormaps · {prog} cmaps", table_box=_CMAP_TABLE_BOX)
    table.add_column("", justify="center", width=1, no_wrap=True)
    table.add_column("Name")
    table.add_column("Description", overflow="fold")
    for name in jarvis:
        table.add_row(
            _nav_cell(False),
            _name_cell(name, False),
            _CMAP_DESCRIPTIONS.get(name, "Jarvis colormap registered in the local colour-map catalogue."),
        )
    if matplotlib_entries:
        # One table, with a real horizontal row between the two registries.
        table.add_section()
        for entry in matplotlib_entries:
            name = str(entry.get("name") or "—")
            cmap_type = str(entry.get("type") or "Colormap")
            samples = entry.get("N")
            reverse = str(entry.get("reverse") or "—")
            description = (
                f"Matplotlib {cmap_type}; N={samples}; reverse pair: {reverse}."
            )
            table.add_row(_nav_cell(False), _name_cell(name, False), description)
    elif payload.get("matplotlib_note"):
        table.add_section()
        table.add_row(
            _nav_cell(False),
            _name_cell("Matplotlib", False),
            str(payload["matplotlib_note"]),
        )
    console.print(table)
    console.print(
        Text(
            "· dim = Jarvis colormap; every name also provides an automatic _r reverse. "
            f"Matplotlib-defined names listed below: {len(matplotlib_entries)}.",
            style="dim",
        )
    )


def _render_funcs(console: Console, payload: dict[str, Any], *, prog: str) -> None:
    names = list(payload.get("names") or [])
    namespaces = list(payload.get("namespaces") or [])
    table = _man_table(f"Expression functions · {prog} funcs")
    table.add_column("Kind")
    table.add_column("Count")
    table.add_column("Description", overflow="fold")
    table.add_row("Public callables", str(len(names)), _join(names[:24]) + ("…" if len(names) > 24 else ""))
    table.add_row("Namespaces", str(len(namespaces)), _join(namespaces))
    table.add_row("Raw callable count", str(payload.get("names_full_count") or "—"), _short(payload.get("note"), 160))
    console.print(table)


def _render_cli(console: Console, payload: dict[str, Any], *, prog: str) -> None:
    commands = list(payload.get("commands") or [])
    if commands:
        table = _man_table(f"Commands · {prog} cli")
        table.add_column("", justify="center", width=1, no_wrap=True)
        table.add_column("Name")
        table.add_column("Kind")
        table.add_column("Group")
        table.add_column("Description", overflow="fold")
        for entry in commands:
            name = str(entry.get("name") or "")
            navigable = name != "file"  # ``file`` is the bare render positional, not a verb.
            table.add_row(
                _nav_cell(navigable),
                _name_cell(name, navigable),
                str(entry.get("kind") or "—"),
                str(entry.get("group") or "—"),
                _short(entry.get("help")),
            )
        console.print(table)
        console.print(Text(_NAV_LEGEND, style="dim"))
        root = _root_prog(prog)
        deeper = [f"{root} {entry.get('name')} -h" for entry in commands if entry.get("name") != "file"]
        if deeper:
            _panel(console, f"{_NAV_EXPAND} Open next", deeper)

    for title, key, columns in (
        ("Positionals", "positionals", ("Name", "Group", "Description")),
        ("Options", "options", ("Option", "Destination", "Description")),
    ):
        rows = list(payload.get(key) or [])
        if not rows:
            continue
        table = _man_table(f"{title} · {prog} cli")
        for column in columns:
            table.add_column(column, overflow="fold")
        for entry in rows:
            if key == "options":
                option = ", ".join(str(value) for value in (entry.get("short"), entry.get("long")) if value)
                table.add_row(option or "—", str(entry.get("dest") or "—"), _short(entry.get("help")))
            else:
                table.add_row(str(entry.get("name") or "—"), str(entry.get("group") or "—"), _short(entry.get("help")))
        console.print(table)


def _render_mapping(console: Console, payload: Any, *, title: str) -> None:
    table = _man_table(title)
    table.add_column("Key", style="bold")
    table.add_column("Value", overflow="fold")
    if isinstance(payload, dict):
        for key, value in payload.items():
            table.add_row(str(key), _short(value, 180))
    else:
        table.add_row("payload", _short(payload, 180))
    console.print(table)


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

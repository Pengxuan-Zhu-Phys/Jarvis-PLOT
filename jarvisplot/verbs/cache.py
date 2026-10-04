"""``jplot cache list|scan|clean``: maintenance without loading the renderer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import sys
from io import StringIO
from typing import Sequence

from ..agent_io import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, envelope, error_payload, system_exit_code
from ..cache_registry import CacheRegistry, CacheUsageError
from ..cli_help import RichArgumentParser, help_panel, terminal_width


def _days(value: str) -> float:
    try:
        days = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("days must be a finite nonnegative number") from None
    if not math.isfinite(days) or days < 0:
        raise argparse.ArgumentTypeError("days must be a finite nonnegative number")
    return days


def build_parser(prog: str = "jplot cache") -> argparse.ArgumentParser:
    parser = RichArgumentParser(
        prog=prog, rich_title="cache",
        description="Manage registered plot caches (~/.jarvis/plot.json; JARVIS_HOME overrides the directory).",
        rich_usage=(f"{prog} list [--json]\n{prog} scan <directory> … [--json]\n"
                    f"{prog} clean (--workdir PATH | --all) [--older-than DAYS] [--dry-run] [--json]"),
    )
    sub = parser.add_subparsers(dest="action", required=True, parser_class=RichArgumentParser)
    listing = sub.add_parser("list", help="show cache paths, sizes, last use, and active/missing status",
                             rich_title="cache list", rich_usage=f"{prog} list [--json]")
    scan = sub.add_parser("scan", help="register existing plot caches under specified directories",
                         rich_title="cache scan", rich_usage=f"{prog} scan <directory> … [--json]")
    scan.add_argument("directories", nargs="+", metavar="DIRECTORY", help="directories to search recursively")
    clean = sub.add_parser("clean", help="clean selected caches; active caches are skipped",
                          rich_title="cache clean",
                          rich_usage=f"{prog} clean (--workdir PATH | --all) [--older-than DAYS] [--dry-run] [--json]")
    targets = clean.add_mutually_exclusive_group(required=True)
    targets.add_argument("--all", action="store_true", help="select all registered plot caches")
    targets.add_argument("--workdir", action="append", metavar="PATH", help="select registered caches in this directory and all subdirectories (repeatable)")
    clean.add_argument("--older-than", type=_days, metavar="DAYS", help="select caches unused for at least this many days")
    clean.add_argument("--dry-run", action="store_true", help="preview selected caches without deleting or changing the index")
    for command in (listing, scan, clean):
        command.add_argument("--json", action="store_true", help="emit one JSON envelope (default when stdout is not a TTY)")
    return parser


def run(argv: Sequence[str], *, prog: str = "jplot cache") -> int:
    try:
        args = build_parser(prog).parse_args(list(argv))
    except SystemExit as exc:
        return system_exit_code(exc)
    as_json = args.json or not sys.stdout.isatty()
    registry = CacheRegistry()
    kind = f"cache.{args.action}"
    try:
        if args.action == "list":
            data = {"registry": str(registry.path), "caches": registry.list()}
            verdict = not any(row["status"] == "invalid" for row in data["caches"])
        elif args.action == "scan":
            data = registry.scan(args.directories)
            verdict = False if data["errors"] else (None if data["skipped"] else True)
        else:
            data = registry.clean(workdirs=args.workdir, older_than=args.older_than, dry_run=args.dry_run)
            verdict = False if data["errors"] else (None if data["skipped"] else True)
        env = envelope(kind, verdict, data=data)
    except CacheUsageError as exc:
        env = envelope(kind, False, error=error_payload("UsageError", str(exc)))
    except (OSError, ValueError) as exc:
        env = envelope(kind, False, error=exc)
    if as_json:
        return emit(env)
    if env["error"]:
        _print_human(args.action, {"error": env["error"]}, registry, prog=prog)
        return EXIT_USAGE if env["error"]["type"] == "UsageError" else EXIT_FAILED
    _print_human(args.action, env["data"], registry, prog=prog)
    return EXIT_FAILED if env["ok"] is False else EXIT_OK


def _print_human(action: str, data: dict, registry: CacheRegistry, *, prog: str) -> None:
    sys.stderr.write(_render_human(action, data, registry, prog=prog))


def _size_label(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{int(value)} B" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024


def _last_used_label(value: str | None) -> str:
    if not value:
        return "—"
    try:
        stamp = datetime.fromisoformat(value)
        if stamp.tzinfo is None:
            return value
        return stamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (TypeError, ValueError):
        return str(value)


def _render_human(action: str, data: dict, registry: CacheRegistry, *, prog: str) -> str:
    """Use Jarvis panels, with full-width records for the cache inventory."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    is_tty = sys.stdout.isatty()
    console = Console(file=StringIO(), width=terminal_width(), force_terminal=is_tty,
                      color_system="standard" if is_tty else None, highlight=False)

    def panel(title: str, body: Text | Table) -> None:
        console.print(Panel(body, title=Text(title, style="bold magenta"),
                            title_align="left", border_style="dim", width=console.width))

    def overview(body: str) -> None:
        panel(f"cache {action}", Text(body))

    def rows(title: str, entries: list[tuple[str, str, str]]) -> None:
        if entries:
            console.print(help_panel(title, entries, width=console.width))

    if data.get("error"):
        overview(f"Registry: {registry.path}\nOperation failed.")
        rows("Error", [(data["error"]["type"], "", data["error"]["message"])])
        return console.file.getvalue()

    errors = list(data.get("errors", []))
    if action == "list":
        caches = data["caches"]
        size = sum(row["bytes"] for row in caches)
        overview(f"Registry: {registry.path}\nCaches: {len(caches)} · Size: {_size_label(size)}")
        inventory = Table.grid(expand=True, padding=(0, 0))
        inventory.add_column(width=max(3, len(str(len(caches))) + 2), no_wrap=True, style="bold cyan")
        inventory.add_column(ratio=1, overflow="fold")
        status_styles = {"ready": "green", "active": "yellow", "missing": "dim", "invalid": "red"}
        for index, row in enumerate(caches, 1):
            if index > 1:
                inventory.add_row("", "")
            record = Text()
            record.append(row["cache_dir"], style="bold cyan")
            record.append("\n")
            record.append(row["status"], style=status_styles.get(row["status"], ""))
            record.append(f" · {_size_label(row['bytes'])} · ")
            record.append("Last used: ", style="dim")
            record.append(_last_used_label(row.get("last_used")))
            for config in row.get("configs", []):
                record.append("\nYAML: ", style="dim")
                record.append(config)
            inventory.add_row(Text(f"{index}."), record)
            if row.get("error"):
                errors.append({"path": row["cache_dir"], "error": row["error"]})
        if caches:
            panel("Caches", inventory)
        if not caches:
            rows("Next", [("No registered caches", "", f"Use {prog} scan <directory> to register existing caches.")])
    elif action == "scan":
        overview(f"Registry: {registry.path}\nRegistered: {len(data['registered'])} cache(s)")
        rows("Registered caches", [(str(index), "", entry["cache_dir"])
                                  for index, entry in enumerate(data["registered"], 1)])
    else:
        label = "Would clean" if data["dry_run"] else "Cleaned"
        overview(f"Registry: {registry.path}\n{label}: {len(data['removed'])} cache(s) · {data['bytes'] / 1024**2:.2f} MiB")
        rows("Selected caches" if data["dry_run"] else "Cleaned caches",
             [(str(index), "", path) for index, path in enumerate(data["removed"], 1)])
        if data["missing"]:
            rows("Missing records", [("Would drop" if data["dry_run"] else "Dropped", "", path)
                                     for path in data["missing"]])
    rows("Skipped caches", [(item["reason"], "", item["path"]) for item in data.get("skipped", [])])
    rows("Errors", [("Failed", "", f"{item['path']}\n{item['error']}") for item in errors])
    return console.file.getvalue()

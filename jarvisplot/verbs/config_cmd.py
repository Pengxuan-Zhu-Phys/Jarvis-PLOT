#!/usr/bin/env python3

"""``jplot config get`` -- named structural reads (F3).

``config set`` with comment preservation stays blocked on DR-02 (ruamel).
This verb only reads.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from ..agent_io import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, envelope, error_payload
from ..config_address import AddressError, parse_address, resolve_address

__all__ = ["build_parser", "run"]


def build_parser(prog: str = "jplot config") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Read values from a plot YAML by named address "
            "(Figures[name].layers[name].…)."
        ),
    )
    sub = parser.add_subparsers(dest="action", required=True)

    get_p = sub.add_parser(
        "get",
        help="read one value (or subtree) by address",
    )
    get_p.add_argument("file", help="path to a YAML plotting configuration")
    get_p.add_argument(
        "address",
        help="e.g. Figures[f1].layers[pts].method or DataSet[samples].path",
    )
    get_p.add_argument("--json", action="store_true")

    path_p = sub.add_parser(
        "paths",
        help="list useful named addresses for Figures / DataSet / layers",
    )
    path_p.add_argument("file", help="path to a YAML plotting configuration")
    path_p.add_argument("--json", action="store_true")
    return parser


def run(argv: Sequence[str], *, prog: str = "jplot config") -> int:
    parser = build_parser(prog)
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        return int(exc.code or EXIT_USAGE)

    as_json = bool(getattr(args, "json", False)) or not sys.stdout.isatty()
    action = args.action

    try:
        config = _load(args.file)
    except Exception as exc:
        env = envelope(
            f"config.{action}",
            False,
            data={"file": args.file},
            error=error_payload(exc),
        )
        return emit(env) if as_json else _fail(env)

    if action == "get":
        try:
            # validate address syntax early
            parse_address(args.address)
            value = resolve_address(config, args.address)
        except AddressError as exc:
            env = envelope(
                "config.get",
                False,
                data={"file": args.file, "address": args.address},
                error=error_payload("AddressError", str(exc)),
            )
            return emit(env) if as_json else _fail(env)

        data = {
            "file": str(Path(args.file).resolve()),
            "address": args.address,
            "value": value,
        }
        env = envelope("config.get", True, data=data)
        if as_json:
            return emit(env)
        _print_value(value)
        return EXIT_OK

    if action == "paths":
        paths = list_named_paths(config)
        env = envelope(
            "config.paths",
            True,
            data={"file": str(Path(args.file).resolve()), "paths": paths},
        )
        if as_json:
            return emit(env)
        for p in paths:
            print(f"  {p}", file=sys.stderr)
        return EXIT_OK

    return EXIT_USAGE


def list_named_paths(config: Any) -> list[str]:
    """Enumerate stable addresses for top-level collections."""
    out: list[str] = []
    if not isinstance(config, dict):
        return out
    for index, entry in enumerate(config.get("DataSet") or ()):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if isinstance(name, str) and name.strip():
            base = f"DataSet[{name}]"
        else:
            base = f"DataSet[{index}]"
        out.append(base)
        out.append(f"{base}.path")
        out.append(f"{base}.type")
    for index, fig in enumerate(config.get("Figures") or ()):
        if not isinstance(fig, dict):
            continue
        fname = fig.get("name")
        fbase = (
            f"Figures[{fname}]"
            if isinstance(fname, str) and fname.strip()
            else f"Figures[{index}]"
        )
        out.append(fbase)
        if "type" in fig:
            out.append(f"{fbase}.type")
        if "style" in fig:
            out.append(f"{fbase}.style")
        for li, layer in enumerate(fig.get("layers") or ()):
            if not isinstance(layer, dict):
                continue
            lname = layer.get("name")
            if isinstance(lname, str) and lname.strip():
                lbase = f"{fbase}.layers[{lname}]"
            else:
                lbase = f"{fbase}.layers[_layer{li}]"
            out.append(lbase)
            if "method" in layer:
                out.append(f"{lbase}.method")
            if "style" in layer:
                out.append(f"{lbase}.style")
    if "output" in config:
        out.append("output")
        if isinstance(config.get("output"), dict) and "dir" in config["output"]:
            out.append("output.dir")
    return out


def _load(path: str) -> Any:
    text = Path(path).expanduser().read_text(encoding="utf-8")
    return yaml.safe_load(text)


def _print_value(value: Any) -> None:
    if isinstance(value, (dict, list)):
        print(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), end="")
    else:
        print(value)


def _fail(env: dict) -> int:
    print(env.get("error", {}).get("message", "failed"), file=sys.stderr)
    return EXIT_FAILED

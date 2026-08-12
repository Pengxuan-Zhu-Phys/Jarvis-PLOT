#!/usr/bin/env python3

"""``jplot config get|paths|set|rm|expand`` -- named structural YAML access (F3/F4).

Write path (set/rm/expand):

1. load (ruamel when available → comments preserved)
2. mutate by named address (or expand ``type:`` → ``layers``)
3. validate in memory
4. ``--diff`` by default; ``--write`` only if validate ok (write-validate-rollback)

``expand`` is the intentional off-ramp from the human-friendly ``type:`` macro
to hand-editable ``layers`` (same engine as render-time expansion).
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from ..agent_io import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, envelope, system_exit_code, error_payload
from ..cli_help import RichArgumentParser
from ..config_address import (
    AddressError,
    delete_address,
    parse_address,
    resolve_address,
    set_address,
)
from ..validation import validate_config
from ..yaml_io import dump_yaml_doc, has_ruamel, load_yaml_doc

__all__ = ["build_parser", "run"]


def build_parser(prog: str = "jplot config") -> argparse.ArgumentParser:
    parser = RichArgumentParser(
        prog=prog,
        description=(
            "Read/write plot YAML by named address "
            "(Figures[name].layers[name].…). Writes validate before disk."
        ),
        rich_title="config",
        rich_usage=(
            f"{prog} get <file> <address> [--json]\n"
            f"{prog} paths <file> [--json]\n"
            f"{prog} set <file> <address> <value> [--write] [--json]\n"
            f"{prog} rm <file> <address> [--write] [--json]\n"
            f"{prog} expand <file> [--figure NAME] [--write] [--json]"
        ),
    )
    sub = parser.add_subparsers(dest="action", required=True, parser_class=RichArgumentParser)

    get_p = sub.add_parser(
        "get",
        help="read one value (or subtree) by address",
        rich_title="config get",
        rich_usage=f"{prog} get <file> <address> [--json]",
    )
    get_p.add_argument("file", help="path to a YAML plotting configuration")
    get_p.add_argument(
        "address",
        help="e.g. Figures[f1].layers[pts].method or DataSet[samples].path",
    )
    get_p.add_argument("--json", action="store_true", help="emit JSON envelope on stdout")

    path_p = sub.add_parser(
        "paths",
        help="list useful named addresses for Figures / DataSet / layers",
        rich_title="config paths",
        rich_usage=f"{prog} paths <file> [--json]",
    )
    path_p.add_argument("file", help="path to a YAML plotting configuration")
    path_p.add_argument("--json", action="store_true", help="emit JSON envelope on stdout")

    set_p = sub.add_parser(
        "set",
        help="set a value by address (validate before write; --diff default)",
        rich_title="config set",
        rich_usage=f"{prog} set <file> <address> <value> [--write] [--diff] [--json]",
    )
    set_p.add_argument("file", help="path to a YAML plotting configuration")
    set_p.add_argument("address", help="named address to set")
    set_p.add_argument(
        "value",
        help="JSON or YAML scalar/mapping/list (e.g. 1.2, viridis, '{s: 6}')",
    )
    set_p.add_argument("--json", action="store_true", help="emit JSON envelope on stdout")
    set_p.add_argument(
        "--write",
        action="store_true",
        help="write the file if validation passes (default: diff only)",
    )
    set_p.add_argument(
        "--diff",
        action="store_true",
        default=None,
        help="print unified diff (default when not --write)",
    )
    set_p.add_argument(
        "--no-columns",
        dest="check_columns",
        action="store_false",
        help="skip column check during post-edit validate",
    )

    rm_p = sub.add_parser(
        "rm",
        help="remove a key or list item by address (validate before write)",
        rich_title="config rm",
        rich_usage=f"{prog} rm <file> <address> [--write] [--json]",
    )
    rm_p.add_argument("file", help="path to a YAML plotting configuration")
    rm_p.add_argument("address", help="named address to remove")
    rm_p.add_argument("--json", action="store_true", help="emit JSON envelope on stdout")
    rm_p.add_argument("--write", action="store_true", help="write if validation passes")
    rm_p.add_argument("--diff", action="store_true", default=None, help="print unified diff")
    rm_p.add_argument(
        "--no-columns",
        dest="check_columns",
        action="store_false",
        help="skip column check during post-edit validate",
    )

    exp_p = sub.add_parser(
        "expand",
        help=(
            "convert figure.type macros to figure.layers (same expansion as render); "
            "default: unified diff; --write only after validate"
        ),
        rich_title="config expand",
        rich_usage=f"{prog} expand <file> [--figure NAME] [--write] [--diff] [--json]",
    )
    exp_p.add_argument("file", help="path to a YAML plotting configuration")
    exp_p.add_argument(
        "--figure",
        action="append",
        dest="figures",
        metavar="NAME",
        help="expand only this figure name (repeatable; default: all type: figures)",
    )
    exp_p.add_argument("--json", action="store_true", help="emit JSON envelope on stdout")
    exp_p.add_argument(
        "--write",
        action="store_true",
        help="write the file if validation passes (default: diff only)",
    )
    exp_p.add_argument(
        "--diff",
        action="store_true",
        default=None,
        help="print unified diff (default when not --write)",
    )
    exp_p.add_argument(
        "--no-columns",
        dest="check_columns",
        action="store_false",
        help="skip column check during post-expand validate",
    )
    return parser


def run(argv: Sequence[str], *, prog: str = "jplot config") -> int:
    parser = build_parser(prog)
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        return system_exit_code(exc)

    as_json = bool(getattr(args, "json", False)) or not sys.stdout.isatty()
    action = args.action

    if action == "get":
        return _run_get(args, as_json=as_json)
    if action == "paths":
        return _run_paths(args, as_json=as_json)
    if action == "set":
        return _run_mutate(
            args,
            as_json=as_json,
            kind="config.set",
            mutator=lambda doc: set_address(doc, args.address, _parse_value(args.value)),
        )
    if action == "rm":
        return _run_mutate(
            args,
            as_json=as_json,
            kind="config.rm",
            mutator=lambda doc: delete_address(doc, args.address),
        )
    if action == "expand":
        return _run_expand(args, as_json=as_json)
    return EXIT_USAGE


def _run_get(args, *, as_json: bool) -> int:
    try:
        config, _meta = load_yaml_doc(args.file)
        parse_address(args.address)
        # resolve on plain dict view
        plain = _to_plain(config)
        value = resolve_address(plain, args.address)
    except Exception as exc:
        env = envelope(
            "config.get",
            False,
            data={"file": args.file, "address": getattr(args, "address", None)},
            error=error_payload(exc),
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


def _run_paths(args, *, as_json: bool) -> int:
    try:
        config, _meta = load_yaml_doc(args.file)
        plain = _to_plain(config)
    except Exception as exc:
        env = envelope(
            "config.paths",
            False,
            data={"file": args.file},
            error=error_payload(exc),
        )
        return emit(env) if as_json else _fail(env)

    paths = list_named_paths(plain)
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


def _run_mutate(args, *, as_json: bool, kind: str, mutator) -> int:
    path = Path(args.file).expanduser()
    extra: dict[str, Any] = {"address": getattr(args, "address", None)}
    try:
        doc, meta = load_yaml_doc(path)
        before = meta.get("raw_text") or dump_yaml_doc(doc, meta=meta)
        mutator(doc)
    except Exception as exc:
        env = envelope(
            kind,
            False,
            data={"file": str(path), **{k: v for k, v in extra.items() if v is not None}},
            error=error_payload(exc),
        )
        return emit(env) if as_json else _fail(env)

    return _finish_write(
        path,
        doc=doc,
        meta=meta,
        before=before,
        as_json=as_json,
        kind=kind,
        write=bool(args.write),
        show_diff=bool(args.diff) if args.diff is not None else not bool(args.write),
        check_columns=bool(getattr(args, "check_columns", True)),
        extra=extra,
    )


def _run_expand(args, *, as_json: bool) -> int:
    """``type:`` → ``layers`` convert; same expansion engine as render.

    Idempotent: if nothing needs expanding, returns ``ok: true`` with
    ``status: unchanged`` (exit 0) so agents can re-run safely.
    """
    from ..Figure.figure_types import expand_typed_figures

    path = Path(args.file).expanduser()
    names = list(args.figures) if args.figures else None
    expanded: list[str] = []

    try:
        doc, meta = load_yaml_doc(path)
        before = meta.get("raw_text") or dump_yaml_doc(doc, meta=meta)
        expanded = expand_typed_figures(
            doc,
            figure_names=names,
            raise_on_error=True,
            allow_noop=True,
        )
    except Exception as exc:
        env = envelope(
            "config.expand",
            False,
            data={
                "file": str(path),
                "figures": names,
                "expanded": expanded,
                "status": "failed",
            },
            error=error_payload(exc),
        )
        return emit(env) if as_json else _fail(env)

    if not expanded:
        data = {
            "file": str(path.resolve()),
            "figures": names,
            "expanded": [],
            "status": "unchanged",
            "wrote": False,
            "comments_preserved": meta.get("comments_preserved", False),
            "engine": meta.get("engine"),
            "diff": None,
            "message": (
                "no type: figures to expand"
                + (f" (looked for: {', '.join(names)})" if names else " (already layers form)")
            ),
        }
        env = envelope("config.expand", True, data=data)
        if as_json:
            return emit(env)
        print(f"{path}: config expand unchanged (already layers form)", file=sys.stderr)
        return EXIT_OK

    return _finish_write(
        path,
        doc=doc,
        meta=meta,
        before=before,
        as_json=as_json,
        kind="config.expand",
        write=bool(args.write),
        show_diff=bool(args.diff) if args.diff is not None else not bool(args.write),
        check_columns=bool(getattr(args, "check_columns", True)),
        extra={"figures": names, "expanded": expanded, "status": "expanded"},
    )


def _finish_write(
    path: Path,
    *,
    doc: Any,
    meta: dict[str, Any],
    before: str,
    as_json: bool,
    kind: str,
    write: bool,
    show_diff: bool,
    check_columns: bool,
    extra: dict[str, Any] | None = None,
) -> int:
    plain = _to_plain(doc)
    bag = validate_config(
        plain,
        base_dir=str(path.parent.resolve()),
        check_columns=check_columns,
    )
    after = dump_yaml_doc(doc, meta=meta)
    diff_text = None
    if show_diff:
        diff_text = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=str(path),
                tofile=str(path) + " (edited)",
            )
        )

    payload_extra = {k: v for k, v in (extra or {}).items() if v is not None}
    wrote = False
    if not bag.ok:
        env = envelope(
            kind,
            False,
            data={
                "file": str(path.resolve()),
                "wrote": False,
                "comments_preserved": meta.get("comments_preserved", False),
                "engine": meta.get("engine"),
                "diff": diff_text,
                **payload_extra,
            },
            diagnostics=bag,
            error=error_payload(
                "ValidationError",
                "edit failed validate; file not written (write-validate-rollback)",
            ),
        )
        if as_json:
            return emit(env)
        print(env["error"]["message"], file=sys.stderr)
        if diff_text:
            print(diff_text, file=sys.stderr)
        print(bag.render_human(), file=sys.stderr)
        return EXIT_FAILED

    if write:
        path.write_text(after, encoding="utf-8")
        wrote = True

    env = envelope(
        kind,
        True,
        data={
            "file": str(path.resolve()),
            "wrote": wrote,
            "comments_preserved": meta.get("comments_preserved", False),
            "engine": meta.get("engine"),
            "diff": diff_text,
            **payload_extra,
        },
        diagnostics=bag,
    )
    if as_json:
        return emit(env)
    if diff_text:
        print(diff_text, file=sys.stderr)
    mode = "wrote" if wrote else "planned"
    action = kind.split(".")[-1]
    detail = ""
    if kind == "config.expand" and payload_extra.get("expanded"):
        detail = f" figures={payload_extra['expanded']}"
    print(
        f"{path}: config {action} {mode}{detail} "
        f"(comments_preserved={meta.get('comments_preserved')})",
        file=sys.stderr,
    )
    return EXIT_OK


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


def _parse_value(raw: str) -> Any:
    text = str(raw)
    # JSON first (numbers, true/false, objects)
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


def _to_plain(doc: Any) -> Any:
    """Convert ruamel CommentedMap/Seq to plain Python for validation/resolve."""
    if has_ruamel():
        try:
            from ruamel.yaml.comments import CommentedMap, CommentedSeq

            if isinstance(doc, CommentedMap):
                return {k: _to_plain(v) for k, v in doc.items()}
            if isinstance(doc, CommentedSeq):
                return [_to_plain(v) for v in doc]
        except Exception:
            pass
    if isinstance(doc, dict):
        return {k: _to_plain(v) for k, v in doc.items()}
    if isinstance(doc, list):
        return [_to_plain(v) for v in doc]
    return doc


def _print_value(value: Any) -> None:
    if isinstance(value, (dict, list)):
        print(dump_yaml_doc(value, meta={"engine": "pyyaml"}), end="")
    else:
        print(value)


def _fail(env: dict) -> int:
    print(env.get("error", {}).get("message", "failed"), file=sys.stderr)
    return EXIT_FAILED

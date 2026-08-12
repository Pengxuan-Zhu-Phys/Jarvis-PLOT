#!/usr/bin/env python3

"""``jplot explain`` -- JP-* codes, figure types, or type: YAML expansion."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import yaml

from ..agent_io import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_USAGE,
    emit,
    envelope,
    error_payload,
    system_exit_code,
)
from ..cli_help import RichArgumentParser
from ..diagnostic_guidance import KNOWN_CODES, guidance_for
from ..yaml_io import dump_yaml_doc

__all__ = ["build_parser", "run"]


def build_parser(prog: str = "jplot explain") -> argparse.ArgumentParser:
    parser = RichArgumentParser(
        prog=prog,
        description=(
            "Explain a JP-* code, a figure type (e.g. posterior_2d), "
            "or expand a type: YAML file/snippet to layers."
        ),
        rich_title="explain",
        rich_usage=(
            f"{prog} <JP-CODE> [--json]\n"
            f"{prog} <figure-type> [--json]\n"
            f"{prog} <yaml-file-or-snippet> [--json]"
        ),
    )
    parser.add_argument(
        "target",
        help="JP-* code, figure type (posterior_2d), or YAML path/snippet with type:",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--expand",
        action="store_true",
        help="force type-expansion mode when target is a YAML file (default when path exists)",
    )
    return parser


def run(argv: Sequence[str], *, prog: str = "jplot explain") -> int:
    parser = build_parser(prog)
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        return system_exit_code(exc)

    as_json = bool(args.json) or not sys.stdout.isatty()
    target = str(args.target).strip()

    if target.startswith("JP-") or target in KNOWN_CODES:
        return _explain_code(target, as_json=as_json)

    type_name = _as_figure_type(target)
    if type_name is not None:
        return _explain_type(type_name, as_json=as_json)

    return _explain_expand(target, as_json=as_json)


def _as_figure_type(target: str) -> str | None:
    from ..Figure.figure_types import KNOWN_FIGURE_TYPES

    token = target.strip().lower().replace("-", "_")
    if token in KNOWN_FIGURE_TYPES:
        return token
    return None


def _explain_code(code: str, *, as_json: bool) -> int:
    suggestion, example = guidance_for(code, "$", "")
    data = {
        "mode": "code",
        "code": code,
        "suggestion": suggestion,
        "example": example,
        "known": code in KNOWN_CODES,
    }
    env = envelope("explain", True, data=data)
    if as_json:
        return emit(env)
    print(f"{code}: {suggestion}", file=sys.stderr)
    if example:
        print("example:", file=sys.stderr)
        print(example, file=sys.stderr)
    return EXIT_OK


def _explain_type(type_name: str, *, as_json: bool) -> int:
    """Document a figure type macro (what cap types advertises)."""
    from ..templates_catalog import get_template, render_template_yaml

    try:
        tmpl = get_template(type_name)
    except Exception as exc:
        env = envelope("explain", False, error=error_payload(exc), data={"type": type_name})
        return emit(env) if as_json else _fail(env)

    try:
        sample_yaml = render_template_yaml(type_name, values=None)
    except Exception:
        sample_yaml = ""

    data = {
        "mode": "type",
        "type": type_name,
        "title": tmpl.get("title") or type_name,
        "summary": tmpl.get("summary") or "",
        "slots": tmpl.get("slots") or [],
        "sample_yaml": sample_yaml,
        "related_cli": [
            {"argv": ["jplot", "template", "show", type_name, "--json"], "why": "slots + skeleton"},
            {
                "argv": ["jplot", "config", "expand", "<yaml>", "--figure", "<name>", "--write"],
                "why": "lower type: to layers in-file",
            },
            {
                "argv": ["jplot", "explain", "<yaml-with-type>", "--json"],
                "why": "print expanded layers for a concrete YAML",
            },
        ],
        "write_yaml": False,
    }
    env = envelope("explain", True, data=data)
    if as_json:
        return emit(env)
    print(f"type: {type_name}", file=sys.stderr)
    if data["summary"]:
        print(data["summary"], file=sys.stderr)
    if sample_yaml:
        print(sample_yaml, end="")
    return EXIT_OK


def _explain_expand(target: str, *, as_json: bool) -> int:
    try:
        text, base = _load_target_yaml(target)
        config = yaml.safe_load(text)
    except Exception as exc:
        env = envelope("explain", False, error=error_payload(exc))
        return emit(env) if as_json else _fail(env)

    if not isinstance(config, dict):
        env = envelope(
            "explain",
            False,
            error=error_payload(
                TypeError(
                    "explain expand expects a YAML mapping (plot config), "
                    f"got {type(config).__name__}. "
                    "For a figure type use `jplot explain posterior_2d`; "
                    "for a JP code use `jplot explain JP-VIZ-003`."
                )
            ),
        )
        return emit(env) if as_json else _fail(env)

    try:
        from ..Figure.figure_types import expand_typed_figures

        names = expand_typed_figures(config, raise_on_error=True)
    except Exception as exc:
        env = envelope("explain", False, error=error_payload(exc))
        return emit(env) if as_json else _fail(env)

    if not names:
        env = envelope(
            "explain",
            False,
            error=error_payload(
                ValueError(
                    "no type: figures to expand in this YAML "
                    "(already layers form, or empty Figures)"
                )
            ),
            data={"source": base},
        )
        return emit(env) if as_json else _fail(env)

    yaml_text = dump_yaml_doc(config, meta={"engine": "pyyaml"})
    data = {
        "mode": "expand",
        "source": base,
        "expanded": names,
        "yaml_text": yaml_text,
        "write_hint": "jplot config expand <file> [--figure NAME] --write",
    }
    env = envelope("explain", True, data=data)
    if as_json:
        return emit(env)
    print(yaml_text, end="")
    return EXIT_OK


def _load_target_yaml(target: str) -> tuple[str, str]:
    from pathlib import Path

    path = Path(target).expanduser()
    if path.is_file():
        return path.read_text(encoding="utf-8"), str(path.resolve())
    # treat as inline YAML
    return target, "<inline>"


def _fail(env: dict) -> int:
    print(env.get("error", {}).get("message", "failed"), file=sys.stderr)
    return EXIT_FAILED

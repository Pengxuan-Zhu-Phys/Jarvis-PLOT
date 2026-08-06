#!/usr/bin/env python3

"""``jplot explain`` -- error-code help or type: expansion (agent knowledge)."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import yaml

from ..agent_io import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, envelope, error_payload
from ..diagnostic_guidance import KNOWN_CODES, guidance_for

__all__ = ["build_parser", "run"]


def build_parser(prog: str = "jplot explain") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Explain a JP-* code, or expand a type: figure block to layers.",
    )
    parser.add_argument(
        "target",
        help="JP-* code (e.g. JP-VIZ-003) or path to a YAML snippet/file with type:",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--expand",
        action="store_true",
        help="force type-expansion mode when target is a YAML file",
    )
    return parser


def run(argv: Sequence[str], *, prog: str = "jplot explain") -> int:
    parser = build_parser(prog)
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        return int(exc.code or EXIT_USAGE)

    as_json = bool(args.json) or not sys.stdout.isatty()
    target = str(args.target)

    if target.startswith("JP-") or target in KNOWN_CODES:
        suggestion, example = guidance_for(target, "$", "")
        data = {
            "mode": "code",
            "code": target,
            "suggestion": suggestion,
            "example": example,
            "known": target in KNOWN_CODES,
        }
        env = envelope("explain", True, data=data)
        if as_json:
            return emit(env)
        print(f"{target}: {suggestion}", file=sys.stderr)
        if example:
            print("example:", file=sys.stderr)
            print(example, file=sys.stderr)
        return EXIT_OK

    # YAML expand path
    try:
        text, base = _load_target_yaml(target)
        config = yaml.safe_load(text)
    except Exception as exc:
        env = envelope("explain", False, error=error_payload(exc))
        return emit(env) if as_json else _fail(env)

    try:
        from ..Figure.figure_types import expand_figure_types_in_config

        expanded = expand_figure_types_in_config(config)
    except Exception as exc:
        env = envelope("explain", False, error=error_payload(exc))
        return emit(env) if as_json else _fail(env)

    yaml_text = yaml.safe_dump(
        expanded, sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    data = {"mode": "expand", "source": base, "yaml_text": yaml_text}
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

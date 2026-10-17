"""Typography CLI persistence, discovery, machine output and human cards."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from rich.cells import cell_len
from rich.text import Text

from jarvisplot.client import main
from jarvisplot.cli import render_help
from jarvisplot.verbs import is_verb
from jarvisplot.verbs.fonts import _render_human, build_parser


def _command(capsys, *arguments, expected=0):
    assert main(["fonts", *arguments, "--json"]) == expected
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    result = json.loads(captured.out)
    assert set(result) == {"api_version", "kind", "ok", "data", "diagnostics", "error"}
    return result


def test_defaults_are_queryable_per_role_without_creating_user_files(capsys):
    result = _command(capsys)
    assert result["kind"] == "fonts.show"
    data = result["data"]
    assert data["card"]["families"] == {"sans": "Helvetica", "serif": "STIXGeneral"}
    assert data["card"]["math"] == {"fontset": "stix"}
    assert set(data["roles"]) == {"title", "axis_label", "tick", "legend", "text", "annotation", "caption"}
    assert data["roles"]["title"]["size_pt"] is None
    assert data["roles"]["title"]["recommended_size_pt"] > 0
    assert data["roles"]["title"]["math_fontfamily"] == "stix"
    assert not Path(data["user_path"]).parent.exists()
    legend = _command(capsys, "show", "legend")["data"]
    assert set(legend["roles"]) == {"legend"}


def test_set_recommended_and_partial_reset_preserve_other_overrides(capsys):
    _command(capsys, "set", "families.sans", "STIXGeneral")
    result = _command(capsys, "set", "roles.title.size_pt", "17.5")["data"]
    assert result["roles"]["title"]["size_pt"] == 17.5
    assert result["roles"]["title"]["family"] == "STIXGeneral"
    assert result["has_overrides"] is True
    recommended = _command(capsys, "show", "title", "--recommended")["data"]
    assert recommended["source"] == "recommended"
    assert recommended["roles"]["title"]["size_pt"] is None
    assert recommended["card"]["families"]["sans"] == "Helvetica"
    partial = _command(capsys, "reset", "roles.title.size_pt")["data"]
    assert partial["roles"]["title"]["size_pt"] is None
    assert partial["card"]["families"]["sans"] == "STIXGeneral"
    _command(capsys, "set", "roles.title.size_pt", "null")
    final = _command(capsys, "reset")["data"]
    assert final["card"] == final["recommended"]
    assert final["has_overrides"] is False


def test_default_card_survives_a_new_cli_process(capsys):
    _command(capsys, "set", "roles.legend.family", "serif")
    env = dict(os.environ, MPLCONFIGDIR="/tmp/jarvisplot-mpl")
    result = subprocess.run(
        [sys.executable, "-c", "from jarvisplot.client import main; raise SystemExit(main())",
         "fonts", "show", "legend", "--json"],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout)["data"]["roles"]["legend"]["family"] == "STIXGeneral"


@pytest.mark.parametrize("arguments", [
    ["unrecognized"], ["resolve"], ["resolve", "STIXGeneral", "--style", "tilted"],
    ["set", "roles.title.size_pt"], ["set", "roles.title.size_pt", "[1, 2]"],
    ["show", "unknown_role"], ["check", "--unsupported"],
])
def test_invalid_arguments_emit_clean_usage_envelopes(capsys, arguments):
    result = _command(capsys, *arguments, expected=2)
    assert result["ok"] is False
    assert result["error"]["type"] == "UsageError"


@pytest.mark.parametrize("path, value", [
    ("roles.title.size_pt", "-1"), ("roles.title.size_pt", "true"),
    ("roles.title.size_pt", "NaN"), ("roles.title.weight", "1001"),
    ("roles.title.style", "tilted"), ("roles.title.recommended_size_pt", "20"),
    ("families.unknown", "STIXGeneral"), ("schema_version", "2"),
])
def test_invalid_edits_leave_persisted_card_unchanged(capsys, path, value):
    data = _command(capsys, "set", "roles.legend.family", "serif")["data"]
    user_path = Path(data["user_path"])
    before = user_path.read_bytes()
    result = _command(capsys, "set", path, value, expected=2)
    assert result["error"]["type"] == "UsageError"
    assert user_path.read_bytes() == before


def test_corrupt_persisted_card_reports_failure_without_overwriting(capsys):
    data = _command(capsys, "set", "roles.legend.family", "serif")["data"]
    user_path = Path(data["user_path"])
    user_path.write_text("{invalid json}")
    _command(capsys, "show", expected=1)
    _command(capsys, "set", "roles.legend.family", "sans", expected=1)
    assert user_path.read_text() == "{invalid json}"
    # Recommendations remain queryable even with a corrupt override file.
    assert _command(capsys, "show", "--recommended")["data"]["source"] == "recommended"


def test_discovery_filtered_inventory_refresh_and_actual_face(capsys):
    data = _command(capsys, "list", "--filter", "stixgeneral", "--refresh")["data"]
    assert data["refreshed"] is True
    assert data["count"] == len(data["fonts"]) > 0
    assert all("stixgeneral" in row["family"].casefold() for row in data["fonts"])
    assert all(Path(row["path"]).is_file() for row in data["fonts"])
    resolved = _command(capsys, "resolve", "STIXGeneral", "--weight", "700")["data"]
    assert resolved["family"] == "STIXGeneral"
    assert Path(resolved["path"]).is_file()
    assert resolved["face"]
    result = _command(capsys, "resolve", "__JarvisMissingFont__", expected=1)
    assert result["ok"] is False


def test_check_reports_roles_and_other_missing_families(capsys):
    _command(capsys, "set", "families.sans", "STIXGeneral")
    result = _command(capsys, "check")
    assert result["ok"] is True
    assert all(row["available"] for row in result["data"]["results"])
    assert set().union(*(row["roles"] for row in result["data"]["results"])) >= {"title", "legend"}
    _command(capsys, "set", "roles.title.family", "__JarvisMissingFont__")
    missing = _command(capsys, "check", expected=1)
    assert any(not row["available"] and "title" in row["roles"] for row in missing["data"]["results"])


def test_json_flag_before_action_and_nontty_default(capsys):
    assert main(["fonts", "--json", "show", "title"]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "fonts.show"
    assert main(["fonts", "show", "legend"]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["roles"].keys() == {"legend"}


def test_fonts_discoverable_through_root_cap_and_manual(capsys):
    assert is_verb("fonts")
    assert "fonts" in render_help()
    assert main(["cap", "cli", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "fonts" in json.dumps(payload["data"])
    assert main(["man", "fonts", "--json"]) == 0
    manual = json.loads(capsys.readouterr().out)
    assert manual["ok"] is True
    assert "recommended_size_pt" in json.dumps(manual["data"])


def test_human_card_has_aligned_panels_and_visible_substitution(capsys):
    payload = _command(capsys, "show")
    rendered = _render_human("show", payload)
    assert "Role defaults" in rendered
    assert "inherit" in rendered
    assert "MathText: stix" in rendered
    assert {cell_len(line) for line in rendered.splitlines() if line.startswith(("╭", "│", "╰"))} == {80}
    fallback = {"api_version": 1, "kind": "fonts.resolve", "ok": True, "error": None,
                "data": {"requested_family": "Helvetica", "family": "DejaVu Sans", "path": "/fonts/sans.ttf",
                         "face": "Book", "weight": "normal", "style": "normal", "fallback": True,
                         "fallback_reason": "Helvetica is unavailable"}}
    human = _render_human("resolve", fallback)
    assert "substituted" in human
    assert "Helvetica" in human and "DejaVu Sans" in human


def test_human_usage_error_and_help_keep_jarvis_panels(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert main(["fonts", "resolve"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "╭─" in Text.from_ansi(captured.err).plain
    assert "set PATH VALUE" in build_parser().format_help()
    assert main(["fonts", "-h"]) == 0
    assert "fonts" in capsys.readouterr().out

"""jplot man: human Rich index/topics + agent --json; no matplotlib."""

from __future__ import annotations

import json
import sys

from rich.cells import cell_len

from jarvisplot.client import main
from jarvisplot.man_catalog import list_topics, load_card, load_manifest
from jarvisplot.man_render_human import render_index, render_topic


def _panel_lines(rendered: str) -> list[str]:
    return [line for line in rendered.splitlines() if line.startswith(("╭", "│", "╰"))]


def test_manifest_topics_all_load():
    topics = list_topics()
    assert "workflow" in topics
    assert "cli-map" in topics
    assert "methods" in topics
    assert "method.scatter" in topics
    for tid in topics:
        card = load_card(tid)
        assert card["id"] == tid
        assert card["title"]
        assert card["summary"]


def test_man_index_human_geometry(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    assert main(["man"]) == 0
    out = capsys.readouterr().out
    assert "Jarvis-PLOT manual" in out
    assert "Topics" in out
    assert "workflow" in out
    assert "{" not in out.split("Topics")[0] or "manual" in out  # not a JSON dump
    assert not out.lstrip().startswith("{")
    panel_lines = _panel_lines(out)
    assert panel_lines
    assert {cell_len(line) for line in panel_lines} == {80}


def test_man_topic_human(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    assert main(["man", "workflow"]) == 0
    out = capsys.readouterr().out
    assert "Coding-agent" in out or "workflow" in out.lower()
    assert "data describe" in out or "Loop" in out
    assert not out.lstrip().startswith("{")


def test_man_index_json(capsys):
    assert main(["man", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["kind"] == "man"
    assert env["ok"] is True
    assert env["data"]["audience"] == "agent"
    assert env["data"]["write_yaml"] is False
    ids = {t["id"] for t in env["data"]["topics"]}
    assert "workflow" in ids
    assert "validate" in ids


def test_man_topic_json_keys(capsys):
    assert main(["man", "workflow", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["kind"] == "man.workflow"
    assert env["ok"] is True
    data = env["data"]
    for key in (
        "topic",
        "audience",
        "title",
        "summary",
        "role",
        "related_cli",
        "live_sources",
        "sections",
        "write_yaml",
    ):
        assert key in data
    assert data["write_yaml"] is False
    assert data["related_cli"]
    assert any("describe" in " ".join(map(str, c.get("argv", []))) for c in data["related_cli"])


def test_man_unknown_topic_json(capsys):
    assert main(["man", "no-such-topic", "--json"]) == 1
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is False
    assert "unknown" in env["error"]["message"].lower()


def test_man_alias_playbook(capsys):
    assert main(["man", "playbook", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["data"]["topic"] == "workflow"


def test_man_does_not_import_matplotlib():
    # Subprocess isolation: man path must not pull matplotlib (same as validate).
    import subprocess

    code = (
        "import sys; "
        "from jarvisplot.client import main; "
        "raise SystemExit(main(['man', 'workflow'])); "
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    # After the man command process exits we cannot see its modules; probe via -c chain:
    probe = (
        "import sys; "
        "from jarvisplot.client import main; "
        "main(['man', 'workflow']); "
        "assert 'matplotlib' not in sys.modules, sorted(m for m in sys.modules if 'matplotlib' in m); "
        "print('ok')"
    )
    proc2 = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc2.returncode == 0, proc2.stderr + proc2.stdout
    assert "ok" in proc2.stdout


def test_render_helpers_nonempty():
    text = render_index(prog="jplot")
    assert "Topics" in text
    text2 = render_topic("validate", prog="jplot")
    assert "validate" in text2.lower()


def test_root_help_lists_man(monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    from jarvisplot.cli import render_help

    rendered = render_help()
    assert "│ man" in rendered
    assert "jplot man" in rendered


def test_man_methods_catalog_json(capsys):
    assert main(["man", "methods", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is True
    assert env["kind"] == "man.methods"
    methods = env["data"].get("methods") or []
    names = {m["name"] for m in methods}
    assert "scatter" in names
    assert "pcolormesh" in names
    assert len(methods) >= 20


def test_man_single_method_json(capsys):
    assert main(["man", "scatter", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is True
    assert env["kind"] == "man.method.scatter"
    data = env["data"]
    assert data["topic"] == "method.scatter"
    assert data["method"]["name"] == "scatter"
    assert "x" in data["method"]["coordinates"]["required"]
    assert data["examples"]


def test_man_corrplot_json_has_both_type_first_forms_and_verification(capsys):
    assert main(["man", "corrplot", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    data = env["data"]

    assert data["method"]["name"] == "corrplot"
    assert data["method"]["style_options"]["corrplot.diamond"]["side"]["values"] == [
        "left", "right"
    ]
    titles = {item["title"] for item in data["examples"]}
    assert titles == {"square correlation matrix", "diamond correlation matrix"}
    yaml_by_title = {item["title"]: item["yaml"] for item in data["examples"]}
    assert "type: correlation_matrix" in yaml_by_title["square correlation matrix"]
    assert "style: [corrplot, diamond]" in yaml_by_title["diamond correlation matrix"]
    verification = next(section for section in data["sections"] if section["id"] == "verification")
    names = {item["name"] for item in verification["items"]}
    assert names == {"validate", "doctor", "render"}
    assert any(
        item["argv"] == ["jplot", "doctor", "<yaml>", "--json"]
        for item in verification["items"]
    )
    assert data["verification"]["doctor"]["argv"] == [
        "jplot",
        "doctor",
        "<yaml>",
        "--json",
    ]


def test_man_correlation_type_json_has_both_forms(capsys):
    assert main(["man", "type-correlation-matrix", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    data = env["data"]
    assert len(data["examples"]) == 2
    assert any("style: [corrplot, diamond]" in item["yaml"] for item in data["examples"])
    assert any(
        item["argv"] == ["jplot", "validate", "<yaml>", "--json"]
        for section in data["sections"]
        if section["id"] == "verification"
        for item in section["items"]
    )
    assert data["verification"]["validate"]["argv"] == [
        "jplot",
        "validate",
        "<yaml>",
        "--json",
    ]


def test_man_method_dot_form(capsys):
    assert main(["man", "method.pcolormesh", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["data"]["method"]["name"] == "pcolormesh"
    assert "z" in env["data"]["method"]["coordinates"]["required"]


def test_man_unknown_method_did_you_mean(capsys):
    assert main(["man", "scater", "--json"]) == 1
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is False
    assert "scatter" in env["error"]["message"].lower() or "did you mean" in env["error"]["message"].lower()


def test_man_index_lists_methods(capsys):
    assert main(["man", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert any(t["id"] == "methods" for t in env["data"]["topics"])
    assert any(m["name"] == "scatter" for m in env["data"]["methods"])


def test_man_method_human(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    assert main(["man", "scatter"]) == 0
    out = capsys.readouterr().out
    assert "scatter" in out.lower()
    assert "coordinates" in out.lower() or "required" in out.lower()


def test_man_transforms_catalog_json(capsys):
    assert main(["man", "transforms", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is True
    names = {t["name"] for t in env["data"]["transforms"]}
    assert "filter" in names
    assert "profile" in names
    assert "make_interp_2d" in names
    assert "posterior_density" in names


def test_man_transform_make_interp_json(capsys):
    assert main(["man", "transform.make_interp_2d", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is True
    tr = env["data"]["transform"]
    assert tr["name"] == "make_interp_2d"
    assert "coordinates" in tr["required"]
    assert "method" in tr["optional"] or "grid" in tr["optional"]
    assert tr["defaults"]
    assert tr["examples"]
    assert tr["input"] == "table"
    assert tr["output"] == "table"


def test_man_transform_filter_bare_name(capsys):
    assert main(["man", "filter", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["data"]["transform"]["name"] == "filter"
    assert env["data"]["transform"]["examples"]


def test_man_transform_profile_prefix(capsys):
    # bare "profile" is aliased to type-profile-2d; use transform.profile
    assert main(["man", "transform.profile", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    tr = env["data"]["transform"]
    assert tr["name"] == "profile"
    assert "coordinates" in tr["required"]
    assert "bridson" in (tr.get("enums") or {}).get("method", [])


def test_cap_transforms_not_delegated(capsys):
    assert main(["cap", "transforms", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    rows = env["data"] if isinstance(env["data"], list) else env["data"].get("transforms") or env["data"]
    # cap envelope: data is the section payload
    items = env["data"]
    if isinstance(items, dict) and "transforms" not in items:
        # section returns list directly under data sometimes wrapped
        pass
    from jarvisplot.capabilities import section

    items = section("transforms")
    by_name = {t["name"]: t for t in items}
    assert "keys" not in by_name["make_interp_2d"] or "delegated" not in str(
        by_name["make_interp_2d"].get("keys", "")
    )
    assert by_name["make_interp_2d"]["required"]
    assert by_name["profile"]["defaults"]


def test_cap_types_no_typed_figures():
    from jarvisplot.capabilities import section

    names = {t["name"] for t in section("types")}
    assert "typed_figures" not in names
    assert "posterior_2d" in names
    assert "profile_2d" in names


def test_cap_funcs_filters_hash_noise():
    from jarvisplot.capabilities import section

    funcs = section("funcs")
    names = funcs["names"]
    assert "exp" in names or "ln" in names
    # no long digit soup
    for n in names:
        assert not (sum(ch.isdigit() for ch in n) >= 6)

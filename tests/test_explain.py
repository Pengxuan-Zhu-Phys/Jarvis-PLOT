"""jplot explain: codes + type expansion."""

from __future__ import annotations

import json
import textwrap

from jarvisplot.client import main


def test_explain_code_json(capsys):
    assert main(["explain", "JP-VIZ-003", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["kind"] == "explain"
    assert env["data"]["mode"] == "code"
    assert env["data"]["suggestion"]


def test_explain_expand_type(tmp_path, capsys):
    path = tmp_path / "fig.yaml"
    path.write_text(
        textwrap.dedent(
            """
            DataSet:
              - {name: samples, path: ./s.csv, type: csv}
            Figures:
              - name: p
                type: posterior_2d
                data: samples
                x: {expr: m_A}
                y: {expr: tanb}
                weight: {expr: exp(LogL)}
                style: [a4paper_2x1, rectcmap]
            """
        ).lstrip(),
        encoding="utf-8",
    )
    assert main(["explain", str(path), "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["data"]["mode"] == "expand"
    assert "layers" in env["data"]["yaml_text"]
    assert "posterior_2d" not in env["data"]["yaml_text"] or "type:" not in env["data"]["yaml_text"]


def test_explain_figure_type_name(capsys):
    assert main(["explain", "posterior_2d", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is True
    assert env["data"]["mode"] == "type"
    assert env["data"]["type"] == "posterior_2d"
    assert env["data"]["slots"]


def test_explain_unknown_bare_word_fails_clearly(capsys):
    # Must not claim "config must be a mapping" as if expand ran on a string.
    assert main(["explain", "not_a_type_or_file", "--json"]) == 1
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is False
    msg = (env.get("error") or {}).get("message", "").lower()
    assert "config must be a mapping" not in msg

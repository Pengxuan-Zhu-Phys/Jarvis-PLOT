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

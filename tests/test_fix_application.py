"""B8: jplot validate --fix applies mechanical renames."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from jarvisplot.client import main
from jarvisplot.fix_apply import apply_fixes, parse_yaml_path, planned_fixes
from jarvisplot.diagnostics import Diagnostic, Fix
from jarvisplot.validation import validate_config


def test_parse_yaml_path():
    assert parse_yaml_path("$.Figures[0].layers[1].method") == [
        "Figures",
        0,
        "layers",
        1,
        "method",
    ]
    assert parse_yaml_path("$") == []


def test_apply_rename_key_preserves_order():
    config = {"Layers": [], "DataSet": [], "Figures": [], "output": {}}
    fixed, applied = apply_fixes(
        config,
        [
            Fix(
                op="rename_key",
                path="$.Layers",
                old="Layers",
                to="layers",
                confidence="certain",
            )
        ],
    )
    assert "layers" in fixed and "Layers" not in fixed
    assert list(fixed.keys())[0] == "layers"
    assert applied[0]["to"] == "layers"


def test_case_a_fix_write(tmp_path):
    """Layers: / outputs: become layers: / output: in one --fix --write pass."""
    (tmp_path / "samples.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    path = tmp_path / "bad.yaml"
    path.write_text(
        textwrap.dedent(
            """
            DataSet:
              - {name: df, path: ./samples.csv, type: csv}
            outputs:
              dir: ./plots
            Figures:
              - name: f1
                Layers:
                  - name: s
                    data: [{source: df}]
                    axes: ax
                    method: scatter
                    coordinates:
                      x: {expr: x}
                      y: {expr: y}
            """
        ).lstrip(),
        encoding="utf-8",
    )
    assert main(["validate", str(path), "--fix", "--write", "--no-columns"]) in {0, 1}
    text = path.read_text(encoding="utf-8")
    assert "layers:" in text
    assert "Layers:" not in text
    assert "output:" in text
    assert "outputs:" not in text

    # second validate: no SCH-001 for the renamed keys
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["validate", str(path), "--json", "--no-columns"])
    env = json.loads(buf.getvalue())
    codes = {d["code"] for d in env["diagnostics"]}
    assert "JP-SCH-001" not in codes
    assert rc == 0


def test_fix_diff_json_lists_applied(tmp_path, capsys):
    path = tmp_path / "case.yaml"
    path.write_text(
        textwrap.dedent(
            """
            DataSet: []
            Figures:
              - name: f1
                Layers: []
            """
        ).lstrip(),
        encoding="utf-8",
    )
    assert main(["validate", str(path), "--fix", "--diff", "--json", "--no-columns"]) in {
        0,
        1,
    }
    env = json.loads(capsys.readouterr().out)
    assert env["data"]["fix"] is True
    assert env["data"]["fixes_planned"] >= 1
    assert env["data"]["wrote"] is False
    assert env["data"]["diff"] and "layers" in env["data"]["diff"]

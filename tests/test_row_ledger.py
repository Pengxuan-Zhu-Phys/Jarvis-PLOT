"""E3: dryrun row ledger + JP-VIZ-003 integration."""

from __future__ import annotations

import json
import textwrap

from jarvisplot.client import main
from jarvisplot.dryrun_runtime import dryrun_file


def test_filter_zero_rows_emits_viz_003(tmp_path):
    (tmp_path / "samples.csv").write_text(
        "x,y,LogL\n1,2,-5\n3,4,-3\n",
        encoding="utf-8",
    )
    config = tmp_path / "empty_filter.yaml"
    config.write_text(
        textwrap.dedent(
            """
            DataSet:
              - {name: df, path: ./samples.csv, type: csv}
            Figures:
              - name: f1
                frame:
                  ax:
                    xlim: [0, 10]
                    ylim: [0, 10]
                layers:
                  - name: pts
                    data:
                      - source: df
                        transform:
                          - filter: "LogL > 100"
                    axes: ax
                    method: scatter
                    coordinates:
                      x: {expr: x}
                      y: {expr: y}
            """
        ).lstrip(),
        encoding="utf-8",
    )
    report, bag = dryrun_file(str(config))
    layers = report["layers"]
    assert layers
    steps = layers[0]["steps"]
    assert steps and steps[0]["rows_in"] == 2 and steps[0]["rows_out"] == 0
    codes = {d.code for d in bag}
    assert "JP-VIZ-003" in codes
    assert "JP-VIZ-001" in codes


def test_jplot_dryrun_json(tmp_path, capsys):
    (tmp_path / "samples.csv").write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
    config = tmp_path / "ok.yaml"
    config.write_text(
        textwrap.dedent(
            """
            DataSet:
              - {name: df, path: ./samples.csv, type: csv}
            Figures:
              - name: f1
                layers:
                  - name: pts
                    data: [{source: df}]
                    method: scatter
                    coordinates:
                      x: {expr: x}
                      y: {expr: y}
            """
        ).lstrip(),
        encoding="utf-8",
    )
    assert main(["dryrun", str(config), "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["kind"] == "dryrun"
    assert env["ok"] is True
    assert env["data"]["datasets"]["df"]["rows"] == 2
    assert env["data"]["layers"][0]["n_points"] == 2


def test_jplot_doctor_combines_validate_and_dryrun(tmp_path, capsys):
    (tmp_path / "samples.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    config = tmp_path / "doc.yaml"
    config.write_text(
        textwrap.dedent(
            """
            DataSet:
              - {name: df, path: ./samples.csv, type: csv}
            Figures:
              - name: f1
                Layers:
                  - name: pts
                    data: [{source: df}]
                    method: scatter
                    coordinates:
                      x: {expr: x}
                      y: {expr: y}
            """
        ).lstrip(),
        encoding="utf-8",
    )
    # Layers: is a schema error; doctor should surface it
    code = main(["doctor", str(config), "--json", "--no-columns"])
    env = json.loads(capsys.readouterr().out)
    assert env["kind"] == "doctor"
    codes = {d["code"] for d in env["diagnostics"]}
    assert "JP-SCH-001" in codes
    assert code == 1

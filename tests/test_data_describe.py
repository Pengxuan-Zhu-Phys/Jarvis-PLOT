"""C1 (+ light C2): jplot data describe --json."""

from __future__ import annotations

import json

import pytest

from jarvisplot.client import main
from jarvisplot.verbs.data import describe_file


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text(
        "m_A,tanb,LogL,weight\n"
        "100,10,-5.0,0.1\n"
        "200,20,-3.0,0.2\n"
        "300,30,-1.0,0.3\n",
        encoding="utf-8",
    )
    return path


def test_describe_csv_columns_and_role_hint(csv_file):
    data = describe_file(str(csv_file))
    assert data["type"] == "csv"
    assert data["rows"] == 3
    names = [c["name"] for c in data["columns"]]
    assert names == ["m_A", "tanb", "LogL", "weight"]
    by_name = {c["name"]: c for c in data["columns"]}
    assert by_name["LogL"]["role_hint"] == "log_likelihood"
    assert by_name["weight"]["role_hint"] == "weight"
    assert "min" in by_name["m_A"]
    assert "q" in by_name["m_A"]


def test_jplot_data_describe_json(csv_file, capsys):
    assert main(["data", "describe", str(csv_file), "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["kind"] == "data.describe"
    assert env["ok"] is True
    assert env["data"]["rows"] == 3
    assert {c["name"] for c in env["data"]["columns"]} == {
        "m_A",
        "tanb",
        "LogL",
        "weight",
    }


def test_jplot_data_describe_missing_file(tmp_path, capsys):
    missing = tmp_path / "nope.csv"
    assert main(["data", "describe", str(missing), "--json"]) == 1
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is False
    assert env["error"]["type"] == "FileNotFoundError"


def test_expr_functions_are_not_missing_columns(tmp_path):
    """Regression: ln / log10 / Max must not become JP-COL-001."""
    import textwrap

    import yaml

    from jarvisplot.validation import validate_config

    (tmp_path / "samples.csv").write_text("x,LogL\n1,-2\n", encoding="utf-8")
    config = yaml.safe_load(
        textwrap.dedent(
            """
            DataSet:
              - {name: df, path: ./samples.csv, type: csv}
            Figures:
              - name: f1
                layers:
                  - name: s
                    data: [{source: df}]
                    axes: ax
                    method: scatter
                    coordinates:
                      x: {expr: "np.log10(x)"}
                      y: {expr: "ln(LogL)"}
                      c: {expr: "Max(x, LogL)"}
            """
        )
    )
    bag = validate_config(config, base_dir=str(tmp_path))
    col_errors = [d for d in bag if d.code == "JP-COL-001"]
    assert col_errors == [], [d.message for d in col_errors]


def test_to_parquet_is_a_legal_transform_step(tmp_path):
    import textwrap

    import yaml

    from jarvisplot.validation import validate_config

    (tmp_path / "samples.csv").write_text("x\n1\n", encoding="utf-8")
    config = yaml.safe_load(
        textwrap.dedent(
            """
            DataSet:
              - name: df
                path: ./samples.csv
                type: csv
                transform:
                  - to_parquet: ./out.parquet
            Figures:
              - name: f1
                layers:
                  - {name: a, data: [{source: df}], method: scatter}
            """
        )
    )
    bag = validate_config(config, base_dir=str(tmp_path), check_columns=False)
    sch = [d for d in bag if d.code.startswith("JP-SCH-")]
    assert not any("to_parquet" in d.message for d in sch), [d.message for d in sch]

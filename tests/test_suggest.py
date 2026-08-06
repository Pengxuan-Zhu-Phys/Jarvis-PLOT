"""F2: jplot suggest data-aware synthesis."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from jarvisplot.client import main
from jarvisplot.validation import validate_config
from jarvisplot.verbs.suggest import suggest_config


def test_suggest_posterior_from_csv(tmp_path):
    path = tmp_path / "egg.csv"
    path.write_text(
        "m_A,tanb,LogL\n"
        "100,10,-5\n"
        "1000,20,-3\n"
        "10000,30,-1\n",
        encoding="utf-8",
    )
    result = suggest_config(data_path=str(path), kind="posterior_2d")
    assert result["decisions"]
    assert all(d.get("reason") for d in result["decisions"])
    config = yaml.safe_load(result["yaml_text"])
    assert config["Figures"][0]["type"] == "posterior_2d"
    bag = validate_config(config, base_dir=str(tmp_path), check_columns=False)
    assert bag.ok, [(d.code, d.message) for d in bag.errors]


def test_jplot_suggest_json(tmp_path, capsys):
    path = tmp_path / "s.csv"
    path.write_text("a,b,LogL\n1,2,-1\n10,20,-2\n", encoding="utf-8")
    assert main(["suggest", "--data", str(path), "--kind", "scatter_2d", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["kind"] == "suggest"
    assert env["ok"] is True
    assert "yaml_text" in env["data"]
    assert env["data"]["decisions"]


def test_suggest_write_validates(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("x,y,LogL\n1,2,-1\n", encoding="utf-8")
    out = tmp_path / "out.yaml"
    assert (
        main(
            [
                "suggest",
                "--data",
                str(path),
                "--kind",
                "posterior_2d",
                "--write",
                str(out),
                "--json",
            ]
        )
        == 0
    )
    assert out.is_file()
    config = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert config["Figures"][0]["type"] == "posterior_2d"

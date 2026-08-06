"""F1: jplot template list/show."""

from __future__ import annotations

import json

import yaml

from jarvisplot.client import main
from jarvisplot.templates_catalog import list_templates, render_template_yaml
from jarvisplot.validation import validate_config


def test_list_includes_posterior_and_profile():
    kinds = {t["kind"] for t in list_templates()}
    assert {"posterior_2d", "profile_2d", "scatter_2d"} <= kinds


def test_show_posterior_yaml_validates_shape(tmp_path):
    text = render_template_yaml(
        "posterior_2d",
        values={"path": str(tmp_path / "samples.csv")},
    )
    (tmp_path / "samples.csv").write_text("m_A,tanb,LogL\n1,2,-3\n", encoding="utf-8")
    config = yaml.safe_load(text)
    # rewrite path absolute already in values
    config["DataSet"][0]["path"] = str(tmp_path / "samples.csv")
    bag = validate_config(config, base_dir=str(tmp_path), check_columns=False)
    assert bag.ok, [d.message for d in bag.errors]


def test_jplot_template_list_json(capsys):
    assert main(["template", "list", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["kind"] == "template.list"
    assert env["data"]["templates"]


def test_jplot_template_show_json(capsys):
    assert main(["template", "show", "profile_2d", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["kind"] == "template.show"
    assert env["data"]["kind"] == "profile_2d"
    assert "yaml_text" in env["data"]
    assert env["data"]["slots"]

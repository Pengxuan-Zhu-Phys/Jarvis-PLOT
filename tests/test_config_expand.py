"""jplot config expand: type: → layers convert with write-validate-rollback."""

from __future__ import annotations

import json
import textwrap

from jarvisplot.Figure.figure_types import expand_typed_figures
from jarvisplot.client import main


def _type_yaml(path, *, name: str = "p") -> None:
    path.write_text(
        textwrap.dedent(
            f"""
            DataSet:
              - {{name: samples, path: ./s.csv, type: csv}}
            Figures:
              - name: {name}
                type: posterior_2d
                data: samples
                x: {{expr: m_A}}
                y: {{expr: tanb}}
                weight: {{expr: exp(LogL)}}
                style: [a4paper_2x1, rectcmap]
              - name: manual
                style: [a4paper_2x1, rect]
                layers:
                  - name: pts
                    data: [{{source: samples}}]
                    method: scatter
                    coordinates:
                      x: {{expr: m_A}}
                      y: {{expr: tanb}}
                    style: {{s: 2}}
            """
        ).lstrip(),
        encoding="utf-8",
    )
    path.parent.joinpath("s.csv").write_text("m_A,tanb,LogL\n1,2,0\n", encoding="utf-8")


def test_expand_typed_figures_single_name():
    cfg = {
        "Figures": [
            {
                "name": "p",
                "type": "posterior_2d",
                "data": "samples",
                "x": {"expr": "m_A"},
                "y": {"expr": "tanb"},
                "weight": {"expr": "exp(LogL)"},
            },
            {"name": "other", "type": "posterior_2d", "data": "s", "x": {"expr": "x"}, "y": {"expr": "y"}, "weight": {"expr": "w"}},
        ]
    }
    names = expand_typed_figures(cfg, figure_names=["p"], raise_on_error=True)
    assert names == ["p"]
    assert "type" not in cfg["Figures"][0]
    assert "layers" in cfg["Figures"][0]
    assert cfg["Figures"][1].get("type") == "posterior_2d"


def test_config_expand_diff_default(tmp_path, capsys):
    path = tmp_path / "c.yaml"
    _type_yaml(path)
    assert main(["config", "expand", str(path), "--json", "--no-columns"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["kind"] == "config.expand"
    assert env["ok"] is True
    assert env["data"]["wrote"] is False
    assert env["data"]["expanded"] == ["p"]
    assert env["data"]["diff"] and "layers:" in (env["data"]["diff"] or "")
    # file unchanged without --write
    text = path.read_text(encoding="utf-8")
    assert "type: posterior_2d" in text
    assert "name: manual" in text


def test_config_expand_write_one_figure(tmp_path, capsys):
    path = tmp_path / "c.yaml"
    _type_yaml(path)
    assert (
        main(
            [
                "config",
                "expand",
                str(path),
                "--figure",
                "p",
                "--write",
                "--json",
                "--no-columns",
            ]
        )
        == 0
    )
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is True
    assert env["data"]["wrote"] is True
    assert env["data"]["expanded"] == ["p"]
    text = path.read_text(encoding="utf-8")
    assert "type: posterior_2d" not in text
    assert "layers:" in text
    assert "name: manual" in text
    # manual figure untouched
    assert "method: scatter" in text


def test_config_expand_missing_figure_fails(tmp_path, capsys):
    path = tmp_path / "c.yaml"
    _type_yaml(path)
    assert (
        main(
            [
                "config",
                "expand",
                str(path),
                "--figure",
                "nope",
                "--json",
                "--no-columns",
            ]
        )
        == 1
    )
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is False
    assert "not found" in env["error"]["message"].lower() or "nope" in env["error"]["message"]


def test_config_expand_already_layers_is_noop(tmp_path, capsys):
    """Idempotent: re-expanding layers form is ok with status=unchanged."""
    path = tmp_path / "c.yaml"
    _type_yaml(path)
    # expand once
    assert main(["config", "expand", str(path), "--write", "--json", "--no-columns"]) == 0
    capsys.readouterr()
    # second expand on whole file (now layers) → no-op success
    assert main(["config", "expand", str(path), "--json", "--no-columns"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is True
    assert env["data"]["status"] == "unchanged"
    assert env["data"]["wrote"] is False
    assert env["data"]["expanded"] == []


def test_config_expand_named_layers_figure_noop(tmp_path, capsys):
    path = tmp_path / "c.yaml"
    _type_yaml(path)
    assert (
        main(
            [
                "config",
                "expand",
                str(path),
                "--figure",
                "manual",
                "--json",
                "--no-columns",
            ]
        )
        == 0
    )
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is True
    assert env["data"]["status"] == "unchanged"

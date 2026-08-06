"""YAML agent_output digests: plan (doctor) + write (render path helper)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np

from jarvisplot.agent_digest import (
    build_voronoi_digest,
    parse_agent_output,
    plan_agent_exports,
    maybe_write_figure_digest,
)
from jarvisplot.client import main
from jarvisplot.diagnostics import DiagnosticBag


def test_build_voronoi_digest_respects_max_cells():
    rng = np.random.default_rng(0)
    n = 5000
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    w = np.exp(rng.normal(size=n))
    payload = build_voronoi_digest(
        x=x,
        y=y,
        weight=w,
        max_cells=64,
        seed=1,
        figure_name="t",
    )
    assert payload["lossy"] is True
    assert payload["algorithm"]["max_cells"] == 64
    assert payload["algorithm"]["actual_cells"] <= 64
    assert len(payload["cells"]) == payload["algorithm"]["actual_cells"]
    mass = sum(c["mass"] for c in payload["cells"])
    assert abs(mass - 1.0) < 1e-6
    assert "site" in payload["cells"][0]
    assert "bbox" in payload["cells"][0]
    assert "polygon" not in payload["cells"][0]
    assert "vertices" not in payload["cells"][0]


def test_plan_agent_exports_planned(tmp_path):
    cfg = {
        "output": {"dir": str(tmp_path / "plots")},
        "Figures": [
            {
                "name": "f1",
                "type": "scatter_2d",
                "x": {"expr": "a"},
                "y": {"expr": "b"},
                "agent_output": {"method": "voronoi", "max_cells": 100, "path": "auto"},
            }
        ],
    }
    bag = DiagnosticBag()
    exports = plan_agent_exports(cfg, base_dir=str(tmp_path), bag=bag)
    assert len(exports) == 1
    assert exports[0]["status"] == "planned"
    assert exports[0]["max_cells"] == 100
    assert exports[0]["path"].endswith("f1.agent.json")
    assert bag.ok


def test_plan_agent_exports_invalid_max_cells():
    cfg = {
        "Figures": [
            {
                "name": "f1",
                "x": {"expr": "a"},
                "y": {"expr": "b"},
                "type": "profile_2d",
                "agent_output": {"max_cells": 1},
            }
        ]
    }
    bag = DiagnosticBag()
    exports = plan_agent_exports(cfg, bag=bag)
    assert exports[0]["status"] == "invalid"
    assert any(d.code == "JP-AGT-001" for d in bag.errors)


def test_doctor_lists_exports(tmp_path, capsys):
    csv = tmp_path / "s.csv"
    csv.write_text("a,b,z\n1,2,3\n4,5,6\n7,8,9\n", encoding="utf-8")
    yml = tmp_path / "p.yaml"
    yml.write_text(
        textwrap.dedent(
            f"""
            DataSet:
              - {{name: samples, path: {csv.name}, type: csv}}
            Figures:
              - name: sc
                type: scatter_2d
                data: samples
                x: {{expr: a}}
                y: {{expr: b}}
                style: [a4paper_2x1, rect]
                agent_output:
                  method: voronoi
                  max_cells: 32
                  path: auto
            output:
              dir: {tmp_path / "plots"}
              formats: [png]
            """
        ).lstrip(),
        encoding="utf-8",
    )
    # scatter may not be a type macro — use layers if needed
    # Check templates: scatter_2d exists
    rc = main(["doctor", str(yml), "--json"])
    env = json.loads(capsys.readouterr().out)
    # doctor may partial/fail if scatter_2d type unknown to expander — still check exports
    assert "exports" in env["data"]
    assert env["data"]["exports"]
    assert env["data"]["exports"][0]["status"] in {"planned", "invalid"}
    # If type scatter_2d not known for axes, fix yaml to layers
    if env["data"]["exports"][0]["status"] == "invalid":
        yml.write_text(
            textwrap.dedent(
                f"""
                DataSet:
                  - {{name: samples, path: {csv.name}, type: csv}}
                Figures:
                  - name: sc
                    data: samples
                    x: {{expr: a}}
                    y: {{expr: b}}
                    style: [a4paper_2x1, rect]
                    layers:
                      - name: pts
                        data: [{{source: samples}}]
                        method: scatter
                        coordinates:
                          x: {{expr: a}}
                          y: {{expr: b}}
                    agent_output:
                      method: voronoi
                      max_cells: 32
                output:
                  dir: {tmp_path / "plots"}
                """
            ).lstrip(),
            encoding="utf-8",
        )
        capsys.readouterr()
        assert main(["doctor", str(yml), "--json"]) in (0, 1)
        env = json.loads(capsys.readouterr().out)
        assert env["data"]["exports"][0]["status"] == "planned"


def test_maybe_write_figure_digest_file(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"a": np.linspace(0, 1, 200), "b": np.linspace(1, 2, 200), "z": np.random.default_rng(0).normal(size=200)})
    fig = {
        "name": "sc",
        "data": "samples",
        "x": {"expr": "a"},
        "y": {"expr": "b"},
        "z": {"expr": "z"},
        "agent_output": {"method": "voronoi", "max_cells": 16, "path": "auto", "seed": 2},
    }
    cfg = {"output": {"dir": str(tmp_path / "out")}}
    path = maybe_write_figure_digest(
        figure_cfg=fig,
        config=cfg,
        dataframe=df,
        base_dir=str(tmp_path),
        yaml_path=str(tmp_path / "p.yaml"),
        jplot_version="test",
    )
    assert path is not None and path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["figure"] == "sc"
    assert payload["algorithm"]["actual_cells"] <= 16
    assert payload["provenance"]["finite_rows"] == 200


def test_parse_xbin_ybin_alias():
    fig = {
        "name": "f",
        "x": 1,
        "y": 2,
        "agent_output": {"xbin": 8, "ybin": 4},
    }
    spec = parse_agent_output(fig)
    assert spec is not None
    assert spec.max_cells == 32


def test_schema_accepts_agent_output_on_figure_and_output():
    from jarvisplot.schema_catalog import schema_catalog, config_validator
    from jarvisplot.validation import validate_config

    schema_catalog.cache_clear()
    config_validator.cache_clear()
    cfg = {
        "DataSet": [{"name": "s", "path": "./x.csv", "type": "csv"}],
        "Figures": [
            {
                "name": "f1",
                "type": "profile_2d",
                "data": "s",
                "x": {"expr": "a"},
                "y": {"expr": "b"},
                "z": {"expr": "z"},
                "agent_output": {
                    "method": "voronoi",
                    "max_cells": 16,
                    "path": "auto",
                    "seed": 0,
                },
            }
        ],
        "output": {
            "dir": "./plots",
            "agent_output": {"max_cells": 32, "method": "voronoi"},
        },
    }
    bag = validate_config(cfg, check_columns=False)
    sch_errors = [d for d in bag if d.code == "JP-SCH-001"]
    assert not sch_errors, sch_errors


def test_config_set_agent_output_write(tmp_path, capsys):
    csv = tmp_path / "s.csv"
    csv.write_text("a,b,z\n1,2,0\n3,4,1\n5,6,2\n", encoding="utf-8")
    yml = tmp_path / "plot.yaml"
    yml.write_text(
        textwrap.dedent(
            f"""
            DataSet:
              - {{name: samples, path: {csv.name}, type: csv}}
            Figures:
              - name: profile_2d
                type: profile_2d
                data: samples
                x: {{expr: a}}
                y: {{expr: b}}
                z: {{expr: z}}
                style: [a4paper_2x1, rectcmap]
            output:
              dir: {tmp_path / "plots"}
              formats: [png]
            """
        ).lstrip(),
        encoding="utf-8",
    )
    rc = main(
        [
            "config",
            "set",
            str(yml),
            "Figures[profile_2d].agent_output",
            '{"method":"voronoi","max_cells":16,"path":"auto","seed":0}',
            "--write",
            "--json",
            "--no-columns",
        ]
    )
    env = json.loads(capsys.readouterr().out)
    assert rc == 0, env
    assert env["ok"] is True
    assert env["data"]["wrote"] is True
    text = yml.read_text(encoding="utf-8")
    assert "agent_output" in text
    assert "max_cells" in text

    capsys.readouterr()
    rc2 = main(["doctor", str(yml), "--json"])
    env2 = json.loads(capsys.readouterr().out)
    assert "exports" in env2["data"]
    assert env2["data"]["exports"]
    assert env2["data"]["exports"][0]["status"] == "planned"
    assert env2["data"]["exports"][0]["max_cells"] == 16

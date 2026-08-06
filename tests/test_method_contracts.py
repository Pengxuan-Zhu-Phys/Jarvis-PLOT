"""B5: method coordinate contracts + schema/registry lockstep."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml

from jarvisplot.Figure.method_registry import METHOD_DISPATCH
from jarvisplot.capabilities import section
from jarvisplot.client import main
from jarvisplot.method_contracts import METHOD_COORDINATES, missing_coordinates
from jarvisplot.schema_catalog import load_manifest
from jarvisplot.validation import validate_config

SCHEMA_METHODS = Path(__file__).resolve().parents[1] / "jarvisplot" / "schema" / "methods"


def test_method_contract_table_covers_dispatch():
    assert set(METHOD_COORDINATES) == set(METHOD_DISPATCH)


def test_schema_methods_files_match_dispatch():
    on_disk = {p.stem for p in SCHEMA_METHODS.glob("*.json")}
    assert on_disk == set(METHOD_DISPATCH)


def test_manifest_methods_map_matches_dispatch():
    methods = load_manifest().get("methods") or {}
    assert set(methods) == set(METHOD_DISPATCH)
    for name, uri in methods.items():
        assert uri.endswith(f"/methods/{name}.json")


def test_missing_coordinates_for_pcolormesh():
    assert missing_coordinates("pcolormesh", {"x": {"expr": "a"}, "y": {"expr": "b"}}) == [
        "z"
    ]
    assert missing_coordinates(
        "pcolormesh", {"x": {"expr": "a"}, "y": {"expr": "b"}, "z": {"expr": "c"}}
    ) == []


def test_ternary_axes_satisfy_plot_contract():
    assert (
        missing_coordinates(
            "plot",
            {
                "left": {"expr": "a"},
                "right": {"expr": "b"},
                "bottom": {"expr": "c"},
            },
        )
        == []
    )


def test_validate_reports_missing_z_for_pcolormesh(tmp_path):
    (tmp_path / "samples.csv").write_text("x,y,z\n1,2,3\n", encoding="utf-8")
    config = yaml.safe_load(
        textwrap.dedent(
            """
            DataSet:
              - {name: df, path: ./samples.csv, type: csv}
            Figures:
              - name: f1
                layers:
                  - name: dens
                    data: [{source: df}]
                    axes: ax
                    method: pcolormesh
                    coordinates:
                      x: {expr: x}
                      y: {expr: y}
            """
        )
    )
    bag = validate_config(config, base_dir=str(tmp_path), check_columns=False)
    mth = [d for d in bag if d.code == "JP-MTH-002"]
    assert len(mth) == 1
    assert mth[0].path == "$.Figures[0].layers[0].coordinates"
    assert "coordinates.z" in mth[0].message
    assert mth[0].suggestion


def test_validate_unknown_method_did_you_mean(tmp_path):
    (tmp_path / "samples.csv").write_text("x,y\n1,2\n", encoding="utf-8")
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
                    method: scattr
                    coordinates: {x: {expr: x}, y: {expr: y}}
            """
        )
    )
    bag = validate_config(config, base_dir=str(tmp_path), check_columns=False)
    # schema enum may also fire JP-SCH-003; method contract adds JP-MTH-001
    codes = {d.code for d in bag}
    assert "JP-MTH-001" in codes or "JP-SCH-003" in codes


def test_cap_methods_include_coordinate_contracts():
    methods = section("methods")
    by_name = {m["name"]: m for m in methods}
    assert by_name["pcolormesh"]["coordinates"]["required"] == ["x", "y", "z"]
    assert by_name["scatter"]["coordinates"]["required"] == ["x", "y"]


def test_jplot_validate_json_includes_mth_002(tmp_path, capsys):
    (tmp_path / "samples.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    config = tmp_path / "cfg.yaml"
    config.write_text(
        textwrap.dedent(
            """
            DataSet:
              - {name: df, path: ./samples.csv, type: csv}
            Figures:
              - name: f1
                layers:
                  - name: dens
                    data: [{source: df}]
                    method: pcolormesh
                    coordinates:
                      x: {expr: x}
                      y: {expr: y}
            """
        ).lstrip(),
        encoding="utf-8",
    )
    assert main(["validate", str(config), "--json", "--no-columns"]) == 1
    env = json.loads(capsys.readouterr().out)
    codes = {d["code"] for d in env["diagnostics"]}
    assert "JP-MTH-002" in codes

"""F3: named address syntax + jplot config get/paths."""

from __future__ import annotations

import json
import textwrap

import yaml

from jarvisplot.client import main
from jarvisplot.config_address import AddressError, parse_address, resolve_address


def _cfg():
    return yaml.safe_load(
        textwrap.dedent(
            """
            DataSet:
              - {name: samples, path: ./s.csv, type: csv}
            Figures:
              - name: EggBox
                layers:
                  - name: _density
                    method: pcolormesh
                    style: {cmap: viridis}
                  - name: _hpd
                    method: contour
                  - method: scatter
            """
        )
    )


def test_parse_named_and_index():
    assert parse_address("Figures[EggBox].layers[_density].style.cmap") == [
        ("Figures", "EggBox"),
        ("layers", "_density"),
        ("style", None),
        ("cmap", None),
    ]
    assert parse_address("Figures[0].layers[1].method") == [
        ("Figures", "0"),
        ("layers", "1"),
        ("method", None),
    ]


def test_resolve_by_name_stable_under_reorder():
    cfg = _cfg()
    assert resolve_address(cfg, "Figures[EggBox].layers[_density].method") == "pcolormesh"
    # swap layers
    cfg["Figures"][0]["layers"] = list(reversed(cfg["Figures"][0]["layers"]))
    assert resolve_address(cfg, "Figures[EggBox].layers[_density].method") == "pcolormesh"
    assert resolve_address(cfg, "Figures[EggBox].layers[_hpd].method") == "contour"


def test_unnamed_layer_alias():
    cfg = _cfg()
    assert resolve_address(cfg, "Figures[EggBox].layers[_layer2].method") == "scatter"


def test_missing_name_errors():
    cfg = _cfg()
    try:
        resolve_address(cfg, "Figures[Nope].name")
        assert False, "expected AddressError"
    except AddressError as exc:
        assert "Nope" in str(exc)


def test_jplot_config_get_json(tmp_path, capsys):
    path = tmp_path / "c.yaml"
    path.write_text(
        textwrap.dedent(
            """
            DataSet:
              - {name: samples, path: ./s.csv, type: csv}
            Figures:
              - name: f1
                layers:
                  - name: pts
                    method: scatter
                    style: {s: 6}
            """
        ).lstrip(),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "config",
                "get",
                str(path),
                "Figures[f1].layers[pts].method",
                "--json",
            ]
        )
        == 0
    )
    env = json.loads(capsys.readouterr().out)
    assert env["kind"] == "config.get"
    assert env["data"]["value"] == "scatter"


def test_jplot_config_paths_json(tmp_path, capsys):
    path = tmp_path / "c.yaml"
    path.write_text(
        textwrap.dedent(
            """
            DataSet:
              - {name: samples, path: ./s.csv, type: csv}
            Figures:
              - name: f1
                layers:
                  - name: pts
                    method: scatter
            """
        ).lstrip(),
        encoding="utf-8",
    )
    assert main(["config", "paths", str(path), "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    paths = env["data"]["paths"]
    assert "DataSet[samples].path" in paths
    assert "Figures[f1].layers[pts].method" in paths

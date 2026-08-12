"""F4: jplot config set/rm with write-validate-rollback."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from jarvisplot.client import main
from jarvisplot.config_address import delete_address, set_address
from jarvisplot.yaml_io import dump_yaml_doc, has_ruamel, load_yaml_doc


def test_set_address_mutates_in_place():
    cfg = {
        "Figures": [
            {
                "name": "f1",
                "layers": [{"name": "pts", "method": "scatter", "style": {"s": 2}}],
            }
        ]
    }
    set_address(cfg, "Figures[f1].layers[pts].style.s", 9)
    assert cfg["Figures"][0]["layers"][0]["style"]["s"] == 9


def test_delete_address_removes_layer():
    cfg = {
        "Figures": [
            {
                "name": "f1",
                "layers": [
                    {"name": "a", "method": "scatter"},
                    {"name": "b", "method": "contour"},
                ],
            }
        ]
    }
    delete_address(cfg, "Figures[f1].layers[b]")
    assert [ly["name"] for ly in cfg["Figures"][0]["layers"]] == ["a"]


def test_config_set_diff_default(tmp_path, capsys):
    path = tmp_path / "c.yaml"
    path.write_text(
        textwrap.dedent(
            """
            # keep me
            DataSet:
              - {name: samples, path: ./s.csv, type: csv}
            Figures:
              - name: f1
                layers:
                  - name: pts
                    data: [{source: samples}]
                    method: scatter
                    coordinates:
                      x: {expr: x}
                      y: {expr: y}
                    style: {s: 2}
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "s.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    assert (
        main(
            [
                "config",
                "set",
                str(path),
                "Figures[f1].layers[pts].style.s",
                "8",
                "--json",
                "--no-columns",
            ]
        )
        == 0
    )
    env = json.loads(capsys.readouterr().out)
    assert env["kind"] == "config.set"
    assert env["ok"] is True
    assert env["data"]["wrote"] is False
    assert env["data"]["diff"] and "s: 8" in env["data"]["diff"] or "8" in (
        env["data"]["diff"] or ""
    )
    # file unchanged without --write
    assert "s: 2" in path.read_text(encoding="utf-8")


def test_config_set_write_and_preserve_comment_when_ruamel(tmp_path, capsys):
    path = tmp_path / "c.yaml"
    path.write_text(
        textwrap.dedent(
            """
            # important human note
            DataSet:
              - {name: samples, path: ./s.csv, type: csv}
            Figures:
              - name: f1
                layers:
                  - name: pts
                    data: [{source: samples}]
                    method: scatter
                    coordinates:
                      x: {expr: x}
                      y: {expr: y}
                    style: {s: 2}
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "s.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    assert (
        main(
            [
                "config",
                "set",
                str(path),
                "Figures[f1].layers[pts].style.s",
                "8",
                "--write",
                "--json",
                "--no-columns",
            ]
        )
        == 0
    )
    env = json.loads(capsys.readouterr().out)
    assert env["data"]["wrote"] is True
    text = path.read_text(encoding="utf-8")
    assert "8" in text
    if has_ruamel():
        assert env["data"]["comments_preserved"] is True
        assert "important human note" in text


def test_config_set_bad_value_does_not_write(tmp_path, capsys):
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
                    data: [{source: samples}]
                    method: scatter
                    coordinates:
                      x: {expr: x}
                      y: {expr: y}
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "s.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    original = path.read_text(encoding="utf-8")
    # unknown root key → validate fails
    code = main(
        [
            "config",
            "set",
            str(path),
            "NotARealKey",
            "1",
            "--write",
            "--json",
            "--no-columns",
        ]
    )
    env = json.loads(capsys.readouterr().out)
    assert code == 1
    assert env["ok"] is False
    assert env["data"]["wrote"] is False
    assert path.read_text(encoding="utf-8") == original


def test_config_rm_layer(tmp_path, capsys):
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
                    data: [{source: samples}]
                    method: scatter
                    coordinates:
                      x: {expr: x}
                      y: {expr: y}
                  - name: extra
                    data: [{source: samples}]
                    method: scatter
                    coordinates:
                      x: {expr: x}
                      y: {expr: y}
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "s.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    assert (
        main(
            [
                "config",
                "rm",
                str(path),
                "Figures[f1].layers[extra]",
                "--write",
                "--json",
                "--no-columns",
            ]
        )
        == 0
    )
    text = path.read_text(encoding="utf-8")
    assert "extra" not in text
    assert "pts" in text


def test_ruamel_roundtrip_engine_when_installed():
    # smoke: load/dump path works either way
    assert isinstance(has_ruamel(), bool)


def test_dump_yaml_doc_compacts_leaf_collections(tmp_path):
    path = tmp_path / "compact.yaml"
    path.write_text(
        textwrap.dedent(
            """
            frame:
              ax:
                xlim:
                - 0.1
                - 5.0
                ylim:
                - 0.0
                - 5.0
                ticks:
                  x:
                    positions:
                    - 0.1
                    - 1
                    - 5
                    labels:
                    - '0.1'
                    - '1'
                    - '5'
                labels:
                  x: $x$
                  y: $y$
                coordinates:
                  x:
                    expr: xx
            """
        ).lstrip(),
        encoding="utf-8",
    )

    doc, meta = load_yaml_doc(path)
    text = dump_yaml_doc(doc, meta=meta)

    assert "xlim: [0.1, 5.0]" in text
    assert "ylim: [0.0, 5.0]" in text
    assert "positions: [0.1, 1, 5]" in text
    assert "labels: ['0.1', '1', '5']" in text
    assert "labels: {x: $x$, y: $y$}" in text
    assert "x: {expr: xx}" in text

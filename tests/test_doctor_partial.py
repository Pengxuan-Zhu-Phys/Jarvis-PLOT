"""doctor/dryrun: heavy-transform skip is partial, not failed."""

from __future__ import annotations

import json
import textwrap

from jarvisplot.client import main


def _profile_yaml(tmp_path) -> str:
    csv = tmp_path / "s.csv"
    csv.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n", encoding="utf-8")
    path = tmp_path / "profile.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            DataSet:
              - {{name: samples, path: {csv.name}, type: csv}}
            Figures:
              - name: p
                type: profile_2d
                data: samples
                x: {{expr: a}}
                y: {{expr: b}}
                z: {{expr: c}}
                style: [a4paper_2x1, rectcmap]
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return str(path)


def test_doctor_profile_2d_is_partial_renderable_not_failed(tmp_path, capsys):
    path = _profile_yaml(tmp_path)
    rc = main(["doctor", path, "--json"])
    env = json.loads(capsys.readouterr().out)
    assert rc == 0, env
    assert env["ok"] is None  # not false — incomplete check only
    assert env["data"]["status"] == "partial_renderable"
    assert env["data"]["coverage"] == "partial"
    assert env["data"]["renderable"] is True
    assert "render" in (env["data"].get("status_note") or "").lower()
    assert env["data"]["heavy_skipped"]
    # No hard errors pretending the YAML is broken
    errors = [d for d in env["diagnostics"] if d.get("level") == "error"]
    assert errors == [], errors


def test_dryrun_profile_2d_partial_renderable(tmp_path, capsys):
    path = _profile_yaml(tmp_path)
    rc = main(["dryrun", path, "--json"])
    env = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert env["ok"] is None
    assert env["data"]["coverage"] == "partial"
    assert env["data"]["status"] == "partial_renderable"
    assert env["data"]["renderable"] is True
    codes = {d.get("code") for d in env["diagnostics"]}
    assert "JP-VIZ-010" in codes


def test_doctor_bad_column_still_failed(tmp_path, capsys):
    csv = tmp_path / "s.csv"
    csv.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    path = tmp_path / "bad.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            DataSet:
              - {{name: samples, path: {csv.name}, type: csv}}
            Figures:
              - name: p
                type: profile_2d
                data: samples
                x: {{expr: nope}}
                y: {{expr: b}}
                z: {{expr: c}}
                style: [a4paper_2x1, rectcmap]
            """
        ).lstrip(),
        encoding="utf-8",
    )
    rc = main(["doctor", str(path), "--json"])
    env = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert env["ok"] is False
    assert env["data"]["status"] == "failed"

"""doctor/dryrun: heavy-transform skip is partial, not failed.

Check phase never re-runs profile/density/interp — only structure/columns/
light steps (+ pre-transform lim proxy). Execution is ``jplot <yaml>``.
"""

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
    assert "deep" not in env["data"]
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
    assert "deep" not in env["data"]
    codes = {d.get("code") for d in env["diagnostics"]}
    assert "JP-VIZ-010" in codes


def test_doctor_rejects_unknown_deep_flag(tmp_path, capsys):
    """--deep must not exist (check phase never re-runs heavy transforms)."""
    path = _profile_yaml(tmp_path)
    rc = main(["doctor", path, "--deep", "--json"])
    # argparse usage error
    assert rc != 0
    out = capsys.readouterr()
    blob = (out.out or "") + (out.err or "")
    assert "deep" in blob.lower() or "unrecognized" in blob.lower() or rc == 2


def test_doctor_type_path_flags_pretransform_clip(tmp_path, capsys):
    """Tight lims on type: figures still raise JP-VIZ-002 via pre-transform proxy."""
    csv = tmp_path / "s.csv"
    rows = ["a,b,c"] + [f"{i},{i},{i}" for i in range(0, 11)]
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    path = tmp_path / "clip.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            DataSet:
              - {{name: samples, path: {csv.name}, type: csv}}
            Figures:
              - name: p
                type: profile_2d
                data: samples
                x: {{expr: a, lim: [0, 10]}}
                y: {{expr: b, lim: [0, 10]}}
                z: {{expr: c}}
                style: [a4paper_2x1, rectcmap]
                frame:
                  ax:
                    xlim: [100, 200]
                    ylim: [100, 200]
            """
        ).lstrip(),
        encoding="utf-8",
    )
    rc = main(["doctor", str(path), "--json"])
    env = json.loads(capsys.readouterr().out)
    codes = [d.get("code") for d in env.get("diagnostics") or []]
    assert "JP-VIZ-002" in codes, env.get("diagnostics")
    assert env["ok"] is False or any(
        d.get("code") == "JP-VIZ-002" and d.get("level") == "error"
        for d in env.get("diagnostics") or []
    )
    ctx = next(
        (d.get("context") or {} for d in env.get("diagnostics") or [] if d.get("code") == "JP-VIZ-002"),
        {},
    )
    assert ctx.get("basis") == "pre-transform"


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

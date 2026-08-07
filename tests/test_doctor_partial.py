"""doctor/dryrun: heavy-transform skip is partial, not failed."""

from __future__ import annotations

import json
import textwrap

from jarvisplot.client import main


def _profile_yaml(tmp_path, *, n_cloud: int = 3) -> str:
    csv = tmp_path / "s.csv"
    if n_cloud <= 3:
        csv.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n", encoding="utf-8")
    else:
        # Non-collinear cloud so natural_neighbor / deep profile can finish cleanly.
        rows = ["a,b,c"]
        for i in range(n_cloud):
            rows.append(f"{i % 10},{i // 10},{i}")
        csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
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
                x: {{expr: a, lim: [0, 20]}}
                y: {{expr: b, lim: [0, 20]}}
                z: {{expr: c}}
                style: [a4paper_2x1, rectcmap]
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return str(path)


def test_doctor_profile_2d_is_partial_renderable_not_failed(tmp_path, capsys):
    """Default doctor is shallow: type: → partial_renderable, not failed."""
    path = _profile_yaml(tmp_path)
    rc = main(["doctor", path, "--json"])
    env = json.loads(capsys.readouterr().out)
    assert rc == 0, env
    assert env["ok"] is None
    assert env["data"]["status"] == "partial_renderable"
    assert env["data"]["coverage"] == "partial"
    assert env["data"]["renderable"] is True
    assert env["data"].get("deep") is False
    assert env["data"]["heavy_skipped"]
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


def test_dryrun_deep_type_path_flags_postmesh_clip(tmp_path, capsys):
    """Deep dryrun uses preprocessor_runtime — JP-VIZ-002 on post-mesh layers."""
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
    rc = main(["dryrun", str(path), "--deep", "--json"])
    env = json.loads(capsys.readouterr().out)
    assert env.get("data", {}).get("deep") is True
    assert not env.get("data", {}).get("heavy_skipped")
    codes = [d.get("code") for d in env.get("diagnostics") or []]
    assert "JP-VIZ-002" in codes, env.get("diagnostics")
    assert "JP-VIZ-010" not in codes
    viz = [d for d in env.get("diagnostics") or [] if d.get("code") == "JP-VIZ-002"]
    assert viz
    assert any((d.get("context") or {}).get("basis") != "pre-transform" for d in viz) or any(
        "pre-transform" not in (d.get("message") or "") for d in viz
    )


def test_doctor_deep_type_path_flags_clip(tmp_path, capsys):
    """doctor --deep matches dryrun --deep for type: clip detection."""
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
    rc = main(["doctor", str(path), "--deep", "--json"])
    env = json.loads(capsys.readouterr().out)
    assert env["data"].get("deep") is True
    codes = [d.get("code") for d in env.get("diagnostics") or []]
    assert "JP-VIZ-002" in codes, env.get("diagnostics")
    assert env["ok"] is False


def test_doctor_type_path_flags_pretransform_clip(tmp_path, capsys):
    """Default (shallow) doctor still uses pre-transform JP-VIZ-002 proxy."""
    csv = tmp_path / "s.csv"
    # data mostly in [0, 10] × [0, 10]
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
    # Should still be failed (clip is a real error), not silent partial_renderable-only
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

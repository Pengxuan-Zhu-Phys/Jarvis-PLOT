"""jplot context — aggregated agent context pack."""

from __future__ import annotations

import json

from jarvisplot.client import main


def test_context_profile_with_logl(tmp_path, capsys):
    path = tmp_path / "s.csv"
    path.write_text("m_A,tanb,LogL\n100,10,-5\n200,20,-3\n", encoding="utf-8")
    assert (
        main(
            [
                "context",
                "--data",
                str(path),
                "--kind",
                "profile_2d",
                "--json",
            ]
        )
        == 0
    )
    env = json.loads(capsys.readouterr().out)
    assert env["kind"] == "context"
    assert env["ok"] is True
    data = env["data"]
    assert data["kind"] == "profile_2d"
    assert "LogL" in data["data"]["column_names"]
    assert data["type"]["slots"]
    assert data["recommendations"].get("z") == "LogL" or data["suggest_ok"]
    assert data["transforms"]["related"]
    names = {t["name"] for t in data["transforms"]["related"]}
    assert "profile" in names
    assert "make_interp_2d" in names
    assert data["yaml_skeleton"]
    assert data["next_cli"]
    assert data["styles"]["usable"] is not None


def test_context_posterior_gap_lists_weight(tmp_path, capsys):
    path = tmp_path / "tri.csv"
    path.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")
    assert main(["context", "--data", str(path), "--kind", "posterior_2d", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is True  # context itself succeeds
    data = env["data"]
    assert data["suggest_ok"] is False
    codes = {g["code"] for g in data["gaps"]}
    assert "JP-TPL-005" in codes
    assert data["recommendations"].get("x") in {"a", "b", "c"}
    # still gives skeleton / next steps
    assert data["next_cli"]


def test_context_scatter_ok(tmp_path, capsys):
    path = tmp_path / "s.csv"
    path.write_text("x,y,c\n1,2,0.1\n3,4,0.2\n", encoding="utf-8")
    assert main(["context", "--data", str(path), "--kind", "scatter_2d", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is True
    assert env["data"]["suggest_ok"] is True
    assert "layers" in (env["data"].get("yaml_skeleton") or "") or "scatter" in (
        env["data"].get("yaml_skeleton") or ""
    )

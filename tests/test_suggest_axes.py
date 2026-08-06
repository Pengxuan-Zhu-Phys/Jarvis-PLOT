"""C6: jplot data suggest-axes."""

from __future__ import annotations

import json

import pytest

from jarvisplot.client import main
from jarvisplot.verbs.data import suggest_axes


@pytest.fixture
def decades_csv(tmp_path):
    """Column m_A spans >2 decades of positive values → log scale."""
    path = tmp_path / "params.csv"
    # 1 … 2000 roughly 3.3 decades
    rows = ["m_A,tanb,flag\n"]
    for i, v in enumerate([1.0, 10.0, 100.0, 500.0, 1000.0, 2000.0]):
        rows.append(f"{v},{1 + i},0\n")
    path.write_text("".join(rows), encoding="utf-8")
    return path


def test_suggest_log_for_wide_positive_range(decades_csv):
    data = suggest_axes(str(decades_csv))
    by_col = {a["col"]: a for a in data["axes"]}
    assert by_col["m_A"]["scale"] == "log"
    assert by_col["m_A"]["lim"] is not None
    assert by_col["m_A"]["lim"][0] < by_col["m_A"]["lim"][1]
    assert by_col["m_A"]["reason"]
    assert "decades" in by_col["m_A"]["reason"]


def test_suggest_linear_for_narrow_range(decades_csv):
    data = suggest_axes(str(decades_csv), cols="tanb")
    assert len(data["axes"]) == 1
    assert data["axes"][0]["scale"] == "linear"


def test_jplot_data_suggest_axes_json(decades_csv, capsys):
    assert main(["data", "suggest-axes", str(decades_csv), "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["kind"] == "data.suggest_axes"
    assert env["ok"] is True
    names = {a["col"] for a in env["data"]["axes"]}
    assert "m_A" in names

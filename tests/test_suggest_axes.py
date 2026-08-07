"""C6: jplot data suggest-axes."""

from __future__ import annotations

import json

import numpy as np
import pytest

from jarvisplot.client import main
from jarvisplot.verbs.data import suggest_axes


@pytest.fixture
def decades_csv(tmp_path):
    """Log-like m_A (wide + right-skewed) → log scale; tanb narrow → linear."""
    path = tmp_path / "params.csv"
    # Geometric sequence: many decades and median ≪ mean (skew_ratio ≪ 0.5).
    rows = ["m_A,tanb,flag\n"]
    for i, v in enumerate([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]):
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


def test_uniform_positive_column_stays_linear(tmp_path):
    """Uniform [0,5]-like data must not be judged log (P1.1).

    Quantile decades alone are ~2.3; shape (median≈mean) keeps linear.
    """
    path = tmp_path / "uniform.csv"
    rng = np.random.default_rng(0)
    # Strictly positive uniform (matches review repro: samples near 0 still ok)
    y = rng.uniform(1e-4, 5.0, size=5000)
    x = rng.uniform(0.0, 1.0, size=5000)
    lines = ["x,y\n"] + [f"{a},{b}\n" for a, b in zip(x, y)]
    path.write_text("".join(lines), encoding="utf-8")
    data = suggest_axes(str(path), cols="y")
    assert len(data["axes"]) == 1
    axis = data["axes"][0]
    assert axis["scale"] == "linear", axis.get("reason")
    stats = axis.get("stats") or {}
    assert stats.get("decades", 0) >= 2.0  # would have been log under min/max or q-only
    assert stats.get("median_over_mean", 0) >= 0.5


def test_jplot_data_suggest_axes_json(decades_csv, capsys):
    assert main(["data", "suggest-axes", str(decades_csv), "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["kind"] == "data.suggest_axes"
    assert env["ok"] is True
    names = {a["col"] for a in env["data"]["axes"]}
    assert "m_A" in names

"""C3–C5: describe cache, data head, data eval."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from jarvisplot.client import main
from jarvisplot import data_access as data_access_mod
from jarvisplot.data_access import describe_file, eval_on_file, head_file


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text(
        "m_A,tanb,LogL,weight\n"
        "100,10,-5.0,0.1\n"
        "200,20,-3.0,0.2\n"
        "300,30,-1.0,0.3\n"
        "400,40,-0.5,0.4\n"
        "500,50,-0.1,0.5\n"
        "600,60,0.0,0.6\n",
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------- #
# C4 head
# --------------------------------------------------------------------------- #


def test_head_default_five_rows(csv_file):
    data = head_file(str(csv_file), n=5)
    assert data["n"] == 5
    assert data["columns"] == ["m_A", "tanb", "LogL", "weight"]
    assert len(data["rows"]) == 5
    assert data["rows"][0]["m_A"] == 100


def test_jplot_data_head_json(csv_file, capsys):
    assert main(["data", "head", str(csv_file), "-n", "3", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["kind"] == "data.head"
    assert env["ok"] is True
    assert env["data"]["n"] == 3
    assert len(env["data"]["rows"]) == 3


def test_head_cols_subset(csv_file):
    data = head_file(str(csv_file), n=2, cols="m_A,LogL")
    assert data["columns"] == ["m_A", "LogL"]
    assert set(data["rows"][0]) == {"m_A", "LogL"}


# --------------------------------------------------------------------------- #
# C5 eval
# --------------------------------------------------------------------------- #


def test_eval_exp_logl(csv_file):
    data = eval_on_file("exp(LogL)", str(csv_file))
    assert data["n"] == 6
    assert data["n_finite"] == 6
    assert data["symbols_used"] == ["LogL"]
    assert data["min"] is not None
    assert data["sample"]


def test_jplot_data_eval_json(csv_file, capsys):
    assert main(["data", "eval", "exp(LogL)", "--data", str(csv_file), "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["kind"] == "data.eval"
    assert env["ok"] is True
    assert env["data"]["expr"] == "exp(LogL)"
    assert env["data"]["n_finite"] == 6


def test_eval_unknown_column_did_you_mean(csv_file, capsys):
    assert main(["data", "eval", "exp(LogLL)", "--data", str(csv_file), "--json"]) == 1
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is False
    assert env["error"]["type"] == "JP-EXP-002"
    assert "LogL" in env["data"]["did_you_mean"] or "LogL" in env["data"]["available_columns"]
    assert env["diagnostics"]
    assert env["diagnostics"][0]["code"] == "JP-EXP-002"


def test_eval_np_log10(csv_file):
    data = eval_on_file("np.log10(m_A)", str(csv_file))
    assert data["n_finite"] == 6
    assert abs(data["min"] - 2.0) < 1e-9  # log10(100)


# --------------------------------------------------------------------------- #
# C3 describe cache
# --------------------------------------------------------------------------- #


def test_describe_cache_hit(csv_file, monkeypatch):
    # first call populates cache
    first = describe_file(str(csv_file), use_cache=True)
    assert first["cache"] == "miss"

    calls = {"n": 0}
    real_load = data_access_mod.load_dataframe

    def counting_load(*args, **kwargs):
        calls["n"] += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(data_access_mod, "load_dataframe", counting_load)
    second = describe_file(str(csv_file), use_cache=True)
    assert second["cache"] == "hit"
    assert calls["n"] == 0
    assert second["rows"] == first["rows"]
    assert [c["name"] for c in second["columns"]] == [c["name"] for c in first["columns"]]


def test_describe_no_cache_flag(csv_file, capsys):
    assert main(["data", "describe", str(csv_file), "--json"]) == 0
    env1 = json.loads(capsys.readouterr().out)
    assert main(["data", "describe", str(csv_file), "--json", "--no-cache"]) == 0
    env2 = json.loads(capsys.readouterr().out)
    assert env2["data"]["cache"] == "miss"
    assert env2["data"]["rows"] == env1["data"]["rows"]

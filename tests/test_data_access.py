"""P1.3: agent/dryrun load through data_access, not a verbs fork."""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd

from jarvisplot import data_access
from jarvisplot.data_access import detect_type, load_dataframe, resolve_data_path


def test_dryrun_runtime_does_not_import_verbs_data():
    import jarvisplot.dryrun_runtime as dryrun_runtime

    src = inspect.getsource(dryrun_runtime)
    assert "verbs.data" not in src
    assert "from .data_access import" in src or "data_access" in src


def test_load_dataframe_csv_full_uses_dataset(tmp_path, monkeypatch):
    path = tmp_path / "s.csv"
    path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    calls: list[str] = []

    real = data_access._load_via_dataset

    def wrapped(*args, **kwargs):
        calls.append("dataset")
        return real(*args, **kwargs)

    monkeypatch.setattr(data_access, "_load_via_dataset", wrapped)
    df = load_dataframe(str(path), kind="csv")
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2
    assert calls == ["dataset"]


def test_load_dataframe_csv_nrows_skips_full_dataset(tmp_path, monkeypatch):
    path = tmp_path / "s.csv"
    path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")

    def boom(*args, **kwargs):
        raise AssertionError("full DataSet path must not run for nrows sample")

    monkeypatch.setattr(data_access, "_load_via_dataset", boom)
    df = load_dataframe(str(path), kind="csv", nrows=2)
    assert len(df) == 2


def test_detect_type_and_resolve(tmp_path):
    path = tmp_path / "x.parquet"
    path.write_bytes(b"PAR1" + b"\0" * 8)
    assert detect_type(str(path), "auto") == "parquet"
    resolved = resolve_data_path(str(path))
    assert Path(resolved).is_file()


def test_verbs_data_is_skin_reexport():
    from jarvisplot.verbs import data as verbs_data
    from jarvisplot import data_access as da

    assert verbs_data.describe_file is da.describe_file
    assert verbs_data.head_file is da.head_file
    assert verbs_data.eval_on_file is da.eval_on_file
    assert verbs_data.suggest_axes is da.suggest_axes

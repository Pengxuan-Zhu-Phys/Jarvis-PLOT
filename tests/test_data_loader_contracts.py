from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from jarvisplot.data_loader import DataSet
from jarvisplot import data_loader_summary
from jarvisplot import data_loader_runtime

try:
    import polars as pl
except Exception:
    pl = None


def _logger():
    return SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )


def test_dataset_setters_return_early_on_none():
    ds = DataSet()
    ds.logger = _logger()

    ds.file = None
    ds.type = None

    assert ds.file is None
    assert ds.type is None


def test_hdf5_fallback_loads_single_dataset(tmp_path):
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("values", data=np.array([[1, 2], [3, 4]], dtype=np.int64))

    ds = DataSet()
    ds.logger = _logger()
    ds.setinfo({"path": str(path), "name": "sample", "type": "hdf5"}, rootpath=str(tmp_path), eager=False, cache=None)
    ds.group = None

    data = ds.load()

    assert data is ds.data
    assert list(ds.data.columns) == [0, 1]
    assert ds.keys == [0, 1]


def test_dataset_load_hdf5_delegates_to_runtime_helper(monkeypatch):
    ds = DataSet()
    ds.logger = _logger()

    called = {}

    def fake_load_hdf5(dataset):
        called["dataset"] = dataset
        dataset.data = pd.DataFrame({"x": [1]})
        dataset.keys = list(dataset.data.columns)
        return "delegated"

    monkeypatch.setattr(data_loader_runtime, "load_hdf5", fake_load_hdf5)

    out = ds.load_hdf5()

    assert out == "delegated"
    assert called["dataset"] is ds
    assert list(ds.data.columns) == ["x"]


def test_materialized_summary_includes_numeric_bounds():
    ds = DataSet()
    ds.logger = _logger()
    ds.name = "sample"
    ds.type = "hdf5"
    ds._materialized_manifest = {"columns": ["a", "b", "c"], "rows": 2, "cols": 3, "parts": 1, "bytes_total": 0}
    ds.data = pd.DataFrame({"a": [1.0, 3.0], "b": ["x", "y"], "c": [2, 5]})

    stats = data_loader_runtime.materialized_numeric_bounds(ds)
    msg = ds._materialized_summary(ds._materialized_manifest, stats=stats)

    assert stats == {"a": {"min": 1.0, "max": 3.0}, "c": {"min": 2.0, "max": 5.0}}
    assert "=== Materialized Stats ===" in msg
    assert "name" in msg
    assert "dtype" in msg
    assert "nonnull%" in msg
    assert "[min]" in msg
    assert "[max]" in msg
    assert "a" in msg
    assert "1" in msg


def test_dataframe_summary_table_uses_scientific_notation_and_uniqs():
    df = pd.DataFrame(
        {
            "x": [1e-7, 2e7],
            "uuid": ["abc", "def"],
            "y": [0.0001128, 5.0],
        }
    )

    msg = data_loader_summary.dataframe_summary(df, name=" CSV loaded!")

    assert "=== DataFrame Summary Table ===" in msg
    assert "name" in msg
    assert "dtype" in msg
    assert "nonnull%" in msg
    assert "[min]" in msg
    assert "[max]" in msg
    assert "uniq=2" in msg
    assert "1e-07" in msg or "1e-07" in msg
    assert "2e+07" in msg or "2e+07" in msg
    assert "0.0001128" in msg


def test_dataset_emit_summary_uses_summary_helper(monkeypatch):
    ds = DataSet()
    ds.logger = _logger()
    ds.name = "sample"
    ds.type = "csv"
    ds.data = pd.DataFrame({"x": [1]})

    called = {}

    def fake_dataframe_summary(df, name=""):
        called["name"] = name
        called["shape"] = df.shape
        return "summary!"

    emitted = []

    monkeypatch.setattr(data_loader_summary, "dataframe_summary", fake_dataframe_summary)
    monkeypatch.setattr(ds, "_emit_summary_text", lambda msg: emitted.append(msg))

    ok = ds.emit_summary(force_load=False)

    assert ok is True
    assert called["shape"] == (1, 1)
    assert emitted == ["summary!"]
    assert called["name"].startswith(" CSV loaded!")


def test_dataset_emit_summary_uses_warning_level(monkeypatch):
    ds = DataSet()
    ds.logger = _logger()
    ds.name = "sample"
    ds.type = "csv"

    calls = []

    class L:
        def warning(self, msg):
            calls.append(str(msg))

        def info(self, *args, **kwargs):
            raise AssertionError("info should not be used for summary emission")

    ds.logger = L()
    ds._emit_summary_text("summary!")

    assert calls == ["\nsummary!"]


def test_hdf5_summary_emitted_before_transform(tmp_path):
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("x", data=np.array([1.0, 2.0, 3.0], dtype=np.float64))

    ds = DataSet()
    ds.logger = _logger()
    ds.setinfo(
            {
                "path": str(path),
                "name": "sample",
                "type": "hdf5",
                "dataset": None,
                "columns": {},
                "transform": [{"add_column": {"name": "y", "expr": "x * 2"}}],
            },
        rootpath=str(tmp_path),
        eager=False,
        cache=None,
    )

    emitted = []
    ds._emit_summary_text = lambda msg: emitted.append(msg)

    data_loader_runtime.load_hdf5(ds)

    assert emitted
    assert "\nx" in emitted[0] or emitted[0].splitlines()[0] == " HDF5 loaded!"
    assert "\ny" not in emitted[0].splitlines()[-1]
    assert "y" in ds.data.columns


def test_hdf5_transform_tocsv_writes_output(tmp_path):
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("x", data=np.array([1.0, 2.0, 3.0], dtype=np.float64))

    ds = DataSet()
    ds.logger = _logger()
    ds.setinfo(
        {
            "path": str(path),
            "name": "sample",
            "type": "hdf5",
            "dataset": None,
            "transform": [
                {"add_column": {"name": "y", "expr": "x * 2"}},
                {"tocsv": "./saved/sample_transformed.csv"},
            ],
        },
        rootpath=str(tmp_path),
        eager=False,
        cache=None,
    )

    data_loader_runtime.load_hdf5(ds)

    out_csv = tmp_path / "saved" / "sample_transformed.csv"
    assert out_csv.exists()
    saved = pd.read_csv(out_csv)
    assert "y" in saved.columns
    assert "x" in saved.columns
    assert "y" in ds.data.columns


def test_hdf5_transform_tocsv_executes_in_order(tmp_path):
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("x", data=np.array([1.0, 2.0, 3.0], dtype=np.float64))

    ds = DataSet()
    ds.logger = _logger()
    ds.setinfo(
        {
            "path": str(path),
            "name": "sample",
            "type": "hdf5",
            "dataset": None,
            "transform": [
                {"add_column": {"name": "y", "expr": "x * 2"}},
                {"tocsv": "./saved/sample_prefix.csv"},
                {"add_column": {"name": "z", "expr": "y + 1"}},
            ],
        },
        rootpath=str(tmp_path),
        eager=False,
        cache=None,
    )

    data_loader_runtime.load_hdf5(ds)

    out_csv = tmp_path / "saved" / "sample_prefix.csv"
    assert out_csv.exists()
    saved = pd.read_csv(out_csv)
    assert "y" in saved.columns
    assert "z" not in saved.columns
    assert "z" in ds.data.columns


def test_hdf5_transform_to_parquet_writes_output(tmp_path):
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("x", data=np.array([1.0, 2.0, 3.0], dtype=np.float64))

    ds = DataSet()
    ds.logger = _logger()
    ds.setinfo(
        {
            "path": str(path),
            "name": "sample",
            "type": "hdf5",
            "dataset": None,
            "transform": [
                {"add_column": {"name": "y", "expr": "x * 2"}},
                {"to_parquet": "./saved/sample_transformed.parquet"},
            ],
        },
        rootpath=str(tmp_path),
        eager=False,
        cache=None,
    )

    data_loader_runtime.load_hdf5(ds)

    out_parquet = tmp_path / "saved" / "sample_transformed.parquet"
    assert out_parquet.exists()
    if pl is not None:
        saved = pl.read_parquet(out_parquet)
        assert "y" in saved.columns
        assert "x" in saved.columns
        assert saved.height == 3
    else:
        saved = pd.read_parquet(out_parquet)
        assert "y" in saved.columns
        assert "x" in saved.columns
        assert len(saved) == 3
    assert "y" in ds.data.columns


def test_hdf5_transform_to_parquet_executes_in_order(tmp_path):
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("x", data=np.array([1.0, 2.0, 3.0], dtype=np.float64))

    ds = DataSet()
    ds.logger = _logger()
    ds.setinfo(
        {
            "path": str(path),
            "name": "sample",
            "type": "hdf5",
            "dataset": None,
            "transform": [
                {"add_column": {"name": "y", "expr": "x * 2"}},
                {"to_parquet": "./saved/sample_prefix.parquet"},
                {"add_column": {"name": "z", "expr": "y + 1"}},
            ],
        },
        rootpath=str(tmp_path),
        eager=False,
        cache=None,
    )

    data_loader_runtime.load_hdf5(ds)

    out_parquet = tmp_path / "saved" / "sample_prefix.parquet"
    assert out_parquet.exists()
    if pl is not None:
        saved = pl.read_parquet(out_parquet)
        assert "y" in saved.columns
        assert "z" not in saved.columns
    else:
        saved = pd.read_parquet(out_parquet)
        assert "y" in saved.columns
        assert "z" not in saved.columns
    assert "z" in ds.data.columns


def test_parquet_dataset_load_emits_summary(tmp_path):
    if pl is None:
        pytest.skip("polars is required for parquet fixture generation")

    path = tmp_path / "sample.parquet"
    pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}).write_parquet(path)

    ds = DataSet()
    ds.logger = _logger()
    ds.setinfo({"path": str(path), "name": "sample", "type": "parquet"}, rootpath=str(tmp_path), eager=False, cache=None)

    emitted = []
    ds._emit_summary_text = lambda msg: emitted.append(msg)

    data = ds.load()

    assert data is ds.data
    assert list(ds.data.columns) == ["x", "y"]
    assert ds.keys == ["x", "y"]
    assert emitted
    assert "Parquet loaded!" in emitted[0]
    assert "=== DataFrame Summary Table ===" in emitted[0]


def test_hdf5_transform_tocsv_and_to_parquet_writes_both_outputs(tmp_path):
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("x", data=np.array([1.0, 2.0, 3.0], dtype=np.float64))

    ds = DataSet()
    ds.logger = _logger()
    ds.setinfo(
        {
            "path": str(path),
            "name": "sample",
            "type": "hdf5",
            "dataset": None,
            "transform": [
                {"add_column": {"name": "y", "expr": "x * 2"}},
                {"tocsv": "./saved/sample_transformed.csv"},
                {"to_parquet": "./saved/sample_transformed.parquet"},
            ],
        },
        rootpath=str(tmp_path),
        eager=False,
        cache=None,
    )

    data_loader_runtime.load_hdf5(ds)

    out_csv = tmp_path / "saved" / "sample_transformed.csv"
    out_parquet = tmp_path / "saved" / "sample_transformed.parquet"
    assert out_csv.exists()
    assert out_parquet.exists()
    saved_csv = pd.read_csv(out_csv)
    assert "x" in saved_csv.columns
    assert "y" in saved_csv.columns
    if pl is not None:
        saved_parquet = pl.read_parquet(out_parquet)
        assert "x" in saved_parquet.columns
        assert "y" in saved_parquet.columns
    else:
        saved_parquet = pd.read_parquet(out_parquet)
        assert "x" in saved_parquet.columns
        assert "y" in saved_parquet.columns


def test_transform_does_not_prune_columns_without_explicit_crop():
    ds = DataSet()
    ds.logger = _logger()
    ds.name = "sample"
    ds.type = "hdf5"
    ds.data = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    ds.transform = [{"add_column": {"name": "y", "expr": "x * 2"}}]
    ds.retained_columns = {"x"}

    data_loader_runtime.apply_dataset_transform(ds, stage="hdf5")

    assert "x" in ds.data.columns
    assert "y" in ds.data.columns
    assert "__jp_row_idx__" in ds.data.columns


def test_keep_and_drop_columns_are_explicit_transform_steps():
    ds = DataSet()
    ds.logger = _logger()
    ds.name = "sample"
    ds.type = "csv"
    ds.data = pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
    ds.transform = [
        {"keep_columns": ["x", "y", "missing"]},
        {"drop_columns": ["y"]},
    ]

    data_loader_runtime.apply_dataset_transform(ds, stage="csv")

    assert list(ds.data.columns) == ["__jp_row_idx__", "x"]

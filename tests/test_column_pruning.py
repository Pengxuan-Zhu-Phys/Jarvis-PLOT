"""Datasets read only the columns the config can ask for."""

from __future__ import annotations

from types import SimpleNamespace

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml
from loguru import logger

from jarvisplot.cache_store import ProjectCache
from jarvisplot.core import JarvisPLOT
from jarvisplot.core_runtime import COLUMN_PRUNE_ENV, plan_dataset_required_columns
from jarvisplot.data_loader import JP_ROW_IDX, DataSet
from jarvisplot.Figure.preprocessor import DataPreprocessor

pl = pytest.importorskip("polars")


class _Log:
    def __init__(self):
        self.messages: list[str] = []

    def __getattr__(self, level):
        if level in {"debug", "info", "warning", "error"}:
            return lambda msg, *a, **k: self.messages.append(str(msg))
        raise AttributeError(level)


WIDE = {f"c{i}": np.arange(6, dtype=float) + i for i in range(20)}
WIDE.update({"x": np.linspace(0, 1, 6), "y": np.linspace(1, 2, 6), "Var0@scan": np.arange(6.0)})


def _dataset(name, transform=None):
    ds = DataSet()
    ds.name = name
    ds.transform = transform
    return ds


def _plan(layers, datasets):
    config = {"Figures": [{"name": "f", "layers": layers}]}
    registry = {ds.name: ds for ds in datasets}
    core = SimpleNamespace(
        yaml=SimpleNamespace(config=config),
        dataset=datasets,
        logger=_Log(),
        preprocessor=DataPreprocessor(context=None, dataset_registry=registry, logger=_Log()),
    )
    plan_dataset_required_columns(core)
    return core


def _scatter(source, *, x="x", y="y", transform=None, method="scatter"):
    entry = {"source": source}
    if transform is not None:
        entry["transform"] = transform
    return {
        "name": "pts",
        "method": method,
        "axes": "ax",
        "data": [entry],
        "coordinates": {"x": {"expr": x}, "y": {"expr": y}},
    }


def test_plan_covers_layer_expressions_transforms_and_raw_column_names():
    ds = _dataset("scan", transform=[{"add_column": {"name": "r", "expr": "c1 / c2"}}])
    _plan([_scatter("scan", y="Var0@scan", transform=[{"filter": "c3 > 0 && c4 < 9"}])], [ds])

    for column in ("x", "Var0@scan", "c3", "c4", "r", JP_ROW_IDX):
        assert column in ds.retained_columns
    assert {"c1", "c2"} <= ds.required_columns
    assert "c5" not in ds.required_columns


@pytest.mark.parametrize(
    "layer, transform",
    [
        (_scatter("scan", method="dynesty_runplot"), None),
        (_scatter("scan"), [{"to_csv": "out.csv"}]),
        (_scatter("scan", transform=[{"correlation": {"regex": "^c"}}]), None),
    ],
    ids=["by-name-method", "dataset-export", "dynamic-correlation"],
)
def test_sources_whose_needs_are_not_names_load_whole(layer, transform):
    ds = _dataset("scan", transform=transform)
    _plan([layer], [ds])
    assert ds.required_columns is None and ds.retained_columns is None


def test_by_name_method_on_an_untracked_table_disables_pruning_everywhere():
    scan, other = _dataset("scan"), _dataset("other")
    _plan(
        [_scatter("scan"), _scatter("shared_table", method="dynesty_runplot")],
        [scan, other],
    )
    assert scan.required_columns is None and other.required_columns is None


def test_dynamic_step_on_a_published_table_disables_pruning_everywhere():
    # `tbl` comes from `scan` via to_df, but that lineage is not tracked: a
    # correlation over "every numeric column" of it needs `scan` whole.
    scan = _dataset("scan")
    publish = _scatter("scan", x="a", y="b", transform=[{"to_df": "tbl"}])
    correlate = {
        "name": "corr",
        "method": "corrplot",
        "axes": "ax",
        "data": [{"source": "tbl", "transform": [{"correlation": {}}]}],
    }

    _plan([publish, correlate], [scan])

    assert scan.required_columns is None


def test_environment_switch_turns_pruning_off(monkeypatch):
    monkeypatch.setenv(COLUMN_PRUNE_ENV, "0")
    ds = _dataset("scan")
    _plan([_scatter("scan")], [ds])
    assert ds.required_columns is None


def _wide_csv(tmp_path):
    pd.DataFrame(WIDE).to_csv(tmp_path / "wide.csv", index=False)
    return tmp_path / "wide.csv"


def test_csv_reads_only_planned_columns(tmp_path):
    _wide_csv(tmp_path)
    ds = DataSet()
    ds.logger = _Log()
    ds.setinfo({"name": "scan", "type": "csv", "path": "wide.csv"}, rootpath=str(tmp_path))
    ds.set_required_columns({"x", "c7", "Var0@scan", JP_ROW_IDX})

    data = ds.get_data()

    assert list(data.columns) == ["c7", "x", "Var0@scan"]
    assert any("reading 3 of 23 columns" in msg for msg in ds.logger.messages)


def test_plan_that_matches_no_column_is_not_trusted(tmp_path):
    _wide_csv(tmp_path)
    ds = DataSet()
    ds.logger = _Log()
    ds.setinfo({"name": "scan", "type": "csv", "path": "wide.csv"}, rootpath=str(tmp_path))
    ds.set_required_columns({"nothing_like_this", JP_ROW_IDX})

    assert len(ds.get_data().columns) == len(WIDE)


def test_parquet_reads_only_planned_columns(tmp_path):
    pl.DataFrame({k: v for k, v in WIDE.items()}).write_parquet(tmp_path / "wide.parquet")
    ds = DataSet()
    ds.logger = _Log()
    ds.setinfo({"name": "scan", "type": "parquet", "path": "wide.parquet"}, rootpath=str(tmp_path))
    ds.set_required_columns({"y", "c0"})

    assert sorted(ds.get_data().columns) == ["c0", "y"]


def test_hdf5_pushdown_collects_retained_columns_and_recovers_the_rest(tmp_path):
    with h5py.File(tmp_path / "scan.h5", "w") as h5:
        group = h5.create_group("data")
        for name in ("x", "y", "c1", "c2"):
            group.create_dataset(name, data=np.asarray(WIDE[name]))
    ds = DataSet()
    ds.logger = _Log()
    ds.setinfo(
        {
            "name": "scan",
            "type": "hdf5",
            "path": "scan.h5",
            "dataset": "data",
            "columns": {"rename": [{"source": f"data/{n}", "target": n} for n in ("x", "y", "c1", "c2")]},
            "transform": [{"filter": "x > 0.1"}],
        },
        rootpath=str(tmp_path),
        cache=ProjectCache(str(tmp_path)),
    )
    ds.set_required_columns({"x", "y", JP_ROW_IDX}, retained={"x", "y", JP_ROW_IDX})

    data = ds.get_data()

    assert sorted(data.columns) == sorted([JP_ROW_IDX, "x", "y"])
    recovered = ds.fetch_rows_columns(data[JP_ROW_IDX].to_numpy(), ["c2"])
    np.testing.assert_array_equal(recovered["c2"].to_numpy(), WIDE["c2"][WIDE["x"] > 0.1])


def test_unprojected_pipeline_key_follows_the_column_plan():
    ds = _dataset("scan")
    pre = DataPreprocessor(context=None, dataset_registry={"scan": ds}, logger=_Log())
    published = [{"to_df": "table"}]

    ds.set_required_columns({"x"})
    key_x = pre._pipeline_key("scan", published, projection=None)
    projected_x = pre._pipeline_key("scan", None, projection=["x"])
    ds.set_required_columns({"x", "y"})

    assert pre._pipeline_key("scan", published, projection=None) != key_x
    assert pre._pipeline_key("scan", None, projection=["x"]) == projected_x


@pytest.fixture
def _clean_render_state():
    plt.close("all")
    yield
    plt.close("all")
    logger.remove()


def _render(workdir, monkeypatch, *, prune: bool):
    workdir.mkdir()
    _wide_csv(workdir)
    name = "pruned" if prune else "whole"
    config = {
        "DataSet": [{"name": "scan", "type": "csv", "path": "wide.csv"}],
        "Figures": [
            {
                "name": name,
                "style": ["a4paper_2x1", "rect"],
                "layers": [_scatter("scan", transform=[{"filter": "c3 > 4 && c4 < 9"}])],
            }
        ],
        "output": {"dir": "./plots", "formats": ["png"], "dpi": 40},
    }
    (workdir / "plot.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv(COLUMN_PRUNE_ENV, "on" if prune else "0")
    JarvisPLOT(argv=[str(workdir / "plot.yaml")]).init()
    return plt.imread(workdir / "plots" / f"{name}.png")


def test_pruned_render_matches_the_unpruned_one(tmp_path, monkeypatch, _clean_render_state):
    plans: list[tuple[int, int]] = []
    monkeypatch.setattr(DataSet, "_log_column_plan", lambda self, kept, total: plans.append((kept, total)))

    pruned = _render(tmp_path / "pruned", monkeypatch, prune=True)
    assert plans == [(4, 23)]  # x, y, c3, c4
    whole = _render(tmp_path / "whole", monkeypatch, prune=False)
    assert plans == [(4, 23)]

    np.testing.assert_array_equal(pruned, whole)

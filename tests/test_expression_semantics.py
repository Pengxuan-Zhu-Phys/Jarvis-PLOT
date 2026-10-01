"""One expression language: the pandas and polars engines must agree.

Dataset-level transforms on HDF5 run as a polars pushdown; everything else runs
on pandas. A filter is only trustworthy if it keeps the same rows on both.
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

from jarvisplot.cache_store import ProjectCache
from jarvisplot.data_loader import JP_ROW_IDX, DataSet
from jarvisplot.Figure.preprocessor_runtime import add_column, filter_df
from jarvisplot.utils import expression as expression_module
from jarvisplot.utils.expression import compile_dataframe_expression, eval_dataframe_expression

pl = pytest.importorskip("polars")
from jarvisplot.utils.expression import polars_expression  # noqa: E402


class _Log:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def _add(self, level):
        return lambda msg, *a, **k: self.messages.append((level, str(msg)))

    def __getattr__(self, level):
        if level in {"debug", "info", "warning", "error"}:
            return self._add(level)
        raise AttributeError(level)

    def at(self, level):
        return [msg for lvl, msg in self.messages if lvl == level]


DATA = {
    "x": [1.0, 2.0, 3.0, 4.0, np.nan, 0.5],
    "y": [4, 3, 2, 1, 5, 6],
    "tag": ["a", "b&&c", "c", "d", "e", "f"],
}

FILTERS = [
    "x > 1 && y > 1",
    "x > 1 and y > 1",
    "(x > 1) & (y > 1)",
    "x > 1 || y == 6",
    "not x > 2",
    "not (x > 2) && y < 5",
    "1 < x < 4",
    "x**2 > 4",
    "abs(x - y) <= 1",
    "log10(x) > 0.4",
    "x > np.nanmax(x) - 2",
    "x != x",
    "tag == 'b&&c'",
    "x > 1 & y > 1",
]


def _pandas_rows(expr):
    out = filter_df(pd.DataFrame(DATA), expr, _Log())
    return out["y"].tolist()


def _polars_rows(expr):
    lf = pl.DataFrame(DATA).lazy()
    mask = polars_expression(expr, lf.collect_schema(), as_mask=True, logger=_Log())
    return lf.filter(mask).collect()["y"].to_list()


@pytest.mark.parametrize("expr", FILTERS)
def test_filter_keeps_the_same_rows_on_pandas_and_polars(expr):
    assert _polars_rows(expr) == _pandas_rows(expr)


@pytest.mark.parametrize(
    ("expr", "rows"),
    [
        ("x > 1 && y > 1", [3, 2]),
        ("x > 1 || y == 6", [3, 2, 1, 6]),
        ("1 < x < 4", [3, 2]),
        ("not x > 2", [4, 3, 5, 6]),
        # numpy compares NaN as False; polars alone would order it above 2.
        ("x > 2", [2, 1]),
        ("tag == 'b&&c'", [3]),
    ],
)
def test_logical_operators_are_elementwise(expr, rows):
    assert _pandas_rows(expr) == rows


def test_double_ampersand_inside_a_string_literal_is_left_alone():
    compiled = compile_dataframe_expression("tag == 'b&&c' && y > 1")
    assert compiled.names == frozenset({"tag", "y"})
    assert _pandas_rows("tag == 'b&&c' && y > 1") == [3]


def test_bitwise_precedence_trap_is_reported_once(monkeypatch):
    monkeypatch.setattr(expression_module, "_precedence_reported", set())
    log = _Log()
    frame = pd.DataFrame(DATA)

    eval_dataframe_expression(frame, "x > 1 & y > 1", logger=log)
    eval_dataframe_expression(frame, "x > 1 & y > 1", logger=log)

    warnings = log.at("warning")
    assert len(warnings) == 1
    assert "'x > (1 & y) > 1'" in warnings[0]


@pytest.mark.parametrize("expr", ["(x > 1) & (y > 1)", "(y & 4) > 0", "x > 1 && y > 1"])
def test_parenthesised_or_logical_forms_are_not_flagged(expr):
    assert compile_dataframe_expression(expr).precedence_hint is None


@pytest.mark.parametrize("expr", ["x * 2 + y", "y > 2", "np.where(y > 2, x, -x)", "x - np.nanmax(x)"])
def test_add_column_matches_on_pandas_and_polars(expr):
    pandas_out = add_column(pd.DataFrame(DATA), {"name": "z", "expr": expr}, _Log())["z"].to_numpy()
    lf = pl.DataFrame(DATA).lazy()
    polars_out = (
        lf.with_columns(polars_expression(expr, lf.collect_schema()).alias("z")).collect()["z"].to_numpy()
    )
    np.testing.assert_array_equal(polars_out, pandas_out)


def test_unknown_name_is_refused_for_pushdown():
    lf = pl.DataFrame(DATA).lazy()
    with pytest.raises(NameError):
        polars_expression("nope > 1", lf.collect_schema(), as_mask=True)


def _write_scan(tmp_path):
    frame = pd.DataFrame({key: DATA[key] for key in ("x", "y")})
    with h5py.File(tmp_path / "scan.h5", "w") as h5:
        group = h5.create_group("data")
        group.create_dataset("x", data=frame["x"].to_numpy(dtype=float))
        group.create_dataset("y", data=frame["y"].to_numpy(dtype=np.int64))
    frame.to_csv(tmp_path / "scan.csv", index=False)


TRANSFORM = [
    {"add_column": {"name": "r", "expr": "x / y"}},
    {"filter": "x > 1 && not y > 3"},
    {"sortby": "-r"},
]


def _load(tmp_path, kind, log):
    entry = {"name": f"scan_{kind}", "type": kind, "path": f"scan.{'h5' if kind == 'hdf5' else 'csv'}"}
    if kind == "hdf5":
        entry["dataset"] = "data"
        entry["columns"] = {
            "rename": [
                {"source": "data/x", "target": "x"},
                {"source": "data/y", "target": "y"},
            ]
        }
    entry["transform"] = TRANSFORM
    ds = DataSet()
    ds.logger = log
    ds.setinfo(entry, rootpath=str(tmp_path), eager=False, cache=ProjectCache(str(tmp_path), logger=log))
    return ds.get_data()


def test_hdf5_pushdown_and_csv_pandas_agree_end_to_end(tmp_path):
    _write_scan(tmp_path)
    log = _Log()

    from_hdf5 = _load(tmp_path, "hdf5", log)
    from_csv = _load(tmp_path, "csv", _Log())

    assert not any("pushdown" in msg and "fail" in msg for msg in log.at("warning")), log.at("warning")
    assert any("polars->pandas" in msg for msg in log.at("warning"))
    columns = ["x", "y", "r"]
    pd.testing.assert_frame_equal(
        from_hdf5[columns].reset_index(drop=True),
        from_csv[columns].reset_index(drop=True),
        check_dtype=False,
    )
    assert from_hdf5["y"].tolist() == [1, 2, 3]
    assert JP_ROW_IDX in from_hdf5.columns

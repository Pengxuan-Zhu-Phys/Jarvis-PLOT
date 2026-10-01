from __future__ import annotations

import pandas as pd
import pytest

from jarvisplot.Figure.preprocessor_runtime import add_column


class _Log:
    def __init__(self):
        self.errors: list[str] = []

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg))

    def debug(self, *args, **kwargs):
        return None

    warning = info = debug


def _frame():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0]})


def test_add_column_adds_the_evaluated_column():
    log = _Log()
    out = add_column(_frame(), {"name": "y", "expr": "x * 2"}, log)

    assert out["y"].tolist() == [2.0, 4.0, 6.0]
    assert log.errors == []


@pytest.mark.parametrize(
    ("spec", "missing"),
    [
        ({"name": "y"}, "expr"),
        ({"name": "y", "expr": "  "}, "expr"),
        ({"expr": "x * 2"}, "name"),
        ({}, "name and expr"),
    ],
)
def test_add_column_without_name_or_expr_is_logged_and_skipped(spec, missing):
    log = _Log()
    out = add_column(_frame(), spec, log)

    assert list(out.columns) == ["x"]
    assert len(log.errors) == 1
    assert f"missing {missing}" in log.errors[0]


def test_add_column_leaves_the_input_frame_alone():
    source = _frame()

    out = add_column(source, {"name": "y", "expr": "x + 1"}, _Log())

    assert "y" in out.columns
    assert list(source.columns) == ["x"]

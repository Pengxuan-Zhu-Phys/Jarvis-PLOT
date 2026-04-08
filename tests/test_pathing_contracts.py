from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from jarvisplot.Figure.figure import Figure
from jarvisplot.Figure.layer_runtime import load_bool_df
from jarvisplot.core import JarvisPLOT
from jarvisplot.data_loader import DataSet
from jarvisplot.utils.pathing import resolve_project_path


class _Recorder:
    def __init__(self):
        self.messages: list[str] = []

    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, message, *args, **kwargs):
        self.messages.append(str(message))


def test_path_resolution_is_consistent_across_owners(tmp_path):
    yaml_dir = tmp_path / "yaml"
    yaml_dir.mkdir()

    rel_path = "assets/example.txt"
    internal_path = "&JP/jarvisplot/cards/style_preference.json"

    fig = Figure()
    fig._yaml_dir = str(yaml_dir)

    ds = DataSet()
    ds.logger = SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None)

    ds.setinfo(
        {"path": rel_path, "name": "sample", "type": "csv"},
        rootpath=str(yaml_dir),
        eager=False,
        cache=None,
    )

    expected_rel = (yaml_dir / rel_path).resolve()
    expected_internal = resolve_project_path(internal_path)

    assert resolve_project_path(rel_path, base_dir=yaml_dir) == expected_rel
    assert Path(fig.load_path(rel_path, base_dir=str(yaml_dir))) == expected_rel
    assert Path(ds.file) == expected_rel

    jp = JarvisPLOT()
    assert jp.load_path(internal_path) == expected_internal
    assert Path(fig.load_path(internal_path, base_dir=str(yaml_dir))) == expected_internal


def test_load_bool_df_invalid_transform_logs_and_returns_input_df():
    fig = Figure()
    recorder = _Recorder()
    fig.logger = recorder
    fig.preprocessor = None

    df = pd.DataFrame({"x": [1, 2, 3]})
    expected = df.copy(deep=True)

    out = load_bool_df(fig, df, {"filter": {"expr": "x > 1"}})

    pd.testing.assert_frame_equal(out, expected)
    assert recorder.messages
    assert "illegal transform format" in recorder.messages[0]


def test_load_bool_df_to_csv_writes_output(tmp_path):
    fig = Figure()
    fig.logger = _Recorder()
    fig.preprocessor = None
    fig._yaml_dir = str(tmp_path)

    df = pd.DataFrame({"x": [1, 2, 3]})
    expected = pd.DataFrame({"x": [1, 2, 3], "y": [2, 3, 4], "z": [3, 4, 5]})

    out = load_bool_df(
        fig,
        df,
        [
            {"add_column": {"name": "y", "expr": "x + 1"}},
            {"to_csv": "./saved/fallback_export.csv"},
            {"add_column": {"name": "z", "expr": "y + 1"}},
        ],
    )

    out_csv = tmp_path / "saved" / "fallback_export.csv"
    saved = pd.read_csv(out_csv)

    pd.testing.assert_frame_equal(out, expected)
    assert list(saved.columns) == ["x", "y"]
    assert list(out.columns) == ["x", "y", "z"]

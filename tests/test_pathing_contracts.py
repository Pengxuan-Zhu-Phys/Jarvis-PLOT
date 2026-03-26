from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from jarvisplot.Figure.figure import Figure
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

    out = fig.load_bool_df(df, {"filter": {"expr": "x > 1"}})

    assert out is df
    assert recorder.messages
    assert "illegal transform format" in recorder.messages[0]

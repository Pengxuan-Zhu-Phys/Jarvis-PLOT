"""Whole-run behaviour of ``jplot <yaml>``: what a render leaves behind."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml
from loguru import logger

from jarvisplot.core import JarvisPLOT


@pytest.fixture(autouse=True)
def _reset_logging_and_figures():
    plt.close("all")
    yield
    plt.close("all")
    logger.remove()


def _scatter_figure(name: str, *, style=("a4paper_2x1", "rect"), axes="ax", y="y") -> dict:
    return {
        "name": name,
        "style": list(style) if isinstance(style, tuple) else style,
        "layers": [
            {
                "name": "points",
                "method": "scatter",
                "axes": axes,
                "data": [{"source": "samples"}],
                "coordinates": {"x": {"expr": "x"}, "y": {"expr": y}},
            }
        ],
    }


def _render(tmp_path: Path, figures: list[dict], *options: str, formats=None) -> int:
    rng = np.random.default_rng(1)
    if not (tmp_path / "samples.csv").exists():
        pd.DataFrame({"x": rng.normal(size=50), "y": rng.normal(size=50)}).to_csv(
            tmp_path / "samples.csv", index=False
        )
    config = {
        "DataSet": [{"name": "samples", "type": "csv", "path": "samples.csv"}],
        "Figures": figures,
        "output": {"dir": "./plots", "formats": ["png"] if formats is None else formats, "dpi": 40},
    }
    yaml_path = tmp_path / "plot.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    try:
        JarvisPLOT(argv=[str(yaml_path), *options]).init()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def test_style_written_as_a_single_string_renders(tmp_path):
    code = _render(tmp_path, [_scatter_figure("one_token", style="a4paper_2x1")])

    assert code == 0
    assert (tmp_path / "plots" / "one_token.png").is_file()


def test_failed_figures_do_not_stay_open(tmp_path):
    figures = [
        _scatter_figure("unknown_axes", axes="no_such_axes"),
        _scatter_figure("unknown_column", y="no_such_column"),
        _scatter_figure("good"),
    ]

    code = _render(tmp_path, figures)

    assert code == 1
    assert (tmp_path / "plots" / "good.png").is_file()
    assert plt.get_fignums() == []


@pytest.mark.parametrize("warm_cache", [False, True])
def test_print_cleans_cache_only_after_all_outputs(tmp_path, monkeypatch, warm_cache):
    from jarvisplot.cache_store import ProjectCache

    figures = [_scatter_figure("first"), _scatter_figure("second")]
    if warm_cache:
        assert _render(tmp_path, figures) == 0
        assert list((tmp_path / ".cache" / "data").glob("*.pkl"))
        # A warm render should read cached data, not recalculate it.
        monkeypatch.setattr(ProjectCache, "put_dataframe", lambda *a, **k: pytest.fail("cache miss"))

    unrelated = tmp_path / ".cache" / "data" / "unrelated.pkl"
    unrelated.parent.mkdir(parents=True, exist_ok=True)
    unrelated.write_bytes(b"other plot")
    original_clear = ProjectCache.clear_used
    cleanups = []

    def clear_after_outputs(cache):
        for name in ("first", "second"):
            for fmt in ("png", "pdf"):
                assert (tmp_path / "plots" / f"{name}.{fmt}").is_file()
        assert list(cache.data_dir.glob("*.json"))
        cleanups.append(True)
        return original_clear(cache)

    monkeypatch.setattr(ProjectCache, "clear_used", clear_after_outputs)
    assert _render(tmp_path, figures, "--print", formats=["png", "pdf"]) == 0
    assert cleanups == [True]
    assert list(unrelated.parent.iterdir()) == [unrelated]
    assert not list((tmp_path / ".cache" / "summary").iterdir())


def test_print_preserves_cache_on_partial_render_failure(tmp_path):
    code = _render(tmp_path, [_scatter_figure("good"), _scatter_figure("bad", y="missing")], "--print")

    assert code == 1
    assert (tmp_path / "plots" / "good.png").is_file()
    assert list((tmp_path / ".cache" / "data").glob("*.pkl"))


def test_print_preserves_cache_when_saving_fails(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    def fail_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Figure, "savefig", fail_save)
    assert _render(tmp_path, [_scatter_figure("failed_save")], "--print") == 1
    assert list((tmp_path / ".cache" / "data").glob("*.pkl"))


@pytest.mark.parametrize("mode", ["disabled", "no_formats"])
def test_print_does_not_clean_without_output(tmp_path, monkeypatch, mode):
    from jarvisplot.cache_store import ProjectCache

    monkeypatch.setattr(ProjectCache, "clear_used", lambda *a: pytest.fail("no output to finalize"))
    figure = _scatter_figure("unused")
    if mode == "disabled":
        figure["enable"] = False
    assert _render(tmp_path, [figure], "--print", formats=[] if mode == "no_formats" else ["png"]) == 0


def test_print_announces_cleanup_before_render(tmp_path, monkeypatch):
    import logging

    messages = []
    log = logging.getLogger("print-lifecycle")
    monkeypatch.setattr(JarvisPLOT, "init_logger", lambda core: setattr(core, "logger", log))
    monkeypatch.setattr(log, "warning", lambda msg: messages.append(str(msg)))
    assert _render(tmp_path, [_scatter_figure("final")], "--print") == 0
    assert messages[0].startswith("--print: accepting all settings")
    assert "will be deleted" in messages[0]


def test_failed_layer_concat_warns_with_the_reason():
    from jarvisplot.Figure.figure import Figure

    warnings: list[str] = []
    fig = Figure()
    fig.logger = type("Log", (), {"warning": lambda self, msg: warnings.append(str(msg))})()
    duplicated = pd.DataFrame([[1.0, 2.0]], columns=["x", "x"])
    other = pd.DataFrame({"x": [3.0], "y": [4.0]})

    out = fig._concat_loaded_data([duplicated, other], layer_name="points")

    assert out is duplicated
    assert len(warnings) == 1
    assert "Layer 'points'" in warnings[0]
    assert "only the first block is drawn" in warnings[0]
    assert "InvalidIndexError" in warnings[0]

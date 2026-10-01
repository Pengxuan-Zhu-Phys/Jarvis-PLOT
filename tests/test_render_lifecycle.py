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


def _render(tmp_path: Path, figures: list[dict]) -> int:
    rng = np.random.default_rng(1)
    pd.DataFrame({"x": rng.normal(size=50), "y": rng.normal(size=50)}).to_csv(
        tmp_path / "samples.csv", index=False
    )
    config = {
        "DataSet": [{"name": "samples", "type": "csv", "path": "samples.csv"}],
        "Figures": figures,
        "output": {"dir": "./plots", "formats": ["png"], "dpi": 40},
    }
    yaml_path = tmp_path / "plot.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    try:
        JarvisPLOT(argv=[str(yaml_path)]).init()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def test_style_written_as_a_single_string_renders(tmp_path):
    code = _render(tmp_path, [_scatter_figure("one_token", style="a4paper_2x1")])

    assert code == 0
    assert (tmp_path / "plots" / "one_token.png").is_file()


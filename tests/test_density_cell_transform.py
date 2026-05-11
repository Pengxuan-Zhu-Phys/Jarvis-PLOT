from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jarvisplot.Figure.density_cell_runtime import density_cell
from jarvisplot.Figure.preprocessor import DataPreprocessor
from jarvisplot.data_loader import JP_ROW_IDX


def _logger():
    return SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )


def _sample_df():
    return pd.DataFrame(
        {
            "xx": [0.1, 0.2, 0.8, 0.9, 0.5, 0.35],
            "yy": [0.1, 0.8, 0.2, 0.9, 0.55, 0.3],
            "LogL": [-4.0, -3.0, -2.0, -1.0, -0.5, -2.5],
        }
    )


def _base_cfg(method):
    return {
        "method": method,
        "coordinates": {
            "x": {"expr": "xx", "name": "x", "lim": [0, 1]},
            "y": {"expr": "yy", "name": "y", "lim": [0, 1]},
            "weight": {"expr": "exp(LogL)", "name": "weight"},
        },
        "diagnostics": False,
        "normalize": True,
    }


def test_density_cell_grid_outputs_only_core_columns_and_normalizes():
    df = _sample_df()
    df["extra_original_column"] = np.arange(len(df))
    cfg = _base_cfg("grid")
    cfg["grid"] = {"bins": 8}

    out = density_cell(df, cfg, _logger())

    assert out is not df
    assert list(out.columns) == ["x", "y", "weight"]
    assert "extra_original_column" not in out.columns
    assert "LogL" not in out.columns
    assert len(out) == 64
    assert float(out["weight"].sum()) == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.isfinite(out["x"]))
    assert np.all(np.isfinite(out["y"]))


def test_density_cell_bridson_outputs_only_core_columns_and_normalizes():
    cfg = _base_cfg("bridson")
    cfg["bridson"] = {"bin": 8, "seed": 123}

    out = density_cell(_sample_df(), cfg, _logger())

    assert list(out.columns) == ["x", "y", "weight"]
    assert len(out) > 0
    assert float(out["weight"].sum()) == pytest.approx(1.0, abs=1e-12)
    assert np.count_nonzero(out["weight"].to_numpy() > 0) <= len(_sample_df())


def test_density_cell_kde_outputs_only_core_columns_and_normalizes():
    cfg = _base_cfg("kde")
    cfg["grid"] = {"bins": 10}
    cfg["bw_method"] = "0.5 * scott"

    out = density_cell(_sample_df(), cfg, _logger())

    assert list(out.columns) == ["x", "y", "weight"]
    assert len(out) == 100
    assert float(out["weight"].sum()) == pytest.approx(1.0, abs=1e-12)
    assert np.nanmax(out["weight"]) > 0


def test_density_cell_custom_output_names():
    cfg = {
        "method": "grid",
        "coordinates": {
            "x": {"expr": "xx", "name": "x_core", "lim": [0, 1]},
            "y": {"expr": "yy", "name": "y_core", "lim": [0, 1]},
            "weight": {"expr": "exp(LogL)", "name": "mass"},
        },
        "grid": {"bins": 4},
        "diagnostics": False,
    }

    out = density_cell(_sample_df(), cfg, _logger())

    assert list(out.columns) == ["x_core", "y_core", "mass"]
    assert float(out["mass"].sum()) == pytest.approx(1.0, abs=1e-12)


def test_density_cell_runtime_transform_and_projection():
    dp = DataPreprocessor(context=None, logger=_logger())
    transform = [
        {
            "make_density_core": {
                "method": "grid",
                "coordinates": {
                    "x": {"expr": "xx", "name": "x", "lim": [0, 1]},
                    "y": {"expr": "yy", "name": "y", "lim": [0, 1]},
                    "weight": {"expr": "exp(LogL)", "name": "weight"},
                },
                "grid": {"bins": 5},
                "diagnostics": False,
            }
        }
    ]

    projection = dp._runtime_projection(transform, [])
    out = dp.apply_runtime_transforms(_sample_df(), transform, source_label="posterior")

    assert JP_ROW_IDX in projection
    assert "xx" in projection
    assert "yy" in projection
    assert "LogL" in projection
    assert "weight" in projection
    assert list(out.columns) == ["x", "y", "weight"]
    assert float(out["weight"].sum()) == pytest.approx(1.0, abs=1e-12)


def test_density_cell_type_form_uses_make_density_core():
    cfg = _base_cfg("grid")
    cfg.update({"type": "make_density_core", "grid": {"bins": 3}})

    out = DataPreprocessor(context=None, logger=_logger()).apply_runtime_transforms(
        _sample_df(),
        [cfg],
        source_label="posterior",
    )

    assert list(out.columns) == ["x", "y", "weight"]
    assert len(out) == 9

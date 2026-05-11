from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jarvisplot.Figure.interp_2d_runtime import make_interp_2d
from jarvisplot.Figure.preprocessor import DataPreprocessor
from jarvisplot.data_loader import JP_ROW_IDX


def _logger():
    return SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )


def _support_df():
    x = np.array([0.0, 1.0, 0.0, 1.0, 0.5, 0.25, 0.75])
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.5, 0.75, 0.25])
    return pd.DataFrame({"x": x, "y": y, "weight": 1.0 + x + 2.0 * y, "raw": np.arange(len(x))})


def _base_cfg(method: str):
    return {
        "method": method,
        "coordinates": {
            "x": {"expr": "x", "name": "x", "lim": [0, 1], "scale": "linear"},
            "y": {"expr": "y", "name": "y", "lim": [0, 1], "scale": "linear"},
            "z": {"expr": "weight", "name": "posterior_pdf"},
        },
        "grid": {"bins": 9},
        "nan_policy": "strict",
        "diagnostics": False,
    }


def test_make_interp_2d_natural_neighbor_outputs_only_grid_columns():
    out = make_interp_2d(_support_df(), _base_cfg("natural_neighbor"), _logger())

    assert list(out.columns) == ["x", "y", "posterior_pdf"]
    assert len(out) == 81
    assert "weight" not in out.columns
    assert np.nanmax(out["posterior_pdf"].to_numpy()) > np.nanmin(out["posterior_pdf"].to_numpy())
    assert out["x"].min() == pytest.approx(0.0)
    assert out["x"].max() == pytest.approx(1.0)


def test_make_interp_2d_triangulation_linear():
    cfg = _base_cfg("triangulation")
    cfg["triangulation"] = {"kind": "linear"}

    out = make_interp_2d(_support_df(), cfg, _logger())

    assert list(out.columns) == ["x", "y", "posterior_pdf"]
    assert len(out) == 81
    center = out.iloc[len(out) // 2]["posterior_pdf"]
    assert float(center) == pytest.approx(2.5, rel=0.25)


def test_make_interp_2d_griddata_nearest_and_custom_grid_shape():
    cfg = _base_cfg("griddata")
    cfg["grid"] = {"nx": 7, "ny": 5}
    cfg["griddata"] = {"kind": "nearest"}

    out = make_interp_2d(_support_df(), cfg, _logger())

    assert list(out.columns) == ["x", "y", "posterior_pdf"]
    assert len(out) == 35
    assert np.all(np.isfinite(out["posterior_pdf"].to_numpy()))


def test_make_interp_2d_accepts_compact_scalar_grid():
    cfg = _base_cfg("griddata")
    cfg["grid"] = 6
    cfg["griddata"] = {"kind": "nearest"}

    out = make_interp_2d(_support_df(), cfg, _logger())

    assert list(out.columns) == ["x", "y", "posterior_pdf"]
    assert len(out) == 36


def test_make_interp_2d_accepts_compact_rectangular_grid():
    cfg = _base_cfg("griddata")
    cfg["grid"] = [7, 5]
    cfg["griddata"] = {"kind": "nearest"}

    out = make_interp_2d(_support_df(), cfg, _logger())

    assert list(out.columns) == ["x", "y", "posterior_pdf"]
    assert len(out) == 35
    assert out["x"].nunique() == 7
    assert out["y"].nunique() == 5


def test_make_interp_2d_omitted_grid_defaults_to_256_square():
    cfg = _base_cfg("griddata")
    cfg.pop("grid")
    cfg["griddata"] = {"kind": "nearest"}

    out = make_interp_2d(_support_df(), cfg, _logger())

    assert len(out) == 256 * 256


def test_make_interp_2d_as_density_converts_grid_mass_to_density():
    df = pd.DataFrame(
        {
            "x": [0.25, 0.75, 0.25, 0.75],
            "y": [0.25, 0.25, 0.75, 0.75],
            "mass": [0.25, 0.25, 0.25, 0.25],
        }
    )
    cfg = {
        "method": "griddata",
        "as_density": True,
        "coordinates": {
            "x": {"expr": "x", "name": "x", "lim": [0, 1]},
            "y": {"expr": "y", "name": "y", "lim": [0, 1]},
            "z": {"expr": "mass", "name": "posterior_pdf"},
        },
        "grid": 5,
        "griddata": {"kind": "nearest"},
        "diagnostics": False,
    }

    out = make_interp_2d(df, cfg, _logger())

    assert list(out.columns) == ["x", "y", "posterior_pdf"]
    assert np.nanmin(out["posterior_pdf"]) == pytest.approx(1.0)
    assert np.nanmax(out["posterior_pdf"]) == pytest.approx(1.0)


def test_make_interp_2d_normalize_rescales_grid_integral():
    cfg = _base_cfg("griddata")
    cfg["grid"] = 12
    cfg["griddata"] = {"kind": "nearest"}
    cfg["normalize"] = True

    out = make_interp_2d(_support_df(), cfg, _logger())
    xs = np.sort(out["x"].unique())
    ys = np.sort(out["y"].unique())
    dx = float(np.mean(np.diff(xs)))
    dy = float(np.mean(np.diff(ys)))
    integral = float(np.nansum(out["posterior_pdf"].to_numpy()) * dx * dy)

    assert integral == pytest.approx(1.0, abs=1e-12)


def test_make_interp_2d_respects_log_scale_and_output_names():
    df = pd.DataFrame(
        {
            "xx": [0.1, 1.0, 10.0, 0.1, 10.0],
            "yy": [0.1, 1.0, 10.0, 10.0, 0.1],
            "zz": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    cfg = {
        "method": "griddata",
        "coordinates": {
            "x": {"expr": "xx", "name": "x_phys", "lim": [0.1, 10.0], "scale": "log"},
            "y": {"expr": "yy", "name": "y_phys", "lim": [0.1, 10.0], "scale": "log"},
            "z": {"expr": "zz", "name": "field"},
        },
        "grid": {"bins": 6},
        "griddata": {"kind": "nearest"},
        "diagnostics": False,
    }

    out = make_interp_2d(df, cfg, _logger())

    assert list(out.columns) == ["x_phys", "y_phys", "field"]
    assert out["x_phys"].min() == pytest.approx(0.1)
    assert out["x_phys"].max() == pytest.approx(10.0)
    assert out["y_phys"].min() == pytest.approx(0.1)
    assert out["y_phys"].max() == pytest.approx(10.0)


def test_make_interp_2d_runtime_transform_and_projection():
    dp = DataPreprocessor(context=None, logger=_logger())
    transform = [{"make_interp_2d": _base_cfg("griddata")}]

    projection = dp._runtime_projection(transform, [])
    out = dp.apply_runtime_transforms(_support_df(), transform, source_label="interp")

    assert JP_ROW_IDX in projection
    assert "x" in projection
    assert "y" in projection
    assert "weight" in projection
    assert "posterior_pdf" in projection
    assert list(out.columns) == ["x", "y", "posterior_pdf"]


def test_make_interp_2d_raises_on_too_few_points():
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [1.0, 2.0]})

    with pytest.raises(ValueError, match="at least three"):
        make_interp_2d(df, {"method": "griddata", "grid": {"bins": 3}}, _logger())

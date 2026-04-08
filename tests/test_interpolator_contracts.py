from __future__ import annotations

import pandas as pd
import pytest

from jarvisplot.core_assets import load_interpolators
from jarvisplot.inner_func import clear_external_funcs, update_funcs
from jarvisplot.utils.interpolator import InterpolatorManager


def test_interpolator_manager_from_yaml_accepts_empty_config(tmp_path):
    mgr = InterpolatorManager.from_yaml({}, tmp_path)

    assert mgr.summary() == {"registered": [], "built": []}


def test_load_interpolators_clears_stale_external_functions(tmp_path):
    csv_path = tmp_path / "points.csv"
    pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 2.0, 4.0]}).to_csv(csv_path, index=False)

    config = {
        "Functions": [
            {
                "name": "f_cut",
                "source": {"type": "csv", "path": "./points.csv", "x": "x", "y": "y"},
                "method": "interp1d",
                "options": {"kind": "linear", "bounds": "clamp"},
            }
        ]
    }

    try:
        clear_external_funcs()

        load_interpolators(config, yaml_dir=tmp_path, shared=None, logger=None)
        funcs = update_funcs({})
        assert "f_cut" in funcs
        assert funcs["f_cut"](1.5) == pytest.approx(3.0)

        load_interpolators({}, yaml_dir=tmp_path, shared=None, logger=None)
        funcs_after = update_funcs({})
        assert "f_cut" not in funcs_after
    finally:
        clear_external_funcs()

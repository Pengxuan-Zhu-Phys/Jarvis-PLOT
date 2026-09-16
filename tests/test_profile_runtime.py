"""Regression coverage for finite, objective-aware profile support selection."""

import logging

import numpy as np
import pandas as pd
import pytest

from jarvisplot.Figure.preprocessor import DataPreprocessor
from jarvisplot.Figure.data_pipelines import DataContext, SharedContent
from jarvisplot.Figure.profile_runtime import _preprofiling, profiling
from jarvisplot.cache_store import ProjectCache


def config(objective="max"):
    return {
        "method": "bridson", "bin": 100, "objective": objective,
        "grid_points": "none", "pregrid": {"bin": 10},
        "coordinates": {
            k: {"expr": k, "name": k, "lim": [0, 1]}
            for k in ("x", "y")
        } | {"z": {"expr": "z", "name": "z"}},
    }


@pytest.mark.parametrize("objective,expected", [("max", [1.0, 0.8]), ("min", [0.0, 0.2])])
def test_preprofile_keeps_only_requested_extremum(objective, expected):
    # Duplicate dataframe indices occur when scan files are concatenated.
    df = pd.DataFrame({
        "x": [0.11, 0.12, 0.41, 0.42, 0.8],
        "y": [0.11, 0.12, 0.41, 0.42, 0.8],
        "z": [0.0, 1.0, 0.2, 0.8, np.nan],
        "row": ["low1", "high1", "low2", "high2", "invalid"],
    }, index=[0, 1, 0, 1, 0])
    out = _preprofiling(df, config(objective), None)
    np.testing.assert_allclose(out.z, expected)
    assert out.row.tolist() == (["high1", "high2"] if objective == "max" else ["low1", "low2"])


@pytest.mark.parametrize("objective", ["max", "min"])
def test_invalid_objective_does_not_disable_bridson_exclusion(objective):
    # The inferior point lies in the annulus where z-aware rejection is needed.
    z = [1.0, 0.0] if objective == "max" else [0.0, 1.0]
    good = pd.DataFrame({"x": [0.1, 0.109], "y": [0.1, 0.1], "z": z})
    dirty = pd.concat([good, pd.DataFrame({
        "x": [0.8, 0.9, 0.7], "y": [0.8, 0.9, 0.7],
        "z": [np.nan, np.inf, -np.inf],
    })], ignore_index=True)
    log = logging.getLogger(__name__)
    expected = profiling(good, config(objective), log)
    actual = profiling(dirty, config(objective), log)
    assert len(actual) == 1
    pd.testing.assert_frame_equal(actual, expected)


def test_constant_profile_still_thins_close_points():
    df = pd.DataFrame({"x": [0.1, 0.101], "y": [0.1, 0.1], "z": [0.0, 0.0]})
    with np.errstate(invalid="raise", divide="raise"):
        out = profiling(df, config(), None)
    assert len(out) == 1
    assert out.z.iloc[0] == 0.0


@pytest.mark.parametrize("objective", ["max", "min"])
def test_suppressed_candidate_still_contributes_to_cell_extremum(objective):
    # A suppresses B, but C lies outside A's exclusion radius. B belongs to
    # C's nearest-seed cell and is better than C: thinning alone leaks C.
    df = pd.DataFrame({
        "x": [0.1, 0.1092, 0.1179], "y": [0.1, 0.1, 0.1],
        "z": [1.0, 0.9, 0.0] if objective == "max" else [0.0, 0.1, 1.0],
        "row": ["A", "B", "C"], "nuisance": [10.0, 20.0, 30.0],
    })
    actual = profiling(df, config(objective), None)
    assert actual.row.tolist() == ["A", "B"]
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), df.iloc[:2].reset_index(drop=True))


def test_all_invalid_profile_returns_empty_named_columns():
    df = pd.DataFrame({"x": [0.1], "y": [0.1], "z": [np.nan]})
    cfg = config()
    cfg["coordinates"]["z"]["name"] = "likelihood"
    for reduce in (_preprofiling, profiling):
        out = reduce(df, cfg, None)
        assert out.empty
        assert "likelihood" in out


def test_log_profile_ignores_nonpositive_coordinates_and_constant_z():
    cfg = config()
    cfg["grid_points"] = "rect"
    cfg["bin"] = 4
    for key in ("x", "y", "z"):
        cfg["coordinates"][key]["scale"] = "log"
    for key in ("x", "y"):
        cfg["coordinates"][key]["lim"] = [1, 10]
    df = pd.DataFrame({"x": [2.0, 0.0, -1.0], "y": [3.0, 3.0, 3.0], "z": [0.05, 0.05, 0.05]})
    with np.errstate(invalid="raise", divide="raise"):
        out = profiling(df, cfg, None)
    actual = out[np.isfinite(out.z)]
    assert actual.x.tolist() == [2.0]
    assert actual.z.tolist() == [0.05]


def test_pregrid_controls_and_objective_change_cache_identity():
    dp = DataPreprocessor(context=None)
    cfg = config()
    base = dp._preprofile_profile_cfg(cfg)
    for change in ({"objective": "min"}, {"pregrid": False}, {"pregrid": {"bin": 20}}, {"pregrid_bin": 20}):
        other = dp._preprofile_profile_cfg(cfg | change)
        assert dp._pipeline_key("scan", [{"profile": base}], mode="preprofile") != dp._pipeline_key(
            "scan", [{"profile": other}], mode="preprofile"
        )
    pre, _ = dp._split_prebuild_transform([{"profile": cfg | {"pregrid": False}}])
    df = pd.DataFrame({"x": [0.11, 0.12], "y": [0.11, 0.12], "z": [0.0, 1.0]})
    assert _preprofiling(df, pre[0]["profile"], None) is df


def test_old_named_profile_signature_is_invalidated(monkeypatch):
    dp = DataPreprocessor(context=None)
    layer = {"name": "support", "data": [{"source": "scan", "transform": [{"profile": config()}]}]}
    current = dp._layer_signature(layer)
    monkeypatch.setattr(dp, "_runtime_profile_signature", lambda tf: "old-algorithm")
    assert dp._layer_signature(layer) != current


def test_prebuild_cached_max_and_min_do_not_reuse_each_others_rows(tmp_path):
    df = pd.DataFrame({"x": [0.11, 0.12], "y": [0.11, 0.12], "z": [0.0, 1.0]})
    for warm in (False, True):
        ctx = DataContext(SharedContent())
        ctx.register("scan", lambda _: df)
        dp = DataPreprocessor(ctx, cache=ProjectCache(str(tmp_path)))
        project = {"Figures": [{"name": "test", "layers": [
            {"name": objective, "data": [{"source": "scan", "transform": [{"profile": config(objective)}]}]}
            for objective in ("max", "min")
        ]}]}
        stats = dp.prebuild_profiles(project)
        assert stats["hits"] == (2 if warm else 0)
        entries = [layer["data"][0] for layer in project["Figures"][0]["layers"]]
        assert entries[0]["source"] != entries[1]["source"]
        assert ctx.get(entries[0]["source"]).z.tolist() == [1.0]
        assert ctx.get(entries[1]["source"]).z.tolist() == [0.0]

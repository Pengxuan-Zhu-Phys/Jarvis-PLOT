from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from jarvisplot.Figure.distribution_1d_runtime import build_cdf, distribution_1d
from jarvisplot.Figure.preprocessor import DataPreprocessor
from jarvisplot.client import main
from jarvisplot.column_demand import plan_source_demand, transform_columns, transform_output_columns
from jarvisplot.data_loader import DataSet
from jarvisplot.data_loader_runtime import apply_dataset_transform
from jarvisplot.distribution_1d_config import settings
from jarvisplot.dryrun_runtime import dryrun_config
from jarvisplot.validation import validate_config


def _logger():
    return SimpleNamespace(**{key: lambda *a, **kw: None for key in ("debug", "info", "warning", "error")})


def _cfg(**x):
    return {"anchors": "all", "interpolation": "pchip",
            "coordinates": {"x": {"expr": "score", "lim": [0, 1], "grid": 101, **x},
                            "weight": {"expr": "weight"}}}


def _samples():
    return pd.DataFrame({"score": [.2, .5, .5, .8], "weight": [1., 1., 2., 4.], "unused": [0] * 4})


def test_raw_duplicate_weights_preserve_empirical_cumulative_and_unit_integral():
    df = _samples()
    cdf = build_cdf(df.score.to_numpy(), df.weight.to_numpy(), (0, 1))
    np.testing.assert_allclose(cdf([.2, .5, .8]), [1 / 8, 4 / 8, 1])
    assert cdf.derivative().integrate(0, 1) == pytest.approx(1, abs=1e-14)
    assert cdf([0, 1]).tolist() == pytest.approx([0, 1])
    out = distribution_1d(df, _cfg(), "PDF1D")
    np.testing.assert_allclose(out.cdf, cdf(out.x))
    np.testing.assert_allclose(out.pdf, cdf.derivative()(out.x), atol=1e-14)
    assert np.all(np.diff(out.cdf) >= -1e-14)
    assert (out.pdf >= 0).all()
    assert out.pdf_std.isna().all() and out.cdf_std.isna().all()
    assert (out.n_repeats == 1).all()
    assert "unused" not in out


def test_each_repeat_normalizes_before_equal_mean_and_sample_std():
    a = _samples().assign(batch="a")
    b = pd.DataFrame({"score": [.1, .4, .7], "weight": [100., 500., 1000.], "batch": "b"})
    df = pd.concat([a, b], ignore_index=True)
    cfg = {**_cfg(), "repeat": "batch"}
    out = distribution_1d(df, cfg, "PDF1D")
    single = [distribution_1d(part, _cfg(), "PDF1D") for part in (a, b)]
    for key in ("cdf", "pdf"):
        values = np.stack([part[key] for part in single])
        np.testing.assert_allclose(out[key], values.mean(axis=0), atol=1e-14)
        np.testing.assert_allclose(out[f"{key}_std"], values.std(axis=0, ddof=1), atol=1e-14)
    assert (out.n_repeats == 2).all()
    pooled = distribution_1d(df, _cfg(), "PDF1D")
    assert not np.allclose(out.cdf, pooled.cdf)


def test_grid_is_evaluation_only_and_pure_cdf_has_no_pdf():
    coarse = distribution_1d(_samples(), _cfg(grid=51), "CDF1D")
    dense = distribution_1d(_samples(), _cfg(grid=101), "PDF1D")
    assert "pdf" not in coarse and "pdf_std" not in coarse
    np.testing.assert_allclose(coarse.x, dense.x[::2])
    np.testing.assert_allclose(coarse.cdf, dense.cdf[::2])


def test_shared_expressions_scalar_weights_custom_name_and_log_grid():
    cfg = {"coordinates": {"x": {"expr": "np.exp(score)", "name": "value", "lim": [1, 3],
                                  "scale": "log", "grid": 23}, "weight": {"expr": 2}},
           "anchors": "all", "interpolation": "pchip"}
    out = distribution_1d(_samples(), cfg, "PDF1D")
    np.testing.assert_allclose(out.value, np.geomspace(1, 3, 23))
    cdf = build_cdf(np.exp(_samples().score.to_numpy()), np.ones(4), (1, 3))
    np.testing.assert_allclose(out.pdf, cdf.derivative()(out.value), atol=1e-14)
    assert transform_columns([{"PDF1D": cfg}]) == {"score"}
    assert transform_output_columns([{"PDF1D": cfg}]) == {"value", "cdf", "cdf_std", "pdf", "pdf_std", "n_repeats"}


def test_inferred_common_support_ignores_zero_weight_rows():
    df = pd.DataFrame({"score": [.2, .4, .6, 100], "weight": [1, 1, 1, 0], "batch": [1, 1, 2, 2]})
    cfg = {"coordinates": {"x": {"name": "score", "grid": 21}, "weight": "weight"}, "repeat": "batch"}
    out = distribution_1d(df, cfg, "PDF1D")
    assert out.score.iloc[[0, -1]].tolist() == pytest.approx([.1, .7])
    assert out.cdf.iloc[[0, -1]].tolist() == pytest.approx([0, 1])
    assert transform_columns([{"PDF1D": cfg}]) == {"score", "weight", "batch"}


def test_support_boundary_mass_is_retained_with_warning():
    messages = []
    logger = _logger()
    logger.warning = messages.append
    df = pd.DataFrame({"score": [0, .4, 1], "weight": [2, 3, 5]})
    out = distribution_1d(df, _cfg(), "PDF1D", logger)
    assert out.cdf.iloc[40] == pytest.approx(.5)
    assert messages and "lower support boundary" in messages[0]
    cdf = build_cdf(df.score.to_numpy(), df.weight.to_numpy(), (0, 1))
    assert cdf.derivative().integrate(0, 1) == pytest.approx(1)


def test_extreme_weights_normalize_without_overflow():
    cdf = build_cdf(np.array([.2, .5, .8]), np.full(3, 1e308), (0, 1))
    assert cdf(.5) == pytest.approx(2 / 3)
    assert cdf.derivative().integrate(0, 1) == pytest.approx(1)


@pytest.mark.parametrize("values,match", [
    ({"weight": [-1, 1, 1, 1]}, "nonnegative"),
    ({"weight": [0, 0, 0, 0]}, "sum to zero"),
    ({"weight": [1, np.nan, 1, 1]}, "finite"),
    ({"score": [np.inf, .4, .5, .6]}, "finite"),
    ({"score": [-.1, .4, .5, .6]}, "must contain"),
])
def test_invalid_samples_fail_explicitly(values, match):
    with pytest.raises(ValueError, match=match):
        distribution_1d(_samples().assign(**values), _cfg(), "PDF1D")


@pytest.mark.parametrize("batch,weights,match", [
    ([1, 1, 2, 2], [1, 1, 0, 0], "repeat 2.*sum to zero"),
    ([1, 1, None, 2], [1, 1, 1, 1], "identifiers must not be missing"),
])
def test_repeat_validation(batch, weights, match):
    df = _samples().assign(batch=batch, weight=weights)
    with pytest.raises(ValueError, match=match):
        distribution_1d(df, {**_cfg(), "repeat": "batch"}, "CDF1D")


@pytest.mark.parametrize("cfg,match", [
    ({**_cfg(), "grid": 5}, "grid belongs"),
    (_cfg(grid=True), "integer >= 2"),
    (_cfg(grid=1), "integer >= 2"),
    (_cfg(lim=[1, 0]), "lo < hi"),
    (_cfg(scale="log"), "positive limits"),
    (_cfg(name="pdf"), "reserved"),
    ({**_cfg(), "groupby": "sample"}, "unknown keys"),
])
def test_closed_runtime_configuration(cfg, match):
    with pytest.raises(ValueError, match=match):
        settings(cfg)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("kind", ["PDF1D", "CDF1D"])
def test_dataset_and_layer_transforms_match_and_keep_new_outputs(backend, kind):
    ds = DataSet()
    ds.logger = _logger()
    ds.name = "events"
    df = _samples().assign(batch=[1, 1, 2, 2])
    if backend == "polars":
        import polars as pl
        ds.data = pl.from_pandas(df).lazy()
    else:
        ds.data = df
    cfg = {**_cfg(), "repeat": "batch"}
    ds.transform = [{"filter": "score > 0"}, {kind: cfg},
                    {"add_column": {"name": "upper", "expr": "cdf + cdf_std"}}]
    ds.retained_columns = {"score", "weight", "batch"}
    apply_dataset_transform(ds)
    dp = DataPreprocessor(context=None, logger=_logger())
    layer = dp.apply_runtime_transforms(df, ds.transform, source_label="events")
    pd.testing.assert_frame_equal(ds.data.drop(columns="__jp_row_idx__"), layer)
    assert set(transform_output_columns([{kind: cfg}])) <= set(ds.data.columns)
    assert np.array_equal(ds.data["__jp_row_idx__"], np.arange(101))


def test_projection_column_demand_and_cache_revision(monkeypatch):
    cfg = {**_cfg(name="evaluation"), "repeat": "batch"}
    tf = [{"filter": 'sample == "signal"'}, {"PDF1D": cfg}]
    layer = {"name": "pdf", "method": "plot", "data": [{"source": "events", "transform": tf}],
             "coordinates": {"x": {"expr": "evaluation"}, "y": {"expr": "pdf"}}}
    config = {"DataSet": [{"name": "events"}], "Figures": [{"name": "f", "layers": [layer]}]}
    assert plan_source_demand(config)["events"].missing_candidates == {"score", "weight", "batch", "sample"}
    dp = DataPreprocessor(context=None, logger=_logger())
    projection = dp._runtime_projection(tf, ["evaluation", "pdf"])
    assert {"score", "weight", "batch", "sample", "evaluation", "pdf", "__jp_row_idx__"} <= set(projection)
    signature = dp._layer_signature(layer)
    monkeypatch.setattr("jarvisplot.Figure.distribution_1d_runtime.ALGORITHM_REVISION", "future-revision")
    assert signature != dp._layer_signature(layer)


def test_dataset_algorithm_revision_invalidates_downstream_cache(monkeypatch):
    ds = DataSet()
    ds.name = "events"
    ds.transform = [{"CDF1D": _cfg()}]
    dp = DataPreprocessor(context=None, dataset_registry={"events": ds}, logger=_logger())
    layer = {"name": "cdf", "data": [{"source": "events"}]}
    fingerprint = ds.fingerprint()
    signature = dp._layer_signature(layer)
    monkeypatch.setattr("jarvisplot.Figure.distribution_1d_runtime.ALGORITHM_REVISION", "future-revision")
    assert ds.fingerprint() != fingerprint
    assert dp._layer_signature(layer) != signature


def _example():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / "Example/pdf_cdf_1d.yaml").read_text())


def test_schema_and_dryrun_report_dataset_and_layer_reconstruction_as_partial():
    config = _example()
    assert list(validate_config(config)) == []
    report, bag = dryrun_config(config)
    assert bag.ok
    assert report["coverage"] == "partial"
    assert report["datasets"]["events"]["heavy_skipped"] == ["PDF1D"]
    assert all(layer["incomplete"] for layer in report["layers"])
    assert any("CDF1D" in step for step in report["heavy_skipped"])


@pytest.mark.parametrize("changes", [{"grid": 5}, {"coordinates": {"x": {"expr": "score", "grid": 1}}},
                                     {"groupby": "sample"}, {"coordinates": {"x": {"expr": "score", "bins": 5}}}])
def test_schema_rejects_old_or_misplaced_configuration(changes):
    config = _example()
    config["DataSet"][0]["transform"][0]["PDF1D"].update(changes)
    assert list(validate_config(config))


@pytest.mark.parametrize("token", ["PDF1D", "transform.pdf1d", "CDF1D", "transform.CDF1D"])
def test_cli_manual_resolves_canonical_and_lowercase_queries(token, capsys):
    assert main(["man", token, "--json"]) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["ok"] is True
    blob = json.dumps(envelope["data"])
    assert "coordinates.x.grid" in blob
    assert "repeat" in blob and "ddof=1" in blob

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jarvisplot.Figure.density_cell_runtime import density_cell
from jarvisplot.Figure.posterior_density_runtime import posterior_density
from jarvisplot.Figure.posterior_mesh import (
    _anisotropic_ownership,
    _greedy_unique_generators,
    assign_nearest_ownership,
    build_posterior_mesh,
)
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


def test_density_cell_bridson_records_internal_mesh_diagnostics():
    cfg = _base_cfg("bridson")
    cfg["bridson"] = {"bin": 8, "seed": 123}

    out = density_cell(_sample_df(), cfg, _logger())
    mesh = density_cell.last_mesh

    assert mesh is not None
    assert mesh.generators.shape == (len(out), 2)
    assert mesh.masses.shape == (len(out),)
    assert float(mesh.masses.sum()) == pytest.approx(float(out["weight"].sum()), abs=1e-12)
    assert mesh.sample_counts.sum() == len(_sample_df())
    assert mesh.centroid_proposal.shape == mesh.generators.shape
    assert mesh.centroid_drift_magnitude.shape == mesh.masses.shape
    assert mesh.diagnostics["area_method"] in {"regular_grid", "clipped_voronoi", "not_performed"}
    payload = mesh.debug_payload()
    assert {
        "generators",
        "centroid_drift",
        "anisotropy_ratio",
        "mass_stress",
        "geometry_stress",
        "stress_gap",
        "stress",
    } <= set(payload)
    assert payload["stress"].shape == mesh.masses.shape
    assert payload["geometry_stress"].shape == mesh.masses.shape
    assert payload["stress_gap"].shape == mesh.masses.shape


def test_density_cell_bridson_can_export_internal_mesh_diagnostics(tmp_path):
    cfg = _base_cfg("bridson")
    cfg["bridson"] = {"bin": 8, "seed": 123}
    cfg["_base_dir"] = str(tmp_path)
    cfg["_mesh_debug"] = {"to_csv": "mesh_debug.csv"}

    density_cell(_sample_df(), cfg, _logger())

    out_path = tmp_path / "mesh_debug.csv"
    assert out_path.exists()
    exported = pd.read_csv(out_path)
    assert {
        "x",
        "y",
        "mass",
        "area",
        "density",
        "sample_count",
        "centroid_x",
        "centroid_y",
        "drift_x",
        "drift_y",
        "drift_x_norm",
        "drift_y_norm",
        "anisotropy_ratio",
        "mass_stress",
        "geometry_stress",
        "stress_gap",
        "stress",
    } <= set(exported.columns)
    assert exported["mass"].sum() == pytest.approx(1.0, abs=1e-12)


def test_build_posterior_mesh_computes_conservative_debug_quantities():
    generators = np.array(
        [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
        ]
    )
    samples = generators.copy()
    weights = np.array([0.1, 0.2, 0.3, 0.4])

    mesh = build_posterior_mesh(
        generators,
        samples,
        weights,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        include_polygons=True,
        strict_geometry=True,
    )

    np.testing.assert_allclose(mesh.masses, weights)
    np.testing.assert_allclose(mesh.areas, np.full(4, 0.25))
    np.testing.assert_allclose(mesh.densities, weights / 0.25)
    np.testing.assert_allclose(mesh.centroid_proposal, generators)
    np.testing.assert_allclose(mesh.centroid_drift_magnitude, np.zeros(4), atol=1e-14)
    assert mesh.geometry_stress.shape == (4,)
    assert mesh.mass_stress.shape == (4,)
    assert mesh.stress_gap.shape == (4,)
    assert mesh.polygons is not None
    assert len(mesh.polygons) == 4


def test_build_posterior_mesh_with_refinement_can_split_and_record_history():
    generators = np.array(
        [
            [0.20, 0.20],
            [0.80, 0.20],
            [0.20, 0.80],
            [0.80, 0.80],
        ]
    )
    ridge_a = np.column_stack(
        [
            np.linspace(0.18, 0.28, 20),
            np.linspace(0.18, 0.28, 20),
        ]
    )
    ridge_b = np.column_stack(
        [
            np.linspace(0.68, 0.78, 20),
            np.linspace(0.68, 0.78, 20),
        ]
    )
    samples = np.vstack([ridge_a, ridge_b])
    weights = np.ones(samples.shape[0], dtype=float)

    mesh = build_posterior_mesh(
        generators,
        samples,
        weights,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        include_polygons=True,
        strict_geometry=True,
        refinement={
            "enabled": True,
            "iterations": 2,
            "alpha": 0.3,
            "eta": 0.5,
            "anisotropic": True,
            "split_enabled": True,
            "merge_enabled": False,
            "split_quantile": 0.0,
            "split_offset": 0.75,
            "min_separation": 0.005,
            "max_splits": 1,
            "max_generators": 8,
            "record_history": True,
            "seed": 7,
        },
    )

    assert mesh.diagnostics["refinement_enabled"] is True
    assert mesh.diagnostics["refinement_iterations_completed"] >= 1
    assert mesh.diagnostics["refinement_splits"] >= 1
    assert mesh.generators.shape[0] >= generators.shape[0]
    assert not np.allclose(mesh.generators[: generators.shape[0]], generators)
    assert isinstance(mesh.diagnostics.get("refinement_trace"), list)
    assert len(mesh.diagnostics["refinement_trace"]) >= 1
    assert isinstance(mesh.diagnostics.get("generator_history"), list)
    assert len(mesh.diagnostics["generator_history"]) >= 2


def test_greedy_unique_generators_truncates_excess_priority_entries():
    generators = np.array(
        [
            [0.10, 0.10],
            [0.40, 0.10],
            [0.70, 0.10],
        ]
    )
    priority = np.array([3.0, 2.0, 1.0, 0.5])

    out = _greedy_unique_generators(
        generators,
        priority,
        min_separation=0.05,
        min_generators=2,
    )

    assert out.shape[0] == 3
    np.testing.assert_allclose(out, generators)


def test_greedy_unique_generators_pads_new_entries_as_low_priority():
    generators = np.array(
        [
            [0.10, 0.10],
            [0.40, 0.10],
            [0.105, 0.10],
        ]
    )
    priority = np.array([10.0, 5.0])

    out = _greedy_unique_generators(
        generators,
        priority,
        min_separation=0.02,
        min_generators=2,
    )

    np.testing.assert_allclose(out, generators[:2])


def test_anisotropic_ownership_can_differ_from_nearest_ownership():
    samples = np.array([[0.75, 0.0]])
    generators = np.array([[0.0, 0.0], [1.0, 0.0]])
    covariance = np.array(
        [
            [[1.0, 0.0], [0.0, 0.01]],
            [[0.01, 0.0], [0.0, 0.01]],
        ]
    )
    diagnostics = {}

    nearest = assign_nearest_ownership(samples, generators)
    anisotropic = _anisotropic_ownership(
        samples,
        generators,
        covariance,
        covariance_floor=1.0e-6,
        anisotropy_cap=100.0,
        candidate_k=2,
        batch_size=4,
        diagnostics=diagnostics,
    )

    assert nearest.tolist() == [1]
    assert anisotropic.tolist() == [0]
    assert diagnostics["anisotropic_fallback"] == "none"
    assert diagnostics["anisotropic_fallback_count"] == 0


def test_refinement_merge_path_preserves_total_mass():
    generators = np.array(
        [
            [0.10, 0.10],
            [0.13, 0.10],
            [0.80, 0.80],
            [0.20, 0.80],
        ]
    )
    samples = generators.copy()
    weights = np.array([0.2, 0.3, 0.4, 0.1])

    mesh = build_posterior_mesh(
        generators,
        samples,
        weights,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        refinement={
            "enabled": True,
            "iterations": 1,
            "eta": 0.0,
            "anisotropic": False,
            "split_enabled": False,
            "merge_enabled": True,
            "merge_quantile": 1.0,
            "merge_distance": 0.05,
            "max_merges": 1,
            "min_generators": 3,
            "empty_reseed": False,
        },
    )

    assert mesh.generators.shape[0] == 3
    assert mesh.diagnostics["refinement_merges"] == 1
    assert float(mesh.masses.sum()) == pytest.approx(float(weights.sum()), abs=1e-12)


def test_refinement_reseeds_empty_generators_and_preserves_total_mass():
    generators = np.array(
        [
            [0.10, 0.10],
            [0.90, 0.90],
            [0.20, 0.80],
            [0.50, 0.50],
        ]
    )
    samples = np.array([[0.11, 0.10], [0.89, 0.90], [0.19, 0.81]])
    weights = np.array([0.2, 0.7, 0.1])

    mesh = build_posterior_mesh(
        generators,
        samples,
        weights,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        refinement={
            "enabled": True,
            "iterations": 1,
            "eta": 0.0,
            "anisotropic": False,
            "split_enabled": False,
            "merge_enabled": False,
            "empty_reseed": True,
            "sample_jitter": 0.0,
            "min_separation": 0.0,
            "min_generators": 4,
        },
    )

    assert mesh.diagnostics["refinement_reseeds"] >= 1
    assert float(mesh.masses.sum()) == pytest.approx(float(weights.sum()), abs=1e-12)


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


def test_density_cell_compact_config_defaults_to_bridson_and_top_level_bins():
    cfg = {
        "x": {"expr": "xx", "lim": [0, 1]},
        "y": {"expr": "yy", "lim": [0, 1]},
        "weight": {"expr": "exp(LogL)"},
        "bins": 8,
        "seed": 123,
        "diagnostics": False,
    }

    out = density_cell(_sample_df(), cfg, _logger())

    assert list(out.columns) == ["x", "y", "weight"]
    assert len(out) > 0
    assert float(out["weight"].sum()) == pytest.approx(1.0, abs=1e-12)
    assert density_cell.last_mesh is not None


def test_posterior_density_grid_compact_transform_outputs_density_grid():
    cfg = {
        "method": "grid",
        "x": {"expr": "xx", "lim": [0, 1]},
        "y": {"expr": "yy", "lim": [0, 1]},
        "weight": {"expr": "exp(LogL)"},
        "bins": 5,
        "diagnostics": False,
    }

    out = posterior_density(_sample_df(), cfg, _logger())
    xs = np.sort(out["x"].unique())
    ys = np.sort(out["y"].unique())
    dx = float(np.mean(np.diff(xs)))
    dy = float(np.mean(np.diff(ys)))
    integral = float(np.nansum(out["density"].to_numpy()) * dx * dy)

    assert list(out.columns) == ["x", "y", "density"]
    assert len(out) == 25
    assert integral == pytest.approx(1.0, abs=1e-12)


def test_posterior_density_runtime_projection_and_output_name():
    dp = DataPreprocessor(context=None, logger=_logger())
    transform = [
        {
            "posterior_density": {
                "method": "grid",
                "x": {"expr": "xx", "lim": [0, 1]},
                "y": {"expr": "yy", "lim": [0, 1]},
                "weight": {"expr": "exp(LogL)"},
                "bins": 4,
                "output": "posterior_pdf",
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
    assert "posterior_pdf" in projection
    assert list(out.columns) == ["x", "y", "posterior_pdf"]

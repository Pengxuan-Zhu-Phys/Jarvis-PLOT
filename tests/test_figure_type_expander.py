from __future__ import annotations

import pytest

from jarvisplot.Figure.figure_types import expand_figure_types_in_config, expand_posterior_2d, expand_profile_2d
from jarvisplot.Figure.posterior_hpd import prepare_hpd_contour_style


def _base_figure(**overrides):
    fig = {
        "name": "posterior",
        "type": "posterior_2d",
        "data": ["df0", "df1"],
        "x": {"expr": "xx", "lim": [0, 5], "label": "$x$"},
        "y": {"expr": "yy", "lim": [0, 5], "label": "$y$"},
        "weight": {"expr": "exp(LogL)"},
    }
    fig.update(overrides)
    return fig


@pytest.mark.parametrize("method", ["voronoi", "adaptive", "kde", "grid"])
def test_posterior_2d_expands_density_methods(method):
    density = {"method": method, "bins": 123}
    if method in {"voronoi", "adaptive"}:
        density["grid"] = 321
    if method == "adaptive":
        density["adaptive"] = {"iterations": 2, "max_generators": 64}
    if method == "kde":
        density["kde"] = {"bw_method": "0.5 * scott"}

    out = expand_posterior_2d(_base_figure(density=density))

    assert "type" not in out
    assert out["style"] == ["a4paper_2x1", "rectcmap"]
    assert out["frame"]["ax"]["labels"] == {"x": "$x$", "y": "$y$"}
    assert out["frame"]["ax"]["xlim"] == [0, 5]
    assert len(out["layers"]) == 2

    density_layer = out["layers"][0]
    transform = density_layer["data"][0]["transform"][0]["posterior_density"]
    assert transform["method"] == method
    assert transform["bins"] == 123
    assert transform["x"] == {"expr": "xx", "lim": [0, 5]}
    assert transform["y"] == {"expr": "yy", "lim": [0, 5]}
    assert transform["weight"] == {"expr": "exp(LogL)"}
    if method in {"voronoi", "adaptive"}:
        assert transform["grid"] == 321
    if method == "adaptive":
        assert transform["adaptive"]["iterations"] == 2
    if method == "kde":
        assert transform["kde"]["bw_method"] == "0.5 * scott"


def test_posterior_2d_expands_colorbar_and_hpd_defaults():
    out = expand_posterior_2d(
        _base_figure(
            style_card=["a4paper_4x1", "rectcmap"],
            colorbar={"label": "posterior PDF", "cmap": "viridis", "scale": "log", "vmin": 1e-3, "vmax": 1.0},
            hpd={"masses": [0.5], "colors": ["white"], "linewidths": [0.4]},
        )
    )

    axc = out["frame"]["axc"]
    assert axc["label"] == {"xlabel": "posterior PDF"}
    assert axc["color"] == {"cmap": "viridis", "scale": "log", "vmin": 1e-3, "vmax": 1.0}

    hpd = out["layers"][1]
    assert hpd["method"] == "contour"
    assert hpd["data"] == [{"source": "_posterior_posterior_density"}]
    assert hpd["style"]["contour_mode"] == "posterior_hpd"
    assert hpd["style"]["masses"] == [0.5]
    assert hpd["style"]["labels"] == ["$1\\sigma$", "$2\\sigma$"]
    assert hpd["style"]["colors"] == ["white"]
    assert hpd["style"]["linewidths"] == [0.4]


def test_posterior_2d_can_disable_hpd_and_rename_density_output():
    out = expand_posterior_2d(
        _base_figure(
            density={"method": "grid", "output": "density_qvor"},
            hpd=False,
        )
    )

    assert len(out["layers"]) == 1
    layer = out["layers"][0]
    assert layer["coordinates"]["z"] == {"expr": "density_qvor"}
    transform = layer["data"][0]["transform"][0]["posterior_density"]
    assert transform["output"] == "density_qvor"


def test_posterior_2d_extra_layers_inherit_figure_data():
    out = expand_posterior_2d(
        _base_figure(
            extra_layers=[
                {
                    "method": "scatter",
                    "coordinates": {"x": {"expr": "xx"}, "y": {"expr": "yy"}},
                    "style": {"s": 1, "color": "gray"},
                },
                {
                    "method": "plot",
                    "data": "reference",
                    "coordinates": {"x": {"expr": "x"}, "y": {"expr": "y"}},
                },
            ]
        )
    )

    assert out["layers"][2]["data"] == [{"source": ["df0", "df1"]}]
    assert out["layers"][2]["axes"] == "ax"
    assert out["layers"][3]["data"] == [{"source": "reference"}]


def test_expand_figure_types_in_config_mutates_figures_before_runtime_planning():
    config = {"Figures": [_base_figure(), {"name": "manual", "layers": []}]}

    out = expand_figure_types_in_config(config)

    assert out is config
    assert "type" not in config["Figures"][0]
    assert config["Figures"][0]["layers"][0]["data"][0]["source"] == ["df0", "df1"]
    assert config["Figures"][1] == {"name": "manual", "layers": []}


def test_expand_typed_figures_raises_on_unknown_type():
    from jarvisplot.Figure.figure_types import expand_typed_figures

    config = {"Figures": [{"name": "x", "type": "not_a_real_type", "data": "s"}]}
    with pytest.raises(ValueError, match="unknown type"):
        expand_typed_figures(config, raise_on_error=True)


def test_posterior_2d_requires_core_fields():
    with pytest.raises(ValueError, match="requires x, y, and weight"):
        expand_posterior_2d({"type": "posterior_2d", "data": "samples", "x": {"expr": "x"}, "y": {"expr": "y"}})


def _base_profile_figure(**overrides):
    fig = {
        "name": "profile",
        "type": "profile_2d",
        "data": ["df0", "df1"],
        "x": {"expr": "xx", "lim": [0.1, 5], "scale": "log", "label": "$x$"},
        "y": {"expr": "yy", "lim": [0, 5], "label": "$y$"},
        "z": {"expr": "LogL", "label": "$\\log\\mathcal{L}$"},
    }
    fig.update(overrides)
    return fig


def test_profile_2d_default_is_bridson_natural_neighbor_pcolormesh():
    out = expand_profile_2d(_base_profile_figure(colorbar={"cmap": "viridis", "vmin": -50, "vmax": 0}))

    assert "type" not in out
    assert out["frame"]["ax"]["labels"] == {"x": "$x$", "y": "$y$"}
    assert out["frame"]["ax"]["xscale"] == "log"
    assert out["frame"]["axc"]["label"] == {"ylabel": "$\\log\\mathcal{L}$"}
    assert out["frame"]["axc"]["color"] == {"cmap": "viridis", "scale": "linear", "vmin": -50, "vmax": 0}
    assert len(out["layers"]) == 2

    profile_layer = out["layers"][0]
    assert profile_layer["method"] == "scatter"
    assert profile_layer["style"]["s"] == 0
    assert profile_layer["share_data"] == "_profile_profile_points"
    transform = profile_layer["data"][0]["transform"][0]["profile"]
    assert transform["method"] == "bridson"
    assert transform["bin"] == 100
    assert transform["objective"] == "max"
    assert transform["coordinates"]["x"] == {"expr": "xx", "lim": [0.1, 5], "scale": "log", "name": "x"}
    assert transform["coordinates"]["y"] == {"expr": "yy", "lim": [0, 5], "name": "y"}
    assert transform["coordinates"]["z"] == {"expr": "LogL", "name": "z"}

    map_layer = out["layers"][1]
    assert map_layer["method"] == "pcolormesh"
    assert map_layer["share_data"] == "_profile_profile_grid"
    assert map_layer["data"][0]["source"] == "_profile_profile_points"
    interp = map_layer["data"][0]["transform"][0]["make_interp_2d"]
    assert interp["method"] == "natural_neighbor"
    assert interp["grid"] == 500
    assert interp["as_density"] is False
    assert interp["normalize"] is False
    assert map_layer["coordinates"]["z"] == {"expr": "z"}


def test_profile_2d_bridson_cell_with_credible_region_uses_hidden_grid_for_contour():
    out = expand_profile_2d(
        _base_profile_figure(
            style_card=["a4paper_4x1", "rectcmap"],
            interp=False,
            bins=77,
            objective="mean",
            credible_region={"levels": [-10], "colors": "white", "linewidths": 0.3},
        )
    )

    assert out["frame"]["axc"]["label"] == {"xlabel": "$\\log\\mathcal{L}$"}
    assert [layer["method"] for layer in out["layers"]] == ["scatter", "voronoi", "scatter", "contour"]
    profile = out["layers"][0]["data"][0]["transform"][0]["profile"]
    assert profile["method"] == "bridson"
    assert profile["bin"] == 77
    assert profile["objective"] == "mean"
    assert out["layers"][1]["data"] == [{"source": "_profile_profile_points"}]
    hidden_interp = out["layers"][2]
    assert hidden_interp["share_data"] == "_profile_profile_grid"
    assert hidden_interp["data"][0]["transform"][0]["make_interp_2d"]["method"] == "natural_neighbor"

    contour = out["layers"][3]
    assert contour["method"] == "contour"
    assert contour["data"] == [{"source": "_profile_profile_grid"}]
    assert contour["style"]["levels"] == [-10]
    assert contour["style"]["colors"] == "white"
    assert contour["style"]["linewidths"] == 0.3


def test_profile_2d_grid_direct_and_grid_interp_modes():
    direct = expand_profile_2d(_base_profile_figure(method="grid", interp=False))
    assert [layer["method"] for layer in direct["layers"]] == ["scatter", "pcolormesh"]
    assert direct["layers"][0]["share_data"] == "_profile_profile_grid"
    assert direct["layers"][0]["data"][0]["transform"][0]["profile"]["method"] == "grid"
    assert direct["layers"][1]["data"] == [{"source": "_profile_profile_grid"}]

    interp = expand_profile_2d(_base_profile_figure(method="grid", interp=True, grid=321))
    assert [layer["method"] for layer in interp["layers"]] == ["scatter", "pcolormesh"]
    assert interp["layers"][0]["share_data"] == "_profile_profile_points"
    assert interp["layers"][1]["share_data"] == "_profile_profile_grid"
    assert interp["layers"][1]["data"][0]["transform"][0]["make_interp_2d"]["grid"] == 321


def test_profile_2d_credible_region_sigma_mode():
    out = expand_profile_2d(
        _base_profile_figure(
            credible_region={"sigma": [1, 2], "colors": ["black", "white"], "linewidths": [0.2, 0.3]}
        )
    )

    assert [layer["method"] for layer in out["layers"]] == ["scatter", "pcolormesh", "contour"]
    contour = out["layers"][2]
    assert contour["data"] == [{"source": "_profile_profile_grid"}]
    assert contour["style"]["contour_mode"] == "profile_likelihood"
    assert contour["style"]["sigma"] == [1, 2]
    assert contour["style"]["ndof"] == 2


def test_profile_likelihood_contour_mode_computes_levels_from_zmax():
    style = prepare_hpd_contour_style(
        [[-10.0, -1.0], [-4.0, 0.0]],
        [0.0, 1.0],
        [0.0, 1.0],
        {"contour_mode": "profile_likelihood", "sigma": [1, 2], "colors": ["black", "white"]},
    )

    assert style["levels"][0] == pytest.approx(-3.090037, rel=1e-5)
    assert style["levels"][1] == pytest.approx(-1.147874, rel=1e-5)
    assert style["colors"] == ["white", "black"]


def test_profile_2d_extra_layer_and_custom_output_names():
    out = expand_profile_2d(
        _base_profile_figure(
            x={"expr": "xx", "name": "x_prof", "lim": [0, 1]},
            y={"expr": "yy", "name": "y_prof", "lim": [0, 1]},
            z={"expr": "LogL", "name": "logL_prof", "label": "log likelihood"},
            extra_layers=[{"method": "scatter", "coordinates": {"x": {"expr": "xx"}, "y": {"expr": "yy"}}}],
        )
    )

    layer = out["layers"][1]
    assert layer["coordinates"] == {"x": {"expr": "x_prof"}, "y": {"expr": "y_prof"}, "z": {"expr": "logL_prof"}}
    transform = out["layers"][0]["data"][0]["transform"][0]["profile"]
    assert transform["coordinates"]["x"]["name"] == "x_prof"
    assert transform["coordinates"]["y"]["name"] == "y_prof"
    assert transform["coordinates"]["z"]["name"] == "logL_prof"
    assert out["layers"][2]["data"] == [{"source": ["df0", "df1"]}]


def test_profile_2d_requires_core_fields():
    with pytest.raises(ValueError, match="requires x, y, and z"):
        expand_profile_2d({"type": "profile_2d", "data": "samples", "x": {"expr": "x"}, "y": {"expr": "y"}})

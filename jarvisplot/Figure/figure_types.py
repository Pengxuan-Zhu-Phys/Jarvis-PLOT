from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


_POSTERIOR_DENSITY_KEYS = {
    "method",
    "bins",
    "bin",
    "grid",
    "seed",
    "voronoi",
    "adaptive",
    "kde",
    "normalize",
    "output",
    "nan_policy",
    "_mesh_debug",
    "diagnostics",
}

_PROFILE_KEYS = {
    "fill_empty",
    "empty_value",
    "grid_points",
    "seed",
}


def _coord_label(spec: Any, fallback: str) -> str:
    if isinstance(spec, Mapping):
        value = spec.get("label", spec.get("name", spec.get("expr", fallback)))
    else:
        value = spec if spec is not None else fallback
    text = str(value).strip()
    return text or fallback


def _coord_transform_spec(spec: Any, fallback: str) -> dict[str, Any]:
    if isinstance(spec, Mapping):
        out: dict[str, Any] = {"expr": spec.get("expr", fallback)}
        for key in ("lim", "limits", "scale", "name"):
            if key in spec:
                out[key] = spec[key]
        return out
    return {"expr": fallback if spec is None else spec}


def _weight_transform_spec(spec: Any) -> dict[str, Any]:
    if isinstance(spec, Mapping):
        out = {"expr": spec.get("expr", spec.get("name", "weight"))}
        if "name" in spec:
            out["name"] = spec["name"]
        return out
    return {"expr": "weight" if spec is None else spec}


def _copy_if_present(src: Mapping[str, Any], dst: dict[str, Any], key: str) -> None:
    if key in src:
        dst[key] = deepcopy(src[key])


def _density_output_name(density: Mapping[str, Any]) -> str:
    output = density.get("output", "density")
    if isinstance(output, Mapping):
        return str(output.get("z", output.get("density", "density"))).strip() or "density"
    return str(output).strip() or "density"


def _colorbar_label_axis(style_tokens: Any) -> str:
    tokens = style_tokens if isinstance(style_tokens, (list, tuple)) else [style_tokens]
    joined = " ".join(str(item).lower() for item in tokens if item is not None)
    if "4x1" in joined:
        return "xlabel"
    return "ylabel"


def _is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"", "auto", "none", "null"}


def _build_frame(info: Mapping[str, Any], style_tokens: Any, *, default_colorbar_label: str = "density", default_vmin: Any = 0.0) -> dict[str, Any]:
    x = info.get("x", {})
    y = info.get("y", {})
    colorbar = info.get("colorbar", {})
    if colorbar is False:
        colorbar = {}
    if not isinstance(colorbar, Mapping):
        colorbar = {}

    ax: dict[str, Any] = {
        "labels": {
            "x": _coord_label(x, "x"),
            "y": _coord_label(y, "y"),
        },
        "xscale": str(x.get("scale", "linear")).strip().lower() if isinstance(x, Mapping) else "linear",
        "yscale": str(y.get("scale", "linear")).strip().lower() if isinstance(y, Mapping) else "linear",
    }
    if isinstance(x, Mapping) and x.get("lim", x.get("limits", None)) is not None:
        ax["xlim"] = deepcopy(x.get("lim", x.get("limits")))
    if isinstance(y, Mapping) and y.get("lim", y.get("limits", None)) is not None:
        ax["ylim"] = deepcopy(y.get("lim", y.get("limits")))

    label = colorbar.get("label", default_colorbar_label)
    label_axis = str(colorbar.get("label_axis", _colorbar_label_axis(style_tokens))).strip() or "ylabel"
    if label_axis not in {"xlabel", "ylabel"}:
        label_axis = "ylabel"

    color: dict[str, Any] = {
        "cmap": colorbar.get("cmap", "jarvis_rainbow2_r"),
        "scale": colorbar.get("scale", "linear"),
    }
    vmin = colorbar.get("vmin", default_vmin)
    vmax = colorbar.get("vmax", "auto")
    if not _is_auto(vmin):
        color["vmin"] = vmin
    if not _is_auto(vmax):
        color["vmax"] = vmax

    axc = {
        "label": {label_axis: label},
        "color": color,
    }
    return {"ax": ax, "axc": axc}


def _posterior_density_config(info: Mapping[str, Any]) -> dict[str, Any]:
    density = info.get("density", {})
    if not isinstance(density, Mapping):
        density = {}

    cfg: dict[str, Any] = {
        "method": density.get("method", "voronoi"),
        "x": _coord_transform_spec(info.get("x", {}), "x"),
        "y": _coord_transform_spec(info.get("y", {}), "y"),
        "weight": _weight_transform_spec(info.get("weight", {})),
    }
    for key in _POSTERIOR_DENSITY_KEYS:
        _copy_if_present(density, cfg, key)
    cfg.setdefault("method", "voronoi")
    return cfg


def _profile_output_name(spec: Any, default: str) -> str:
    if isinstance(spec, Mapping):
        text = str(spec.get("name", default)).strip()
        return text or default
    return default


def _profile_coord_spec(spec: Any, key: str) -> dict[str, Any]:
    out = _coord_transform_spec(spec, key)
    out["name"] = _profile_output_name(spec, key)
    return out


def _profile_options(info: Mapping[str, Any]) -> dict[str, Any]:
    legacy = info.get("profile", {})
    if not isinstance(legacy, Mapping):
        legacy = {}

    method_raw = str(info.get("method", legacy.get("method", "bridson"))).strip().lower()
    interp_default: Any = True
    aliases = {
        "cell": "bridson",
        "voronoi": "bridson",
        "mesh": "grid",
        "heatmap": "grid",
        "pcolormesh": "grid",
    }
    if method_raw in {"cell", "voronoi"}:
        interp_default = False
    elif method_raw == "pcolormesh" and "interp" not in legacy and "interp" not in info:
        interp_default = False
    method = aliases.get(method_raw, method_raw)
    if method not in {"bridson", "grid"}:
        raise ValueError("profile_2d method must be one of: bridson, grid.")

    interp_raw = info.get("interp", legacy.get("interp", interp_default))
    interp_cfg = dict(interp_raw) if isinstance(interp_raw, Mapping) else {}
    interp_enabled = False if interp_raw is False else bool(interp_raw)
    grid = info.get("grid", legacy.get("grid", interp_cfg.get("grid", 500)))

    return {
        "method": method,
        "bins": info.get("bins", info.get("bin", legacy.get("bins", legacy.get("bin", 100)))),
        "objective": info.get("objective", legacy.get("objective", "max")),
        "seed": info.get("seed", legacy.get("seed", None)),
        "grid_points": info.get("grid_points", legacy.get("grid_points", "rect")),
        "fill_empty": info.get("fill_empty", legacy.get("fill_empty", None)),
        "empty_value": info.get("empty_value", legacy.get("empty_value", None)),
        "interp": interp_enabled,
        "interp_cfg": interp_cfg,
        "grid": grid,
    }


def _profile_transform_config(info: Mapping[str, Any], opts: Mapping[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "method": opts.get("method", "bridson"),
        "bin": opts.get("bins", 100),
        "objective": opts.get("objective", "max"),
        "grid_points": opts.get("grid_points", "rect"),
        "coordinates": {
            "x": _profile_coord_spec(info.get("x", {}), "x"),
            "y": _profile_coord_spec(info.get("y", {}), "y"),
            "z": _profile_coord_spec(info.get("z", {}), "z"),
        },
    }
    for key in _PROFILE_KEYS:
        value = opts.get(key, None)
        if value is not None:
            cfg[key] = deepcopy(value)
    return cfg


def _profile_interp_transform(info: Mapping[str, Any], opts: Mapping[str, Any]) -> dict[str, Any]:
    interp_cfg = dict(opts.get("interp_cfg", {})) if isinstance(opts.get("interp_cfg", {}), Mapping) else {}
    x_name = _profile_output_name(info.get("x", {}), "x")
    y_name = _profile_output_name(info.get("y", {}), "y")
    z_name = _profile_output_name(info.get("z", {}), "z")
    x_spec = info.get("x", {})
    y_spec = info.get("y", {})
    transform = {
        "method": interp_cfg.get("method", "natural_neighbor"),
        "coordinates": {
            "x": {
                "expr": x_name,
                "name": x_name,
                "scale": x_spec.get("scale", "linear") if isinstance(x_spec, Mapping) else "linear",
            },
            "y": {
                "expr": y_name,
                "name": y_name,
                "scale": y_spec.get("scale", "linear") if isinstance(y_spec, Mapping) else "linear",
            },
            "z": {"expr": z_name, "name": z_name},
        },
        "grid": opts.get("grid", interp_cfg.get("grid", 500)),
        "nan_policy": interp_cfg.get("nan_policy", "strict"),
        "as_density": False,
        "normalize": False,
    }
    if isinstance(x_spec, Mapping) and x_spec.get("lim", x_spec.get("limits", None)) is not None:
        transform["coordinates"]["x"]["lim"] = deepcopy(x_spec.get("lim", x_spec.get("limits")))
    if isinstance(y_spec, Mapping) and y_spec.get("lim", y_spec.get("limits", None)) is not None:
        transform["coordinates"]["y"]["lim"] = deepcopy(y_spec.get("lim", y_spec.get("limits")))
    for key, value in interp_cfg.items():
        if key not in {"method", "grid", "nan_policy"}:
            transform[key] = deepcopy(value)
    return transform


def _hpd_style(hpd: Any) -> dict[str, Any] | None:
    if hpd is False or hpd is None:
        return None
    if isinstance(hpd, Mapping) and hpd.get("enabled", True) is False:
        return None
    style = {
        "contour_mode": "posterior_hpd",
        "masses": [0.6827, 0.9545],
        "labels": ["$1\\sigma$", "$2\\sigma$"],
        "colors": ["black", "white"],
        "linestyles": ["solid", "solid"],
        "linewidths": [0.2, 0.2],
        "zorder": 20,
    }
    if isinstance(hpd, Mapping):
        for key, value in hpd.items():
            if key == "enabled":
                continue
            style[key] = deepcopy(value)
    return style


def _normalize_extra_layer(layer: Mapping[str, Any], default_source: Any) -> dict[str, Any]:
    out = deepcopy(dict(layer))
    out.setdefault("axes", "ax")
    if "data" not in out:
        if "source" in out:
            out["data"] = [{"source": out.pop("source")}]
        else:
            out["data"] = [{"source": deepcopy(default_source)}]
    elif isinstance(out["data"], Mapping):
        out["data"] = [deepcopy(out["data"])]
    elif isinstance(out["data"], (str, list, tuple)) and not (
        isinstance(out["data"], list) and out["data"] and isinstance(out["data"][0], Mapping)
    ):
        out["data"] = [{"source": deepcopy(out["data"])}]
    return out


def _style_tokens(value: Any) -> list[Any]:
    if value is None:
        return ["a4paper_2x1", "rectcmap"]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    out = deepcopy(base)
    if not isinstance(override, Mapping):
        return out
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def expand_posterior_2d(info: Mapping[str, Any]) -> dict[str, Any]:
    """Expand a `type: posterior_2d` figure into ordinary Jarvis-PLOT layers."""
    out = deepcopy(dict(info))
    out.pop("type", None)

    name = str(info.get("name", "posterior_2d")).strip() or "posterior_2d"
    source = deepcopy(info.get("data"))
    if source is None:
        raise ValueError("posterior_2d requires a data source.")
    if "x" not in info or "y" not in info or "weight" not in info:
        raise ValueError("posterior_2d requires x, y, and weight.")

    style_tokens = _style_tokens(deepcopy(info.get("style_card", info.get("style", None))))
    frame = _build_frame(info, style_tokens)
    frame = _deep_merge(frame, info.get("frame"))

    density_cfg = _posterior_density_config(info)
    z_name = _density_output_name(density_cfg)
    share_name = str(info.get("share_data", f"_{name}_posterior_density")).strip() or f"_{name}_posterior_density"

    layers = [
        {
            "name": "_density",
            "data": [{"source": source, "transform": [{"posterior_density": density_cfg}]}],
            "share_data": share_name,
            "axes": "ax",
            "method": "pcolormesh",
            "coordinates": {"x": {"expr": "x"}, "y": {"expr": "y"}, "z": {"expr": z_name}},
            "style": {"edgecolor": "none", "linewidth": 0},
            "colorbar": "axc",
        }
    ]

    hpd_style = _hpd_style(info.get("hpd", {}))
    if hpd_style is not None:
        layers.append(
            {
                "name": "_hpd",
                "data": [{"source": share_name}],
                "axes": "ax",
                "method": "contour",
                "coordinates": {"x": {"expr": "x"}, "y": {"expr": "y"}, "z": {"expr": z_name}},
                "style": hpd_style,
            }
        )

    extra_layers = info.get("extra_layers", [])
    if isinstance(extra_layers, list):
        for layer in extra_layers:
            if isinstance(layer, Mapping):
                layers.append(_normalize_extra_layer(layer, source))

    for key in (
        "style_card",
        "data",
        "x",
        "y",
        "weight",
        "density",
        "colorbar",
        "hpd",
        "extra_layers",
        "share_data",
    ):
        out.pop(key, None)

    out["name"] = name
    out["style"] = style_tokens
    out["frame"] = frame
    out["layers"] = layers
    return out


def _profile_hidden_style() -> dict[str, Any]:
    return {"marker": ".", "s": 0, "color": "none"}


def _profile_render_style(info: Mapping[str, Any], render_method: str) -> dict[str, Any]:
    base: dict[str, Any]
    if render_method == "pcolormesh":
        base = {"edgecolor": "none", "linewidth": 0}
    elif render_method == "voronoi":
        base = {}
    else:
        base = {}
    style = info.get(render_method, {})
    if isinstance(style, Mapping):
        base.update(deepcopy(dict(style)))
    return base


def _credible_region_style(region: Any) -> dict[str, Any] | None:
    if region is None or region is False:
        return None
    if isinstance(region, Mapping) and region.get("enabled", True) is False:
        return None
    style = {
        "colors": ["black", "white"],
        "linestyles": "solid",
        "linewidths": [0.2, 0.2],
        "zorder": 20,
    }
    if isinstance(region, Mapping):
        if "levels" in region:
            style["levels"] = deepcopy(region["levels"])
        elif "sigma" in region:
            style["contour_mode"] = "profile_likelihood"
            style["sigma"] = deepcopy(region["sigma"])
            style["ndof"] = deepcopy(region.get("ndof", 2))
        for key, value in region.items():
            if key in {"enabled", "levels", "sigma", "ndof"}:
                continue
            style[key] = deepcopy(value)
    else:
        style["contour_mode"] = "profile_likelihood"
        style["sigma"] = [1, 2]
    return style


def expand_profile_2d(info: Mapping[str, Any]) -> dict[str, Any]:
    """Expand a `type: profile_2d` figure into ordinary Jarvis-PLOT layers."""
    out = deepcopy(dict(info))
    out.pop("type", None)

    name = str(info.get("name", "profile_2d")).strip() or "profile_2d"
    source = deepcopy(info.get("data"))
    if source is None:
        raise ValueError("profile_2d requires a data source.")
    if "x" not in info or "y" not in info or "z" not in info:
        raise ValueError("profile_2d requires x, y, and z.")

    opts = _profile_options(info)
    reduce_method = str(opts["method"])
    interp_enabled = bool(opts["interp"])
    render_method = "pcolormesh" if interp_enabled or reduce_method == "grid" else "voronoi"

    style_tokens = _style_tokens(deepcopy(info.get("style_card", info.get("style", None))))
    z_label = _coord_label(info.get("z", {}), "z")
    frame = _build_frame(info, style_tokens, default_colorbar_label=z_label, default_vmin="auto")
    frame = _deep_merge(frame, info.get("frame"))

    x_name = _profile_output_name(info.get("x", {}), "x")
    y_name = _profile_output_name(info.get("y", {}), "y")
    z_name = _profile_output_name(info.get("z", {}), "z")
    points_share = f"_{name}_profile_points"
    grid_share = f"_{name}_profile_grid"
    reduced_share = grid_share if reduce_method == "grid" and not interp_enabled else points_share

    profile_layer = {
        "name": f"_profile_{reduce_method}",
        "data": [{"source": source, "transform": [{"profile": _profile_transform_config(info, opts)}]}],
        "share_data": reduced_share,
        "axes": "ax",
        "method": "scatter",
        "coordinates": {"x": {"expr": x_name}, "y": {"expr": y_name}},
        "style": _profile_hidden_style(),
    }
    layers = [profile_layer]

    if interp_enabled:
        layers.append(
            {
                "name": "_profile_interp",
                "data": [{"source": reduced_share, "transform": [{"make_interp_2d": _profile_interp_transform(info, opts)}]}],
                "share_data": grid_share,
                "axes": "ax",
                "method": "pcolormesh",
                "coordinates": {"x": {"expr": x_name}, "y": {"expr": y_name}, "z": {"expr": z_name}},
                "style": _profile_render_style(info, "pcolormesh"),
                "colorbar": "axc",
            }
        )
        contour_source = grid_share
    else:
        layers.append(
            {
                "name": "_profile_cells",
                "data": [{"source": reduced_share}],
                "axes": "ax",
                "method": render_method,
                "coordinates": {"x": {"expr": x_name}, "y": {"expr": y_name}, "z": {"expr": z_name}},
                "style": _profile_render_style(info, render_method),
                "colorbar": "axc",
            }
        )
        contour_source = reduced_share

    region_style = _credible_region_style(info.get("credible_region", info.get("contour", None)))
    if region_style is not None:
        if not interp_enabled and reduce_method == "bridson":
            layers.append(
                {
                    "name": "_profile_contour_grid",
                    "data": [{"source": reduced_share, "transform": [{"make_interp_2d": _profile_interp_transform(info, opts)}]}],
                    "share_data": grid_share,
                    "axes": "ax",
                    "method": "scatter",
                    "coordinates": {"x": {"expr": x_name}, "y": {"expr": y_name}},
                    "style": _profile_hidden_style(),
                }
            )
            contour_source = grid_share
        layers.append(
            {
                "name": "_credible_region",
                "data": [{"source": contour_source}],
                "axes": "ax",
                "method": "contour",
                "coordinates": {"x": {"expr": x_name}, "y": {"expr": y_name}, "z": {"expr": z_name}},
                "style": region_style,
            }
        )

    extra_layers = info.get("extra_layers", [])
    if isinstance(extra_layers, list):
        for layer in extra_layers:
            if isinstance(layer, Mapping):
                layers.append(_normalize_extra_layer(layer, source))

    for key in (
        "style_card",
        "data",
        "x",
        "y",
        "z",
        "profile",
        "method",
        "bins",
        "bin",
        "objective",
        "seed",
        "grid_points",
        "fill_empty",
        "empty_value",
        "interp",
        "grid",
        "colorbar",
        "contour",
        "credible_region",
        "extra_layers",
        "share_data",
        "pcolormesh",
        "voronoi",
    ):
        out.pop(key, None)

    out["name"] = name
    out["style"] = style_tokens
    out["frame"] = frame
    out["layers"] = layers
    return out


def expand_figure_type(info: Any) -> Any:
    if not isinstance(info, Mapping):
        return info
    fig_type = str(info.get("type", "")).strip().lower()
    if fig_type == "posterior_2d":
        return expand_posterior_2d(info)
    if fig_type == "profile_2d":
        return expand_profile_2d(info)
    return deepcopy(dict(info))


def expand_figure_types_in_config(config: Any, logger=None) -> Any:
    if not isinstance(config, dict):
        return config
    figures = config.get("Figures", [])
    if not isinstance(figures, list):
        return config
    expanded = []
    for fig in figures:
        try:
            expanded.append(expand_figure_type(fig))
        except Exception as exc:
            if logger is not None:
                try:
                    name = fig.get("name", "<unknown>") if isinstance(fig, Mapping) else "<unknown>"
                    logger.warning(f"Figure type expansion failed for {name}: {exc}")
                except Exception:
                    pass
            expanded.append(fig)
    config["Figures"] = expanded
    return config

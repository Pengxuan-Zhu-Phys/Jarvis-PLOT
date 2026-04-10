from __future__ import annotations

from copy import deepcopy
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessor_runtime import add_column, filter_df, sort_by
from .profile_runtime import grid_profiling, profiling
from .method_registry import resolve_callable
from .colorbar_runtime import collect_and_attach_colorbar
from .interp_natural_neighbor import resolve_backend


def _resolve_csv_export_path(fig, target):
    if isinstance(target, dict):
        raw = target.get("path", target.get("file", target.get("target", target.get("value", ""))))
    else:
        raw = target
    path = str(raw).strip()
    if not path:
        raise ValueError("tocsv requires a non-empty path")
    return Path(fig.load_path(path))


def _save_dataframe_csv(fig, df, target):
    out_path = _resolve_csv_export_path(fig, target)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df, pd.DataFrame):
        df.to_csv(out_path, index=False)
    else:
        pd.DataFrame(df).to_csv(out_path, index=False)
    if fig.logger:
        fig.logger.debug(f"Saved transformed dataframe to CSV -> {out_path}")
    return out_path


def _coerce_positive_int(value, default: int) -> int:
    try:
        n = int(value)
        if n > 0:
            return n
    except Exception:
        pass
    return int(default)


def _resolve_interp_grid_size(interp_cfg: dict | None, x_size: int, y_size: int) -> tuple[int, int]:
    default_n = 500
    if not isinstance(interp_cfg, dict):
        return default_n, default_n

    base = interp_cfg.get("bin", interp_cfg.get("resolution", default_n))
    nx = interp_cfg.get("nx", interp_cfg.get("xbin", base))
    ny = interp_cfg.get("ny", interp_cfg.get("ybin", base))

    grid_cfg = interp_cfg.get("grid", None)
    if isinstance(grid_cfg, dict):
        nx = grid_cfg.get("nx", grid_cfg.get("xbin", nx))
        ny = grid_cfg.get("ny", grid_cfg.get("ybin", ny))

    return _coerce_positive_int(nx, default_n), _coerce_positive_int(ny, default_n)


def _resolve_query_axis(
    ax,
    axis_name: str,
    values,
    *,
    npts: int,
    interp_cfg: dict | None = None,
    logger=None,
):
    values = np.asarray(values, dtype=float).reshape(-1)
    scale = str(ax.get_xscale() if axis_name == "x" else ax.get_yscale()).lower()

    bounds = None
    if isinstance(interp_cfg, dict):
        bounds = interp_cfg.get(f"{axis_name}lim", None)

    if bounds is None:
        try:
            bounds = ax.get_xlim() if axis_name == "x" else ax.get_ylim()
        except Exception:
            bounds = None

    finite = values[np.isfinite(values)]
    if bounds is None:
        if finite.size:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
        else:
            lo, hi = 0.0, 1.0
    else:
        try:
            lo = float(bounds[0])
            hi = float(bounds[1])
        except Exception:
            lo, hi = 0.0, 1.0

    if not np.isfinite(lo) or not np.isfinite(hi):
        if finite.size:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
        else:
            lo, hi = 0.0, 1.0

    if lo == hi:
        if scale == "log" and lo > 0:
            hi = lo * 10.0
        else:
            hi = lo + 1.0

    if scale == "log":
        positive = finite[finite > 0]
        if lo <= 0 or hi <= 0:
            if positive.size:
                lo = float(np.min(positive))
                hi = float(np.max(positive))
            else:
                if logger:
                    logger.warning(
                        f"natural_neighbor: {axis_name}-axis is log-scaled but has no positive "
                        "finite bounds; falling back to a linear grid"
                    )
                return np.linspace(lo, hi, npts)
        lo = max(lo, np.finfo(float).tiny)
        hi = max(hi, lo * 10.0)
        return np.geomspace(lo, hi, npts)

    return np.linspace(lo, hi, npts)


def _build_contour_query_grid(ax, x, y, *, interp_cfg: dict | None = None, logger=None):
    nx, ny = _resolve_interp_grid_size(interp_cfg, np.size(x), np.size(y))
    xq = _resolve_query_axis(ax, "x", x, npts=nx, interp_cfg=interp_cfg, logger=logger)
    yq = _resolve_query_axis(ax, "y", y, npts=ny, interp_cfg=interp_cfg, logger=logger)
    return np.meshgrid(xq, yq)


def _log_natural_neighbor_diagnostics(logger, diag) -> None:
    if logger is None or diag is None:
        return
    try:
        if getattr(diag, "degenerate_input", False):
            logger.warning("natural_neighbor: input data are too sparse or degenerate for full interpolation")
        if getattr(diag, "all_nan_cores", False):
            logger.warning("natural_neighbor: all input z cores are NaN")
        exact_duplicate_groups = getattr(diag, "exact_duplicate_groups", 0)
        near_duplicate_groups = getattr(diag, "near_duplicate_groups", 0)
        merged_points = getattr(diag, "merged_points", 0)
        if any(int(v) for v in (exact_duplicate_groups, near_duplicate_groups, merged_points)):
            logger.debug(
                "natural_neighbor: merged {} exact-duplicate groups, {} near-duplicate groups, {} points merged".format(
                    int(exact_duplicate_groups),
                    int(near_duplicate_groups),
                    int(merged_points),
                )
            )
        nominal_spacing = getattr(diag, "nominal_point_spacing", None)
        if isinstance(nominal_spacing, (int, float)) and nominal_spacing > 0:
            logger.debug(
                "natural_neighbor: nominal point spacing {:.6g}".format(float(nominal_spacing))
            )
        vertex_tolerance = getattr(diag, "vertex_tolerance", None)
        if isinstance(vertex_tolerance, (int, float)) and vertex_tolerance > 0:
            logger.debug(
                "natural_neighbor: vertex tolerance {:.6g}".format(float(vertex_tolerance))
            )
        boundary_tolerance = getattr(diag, "boundary_tolerance", None)
        if isinstance(boundary_tolerance, (int, float)) and boundary_tolerance > 0:
            logger.debug(
                "natural_neighbor: boundary tolerance {:.6g}".format(float(boundary_tolerance))
            )
        if getattr(diag, "masked_by_nan", 0):
            logger.warning(
                "natural_neighbor: {} query points were masked because a contributing core is NaN".format(
                    int(getattr(diag, "masked_by_nan", 0))
                )
            )
        if getattr(diag, "outside_hull", 0):
            logger.debug(
                "natural_neighbor: {} query points lie outside the convex hull".format(
                    int(getattr(diag, "outside_hull", 0))
                )
            )
        if getattr(diag, "exact_hits", 0):
            logger.debug(
                "natural_neighbor: {} query points matched an input core exactly".format(
                    int(getattr(diag, "exact_hits", 0))
                )
            )
        if getattr(diag, "cavity_triangles", 0):
            logger.debug(
                "natural_neighbor: visited {} cavity triangles".format(
                    int(getattr(diag, "cavity_triangles", 0))
                )
            )
        area_of_embedded_polygon = getattr(diag, "area_of_embedded_polygon", None)
        if isinstance(area_of_embedded_polygon, (int, float)) and area_of_embedded_polygon != 0:
            logger.debug(
                "natural_neighbor: area of embedded polygon {:.6f}".format(
                    float(area_of_embedded_polygon)
                )
            )
        barycentric_coordinate_deviation = getattr(diag, "barycentric_coordinate_deviation", None)
        if isinstance(barycentric_coordinate_deviation, (int, float)) and barycentric_coordinate_deviation != 0:
            logger.debug(
                "natural_neighbor: barycentric coordinate deviation {:.6e}".format(
                    float(barycentric_coordinate_deviation)
                )
            )
        build_seconds = getattr(diag, "build_seconds", None)
        eval_seconds = getattr(diag, "eval_seconds", None)
        if isinstance(build_seconds, (int, float)) and build_seconds > 0:
            logger.debug(f"natural_neighbor: build time {float(build_seconds):.4f}s")
        if isinstance(eval_seconds, (int, float)) and eval_seconds > 0:
            logger.debug(f"natural_neighbor: eval time {float(eval_seconds):.4f}s")
    except Exception:
        pass


def _prepare_contour_args(fig, ax, method_key: str, style: dict, coor: dict):
    """Prepare positional contour/contourf args and remove interpolation config."""
    interp_cfg = style.pop("interp", None)
    interp_cfg = dict(interp_cfg) if isinstance(interp_cfg, dict) else None

    x = np.asarray(coor.get("x"), dtype=float)
    y = np.asarray(coor.get("y"), dtype=float)
    z = np.asarray(coor.get("z"), dtype=float)

    # Already-gridded inputs should pass straight through.
    if x.ndim == 2 and y.ndim == 2 and z.ndim == 2 and x.shape == y.shape == z.shape:
        return (x, y, np.ma.masked_invalid(z)), style

    if x.ndim == 1 and y.ndim == 1 and z.ndim == 2:
        if z.shape == (y.size, x.size):
            return (x, y, np.ma.masked_invalid(z)), style
        if z.shape == (x.size, y.size):
            return (x, y, np.ma.masked_invalid(z.T)), style
        raise ValueError(
            f"{method_key} expects Z to have shape (len(y), len(x)) or the transpose when X/Y are 1D"
        )

    if interp_cfg is None:
        raise ValueError(
            f"{method_key} requires gridded X/Y/Z inputs or style.interp.method: natural_neighbor"
        )

    backend_name = str(interp_cfg.get("method", "natural_neighbor")).strip()
    nan_policy = str(interp_cfg.get("nan_policy", "strict"))
    diagnostics = bool(interp_cfg.get("diagnostics", False))
    backend_options = interp_cfg.get("backend_options", None)
    X, Y = _build_contour_query_grid(ax, x, y, interp_cfg=interp_cfg, logger=getattr(fig, "logger", None))
    try:
        backend = resolve_backend(backend_name)
    except Exception as exc:
        raise ValueError(f"Unsupported contour interpolation backend: {backend_name!r}") from exc
    Z = backend(
        x,
        y,
        z,
        X,
        Y,
        nan_policy=nan_policy,
        diagnostics=diagnostics,
        backend_options=backend_options,
    )
    _log_natural_neighbor_diagnostics(getattr(fig, "logger", None), getattr(backend, "last_diagnostics", None))
    Z = np.asarray(Z, dtype=float)

    if not np.isfinite(Z).any():
        if getattr(fig, "logger", None):
            fig.logger.warning(
                f"natural_neighbor returned only NaNs for layer '{getattr(fig, 'name', '<noname>')}'"
            )
        return None, style

    return (X, Y, np.ma.masked_invalid(Z)), style


def _prepare_jpcontour_style(
    fig,
    method_key: str,
    style: dict,
    coor: dict,
    df,
    layer_name: str = "",
    coord_keys: tuple[str, ...] | None = None,
    required_keys: tuple[str, ...] = (),
    include_diagnostics: bool = False,
):
    """Prepare scattered jpcontour/jpcontourf/jpfield kwargs and map style.interp."""
    call_style = dict(style)
    interp_cfg = call_style.pop("interp", None)
    interp_cfg = dict(interp_cfg) if isinstance(interp_cfg, dict) else None
    if interp_cfg is not None:
        call_style.setdefault("interp_method", interp_cfg.get("method", "natural_neighbor"))
        for key in ("bin", "nx", "ny", "xlim", "ylim", "nan_policy", "backend_options"):
            if key in interp_cfg and key not in call_style:
                call_style[key] = interp_cfg[key]
        if include_diagnostics and "diagnostics" in interp_cfg and "diagnostics" not in call_style:
            call_style["diagnostics"] = bool(interp_cfg.get("diagnostics", False))

    coords = {}
    items = coor.items() if coord_keys is None else ((kk, coor[kk]) for kk in coord_keys if kk in coor)
    for kk, vv in items:
        if isinstance(vv, dict) and "expr" in vv:
            if df is None:
                raise ValueError(
                    f"Layer '{layer_name}' defines expression-based "
                    f"coordinate for '{kk}' but has no data source."
                )
            coords[kk] = fig._eval_series(df, vv)
        else:
            coords[kk] = vv

    for kk in required_keys:
        if kk not in coords:
            required_list = ", ".join(required_keys) if required_keys else "required coordinates"
            raise ValueError(f"{method_key} layer must define coordinates: {{{required_list}}}")

    call_style.update(coords)
    return call_style


def load_layer_data(fig, layer):
    lyinfo = layer.get("data", False)
    lycomb = layer.get("combine", "concat")
    share_name = layer.get("share_data")
    layer_demand = None
    if fig.preprocessor is not None:
        try:
            layer_demand = fig.preprocessor.layer_demand_columns(layer)
        except Exception:
            layer_demand = None

    if fig.preprocessor is not None and share_name:
        try:
            named = fig.preprocessor.load_named_layer(share_name, layer, demand_columns=layer_demand)
            if named is not None:
                fig.logger.debug(f"Loaded share_data '{share_name}' from named cache.")
                return named, None
        except Exception as e:
            fig.logger.debug(f"Named share_data cache load failed: {e}")

    if lyinfo:
        if lycomb == "concat":
            dts = []
            cache_keys = []
            for ds in lyinfo:
                src = ds.get("source")
                use_cache = bool(ds.get("cache", True))
                fig.logger.debug("Loading layer data source -> {}".format(src))
                if src and fig.context:
                    if fig.preprocessor is not None:
                        dt, cache_key, _ = fig.preprocessor.run_pipeline(
                            source=src,
                            transform=ds.get("transform", None),
                            combine="concat",
                            use_cache=use_cache,
                            demand_columns=layer_demand,
                        )
                        if dt is not None:
                            dts.append(dt)
                            cache_keys.append(cache_key if isinstance(cache_key, str) and use_cache else None)
                    else:
                        if isinstance(src, (list, tuple)):
                            fig.logger.debug("loading datasets in list mode")
                            dsrc = []
                            for srcitem in src:
                                fig.logger.debug("loading layer data source item -> {}".format(srcitem))
                                dt = deepcopy(fig.context.get(srcitem))
                                dsrc.append(dt)
                            dfsrcs = pd.concat(dsrc, ignore_index=False)
                            dt = load_bool_df(fig, dfsrcs, ds.get("transform", None))
                            dts.append(dt)
                        elif fig.context.get(src) is not None:
                            dt = deepcopy(fig.context.get(src))
                            dt = load_bool_df(fig, dt, ds.get("transform", None))
                            dts.append(dt)
                else:
                    fig.logger.error("DataSet -> {} not specified".format(src))
            if len(dts) == 0:
                return None, None
            combined = fig._concat_loaded_data(dts)
            cache_ref = None
            if len(dts) == 1 and combined is dts[0] and len(cache_keys) == 1:
                cache_ref = cache_keys[0]
            return combined, cache_ref
        elif lycomb == "seperate":
            dts = {}
            for ds in lyinfo:
                src = ds.get("source")
                label = ds.get("label")
                use_cache = bool(ds.get("cache", True))
                fig.logger.debug("Loading layer data source -> {}".format(src))
                if src and fig.context:
                    if fig.preprocessor is not None:
                        dt, _, _ = fig.preprocessor.run_pipeline(
                            source=src,
                            transform=ds.get("transform", None),
                            combine="concat",
                            use_cache=use_cache,
                            demand_columns=layer_demand,
                        )
                    else:
                        if fig.context.get(src) is None:
                            dt = None
                        else:
                            dt = deepcopy(fig.context.get(src))
                            dt = load_bool_df(fig, dt, ds.get("transform", None))
                    if dt is not None:
                        dts[label] = dt
                else:
                    fig.logger.error("DataSet -> {} not specified".format(src))
            if len(dts) == 0:
                return None, None
            return dts, None
    return None, None


def load_bool_df(fig, df, transform):
    if fig.preprocessor is not None:
        return fig.preprocessor.apply_runtime_transforms(df, transform)
    if transform is None:
        return df
    elif not isinstance(transform, list):
        fig.logger.error(f"illegal transform format, list type needed -> {transform!r}")
        return df
    else:
        for trans in transform:
            fig.logger.debug("Applying the transform ... ")
            if "filter" in trans.keys():
                fig.logger.debug("Before filtering -> {}".format(df.shape))
                df = filter_df(df, trans["filter"], fig.logger)
                fig.logger.debug("After filtering -> {}".format(df.shape))
            elif "profile" in trans.keys():
                df = profiling(df, trans["profile"], fig.logger)
                fig.logger.debug("After profiling -> {}".format(df.shape))
            elif "grid_profile" in trans.keys():
                cfg = trans.get("grid_profile", {})
                if isinstance(cfg, dict):
                    cfg = cfg.copy()
                    cfg.setdefault("method", "grid")
                else:
                    cfg = {"method": "grid"}
                df = grid_profiling(df, cfg, fig.logger)
                fig.logger.debug("After grid profiling -> {}".format(df.shape))
            elif "sortby" in trans.keys():
                df = sort_by(df, trans["sortby"], fig.logger)
                fig.logger.debug("After sortby -> {}".format(df.shape))
            elif "add_column" in trans.keys():
                df = add_column(df, trans["add_column"], fig.logger)
                fig.logger.debug("After Add-column -> {}".format(df.shape))
            elif "tocsv" in trans.keys() or "to_csv" in trans.keys():
                _save_dataframe_csv(fig, df, trans.get("tocsv", trans.get("to_csv")))

        return df


def load_layer_runtime_data(fig, layer_info):
    if layer_info.get("data_loaded", False):
        return layer_info.get("data")
    layer = layer_info.get("layer_spec", {})
    loaded = load_layer_data(fig, layer)
    cache_ref = None
    if isinstance(loaded, tuple) and len(loaded) == 2:
        data, cache_ref = loaded
    else:
        data = loaded
    layer_info["data"] = data
    layer_info["data_loaded"] = True
    layer_info["share_cache_ref"] = cache_ref
    fig._store_share_data_if_needed(layer, data, cache_ref=cache_ref)
    return data


def release_layer_runtime_data(fig, layer_info, consume_sources: bool = True):
    layer_info["data"] = None
    layer_info["data_loaded"] = False

    if consume_sources:
        share_name = layer_info.get("share_name")
        if share_name and fig.context is not None and fig.context.remaining_uses(share_name) <= 0:
            fig.context.invalidate(share_name)

        if fig.context is not None:
            for ref in layer_info.get("source_refs", []):
                remain = fig.context.consume(ref)
                if remain > 0 and fig.preprocessor is not None:
                    try:
                        if fig.preprocessor.should_release_between_uses(ref):
                            fig.context.invalidate(ref)
                    except Exception as e:
                        fig.logger.debug(f"transient source release failed for '{ref}': {e}")
    gc.collect()


def render_layer(fig, ax, layer_info):
    """Render one layer on the given axes using METHOD_DISPATCH."""
    try:
        fig.logger.debug(f"Drawing layer -> {layer_info['name']}")
    except Exception:
        pass
    method_key = str(layer_info.get("method", "scatter")).lower()
    axes_type = getattr(ax, "_type", "any")

    method, warn = resolve_callable(ax, method_key, axes_type=axes_type, strict=True)
    if warn and fig.logger:
        try:
            fig.logger.warning(warn)
        except Exception:
            pass

    style = dict(fig.style.get(method_key, {}))
    if not style:
        try:
            style = dict(fig.style.get(getattr(method, "__name__", ""), {}))
        except Exception:
            pass
    if layer_info.get("style", {}) is not None:
        style.update(layer_info.get("style", {}))

    cb_name = layer_info.get("colorbar", "axc")

    if getattr(ax, "_type", None) == "tri":
        df = fig._ensure_pandas_data(layer_info["data"], reason=f"render:{layer_info.get('name', '')}")
        coor = layer_info.get("coor", {})
        try:
            style = collect_and_attach_colorbar(fig, style, coor, method_key, df, cb_name=cb_name)
            fig.logger.debug("Successful loading colorbar style")
        except Exception as _e:
            if fig.logger:
                fig.logger.debug(f"colorbar lazy-attach failed: {_e}")
        requiredlbr = {"left", "right", "bottom"}
        requiredxy = {"x", "y"}
        if not ((requiredlbr <= set(coor.keys())) or (requiredxy <= set(coor.keys()))):
            raise ValueError("Ternary layer must define coordinates: {left, right, bottom} or {x, y} with exprs.")
        for kk, vv in coor.items():
            style[kk] = fig._eval_series(df, vv)
        if method_key == "grid_profile":
            style["__df__"] = df
        if method_key in {"jpcontour", "jpcontourf", "jpfield"}:
            jp_kwargs = _prepare_jpcontour_style(
                fig,
                method_key,
                style,
                coor,
                df,
                layer_name=str(layer_info.get("name", "")),
                coord_keys=None,
                required_keys=("z",),
                include_diagnostics=method_key in {"jpcontour", "jpcontourf"},
            )
            return method(**jp_kwargs)
        return method(**style)

    elif getattr(ax, "_type", None) == "rect":
        df = fig._ensure_pandas_data(layer_info["data"], reason=f"render:{layer_info.get('name', '')}")
        coor = layer_info.get("coor", {})
        try:
            style = collect_and_attach_colorbar(fig, style, coor, method_key, df, cb_name=cb_name)
            fig.logger.debug("Successful loading colorbar style")
        except Exception as _e:
            if fig.logger:
                fig.logger.debug(f"colorbar lazy-attach failed: {_e}")

        if method_key == "hist":
            style.pop("interp", None)
            if isinstance(df, dict):
                if "label" not in style.keys():
                    style["label"] = []
                for kk, vv in coor.items():
                    style[kk] = []
                for dn, ddf in df.items():
                    style["label"].append(dn)
                    for kk, vv in coor.items():
                        style[kk].append(fig._eval_series(ddf, vv))
            else:
                for kk, vv in coor.items():
                    style[kk] = fig._eval_series(df, vv)

            return method(**style)
        if method_key in {"jpcontour", "jpcontourf", "jpfield"}:
            jp_kwargs = _prepare_jpcontour_style(
                fig,
                method_key,
                style,
                coor,
                df,
                layer_name=str(layer_info.get("name", "")),
                coord_keys=("x", "y", "z"),
                required_keys=("x", "y", "z"),
                include_diagnostics=method_key in {"jpcontour", "jpcontourf"},
            )
            return method(**jp_kwargs)
        if method_key in {"contour", "contourf"}:
            contour_coor = {}
            for kk in ("x", "y", "z"):
                if kk not in coor:
                    raise ValueError(
                        f"Rectangular {method_key} layer must define coordinates: {{x, y, z}}"
                    )
                vv = coor[kk]
                if isinstance(vv, dict) and "expr" in vv:
                    if df is None:
                        raise ValueError(
                            f"Layer '{layer_info.get('name', '')}' defines expression-based "
                            f"coordinate for '{kk}' but has no data source."
                        )
                    contour_coor[kk] = fig._eval_series(df, vv)
                else:
                    contour_coor[kk] = vv

            contour_args, style = _prepare_contour_args(fig, ax.ax if hasattr(ax, "ax") else ax, method_key, style, contour_coor)
            if contour_args is None:
                return []
            return method(*contour_args, **style)
        else:
            if not ({"x", "y"} <= set(coor.keys())):
                raise ValueError("Rectangular layer must define coordinates: {x,y} with exprs.")

            for kk, vv in coor.items():
                if isinstance(vv, dict) and "expr" in vv:
                    if df is None:
                        raise ValueError(
                            f"Layer '{layer_info.get('name', '')}' defines expression-based "
                            f"coordinate for '{kk}' but has no data source."
                        )
                    style[kk] = fig._eval_series(df, vv)
                else:
                    style[kk] = vv
            if method_key == "grid_profile":
                style["__df__"] = df
            style.pop("interp", None)
            return method(**style)

    else:
        raise ValueError(f"Axes '{ax}' has unknown _type='{getattr(ax, '_type', None)}'.")

from __future__ import annotations

from copy import deepcopy
import gc
from pathlib import Path

import pandas as pd

from .method_registry import resolve_callable
from .colorbar_runtime import collect_and_attach_colorbar


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
                            dt = fig.load_bool_df(dfsrcs, ds.get("transform", None))
                            dts.append(dt)
                        elif fig.context.get(src) is not None:
                            dt = deepcopy(fig.context.get(src))
                            dt = fig.load_bool_df(dt, ds.get("transform", None))
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
                            dt = fig.load_bool_df(dt, ds.get("transform", None))
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
                from .load_data import filter

                df = filter(df, trans["filter"], fig.logger)
                fig.logger.debug("After filtering -> {}".format(df.shape))
            elif "profile" in trans.keys():
                from .load_data import profiling

                df = profiling(df, trans["profile"], fig.logger)
                fig.logger.debug("After profiling -> {}".format(df.shape))
            elif "grid_profile" in trans.keys():
                from .load_data import grid_profiling

                cfg = trans.get("grid_profile", {})
                if isinstance(cfg, dict):
                    cfg = cfg.copy()
                    cfg.setdefault("method", "grid")
                else:
                    cfg = {"method": "grid"}
                df = grid_profiling(df, cfg, fig.logger)
                fig.logger.debug("After grid profiling -> {}".format(df.shape))
            elif "sortby" in trans.keys():
                from .load_data import sortby

                df = sortby(df, trans["sortby"], fig.logger)
                fig.logger.debug("After sortby -> {}".format(df.shape))
            elif "add_column" in trans.keys():
                from .load_data import addcolumn

                df = addcolumn(df, trans["add_column"], fig.logger)
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


def release_layer_runtime_data(fig, layer_info):
    layer_info["data"] = None
    layer_info["data_loaded"] = False

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
        if method_key in {"grid_profile", "grid_profiling"}:
            style["__df__"] = df
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
            if method_key in {"grid_profile", "grid_profiling"}:
                style["__df__"] = df
            return method(**style)

    else:
        raise ValueError(f"Axes '{ax}' has unknown _type='{getattr(ax, '_type', None)}'.")

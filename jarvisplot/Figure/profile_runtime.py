from __future__ import annotations

import numpy as np
import pandas as pd

from ..memtrace import memtrace_checkpoint
from ..utils.expression import eval_dataframe_expression


def eval_series(df: pd.DataFrame, set: dict, logger):
    """
    Evaluate an expression/column name against df safely.
    - If expr is a direct column name, returns that series.
    - If expr is a python expression, eval with df columns in scope.
    """
    try:
        logger.debug("Loading variable expression -> {}".format(set["expr"]))
    except Exception:
        pass
    if not "expr" in set.keys():
        raise ValueError(f"expr need for axes {set}.")
    arr = eval_dataframe_expression(
        df,
        set["expr"],
        logger=logger,
        fillna=set.get("fillna", None),
    )
    return np.asarray(arr)


def _profile_eval_axis(df: pd.DataFrame, cfg: dict, default_col: str, logger):
    if isinstance(cfg, dict) and ("expr" in cfg):
        return eval_series(df, cfg, logger)

    col = default_col
    if isinstance(cfg, dict):
        col = cfg.get("name", default_col)
    if col in df.columns:
        return np.asarray(df[col])
    if default_col in df.columns:
        return np.asarray(df[default_col])
    raise KeyError(f"Cannot resolve profiling axis '{default_col}' from dataframe.")


def _safe_minmax(arr):
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return [0.0, 1.0]
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if lo == hi:
        hi = lo + 1.0
    return [lo, hi]


def _normalize_profile_axis(arr, lim, scale):
    arr = np.asarray(arr, dtype=float)
    lo, hi = float(lim[0]), float(lim[1])

    if str(scale).lower() == "log":
        tiny = np.finfo(float).tiny
        lo = max(lo, tiny)
        hi = max(hi, lo * 10.0)
        den = np.log(hi) - np.log(lo)
        if den == 0:
            den = 1.0
        arr = np.where(arr > 0, arr, np.nan)
        return (np.log(arr) - np.log(lo)) / den

    den = hi - lo
    if den == 0:
        den = 1.0
    return (arr - lo) / den


def _grid_edges(lo, hi, nbin, scale):
    lo = float(lo)
    hi = float(hi)
    if str(scale).lower() == "log":
        tiny = np.finfo(float).tiny
        lo = max(lo, tiny)
        hi = max(hi, lo * 10.0)
        if hi == lo:
            hi = lo * 10.0
        return np.geomspace(lo, hi, int(nbin) + 1)
    if hi == lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, int(nbin) + 1)


def grid_profile_mesh(
    x,
    y,
    z,
    df=None,
    *,
    grid_bin=None,
    xlim=None,
    ylim=None,
    xscale="linear",
    yscale="linear",
    objective="max",
    objective_from_style=False,
):
    """Reconstruct grid_profile pcolormesh inputs from compact table metadata."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    objective = str(objective).lower()
    xscale = str(xscale).lower()
    yscale = str(yscale).lower()

    ix = None
    iy = None
    cols = getattr(df, "columns", [])
    if df is not None and ("__grid_ix__" in cols) and ("__grid_iy__" in cols):
        try:
            ix = np.asarray(df["__grid_ix__"], dtype=np.int32)
            iy = np.asarray(df["__grid_iy__"], dtype=np.int32)
            if "__grid_bin__" in cols and grid_bin is None:
                grid_bin = int(np.asarray(df["__grid_bin__"])[0])
            if "__grid_xmin__" in cols and "__grid_xmax__" in cols and xlim is None:
                xlim = [
                    float(np.asarray(df["__grid_xmin__"])[0]),
                    float(np.asarray(df["__grid_xmax__"])[0]),
                ]
            if "__grid_ymin__" in cols and "__grid_ymax__" in cols and ylim is None:
                ylim = [
                    float(np.asarray(df["__grid_ymin__"])[0]),
                    float(np.asarray(df["__grid_ymax__"])[0]),
                ]
            if "__grid_xscale__" in cols:
                xscale = str(np.asarray(df["__grid_xscale__"])[0]).lower()
            if "__grid_yscale__" in cols:
                yscale = str(np.asarray(df["__grid_yscale__"])[0]).lower()
            if (not objective_from_style) and ("__grid_objective__" in cols):
                objective = str(np.asarray(df["__grid_objective__"])[0]).lower()
        except Exception:
            ix, iy = None, None

    if grid_bin is None:
        if ix is not None and ix.size > 0:
            try:
                grid_bin = int(max(np.nanmax(ix), np.nanmax(iy)) + 1)
            except Exception:
                grid_bin = None
    if grid_bin is None:
        grid_bin = max(1, int(np.sqrt(max(len(x), 1))))
    grid_bin = max(int(grid_bin), 1)

    if xlim is None:
        xlim = _safe_minmax(x)
    if ylim is None:
        ylim = _safe_minmax(y)

    n = min(len(x), len(y), len(z))
    if n == 0:
        return None
    x = x[:n]
    y = y[:n]
    z = z[:n]

    if ix is None or iy is None:
        xn = _normalize_profile_axis(x, xlim, xscale)
        yn = _normalize_profile_axis(y, ylim, yscale)
        valid = np.isfinite(xn) & np.isfinite(yn)
        if not np.any(valid):
            return None
        xv = np.clip(xn[valid], 0.0, 1.0 - 1e-12)
        yv = np.clip(yn[valid], 0.0, 1.0 - 1e-12)
        ix = (xv * grid_bin).astype(np.int32)
        iy = (yv * grid_bin).astype(np.int32)
        zv = z[valid]
    else:
        ix = np.asarray(ix, dtype=np.int32)[:n]
        iy = np.asarray(iy, dtype=np.int32)[:n]
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        valid &= np.isfinite(ix) & np.isfinite(iy)
        if not np.any(valid):
            return None
        ix = ix[valid]
        iy = iy[valid]
        zv = z[valid]

    try:
        zkey = np.where(np.isfinite(zv), zv, np.inf if objective == "min" else -np.inf)
        tmp = pd.DataFrame({"ix": ix, "iy": iy, "z": zv, "zkey": zkey})
        if objective == "min":
            loc = tmp.groupby(["ix", "iy"], sort=False)["zkey"].idxmin()
        else:
            loc = tmp.groupby(["ix", "iy"], sort=False)["zkey"].idxmax()
        pick = tmp.loc[loc, ["ix", "iy", "z"]]
        ix_u = np.asarray(pick["ix"], dtype=np.int32)
        iy_u = np.asarray(pick["iy"], dtype=np.int32)
        z_u = np.asarray(pick["z"], dtype=float)
    except Exception:
        ix_u = np.asarray(ix, dtype=np.int32)
        iy_u = np.asarray(iy, dtype=np.int32)
        z_u = np.asarray(zv, dtype=float)

    x_edges = _grid_edges(xlim[0], xlim[1], grid_bin, xscale)
    y_edges = _grid_edges(ylim[0], ylim[1], grid_bin, yscale)

    in_range = (ix_u >= 0) & (ix_u < grid_bin) & (iy_u >= 0) & (iy_u < grid_bin)
    if not np.any(in_range):
        return None
    ix_u = ix_u[in_range]
    iy_u = iy_u[in_range]
    z_u = z_u[in_range]

    grid = np.full((grid_bin, grid_bin), np.nan, dtype=float)
    if len(ix_u) > 0:
        grid[iy_u, ix_u] = z_u
    return x_edges, y_edges, np.ma.masked_invalid(grid)


def grid_profiling(df, prof, logger):
    try:
        bin = int(prof.get("bin", 100) or 100)
    except Exception:
        bin = 100
    bin = max(bin, 1)
    memtrace_checkpoint(
        logger,
        "grid_profile.before",
        df,
        extra={
            "bin": bin,
            "objective": prof.get("objective", "max") if isinstance(prof, dict) else "max",
            "grid_points": prof.get("grid_points", "rect") if isinstance(prof, dict) else "rect",
        },
    )

    coors = prof.get("coordinates", {})
    obj = str(prof.get("objective", "max")).lower()
    grid = str(prof.get("grid_points", "rect")).lower()

    xcfg = coors.get("x", {}) if isinstance(coors, dict) else {}
    ycfg = coors.get("y", {}) if isinstance(coors, dict) else {}
    zcfg = coors.get("z", {}) if isinstance(coors, dict) else {}

    if grid == "ternary" and ("bottom" in coors) and ("right" in coors):
        bcfg = coors.get("bottom", {})
        rcfg = coors.get("right", {})
        b = _profile_eval_axis(df, bcfg, "bottom", logger)
        r = _profile_eval_axis(df, rcfg, "right", logger)
        x = np.asarray(b, dtype=float) + 0.5 * np.asarray(r, dtype=float)
        y = np.asarray(r, dtype=float)
    else:
        x = _profile_eval_axis(df, xcfg, "x", logger)
        y = _profile_eval_axis(df, ycfg, "y", logger)
    z = _profile_eval_axis(df, zcfg, "z", logger)

    xind = xcfg.get("name", "x") if isinstance(xcfg, dict) else "x"
    yind = ycfg.get("name", "y") if isinstance(ycfg, dict) else "y"
    zind = zcfg.get("name", "z") if isinstance(zcfg, dict) else "z"

    if grid == "ternary":
        xlim = xcfg.get("lim", [0, 1]) if isinstance(xcfg, dict) else [0, 1]
        ylim = ycfg.get("lim", [0, 1]) if isinstance(ycfg, dict) else [0, 1]
    else:
        xlim = xcfg.get("lim", _safe_minmax(x)) if isinstance(xcfg, dict) else _safe_minmax(x)
        ylim = ycfg.get("lim", _safe_minmax(y)) if isinstance(ycfg, dict) else _safe_minmax(y)

    xscale = xcfg.get("scale", "linear") if isinstance(xcfg, dict) else "linear"
    yscale = ycfg.get("scale", "linear") if isinstance(ycfg, dict) else "linear"

    out = df.copy(deep=False)
    out[xind] = x
    out[yind] = y
    out[zind] = z

    xnorm = _normalize_profile_axis(out[xind].to_numpy(), xlim, xscale)
    ynorm = _normalize_profile_axis(out[yind].to_numpy(), ylim, yscale)
    valid = np.isfinite(xnorm) & np.isfinite(ynorm)
    if not np.any(valid):
        if logger:
            logger.warning("Grid profiling got no finite points; returning original dataframe.")
        return out

    xv = np.clip(xnorm[valid], 0.0, 1.0 - 1e-12)
    yv = np.clip(ynorm[valid], 0.0, 1.0 - 1e-12)
    ix = (xv * bin).astype(np.int32)
    iy = (yv * bin).astype(np.int32)
    cell = iy.astype(np.int64) * np.int64(bin) + ix.astype(np.int64)
    pos_src = np.flatnonzero(valid).astype(np.int64)
    zraw = np.asarray(out[zind], dtype=float)[valid]

    finite_zraw = zraw[np.isfinite(zraw)]
    fill_empty = bool(prof.get("fill_empty", False))
    if "empty_value" in prof:
        try:
            empty_value = float(prof.get("empty_value"))
        except Exception:
            empty_value = np.nan
    elif fill_empty and finite_zraw.size > 0:
        empty_value = float(np.min(finite_zraw)) - 0.1
    else:
        empty_value = np.nan

    if obj == "min":
        zkey = np.where(np.isfinite(zraw), zraw, np.inf)
        tmp = pd.DataFrame({
            "__cell__": cell,
            "__zkey__": zkey,
            "__zraw__": zraw,
            "__pos__": pos_src,
            "__ix__": ix,
            "__iy__": iy,
        })
        loc = tmp.groupby("__cell__", sort=False)["__zkey__"].idxmin()
    else:
        zkey = np.where(np.isfinite(zraw), zraw, -np.inf)
        tmp = pd.DataFrame({
            "__cell__": cell,
            "__zkey__": zkey,
            "__zraw__": zraw,
            "__pos__": pos_src,
            "__ix__": ix,
            "__iy__": iy,
        })
        loc = tmp.groupby("__cell__", sort=False)["__zkey__"].idxmax()

    memtrace_checkpoint(
        logger,
        "grid_profile.groupby_ready",
        tmp,
        extra={"bin": bin, "objective": obj},
    )
    pick = tmp.loc[loc, ["__ix__", "__iy__", "__zraw__"]].copy()

    grid = np.full((bin, bin), empty_value, dtype=float)
    if len(pick) > 0:
        ixa = pick["__ix__"].to_numpy(dtype=np.int32)
        iya = pick["__iy__"].to_numpy(dtype=np.int32)
        zva = pick["__zraw__"].to_numpy(dtype=float)
        keep = (ixa >= 0) & (ixa < bin) & (iya >= 0) & (iya < bin)
        if np.any(keep):
            grid[iya[keep], ixa[keep]] = zva[keep]

    x_edges = _grid_edges(xlim[0], xlim[1], bin, xscale)
    y_edges = _grid_edges(ylim[0], ylim[1], bin, yscale)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    ixm, iym = np.meshgrid(
        np.arange(bin, dtype=np.int32),
        np.arange(bin, dtype=np.int32),
    )
    xg, yg = np.meshgrid(x_centers, y_centers)

    reduced = pd.DataFrame({
        xind: xg.ravel(),
        yind: yg.ravel(),
        zind: grid.ravel(),
        "__grid_ix__": ixm.ravel(),
        "__grid_iy__": iym.ravel(),
    })
    reduced["__grid_bin__"] = np.int32(bin)
    reduced["__grid_xmin__"] = float(xlim[0])
    reduced["__grid_xmax__"] = float(xlim[1])
    reduced["__grid_ymin__"] = float(ylim[0])
    reduced["__grid_ymax__"] = float(ylim[1])
    reduced["__grid_xscale__"] = str(xscale).lower()
    reduced["__grid_yscale__"] = str(yscale).lower()
    reduced["__grid_objective__"] = obj
    reduced["__grid_empty_value__"] = empty_value

    if logger:
        n_filled = int(np.isfinite(grid).sum())
        logger.debug(
            f"After grid profiling ({bin}x{bin}) -> {reduced.shape}, filled={n_filled}/{bin*bin}"
        )
    memtrace_checkpoint(
        logger,
        "grid_profile.after",
        reduced,
        extra={"bin": bin, "filled": int(np.isfinite(grid).sum()), "cells": int(bin * bin)},
    )
    return reduced


def profiling(df, prof, logger):
    mode = "bridson"
    if isinstance(prof, dict):
        mode = str(prof.get("method", "bridson")).lower()
    if mode in {"grid", "grid_profile"}:
        return grid_profiling(df, prof, logger)
    memtrace_checkpoint(
        logger,
        "profile.before",
        df,
        extra={
            "method": mode,
            "bin": prof.get("bin", 100) if isinstance(prof, dict) else 100,
            "grid_points": prof.get("grid_points", "rect") if isinstance(prof, dict) else "rect",
            "objective": prof.get("objective", "max") if isinstance(prof, dict) else "max",
        },
    )

    def profile_bridson_sorted(xx, yy, zz, radius):
        msk = np.full(xx.shape, True, dtype=bool)
        for i in range(len(xx)):
            if not msk[i]:
                continue
            if i + 1 >= len(xx):
                continue
            dx = xx[i + 1 :] - xx[i]
            dy = yy[i + 1 :] - yy[i]
            dz = zz[i + 1 :] - zz[i]
            dist0 = (dx**2 + dy**2)**0.5
            dist1 = (dx**2 + dy**2 + dz**2)**0.5
            near0 = (dist0 < 0.707 * radius) | (dist0 < radius) & (dist1 > radius)
            msk[i + 1 :] &= ~near0
        return msk

    bin = prof.get("bin", 100)
    coors = prof.get("coordinates", {})
    obj = prof.get("objective", "max")
    grid = prof.get("grid_points", "rect")

    radius = 1 / bin
    xcfg = coors.get("x", {}) if isinstance(coors, dict) else {}
    ycfg = coors.get("y", {}) if isinstance(coors, dict) else {}
    zcfg = coors.get("z", {}) if isinstance(coors, dict) else {}

    if "expr" in xcfg.keys():
        x = eval_series(df, xcfg, logger)
    else:
        x = df["x"]

    if "expr" in ycfg.keys():
        y = eval_series(df, ycfg, logger)
    else:
        y = df["y"]

    if "expr" in zcfg.keys():
        z = eval_series(df, zcfg, logger)
    else:
        z = df["z"]

    logger.debug("After loading profiling x, y, z. ")

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    if grid == "ternary":
        xlim = xcfg.get("lim", [0, 1])
        ylim = ycfg.get("lim", [0, 1])
        zlim = zcfg.get("lim", [np.min(z), np.max(z)])
        xscale = xcfg.get("scale", "linear")
        yscale = ycfg.get("scale", "linear")
        zscale = zcfg.get("scale", "linear")
        zind = zcfg.get("name", "z")
        xind = xcfg.get("name", "x")
        yind = ycfg.get("name", "y")
    elif grid == "rect":
        xlim = xcfg.get("lim", [np.min(x), np.max(x)])
        ylim = ycfg.get("lim", [np.min(y), np.max(y)])
        zlim = zcfg.get("lim", [np.min(z), np.max(z)])
        xscale = xcfg.get("scale", "linear")
        yscale = ycfg.get("scale", "linear")
        zscale = zcfg.get("scale", "linear")
        zind = zcfg.get("name", "z")
        xind = xcfg.get("name", "x")
        yind = ycfg.get("name", "y")
    else:
        xlim = xcfg.get("lim", [np.min(x), np.max(x)])
        ylim = ycfg.get("lim", [np.min(y), np.max(y)])
        zlim = zcfg.get("lim", [np.min(z), np.max(z)])
        xscale = xcfg.get("scale", "linear")
        yscale = ycfg.get("scale", "linear")
        zscale = zcfg.get("scale", "linear")
        zind = zcfg.get("name", "z")
        xind = xcfg.get("name", "x")
        yind = ycfg.get("name", "y")

    zord = np.asarray(z, dtype=float)
    if obj == "max":
        zkey = np.where(np.isfinite(zord), zord, -np.inf)
        order = np.argsort(zkey, kind="quicksort")[::-1].astype(np.int64, copy=False)
    elif obj == "min":
        zkey = np.where(np.isfinite(zord), zord, np.inf)
        order = np.argsort(zkey, kind="quicksort").astype(np.int64, copy=False)
    else:
        zkey = np.where(np.isfinite(zord), zord, -np.inf)
        order = np.argsort(zkey, kind="quicksort")[::-1].astype(np.int64, copy=False)
        logger.error("Sort dataset method: objective: {} not support, using default value -> 'max'".format(obj))

    if grid == "ternary":
        bb = np.linspace(0, 1, bin + 1)
        rr = np.linspace(0, 1, bin + 1)
        Bg, Rg = np.meshgrid(bb, rr)
        r = Rg.ravel()
        b = Bg.ravel()
        l = 1.0 - b - r
        mask = (l >= 0) & (b >= 0) & (r >= 0)
        x_grid = (b + 0.5 * r)[mask]
        y_grid = r[mask]
    elif grid == "rect":
        xx = np.linspace(xlim[0], xlim[1], bin + 1)
        yy = np.linspace(ylim[0], ylim[1], bin + 1)
        xg, yg = np.meshgrid(xx, yy)
        x_grid = xg.ravel()
        y_grid = yg.ravel()
    else:
        x_grid = np.asarray([], dtype=float)
        y_grid = np.asarray([], dtype=float)
    z_grid = np.ones(x_grid.shape, dtype=float) * (np.min(z) - 0.1)

    xx = np.concatenate([x[order], x_grid], axis=0)
    yy = np.concatenate([y[order], y_grid], axis=0)
    zz = np.concatenate([z[order], z_grid], axis=0)
    memtrace_checkpoint(
        logger,
        "profile.concat_ready",
        None,
        extra={
            "source_rows": int(order.shape[0]),
            "grid_rows": int(x_grid.shape[0]),
            "concat_rows": int(xx.shape[0]),
        },
    )

    if xscale == "log":
        xx = (np.log(xx) - np.log(xlim[0])) / (np.log(xlim[1]) - np.log(xlim[0]))
    else:
        xx = (xx - xlim[0]) / (xlim[1] - xlim[0])

    if yscale == "log":
        yy = (np.log(yy) - np.log(ylim[0])) / (np.log(ylim[1]) - np.log(ylim[0]))
    else:
        yy = (yy - ylim[0]) / (ylim[1] - ylim[0])

    if zscale == "log":
        zz = (np.log(zz) - np.log(zlim[0])) / (np.log(zlim[1]) - np.log(zlim[0]))
    else:
        zz = (zz - zlim[0]) / (zlim[1] - zlim[0])

    msk = profile_bridson_sorted(xx, yy, zz, radius)
    keep_pos = np.flatnonzero(msk).astype(np.int64, copy=False)

    source_count = int(order.shape[0])
    keep_source = keep_pos[keep_pos < source_count]
    keep_grid = keep_pos[keep_pos >= source_count] - source_count

    source_rows = order[keep_source]
    source_x = x[source_rows]
    source_y = y[source_rows]
    source_z = z[source_rows]
    grid_x_keep = x_grid[keep_grid]
    grid_y_keep = y_grid[keep_grid]
    xx = None
    yy = None
    zz = None
    x_grid = None
    y_grid = None
    z_grid = None

    out = df.iloc[source_rows].copy()
    out.index = keep_source
    out[xind] = source_x
    out[yind] = source_y
    out[zind] = source_z

    if keep_grid.size > 0:
        grid_out = pd.DataFrame(index=keep_grid + source_count)
        grid_out[xind] = grid_x_keep
        grid_out[yind] = grid_y_keep
        grid_out[zind] = np.nan
        grid_out = grid_out.reindex(columns=out.columns)
        out = pd.concat([out, grid_out], axis=0, ignore_index=False)

    memtrace_checkpoint(
        logger,
        "profile.after",
        out,
        extra={"source_rows": int(source_rows.shape[0]), "grid_rows": int(keep_grid.shape[0])},
    )
    return out


def _preprofiling(df, prof, logger):
    """Lightweight pre-profiling for cache prebuild."""

    def _norm(arr, lim, scale):
        arr = np.asarray(arr, dtype=float)
        lo, hi = float(lim[0]), float(lim[1])
        if str(scale).lower() == "log":
            tiny = np.finfo(float).tiny
            lo = max(lo, tiny)
            hi = max(hi, lo * 10.0)
            den = np.log(hi) - np.log(lo)
            if den == 0:
                den = 1.0
            arr = np.where(arr > 0, arr, np.nan)
            return (np.log(arr) - np.log(lo)) / den
        den = hi - lo
        if den == 0:
            den = 1.0
        return (arr - lo) / den

    def _auto_prebin(nrows: int):
        nrows = int(max(nrows, 0))
        if nrows > 1_000_000:
            return 1000, "rows > 1000000"
        if nrows > 250_000:
            return 500, "250000 < rows <= 1000000"
        if nrows > 90_000:
            return 300, "90000 < rows <= 250000"
        return 300, "rows <= 90000"

    pre_cfg = prof.get("pregrid", {})
    rows_in = int(df.shape[0]) if isinstance(df, pd.DataFrame) else 0
    explicit_prebin = None
    explicit_source = None

    if isinstance(pre_cfg, dict) and ("bin" in pre_cfg):
        explicit_prebin = pre_cfg.get("bin")
        explicit_source = "pregrid.bin"
    elif isinstance(prof, dict) and ("pregrid_bin" in prof):
        explicit_prebin = prof.get("pregrid_bin")
        explicit_source = "pregrid_bin"

    if explicit_prebin is not None:
        try:
            prebin = int(explicit_prebin)
        except Exception:
            prebin, rule = _auto_prebin(rows_in)
            explicit_source = None
    else:
        prebin, rule = _auto_prebin(rows_in)

    if isinstance(pre_cfg, dict):
        enabled = bool(pre_cfg.get("enable", True))
    else:
        enabled = False if pre_cfg is False else True
    prebin = max(prebin, 1)

    if logger:
        if explicit_source is None:
            logger.info(
                "Preprofiling auto-prebin:\n\t rows_in -> {}\n\t prebin -> {}\n\t rule -> {}\n\t enabled -> {}.".format(
                    rows_in,
                    prebin,
                    rule,
                    enabled,
                )
            )
        else:
            logger.info(
                "Preprofiling explicit prebin:\n\t rows_in -> {}\n\t prebin -> {}\n\t source -> {}\n\t enabled -> {}.".format(
                    rows_in,
                    prebin,
                    explicit_source,
                    enabled,
                )
            )

    if not enabled:
        return df

    coors = prof.get("coordinates", {})
    if not isinstance(coors, dict) or not coors:
        if logger:
            logger.warning("Preprofiling skipped: profile.coordinates is missing or invalid.")
        return df

    xset = coors.get("x", {})
    yset = coors.get("y", {})
    zset = coors.get("z", {})
    if not isinstance(xset, dict):
        xset = {}
    if not isinstance(yset, dict):
        yset = {}
    if not isinstance(zset, dict):
        zset = {}

    grid_hint = str(prof.get("grid_points", "")).lower()
    use_ternary = (
        grid_hint == "ternary"
        or (
            ("bottom" in coors)
            and ("right" in coors)
            and (("x" not in coors) or ("y" not in coors))
        )
    )

    if use_ternary and ("bottom" in coors) and ("right" in coors):
        bset = coors["bottom"]
        rset = coors["right"]
        if isinstance(bset, dict) and "expr" in bset:
            b = eval_series(df, bset, logger)
        else:
            b = np.asarray(df.get("bottom"))
        if isinstance(rset, dict) and "expr" in rset:
            r = eval_series(df, rset, logger)
        else:
            r = np.asarray(df.get("right"))
        x = np.asarray(b, dtype=float) + 0.5 * np.asarray(r, dtype=float)
        y = np.asarray(r, dtype=float)
    else:
        if "expr" in xset.keys():
            x = eval_series(df, xset, logger)
        else:
            xcol = xset.get("name", "x")
            if xcol in df.columns:
                x = df[xcol]
            elif "x" in df.columns:
                x = df["x"]
            else:
                raise KeyError("Preprofiling x axis not found (need coordinates.x expr/name or dataframe column 'x').")

        if "expr" in yset.keys():
            y = eval_series(df, yset, logger)
        else:
            ycol = yset.get("name", "y")
            if ycol in df.columns:
                y = df[ycol]
            elif "y" in df.columns:
                y = df["y"]
            else:
                raise KeyError("Preprofiling y axis not found (need coordinates.y expr/name or dataframe column 'y').")

    if "expr" in zset.keys():
        z = eval_series(df, zset, logger)
    else:
        zcol = zset.get("name", "z")
        if zcol in df.columns:
            z = df[zcol]
        elif "z" in df.columns:
            z = df["z"]
        else:
            raise KeyError("Preprofiling z axis not found (need coordinates.z expr/name or dataframe column 'z').")

    if use_ternary:
        xlim = xset.get("lim", [0, 1])
        ylim = yset.get("lim", [0, 1])
    else:
        xlim = xset.get("lim", [np.nanmin(x), np.nanmax(x)])
        ylim = yset.get("lim", [np.nanmin(y), np.nanmax(y)])

    xscale = xset.get("scale", "linear")
    yscale = yset.get("scale", "linear")
    xind = xset.get("name", "x")
    yind = yset.get("name", "y")
    zind = zset.get("name", "z")

    xvals = np.asarray(x)
    yvals = np.asarray(y)
    zvals = np.asarray(z)

    xnorm = _norm(xvals, xlim, xscale)
    ynorm = _norm(yvals, ylim, yscale)
    valid = np.isfinite(xnorm) & np.isfinite(ynorm)
    if not np.any(valid):
        if logger:
            logger.warning("Preprofiling got no finite points; returning original dataframe.")
        out = df.copy(deep=False)
        out[xind] = xvals
        out[yind] = yvals
        out[zind] = zvals
        return out

    xv = np.clip(xnorm[valid], 0.0, 1.0 - 1e-12)
    yv = np.clip(ynorm[valid], 0.0, 1.0 - 1e-12)
    ix = (xv * prebin).astype(np.int32)
    iy = (yv * prebin).astype(np.int32)
    cell = iy.astype(np.int64) * np.int64(prebin) + ix.astype(np.int64)

    idx_src = np.flatnonzero(valid).astype(np.int64)
    zraw = np.asarray(zvals, dtype=float)[valid]
    zmax = np.where(np.isfinite(zraw), zraw, -np.inf)
    zmin = np.where(np.isfinite(zraw), zraw, np.inf)
    tmp = pd.DataFrame(
        {
            "__cell__": cell,
            "__zmax__": zmax,
            "__zmin__": zmin,
            "__pos__": idx_src,
        }
    )

    loc_max = tmp.groupby("__cell__", sort=False)["__zmax__"].idxmax().to_numpy(dtype=np.int64, copy=False)
    loc_min = tmp.groupby("__cell__", sort=False)["__zmin__"].idxmin().to_numpy(dtype=np.int64, copy=False)
    loc_all = np.unique(np.concatenate([loc_max, loc_min], axis=0))
    keep_pos = tmp.iloc[loc_all]["__pos__"].to_numpy(dtype=np.int64, copy=False)
    reduced = df.iloc[keep_pos].copy()
    reduced[xind] = xvals[keep_pos]
    reduced[yind] = yvals[keep_pos]
    reduced[zind] = zvals[keep_pos]
    if logger:
        logger.debug(f"After preprofiling ({prebin}x{prebin}) -> {reduced.shape}")
    return reduced

#!/usr/bin/env python3 

import numpy as np 
import pandas as pd 
from copy import deepcopy

def eval_series(df: pd.DataFrame, set: dict, logger):
    """
    Evaluate an expression/column name against df safely.
    - If expr is a direct column name, returns that series.
    - If expr is a python expression, eval with df columns in scope.
    """
    try: 
        logger.debug("Loading variable expression -> {}".format(set['expr'])) 
    except: 
        pass 
    if not "expr" in set.keys():
        raise ValueError(f"expr need for axes {set}.")
    if set["expr"] in df.columns:
        arr = df[set["expr"]].values
        if np.isnan(arr).sum() and "fillna" in set.keys():
            arr = np.where(np.isnan(arr), float(set['fillna']), arr)
    else: 
        # safe-ish eval with only df columns in locals
        local_vars = df.to_dict("series")
        import math
        from ..inner_func import update_funcs
        allowed_globals = update_funcs({"np": np, "math": math})
        arr = eval(set["expr"], allowed_globals, local_vars)
        if np.isnan(arr).sum() and "fillna" in set.keys():
            arr = np.where(np.isnan(arr), float(set['fillna']), arr)
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


def grid_profiling(df, prof, logger):
    try:
        bin = int(prof.get("bin", 100) or 100)
    except Exception:
        bin = 100
    bin = max(bin, 1)

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

    out = df.copy()
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
    # Default behavior: keep empty cells as NaN so they are not rendered.
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
    return reduced


def profiling(df, prof, logger):
    mode = "bridson"
    if isinstance(prof, dict):
        mode = str(prof.get("method", "bridson")).lower()
    if mode in {"grid", "grid_profile", "grid_profiling"}:
        return grid_profiling(df, prof, logger)

    def profile_bridson_sorted(idx, xx, yy, zz, radius, msk):
        for i in range(len(idx)):
            if not msk[i]:
                continue
            dx = xx[idx > idx[i]] - xx[i]
            dy = yy[idx > idx[i]] - yy[i]
            dz = zz[idx > idx[i]] - zz[i]
            dist0 = (dx**2 + dy**2)**0.5
            dist1 = (dx**2 + dy**2 + dz**2)**0.5
            near0 = (dist0 < 0.707 * radius) | (dist0 < radius) & (dist1 > radius)
            sel = (idx > idx[i])
            msk[sel] &= ~near0                     
        return msk
            
    bin     = prof.get("bin", 100)
    coors   = prof.get("coordinates", {})
    obj     = prof.get("objective", "max")
    grid    = prof.get("grid_points", "rect")
    gdata   = None 

    radius  = 1 / bin 
    if "expr" in coors['x'].keys():
        x = eval_series(df, coors['x'], logger)
    else: 
        x = df['x']
    
    if "expr" in coors['y'].keys():
        y = eval_series(df, coors['y'], logger)
    else: 
        y = df['y']
        
    if "expr" in coors['z'].keys():
        z = eval_series(df, coors['z'], logger)
    else: 
        z = df['z']

    logger.debug("After loading profiling x, y, z. ")

    if grid == "ternary":
        xlim = coors['x'].get("lim", [0, 1])
        ylim = coors['y'].get("lim", [0, 1])
        zlim = coors['z'].get("lim", [np.min(z), np.max(z)])
        xscale = coors['x'].get("scale", "linear")
        yscale = coors['y'].get("scale", "linear")
        zscale = coors['z'].get("scale", "linear")
        zind   = coors['z'].get("name", "z")
        xind   = coors['x'].get("name", "x")
        yind   = coors['y'].get("name", "y")
    elif grid == "rect":
        xlim = coors['x'].get("lim", [np.min(x), np.max(x)])
        ylim = coors['y'].get("lim", [np.min(y), np.max(y)])
        zlim = coors['z'].get("lim", [np.min(z), np.max(z)])

        xscale = coors['x'].get("scale", "linear")
        yscale = coors['y'].get("scale", "linear")
        zscale = coors['z'].get("scale", "linear")

        zind = coors['z'].get("name", "z")
        xind = coors['x'].get("name", "x")
        yind = coors['y'].get("name", "y") 


    # profiling will add new columns into dataframe, so that can be used in the next step
    df[xind] = x 
    df[yind] = y
    df[zind] = z
            # print(x.min(), x.max(), y.min(), y.max(), z.min(), z.max())


    if grid == "ternary":
        bb = np.linspace(0, 1, bin + 1)
        rr = np.linspace(0, 1, bin + 1)
        Bg, Rg = np.meshgrid(bb, rr)
        r = Rg.ravel()
        b = Bg.ravel() 
        l = 1.0 - b - r
        mask = (l >= 0) & (b >= 0) & (r >= 0)
        x = b + 0.5 * r 
        y = r 
        xxg, yyg = x[mask], y[mask]
        llg, bbg, rrg, = l[mask], b[mask], r[mask]
        gdata = pd.DataFrame({
            xind: xxg, 
            yind: yyg, 
            zind: np.ones(xxg.shape) * (np.min(z) - 0.1)
        })

    elif grid == "rect":
        xx = np.linspace(xlim[0], xlim[1], bin+1)
        yy = np.linspace(ylim[0], ylim[1], bin+1)
        xg, yg = np.meshgrid(xx, yy)

        gdata = pd.DataFrame({
            xind: xg.ravel(),
            yind: yg.ravel(),
            zind: np.ones(xg.ravel().shape) * (np.min(z) - 0.1)
        })

    if obj == "max":    
        df = df.sort_values(zind, ascending=False).reset_index(drop=True)
    elif obj == "min":
        df = df.sort_values(zind, ascending=True).reset_index(drop=True)
    else:
        df = df.sort_values(zind, ascending=False).reset_index(drop=True)
        logger.error("Sort dataset method: objective: {} not support, using default value -> 'max'".format(obj))
    df = pd.concat([df, gdata], ignore_index=True)
                        
    idx = deepcopy(np.array(df.index))
    xx  = deepcopy(np.array(df[xind]))
    yy  = deepcopy(np.array(df[yind]))
    zz  = deepcopy(np.array(df[zind]))
            # mapping xx, yy, zz to range [0, 1]
    if xscale == "log":
        xx = (np.log(xx) - np.log(xlim[0])) / (np.log(xlim[1]) - np.log(xlim[0]))
    else:  # linear scale
        xx = (xx - xlim[0]) / (xlim[1] - xlim[0])

    if yscale == "log":
        yy = (np.log(yy) - np.log(ylim[0])) / (np.log(ylim[1]) - np.log(ylim[0]))
    else:  # linear scale
        yy = (yy - ylim[0]) / (ylim[1] - ylim[0])

    if zscale == "log":
        zz = (np.log(zz) - np.log(zlim[0])) / (np.log(zlim[1]) - np.log(zlim[0]))
    else:  # linear scale
        zz = (zz - zlim[0]) / (zlim[1] - zlim[0])

    # (removed print(radius))
    msk = np.full(idx.shape, True)
    msk = profile_bridson_sorted(idx, xx, yy, zz, radius, msk)
    df = df.iloc[idx[msk]]

    return df 


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

    out = df.copy()
    out[xind] = x
    out[yind] = y
    out[zind] = z

    xnorm = _norm(out[xind].to_numpy(), xlim, xscale)
    ynorm = _norm(out[yind].to_numpy(), ylim, yscale)
    valid = np.isfinite(xnorm) & np.isfinite(ynorm)
    if not np.any(valid):
        if logger:
            logger.warning("Preprofiling got no finite points; returning original dataframe.")
        return out

    xv = np.clip(xnorm[valid], 0.0, 1.0 - 1e-12)
    yv = np.clip(ynorm[valid], 0.0, 1.0 - 1e-12)
    ix = (xv * prebin).astype(np.int32)
    iy = (yv * prebin).astype(np.int32)
    cell = iy.astype(np.int64) * np.int64(prebin) + ix.astype(np.int64)

    # Use positional index (not label index) to avoid expansion when source
    # frames were concatenated with duplicate index labels.
    idx_src = np.flatnonzero(valid).astype(np.int64)
    zraw = np.asarray(out[zind], dtype=float)[valid]
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
    reduced = out.iloc[keep_pos].copy()
    if logger:
        logger.debug(f"After preprofiling ({prebin}x{prebin}) -> {reduced.shape}")
    return reduced


def filter(df, condition, logger):
    try:
        if isinstance(condition, bool):
            return df.copy() if condition else df.iloc[0:0].copy()
        if isinstance(condition, (int, float)) and condition in (0, 1):
            return df.copy() if int(condition) == 1 else df.iloc[0:0].copy()
        
        if isinstance(condition, str):
            s = condition.strip()
            low = s.lower()
            if low in {"true", "t", "yes", "y"}:
                return df.copy()
            if low in {"false", "f", "no", "n"}:
                return df.iloc[0:0].copy()
            s = s.replace("&&", " & ").replace("||", " | ")
            condition = s
        else:
            raise TypeError(f"Unsupported condition type: {type(condition)}")

        from ..inner_func import update_funcs
        import math
        allowed_globals = update_funcs({"np": np, "math": math})
        local_vars = df.to_dict("series")
        mask = eval(condition, allowed_globals, local_vars)

        if isinstance(mask, (bool, np.bool_, int, float)):
            return df.copy() if bool(mask) else df.iloc[0:0].copy()
        if not isinstance(mask, pd.Series):
            mask = pd.Series(mask, index=df.index)
        mask = mask.astype(bool)
        return df[mask].copy()
    except Exception as e:
        logger.error(f"Errors when evaluating condition -> {condition}:\n\t{e}")
        return pd.DataFrame(index=df.index).iloc[0:0].copy()

def addcolumn(df, adds, logger):
    try: 
        name = adds.get("name", False)
        expr = adds.get("expr", False)
        if not (name and expr):
            logger.error("Error in loading add_column -> {}".format(adds))
        from ..inner_func import update_funcs
        import math
        allowed_globals = update_funcs({"np": np, "math": math})
        local_vars = df.to_dict("series") 
        value = eval(str(expr), allowed_globals, local_vars)
        df[name] = value 
        return df
    except Exception as e: 
        logger.error(
            "Errors when add new column -> {}:\n\t{}: {}".format(
                adds, e.__class__.__name__, e
            )
        )
        return df               
        
def sortby(df, expr, logger):
    try:
        return sort_df_by_expr(df, expr, logger=logger)
    except Exception as e:
        logger.warning(f"sortby failed for expr={expr}: {e}")
        return df

def sort_df_by_expr(df: pd.DataFrame, expr: str, logger) -> pd.DataFrame:
    """
    Sort the dataframe by evaluating the given expression.
    The expression can be a column name or a valid expression understood by _eval_series.
    Returns a new DataFrame sorted ascending by the evaluated values.
    """
    if df is None or expr is None:
        return df
    try:
        # Try evaluate as expression (could be column or expression)
        values = eval_series(df, {"expr": expr}, logger)
        df = df.assign(__sortkey__=values)
        df = df.sort_values(by="__sortkey__", ascending=True)
        df = df.drop(columns=["__sortkey__"])
        return df
    except Exception as e:
        logger.warning(f"LB: sortby failed for expr={expr}: {e}")
        return df   

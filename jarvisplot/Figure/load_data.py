#!/usr/bin/env python3 

import numpy as np 
import pandas as pd 
import json
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


def profiling(df, prof, logger):
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
    """Lightweight pre-profiling (default 1000x1000 grid) for cache prebuild."""

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

    pre_cfg = prof.get("pregrid", {})
    if isinstance(pre_cfg, dict):
        prebin = int(pre_cfg.get("bin", prof.get("pregrid_bin", 1000)) or 1000)
        enabled = bool(pre_cfg.get("enable", True))
    else:
        prebin = int(prof.get("pregrid_bin", 1000) or 1000)
        enabled = False if pre_cfg is False else True
    prebin = max(prebin, 1)

    if not enabled:
        return df

    coors = prof.get("coordinates", {})
    obj = str(prof.get("objective", "max")).lower()
    grid = str(prof.get("grid_points", "rect")).lower()

    if grid == "ternary" and ("bottom" in coors) and ("right" in coors):
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
        if "expr" in coors["x"].keys():
            x = eval_series(df, coors["x"], logger)
        else:
            x = df["x"]

        if "expr" in coors["y"].keys():
            y = eval_series(df, coors["y"], logger)
        else:
            y = df["y"]

    if "expr" in coors["z"].keys():
        z = eval_series(df, coors["z"], logger)
    else:
        z = df["z"]

    if grid == "ternary":
        xlim = coors["x"].get("lim", [0, 1])
        ylim = coors["y"].get("lim", [0, 1])
    else:
        xlim = coors["x"].get("lim", [np.nanmin(x), np.nanmax(x)])
        ylim = coors["y"].get("lim", [np.nanmin(y), np.nanmax(y)])

    xscale = coors["x"].get("scale", "linear")
    yscale = coors["y"].get("scale", "linear")
    xind = coors["x"].get("name", "x")
    yind = coors["y"].get("name", "y")
    zind = coors["z"].get("name", "z")

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

    idx_src = np.asarray(out.index)[valid]
    zraw = np.asarray(out[zind], dtype=float)[valid]
    if obj == "min":
        zkey = np.where(np.isfinite(zraw), zraw, np.inf)
        tmp = pd.DataFrame({"__cell__": cell, "__z__": zkey, "__idx__": idx_src})
        loc = tmp.groupby("__cell__", sort=False)["__z__"].idxmin()
    else:
        zkey = np.where(np.isfinite(zraw), zraw, -np.inf)
        tmp = pd.DataFrame({"__cell__": cell, "__z__": zkey, "__idx__": idx_src})
        loc = tmp.groupby("__cell__", sort=False)["__z__"].idxmax()

    keep_idx = tmp.loc[loc, "__idx__"].to_numpy()
    reduced = out.loc[keep_idx].copy()
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
        logger.error("Errors when add new column -> {}:\n\t{}".format(adds, json.dumps(e)))   
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

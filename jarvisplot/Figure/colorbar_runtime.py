from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import matplotlib.colors as mcolors

from ..utils.expression import eval_dataframe_expression


def axc_is_horizontal(frame: Mapping[str, Any]) -> bool:
    return str(frame.get("axc", {}).get("orientation", "")).lower() == "horizontal"


def axc_color_config(frame: Mapping[str, Any]) -> dict:
    """Return normalized frame.axc.color config with legacy fallbacks."""
    out: dict[str, Any] = {}
    axc_cfg = frame.get("axc", {}) if isinstance(frame, Mapping) else {}
    color_cfg = axc_cfg.get("color", {})
    if not isinstance(color_cfg, dict):
        color_cfg = {}

    if "cmap" in color_cfg:
        out["cmap"] = color_cfg.get("cmap")

    for key in ("vmin", "vmax"):
        if key in color_cfg:
            try:
                val = float(color_cfg.get(key))
                if np.isfinite(val):
                    out[key] = val
            except Exception:
                pass

    scale = color_cfg.get("scale", None)
    if scale is None:
        if axc_is_horizontal(frame):
            scale = axc_cfg.get("xscale", axc_cfg.get("yscale", None))
        else:
            scale = axc_cfg.get("yscale", axc_cfg.get("xscale", None))
    if isinstance(scale, str):
        scale = scale.strip().lower()
        if scale:
            out["scale"] = scale

    return out


def _coerce_scalar(v):
    try:
        vv = float(v)
        if np.isfinite(vv):
            return vv
    except Exception:
        pass
    return None


def _resolve_norm(nv, *, vmin=None, vmax=None, logger=None):
    if nv is None:
        return None
    if isinstance(nv, mcolors.Normalize):
        return nv
    if isinstance(nv, str):
        key = nv.strip().lower()
        if key in {"log", "lognorm"}:
            if (vmin is None) or (vmax is None) or (vmin <= 0) or (vmax <= 0):
                if logger:
                    logger.warning("axc log scale requires positive vmin/vmax; fallback to linear Normalize.")
                return mcolors.Normalize(vmin=vmin, vmax=vmax)
            return mcolors.LogNorm(vmin=vmin, vmax=vmax)
        if key in {"twoslopenorm", "diverging"}:
            return mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
        if key in {"symlog", "symlognorm"}:
            return mcolors.SymLogNorm(linthresh=1.0, linscale=1.0, base=10, vmin=vmin, vmax=vmax)
        return mcolors.Normalize(vmin=vmin, vmax=vmax)
    if isinstance(nv, dict):
        t = str(nv.get("type", "Normalize")).strip().lower()
        _vmin = nv.get("vmin", vmin)
        _vmax = nv.get("vmax", vmax)
        if t in {"log", "lognorm"}:
            if (_vmin is None) or (_vmax is None) or (_vmin <= 0) or (_vmax <= 0):
                if logger:
                    logger.warning("axc log norm requires positive vmin/vmax; fallback to linear Normalize.")
                return mcolors.Normalize(vmin=_vmin, vmax=_vmax)
            return mcolors.LogNorm(vmin=_vmin, vmax=_vmax)
        if t in {"twoslopenorm", "diverging"}:
            vcenter = nv.get("vcenter", 0.0)
            return mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=_vmin, vmax=_vmax)
        if t in {"symlog", "symlognorm"}:
            linthresh = nv.get("linthresh", 1.0)
            linscale = nv.get("linscale", 1.0)
            base = nv.get("base", 10)
            return mcolors.SymLogNorm(
                linthresh=linthresh,
                linscale=linscale,
                base=base,
                vmin=_vmin,
                vmax=_vmax,
            )
        return mcolors.Normalize(vmin=_vmin, vmax=_vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def collect_and_attach_colorbar(fig, style: dict, coor: dict, method_key: str, df):
    axc = fig.axes.get("axc")
    if axc is None or not hasattr(axc, "_cb"):
        return style

    s = dict(style)
    colored_z_methods = {
        "contour",
        "contourf",
        "tricontour",
        "tricontourf",
        "tripcolor",
        "tripcolor_axes",
        "pcolor",
        "pcolormesh",
        "imshow",
        "voronoi",
        "voronoif",
        "grid_profile",
        "grid_profiling",
    }
    uses_color = bool(s.get("cmap")) or ("c" in coor) or (("z" in coor) and (method_key in colored_z_methods))
    if not uses_color:
        return style

    color_cfg = axc_color_config(fig.frame)

    def _load_color_series():
        arr = None
        if isinstance(coor.get("z"), dict) and ("expr" in coor["z"]):
            arr = eval_dataframe_expression(df, coor["z"]["expr"], logger=fig.logger)
        elif isinstance(coor.get("c"), dict) and ("expr" in coor["c"]):
            arr = eval_dataframe_expression(df, coor["c"]["expr"], logger=fig.logger)
        elif "z" in s:
            arr = s.get("z")
        elif "c" in s and not isinstance(s.get("c"), str):
            arr = s.get("c")
        if arr is None:
            return None
        try:
            arr = np.asarray(arr, dtype=float)
        except Exception:
            return None
        arr = arr[np.isfinite(arr)]
        return arr if arr.size else None

    chosen_cmap = color_cfg.get("cmap", s.get("cmap"))
    if chosen_cmap is not None:
        s["cmap"] = chosen_cmap
    fig.axc._cb["cmap"] = chosen_cmap

    data_series = _load_color_series()
    data_vmin = float(np.min(data_series)) if data_series is not None else None
    data_vmax = float(np.max(data_series)) if data_series is not None else None

    frame_vmin = _coerce_scalar(color_cfg.get("vmin", None))
    frame_vmax = _coerce_scalar(color_cfg.get("vmax", None))
    style_vmin = _coerce_scalar(s.get("vmin", None))
    style_vmax = _coerce_scalar(s.get("vmax", None))

    if frame_vmin is not None:
        fig.axc._cb["vmin"] = frame_vmin
    else:
        cand_min = style_vmin if style_vmin is not None else data_vmin
        if cand_min is not None:
            cur_min = _coerce_scalar(fig.axc._cb.get("vmin", None))
            fig.axc._cb["vmin"] = cand_min if cur_min is None else min(cur_min, cand_min)

    if frame_vmax is not None:
        fig.axc._cb["vmax"] = frame_vmax
    else:
        cand_max = style_vmax if style_vmax is not None else data_vmax
        if cand_max is not None:
            cur_max = _coerce_scalar(fig.axc._cb.get("vmax", None))
            fig.axc._cb["vmax"] = cand_max if cur_max is None else max(cur_max, cand_max)

    if (fig.axc._cb["vmin"] is not None) and (fig.axc._cb["vmax"] is not None):
        vmin = float(fig.axc._cb["vmin"])
        vmax = float(fig.axc._cb["vmax"])
        if vmax < vmin:
            vmin, vmax = vmax, vmin
            fig.axc._cb["vmin"], fig.axc._cb["vmax"] = vmin, vmax

        requested_scale = str(color_cfg.get("scale", "")).lower()
        if requested_scale in {"linear", "norm", "normalize"}:
            resolved = mcolors.Normalize(vmin=vmin, vmax=vmax)
        elif requested_scale == "log":
            resolved = _resolve_norm("lognorm", vmin=vmin, vmax=vmax, logger=fig.logger)
        else:
            user_norm = style.get("norm", None)
            resolved = _resolve_norm(user_norm, vmin=vmin, vmax=vmax, logger=fig.logger)
            if resolved is None:
                resolved = mcolors.Normalize(vmin=vmin, vmax=vmax)

        fig.axc._cb["norm"] = resolved
        s["norm"] = fig.axc._cb["norm"]
        if isinstance(fig.axc._cb["norm"], mcolors.LogNorm):
            fig.axc._cb["mode"] = "log"
        elif isinstance(fig.axc._cb["norm"], mcolors.TwoSlopeNorm):
            fig.axc._cb["mode"] = "diverging"
        elif isinstance(fig.axc._cb["norm"], mcolors.SymLogNorm):
            fig.axc._cb["mode"] = "log"
        else:
            fig.axc._cb["mode"] = "norm"

    if method_key in ("contour", "contourf", "tricontour", "tricontourf") and fig.axc._cb["levels"] is None:
        lv = s.get("levels", 10)
        if isinstance(lv, int) and fig.axc._cb["vmin"] is not None and fig.axc._cb["vmax"] is not None:
            fig.axc._cb["levels"] = np.linspace(fig.axc._cb["vmin"], fig.axc._cb["vmax"], lv)
        elif hasattr(lv, "__len__"):
            fig.axc._cb["levels"] = lv
    if "norm" in s and s["norm"] is not None:
        s.pop("vmin", None)
        s.pop("vmax", None)
        s.pop("mode", None)
    fig.axc._cb["used"] = uses_color
    return s

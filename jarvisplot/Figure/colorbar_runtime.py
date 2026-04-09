from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import matplotlib.colors as mcolors

from ..utils.expression import eval_dataframe_expression


# ---------------------------------------------------------------------------
# Frame config helpers
# ---------------------------------------------------------------------------

def axc_is_horizontal(frame: Mapping[str, Any], cb_name: str = "axc") -> bool:
    cb_cfg = frame.get(cb_name, {}) if isinstance(frame, Mapping) else {}
    return str(cb_cfg.get("orientation", "")).lower() == "horizontal"


def axc_color_config(frame: Mapping[str, Any], cb_name: str = "axc") -> dict:
    """Return normalized color config for a named colorbar axis.

    Reads frame[cb_name].color.  A value of None or the string "auto" means
    auto-range from data.
    """
    out: dict[str, Any] = {}
    cb_cfg = frame.get(cb_name, {}) if isinstance(frame, Mapping) else {}
    if not isinstance(cb_cfg, dict):
        cb_cfg = {}
    color_cfg = cb_cfg.get("color", {})
    if not isinstance(color_cfg, dict):
        color_cfg = {}

    if "cmap" in color_cfg:
        out["cmap"] = color_cfg.get("cmap")

    for key in ("vmin", "vmax"):
        raw = color_cfg.get(key, None)
        if raw is None or str(raw).strip().lower() == "auto":
            out[key] = None          # explicit auto → leave as None
        else:
            try:
                val = float(raw)
                if np.isfinite(val):
                    out[key] = val
                # non-finite treated as auto (leave as None)
            except Exception:
                pass                 # non-numeric treated as auto

    scale = color_cfg.get("scale", None)
    if isinstance(scale, str):
        scale = scale.strip().lower()
        if scale:
            out["scale"] = scale

    return out


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------

def _coerce_scalar(v) -> Optional[float]:
    try:
        vv = float(v)
        if np.isfinite(vv):
            return vv
    except Exception:
        pass
    return None


def _is_numeric_color_source(spec) -> bool:
    """Return True when a color spec looks like numeric scalar/mapping data.

    We treat expression dictionaries and numeric array-likes as color sources,
    but ignore plain strings such as literal colors or colormap names.
    """
    if spec is None:
        return False
    if isinstance(spec, str):
        return False
    if isinstance(spec, Mapping):
        return "expr" in spec
    try:
        arr = np.asarray(spec, dtype=float)
    except Exception:
        return False
    if arr.size == 0:
        return False
    return bool(np.isfinite(arr).any())


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
                linthresh=linthresh, linscale=linscale, base=base,
                vmin=_vmin, vmax=_vmax,
            )
        return mcolors.Normalize(vmin=_vmin, vmax=_vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


# ---------------------------------------------------------------------------
# Pre-scan helpers
# ---------------------------------------------------------------------------

_COLORED_Z_METHODS = frozenset({
    "contour", "contourf", "tricontour", "tricontourf",
    "jpcontour", "jpcontourf", "jpfield",
    "tripcolor", "tripcolor_axes",
    "pcolor", "pcolormesh", "imshow",
    "voronoi", "voronoif",
    "grid_profile",
})

_CONTOUR_LIKE_METHODS = frozenset({
    "contour", "contourf", "tricontour", "tricontourf",
    "jpcontour", "jpcontourf",
})


def layer_uses_color(style: dict, coor: dict, method_key: str) -> bool:
    """Return True if this layer will need a colorbar."""
    c_src = coor.get("c")
    z_src = coor.get("z")

    if _is_numeric_color_source(c_src):
        return True
    if method_key in _COLORED_Z_METHODS and _is_numeric_color_source(z_src):
        return True

    style_c = style.get("c")
    style_z = style.get("z")
    if _is_numeric_color_source(style_c):
        return True
    if method_key in _COLORED_Z_METHODS and _is_numeric_color_source(style_z):
        return True

    return False


def collect_layer_color_range(df, coor: dict, style: dict, *, scale: str | None = None):
    """Extract (data_min, data_max) for the colour channel of a single layer.

    Returns (None, None) when no finite data is available.
    """
    arr = None
    if isinstance(coor.get("z"), dict) and "expr" in coor["z"]:
        arr = eval_dataframe_expression(df, coor["z"]["expr"], logger=None)
    elif isinstance(coor.get("c"), dict) and "expr" in coor["c"]:
        arr = eval_dataframe_expression(df, coor["c"]["expr"], logger=None)
    elif "z" in style and not isinstance(style.get("z"), str):
        arr = style.get("z")
    elif "c" in style and not isinstance(style.get("c"), str):
        arr = style.get("c")

    if arr is None:
        return None, None
    try:
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if str(scale or "").strip().lower() == "log":
            arr = arr[arr > 0]
        if arr.size == 0:
            return None, None
        return float(np.min(arr)), float(np.max(arr))
    except Exception:
        return None, None


def _validate_colorbar_limits(vmin, vmax, scale: str, logger=None):
    """Validate vmin/vmax; replace invalid values with None (auto).

    Rules applied in order:
    1. Non-finite value  → warn, fall back to auto.
    2. log scale + value <= 0  → warn, fall back to auto for that value.
    3. Both set but vmin >= vmax  → warn, fall back to auto for both.
    """
    scale = str(scale or "").strip().lower()

    # Rule 1: finiteness
    if vmin is not None and not np.isfinite(vmin):
        if logger:
            logger.warning(f"colorbar vmin={vmin!r} is not finite; falling back to auto.")
        vmin = None
    if vmax is not None and not np.isfinite(vmax):
        if logger:
            logger.warning(f"colorbar vmax={vmax!r} is not finite; falling back to auto.")
        vmax = None

    # Rule 2: log scale positivity
    if scale == "log":
        if vmin is not None and vmin <= 0:
            if logger:
                logger.warning(
                    f"colorbar log scale requires vmin > 0, got vmin={vmin}; "
                    "falling back to auto."
                )
            vmin = None
        if vmax is not None and vmax <= 0:
            if logger:
                logger.warning(
                    f"colorbar log scale requires vmax > 0, got vmax={vmax}; "
                    "falling back to auto."
                )
            vmax = None

    # Rule 3: ordering
    if vmin is not None and vmax is not None and vmin >= vmax:
        if logger:
            logger.warning(
                f"colorbar vmin={vmin} >= vmax={vmax}; "
                "falling back to auto for both."
            )
        vmin = None
        vmax = None

    return vmin, vmax


def precompute_colorbar_cb(color_cfg: dict, data_ranges: list, logger=None) -> dict:
    """Build the complete _cb state dict from frame color config + data ranges.

    vmin and vmax are resolved independently:
    - Explicit value in frame config that passes validation → used as-is.
    - Missing or invalid value → auto from the data range of bound layers.

    This function is called once per colorbar during the pre-scan phase in
    Figure._prescan_colorbar_ranges(), before any layer is rendered.  After
    this call _cb is treated as read-only during rendering.

    Args:
        color_cfg: result of axc_color_config() for this colorbar.
        data_ranges: list of (data_min, data_max) collected from pre-scanned layers.
        logger: optional loguru logger.

    Returns:
        Complete _cb dict to be stored on the colorbar axes object.
    """
    scale = str(color_cfg.get("scale", "linear")).strip().lower()

    cb: dict[str, Any] = {
        "mode":   "auto",
        "levels": None,
        "vmin":   None,
        "vmax":   None,
        "norm":   None,
        "cmap":   color_cfg.get("cmap"),
        "used":   bool(data_ranges),
    }

    if not data_ranges:
        return cb

    # --- Resolve vmin (explicit wins; absent → auto from data) ---
    frame_vmin = color_cfg.get("vmin")
    if frame_vmin is not None:
        cb["vmin"] = frame_vmin
    else:
        valid_mins = [r[0] for r in data_ranges if r[0] is not None]
        cb["vmin"] = min(valid_mins) if valid_mins else None

    # --- Resolve vmax (explicit wins; absent → auto from data) ---
    frame_vmax = color_cfg.get("vmax")
    if frame_vmax is not None:
        cb["vmax"] = frame_vmax
    else:
        valid_maxes = [r[1] for r in data_ranges if r[1] is not None]
        cb["vmax"] = max(valid_maxes) if valid_maxes else None

    # --- Validate (may reset invalid values back to None) ---
    cb["vmin"], cb["vmax"] = _validate_colorbar_limits(
        cb["vmin"], cb["vmax"], scale, logger=logger
    )

    # --- Second-pass auto: if validation reset a value, refill from data ---
    if cb["vmin"] is None:
        valid_mins = [r[0] for r in data_ranges if r[0] is not None]
        cb["vmin"] = min(valid_mins) if valid_mins else None
    if cb["vmax"] is None:
        valid_maxes = [r[1] for r in data_ranges if r[1] is not None]
        cb["vmax"] = max(valid_maxes) if valid_maxes else None

    # --- Build norm ---
    if cb["vmin"] is not None and cb["vmax"] is not None:
        if scale in {"linear", "norm", "normalize", ""}:
            resolved = mcolors.Normalize(vmin=cb["vmin"], vmax=cb["vmax"])
        elif scale == "log":
            resolved = _resolve_norm("lognorm", vmin=cb["vmin"], vmax=cb["vmax"], logger=logger)
        else:
            resolved = _resolve_norm(scale, vmin=cb["vmin"], vmax=cb["vmax"], logger=logger)
            if resolved is None:
                resolved = mcolors.Normalize(vmin=cb["vmin"], vmax=cb["vmax"])

        cb["norm"] = resolved
        if isinstance(resolved, mcolors.LogNorm):
            cb["mode"] = "log"
        elif isinstance(resolved, mcolors.TwoSlopeNorm):
            cb["mode"] = "diverging"
        elif isinstance(resolved, mcolors.SymLogNorm):
            cb["mode"] = "log"
        else:
            cb["mode"] = "norm"

    return cb


# ---------------------------------------------------------------------------
# Render-time attachment  (read-only after pre-scan)
# ---------------------------------------------------------------------------

def collect_and_attach_colorbar(
    fig, style: dict, coor: dict, method_key: str, df, cb_name: str = "axc"
) -> dict:
    """Attach pre-computed colorbar norm/cmap to the layer style dict.

    Called once per layer during rendering.  By this point _cb has already
    been fully built by Figure._prescan_colorbar_ranges() so this function
    is read-only — it never mutates _cb.

    The pre-scan / read-only split ensures:
    - colorbar limits are determined by ALL bound layers, not just those
      rendered so far (no render-order dependency).
    - the final colorbar is always predictable from the YAML alone:
      explicit frame.axc.color.vmin/vmax win; absent values fall back to
      the data-driven auto range.
    """
    if not layer_uses_color(style, coor, method_key):
        return style

    # If the layer already declares an explicit contour color, do not inject a
    # colormap.  Matplotlib treats `colors` and `cmap` as mutually exclusive for
    # contour-style artists, and overlay layers often want a fixed line color
    # while another layer owns the shared colorbar.
    if method_key in _CONTOUR_LIKE_METHODS and (
        "colors" in style or "color" in style
    ):
        s = dict(style)
        s.pop("cmap", None)
        s.pop("norm", None)
        s.pop("vmin", None)
        s.pop("vmax", None)
        s.pop("mode", None)
        return s

    axc = fig.axes.get(cb_name)
    if axc is None or not hasattr(axc, "_cb"):
        return style

    if not isinstance(axc._cb, dict):
        axc._cb = {
            "mode": "auto",
            "levels": None,
            "vmin": None,
            "vmax": None,
            "norm": None,
            "cmap": None,
            "used": False,
        }

    if not axc._cb.get("used"):
        frame = getattr(fig, "frame", {})
        color_cfg = axc_color_config(frame, cb_name)
        data_range = collect_layer_color_range(
            df,
            coor,
            style,
            scale=color_cfg.get("scale"),
        )
        if data_range != (None, None):
            axc._cb.update(
                precompute_colorbar_cb(color_cfg, [data_range], logger=getattr(fig, "logger", None))
            )

    if not axc._cb.get("used"):
        return style

    s = dict(style)

    # Attach cmap
    chosen_cmap = axc._cb.get("cmap")
    if chosen_cmap is not None:
        s["cmap"] = chosen_cmap

    # Attach norm (strip raw vmin/vmax so matplotlib uses the norm object)
    norm = axc._cb.get("norm")
    if norm is not None:
        s["norm"] = norm
        s.pop("vmin", None)
        s.pop("vmax", None)
        s.pop("mode", None)

    # Contour levels (lazily generated on first contour layer, then reused)
    if (
        method_key in ("contour", "contourf", "tricontour", "tricontourf", "jpcontour", "jpcontourf")
        and axc._cb.get("levels") is None
        and axc._cb.get("vmin") is not None
        and axc._cb.get("vmax") is not None
    ):
        lv = s.get("levels", 10)
        if isinstance(lv, int):
            axc._cb["levels"] = np.linspace(axc._cb["vmin"], axc._cb["vmax"], lv)
        elif hasattr(lv, "__len__"):
            axc._cb["levels"] = lv
    if axc._cb.get("levels") is not None:
        s["levels"] = axc._cb["levels"]

    return s

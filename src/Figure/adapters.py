# src/Figure/adapters.py
#!/usr/bin/env python3 

from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
import os
import json

from matplotlib.axes import Axes

# ---- helpers for masking by extend on filled contours ----

def _resolve_vlim(z, vmin=None, vmax=None, levels=None, norm=None):
    """Derive effective vmin/vmax from norm/levels/fallback to data."""
    import numpy as _np
    if norm is not None:
        vmin = getattr(norm, "vmin", vmin)
        vmax = getattr(norm, "vmax", vmax)
    if levels is not None and _np.ndim(levels) > 0:
        vmin = levels[0] if vmin is None else vmin
        vmax = levels[-1] if vmax is None else vmax
    if vmin is None:
        vmin = float(_np.nanmin(z))
    if vmax is None:
        vmax = float(_np.nanmax(z))
    return float(vmin), float(vmax)


def _mask_by_extend(z, *, extend="neither", vmin=None, vmax=None, levels=None, norm=None):
    """
    Return masked z according to extend semantics:
      - 'min'  : mask z < vmin
      - 'max'  : mask z > vmax
      - 'both' : mask outside [vmin, vmax]
      - 'neither': no masking
    Also returns effective (vmin, vmax).
    """
    import numpy as _np
    z = _np.asarray(z)
    e = (extend or "neither").lower()
    if e not in ("neither", "min", "max", "both"):
        e = "neither"
    vmin_eff, vmax_eff = _resolve_vlim(z, vmin=vmin, vmax=vmax, levels=levels, norm=norm)
    mask = _np.zeros_like(z, dtype=bool)
    if e in ("min", "both"):
        mask |= (z < vmin_eff)
    if e in ("max", "both"):
        mask |= (z > vmax_eff)
    return _np.ma.masked_array(z, mask=mask), vmin_eff, vmax_eff

# —— 小工具：对 artist 或容器做统一 clip_path 应用 ——
def _auto_clip(artists, ax: Axes, clip_path):
    if clip_path is None:
        return artists
    def _apply_one(a):
        try:
            a.set_clip_path(clip_path, transform=ax.transData)
        except Exception:
            pass
    # always apply to the container itself first
    _apply_one(artists)
    try:
        iter(artists)
    except TypeError:
        return artists
    else:
        for item in artists:
            # 先试 artist 本体
            _apply_one(item)
            # matplotlib 常见容器：collections / patches / lines
            coll = getattr(item, "collections", None)
            if coll:
                for c in coll:
                    _apply_one(c)
            patches = getattr(item, "patches", None)
            if patches:
                for p in patches:
                    _apply_one(p)
            lines = getattr(item, "lines", None)
            if lines:
                for ln in lines:
                    _apply_one(ln)
        return artists

# —— 基础适配器：转发到底层 Axes，合并默认参数，做自动裁剪 ——
class StdAxesAdapter:
    def __init__(self, ax: Axes, defaults: Optional[Dict[str, Dict[str, Any]]] = None,
                 clip_path=None):
        """
        ax: 原始 matplotlib Axes
        defaults: 分方法的默认参数，如 {"scatter": {"s": 8, "alpha": 0.8}}
        clip_path: 可选 Path/PathPatch，用于 set_clip_path
        """
        self.ax = ax
        self._defaults = defaults or {}
        self._clip_path = clip_path  # None 表示不用裁剪
        self.config = self._load_internal_config()
        self._legend = False 

    def _load_internal_config(self):
        default_path = os.path.join(os.path.dirname(__file__), "cards", "std_axes_adapter_config.json")
        try:
            with open(default_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # 可选：返回默认空配置或 raise
            return {}

    # 参数合并：用户优先，默认兜底
    def _merge(self, method: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        base = dict(self._defaults.get(method, {}))
        base.update(kwargs or {})
        return base

    # —— 常用方法转发（需要的就加） ——
    def scatter(self, **kwargs):
        x, y = kwargs.pop("x"), kwargs.pop("y")
        print(x, kwargs)
        kw = self._merge("scatter", kwargs)
        artists = self.ax.scatter(x, y, **kw)
        return _auto_clip(artists, self.ax, self._clip_path)

    def plot(self, *args, **kwargs):
        x, y = kwargs.pop("x"), kwargs.pop("y")
        kw = self._merge("plot", kwargs)
        artists = self.ax.plot(x, y, **kw)        
        print("Plot method succesful")
        return _auto_clip(artists, self.ax, self._clip_path)

    def contour(self, *args, **kwargs):
        kw = self._merge("contour", kwargs)
        artists = self.ax.contour(*args, **kw)
        return _auto_clip(artists, self.ax, self._clip_path)

    def contourf(self, *args, **kwargs):
        kw = self._merge("contourf", kwargs)
        artists = self.ax.contourf(*args, **kw)
        
        return _auto_clip(artists, self.ax, self._clip_path)

    def imshow(self, *args, **kwargs):
        kw = self._merge("imshow", kwargs)
        artists = self.ax.imshow(*args, **kw)
        return _auto_clip(artists, self.ax, self._clip_path)
    
    def tricontour(self, **kwargs):
        x, y, z = kwargs.pop("x"), kwargs.pop("y"), kwargs.pop("z")
        import matplotlib.tri as tri
        triang = tri.Triangulation(x, y)
        refiner = tri.UniformTriRefiner(triang)
        tri_refi, z_test_refi = refiner.refine_field(z, subdiv=3)
        kw = self._merge("tricontour", kwargs)
        artists = self.ax.tricontour(tri_refi, z_test_refi, **kw)
        return _auto_clip(artists, self.ax, self._clip_path)

    def tricontourf(self, **kwargs):
        x, y, z = kwargs.pop("x"), kwargs.pop("y"), kwargs.pop("z")
        import matplotlib.tri as tri
        import numpy as _np

        triang = tri.Triangulation(x, y)
        refiner = tri.UniformTriRefiner(triang)
        tri_refi, z_refi = refiner.refine_field(z, subdiv=3)

        # Merge defaults for tricontourf (not tricontour)
        kw = self._merge("tricontourf", kwargs)

        # Mask by extend; also resolve effective vmin/vmax for color scaling
        z_masked, vmin_eff, vmax_eff = _mask_by_extend(
            z_refi,
            extend=kw.get("extend", "neither"),
            vmin=kw.get("vmin"),
            vmax=kw.get("vmax"),
            levels=kw.get("levels"),
            norm=kw.get("norm"),
        )

        # Avoid Matplotlib conflict: norm cannot be combined with vmin/vmax
        if kw.get("norm") is not None:
            kw.pop("vmin", None)
            kw.pop("vmax", None)
        else:
            kw.setdefault("vmin", vmin_eff)
            kw.setdefault("vmax", vmax_eff)

        # Mask triangles that include any masked vertex
        z_mask_arr = _np.ma.getmaskarray(z_masked)
        if z_mask_arr is not False and z_mask_arr is not None:
            tri_mask = _np.any(z_mask_arr[tri_refi.triangles], axis=1)
            tri_refi.set_mask(tri_mask)
            z_for_plot = _np.asarray(_np.ma.filled(z_masked, 0.0))
        else:
            z_for_plot = _np.asarray(z_masked)

        artists = self.ax.tricontourf(tri_refi, z_for_plot, **kw)
        return _auto_clip(artists, self.ax, self._clip_path)

    # 为了兼容现有框架，暴露底层的方法/属性
    def __getattr__(self, name: str):
        # 未覆写的方法透传给原始 Axes
        return getattr(self.ax, name)

    def hist(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        stacked = kwargs.get('stacked', False)
        colors = kwargs.get('color', None) or kwargs.get('colors', None)

        # 获取数据分组数：支持 x 作为 args[0] 或 kwargs['x']
        x = kwargs.get('x', args[0] if args else None)
        n_groups = None
        if stacked and colors is None and x is not None:
            # 只有当x为二维数据时，len(x)为分组数（多数组堆叠情况）
            try:
                n_groups = len(x) if hasattr(x, '__len__') and not isinstance(x, (str, bytes)) else 1
            except Exception:
                n_groups = 1
            default_colors = self._get_default_colors(n_groups)
            kwargs['color'] = default_colors

        kw = self._merge("hist", kwargs)
        artists = self.ax.hist(*args, **kw)
        return _auto_clip(artists, self.ax, self._clip_path)

    def _get_default_colors(self, n):
        import matplotlib.pyplot as plt
        # print(getattr(self, "color_palette"))
        palette = self.config['hist']['color']
        print(len(palette), n)
        
        # palette = getattr(self, 'color_palette', plt.cm.tab10.colors)
        # print(palette)
        return [palette[i % len(palette)] for i in range(n)]

# —— Ternary 适配器：在 Std 基础上增加 (a,b,c)->(x,y) 投影 ——
class TernaryAxesAdapter(StdAxesAdapter):
    def __init__(self, ax: Axes, defaults: Optional[Dict[str, Dict[str, Any]]] = None,
                 clip_path=None):
        super().__init__(ax, defaults=defaults, clip_path=clip_path)

    @staticmethod
    def _lbr_to_xy(a, b, c):
        s = (a + b + c)
        s = np.where(s == 0.0, 1.0, s)  # 避免除零
        aa, bb, cc = a/s, b/s, c/s
        x = bb + 0.5 * cc
        y = cc
        return x, y

    def scatter(self, **kwargs):
        if {"left", "right", "bottom"}.issubset(kwargs.keys()): 
            x, y = self._lbr_to_xy(kwargs.pop('left'), kwargs.pop('right'), kwargs.pop('bottom'))
            kwargs['x'] = x 
            kwargs['y'] = y 
        return super().scatter(**kwargs)
        
    
    def plot(self, **kwargs):
        if {"left", "right", "bottom"}.issubset(kwargs.keys()): 
            x, y = self._lbr_to_xy(kwargs.pop('left'), kwargs.pop('right'), kwargs.pop('bottom'))
            kwargs['x'] = x 
            kwargs['y'] = y 
        return super().plot(**kwargs)

        
    def tricontour(self, **kwargs):
        if {"left", "right", "bottom"}.issubset(kwargs.keys()): 
            x, y = self._lbr_to_xy(kwargs.pop('left'), kwargs.pop('right'), kwargs.pop('bottom'))
            kwargs['x'] = x 
            kwargs['y'] = y         
        return super().tricontour( **kwargs)

        
    def tricontourf(self, **kwargs):
        if {"left", "right", "bottom"}.issubset(kwargs.keys()): 
            x, y = self._lbr_to_xy(kwargs.pop('left'), kwargs.pop('right'), kwargs.pop('bottom'))
            kwargs['x'] = x 
            kwargs['y'] = y 

        return super().tricontourf(**kwargs)
        # else:
        #     raise ValueError("scatter() needs either (a,b,c) or (x,y) inputs")


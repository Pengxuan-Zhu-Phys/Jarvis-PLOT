# src/Figure/adapters.py
#!/usr/bin/env python3 

from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from matplotlib.axes import Axes

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
        print("tricontour method ...")
        x, y, z = kwargs.pop("x"), kwargs.pop("y"), kwargs.pop("z")
        import matplotlib.tri as tri 
        triang = tri.Triangulation(x, y)
        refiner = tri.UniformTriRefiner(triang)
        tri_refi, z_test_refi = refiner.refine_field(z, subdiv=3)
        kw = self._merge("tricontour", kwargs)
        artists = self.ax.tricontour(tri_refi, z_test_refi, **kw)
        print("tricontour method succesful")
        return _auto_clip(artists, self.ax, self._clip_path)

    def tricontourf(self, **kwargs):
        print("Line 99 -> ", kwargs.keys())
        x, y, z = kwargs.pop("x"), kwargs.pop("y"), kwargs.pop("z")
        print(x, y, z)
        import matplotlib.tri as tri 
        triang = tri.Triangulation(x, y)
        refiner = tri.UniformTriRefiner(triang)
        tri_refi, z_test_refi = refiner.refine_field(z, subdiv=3)
        kw = self._merge("tricontour", kwargs)
        artists = self.ax.tricontourf(tri_refi, z_test_refi, **kw)
        # return _auto_clip(artists, self.ax, self._clip_path)
        return artists

    # 为了兼容现有框架，暴露底层的方法/属性
    def __getattr__(self, name: str):
        # 未覆写的方法透传给原始 Axes
        return getattr(self.ax, name)

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
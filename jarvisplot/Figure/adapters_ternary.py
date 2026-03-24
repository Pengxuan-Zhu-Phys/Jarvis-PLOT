#!/usr/bin/env python3

from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
import os
import json

from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
#
from .helper import _auto_clip, _mask_by_extend, voronoi_finite_polygons_2d, _clip_poly_to_rect
from .adapters_rect import StdAxesAdapter


# —— Basic Adapter: Forward to the underlying Axes, merge default parameters, perform automatic clipping ——

class TernaryAxesAdapter(StdAxesAdapter):
    def __init__(self, ax: Axes, defaults: Optional[Dict[str, Any]] = None,
                 clip_path=None):
        # Allow a flat defaults dict like {"facecolor": "..."} and keep it internal.
        d = dict(defaults) if isinstance(defaults, dict) else {}
        facecolor = d.pop("facecolor", None)

        super().__init__(ax, defaults=d or None, clip_path=clip_path)

        if facecolor is not None:
            self.set_facecolor(facecolor)

        self.status = "init"

    def set_facecolor(self, color, zorder=-100):
        self.ax.fill(
            [0.0, 1.0, 0.5],
            [0.0, 0.0, 1.0],
            facecolor=color,
            edgecolor="none",
            zorder=zorder
        )

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

    def grid_profile(self, **kwargs):
        if {"left", "right", "bottom"}.issubset(kwargs.keys()):
            x, y = self._lbr_to_xy(kwargs.pop('left'), kwargs.pop('right'), kwargs.pop('bottom'))
            kwargs['x'] = x
            kwargs['y'] = y
        return super().grid_profile(**kwargs)


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

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


# —— Basic Adapter: Forward to the underlying Axes, merge default parameters, perform automatic clipping ——
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
        self._clip_path = clip_path  # None means no cropping
        self.config = self._load_internal_config()
        self._legend = False
        self.status = "init"           # lifecycle: init -> configured -> drawn -> finalized
        self.needs_finalize = True      # allow some axes (e.g., logo) to opt out

    def finalize(self):
        """Finalize axes after all layers/legends/colorbars applied.
        Override in specialized adapters if needed. Here we just mark status.
        """
        self.status = "finalized"

    def _load_internal_config(self):
        default_path = os.path.join(os.path.dirname(__file__), "cards", "std_axes_adapter_config.json")
        try:
            with open(default_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # Optional: return default empty configuration or raise
            return {}

    # Parameter merging: user preference takes priority, default as fallback
    def _merge(self, method: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        base = dict(self._defaults.get(method, {}))
        base.update(kwargs or {})
        return base

    # —— Common method forwarding (add as needed) ——
    def scatter(self, **kwargs):
        x, y = kwargs.pop("x"), kwargs.pop("y")
        kw = self._merge("scatter", kwargs)
        artists = self.ax.scatter(x, y, **kw)
        return _auto_clip(artists, self.ax, self._clip_path)

    def plot(self, *args, **kwargs):
        x, y = kwargs.pop("x"), kwargs.pop("y")
        kw = self._merge("plot", kwargs)
        artists = self.ax.plot(x, y, **kw)
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

        triang = tri.Triangulation(x, y)
        refiner = tri.UniformTriRefiner(triang)
        tri_refi, z_refi = refiner.refine_field(z, subdiv=3)

        kw = self._merge("tricontourf", kwargs)

        z_masked, vmin_eff, vmax_eff = _mask_by_extend(
            z_refi,
            extend=kw.get("extend", "neither"),
            vmin=kw.get("vmin"),
            vmax=kw.get("vmax"),
            levels=kw.get("levels"),
            norm=kw.get("norm"),
        )
        try:
            print("Adapter 184 -> ", z_refi.max(), z_refi.min())
            if kw.get("levels", False) and isinstance(kw.get("levels", False), int):
                kw["levels"] = np.linspace(kw.get("vmin"), kw.get("vmax"), kw.get("levels"))
        except TypeError:
            pass
        if kw.get("norm") is not None:
            kw.pop("vmin", None)
            kw.pop("vmax", None)
        else:
            kw.setdefault("vmin", vmin_eff)
            kw.setdefault("vmax", vmax_eff)

        z_mask_arr = np.ma.getmaskarray(z_masked)
        if z_mask_arr is not False and z_mask_arr is not None:
            tri_mask = np.any(z_mask_arr[tri_refi.triangles], axis=1)
            tri_refi.set_mask(tri_mask)
            z_for_plot = np.asarray(np.ma.filled(z_masked, 0.0))
        else:
            z_for_plot = np.asarray(z_masked)

        artists = self.ax.tricontourf(tri_refi, z_for_plot, **kw)
        return _auto_clip(artists, self.ax, self._clip_path)

    def voronoi(self, **kwargs):
        """Fill regions by Voronoi cells colored by z at each site."""
        import matplotlib as mpl
        x = kwargs.pop("x")
        y = kwargs.pop("y")
        z = kwargs.pop("z")
        cmap = kwargs.pop("cmap", None)
        # Matplotlib colormap resolution for string names
        if isinstance(cmap, str):
            try:
                cmap = mpl.colormaps.get(cmap)
            except Exception:
                cmap = None
        # Now proceed as before
        import numpy as _np
        try:
            from scipy.spatial import Voronoi
        except Exception as e:
            raise ImportError("voronoi requires scipy.spatial.Voronoi. Please install scipy.") from e

        # ---- inputs ----
        x = _np.asarray(x)
        y = _np.asarray(y)
        z = _np.asarray(z)
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        edgecolor = kwargs.pop("edgecolor", 'none')
        lw = kwargs.pop("linewidth", kwargs.pop("linewidths", 0.0))
        extent = kwargs.pop("extent", None)   # data-space
        radius = kwargs.pop("radius", None)
        nan_color = kwargs.pop("nan_color", None)
        zorder = kwargs.pop("zorder", None)

        # ---- derive view box & transforms from axes ----
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        if extent is None:
            extent = (min(xlim), max(xlim), min(ylim), max(ylim))
        xmin, xmax, ymin, ymax = extent

        t_data_to_disp = self.ax.transData
        t_disp_to_data = t_data_to_disp.inverted()

        disp_ll = t_data_to_disp.transform((xmin, ymin))
        disp_ur = t_data_to_disp.transform((xmax, ymax))
        disp_x0, disp_y0 = disp_ll
        disp_x1, disp_y1 = disp_ur
        if disp_x1 == disp_x0:
            disp_x1 = disp_x0 + 1.0
        if disp_y1 == disp_y0:
            disp_y1 = disp_y0 + 1.0
        if disp_x1 < disp_x0:
            disp_x0, disp_x1 = disp_x1, disp_x0
        if disp_y1 < disp_y0:
            disp_y0, disp_y1 = disp_y1, disp_y0
        disp_w = (disp_x1 - disp_x0)
        disp_h = (disp_y1 - disp_y0)

        pts_disp = t_data_to_disp.transform(_np.c_[x, y])
        pts_norm = _np.c_[ (pts_disp[:,0] - disp_x0)/disp_w, (pts_disp[:,1] - disp_y0)/disp_h ]

        if not _np.all(_np.isfinite(pts_norm)):
            pts_norm = _np.nan_to_num(pts_norm, nan=0.5, posinf=1.0, neginf=0.0)

        vor = Voronoi(pts_norm)

        from .helper import voronoi_finite_polygons_2d, _clip_poly_to_rect
        regions, vertices = voronoi_finite_polygons_2d(vor, radius=radius)
        unit_rect = (0.0, 1.0, 0.0, 1.0)

        polys_valid, zvals_valid = [], []
        polys_bg = []
        def _is_invalid(val):
            try:
                return (val is None) or (not _np.isfinite(float(val)))
            except Exception:
                return True

        for i_pt, region in enumerate(regions):
            if not region:
                continue
            poly = vertices[region]
            poly = [(float(px), float(py)) for px, py in poly]
            poly = _clip_poly_to_rect(poly, unit_rect)
            if len(poly) < 3:
                continue
            poly_disp = _np.c_[_np.array([p[0] for p in poly])*disp_w + disp_x0,
                               _np.array([p[1] for p in poly])*disp_h + disp_y0]
            poly_data = [tuple(p) for p in t_disp_to_data.transform(poly_disp)]
            val = z[i_pt]
            if _is_invalid(val):
                polys_bg.append(poly_data)
            else:
                polys_valid.append(poly_data)
                zvals_valid.append(float(val))

        from matplotlib.collections import PolyCollection
        artists = []
        if polys_bg:
            pc_bg = PolyCollection(polys_bg, facecolor=(nan_color if nan_color is not None else self.ax.get_facecolor()),
                                   edgecolor=edgecolor, linewidth=lw)
            artists.append(self.ax.add_collection(pc_bg))

        from matplotlib import colors as mcolors
        norm = kwargs.pop("norm", None)
        if norm is None and (vmin is not None or vmax is not None):
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        pc = PolyCollection(
            polys_valid,
            array=_np.asarray(zvals_valid),
            cmap=cmap,
            edgecolor=edgecolor,
            linewidth=lw,
            norm=norm,
        )
        if zorder is not None:
            pc.set_zorder(zorder)
        if norm is None and (vmin is not None or vmax is not None):
            pc.set_clim(vmin=vmin, vmax=vmax )

        artists.append(self.ax.add_collection(pc))
        return _auto_clip(artists, self.ax, self._clip_path)

    # 为了兼容现有框架，暴露底层的方法/属性
    def __getattr__(self, name: str):
        # 未覆写的方法透传给原始 Axes
        return getattr(self.ax, name)

    def hist(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        stacked = kwargs.get('stacked', False)
        colors = kwargs.get('color', None) or kwargs.get('colors', None)

        # Get the number of data groups: supports x as args[0] or kwargs['x']
        x = kwargs.get('x', args[0] if args else None)
        n_groups = None
        if stacked and colors is None and x is not None:
            # Only when x is two-dimensional data, len(x) is the number of groups (in the case of multiple arrays stacked)
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
        self.status = "init"

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

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

    def grid_profile(self, **kwargs):
        """Grid-cell partition rendering for profiled scalar fields.

        Compared with tripcolor, this draws axis-aligned cells (like a regular
        partition of the plane), closer to Voronoi-style region coloring.
        """
        x = np.asarray(kwargs.pop("x"), dtype=float)
        y = np.asarray(kwargs.pop("y"), dtype=float)
        z = np.asarray(kwargs.pop("z", np.zeros_like(x)), dtype=float)
        df = kwargs.pop("__df__", None)
        kw = self._merge("grid_profile", kwargs)

        # Keep compatibility with configs copied from tripcolor/scatter.
        for k in ("shading", "levels", "extend", "space", "marker", "s", "c"):
            kw.pop(k, None)

        cmap = kw.pop("cmap", None)
        norm = kw.pop("norm", None)
        vmin = kw.pop("vmin", None)
        vmax = kw.pop("vmax", None)
        alpha = kw.pop("alpha", None)
        zorder = kw.pop("zorder", None)
        antialiased = kw.pop("antialiased", False)
        edgecolor = kw.pop("edgecolor", kw.pop("ec", "none"))
        linewidth = kw.pop("linewidth", kw.pop("linewidths", 0.0))
        linestyle = kw.pop("linestyle", kw.pop("ls", "solid"))
        objective_from_style = ("objective" in kw)
        objective = str(kw.pop("objective", "max")).lower()

        # Optional explicit controls in style
        grid_bin = kw.pop("bin", None)
        xlim = kw.pop("xlim", None)
        ylim = kw.pop("ylim", None)
        xscale = str(kw.pop("xscale", self.ax.get_xscale())).lower()
        yscale = str(kw.pop("yscale", self.ax.get_yscale())).lower()

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
            # fallback when metadata is absent
            grid_bin = max(1, int(np.sqrt(max(len(x), 1))))
        grid_bin = max(int(grid_bin), 1)

        if xlim is None:
            xlim = [float(self.ax.get_xlim()[0]), float(self.ax.get_xlim()[1])]
        if ylim is None:
            ylim = [float(self.ax.get_ylim()[0]), float(self.ax.get_ylim()[1])]

        def _grid_edges(lo, hi, n, scale):
            lo = float(lo)
            hi = float(hi)
            if str(scale).lower() == "log":
                tiny = np.finfo(float).tiny
                lo = max(lo, tiny)
                hi = max(hi, lo * 10.0)
                if hi == lo:
                    hi = lo * 10.0
                return np.geomspace(lo, hi, n + 1)
            if hi == lo:
                hi = lo + 1.0
            return np.linspace(lo, hi, n + 1)

        def _norm(arr, lim, scale):
            arr = np.asarray(arr, dtype=float)
            lo = float(lim[0])
            hi = float(lim[1])
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

        n = min(len(x), len(y), len(z))
        if n == 0:
            return []
        x = x[:n]
        y = y[:n]
        z = z[:n]

        if ix is None or iy is None:
            xn = _norm(x, xlim, xscale)
            yn = _norm(y, ylim, yscale)
            valid = np.isfinite(xn) & np.isfinite(yn)
            if not np.any(valid):
                return []
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
                return []
            ix = ix[valid]
            iy = iy[valid]
            zv = z[valid]

        try:
            import pandas as _pd
            zkey = np.where(np.isfinite(zv), zv, np.inf if objective == "min" else -np.inf)
            tmp = _pd.DataFrame({"ix": ix, "iy": iy, "z": zv, "zkey": zkey})
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
            return []
        ix_u = ix_u[in_range]
        iy_u = iy_u[in_range]
        z_u = z_u[in_range]

        grid = np.full((grid_bin, grid_bin), np.nan, dtype=float)
        if len(ix_u) > 0:
            grid[iy_u, ix_u] = z_u
        grid = np.ma.masked_invalid(grid)

        kw.setdefault("shading", "flat")
        if edgecolor is not None:
            kw.setdefault("edgecolors", edgecolor)
        if linewidth is not None:
            kw.setdefault("linewidth", linewidth)
        if linestyle is not None:
            kw.setdefault("linestyle", linestyle)
        kw.setdefault("antialiased", antialiased)
        if alpha is not None:
            kw.setdefault("alpha", alpha)
        if zorder is not None:
            kw.setdefault("zorder", zorder)

        # histo2d-like rendering via pcolormesh
        if norm is not None:
            artist = self.ax.pcolormesh(
                x_edges,
                y_edges,
                grid,
                cmap=cmap,
                norm=norm,
                **kw,
            )
        else:
            artist = self.ax.pcolormesh(
                x_edges,
                y_edges,
                grid,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **kw,
            )
        return _auto_clip(artist, self.ax, self._clip_path)

    def plot(self, *args, **kwargs):
        x, y = kwargs.pop("x"), kwargs.pop("y")
        # print("x:", x, "y:", y)
        kw = self._merge("plot", kwargs)
        fmt = kw.pop("fmt", None)
        if fmt is not None: 
            artists = self.ax.plot(x, y, fmt, **kw)
        else: 
            artists = self.ax.plot(x, y, **kw)
        return _auto_clip(artists, self.ax, self._clip_path)

    def fill(self, **kwargs):
        x, y = kwargs.pop("x"), kwargs.pop("y")
        kw = self._merge("fill", kwargs)
        artists = self.ax.fill(x, y, **kw)
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
        kw = self._merge("tricontour", kwargs)
        subdiv = int(kw.pop("subdiv", 0) or 0)
        if subdiv > 0:
            refiner = tri.UniformTriRefiner(triang)
            triang, z = refiner.refine_field(z, subdiv=subdiv)
        artists = self.ax.tricontour(triang, z, **kw)
        return _auto_clip(artists, self.ax, self._clip_path)

    def tricontourf(self, **kwargs):
        x, y, z = kwargs.pop("x"), kwargs.pop("y"), kwargs.pop("z")
        import matplotlib.tri as tri

        triang = tri.Triangulation(x, y)
        kw = self._merge("tricontourf", kwargs)
        subdiv = int(kw.pop("subdiv", 0) or 0)
        if subdiv > 0:
            refiner = tri.UniformTriRefiner(triang)
            triang, z = refiner.refine_field(z, subdiv=subdiv)

        z_masked, vmin_eff, vmax_eff = _mask_by_extend(
            z,
            extend=kw.get("extend", "neither"),
            vmin=kw.get("vmin"),
            vmax=kw.get("vmax"),
            levels=kw.get("levels"),
            norm=kw.get("norm"),
        )
        # If user provided levels as an int, let Matplotlib handle it.
        # Only expand to an explicit array if BOTH vmin and vmax are provided.
        if isinstance(kw.get("levels", None), int):
            if kw.get("vmin") is not None and kw.get("vmax") is not None:
                kw["levels"] = np.linspace(kw["vmin"], kw["vmax"], int(kw["levels"]))
        if kw.get("norm") is not None:
            kw.pop("vmin", None)
            kw.pop("vmax", None)
        else:
            kw.setdefault("vmin", vmin_eff)
            kw.setdefault("vmax", vmax_eff)

        z_mask_arr = np.ma.getmaskarray(z_masked)
        if z_mask_arr is not False and z_mask_arr is not None:
            tri_mask = np.any(z_mask_arr[triang.triangles], axis=1)
            triang.set_mask(tri_mask)
            z_for_plot = np.asarray(np.ma.filled(z_masked, 0.0))
        else:
            z_for_plot = np.asarray(z_masked)

        artists = self.ax.tricontourf(triang, z_for_plot, **kw)
        return _auto_clip(artists, self.ax, self._clip_path)

    

    def voronoi(self, **kwargs):
        if {"x", "y", "z"}.issubset(kwargs.keys()):
            return self.voronoi_cmapfill(**kwargs)
        elif {"x", "y"}.issubset(kwargs.keys()): 
            return self.voronoi_colorfill(**kwargs)

    def voronoi_colorfill(self, **kwargs):
        """Fill selected Voronoi cells with a single facecolor (no z / no colorbar).

        Required:
          - x, y
        Optional:
          - where: boolean mask (same shape as x/y). If provided, only True cells are filled.
          - facecolor, edgecolor, linewidth/linewidths, draw_edges, antialiased, extent, radius, zorder
        """
        import numpy as _np
        try:
            from scipy.spatial import Voronoi
        except Exception as e:
            raise ImportError("voronoi requires scipy.spatial.Voronoi. Please install scipy.") from e

        x = kwargs.pop("x")
        y = kwargs.pop("y")
        where = kwargs.pop("where", None)

        # Keep matplotlib fill kwargs intact (facecolor/edgecolor/linewidth/alpha/etc.).
        # Only consume voronoi-specific options here.
        extent = kwargs.pop("extent", None)   # data-space
        radius = kwargs.pop("radius", None)

        if x.size == 0:
            return []

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

        # robust ordering / non-zero spans
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
        pts_norm = _np.c_[
            (pts_disp[:, 0] - disp_x0) / disp_w,
            (pts_disp[:, 1] - disp_y0) / disp_h,
        ]

        if not _np.all(_np.isfinite(pts_norm)):
            pts_norm = _np.nan_to_num(pts_norm, nan=0.5, posinf=1.0, neginf=0.0)

        vor = Voronoi(pts_norm)

        regions, vertices = voronoi_finite_polygons_2d(vor, radius=radius)
        unit_rect = (0.0, 1.0, 0.0, 1.0)

        polys_fill = []
        for i_pt, region in enumerate(regions):
            if not region:
                continue
            if where is not None and (not bool(where[i_pt])):
                continue

            poly = vertices[region]
            poly = [(float(px), float(py)) for px, py in poly]
            poly = _clip_poly_to_rect(poly, unit_rect)
            if len(poly) < 3:
                continue

            # norm -> display -> data
            poly_disp = _np.c_[
                _np.array([p[0] for p in poly]) * disp_w + disp_x0,
                _np.array([p[1] for p in poly]) * disp_h + disp_y0,
            ]
            poly_data = [tuple(p) for p in t_disp_to_data.transform(poly_disp)]
            polys_fill.append(poly_data)

        if len(polys_fill) == 0:
            return []

        # Merge selected Voronoi cells into one (possibly multi-) polygon, then fill using ax.fill
        try:
            from shapely.geometry import Polygon as _SHPPolygon
            from shapely.ops import unary_union as _shp_unary_union
        except Exception as e:
            raise ImportError("voronoi_colorfill merge requires shapely. Please install shapely.") from e

        shp_polys = []
        for poly in polys_fill:
            try:
                g = _SHPPolygon(poly)
                if not g.is_valid:
                    g = g.buffer(0)
                if (not g.is_empty) and g.area > 0:
                    shp_polys.append(g)
            except Exception:
                continue

        if not shp_polys:
            return []

        merged = _shp_unary_union(shp_polys)
        if merged.is_empty:
            return []

        # Inherit matplotlib fill kwargs (plus any defaults for 'fill')
        kw = self._merge("fill", kwargs)

        artists = []
        if merged.geom_type == "Polygon":
            parts = [merged]
        else:
            parts = list(getattr(merged, "geoms", []))

        for g in parts:
            if g.is_empty:
                continue
            xs, ys = g.exterior.coords.xy
            artists.extend(self.ax.fill(list(xs), list(ys), **kw))

        return _auto_clip(artists, self.ax, self._clip_path)

    def _tripcolor_impl(self, x, y, z, kw, axes_space: bool):
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

        if axes_space:
            # Map data -> axes coordinates; triangulation in display space is
            # robust under log scales and avoids visually stretched cells.
            pts = np.c_[x, y]
            pts_disp = self.ax.transData.transform(pts)
            pts_axes = self.ax.transAxes.inverted().transform(pts_disp)
            xa = pts_axes[:, 0]
            ya = pts_axes[:, 1]
            m = np.isfinite(xa) & np.isfinite(ya)
            # Keep only visible points in the axes box for stable triangulation.
            m &= (xa >= 0.0) & (xa <= 1.0) & (ya >= 0.0) & (ya <= 1.0)
            if m.sum() < 3:
                return []
            xa = xa[m]
            ya = ya[m]
            z = z[m]
        else:
            xa = x
            ya = y
            m = np.isfinite(xa) & np.isfinite(ya)
            if m.sum() < 3:
                return []
            xa = xa[m]
            ya = ya[m]
            z = z[m]

        # non-finite z should produce holes
        z_nonfinite = ~np.isfinite(z)

        # options consumed for masking behavior; remove before calling tripcolor
        extend_opt = kw.pop("extend", "neither")
        levels_opt = kw.pop("levels", None)
        z_masked, vmin_eff, vmax_eff = _mask_by_extend(
            z,
            extend=extend_opt,
            vmin=kw.get("vmin"),
            vmax=kw.get("vmax"),
            levels=levels_opt,
            norm=kw.get("norm"),
        )

        try:
            base_mask = np.ma.getmaskarray(z_masked)
            z_masked = np.ma.array(z_masked, mask=(base_mask | z_nonfinite))
        except Exception:
            z_masked = np.ma.array(z, mask=z_nonfinite)

        if kw.get("norm") is not None:
            kw.pop("vmin", None)
            kw.pop("vmax", None)
        else:
            kw.setdefault("vmin", vmin_eff)
            kw.setdefault("vmax", vmax_eff)

        import matplotlib.tri as tri

        tri_obj = tri.Triangulation(xa, ya)
        mask_v = np.ma.getmaskarray(z_masked)
        if mask_v is not False and mask_v is not None:
            tri_mask = np.any(mask_v[tri_obj.triangles], axis=1)
            tri_obj.set_mask(tri_mask)

        z_for_plot = np.asarray(np.ma.filled(z_masked, 0.0))
        if axes_space:
            artists = self.ax.tripcolor(
                tri_obj,
                z_for_plot,
                transform=self.ax.transAxes,
                **kw,
            )
        else:
            artists = self.ax.tripcolor(tri_obj, z_for_plot, **kw)

        return _auto_clip(artists, self.ax, self._clip_path)

    def tripcolor(self, **kwargs):
        """Triangulated color field.

        Default behavior uses axes-space triangulation (`axes.transAxes`) to
        keep cell geometry visually stable under log/linear axis transforms.
        Set `space: data` to force data-space triangulation.
        """
        x = kwargs.pop("x")
        y = kwargs.pop("y")
        z = kwargs.pop("z")

        # default: axes space
        space = str(kwargs.pop("space", "axes")).lower()
        use_axes_space = space != "data"
        kw = self._merge("tripcolor", kwargs)
        return self._tripcolor_impl(x=x, y=y, z=z, kw=kw, axes_space=use_axes_space)

    def tripcolor_axes(self, **kwargs):
        """Triangulated pseudocolor in axes coordinates.

        We map the input (x, y) from data coordinates to Axes coordinates (0..1)
        using the current axis scale and limits via Matplotlib transforms:
            (x, y)_data -> display -> axes

        Then we call `ax.tripcolor` with `transform=ax.transAxes` so the triangles
        are drawn in the Axes coordinate system. This makes the rendering robust
        and consistent with other "axes-space" primitives (e.g. Voronoi fills).

        Required:
          - x, y, z

        Common options:
          - shading (default: 'gouraud'), cmap, norm, vmin, vmax, alpha, zorder, etc.
          - extend/levels (optional): used to mask out-of-range z values consistently.
        """
        x = kwargs.pop("x")
        y = kwargs.pop("y")
        z = kwargs.pop("z")
        kw = self._merge("tripcolor", kwargs)
        kw.setdefault("shading", "gouraud")
        return self._tripcolor_impl(x=x, y=y, z=z, kw=kw, axes_space=True)

    def voronoi_cmapfill(self, **kwargs): 
        import matplotlib as mpl
        x = kwargs.pop("x")
        y = kwargs.pop("y")
        z = kwargs.pop("z")
        where = kwargs.pop("where", None)
        cmap = kwargs.pop("cmap", None)
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
        if where is not None:
            where = _np.asarray(where, dtype=bool)
            if where.shape != x.shape:
                raise ValueError("voronoi: 'where' must have the same shape as x/y")
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        edgecolor = kwargs.pop("edgecolor", 'none')
        draw_edges = kwargs.pop("draw_edges", True)
        antialiased = kwargs.pop("antialiased", False)
        orig_lw = kwargs.pop("linewidth", kwargs.pop("linewidths", 0.0))
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
            if where is not None and (not bool(where[i_pt])):
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

        artists = []
        from matplotlib import colors as mcolors
        norm = kwargs.pop("norm", None)
        if norm is None and (vmin is not None or vmax is not None):
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        pc = PolyCollection(
            polys_valid,
            array=_np.asarray(zvals_valid),
            cmap=cmap,
            edgecolor='none',
            linewidth=0.0,
            norm=norm,
            antialiased=antialiased,
        )
        if zorder is not None:
            pc.set_zorder(zorder)

        artists.append(self.ax.add_collection(pc))

        return _auto_clip(artists, self.ax, self._clip_path)

    def voronoif(self, **kwargs):
        """Hatched fill for a boundary layer of a where-selected Voronoi region.

        Algorithm:
          1) Use `where` to select Voronoi cells and union them into region A.
          2) For each selected cell, keep it only if the site point (core) is within
             `core_dist` (in axes-intrinsic unit-square coords) of the *boundary of A*.
             The kept cells form region B.
          3) Hatch-fill the union of B.

        Inputs:
          - x, y: 1D arrays of site positions (data coordinates)
          - where: optional 1D boolean array; True selects the corresponding Voronoi cell for region A

        Keyword options:
          - core_dist: float, core-to-boundary(A) distance threshold in unit-square coords (default 0.05)
          - hatch: str (default '///')
          - extent: (xmin, xmax, ymin, ymax) in data coords (default: current view)
          - radius: passed to voronoi_finite_polygons_2d
          - frame_strip: float, exclude hatch within this strip near axes frame (unit-square, default 0.0)
          - All standard `fill` kwargs are inherited (facecolor/edgecolor/linewidth/alpha/...) via adapter defaults.
        """
        import numpy as _np

        x = kwargs.pop("x")
        y = kwargs.pop("y")
        where = kwargs.pop("where", None)

        core_dist = 0.025
        frame_strip = 0.0

        kw_all = self._merge("fill", kwargs)
        from .helper import split_fill_kwargs
        kw_edge, kw_face, kw_rest = split_fill_kwargs(kw_all)
        kw_edge.update(kw_rest)
        kw_face.update(kw_rest)
        # ---- inputs ----
        x = _np.asarray(x)
        y = _np.asarray(y)
        if x.shape != y.shape:
            raise ValueError("voronoif: x and y must have the same shape")
        if x.size == 0:
            return []

        if where is None:
            where = _np.ones_like(x, dtype=bool)

        try:
            from scipy.spatial import Voronoi
        except Exception as e:
            raise ImportError("voronoif requires scipy.spatial.Voronoi. Please install scipy.") from e

        try:
            from shapely.geometry import Polygon as _SHPPolygon, Point as _SHPPoint, box as _SHPBox
            from shapely.ops import unary_union as _shp_unary_union
        except Exception as e:
            raise ImportError("voronoif requires shapely. Please install shapely.") from e

        # ---- derive view box & transforms from axes ----
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        extent = (min(xlim), max(xlim), min(ylim), max(ylim))
        xmin, xmax, ymin, ymax = extent

        t_data_to_disp = self.ax.transData

        disp_ll = t_data_to_disp.transform((xmin, ymin))
        disp_ur = t_data_to_disp.transform((xmax, ymax))
        disp_x0, disp_y0 = disp_ll
        disp_x1, disp_y1 = disp_ur

        # robust ordering / non-zero spans
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

        # ---- sites in axes-intrinsic unit-square coords ----
        pts_disp = t_data_to_disp.transform(_np.c_[x, y])
        pts_norm = _np.c_[
            (pts_disp[:, 0] - disp_x0) / disp_w,
            (pts_disp[:, 1] - disp_y0) / disp_h,
        ]
        if not _np.all(_np.isfinite(pts_norm)):
            pts_norm = _np.nan_to_num(pts_norm, nan=0.5, posinf=1.0, neginf=0.0)

        vor = Voronoi(pts_norm)
        regions, vertices = voronoi_finite_polygons_2d(vor)
        unit_rect = (0.0, 1.0, 0.0, 1.0)

        # ---- collect A: where-selected cell polygons (unit-square) ----
        polys_A_unit = []
        idx_A = []
        poly_by_idx = {}
        for i_pt, region in enumerate(regions):
            if (not where[i_pt]) or (not region):
                continue
            poly = vertices[region]
            poly = [(float(px), float(py)) for px, py in poly]
            poly = _clip_poly_to_rect(poly, unit_rect)
            if len(poly) < 3:
                continue
            try:
                g = _SHPPolygon(poly)
                if not g.is_valid:
                    g = g.buffer(0)
                if g.is_empty or g.area <= 0:
                    continue
                polys_A_unit.append(g)
                idx_A.append(i_pt)
                poly_by_idx[i_pt] = g
            except Exception:
                continue

        if not polys_A_unit:
            return []

        A = _shp_unary_union(polys_A_unit)
        if A.is_empty:
            return []

        # ---- filter cells in A by core distance to boundary(A) ----
        B_polys = []
        bnd = A.boundary

        # Exclude boundary segments that coincide with the axes frame (unit-square edges).
        # We do this by intersecting with a slightly shrunken unit box, removing edges at u/v=0/1.
        eps = 1.e-9
        inner = _SHPBox(eps, eps, 1.0 - eps, 1.0 - eps)
        bnd = bnd.intersection(inner)
        # bnd: shapely LineString/MultiLineString in unit-square coords
        if not (bnd is None or bnd.is_empty):
            lines = []
            gt = bnd.geom_type
            if gt == "LineString":
                lines = [bnd]
            elif gt == "MultiLineString":
                lines = list(bnd.geoms)
            elif gt == "GeometryCollection":
                for g in bnd.geoms:
                    if g.geom_type == "LineString":
                        lines.append(g)
                    elif g.geom_type == "MultiLineString":
                        lines.extend(list(g.geoms))
            for line in lines: 
                lind = line.buffer(0.001, cap_style=2, join_style=2)
                xs, ys = lind.exterior.coords.xy 
                self.ax.fill(list(xs), list(ys), **kw_edge, transform=self.ax.transAxes)

        
        for i_pt in idx_A:
            u, v = float(pts_norm[i_pt, 0]), float(pts_norm[i_pt, 1])
            try:
                d = _SHPPoint(u, v).distance(bnd)
            except Exception:
                continue
            if d < core_dist:
                B_polys.append(poly_by_idx[i_pt])

        if not B_polys:
            return []

        B = _shp_unary_union(B_polys)
        if B.is_empty:
            return []

        # Inherit matplotlib fill kwargs (plus any defaults for 'fill')
        kw_face = self._merge("fill", kw_face)
        kw_face['linewidth'] = 0.

        artists = []
        if B.geom_type == "Polygon":
            parts = [B]
        else:
            parts = list(getattr(B, "geoms", []))

        for g in parts:
            if g.is_empty:
                continue
            xs, ys = g.exterior.coords.xy
            artists.extend(self.ax.fill(list(xs), list(ys), **kw_face, transform=self.ax.transAxes))

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
        palette = self.config['hist']['color']
        return [palette[i % len(palette)] for i in range(n)]

# —— Ternary 适配器：在 Std 基础上增加 (a,b,c)->(x,y) 投影 ——
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

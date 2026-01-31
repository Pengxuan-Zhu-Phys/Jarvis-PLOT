# jarvisplot/Figure/helper.py
#!/usr/bin/env python3

from matplotlib.axes import Axes
import numpy as np

def split_fill_kwargs(kw_all):
    edge_kws = {
        "edgecolor", "ec",
        "linewidth", "lw",
        "linestyle", "ls",
        "joinstyle", "capstyle",
        "alpha"
    }
    face_kws = {
        "facecolor", "fc",
        "hatch",
        "hatch_linewidth",
        "alpha", "edgecolor"
    }
    kw_edge = {}
    kw_face = {}
    kw_rest = {}
    for k, v in kw_all.items():
        if k in edge_kws:
            kw_edge[k] = v
        if k in face_kws:
            kw_face[k] = v
        else:
            kw_rest[k] = v
    return kw_edge, kw_face, kw_rest

def plot_shapely_boundary(ax, geom, *, transform=None, **plot_kw):
    if geom is None or geom.is_empty:
        return []

    lines = []
    gt = geom.geom_type
    if gt == "LineString":
        lines = [geom]
    elif gt == "MultiLineString":
        lines = list(geom.geoms)
    elif gt == "GeometryCollection":
        for g in geom.geoms:
            if g.geom_type == "LineString":
                lines.append(g)
            elif g.geom_type == "MultiLineString":
                lines.extend(list(g.geoms))
    else:
        # 兜底：有时 boundary 可能给出别的类型
        try:
            b = geom.boundary
            return _plot_shapely_boundary(ax, b, transform=transform, **plot_kw)
        except Exception:
            return []

    artists = []
    for ln in lines:
        xs, ys = ln.coords.xy
        artists += ax.plot(list(xs), list(ys), transform=transform, **plot_kw)
    return artists


# helper: convert infinite regions to finite polygons (public-domain recipe)
# Clip polygon to rectangle extent (Sutherland-Hodgman for convex quad)
def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points).max() * 2
    # Construct a map ridge_vertices -> points (pairs)
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    # Reconstruct infinite regions
    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if -1 not in vertices:
            new_regions.append(vertices)
            continue
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v != -1]
        for p2, v1, v2 in ridges:
            if v2 < 0: v1, v2 = v2, v1
            if v1 >= 0 and v2 >= 0:
                continue
            # Compute the missing endpoint at infinity
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        # Sort region vertices counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)].tolist()
        new_regions.append(new_region)
    return new_regions, np.asarray(new_vertices)


def _clip_poly_to_rect(poly, rect):
    # rect: (xmin, xmax, ymin, ymax)
    xmin, xmax, ymin, ymax = rect
    def _clip(edges, inside, intersect):
        out = []
        if not edges:
            return out
        S = edges[-1]
        for E in edges:
            if inside(E):
                if inside(S):
                    out.append(E)
                else:
                    out.append(intersect(S, E))
                    out.append(E)
            elif inside(S):
                out.append(intersect(S, E))
            S = E
        return out
    def clip_left(P):
        return _clip(P, lambda p: p[0] >= xmin, lambda s ,e: (xmin, s[1] + (e[1 ] -s[1] ) *(xmin - s[0] ) /(e[0 ] -s[0]) ))
    def clip_right(P):
        return _clip(P, lambda p: p[0] <= xmax, lambda s ,e: (xmax, s[1] + (e[1 ] -s[1] ) *(xmax - s[0] ) /(e[0 ] -s[0]) ))
    def clip_bottom(P):
        return _clip(P, lambda p: p[1] >= ymin, lambda s ,e: (s[0] + (e[0 ] -s[0] ) *(ymin - s[1] ) /(e[1 ] -s[1]), ymin ))
    def clip_top(P):
        return _clip(P, lambda p: p[1] <= ymax, lambda s ,e: (s[0] + (e[0 ] -s[0] ) *(ymax - s[1] ) /(e[1 ] -s[1]), ymax ))
    P = poly
    for fn in (clip_left, clip_right, clip_bottom, clip_top):
        P = fn(P)
        if not P:
            break
    return P



# ---- helpers for masking by extend on filled contours ----

def _resolve_vlim(z, vmin=None, vmax=None, levels=None, norm=None):
    """Derive effective vmin/vmax from norm/levels/fallback to data."""
    import numpy as np
    if norm is not None:
        vmin = getattr(norm, "vmin", vmin)
        vmax = getattr(norm, "vmax", vmax)
    if levels is not None and np.ndim(levels) > 0:
        vmin = levels[0] if vmin is None else vmin
        vmax = levels[-1] if vmax is None else vmax
    if vmin is None:
        vmin = float(np.nanmin(z))
    if vmax is None:
        vmax = float(np.nanmax(z))
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
    import numpy as np
    z = np.asarray(z)
    e = (extend or "neither").lower()
    if e not in ("neither", "min", "max", "both"):
        e = "neither"
    vmin_eff, vmax_eff = _resolve_vlim(z, vmin=vmin, vmax=vmax, levels=levels, norm=norm)
    mask = np.zeros_like(z, dtype=bool)
    if e in ("min", "both"):
        mask |= (z < vmin_eff)
    if e in ("max", "both"):
        mask |= (z > vmax_eff)
    return np.ma.masked_array(z, mask=mask), vmin_eff, vmax_eff

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

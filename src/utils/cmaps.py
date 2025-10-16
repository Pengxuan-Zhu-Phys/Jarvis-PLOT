#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JarvisPLOT colormap loader: JSON-only, no built-ins.
Public API:
  - setup(json_path: str|None = None, force: bool = True) -> dict
  - register_from_json(json_path: str|os.PathLike, force: bool = True) -> dict
  - list_available() -> list[str]
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import os

import matplotlib as mpl
import matplotlib.cm as mcm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Colormap

import inspect
import matplotlib.pyplot as plt
def _register(name: str, cmap: Colormap, force: bool) -> bool:
    """
    Try multiple registration paths depending on Matplotlib version:
      1) mpl.colormaps.register(...), passing 'name' and 'override' only if supported
      2) plt.register_cmap(name=..., cmap=...)
      3) update legacy cm.cmap_d if present
    Returns True on success.
    """
    name = str(name)
    # Path 1: modern registry
    try:
        reg = getattr(mpl.colormaps, "register", None)
        if reg is not None:
            kwargs = {}
            try:
                sig = inspect.signature(reg)
                if "name" in sig.parameters:
                    kwargs["name"] = name
                else:
                    try:
                        cmap.name = name
                    except Exception:
                        pass
                if "override" in sig.parameters:
                    kwargs["override"] = bool(force)
            except Exception:
                # If signature introspection fails, try safest call (cmap only).
                try:
                    cmap.name = name
                except Exception:
                    pass
            # Call register
            if kwargs:
                reg(cmap, **kwargs)
            else:
                reg(cmap)
            ok = True
        else:
            ok = False
    except Exception:
        ok = False
    # Path 2: pyplot register_cmap
    if not ok:
        try:
            plt.register_cmap(name=name, cmap=cmap)
            ok = True
        except Exception:
            ok = False
    # Path 3: legacy dict
    if not ok and hasattr(mcm, "cmap_d"):
        try:
            mcm.cmap_d[name] = cmap
            ok = True
        except Exception:
            ok = False
    # final visibility sanity check
    try:
        visible = (name in list(mpl.colormaps)) or (hasattr(mcm, "cmap_d") and name in mcm.cmap_d)
    except Exception:
        visible = False
    return bool(ok and visible)

class CmapSpecError(Exception):
    pass

def _norm_color_list(seq):
    """
    Normalize a list of colors into formats Matplotlib accepts.
    Accepts:
      - hex strings "#RRGGBB" / "#RRGGBBAA"
      - RGB/RGBA tuples or lists in 0..1 or 0..255
    Returns a new list; raises CmapSpecError on invalid entries.
    """
    out = []
    for c in list(seq):
        if isinstance(c, str):
            out.append(c)
            continue
        if isinstance(c, (list, tuple)):
            vals = list(c)
            if not vals:
                raise CmapSpecError("empty color tuple")
            if all(isinstance(v, (int, float)) for v in vals):
                # If any component >1, assume 0..255 and scale
                mx = max(abs(float(v)) for v in vals)
                if mx > 1.0:
                    vals = [float(v)/255.0 for v in vals]
                out.append(tuple(vals))
                continue
        raise CmapSpecError(f"unsupported color value: {c!r}")
    return out

JsonLike = Union[Dict[str, Any], List[Dict[str, Any]]]

def _read_json(path: Path) -> Optional[JsonLike]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _to_linear(name: str, colors: List[Any]) -> Optional[LinearSegmentedColormap]:
    try:
        # Accept either ["#hex", ...] or [[pos, color], ...]
        if colors and isinstance(colors[0], (list, tuple)) and len(colors[0]) == 2 and not isinstance(colors[0][1], (list, tuple)) and not (isinstance(colors[0][1], str) and colors[0][1].startswith("#") or isinstance(colors[0][1], str)):
            # The above heuristic is too brittle; simplify by explicit shape check below
            pass
    except Exception:
        pass
    try:
        if colors and isinstance(colors[0], (list, tuple)) and len(colors[0]) == 2:
            pts: List[Tuple[float, Any]] = []
            for p, c in colors:
                try:
                    pp = float(p)
                except Exception:
                    continue
                pp = max(0.0, min(1.0, pp))
                pts.append((pp, c))
            pts.sort(key=lambda t: t[0])
            # Normalize color payload
            pts = [(p, _norm_color_list([c])[0]) for (p, c) in pts]
            return LinearSegmentedColormap.from_list(name, pts)
        # Plain list of colors
        return LinearSegmentedColormap.from_list(name, _norm_color_list(list(colors)))
    except Exception:
        return None

def _to_listed(name: str, colors: List[Any]) -> Optional[ListedColormap]:
    try:
        return ListedColormap(_norm_color_list(list(colors)), name=name)
    except Exception:
        return None

def _build(spec: Dict[str, Any]) -> Colormap:
    name = spec.get("name")
    colors = spec.get("colors")
    if not isinstance(name, str) or not name.strip():
        raise CmapSpecError("missing or invalid 'name'")
    if not isinstance(colors, (list, tuple)) or not colors:
        raise CmapSpecError(f"cmap {name!r}: missing or invalid 'colors'")
    t = str(spec.get("type", "listed")).lower()
    if t in ("linear", "segmented", "continuous"):
        cm = _to_linear(name, list(colors))
    else:
        cm = _to_listed(name, list(colors))
    if cm is None:
        raise CmapSpecError(f"failed to build colormap {name!r}")
    return cm

def register_from_json(json_path: Union[str, os.PathLike], force: bool = True) -> Dict[str, Any]:
    p = Path(json_path).expanduser().resolve()
    reg: List[str] = []
    fail: List[str] = []
    err: Dict[str, str] = {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"registered": [], "failed": [], "errors": {str(p): f"read json failed: {e}"}, "path": str(p)}
    # Accept three shapes: dict with 'colormaps', single dict, or top-level list
    specs: List[Dict[str, Any]] = []
    aliases: Dict[str, str] = {}
    if isinstance(data, dict):
        if "name" in data and "colors" in data:
            specs.append(data)
        else:
            cmaps_list = data.get("colormaps")
            if isinstance(cmaps_list, list):
                for item in cmaps_list:
                    if isinstance(item, dict):
                        specs.append(item)
            aliases = data.get("aliases", {}) or {}
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                specs.append(item)
    # Build + register
    for spec in specs:
        try:
            name = spec.get("name")
            colors = spec.get("colors")
            ctype = str(spec.get("type", "listed")).lower()
            if not isinstance(name, str) or not name or not isinstance(colors, (list, tuple)) or not colors:
                raise ValueError("invalid spec (need 'name' and non-empty 'colors')")
            if ctype in ("linear", "segmented", "continuous"):
                cm = _to_linear(name, list(colors))
            else:
                cm = _to_listed(name, list(colors))
            if cm is None:
                raise ValueError("failed to build colormap")
            if _register(name, cm, force=force):
                reg.append(name)
                try:
                    cm_r = cm.reversed()
                    if _register(f"{name}_r", cm_r, force=force):
                        reg.append(f"{name}_r")
                except Exception:
                    pass
            else:
                fail.append(name)
        except Exception as e:
            nm = spec.get("name", "<missing-name>")
            fail.append(nm)
            err[nm] = str(e)
    # aliases (if any): only when target is visible
    for alias, target in (aliases.items() if isinstance(aliases, dict) else []):
        try:
            base = mpl.colormaps.get(target)
        except Exception:
            base = None
        if base is None and hasattr(mcm, "cmap_d"):
            base = mcm.cmap_d.get(target)
        if base is None:
            fail.append(alias)
            err[alias] = f"alias target not found: {target}"
        elif _register(alias, base, force=force):
            reg.append(alias)
        else:
            fail.append(alias)
    return {"registered": reg, "failed": fail, "errors": err, "path": str(p)}

def setup(json_path: Optional[Union[str, os.PathLike]] = None, force: bool = True) -> Dict[str, Any]:
    src = json_path
    if not src:
        return {"registered": [], "failed": [], "errors": {}, "path": None}
    return register_from_json(src, force=force)

def list_available() -> List[str]:
    try:
        return list(mpl.colormaps)
    except Exception:
        if hasattr(mcm, "cmap_d"):
            return sorted(mcm.cmap_d.keys())
        return []

__all__ = ["setup", "register_from_json", "list_available"]

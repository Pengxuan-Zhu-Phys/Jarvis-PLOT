#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict


def load_cmaps(load_path: Callable[[str], str], logger=None) -> None:
    """Load and register JarvisPLOT colormaps from the internal JSON bundle."""
    try:
        from .utils import cmaps

        json_path = "&JP/jarvisplot/cards/colors/colormaps.json"
        cmap_summary = cmaps.setup(load_path(json_path), force=True)
        if logger:
            logger.debug(f"JarvisPLOT: colormaps registered: {cmap_summary}")
            try:
                logger.debug(f"JarvisPLOT: available colormaps sample: {cmaps.list_available()}")
            except Exception:
                pass
    except Exception as e:
        if logger:
            logger.warning(f"JarvisPLOT: failed to initialize colormaps: {e}")


def load_interpolators(config: Dict[str, Any], yaml_dir, shared=None, logger=None) -> None:
    """Parse YAML interpolator specs and register them for lazy use in expressions."""
    from .inner_func import clear_external_funcs

    clear_external_funcs()
    cfg = config.get("Functions", None) if isinstance(config, dict) else None
    if cfg is not None:
        from .inner_func import set_external_funcs_getter
        from .utils.interpolator import InterpolatorManager

        mgr = InterpolatorManager.from_yaml(
            cfg,
            yaml_dir=yaml_dir,
            shared=shared,
            logger=logger,
        )
        set_external_funcs_getter(lambda: (mgr.as_eval_funcs() or {}))
        if logger:
            logger.debug(f"JarvisPLOT: Functions registered: {mgr.summary()}")


def load_styles(load_path: Callable[[str], str], logger=None) -> Dict[str, Dict[str, Any]]:
    """Load the internal style preference and bundle files."""
    spp = "&JP/jarvisplot/cards/style_preference.json"
    style: Dict[str, Dict[str, Any]] = {}
    if logger:
        logger.debug("Loading internal Format set -> {}".format(load_path(spp)))
    with open(load_path(spp), "r") as f1:
        stl = json.load(f1)
        for sty, boudle in stl.items():
            style[sty] = {}
            for kk, vv in boudle.items():
                vpath = load_path(vv)
                if vpath and os.path.exists(vpath):
                    if logger:
                        logger.debug("Loading '{}' boudle, {} Style \n\t-> {}".format(sty, kk, vpath))
                    with open(vpath, "r") as f2:
                        style[sty][kk] = json.load(f2)
                else:
                    if logger:
                        logger.error("Style Not Found: '{}' boudle, {} Style \n\t-> {}".format(sty, kk, vpath))
    return style

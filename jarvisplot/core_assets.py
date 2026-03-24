#!/usr/bin/env python3
from __future__ import annotations

import json
import os


def load_cmaps(core) -> None:
    """Load and register JarvisPLOT colormaps from the internal JSON bundle."""
    try:
        from .utils import cmaps

        json_path = "&JP/jarvisplot/cards/colors/colormaps.json"
        cmap_summary = cmaps.setup(core.load_path(json_path), force=True)
        if core.logger:
            core.logger.debug(f"JarvisPLOT: colormaps registered: {cmap_summary}")
            try:
                core.logger.debug(f"JarvisPLOT: available colormaps sample: {cmaps.list_available()}")
            except Exception:
                pass
    except Exception as e:
        if core.logger:
            core.logger.warning(f"JarvisPLOT: failed to initialize colormaps: {e}")


def load_interpolators(core) -> None:
    """Parse YAML interpolator specs and register them for lazy use in expressions."""
    cfg = core.yaml.config.get("Functions", None)
    if cfg is not None:
        from .inner_func import set_external_funcs_getter
        from .utils.interpolator import InterpolatorManager

        mgr = InterpolatorManager.from_yaml(
            cfg,
            yaml_dir=core.yaml.dir,
            shared=core.shared,
            logger=core.logger,
        )
        core.interpolators = mgr
        set_external_funcs_getter(lambda: (mgr.as_eval_funcs() or {}))
        if core.interpolators and core.logger:
            core.logger.debug(f"JarvisPLOT: Functions registered: {mgr.summary()}")


def load_styles(core) -> None:
    """Load the internal style preference and bundle files."""
    spp = "&JP/jarvisplot/cards/style_preference.json"
    if core.logger:
        core.logger.debug("Loading internal Format set -> {}".format(core.load_path(spp)))
    with open(core.load_path(spp), "r") as f1:
        stl = json.load(f1)
        for sty, boudle in stl.items():
            core.style[sty] = {}
            for kk, vv in boudle.items():
                vpath = core.load_path(vv)
                if vpath and os.path.exists(vpath):
                    if core.logger:
                        core.logger.debug("Loading '{}' boudle, {} Style \n\t-> {}".format(sty, kk, vpath))
                    with open(vpath, "r") as f2:
                        core.style[sty][kk] = json.load(f2)
                else:
                    if core.logger:
                        core.logger.error("Style Not Found: '{}' boudle, {} Style \n\t-> {}".format(sty, kk, vpath))

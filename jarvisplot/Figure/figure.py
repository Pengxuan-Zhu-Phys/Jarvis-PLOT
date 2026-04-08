#!/usr/bin/env python3
from copy import deepcopy
import gc
from typing import Optional, Mapping
import numpy as np 
import os, sys 
import pandas as pd 
import matplotlib as mpl
from types import MethodType

from .adapters import StdAxesAdapter, TernaryAxesAdapter
from .method_registry import resolve_callable
from .style_runtime import resolve_style_bundle
from .config_runtime import apply_figure_config
from .layer_runtime import (
    load_layer_data as runtime_load_layer_data,
    load_bool_df as runtime_load_bool_df,
    load_layer_runtime_data as runtime_load_layer_runtime_data,
    release_layer_runtime_data as runtime_release_layer_runtime_data,
    render_layer as runtime_render_layer,
)
from .layout_runtime import (
    apply_axis_endpoints,
    apply_auto_ticks,
    apply_manual_ticks,
    ensure_numbered_rect_axes,
    has_manual_ticks,
    is_numbered_ax,
)
from .colorbar_runtime import (
    axc_color_config,
    axc_is_horizontal,
    collect_layer_color_range,
    layer_uses_color,
    precompute_colorbar_cb,
)
from ..utils.expression import eval_dataframe_expression
from ..utils.pathing import resolve_project_path
from ..memtrace import memtrace_checkpoint

import json
import time

try:
    import polars as pl
except Exception:
    pl = None


class Figure:
    def _is_numbered_ax(self, name: str) -> bool:
        """Return True iff name matches ax<NUMBER>, e.g. ax1, ax2, ax10."""
        return is_numbered_ax(name)

    def _ensure_numbered_rect_axes(self, ax_name: str, kwgs: dict):
        """Create/configure a numbered rectangular axes (ax1, ax2, ...) using layout helpers."""
        return ensure_numbered_rect_axes(self, ax_name, kwgs)
    def _has_manual_ticks(self, ax_key: str, which: str) -> bool:
        """Return True if YAML provides manual tick positions for given axis."""
        return has_manual_ticks(self.frame, ax_key, which)

    def _axc_is_horizontal(self) -> bool:
        return axc_is_horizontal(self.frame)

    def _axc_color_config(self) -> dict:
        """Return normalized frame.axc.color config with legacy fallbacks."""
        return axc_color_config(self.frame)

    def _apply_axis_endpoints(self, ax_obj, axis_cfg: dict, which: str):
        """
        which: 'x' or 'y'
        axis_cfg: self.frame['ax'].get('xaxis', {}) / 'yaxis'
        """
        return apply_axis_endpoints(self, ax_obj, axis_cfg, which)
                
    def _apply_auto_ticks(self, ax_obj, which: str):
        """Lightweight auto-tick post-processing at finalize stage.

        Goals:
        - Never print/debug here.
        - X: rotate long labels a bit.
        - Y: if log-like scale -> keep log spacing and use compact decimals for decades in [1e-2, 1e2],
             otherwise defer to Matplotlib's LogFormatter.
             if linear scale -> use ScalarFormatter with sci notation, but never touch log formatters.
        """
        return apply_auto_ticks(ax_obj, which)
        
    def __init__(self, info: Optional[Mapping] = None):
        self.t0 = time.perf_counter()

        # internal state
        self._name: Optional[str]       = None
        self._jpstyles: Optional[dict]  = None
        self._style: Optional[dict]     = {}
        self.print      = False
        self.mode       = "Jarvis"
        # self._jpdatas:  Optional[list]  =   []
        self._logger    = None
        self._frame     = {}
        self._outinfo   = {}
        self._yaml_dir  = None  # directory of the active YAML file (used to resolve relative paths)
        self.axes       = {}
        self.debug      = False
        # self._axtri     = None
        self._layers    = {}
        self._render_queue = []
        self._ctx       = None
        self._preprocessor = None
        # allow optional initialization from a dict
        if info:
            self.from_dict(info)

    # --- name property ---
    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        """Set figure name as a string (or None)."""
        if value is None:
            self._name = None
            return
        if not isinstance(value, str):
            raise TypeError("Figure.name must be a string or None")
        self._name = value

    @property
    def config(self):   
        return None 
    
    @config.setter
    def config(self, infos):
        self.dir = self.load_path(infos['output'].get("dir", "."), base_dir=self._yaml_dir)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.fmts   = infos['output'].get("formats", ['png'])
        self.dpi    = infos['output'].get('dpi', 600)

    @property
    def jpstyles(self) -> Optional[dict]: 
        return self._jpstyles 
    
    @jpstyles.setter
    def jpstyles(self, value) -> None:             
        self._jpstyles = value
        
    @property
    def logger(self):
        return self._logger 
    
    @logger.setter
    def logger(self, value):
        if value is not None: 
            self._logger = value 

    @property
    def frame(self):
        return self._frame 
    
    @frame.setter
    def frame(self, value) -> None: 
        if self._frame is None:
            self._frame = value
        else:
            from deepmerge import always_merger
            self._frame = always_merger.merge(self._frame, value)
            
    @property
    def style(self) -> Optional[dict]: 
        return self._style 
    
    
    @style.setter
    def style(self, value) -> None: 
        if len(value) not in (1, 2):
            self.logger.error("Undefined style -> {}".format(value))
            raise TypeError
        family, selected, frame, style = resolve_style_bundle(self.jpstyles, value)
        self._frame = frame
        self._style = style
        if self.logger:
            self.logger.debug("Style: [{} : {}] used for figure -> {}".format(family, selected, self.name))
    
    @property
    def context(self):
        return self._ctx

    @context.setter
    def context(self, value):
        self._ctx = value  # 期望是 DataContext

    @property
    def preprocessor(self):
        return self._preprocessor

    @preprocessor.setter
    def preprocessor(self, value):
        self._preprocessor = value


    
    
    @property
    def layers(self): 
        return self._layers 
    
    @layers.setter
    def layers(self, infos):
        self._render_queue = []
        for layer in infos: 
            info = {}
            ax  = self.axes[layer['axes']]
            info['name'] = layer.get("name", "")
            info['data'] = None
            info['data_loaded'] = False
            info['layer_spec'] = layer
            info['source_refs'] = self._source_refs_from_layer(layer)
            info['share_name'] = layer.get("share_data")
            info['combine'] = layer.get("combine", "concat")
            info['coor'] = layer['coordinates']
            info['method'] = layer.get("method", "scatter")
            info['style'] = layer.get("style", {})
            info['colorbar'] = layer.get("colorbar", "axc")
            ax.layers.append(info)
            self._render_queue.append((ax, info))
            self.logger.debug("Successfully loaded layer -> {}".format(info["name"]))

    @staticmethod
    def _is_polars_frame(obj) -> bool:
        if pl is None:
            return False
        return isinstance(obj, (pl.DataFrame, pl.LazyFrame))

    def _polars_to_pandas_compat(self, data, reason: str = "render"):
        if pl is None:
            return data
        if isinstance(data, pl.LazyFrame):
            memtrace_checkpoint(self.logger, "figure.polars_collect.before", data, extra={"reason": reason})
            self.logger.debug(f"Collecting polars LazyFrame for pandas step -> {reason}")
            data = data.collect()
            memtrace_checkpoint(self.logger, "figure.polars_collect.after", data, extra={"reason": reason})
        if isinstance(data, pl.DataFrame):
            memtrace_checkpoint(self.logger, "figure.pandas_convert.before", data, extra={"reason": reason})
            self.logger.debug(f"Materializing polars DataFrame for pandas step -> {reason}")
            try:
                out = data.to_pandas()
            except ModuleNotFoundError:
                self.logger.warning(
                    f"pyarrow unavailable during polars->pandas conversion; using dict fallback -> {reason}"
                )
                out = pd.DataFrame(data.to_dict(as_series=False))
            memtrace_checkpoint(self.logger, "figure.pandas_convert.after", out, extra={"reason": reason})
            return out
        return data

    def _source_refs_from_layer(self, layer) -> list[str]:
        refs: list[str] = []
        entries = layer.get("data", [])
        if not isinstance(entries, list):
            return refs
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            source = entry.get("source")
            if isinstance(source, str):
                refs.append(source)
            elif isinstance(source, (list, tuple)):
                refs.extend(str(item) for item in source if isinstance(item, str))
        return refs

    def _ensure_pandas_data(self, data, reason: str = "render"):
        if pl is None:
            return data
        if isinstance(data, (pl.LazyFrame, pl.DataFrame)):
            return self._polars_to_pandas_compat(data, reason=reason)
        if isinstance(data, dict):
            return {kk: self._ensure_pandas_data(vv, reason=reason) for kk, vv in data.items()}
        return data

    def _concat_loaded_data(self, items):
        if len(items) == 0:
            return None
        if len(items) == 1:
            return items[0]
        if pl is not None and all(self._is_polars_frame(item) for item in items):
            lazy_items = [item if isinstance(item, pl.LazyFrame) else item.lazy() for item in items]
            return pl.concat(lazy_items, how="vertical_relaxed")
        pandas_items = [self._ensure_pandas_data(item, reason="concat-layer") for item in items]
        try:
            return pd.concat(pandas_items, ignore_index=False)
        except Exception:
            return pandas_items[0]

    def _store_share_data_if_needed(self, layer, data, cache_ref: str | None = None):
        share_name = layer.get("share_data")
        if not share_name or data is None or self.context is None:
            return

        registered = False
        if self.preprocessor is not None:
            try:
                self.preprocessor.persist_named_layer(share_name, layer, data, cache_ref=cache_ref)
                registered = bool(self.preprocessor.register_named_layer(share_name, layer))
            except Exception as e:
                self.logger.debug(f"persist share_data cache failed: {e}")

        if self.context.remaining_uses(share_name) <= 0:
            return
        if not registered:
            self.context.update(share_name, data)

    def _load_layer_runtime_data(self, layer_info):
        return runtime_load_layer_runtime_data(self, layer_info)

    def _release_layer_runtime_data(self, layer_info, consume_sources: bool = True):
        return runtime_release_layer_runtime_data(self, layer_info, consume_sources=consume_sources)
                        
    def load_layer_data(self, layer):
        return runtime_load_layer_data(self, layer)
               
         
    def load_bool_df(self, df, transform):
        return runtime_load_bool_df(self, df, transform)
                         
         
            
    @property    
    def axlogo(self):
        return self.axes['axlogo']
    
    @axlogo.setter
    def axlogo(self, kwgs):
        if "axlogo" not in self.axes.keys():
            axtp = self.fig.add_axes(**kwgs)
            axtp.set_zorder(200)
            axtp.patch.set_alpha(0) 
            
            self.axes['axlogo'] = axtp
            self.axes['axlogo'].needs_finalize = False
            self.axes['axlogo'].status = 'finalized'

        self.axlogo.layers  = []
        jhlogo = self.load_path(self.frame['axlogo']['file'])
        from PIL import Image
        with Image.open(jhlogo) as image:
            arr = np.asarray(image.convert("RGBA"))
            self.axlogo.imshow(arr)
            if self.frame['axlogo'].get("text"):
                for txt in self.frame['axlogo']['text']:
                    self.axlogo.text(**txt,  transform=self.axlogo.transAxes)
            # else: 
                # self.axlogo.text(1., 0., "Jarvis-HEP", ha="left", va='bottom', color="black", fontfamily="Fira code", fontsize="x-small", fontstyle="normal", fontweight="bold", transform=self.axlogo.transAxes)
                # self.axlogo.text(1., 0.9, "  Powered by", ha="left", va='top', color="black", fontfamily="Fira code", fontsize="xx-small", fontstyle="normal", fontweight="normal", transform=self.axlogo.transAxes)

    @property
    def axtri(self):
        return self.axes['axtri']
    
    @axtri.setter
    def axtri(self, kwgs):
        if "axtri" not in self.axes.keys():
            facecolor = kwgs.pop("facecolor", None)
            raw_ax = self.fig.add_axes(**kwgs) 
            # Booking Ternary Plot Clip_path 
            from matplotlib.path import Path
            vertices = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0), (0.0, 0.0)]
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
            raw_ax._clip_path = Path(vertices, codes)
            self._install_tri_auto_clip(raw_ax)

            # Keep rect axes patch transparent; ternary background is handled by adapter.
            raw_ax.patch.set_alpha(0)

            adapter = TernaryAxesAdapter(
                raw_ax,
                defaults={"facecolor": facecolor} if facecolor is not None else None,
                clip_path=Path(vertices, codes)  # 用 path 做 clip，transform 使用 ax.transData 已在适配器里处理
            )
            adapter._type = 'tri'
            adapter._legend = False
            adapter.layers = []
            adapter.status = 'configured'
            self.axes["axtri"] = adapter
        
        self.axtri.plot(
            x=[0.5, 1.0, 0.5, 0.0, 0.5], 
            y=[0.0, 0.0, 1.0, 0.0, 0.0],
            **self.frame['axtri']['frame'])

        arr = np.arange(0., 1.0, self.frame['axtri']['grid']['sep'])  
        seps = np.empty(len(arr) * 2 - 1)
        seps[0::2] = arr
        seps[1::2] = np.nan
        
        x0 = seps 
        x1 = 0.5 * seps 
        x2 = 0.5 + 0.5 * seps
        x3 = 1.0 - 0.5 * seps 
                
        y0 = 0.0 * seps 
        y1 = seps 
        y2 = 1 - seps
        
        # Major ticks 
        arrt = np.arange(0., 1.0001, self.frame['axtri']['ticks']['majorsep'])
        sept = np.empty(len(arrt) * 2 - 1)
        sept[0::2] = arrt
        sept[1::2] = np.nan 
        matl = self.frame['axtri']['ticks']['majorlength']
        
        txb0 = sept 
        txb1 = sept - 0.5 * matl 
        tyb0 = 0.0 * sept 
        tyb1 = 0.0 * sept  - matl 
        
        txl0 = 0.5 * sept 
        txl1 = 0.5 * sept - 0.5 * matl 
        tyl0 = sept 
        tyl1 = sept + matl 
        
        txr0 = 1.0 - 0.5 * sept 
        txr1 = 1.0 - 0.5 * sept + matl
        tyr0 = sept 
        tyr1 = sept

        
        # Minor ticks
        arrm = np.arange(0., 1.0001, self.frame['axtri']['ticks']['minorsep'])
        sepm = np.empty(len(arrm) * 2 - 1)
        sepm[0::2] = arrm
        sepm[1::2] = np.nan 
        matm = self.frame['axtri']['ticks']['minorlength']
        
        mxb0 = sepm 
        mxb1 = sepm - 0.5 * matm 
        myb0 = 0.0 * sepm 
        myb1 = 0.0 * sepm  - matm 
        
        mxl0 = 0.5 * sepm 
        mxl1 = 0.5 * sepm - 0.5 * matm 
        myl0 = sepm 
        myl1 = sepm + matm 
        
        mxr0 = 1.0 - 0.5 * sepm 
        mxr1 = 1.0 - 0.5 * sepm + matm
        myr0 = sepm 
        myr1 = sepm
        
        ticklabels = [f"{v*100:.0f}%" for v in arrt]
        # Ticks label positions 
        lbx  = arrt - 0.7 * matl
        lby  = 0.0 * arrt - 1.4 * matl 
        
        llx  = (1.0 - arrt) / 2.0 - 1.5 * matl
        lly  = 1 - arrt + 2.0 * matl 
        
        lrx  = 1.0 - arrt / 2.0 + 1.3 * matl 
        lry  = arrt
        
        # Bottom Axis 
        gridbx = np.array([val for pair in zip(x0, x1) for val in pair]) 
        gridby = np.array([val for pair in zip(y0, y1) for val in pair]) 
        tickbx = np.array([val for pair in zip(txb0, txb1) for val in pair]) 
        tickby = np.array([val for pair in zip(tyb0, tyb1) for val in pair]) 
        minorbx = np.array([val for pair in zip(mxb0, mxb1) for val in pair]) 
        minorby = np.array([val for pair in zip(myb0, myb1) for val in pair]) 

        # Left Axis 
        gridlx = np.array([val for pair in zip(x0, x2) for val in pair])
        gridly = np.array([val for pair in zip(y0, y2) for val in pair])
        ticklx = np.array([val for pair in zip(txl0, txl1) for val in pair]) 
        tickly = np.array([val for pair in zip(tyl0, tyl1) for val in pair]) 
        minorlx = np.array([val for pair in zip(mxl0, mxl1) for val in pair]) 
        minorly = np.array([val for pair in zip(myl0, myl1) for val in pair]) 

        # Right Axis 
        gridrx = np.array([val for pair in zip(x1, x3) for val in pair])
        gridry = np.array([val for pair in zip(y1, y1) for val in pair])
        tickrx = np.array([val for pair in zip(txr0, txr1) for val in pair]) 
        tickry = np.array([val for pair in zip(tyr0, tyr1) for val in pair]) 
        minorrx = np.array([val for pair in zip(mxr0, mxr1) for val in pair]) 
        minorry = np.array([val for pair in zip(myr0, myr1) for val in pair]) 

        # Grids // Major Ticks // Minor Ticks // Tick Lables 
        # Bottom Axis 

        self.axtri.plot(x=gridbx, y=gridby, **self.frame['axtri']['grid']['style'])
        self.axtri.plot(x=tickbx, y=tickby, **self.frame['axtri']['ticks']['majorstyle'])
        self.axtri.plot(x=minorbx, y=minorby, **self.frame['axtri']['ticks']['minorstyle'])
        self.axtri.text(s=self.frame['axtri']['labels']['bottom'], **self.frame['axtri']['labels']['bottomstyle'])
        for x, y, label in zip(lbx, lby, ticklabels):
            self.axtri.text(x, y, label, **self.frame['axtri']['ticks']['bottomticklables'])

        # Right Axis 
        self.axtri.plot(x=gridrx, y=gridry, **self.frame['axtri']['grid']['style'])
        self.axtri.plot(x=tickrx, y=tickry, **self.frame['axtri']['ticks']['majorstyle'])
        self.axtri.plot(x=minorrx, y=minorry, **self.frame['axtri']['ticks']['minorstyle'])
        self.axtri.text(s=self.frame['axtri']['labels']['right'], **self.frame['axtri']['labels']['rightstyle'])
        for x, y, label in zip(lrx, lry, ticklabels):
            self.axtri.text(x, y, label, **self.frame['axtri']['ticks']['rightticklables'])

        # Left Axis 
        self.axtri.plot(x=gridlx, y=gridly, **self.frame['axtri']['grid']['style'])
        self.axtri.plot(x=ticklx, y=tickly, **self.frame['axtri']['ticks']['majorstyle'])
        self.axtri.plot(x=minorlx, y=minorly, **self.frame['axtri']['ticks']['minorstyle'])
        self.axtri.text(s=self.frame['axtri']['labels']['left'], **self.frame['axtri']['labels']['leftstyle'])
        for x, y, label in zip(llx, lly, ticklabels):
            self.axtri.text(x, y, label, **self.frame['axtri']['ticks']['leftticklables'])

        if self.debug:
            self.axtri.scatter(x=lbx, y=lby, s=1.0, marker='.', c="#FF42A1", clip_on=False)
            self.axtri.scatter(x=lrx, y=lry, s=1.0, marker='.', c="#FF42A1", clip_on=False)
            self.axtri.scatter(x=llx, y=lly, s=1.0, marker='.', c="#FF42A1", clip_on=False)
            self.axtri.plot(x=[0., 0.75, 0.84], y=[0., 0.5, 0.56], marker=".", linestyle="-", lw=0.3, markersize=1, c="#FF42A1", clip_on=False)
            self.axtri.plot(x=[0.5, 0.5, 0.5], y=[1., 0.0, -0.12], marker=".", linestyle="-", lw=0.3, markersize=1, c="#FF42A1", clip_on=False)
            self.axtri.plot(x=[1., 0.25, 0.16], y=[0., 0.5, 0.56], marker=".", linestyle="-", lw=0.3, markersize=1, c="#FF42A1", clip_on=False)
       
    @property
    def axc(self):
        return self.axes['axc']
    
    @axc.setter
    def axc(self, kwgs):
        if "axc" not in self.axes.keys():
            # Creation: initialise axes and empty _cb state
            self._init_axc_axes("axc", kwgs)

    # ------------------------------------------------------------------
    # Named colorbar axes support  (axc2, axc_mass, axc_logL, …)
    # ------------------------------------------------------------------

    def _init_axc_axes(self, name: str, kwgs: dict) -> None:
        """Create a named colorbar axes and attach an empty _cb state dict."""
        if name in self.axes:
            return
        axc_obj = self.fig.add_axes(**kwgs)
        axc_obj._cb = {
            "mode":   "auto",
            "levels": None,
            "vmin":   None,
            "vmax":   None,
            "norm":   None,
            "cmap":   None,
            "used":   False,
        }
        self.axes[name] = axc_obj
        if self.logger:
            self.logger.debug(f"Created named colorbar axes -> {name}")

    def _finalize_axc(self, name: str) -> None:
        """Draw the colorbar for a named axc* axes using its pre-built _cb state."""
        axc_obj = self.axes.get(name)
        if axc_obj is None or not hasattr(axc_obj, "_cb"):
            return
        if not axc_obj._cb.get("used"):
            return

        cb_frame = self.frame.get(name, {})
        is_h = axc_is_horizontal(self.frame, name)
        ticks_cfg = cb_frame.get("ticks", {}) if isinstance(cb_frame, dict) else {}
        if not isinstance(ticks_cfg, dict):
            ticks_cfg = {}

        mappable = mpl.cm.ScalarMappable(
            cmap=axc_obj._cb.get("cmap") or mpl.rcParams.get("image.cmap", "rainbow"),
            norm=axc_obj._cb.get("norm"),
        )
        mappable.set_array([])

        if not is_h:
            cbar = self.fig.colorbar(mappable, cax=axc_obj)
            cbar.minorticks_on()
            if axc_obj._cb.get("vmin") is not None and axc_obj._cb.get("vmax") is not None:
                axc_obj.set_ylim(axc_obj._cb["vmin"], axc_obj._cb["vmax"])
            if str(axc_obj._cb.get("mode", "auto")).lower() == "log":
                from matplotlib.ticker import LogLocator
                axc_obj.yaxis.set_minor_locator(LogLocator(subs="auto"))
            else:
                from matplotlib.ticker import AutoMinorLocator
                axc_obj.yaxis.set_minor_locator(AutoMinorLocator())
            _tp = ticks_cfg.get("ticks_position", "right")
            axc_obj.yaxis.set_ticks_position(_tp)
            axc_obj.yaxis.set_label_position("left" if _tp == "left" else "right")
            axc_obj.tick_params(**ticks_cfg.get("both", {}))
            axc_obj.tick_params(**ticks_cfg.get("major", {}))
            axc_obj.tick_params(**ticks_cfg.get("minor", {}))
            label_cfg = cb_frame.get("label", {}) if isinstance(cb_frame, dict) else {}
            if not isinstance(label_cfg, dict):
                label_cfg = {}
            if label_cfg:
                label_kwargs = dict(label_cfg)
                label_text = label_kwargs.pop("ylabel", "")
                axc_obj.set_ylabel(label_text if label_text is not None else "", **label_kwargs)
            ylabel_coords = cb_frame.get("ylabel_coords") if isinstance(cb_frame, dict) else None
            if ylabel_coords:
                axc_obj.yaxis.set_label_coords(ylabel_coords["x"], ylabel_coords["y"])
            self._apply_manual_ticks(axc_obj, "y", ticks_cfg.get("y", {}))
        else:
            cbar = self.fig.colorbar(mappable, cax=axc_obj, orientation="horizontal")
            cbar.minorticks_on()
            if axc_obj._cb.get("vmin") is not None and axc_obj._cb.get("vmax") is not None:
                axc_obj.set_xlim(axc_obj._cb["vmin"], axc_obj._cb["vmax"])
            if str(axc_obj._cb.get("mode", "auto")).lower() == "log":
                from matplotlib.ticker import LogLocator
                axc_obj.xaxis.set_minor_locator(LogLocator(subs="auto"))
            else:
                from matplotlib.ticker import AutoMinorLocator
                axc_obj.xaxis.set_minor_locator(AutoMinorLocator())
            _tp = ticks_cfg.get("ticks_position", "top")
            axc_obj.xaxis.set_ticks_position(_tp)
            axc_obj.xaxis.set_label_position("top" if _tp == "top" else "bottom")
            axc_obj.tick_params(**ticks_cfg.get("both", {}))
            axc_obj.tick_params(**ticks_cfg.get("major", {}))
            axc_obj.tick_params(**ticks_cfg.get("minor", {}))
            label_cfg = cb_frame.get("label", {}) if isinstance(cb_frame, dict) else {}
            if not isinstance(label_cfg, dict):
                label_cfg = {}
            if cb_frame.get("isxlabel") and label_cfg:
                label_kwargs = dict(label_cfg)
                label_text = label_kwargs.pop("xlabel", "")
                axc_obj.set_xlabel(label_text if label_text is not None else "", **label_kwargs)
            self._apply_manual_ticks(axc_obj, "x", ticks_cfg.get("x", {}))

        if self.logger:
            self.logger.debug(f"Finalized colorbar axes -> {name}")

    # ------------------------------------------------------------------
    # Colorbar pre-scan (called at the start of render())
    # ------------------------------------------------------------------

    def _prescan_colorbar_ranges(self) -> None:
        """Pre-scan all coloured layers to build _cb for every axc* axes.

        For each colorbar axis that has at least one bound layer:
        1. Load the layer data (DataPreprocessor cache makes this cheap).
        2. Collect the colour-channel data range.
        3. Release the data immediately (memory profile unchanged).
        4. Resolve vmin/vmax from frame config + data ranges, validate,
           and build the norm object via precompute_colorbar_cb().

        After this call all _cb dicts are fully populated and treated as
        read-only during the main render loop.
        """
        cb_ranges: dict[str, list] = {}

        for ax, ly in self._render_queue:
            cb_name = ly.get("colorbar", "axc")
            if cb_name not in self.axes:
                continue

            coor = ly.get("coor", {})
            style = dict(ly.get("style") or {})
            method_key = str(ly.get("method", "scatter")).lower()

            if not layer_uses_color(style, coor, method_key):
                continue

            # Load data (pipeline cache hit on second call in render loop)
            self._load_layer_runtime_data(ly)
            df = ly.get("data")
            if df is not None:
                df = self._ensure_pandas_data(df, reason="prescan:colorbar")
                color_cfg = axc_color_config(self.frame, cb_name)
                lo, hi = collect_layer_color_range(df, coor, style, scale=color_cfg.get("scale"))
                if lo is not None or hi is not None:
                    cb_ranges.setdefault(cb_name, []).append((lo, hi))
            # Release immediately to preserve memory profile
            self._release_layer_runtime_data(ly, consume_sources=False)

        # Build _cb for each colorbar that has data
        for cb_name, ranges in cb_ranges.items():
            axc_obj = self.axes.get(cb_name)
            if axc_obj is None or not hasattr(axc_obj, "_cb"):
                continue
            color_cfg = axc_color_config(self.frame, cb_name)
            axc_obj._cb.update(
                precompute_colorbar_cb(color_cfg, ranges, logger=self.logger)
            )




    @property
    def ax(self): 
        return self.axes['ax']

    @ax.setter
    def ax(self, kwgs):
        if "ax" not in self.axes.keys():
            raw_ax = self.fig.add_axes(**kwgs)
            if "facecolor" in kwgs.keys():
                raw_ax.set_facecolor(kwgs['facecolor'])
            adapter = StdAxesAdapter(raw_ax)
            adapter._type = "rect"
            adapter.layers = []
            adapter._legend = self.frame['ax'].get("legend", False)
            self.axes['ax'] = adapter 
            adapter.status = 'configured'
        
        if self.frame['ax'].get("spines"): 
            if "color" in self.frame['ax']['spines']:
                for s in self.axes['ax'].spines.values():
                    s.set_color(self.frame['ax']['spines']['color'])
             
        def _safe_cast(v):
            try:
                return float(v)
            except Exception:
                return v
                    
        xlim = self.frame["ax"].get("xlim")
        if xlim:
            xlim = list(map(_safe_cast, xlim))
            self.ax.set_xlim(xlim)

        ylim = self.frame["ax"].get("ylim")
        if ylim:
            ylim = list(map(_safe_cast, ylim))
            self.ax.set_ylim(ylim)

                    
        from matplotlib.ticker import AutoMinorLocator, MaxNLocator, AutoLocator
        if self.frame['ax'].get("yscale", "").lower() == 'log':
            self.ax.set_yscale("log")
            from matplotlib.ticker import LogLocator
            self.ax.yaxis.set_minor_locator(LogLocator(subs='auto'))
        else:
            self.ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
            self.ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        if self.frame['ax'].get("xscale", "").lower() == 'log':
            self.ax.set_xscale("log")
            from matplotlib.ticker import LogLocator
            self.ax.xaxis.set_minor_locator(LogLocator(subs='auto'))
        else:
            self.ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
            self.ax.xaxis.set_minor_locator(AutoMinorLocator())
        


        if self.frame["ax"].get("text"): 
            for txt in self.frame["ax"]["text"]:
                if txt.get("transform", False):
                    txt.pop("transform")
                    self.ax.text(**txt, transform=self.ax.transAxes)
                else:
                    self.ax.text(**txt)



        if self.frame['ax']['labels'].get("x"):
            self.ax.set_xlabel(self.frame['ax']['labels']['x'], **self.frame['ax']['labels']['xlabel'])
        if self.frame['ax']['labels'].get("y"):
            self.ax.set_ylabel(self.frame['ax']['labels']['y'], **self.frame['ax']['labels']['ylabel'])
            self.ax.yaxis.set_label_coords(self.frame['ax']['labels']['ylabel_coords']['x'], self.frame['ax']['labels']['ylabel_coords']['y'])

        if self.frame['ax']['labels'].get("zorder"):
            for spine in self.ax.spines.values():
                spine.set_zorder(self.frame['ax']['labels']['zorder'])

        # Apply manual ticks here at initialization if provided in YAML
        ax_ticks_cfg = self.frame.get('ax', {}).get('ticks', {})
        # self._apply_manual_ticks(self.ax, "x", ax_ticks_cfg.get('x', {}))
        # self._apply_manual_ticks(self.ax, "y", ax_ticks_cfg.get('y', {}))


        self.ax.tick_params(**self.frame['ax']['ticks'].get("both", {}))
        self.ax.tick_params(**self.frame['ax']['ticks'].get("major", {}))
        self.ax.tick_params(**self.frame['ax']['ticks'].get("minor", {}))
        
        self._apply_axis_endpoints(self.axes['ax'], self.frame['ax'].get('xaxis', {}), "x")
        self._apply_axis_endpoints(self.axes['ax'], self.frame['ax'].get('yaxis', {}), "y")

        # ---- Finalize logic with auto-ticks injection ----
        if getattr(self.ax, 'needs_finalize', True) and hasattr(self.ax, 'finalize'):
            orig_finalize = self.ax.finalize
            def wrapped_finalize():
                # try:
                #     if not self._has_manual_ticks('ax', 'x'):
                #         self._apply_auto_ticks(self.ax, 'x')
                #     if not self._has_manual_ticks('ax', 'y'):
                #         self._apply_auto_ticks(self.ax, 'y')
                # except Exception as e:
                #     if self.logger:
                #         self.logger.warning(f"Auto ticks failed on ax: {e}")
                try:
                    orig_finalize()
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Finalize failed on axes 'ax': {e}")
            self.ax.finalize = wrapped_finalize
            self.ax.finalize()

        self.logger.debug("Loaded main rectangle axes -> ax")

    def _apply_legend_on_axes(self, ax_name: str, ax_obj, leg_cfg: dict):
        """Apply a legend on a specific axes using a YAML dict stored under frame['axes'][ax_name]['legend'].
        Supports an optional 'enabled' key (default True). Any 'axes' key will be ignored here."""
        if not isinstance(leg_cfg, dict):
            return
        if leg_cfg.get("enabled", True) is False:
            return
        kw = dict(leg_cfg)
        kw.pop("axes", None)  # per-axes legend doesn't need this
        try:
            (ax_obj.ax if hasattr(ax_obj, "ax") else ax_obj).legend(**kw)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Legend apply failed on '{ax_name}': {e}")


    def _install_tri_auto_clip(self, ax):
        """
        Install auto-clip wrappers on this Axes so that any newly created or
        added artists are clipped to ax._clip_path (data coords) automatically.
        This affects high-level draw calls (plot/scatter/contour/contourf/imshow)
        and low-level add_* entry points (add_line/add_collection/add_patch/add_artist).
        """
        if not hasattr(ax, "_jp_orig"):
            ax._jp_orig = {}

        def _wrap_high_level(name):
            if hasattr(ax, name) and name not in ax._jp_orig:
                ax._jp_orig[name] = getattr(ax, name)
                def wrapped(self_ax, *args, **kwargs):
                    out = ax._jp_orig[name](*args, **kwargs)
                    # Only auto-clip if a triangular clip path is defined
                    if getattr(self_ax, "_clip_path", None) is not None:
                        auto_clip(out, self_ax)
                    return out
                setattr(ax, name, MethodType(wrapped, ax))

        # Wrap common high-level APIs
        for m in ("plot", "scatter", "contour", "contourf", "imshow"):
            _wrap_high_level(m)

        # Wrap low-level add_* so indirect additions are also clipped
        def _wrap_add(name):
            if hasattr(ax, name) and name not in ax._jp_orig:
                ax._jp_orig[name] = getattr(ax, name)
                def wrapped_add(self_ax, artist, *args, **kwargs):
                    if getattr(self_ax, "_clip_path", None) is not None:
                        try:
                            artist.set_clip_path(self_ax._clip_path, transform=self_ax.transData)
                        except Exception:
                            pass
                    return ax._jp_orig[name](artist, *args, **kwargs)
                setattr(ax, name, MethodType(wrapped_add, ax))

        for m in ("add_line", "add_collection", "add_patch", "add_artist"):
            _wrap_add(m)

    def savefig(self):
        # self.ax.tight_layout()
        for fmt in self.fmts: 
            spf = os.path.join(self.dir, "{}.{}".format(self.name, fmt))
            try:
                self.logger.warning(
                        "JarvisPlot successfully draw {}\t in {:.3f}s sec\n\t-> {}".format(self.name, float(time.perf_counter() - self.t0), spf)
                    )
            except Exception:
                pass 
            self.fig.savefig(spf, dpi=self.dpi)

    def load_axes(self):
        for ax, kws in self.frame['axes'].items():
            try:
                self.logger.debug("Loading axes -> {}".format(ax))
            except Exception:
                pass

            if ax == "axlogo":
                self.axlogo = kws
            elif ax == "axtri":
                self.axtri  = kws
            elif ax == "axc":
                self.axc    = kws
            elif ax.startswith("axc") and len(ax) > 3:
                # named secondary colorbar axes: axc2, axc_mass, axc_logL, …
                self._init_axc_axes(ax, kws)
            elif ax == "ax":
                self.ax     = kws
            elif self._is_numbered_ax(ax):
                self._ensure_numbered_rect_axes(ax, kws)
            else:
                try:
                    self.logger.warning(f"Unsupported axes key '{ax}'. Only 'ax' or 'ax<NUMBER>' are allowed.")
                except Exception:
                    pass
        
        # import matplotlib.pyplot as plt
        # plt.show()
    
    def plot(self):
        self.render()
        # for layer in self.layers:
        if self.debug: 
            if "axtri" in self.axes.keys():
                # Demo of Scatter Clip
                x = np.linspace(-1, 2, 121)
                y = np.linspace(-1, 2, 121)
                X, Y = np.meshgrid(x, y)
                self.axtri.scatter(x=X.ravel() + 0.5 *  Y.ravel(), y=Y.ravel(), marker='.', s=1, facecolor="#0277BA", edgecolor="None")

                # Demo of Plot Clip 
                self.axtri.plot(x=[-1, 0.5, 0.5, 2], y=[-1.1, 0.6, 0.3, 1.8], linestyle="-", color="#0277BA")
        self.savefig()
        import matplotlib.pyplot as plt
        plt.close(self.fig)
        
    def render(self):
        """
        Render all layers attached to each axes (we appended them in axtri/axlogo setters).
        """
        memtrace_checkpoint(
            self.logger,
            "figure.render.before",
            None,
            extra={
                "figure": self.name,
                "layers": len(self._render_queue),
            },
        )

        # Pre-scan: collect colour ranges and build _cb for all axc* axes
        # before any layer is rendered.  This makes colorbar limits fully
        # determined by frame config + data — never by render order.
        self._prescan_colorbar_ranges()

        for ax, ly in self._render_queue:
            self._load_layer_runtime_data(ly)
            self.render_layer(ax, ly)
            self._release_layer_runtime_data(ly)

        for ax_name, ax in self.axes.items():
            # mark drawn after all layers on this axes
            if hasattr(ax, 'status'):
                ax.status = 'drawn'

        for name, ax in self.axes.items():
            try:
                if hasattr(ax, "_legend") and ax._legend:
                    target_ax = ax.ax if hasattr(ax, "ax") else ax
                    target_ax.legend(**ax._legend)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Legend draw failed on axes '{name}': {e}")

        # Finalize all colorbar axes (axc and any named axc*)
        for name in list(self.axes.keys()):
            if name == "axc" or (name.startswith("axc") and len(name) > 3):
                try:
                    self._finalize_axc(name)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Colorbar finalize failed on '{name}': {e}")

        # ---- Finalize axes that want it ----
        for name, ax in self.axes.items():
            # Auto ticks only if user did not provide manual ticks for this axis
            if name == 'ax':
                if not self._has_manual_ticks('ax', 'x'):
                    self._apply_auto_ticks(ax, 'x')
                if not self._has_manual_ticks('ax', 'y'):
                    self._apply_auto_ticks(ax, 'y')
            elif name == "axc" or (name.startswith("axc") and len(name) > 3):
                axc_tick_axis = 'x' if axc_is_horizontal(self.frame, name) else 'y'
                if not self._has_manual_ticks(name, axc_tick_axis):
                    self._apply_auto_ticks(self.axes[name], axc_tick_axis)
                continue  # already handled above, skip finalize below
            if getattr(ax, 'needs_finalize', True) and hasattr(ax, 'finalize'):
                try:
                    ax.finalize()
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Finalize failed on axes '{name}': {e}")
        memtrace_checkpoint(
            self.logger,
            "figure.render.after",
            None,
            extra={
                "figure": self.name,
                "axes": len(self.axes),
            },
        )

    def _apply_manual_ticks(self, ax_obj, which: str, ticks_cfg: dict):
        """Apply manual ticks if YAML provides them; otherwise keep auto.
        YAML:
          frame.ax.ticks.x: { positions: [...], labels: [...] }
          frame.ax.ticks.y: { positions: [...], labels: [...] }
          frame.axc.ticks.y: { positions: [...], labels: [...] }
        """
        return apply_manual_ticks(self, ax_obj, which, ticks_cfg)
            
    # --- config ingestion ---
    def from_dict(self, info: Mapping) -> bool:
        """Apply settings from a dict. Returns True if any field was set."""
        return apply_figure_config(self, info)


    # Backward-compatible alias if other code still calls `set(info)`

    def set(self, info: Mapping) -> bool:
        return self.from_dict(info)
    
    def load_path(self, path, base_dir=None):
        """Resolve a path string.

        Rules:
          - "&JP/..." is resolved relative to the Jarvis-PLOT repository root.
          - Absolute paths are kept.
          - Relative paths are resolved relative to `base_dir` if provided, otherwise CWD.
        """
        return str(resolve_project_path(path, base_dir=base_dir or self._yaml_dir or "."))
    

    def _eval_series(self, df: pd.DataFrame, set: dict):
        """
        Evaluate an expression/column name against df safely.
        - If expr is a direct column name, returns that series.
        - If expr is a python expression, eval with df columns in scope.
        """
        try: 
            self.logger.debug("Loading variable expression -> {}".format(set['expr'])) 
        except Exception:
            pass 
        if not "expr" in set.keys():
            raise ValueError(f"expr need for axes {set}.")
        arr = eval_dataframe_expression(
            df,
            set["expr"],
            logger=self.logger,
            fillna=set.get("fillna", None),
        )
        return np.asarray(arr)





    def render_layer(self, ax, layer_info):
        return runtime_render_layer(self, ax, layer_info)



    

    # def auto_clip(artists, ax, clip_obj=None, transform=None):
def auto_clip(artists, ax):
        
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import Patch as MplPatch
    from matplotlib.container import BarContainer, ErrorbarContainer
    
    def _apply_to_one(a):
        try:
            a.set_clip_path(ax._clip_path, transform=ax.transData)
            return True
        except Exception:
            return False
        
    def _apply(obj):
        if _apply_to_one(obj):
            return True
        coll = getattr(obj, "collections", None)
        if coll is not None:
            for c in coll:
                _apply_to_one(c)
            return True
        if isinstance(obj, BarContainer):
            for p in obj.patches:
                _apply_to_one(p)
            return True
        if isinstance(obj, ErrorbarContainer):
            for line in obj.lines:
                _apply_to_one(line)
            if hasattr(obj, "has_xerr") and obj.has_xerr and obj.has_yerr:
                for lc in getattr(obj, "barlinecols", []):
                    _apply_to_one(lc)
            return True
        # 4) violinplot 
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, (list, tuple)):
                    for a in v:
                        _apply_to_one(a)
                else:
                    _apply_to_one(v)
            return True
        # 5) iterabile object 
        try:
            iterator = iter(obj)
        except TypeError:
            return False
        else:
            for a in iterator:
                _apply_to_one(a)
            return True

    _apply(artists)
    return artists

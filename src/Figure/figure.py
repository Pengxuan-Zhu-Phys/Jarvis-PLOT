#!/usr/bin/env python3
from copy import deepcopy
from typing import Optional, Mapping
import numpy as np 
import os, sys 
from ..core import jppwd
import matplotlib.ticker as mticker
import pandas as pd 
import matplotlib as mpl
from types import MethodType
from .adapters import StdAxesAdapter, TernaryAxesAdapter

import json
import time
import re


class Figure:
    def _is_numbered_ax(self, name: str) -> bool:
        """Return True iff name matches ax<NUMBER>, e.g. ax1, ax2, ax10."""
        return isinstance(name, str) and re.fullmatch(r"ax\d+", name) is not None

    def _ensure_numbered_rect_axes(self, ax_name: str, kwgs: dict):
        """Create/configure a numbered rectangular axes (ax1, ax2, ...) using ax-style logic."""
        if not self._is_numbered_ax(ax_name):
            raise ValueError(f"Illegal dynamic axes name '{ax_name}'. Only ax<NUMBER> is allowed.")

        # Reuse ax-style construction, but read frame config from frame[ax_name]
        if ax_name not in self.axes.keys():
            raw_ax = self.fig.add_axes(**kwgs)
            if isinstance(kwgs, dict) and ("facecolor" in kwgs):
                raw_ax.set_facecolor(kwgs['facecolor'])
            adapter = StdAxesAdapter(raw_ax)
            adapter._type = "rect"
            adapter.layers = []
            adapter._legend = self.frame.get(ax_name, {}).get("legend", False)
            self.axes[ax_name] = adapter
            adapter.status = 'configured'

        ax_obj = self.axes[ax_name]

        # ---- identical configuration path as ax, but keyed by ax_name ----
        if self.frame.get(ax_name, {}).get("spines"):
            if "color" in self.frame[ax_name]["spines"]:
                for s in ax_obj.spines.values():
                    s.set_color(self.frame[ax_name]['spines']['color'])

        # y scale
        if self.frame.get(ax_name, {}).get("yscale", "").lower() == 'log':
            ax_obj.set_yscale("log")
            from matplotlib.ticker import LogLocator
            ax_obj.yaxis.set_minor_locator(LogLocator(subs='auto'))
        else:
            from matplotlib.ticker import AutoMinorLocator
            ax_obj.yaxis.set_minor_locator(AutoMinorLocator())

        # x scale
        if self.frame.get(ax_name, {}).get("xscale", "").lower() == 'log':
            ax_obj.set_xscale("log")
            from matplotlib.ticker import LogLocator
            ax_obj.xaxis.set_minor_locator(LogLocator(subs='auto'))
        else:
            from matplotlib.ticker import AutoMinorLocator
            ax_obj.xaxis.set_minor_locator(AutoMinorLocator())

        def _safe_cast(v):
            try:
                return float(v)
            except Exception:
                return v

        # text
        if self.frame.get(ax_name, {}).get("text"):
            for txt in self.frame[ax_name]["text"]:
                if txt.get("transform", False):
                    txt.pop("transform")
                    ax_obj.text(**txt, transform=ax_obj.transAxes)
                else:
                    ax_obj.text(**txt)

        # limits
        xlim = self.frame.get(ax_name, {}).get("xlim")
        if xlim:
            ax_obj.set_xlim(list(map(_safe_cast, xlim)))

        ylim = self.frame.get(ax_name, {}).get("ylim")
        if ylim:
            ax_obj.set_ylim(list(map(_safe_cast, ylim)))

        # labels
        if self.frame.get(ax_name, {}).get('labels', {}).get("x"):
            ax_obj.set_xlabel(self.frame[ax_name]['labels']['x'], **self.frame[ax_name]['labels']['xlabel'])
        if self.frame.get(ax_name, {}).get('labels', {}).get("y"):
            ax_obj.set_ylabel(self.frame[ax_name]['labels']['y'], **self.frame[ax_name]['labels']['ylabel'])
            # ax_obj.yaxis.set_label_coords(
            #     self.frame[ax_name]['labels']['ylabel_coords']['x'],
            #     self.frame[ax_name]['labels']['ylabel_coords']['y']
            # )

        # ticks
        ax_obj.tick_params(**self.frame.get(ax_name, {}).get('ticks', {}).get("both", {}))
        ax_obj.tick_params(**self.frame.get(ax_name, {}).get('ticks', {}).get("major", {}))
        ax_obj.tick_params(**self.frame.get(ax_name, {}).get('ticks', {}).get("minor", {}))

        self._apply_axis_endpoints(ax_obj, self.frame.get(ax_name, {}).get('xaxis', {}), "x")
        self._apply_axis_endpoints(ax_obj, self.frame.get(ax_name, {}).get('yaxis', {}), "y")

        # finalize
        if getattr(ax_obj, 'needs_finalize', True) and hasattr(ax_obj, 'finalize'):
            try:
                ax_obj.finalize()
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Finalize failed on axes '{ax_name}': {e}")

        try:
            self.logger.debug(f"Loaded numbered rectangle axes -> {ax_name}")
        except Exception:
            pass

        return ax_obj
    def _has_manual_ticks(self, ax_key: str, which: str) -> bool:
        """Return True if YAML provides manual tick positions for given axis."""
        try:
            if ax_key == 'ax':
                ticks_cfg = self.frame.get('ax', {}).get('ticks', {})
            elif ax_key == 'axc':
                ticks_cfg = self.frame.get('axc', {}).get('ticks', {})
            else:
                return False
            node = ticks_cfg.get(which, {})
            return isinstance(node, dict) and ((node.get('positions') is not None) or (node.get('pos') is not None))
        except Exception:
            return False

    def _apply_axis_endpoints(self, ax_obj, axis_cfg: dict, which: str):
        """
        which: 'x' or 'y'
        axis_cfg: self.frame['ax'].get('xaxis', {}) / 'yaxis'
        """
        if not isinstance(axis_cfg, dict):
            return

        # Resolve underlying Matplotlib Axes (StdAxesAdapter or plain Axes)
        target = ax_obj.ax if hasattr(ax_obj, "ax") else ax_obj

        if which == 'x':
            ticks = target.xaxis.get_major_ticks()
            locs  = target.xaxis.get_majorticklocs()
        else:
            ticks = target.yaxis.get_major_ticks()
            locs  = target.yaxis.get_majorticklocs()
        if not ticks:
            return

        # Get axis limits for boundary check
        if which == 'x':
            lim0, lim1 = target.get_xlim()
        else:
            lim0, lim1 = target.get_ylim()

        min_cfg = axis_cfg.get("min_endpoints", {})
        max_cfg = axis_cfg.get("max_endpoints", {})
        width   = abs(lim0 - lim1)

        # 第一个 tick = min 端点
        t0 = ticks[0]
        # Check whether first tick is at the lower boundary
        t0_loc = locs[0]

        if abs(t0_loc - lim0) < 1e-3 * width:
            if min_cfg.get("tick") is False:
                t0.tick1line.set_visible(False)
                t0.tick2line.set_visible(False)
            if min_cfg.get("label") is False:
                t0.label1.set_visible(False)
                t0.label2.set_visible(False)

        # 最后一个 tick = max 端点
        t1 = ticks[-1]
        # Check whether last tick is at the upper boundary
        t1_loc = locs[-1]

        if abs(t1_loc - lim1) < 1e-3 * width:
            if max_cfg.get("tick") is False:
                t1.tick1line.set_visible(False)
                t1.tick2line.set_visible(False)
            if max_cfg.get("label") is False:
                t1.label1.set_visible(False)
                t1.label2.set_visible(False)
                
    def _apply_auto_ticks(self, ax_obj, which: str):
        """Lightweight auto-tick post-processing at finalize stage.

        Goals:
        - Never print/debug here.
        - X: rotate long labels a bit.
        - Y: if log-like scale -> keep log spacing and use compact decimals for decades in [1e-2, 1e2],
             otherwise defer to Matplotlib's LogFormatter.
             if linear scale -> use ScalarFormatter with sci notation, but never touch log formatters.
        """
        target = ax_obj.ax if hasattr(ax_obj, "ax") else ax_obj
        axis = target.xaxis if which == 'x' else target.yaxis

        try:
            # Ensure ticks/labels exist
            labels = axis.get_ticklabels()

            # --- X axis formatting (match Y axis behavior)
            if which == 'x':
                try:
                    xscale = target.get_xscale()
                except Exception:
                    xscale = None

                # only touch linear x-axis
                if xscale not in ('log', 'symlog', 'logit'):
                    fmt = mticker.ScalarFormatter(useMathText=True)
                    # Do not use sci/offset for small exponents like 10^-1; reserve for <=1e-2 or >=1e4
                    fmt.set_powerlimits((-2, 4))
                    axis.set_major_formatter(fmt)
                    try:
                        target.ticklabel_format(style='sci', axis='x', scilimits=(-2, 4))
                    except Exception:
                        pass
                    try:
                        axis.set_offset_position('bottom')
                    except Exception:
                        pass
                    mpl.rcParams['axes.formatter.useoffset'] = True

                    # Ensure offset text is rendered and sized like tick labels (avoid huge ×10^n)
                    try:
                        target.figure.canvas.draw_idle()
                        tl = axis.get_ticklabels()
                        if tl:
                            axis.offsetText.set_fontsize(tl[0].get_size() * 0.8)
                        # Place the offset near the left of the x-axis (like y-axis), and nudge up/right
                        axis.offsetText.set_horizontalalignment('left')
                        # axis.offsetText.set_verticalalignment('top')
                        axis.offsetText.set_x(1.02)   # axes fraction
                        # axis.offsetText.set_y(1.08)   # small upward nudge (axes fraction)

                    except Exception:
                        pass

                # optional: rotate long labels
                try:
                    long = any((t is not None) and (len(t) > 6) for t in (l.get_text() for l in labels))
                except Exception:
                    long = False
                # if long:
                #     target.tick_params(axis='x', labelrotation=35)

                return

            # --- Y axis formatting
            if which != 'y':
                return

            # Detect scale
            try:
                yscale = target.get_yscale()
            except Exception:
                yscale = None

            # 1) Log-like y-axis: compact decimals in range, otherwise default log formatter
            if yscale in ('log', 'symlog', 'logit'):
                from matplotlib.ticker import LogFormatterMathtext, FuncFormatter

                base = LogFormatterMathtext()
                lo, hi = 1e-2, 1e2  # compact decimal range

                def _fmt(val, pos=None):
                    if val is None or val <= 0:
                        return ""
                    try:
                        exp = np.log10(val)
                    except Exception:
                        return ""
                    # Only label exact decades
                    if (not np.isfinite(exp)) or (not np.isclose(exp, round(exp))):
                        return ""

                    if lo <= val <= hi:
                        e = int(round(exp))
                        if e >= 0:
                            return f"{int(10**e)}"
                        # e=-1 -> 0.1 (1 dp), e=-2 -> 0.01 (2 dp)
                        return f"{10**e:.{abs(e)}f}"

                    # outside compact range: defer to Matplotlib
                    return base(val, pos)

                axis.set_major_formatter(FuncFormatter(_fmt))
                return

            # 2) Linear y-axis: ScalarFormatter sci notation
            fmt = mticker.ScalarFormatter(useMathText=True)
            # Do not use sci/offset for small exponents like 10^-1; reserve for <=1e-2 or >=1e4
            fmt.set_powerlimits((-2, 4))
            axis.set_major_formatter(fmt)
            try:
                target.ticklabel_format(style='sci', axis='y', scilimits=(-2, 4))
            except Exception:
                pass
            try:
                axis.set_offset_position('left')
            except Exception:
                pass
            mpl.rcParams['axes.formatter.useoffset'] = True
            try:
                target.figure.canvas.draw_idle()
                # shrink offset text a bit
                try:
                    tl = axis.get_ticklabels()
                    if tl:
                        axis.offsetText.set_fontsize(tl[0].get_size() * 0.8)
                except Exception:
                    pass
            except Exception:
                pass

        except Exception:
            # Always fail silently here; tick post-processing must never crash plotting.
            return
        
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
        self._ctx       = None
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
        from copy import deepcopy 
        if len(value) == 2: 
            self._frame = deepcopy(self.jpstyles[value[0]][value[1]]['Frame'])
            self._style = deepcopy(self.jpstyles[value[0]][value[1]]['Style'])
            self.logger.debug("Style: [{} : {}] used for figure -> {}".format(value[0], value[1], self.name))
        elif len(value) == 1: 
            self._frame = deepcopy(self.jpstyles[value[0]]["default"]['Frame'])
            self._style = deepcopy(self.jpstyles[value[0]]["default"]['Style'])
            self.logger.debug("Style: [{} : {}] used for figure -> {}".format(value[0], "default", self.name))
        else:
            self.logger.error("Undefined style -> {}".format(value))
            raise TypeError
    
    @property
    def context(self):
        return self._ctx

    @context.setter
    def context(self, value):
        self._ctx = value  # 期望是 DataContext


    
    
    @property
    def layers(self): 
        return self._layers 
    
    @layers.setter
    def layers(self, infos):
        for layer in infos: 
            info = {}
            ax  = self.axes[layer['axes']]
            info['name'] = layer.get("name", "")
            info['data'] = self.load_layer_data(layer)
            info['combine'] = layer.get("combine", "concat")
            if layer.get("share_data") and info['data'] is not None:
                from copy import deepcopy
                self.context.update(layer["share_data"], deepcopy(info['data']))
            info['coor'] = layer['coordinates']
            info['method'] = layer.get("method", "scatter")
            info['style'] = layer.get("style", {})
            ax.layers.append(info)
            self.logger.debug("Successfully loaded layer -> {}".format(info["name"]))
                        
    def load_layer_data(self, layer):
        lyinfo = layer.get("data", False)
        lycomb = layer.get("combine", "concat")
        if lyinfo:
            if lycomb == "concat":
                dts = []
                for ds in lyinfo:
                    src = ds.get('source')
                    self.logger.debug("Loading layer data source -> {}".format(src))
                    if src and self.context:
                        from copy import deepcopy
                        if isinstance(src, (list, tuple)):
                            self.logger.debug("loading datasets in list mode")
                            dsrc = []
                            for srcitem in src: 
                                self.logger.debug("loading layer data source item -> {}".format(srcitem))
                                dt = deepcopy(self.context.get(srcitem))
                                dsrc.append(dt)
                            dfsrcs = pd.concat(dsrc, ignore_index=False)
                            dt = self.load_bool_df(dfsrcs, ds.get("transform", None))
                            dts.append(dt)
                        elif self.context.get(src) is not None:
                            dt = deepcopy(self.context.get(src))
                            dt = self.load_bool_df(dt, ds.get("transform", None))
                            dts.append(dt)
                    else:
                        self.logger.error("DataSet -> {} not specified".format(src))
                if len(dts) == 0:
                    return None
                try:
                        return pd.concat(dts, ignore_index=False)
                except Exception:
                    return dts[0]
            elif lycomb == "seperate":
                dts = {}
                for ds in lyinfo: 
                    src = ds.get("source")
                    label = ds.get("label")
                    self.logger.debug("Loading layer data source -> {}".format(src))
                    if src and self.context and self.context.get(src) is not None:
                        from copy import deepcopy
                        dt = deepcopy(self.context.get(src))
                        dt = self.load_bool_df(dt, ds.get("transform", None))
                        dts[label] = dt
                    else:
                        self.logger.error("DataSet -> {} not specified".format(src))
                if len(dts) == 0: 
                    return None 
                return dts 
        # Unsupported lyinfo shape -> no data
        return None
               
         
    def load_bool_df(self, df, transform):
        if transform is None:
            return df 
        elif not isinstance(transform, list): 
            self.logger.error("illegal transform format, list type needed ->".format(json.dump(transform)))
            return df 
        else: 
            for trans in transform:
                self.logger.debug("Applying the transform ... ")
                if "filter" in trans.keys():
                    self.logger.debug("Before filtering -> {}".format(df.shape))
                    from .load_data import filter
                    df = filter(df, trans['filter'], self.logger)
                    self.logger.debug("After filtering -> {}".format(df.shape))
                elif "profile" in trans.keys(): 
                    from .load_data import profiling
                    df = profiling(df, trans['profile'], self.logger)
                    self.logger.debug("After profiling -> {}".format(df.shape))
                elif "sortby" in trans.keys(): 
                    from .load_data import sortby
                    df = sortby(df, trans['sortby'], self.logger)
                    self.logger.debug("After sortby -> {}".format(df.shape))
                elif "add_column" in trans.keys(): 
                    from .load_data import addcolumn
                    df = addcolumn(df, trans['add_column'], self.logger)
                    self.logger.debug("After Add-column -> {}".format(df.shape))
                
            return df 
                         
         
            
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
            axc = self.fig.add_axes(**kwgs)
            axc._cb = {
                "mode":  "auto",     # auto|linear|log|diverging
                "levels": None,
                "label": None,
                "vmin": None, "vmax": None,
                "norm": None,
                "used": False
            }
            self.axes["axc"] = axc
        else:
            if not self.axc._cb.get("used"):
                return

            # Build mappable for the colorbar
            mappable = mpl.cm.ScalarMappable(
                cmap=self.axc._cb.get("cmap") or mpl.rcParams.get("image.cmap", "rainbow"),
                norm=self.axc._cb.get("norm")
            )
            mappable.set_array([])
            if self.frame['axc'].get('orientation') != "horizontal":
                cbar = self.fig.colorbar(mappable, cax=self.axc)
                cbar.minorticks_on()
                self.axc.set_ylim(self.axc._cb['vmin'], self.axc._cb['vmax'])

                if str(self.axc._cb.get('mode', 'auto')).lower() == 'log':
                    from matplotlib.ticker import LogLocator
                    # Use default subs for log scale minor ticks
                    self.axc.yaxis.set_minor_locator(LogLocator(subs='auto'))
                else:
                    from matplotlib.ticker import AutoMinorLocator
                    self.axc.yaxis.set_minor_locator(AutoMinorLocator())

                self.axc.yaxis.set_ticks_position(self.frame['axc']['ticks']['ticks_position'])
                self.axc.yaxis.set_label_position("right")

                # Apply tick params (major/minor) as provided in frame config
                self.axc.tick_params(**self.frame['axc']['ticks'].get('both', {}))
                self.axc.tick_params(**self.frame['axc']['ticks'].get('major', {}))
                self.axc.tick_params(**self.frame['axc']['ticks'].get('minor', {}))

                self.axc.yaxis.set_ticks_position(self.frame['axc']['ticks']['ticks_position'])
                self.axc.yaxis.set_label_position("right")

                # Apply tick params (major/minor) as provided in frame config
                self.axc.tick_params(**self.frame['axc']['ticks'].get('both', {}))
                self.axc.tick_params(**self.frame['axc']['ticks'].get('major', {}))
                self.axc.tick_params(**self.frame['axc']['ticks'].get('minor', {}))
                self.axc.set_ylabel(**self.frame['axc'].get('label', {}))
                if self.frame['axc'].get('ylabel_coords'): 
                    self.axc.yaxis.set_label_coords(self.frame['axc']['ylabel_coords']['x'], self.frame['axc']['ylabel_coords']['y'])
                                    # Apply manual ticks for colorbar (y-axis) at initialization if provided
                cbar_ticks_cfg = self.frame.get('axc', {}).get('ticks', {}).get('y', {})
                # self._apply_manual_ticks(self.axc, 'y', cbar_ticks_cfg)
                self.logger.debug("Loaded colorbar axes -> axc")
            # else:
            #     cbar = self.fig.colorbar(mappable, cax=self.axc, orientation="horizontal")
            #     cbar.minorticks_on()
            #     self.axc.xaxis.set_ticks_position(self.frame['axc']['ticks']['ticks_position'])
            #     self.axc.xaxis.set_label_position("top")    
            #     self.axc.set_xlim(self.axc._cb['vmin'], self.axc._cb['vmax'])

            #     self.axc.set_xlabel(**self.frame['axc'].get("label", {}))
                
            else:
                # Horizontal colorbar: drive everything from x-axis
                cbar = self.fig.colorbar(mappable, cax=self.axc, orientation="horizontal")
                cbar.minorticks_on()

                # Limits use xlim for horizontal bars
                self.axc.set_xlim(self.axc._cb['vmin'], self.axc._cb['vmax'])

                # Minor locator depends on mode
                if str(self.axc._cb.get('mode', 'auto')).lower() == 'log':
                    from matplotlib.ticker import LogLocator
                    self.axc.xaxis.set_minor_locator(LogLocator(subs='auto'))
                else:
                    from matplotlib.ticker import AutoMinorLocator
                    self.axc.xaxis.set_minor_locator(AutoMinorLocator())

                # Ticks/label positions (top/bottom for horizontal)
                _tp = self.frame.get('axc', {}).get('ticks', {}).get('ticks_position', 'top')
                self.axc.xaxis.set_ticks_position(_tp)
                self.axc.xaxis.set_label_position('top' if _tp == 'top' else 'bottom')

                # Apply tick params (both/major/minor)
                self.axc.tick_params(**self.frame.get('axc', {}).get('ticks', {}).get('both', {}))
                self.axc.tick_params(**self.frame.get('axc', {}).get('ticks', {}).get('major', {}))
                self.axc.tick_params(**self.frame.get('axc', {}).get('ticks', {}).get('minor', {}))
                # Apply label if configured
                if self.frame.get('axc', {}).get('isxlabel'):
                    self.axc.set_xlabel(**self.frame.get('axc', {}).get('label', {}))






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
                    
        if self.frame['ax'].get("yscale", "").lower() == 'log':
            self.ax.set_yscale("log")
            from matplotlib.ticker import LogLocator
            self.ax.yaxis.set_minor_locator(LogLocator(subs='auto'))
        else:
            from matplotlib.ticker import AutoMinorLocator
            self.ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        if self.frame['ax'].get("xscale", "").lower() == 'log':
            self.ax.set_xscale("log")
            from matplotlib.ticker import LogLocator
            self.ax.xaxis.set_minor_locator(LogLocator(subs='auto'))
        else:
            from matplotlib.ticker import AutoMinorLocator
            self.ax.xaxis.set_minor_locator(AutoMinorLocator())
        
        def _safe_cast(v):
            try:
                return float(v)
            except Exception:
                return v

        if self.frame["ax"].get("text"): 
            for txt in self.frame["ax"]["text"]:
                if txt.get("transform", False):
                    txt.pop("transform")
                    self.ax.text(**txt, transform=self.ax.transAxes)
                else:
                    self.ax.text(**txt)


        xlim = self.frame["ax"].get("xlim")
        if xlim:
            xlim = list(map(_safe_cast, xlim))
            self.ax.set_xlim(xlim)

        ylim = self.frame["ax"].get("ylim")
        if ylim:
            ylim = list(map(_safe_cast, ylim))
            self.ax.set_ylim(ylim)

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
            except:
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
        for ax_name, ax in self.axes.items():
            ly_list = getattr(ax, "layers", [])
            for ly in ly_list:
                self.render_layer(ax, ly)
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

        # finalize colorbar lazily (only if any colored layer appeared)
        if "axc" in self.axes:
            self.axc = True 

        # ---- Finalize axes that want it ----
        for name, ax in self.axes.items():
            # Auto ticks only if user did not provide manual ticks for this axis
            if name == 'ax':
                if not self._has_manual_ticks('ax', 'x'):
                    self._apply_auto_ticks(ax, 'x')
                if not self._has_manual_ticks('ax', 'y'):
                    self._apply_auto_ticks(ax, 'y')
            elif name == 'axc':
                if not self._has_manual_ticks('axc', 'y'):
                    self._apply_auto_ticks(ax, 'y')
            if getattr(ax, 'needs_finalize', True) and hasattr(ax, 'finalize'):
                try:
                    ax.finalize()
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Finalize failed on axes '{name}': {e}")

    def _apply_manual_ticks(self, ax_obj, which: str, ticks_cfg: dict):
        """Apply manual ticks if YAML provides them; otherwise keep auto.
        YAML:
          frame.ax.ticks.x: { positions: [...], labels: [...] }
          frame.ax.ticks.y: { positions: [...], labels: [...] }
          frame.axc.ticks.y: { positions: [...], labels: [...] }
        """
        if not isinstance(ticks_cfg, dict):
            return
        pos = ticks_cfg.get("positions") or ticks_cfg.get("pos")
        labs = ticks_cfg.get("labels")
        if pos is None:
            return
        target = ax_obj.ax if hasattr(ax_obj, "ax") else ax_obj
        try:
            if which == "x":
                target.set_xticks(pos)
                if labs is not None:
                    target.set_xticklabels(labs)
            elif which == "y":
                target.set_yticks(pos)
                if labs is not None:
                    target.set_yticklabels(labs)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Manual ticks apply failed on {which}-axis: {e}")
            
    # --- config ingestion ---
    def from_dict(self, info: Mapping) -> bool:
        """Apply settings from a dict. Returns True if any field was set.
        Expected keys (so far): 'name'.
        """
        if not isinstance(info, Mapping):
            raise TypeError("from_dict expects a mapping/dict")
        
        try: 
            changed = True
            if "name" in info:
                self.name = info["name"]  # use the property setter correctly
            else:
                changed = False

            # Base directory for resolving relative paths (e.g. output.dir, resources, etc.)
            # Core should pass one of these keys.
            if "yaml_dir" in info:
                self._yaml_dir = info.get("yaml_dir")
            elif "_yaml_dir" in info:
                self._yaml_dir = info.get("_yaml_dir")
            elif "yaml_path" in info:
                try:
                    from pathlib import Path
                    self._yaml_dir = str(Path(info.get("yaml_path")).expanduser().resolve().parent)
                except Exception:
                    pass

            if "debug" in info:
                self.debug = info['debug']
                try:
                    self.logger.debug("Loading plot -> {} in debug mode".format(self.name))
                except:
                    pass 
            self._enable = info.get("enable", True)
            if not self._enable:
                self.logger.warning("Skip plot -> {}".format(self.name))
                return False

            if "style" in info:
                self.style = info['style']
            else:
                self.style = ["a4paper_2x1", "default"]
            self.logger.debug("Figure style loaded")
            if "gambit" in info['style'][0]:
                self.mode = "gambit"


            if "frame" in info:
                self.frame = info['frame']

            import matplotlib.pyplot as plt
            plt.rcParams['mathtext.fontset'] = 'stix'
            # --- Ensure JarvisPLOT colormaps are registered globally before plotting ---
            try:
                from ..utils import cmaps as _jp_cmaps
                _cmaps_summary = _jp_cmaps.setup(force=True)
                if self.logger:
                    try: 
                        self.logger.debug(f"JarvisPLOT: colormaps registered (builtin/external): {_cmaps_summary}")
                        self.logger.debug(f"JarvisPLOT: available cmaps sample: {_jp_cmaps.list_available()[:10]} ...")
                    except: 
                        pass 
            except Exception as _e:
                if self.logger:
                    self.logger.warning(f"JarvisPLOT: failed to register colormaps: {_e}")
            # plt.rcParams['font.family'] = 'STIXGeneral'
            self.fig = plt.figure(**self.frame['figure'])
            # CLI override: disable logo panel
            if self.print:
                try:
                    if self.mode == "Jarvis": 
                        if isinstance(self.frame.get("axes"), dict):
                            self.frame["axes"].pop("axlogo", None)
                        self.frame.pop("axlogo", None)  # optional: drop logo content too
                except Exception:
                    pass
            self.load_axes()

            if "layers" in info: 
                self.layers = info['layers']
            else: 
                changed = False 
                
            return changed
        except: 
            return False 


    # Backward-compatible alias if other code still calls `set(info)`

    def set(self, info: Mapping) -> bool:
        return self.from_dict(info)
    
    def load_path(self, path, base_dir=None):
        """Resolve a path string.

        Rules:
          - "&JP/..." is resolved relative to JarvisPLOT project root (jppwd).
          - Absolute paths are kept.
          - Relative paths are resolved relative to `base_dir` if provided, otherwise CWD.
        """
        path = str(path)
        if path.startswith("&JP/"):
            return os.path.abspath(os.path.join(jppwd, path[4:]))

        from pathlib import Path
        p = Path(path).expanduser()
        if p.is_absolute():
            return str(p.resolve())

        if base_dir:
            try:
                bd = Path(str(base_dir)).expanduser().resolve()
                return str((bd / p).resolve())
            except Exception:
                pass

        return str(p.resolve())
    
    # --- unified method dispatch ---
    METHOD_DISPATCH = {
        "scatter": "scatter",
        "plot": "plot",
        "fill": "fill",
        "contour": "contour",
        "contourf": "contourf",
        "imshow": "imshow",
        "hist": "hist",
        "hexbin": "hexbin",
        "tricontour": "tricontour",
        "tricontourf": "tricontourf",
        "voronoi": "voronoi",
        "voronoif": "voronoif"
    }

    def _eval_series(self, df: pd.DataFrame, set: dict):
        """
        Evaluate an expression/column name against df safely.
        - If expr is a direct column name, returns that series.
        - If expr is a python expression, eval with df columns in scope.
        """
        try: 
            self.logger.debug("Loading variable expression -> {}".format(set['expr'])) 
        except: 
            pass 
        if not "expr" in set.keys():
            raise ValueError(f"expr need for axes {set}.")
        if set["expr"] in df.columns:
            arr = df[set["expr"]].values
            if np.isnan(arr).sum() and "fillna" in set.keys():
                arr = np.where(np.isnan(arr), float(set['fillna']), arr)
        else: 
            # safe-ish eval with only df columns in locals
            local_vars = df.to_dict("series")
            import math
            from ..inner_func import update_funcs
            allowed_globals = update_funcs({"np": np, "math": math})
            arr = eval(set["expr"], allowed_globals, local_vars)
            if np.isnan(arr).sum() and "fillna" in set.keys():
                arr = np.where(np.isnan(arr), float(set['fillna']), arr)
        return np.asarray(arr)

    def _cb_collect_and_attach(self, style: dict, coor: dict, method_key: str, df: pd.DataFrame) -> dict:
        import matplotlib.colors as mcolors
        axc = self.axes.get("axc")
        if axc is None or not hasattr(axc, "_cb"):
            return style

        s = dict(style)
        uses_color = bool(style.get("cmap")) or ("c" in coor)
        if not uses_color:
            return style
        self.axc._cb["cmap"] = s.get("cmap")

        # ---- 1) records vmin and vmax ----
        z = None
        if self.axc._cb["vmin"] is None:
            if ("vmin" in s and isinstance(s['vmin'], (int, float))):
                self.axc._cb['vmin'] = s['vmin']
            else:
                if z is None:
                    if "z" in coor and isinstance(coor["z"], dict) and "expr" in coor["z"]:
                        z = self._eval_series(df, {"expr": coor["z"]["expr"]})
                    elif "c" in coor and isinstance(coor["c"], dict) and "expr" in coor["c"]:
                        z = self._eval_series(df, {"expr": coor["c"]["expr"]})
                if z is not None:
                    z = z[np.isfinite(z)]
                    if z.size:
                        self.axc._cb["vmin"] = float(np.min(z))

        if self.axc._cb["vmax"] is None:
            if ("vmax" in s and isinstance(s['vmax'], (int, float))):
                self.axc._cb['vmax'] = s['vmax']
            else:
                if z is None:
                    if "z" in coor and isinstance(coor["z"], dict) and "expr" in coor["z"]:
                        z = self._eval_series(df, {"expr": coor["z"]["expr"]})
                    elif "c" in coor and isinstance(coor["c"], dict) and "expr" in coor["c"]:
                        z = self._eval_series(df, {"expr": coor["c"]["expr"]})
                if z is not None:
                    z = z[np.isfinite(z)]
                    if z.size:
                        self.axc._cb["vmax"] = float(np.max(z))

        # ---- 2) Resolve/attach norm (priority: explicit 'norm' in style) ----
        if (self.axc._cb["vmin"] is not None) and (self.axc._cb["vmax"] is not None):
            vmin, vmax = self.axc._cb["vmin"], self.axc._cb["vmax"]

            def _resolve_norm(nv, *, vmin=None, vmax=None):
                """Turn user-specified 'norm' into a matplotlib.colors.Normalize instance."""
                if nv is None:
                    return None
                # Already a Normalize subclass
                if isinstance(nv, mcolors.Normalize):
                    return nv
                # String shorthand, e.g. "LogNorm", "TwoSlopeNorm", "Normalize"
                if isinstance(nv, str):
                    key = nv.strip().lower()
                    if key in {"log", "lognorm"}:
                        return mcolors.LogNorm(vmin=vmin, vmax=vmax)
                    if key in {"twoslopenorm", "diverging"}:
                        # default vcenter=0 for diverging data; user can override via dict form
                        return mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
                    if key in {"norm", "normalize", "linear"}:
                        return mcolors.Normalize(vmin=vmin, vmax=vmax)
                    # Fallback: unknown string → default linear
                    return mcolors.Normalize(vmin=vmin, vmax=vmax)
                # Dict form: {"type": "LogNorm", "vmin": ..., "vmax": ..., ...}
                if isinstance(nv, dict):
                    t = str(nv.get("type", "Normalize")).strip().lower()
                    # Prefer explicit vmin/vmax in dict; otherwise use inferred values
                    _vmin = nv.get("vmin", vmin)
                    _vmax = nv.get("vmax", vmax)
                    if t in {"log", "lognorm"}:
                        return mcolors.LogNorm(vmin=_vmin, vmax=_vmax)
                    if t in {"twoslopenorm", "diverging"}:
                        vcenter = nv.get("vcenter", 0.0)
                        return mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=_vmin, vmax=_vmax)
                    if t in {"symlog", "symlognorm"}:
                        # Optional parameters for SymLogNorm
                        linthresh = nv.get("linthresh", 1.0)
                        linscale = nv.get("linscale", 1.0)
                        base = nv.get("base", 10)
                        return mcolors.SymLogNorm(linthresh=linthresh, linscale=linscale, base=base, vmin=_vmin, vmax=_vmax)
                    # Default linear
                    return mcolors.Normalize(vmin=_vmin, vmax=_vmax)
                # Anything else → default linear
                return mcolors.Normalize(vmin=vmin, vmax=vmax)

            # Priority 1: explicit norm in style
            user_norm = style.get("norm", None)
            resolved = _resolve_norm(user_norm, vmin=vmin, vmax=vmax)

            # If no explicit norm provided, keep existing (if any) or use linear as default
            if resolved is None:
                if self.axc._cb["norm"] is None:
                    self.axc._cb["norm"] = mcolors.Normalize(vmin=vmin, vmax=vmax)
            else:
                self.axc._cb["norm"] = resolved

            # Also store the resolved norm back into the outgoing style so that plotting receives a proper Normalize
            if self.axc._cb["norm"] is not None:
                s["norm"] = self.axc._cb["norm"]

            # Set a backward-compatible 'mode' tag based on the resolved norm
            if isinstance(self.axc._cb["norm"], mcolors.LogNorm):
                self.axc._cb['mode'] = "log"
            elif isinstance(self.axc._cb["norm"], mcolors.TwoSlopeNorm):
                self.axc._cb['mode'] = "diverging"
            elif isinstance(self.axc._cb["norm"], mcolors.SymLogNorm):
                # Mark as 'log' to trigger log-style minor locators on colorbar if needed
                self.axc._cb['mode'] = "log"
            else:
                self.axc._cb['mode'] = "norm"

        if method_key in ("contour","contourf","tricontour","tricontourf") and self.axc._cb["levels"] is None:
            lv = s.get("levels", 10)
            if isinstance(lv, int) and self.axc._cb["vmin"] is not None and self.axc._cb["vmax"] is not None:
                self.axc._cb["levels"] = np.linspace(self.axc._cb["vmin"], self.axc._cb["vmax"], lv)
            elif hasattr(lv, "__len__"):
                self.axc._cb["levels"] = lv
        if 'norm' in s and s['norm'] is not None:
            s.pop('vmin', None)
            s.pop('vmax', None)
            s.pop('mode', None)
        self.axc._cb["used"] = uses_color
        return s





    def render_layer(self, ax, layer_info):
        """
        Render one layer on the given axes using METHOD_DISPATCH and the layer's
        data/coordinates/style fields assembled earlier in self.layers setter.
        This function now routes arguments based on the axes type:
          - ternary axes (ax._type == 'tri'): methods expect (a,b,c, ...)
              * profile_scatter additionally expects z -> (a,b,c,z, ...)
          - rectangular axes (ax._type == 'rect'): methods expect (x,y, ...)
        """
        # 1) Resolve method
        try: 
            self.logger.debug(f"Drawing layer -> {layer_info['name']}")
        except: 
            pass 
        method_key = str(layer_info.get("method", "scatter")).lower()
        method_name = self.METHOD_DISPATCH.get(method_key)
        if not method_name or not hasattr(ax, method_name):
            raise ValueError(f"Unknown/unsupported method '{method_key}' for axes {ax}.")
        method = getattr(ax, method_name)

        # 2) Merge style (bundle default -> layer override)
        style = dict(self.style.get(method_key, {}))
        if layer_info.get("style", {}) is not None: 
            style.update(layer_info.get("style", {}))

        if getattr(ax, "_type", None) == "tri":
            df = layer_info["data"]
            coor = layer_info.get("coor", {})

            # Apply per-figure shared colorbar (lazy) if an axc exists
            try:
                style = self._cb_collect_and_attach(style, coor, method_key, df)
                self.logger.debug("Successful loading colorbar style")
            except Exception as _e:
                self._logger.debug(f"colorbar lazy-attach failed: {_e}")
            # Ternary coordinates required: left/right/bottom
            requiredlbr = {"left", "right", "bottom"}
            requiredxy  = {"x", "y"}
            if not ((requiredlbr <= set(coor.keys())) or (requiredxy <= set(coor.keys()))):
                raise ValueError("Ternary layer must define coordinates: {left, right, bottom} or {x, y} with exprs.")
            for kk, vv in coor.items(): 
                style[kk] = self._eval_series(df, vv)
            return method(**style)

        elif getattr(ax, "_type", None) == "rect":
            df = layer_info["data"]
            coor = layer_info.get("coor", {})
            try:
                style = self._cb_collect_and_attach(style, coor, method_key, df)
                self.logger.debug("Successful loading colorbar style")
            except Exception as _e:
                self._logger.debug(f"colorbar lazy-attach failed: {_e}")
                
            if layer_info['method'] == "hist":
                if isinstance(layer_info['data'], dict): 
                    if "label" not in style.keys():
                        style['label'] = []
                    for kk, vv in coor.items():
                        style[kk] = []
                    for dn, ddf in df.items(): 
                        style['label'].append(dn)
                        for kk, vv in coor.items():
                            style[kk].append( self._eval_series(ddf, vv) )
                else: 
                    for kk, vv in coor.items(): 
                        style[kk] = self._eval_series(df, vv)
                        
                return method(**style)
            # Generic x/y coordinates required
            else:
                if not ({"x", "y"} <= set(coor.keys())):
                    raise ValueError("Rectangular layer must define coordinates: {x,y} with exprs.")

                for kk, vv in coor.items():
                    # style[kk] = self._eval_series(df, vv)
                    # Mode 1：expr → DataFrame evaluation
                    if isinstance(vv, dict) and "expr" in vv:
                        if df is None:
                            raise ValueError(
                                f"Layer '{layer_info.get('name', '')}' defines expression-based "
                                f"coordinate for '{kk}' but has no data source."
                            )
                        style[kk] = self._eval_series(df, vv)
                    else:
                        # Mode 2：(list/tuple/ndarray/scalar) 
                        style[kk] = vv
                # if "norm" in style.keys():
                #     style = self.load_norm(style)
                return method(**style)

        else:
            # Unknown axes adapter type
            raise ValueError(f"Axes '{ax}' has unknown _type='{getattr(ax, '_type', None)}'.")



    

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

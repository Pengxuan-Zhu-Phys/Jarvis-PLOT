from __future__ import annotations

import re
from typing import Any, Mapping

from matplotlib.ticker import AutoMinorLocator, LogLocator, ScalarFormatter


def is_numbered_ax(name: str) -> bool:
    return isinstance(name, str) and re.fullmatch(r"ax\d+", name) is not None


def ensure_numbered_rect_axes(fig, ax_name: str, kwgs: dict):
    if not is_numbered_ax(ax_name):
        raise ValueError(f"Illegal dynamic axes name '{ax_name}'. Only ax<NUMBER> is allowed.")

    if ax_name not in fig.axes.keys():
        raw_ax = fig.fig.add_axes(**kwgs)
        if isinstance(kwgs, dict) and ("facecolor" in kwgs):
            raw_ax.set_facecolor(kwgs["facecolor"])
        from .adapters_rect import StdAxesAdapter

        adapter = StdAxesAdapter(raw_ax)
        adapter._type = "rect"
        adapter.layers = []
        adapter._legend = fig.frame.get(ax_name, {}).get("legend", False)
        fig.axes[ax_name] = adapter
        adapter.status = "configured"

    ax_obj = fig.axes[ax_name]

    if fig.frame.get(ax_name, {}).get("spines"):
        if "color" in fig.frame[ax_name]["spines"]:
            for s in ax_obj.spines.values():
                s.set_color(fig.frame[ax_name]["spines"]["color"])

    if fig.frame.get(ax_name, {}).get("yscale", "").lower() == "log":
        ax_obj.set_yscale("log")
        ax_obj.yaxis.set_minor_locator(LogLocator(subs="auto"))
    else:
        ax_obj.yaxis.set_minor_locator(AutoMinorLocator())

    if fig.frame.get(ax_name, {}).get("xscale", "").lower() == "log":
        ax_obj.set_xscale("log")
        ax_obj.xaxis.set_minor_locator(LogLocator(subs="auto"))
    else:
        ax_obj.xaxis.set_minor_locator(AutoMinorLocator())

    def _safe_cast(v):
        try:
            return float(v)
        except Exception:
            return v

    if fig.frame.get(ax_name, {}).get("text"):
        for txt in fig.frame[ax_name]["text"]:
            if txt.get("transform", False):
                txt.pop("transform")
                ax_obj.text(**txt, transform=ax_obj.transAxes)
            else:
                ax_obj.text(**txt)

    xlim = fig.frame.get(ax_name, {}).get("xlim")
    if xlim:
        ax_obj.set_xlim(list(map(_safe_cast, xlim)))

    ylim = fig.frame.get(ax_name, {}).get("ylim")
    if ylim:
        ax_obj.set_ylim(list(map(_safe_cast, ylim)))

    if fig.frame.get(ax_name, {}).get("labels", {}).get("x"):
        ax_obj.set_xlabel(fig.frame[ax_name]["labels"]["x"], **fig.frame[ax_name]["labels"]["xlabel"])
    if fig.frame.get(ax_name, {}).get("labels", {}).get("y"):
        ax_obj.set_ylabel(fig.frame[ax_name]["labels"]["y"], **fig.frame[ax_name]["labels"]["ylabel"])

    ax_obj.tick_params(**fig.frame.get(ax_name, {}).get("ticks", {}).get("both", {}))
    ax_obj.tick_params(**fig.frame.get(ax_name, {}).get("ticks", {}).get("major", {}))
    ax_obj.tick_params(**fig.frame.get(ax_name, {}).get("ticks", {}).get("minor", {}))

    apply_axis_endpoints(fig, ax_obj, fig.frame.get(ax_name, {}).get("xaxis", {}), "x")
    apply_axis_endpoints(fig, ax_obj, fig.frame.get(ax_name, {}).get("yaxis", {}), "y")

    if getattr(ax_obj, "needs_finalize", True) and hasattr(ax_obj, "finalize"):
        try:
            ax_obj.finalize()
        except Exception as e:
            if fig.logger:
                fig.logger.warning(f"Finalize failed on axes '{ax_name}': {e}")

    try:
        fig.logger.debug(f"Loaded numbered rectangle axes -> {ax_name}")
    except Exception:
        pass

    return ax_obj


def has_manual_ticks(frame: Mapping[str, Any], ax_key: str, which: str) -> bool:
    try:
        if ax_key == "ax":
            ticks_cfg = frame.get("ax", {}).get("ticks", {})
        elif ax_key == "axc":
            ticks_cfg = frame.get("axc", {}).get("ticks", {})
        else:
            return False
        node = ticks_cfg.get(which, {})
        return isinstance(node, dict) and ((node.get("positions") is not None) or (node.get("pos") is not None))
    except Exception:
        return False


def apply_axis_endpoints(fig, ax_obj, axis_cfg: dict, which: str):
    if not isinstance(axis_cfg, dict):
        return

    target = ax_obj.ax if hasattr(ax_obj, "ax") else ax_obj

    if which == "x":
        ticks = target.xaxis.get_major_ticks()
        locs = target.xaxis.get_majorticklocs()
    else:
        ticks = target.yaxis.get_major_ticks()
        locs = target.yaxis.get_majorticklocs()
    if not ticks:
        return

    if which == "x":
        lim0, lim1 = target.get_xlim()
    else:
        lim0, lim1 = target.get_ylim()

    min_cfg = axis_cfg.get("min_endpoints", {})
    max_cfg = axis_cfg.get("max_endpoints", {})
    width = abs(lim0 - lim1)

    t0 = ticks[0]
    t0_loc = locs[0]
    if abs(t0_loc - lim0) < 1e-3 * width:
        if min_cfg.get("tick") is False:
            t0.tick1line.set_visible(False)
            t0.tick2line.set_visible(False)
        if min_cfg.get("label") is False:
            t0.label1.set_visible(False)
            t0.label2.set_visible(False)

    t1 = ticks[-1]
    t1_loc = locs[-1]
    if abs(t1_loc - lim1) < 1e-3 * width:
        if max_cfg.get("tick") is False:
            t1.tick1line.set_visible(False)
            t1.tick2line.set_visible(False)
        if max_cfg.get("label") is False:
            t1.label1.set_visible(False)
            t1.label2.set_visible(False)


def apply_auto_ticks(ax_obj, which: str):
    target = ax_obj.ax if hasattr(ax_obj, "ax") else ax_obj
    axis = target.xaxis if which == "x" else target.yaxis

    try:
        labels = axis.get_ticklabels()
        if which == "x":
            xscale = target.get_xscale()
            if xscale not in ("log", "symlog", "logit"):
                fmt = ScalarFormatter(useMathText=True)
                fmt.set_powerlimits((-3, 4))
                axis.set_major_formatter(fmt)
                try:
                    target.ticklabel_format(style="sci", axis="x", scilimits=(-3, 4))
                except Exception:
                    pass
                try:
                    axis.set_offset_position("bottom")
                except Exception:
                    pass
                target.figure.canvas.draw_idle()
                tl = axis.get_ticklabels()
                if tl:
                    axis.offsetText.set_fontsize(tl[0].get_size() * 0.8)
                axis.offsetText.set_horizontalalignment("left")
                axis.offsetText.set_x(1.02)
            return

        if which == "y":
            yscale = target.get_yscale()
            if yscale in ("log", "symlog", "logit"):
                from matplotlib.ticker import LogFormatterMathtext, FuncFormatter

                base = LogFormatterMathtext()
                lo, hi = 1e-2, 1e2

                def _fmt(val, pos=None):
                    if val <= 0:
                        return ""
                    if lo <= val <= hi:
                        if abs(val - round(val)) < 1e-10:
                            return f"{int(round(val))}"
                        return f"{val:.3g}"
                    return base(val, pos)

                axis.set_major_formatter(FuncFormatter(_fmt))
            else:
                fmt = ScalarFormatter(useMathText=True)
                fmt.set_powerlimits((-3, 4))
                axis.set_major_formatter(fmt)
    except Exception:
        return


def apply_manual_ticks(fig, ax_obj, which: str, ticks_cfg: dict):
    pos = ticks_cfg.get("positions", ticks_cfg.get("pos", None))
    labs = ticks_cfg.get("labels", ticks_cfg.get("labs", None))
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
        if fig.logger:
            fig.logger.warning(f"Manual ticks apply failed on {which}-axis: {e}")

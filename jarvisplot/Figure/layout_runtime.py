from __future__ import annotations

import re
from typing import Any, Mapping

from matplotlib.ticker import AutoMinorLocator, LogLocator, NullLocator, ScalarFormatter


def is_numbered_ax(name: str) -> bool:
    return isinstance(name, str) and re.fullmatch(r"ax\d+", name) is not None


#: Side panels a card may attach to the main axes: a ratio strip underneath,
#: and the left / bottom marginal panels of a corner-plot layout.  Each is a
#: full rectangular axes and reads its own ``Frame`` node.
SIDE_AXES = ("axr", "axl", "axb")

#: The correlation matrix owns a panel and a colorbar of its own, and both
#: names are reserved.  A correlation matrix is not a matplotlib primitive --
#: it is a categorical grid whose two axes carry variable names in an order
#: the data decides -- so dropping one onto a general panel would not fail,
#: it would draw a plausible-looking wrong figure.  Giving it names no other
#: card declares makes that impossible to express rather than merely
#: discouraged.
CORR_AXES = "axcorr"
CORR_COLORBAR = "axccorr"


def is_corr_ax(name) -> bool:
    """True for the reserved correlation-matrix panel."""
    return name == CORR_AXES


def is_colorbar_ax(name) -> bool:
    """True for ``axc`` and the named secondary colorbars (``axc2``, ``axccorr``).

    ``axcorr`` also starts with ``axc``, so the matrix panel has to be
    excluded by name: every caller of this rule routes its matches into the
    colorbar machinery, which would swallow the panel whole.
    """
    if not isinstance(name, str) or is_corr_ax(name):
        return False
    return name == "axc" or (name.startswith("axc") and len(name) > 3)


def is_rect_ax(name: str) -> bool:
    """Return True for dynamically supported rectangular axes."""
    return is_numbered_ax(name) or name in SIDE_AXES


_TITLE_POSITION_PARAMS = {
    # `top` is the historical top-left placement used by the cards.
    "top": {"x": 0.005, "y": 1.0, "ha": "left", "va": "bottom"},
    "center": {"x": 0.5, "y": 1.0, "ha": "center", "va": "bottom"},
    "right": {"x": 0.995, "y": 1.0, "ha": "right", "va": "bottom"},
}


def apply_axis_title(fig, ax_obj, ax_name: str) -> None:
    """Render the configured title, with numbered-panel ownership rules."""
    axis_cfg = fig.frame.get(ax_name, {})
    if not isinstance(axis_cfg, dict) or "title" not in axis_cfg:
        return

    if is_numbered_ax(ax_name) and ax_name != "ax0":
        if getattr(fig, "logger", None):
            fig.logger.warning(
                f"Ignoring title for axes '{ax_name}': title rendering is only supported on 'ax0'."
            )
        return

    title = axis_cfg.get("title")
    if title is None or str(title) == "":
        return

    title_params = dict(axis_cfg.get("title_params") or {})
    position = str(title_params.pop("position", "top")).strip().lower()
    if position not in _TITLE_POSITION_PARAMS:
        if getattr(fig, "logger", None):
            fig.logger.warning(
                f"Unknown title position '{position}' for axes '{ax_name}'; using 'top'."
            )
        position = "top"

    legacy_position_params = [key for key in ("x", "y", "ha", "va") if key in title_params]
    if legacy_position_params and getattr(fig, "logger", None):
        fig.logger.warning(
            f"Ignoring title_params {legacy_position_params} for axes '{ax_name}'; use position instead."
        )
    for key in ("x", "y", "ha", "va"):
        title_params.pop(key, None)

    text_params = dict(_TITLE_POSITION_PARAMS[position])
    text_params.update(title_params)
    # The title text is owned by frame.<axis>.title; do not let an
    # accidental `s` value in title_params override it.
    text_params.pop("s", None)
    text_params["transform"] = ax_obj.transAxes
    ax_obj.text(s=str(title), **text_params)


def hide_log_minor_tick_labels(ax_obj, which: str) -> None:
    """Keep log minor tick marks, but hide their labels by default."""
    target = ax_obj.ax if hasattr(ax_obj, "ax") else ax_obj
    if which == "x":
        target.tick_params(
            axis="x", which="minor", labelbottom=False, labeltop=False
        )
    elif which == "y":
        target.tick_params(
            axis="y", which="minor", labelleft=False, labelright=False
        )


def apply_grid(fig, ax_obj, ax_name: str) -> None:
    """Draw the configured major / minor grid on a rectangular axes.

    Shaped like ``ticks``: one block per matplotlib call, splatted verbatim
    into ``Axes.grid``.  Ternary cards keep their own hand-drawn grid under
    ``grid.sep`` / ``grid.style``, so only the two call blocks are read here
    and every other key in the node is left alone -- which is also why the
    dead ``grid`` blocks the rect cards inherited stay dead.

    ``axisbelow`` is a sibling key rather than a ``grid`` kwarg because it is
    the only thing that decides whether the grid lands over or under the data:
    a ``zorder`` passed to ``Axes.grid`` sets the gridline's own zorder, but
    the Axis artist is drawn as a unit at the zorder ``set_axisbelow`` gives
    it, so the kwarg alone never lifts a grid above a filled layer.
    """
    grid_cfg = fig.frame.get(ax_name, {}).get("grid", {})
    if not isinstance(grid_cfg, dict):
        return
    target = ax_obj.ax if hasattr(ax_obj, "ax") else ax_obj

    if "axisbelow" in grid_cfg:
        target.set_axisbelow(grid_cfg["axisbelow"])

    for which in ("major", "minor"):
        block = grid_cfg.get(which)
        if not isinstance(block, dict) or not block:
            continue
        try:
            target.grid(**block)
        except Exception as exc:
            if getattr(fig, "logger", None):
                fig.logger.warning(
                    f"grid.{which} on axes '{ax_name}' ignored: {exc}"
                )


def ensure_rect_axes(fig, ax_name: str, kwgs: dict):
    if not is_rect_ax(ax_name):
        allowed = ", ".join(repr(n) for n in SIDE_AXES)
        raise ValueError(
            f"Illegal dynamic axes name '{ax_name}'. Only {allowed} or ax<NUMBER> is allowed."
        )

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
        hide_log_minor_tick_labels(ax_obj, "y")
    else:
        ax_obj.yaxis.set_minor_locator(AutoMinorLocator())

    if fig.frame.get(ax_name, {}).get("xscale", "").lower() == "log":
        ax_obj.set_xscale("log")
        ax_obj.xaxis.set_minor_locator(LogLocator(subs="auto"))
        hide_log_minor_tick_labels(ax_obj, "x")
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

    apply_axis_title(fig, ax_obj, ax_name)

    xlim = fig.frame.get(ax_name, {}).get("xlim")
    if xlim:
        ax_obj.set_xlim(list(map(_safe_cast, xlim)))

    ylim = fig.frame.get(ax_name, {}).get("ylim")
    if ylim:
        ax_obj.set_ylim(list(map(_safe_cast, ylim)))

    labels_cfg = fig.frame.get(ax_name, {}).get("labels", {})
    if labels_cfg.get("x"):
        ax_obj.set_xlabel(labels_cfg["x"], **labels_cfg.get("xlabel", {}))
    if labels_cfg.get("y"):
        ax_obj.set_ylabel(labels_cfg["y"], **labels_cfg.get("ylabel", {}))
        ylabel_coords = labels_cfg.get("ylabel_coords")
        if isinstance(ylabel_coords, dict) and {"x", "y"} <= set(ylabel_coords):
            ax_obj.yaxis.set_label_coords(ylabel_coords["x"], ylabel_coords["y"])

    ticks_cfg = fig.frame.get(ax_name, {}).get("ticks", {})
    ax_obj.tick_params(**ticks_cfg.get("both", {}))
    ax_obj.tick_params(**ticks_cfg.get("major", {}))
    ax_obj.tick_params(**ticks_cfg.get("minor", {}))

    # `ticks.x/y.positions` was honoured on ax and on the colorbars but never
    # here, so a side or numbered axes silently kept matplotlib's automatic
    # locator -- even though has_manual_ticks already reads this same node.
    apply_manual_ticks(fig, ax_obj, "x", ticks_cfg.get("x", {}) or {})
    apply_manual_ticks(fig, ax_obj, "y", ticks_cfg.get("y", {}) or {})
    apply_tick_label_props(fig, ax_obj, "x", ticks_cfg.get("x", {}) or {})
    apply_tick_label_props(fig, ax_obj, "y", ticks_cfg.get("y", {}) or {})

    apply_grid(fig, ax_obj, ax_name)

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


def ensure_numbered_rect_axes(fig, ax_name: str, kwgs: dict):
    """Backward-compatible wrapper for numbered rectangular axes."""
    if not is_numbered_ax(ax_name):
        raise ValueError(f"Illegal numbered axes name '{ax_name}'. Only ax<NUMBER> is allowed.")
    return ensure_rect_axes(fig, ax_name, kwgs)


#: Everything ``axcorr`` reads out of its ``Frame`` node.  A card key outside
#: this set is a mistake worth reporting rather than a setting that quietly
#: does nothing -- which is what the general ``ax`` path does with the keys it
#: happens not to consume.
CORR_FRAME_KEYS = frozenset(
    {"rect", "spines", "xlim", "ylim", "ticks", "grid", "title", "title_params"}
)


def ensure_corr_axes(fig, kwgs: dict):
    """Create and configure the reserved correlation-matrix panel.

    Deliberately narrower than the general ``ax`` path.  A correlation matrix
    is a categorical grid: both axes are variable names at integer positions,
    so there is no log scale to switch, no minor locator to place, no axis
    label to centre and no endpoint tick to trim.  Supporting those here would
    only create ways to configure a matrix into something that is no longer a
    matrix.
    """
    from .adapters_rect import StdAxesAdapter

    name = CORR_AXES
    node = fig.frame.get(name, {}) or {}

    unknown = sorted(set(node) - CORR_FRAME_KEYS)
    if unknown and getattr(fig, "logger", None):
        fig.logger.warning(
            "frame.{} ignores {}: the correlation panel reads only {}.".format(
                name, unknown, ", ".join(sorted(CORR_FRAME_KEYS))
            )
        )

    if name not in fig.axes:
        if not (isinstance(kwgs, Mapping) and kwgs.get("rect")):
            raise ValueError(
                "frame.axes.{} has no rect. This card solves its geometry from "
                "the matrix rather than carrying a fixed one, so the figure has "
                "to be rendered through `type: correlation_matrix`, which runs "
                "the solve. Nothing else can supply a sensible rect here."
                .format(name)
            )
        raw_ax = fig.fig.add_axes(**kwgs)
        if "facecolor" in kwgs:
            raw_ax.set_facecolor(kwgs["facecolor"])
        adapter = StdAxesAdapter(raw_ax)
        adapter._type = "rect"
        adapter.layers = []
        adapter._legend = False
        fig.axes[name] = adapter
        adapter.status = "configured"

    ax_obj = fig.axes[name]
    target = ax_obj.ax if hasattr(ax_obj, "ax") else ax_obj

    spines = node.get("spines") or {}
    for key, setter in (("color", "set_color"), ("linewidth", "set_linewidth")):
        if spines.get(key) is not None:
            for spine in target.spines.values():
                getattr(spine, setter)(spines[key])

    for key, setter in (("xlim", target.set_xlim), ("ylim", target.set_ylim)):
        lim = node.get(key)
        if lim:
            setter([float(v) for v in lim])

    # A categorical axis has nowhere to put a minor tick: AutoMinorLocator
    # would drop them between the variable names.
    target.xaxis.set_minor_locator(NullLocator())
    target.yaxis.set_minor_locator(NullLocator())

    apply_axis_title(fig, ax_obj, name)

    ticks_cfg = node.get("ticks", {}) or {}
    apply_manual_ticks(fig, ax_obj, "x", ticks_cfg.get("x", {}) or {})
    apply_manual_ticks(fig, ax_obj, "y", ticks_cfg.get("y", {}) or {})
    target.tick_params(**ticks_cfg.get("both", {}))
    target.tick_params(**ticks_cfg.get("major", {}))
    apply_tick_label_props(fig, ax_obj, "x", ticks_cfg.get("x", {}) or {})
    apply_tick_label_props(fig, ax_obj, "y", ticks_cfg.get("y", {}) or {})

    apply_grid(fig, ax_obj, name)

    if getattr(ax_obj, "needs_finalize", True) and hasattr(ax_obj, "finalize"):
        try:
            ax_obj.finalize()
        except Exception as e:
            if getattr(fig, "logger", None):
                fig.logger.warning(f"Finalize failed on axes '{name}': {e}")

    try:
        fig.logger.debug(f"Loaded correlation matrix axes -> {name}")
    except Exception:
        pass

    return ax_obj


def has_manual_ticks(frame: Mapping[str, Any], ax_key: str, which: str) -> bool:
    try:
        if ax_key == "ax" or ax_key in SIDE_AXES or is_corr_ax(ax_key):
            # Each side panel reads its own node, so a later auto-tick pass
            # cannot hand it the main axes' positions.  The correlation panel
            # is here for the same reason: its ticks are variable names, and
            # a formatter would replace them with numbers.
            ticks_cfg = frame.get(ax_key, {}).get("ticks", {})
        elif is_colorbar_ax(ax_key):
            # Colorbars can be named axc, axc2, etc.  Each must read its
            # own frame node so a later auto-tick pass cannot replace YAML
            # positions/labels with a Matplotlib formatter.
            ticks_cfg = frame.get(ax_key, {}).get("ticks", {})
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


#: Tick-label text properties a per-axis ``ticks.x`` / ``ticks.y`` node may
#: set, mapped to the ``Text`` property each one becomes.  The card's
#: ``both`` / ``major`` / ``minor`` blocks go through ``tick_params``, which
#: reaches *both* axes at once, so an axis carrying names rather than numbers
#: -- rotated on x, horizontal on y -- has nowhere else to say so.  These sit
#: next to ``positions`` because they describe the same labels.
TICK_LABEL_PROPS = {
    "labelrotation": "rotation",
    "labelrotation_mode": "rotation_mode",
    "labelha": "ha",
    "labelva": "va",
    "labelcolor": "color",
    "labelsize": "fontsize",
    "labelfamily": "fontfamily",
}


def tick_label_props(ticks_cfg: Mapping[str, Any]) -> dict:
    """Text properties declared on one axis' tick node."""
    if not isinstance(ticks_cfg, Mapping):
        return {}
    return {
        prop: ticks_cfg[key]
        for key, prop in TICK_LABEL_PROPS.items()
        if ticks_cfg.get(key) is not None
    }


def apply_tick_label_props(fig, ax_obj, which: str, ticks_cfg: dict) -> None:
    """Style one axis' tick labels.

    Must run *after* the ``both`` / ``major`` / ``minor`` blocks: ``tick_params``
    rewrites each label from the axis' stored parameters, so a rotation set
    before it is silently reset to zero.
    """
    props = tick_label_props(ticks_cfg)
    if not props:
        return
    target = ax_obj.ax if hasattr(ax_obj, "ax") else ax_obj
    try:
        artists = target.get_xticklabels() if which == "x" else target.get_yticklabels()
        for artist in artists:
            artist.set(**props)
    except Exception as e:
        if getattr(fig, "logger", None):
            fig.logger.warning(f"Tick label props failed on {which}-axis: {e}")


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

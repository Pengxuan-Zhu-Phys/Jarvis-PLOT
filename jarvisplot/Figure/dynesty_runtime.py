from __future__ import annotations

from collections.abc import Mapping, Sequence
import warnings

import numpy as np
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter
from scipy.stats import gaussian_kde


_SQRTEPS = np.sqrt(np.finfo(float).eps)


_DEFAULT_COLUMNS = {
    "logvol": ("log_PriorVolume", "logvol"),
    "logl": ("log_Like", "logl"),
    "logwt": ("log_weight", "logwt"),
    "logz": ("log_Evidence", "logz"),
    "logzerr": ("log_Evidence_err", "logzerr"),
    "nlive": ("samples_nlive", "samples_n", "nlive"),
    "iters": ("samples_it", "samples_iter", "it"),
}


_RUNPLOT_CONFIG_KEYS = {
    "axes",
    "panels",
    "columns",
    "span",
    "logplot",
    "kde",
    "nkde",
    "seed",
    "lnz_error",
    "lnz_error_levels",
    "lnz_truth",
    "evidence_summary",
    "evidence_summary_fmt",
    "evidence_summary_line_kwargs",
    "evidence_summary_text_kwargs",
    "truth_color",
    "truth_kwargs",
    "label_kwargs",
    "plot_kwargs",
    "fill_kwargs",
    "final_live_kwargs",
    "max_x_ticks",
    "max_y_ticks",
    "ylim_factor",
    "use_math_text",
    "mark_final_live",
    "scatter",
    "scatter_panels",
    "scatter_kwargs",
    "overlay_scatter",
    "overlay_scatter_panels",
    "overlay_scatter_kwargs",
    "importance_scatter_normalize",
    "set_labels",
}


def _raw_axes(ax):
    return ax.ax if hasattr(ax, "ax") else ax


def _rng(seed=None):
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.Generator(np.random.PCG64(seed))


def _resample_equal(samples, weights, rstate=None):
    if rstate is None:
        rstate = _rng()
    samples = np.asarray(samples)
    weights = np.asarray(weights, dtype=float)
    cumulative_sum = np.cumsum(weights)
    if (
        cumulative_sum.size == 0
        or not np.isfinite(cumulative_sum[-1])
        or cumulative_sum[-1] <= 0
    ):
        raise ValueError("dynesty_runplot requires at least one positive finite importance weight")
    if abs(cumulative_sum[-1] - 1.0) > _SQRTEPS:
        warnings.warn("Weights do not sum to 1 and have been renormalized.")
    cumulative_sum /= cumulative_sum[-1]

    nsamples = len(weights)
    positions = (rstate.random() + np.arange(nsamples)) / nsamples
    idx = np.zeros(nsamples, dtype=int)
    i, j = 0, 0
    while i < nsamples:
        if positions[i] < cumulative_sum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    return rstate.permutation(samples[idx])


def _first_existing_column(df, names, logical_name):
    for name in names:
        if name in df:
            return name
    raise KeyError(
        "dynesty_runplot could not find a column for '{}'. Tried: {}".format(
            logical_name, ", ".join(str(n) for n in names)
        )
    )


def _column_array(df, columns, key, *, required=True, default=None):
    raw = columns.get(key, None) if isinstance(columns, Mapping) else None
    if raw is None:
        candidates = _DEFAULT_COLUMNS[key]
    elif isinstance(raw, str):
        candidates = (raw,)
    else:
        candidates = tuple(raw)

    try:
        name = _first_existing_column(df, candidates, key)
    except KeyError:
        if required:
            raise
        return default
    return np.asarray(df[name], dtype=float)


def _as_list(value, default):
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return list(default)


def _as_panel_set(value, default=()):
    if value is True:
        return set(default)
    if value in (None, False):
        return set()
    return set(str(item).strip().lower() for item in _as_list(value, default))


def _as_error_levels(value):
    if value is None:
        return (1, 2, 3)
    if isinstance(value, (int, float)):
        levels = [int(value)]
    else:
        levels = []
        for item in _as_list(value, ()):
            try:
                levels.append(int(item))
            except Exception:
                pass
    return tuple(level for level in levels if level > 0) or (1,)


def _evidence_value_and_error(logz_value, logzerr_value, *, logplot: bool):
    logz_value = float(logz_value)
    try:
        logzerr_value = float(logzerr_value)
    except Exception:
        logzerr_value = 0.0
    if not np.isfinite(logzerr_value):
        logzerr_value = 0.0

    if logplot:
        return logz_value, logzerr_value

    value = float(np.exp(logz_value))
    err_hi = float(np.exp(logz_value + logzerr_value) - value)
    err_lo = float(value - np.exp(logz_value - logzerr_value))
    error = 0.5 * (err_hi + err_lo)
    return value, error


def _format_evidence_summary(value, error, fmt):
    if not fmt:
        fmt = "{value:.3g} ± {error:.2g}"
    try:
        return str(fmt).format(value=value, error=error)
    except Exception:
        return f"{value:.3g} ± {error:.2g}"


def _line_kwargs(style):
    out = {}
    out.update(
        style.get("plot_kwargs", {})
        if isinstance(style.get("plot_kwargs"), Mapping)
        else {}
    )
    for key, value in style.items():
        if key not in _RUNPLOT_CONFIG_KEYS:
            out[key] = value
    out.setdefault("linewidth", 5)
    out.setdefault("alpha", 0.7)
    return out


def _scatter_kwargs(style):
    out = {"s": 3, "marker": ".", "alpha": 0.45}
    cfg = style.get("scatter_kwargs", {})
    if isinstance(cfg, Mapping):
        out.update(cfg)
    return out


def _overlay_scatter_kwargs(style):
    out = _scatter_kwargs(style)
    out["alpha"] = 0.35
    out["zorder"] = 1
    cfg = style.get("overlay_scatter_kwargs", {})
    if isinstance(cfg, Mapping):
        out.update(cfg)
    return out


def _resolve_axes(fig, style):
    names = _as_list(style.get("axes"), ("ax0", "ax1", "ax2", "ax3"))
    axes = []
    for name in names:
        if name not in fig.axes:
            raise KeyError(f"dynesty_runplot requested missing axes '{name}'")
        axes.append(_raw_axes(fig.axes[name]))
    if len(axes) < 4:
        raise ValueError("dynesty_runplot requires at least four axes")
    return names, axes


def _prepare_arrays(df, style):
    columns = style.get("columns", {})
    logvol = _column_array(df, columns, "logvol")
    logl = _column_array(df, columns, "logl")
    logwt_raw = _column_array(df, columns, "logwt")
    logz = _column_array(df, columns, "logz")
    logzerr = _column_array(
        df, columns, "logzerr", required=False, default=np.zeros_like(logz)
    )
    nlive = _column_array(
        df, columns, "nlive", required=False, default=np.ones_like(logvol)
    )
    iters = _column_array(
        df,
        columns,
        "iters",
        required=False,
        default=np.arange(len(logvol), dtype=float),
    )

    size = min(
        *(len(a) for a in (logvol, logl, logwt_raw, logz, logzerr, nlive, iters))
    )
    arrays = [
        np.asarray(a[:size], dtype=float)
        for a in (logvol, logl, logwt_raw, logz, logzerr, nlive, iters)
    ]
    mask = np.ones(size, dtype=bool)
    for arr in arrays[:5]:
        mask &= np.isfinite(arr)
    for arr in arrays[5:]:
        mask &= np.isfinite(arr)
    if not mask.any():
        raise ValueError("dynesty_runplot has no finite rows to plot")
    logvol, logl, logwt_raw, logz, logzerr, nlive, iters = [arr[mask] for arr in arrays]
    logzerr = np.asarray(logzerr, dtype=float)
    logzerr[~np.isfinite(logzerr)] = 0.0
    return logvol, logl, logwt_raw, logz, logzerr, nlive, iters


def _positive_ymax(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    ymax = float(np.nanmax(arr))
    if not np.isfinite(ymax) or ymax <= 0:
        return 1.0
    return ymax


def _span_from_panels(panels, data_by_panel, span, *, logplot, logz, logzerr, lnz_error, ylim_factor):
    if span is None:
        out = []
        for panel in panels:
            if panel not in data_by_panel:
                continue
            ydata = data_by_panel[panel][1]
            if panel == "evidence" and lnz_error:
                if logplot:
                    ymax = _positive_ymax(logz + 3.0 * logzerr)
                else:
                    ymax = _positive_ymax(np.exp(logz + 3.0 * logzerr))
            else:
                ymax = _positive_ymax(ydata)
            out.append((0.0, float(ylim_factor) * ymax))
        return out

    out = list(span)
    panel_ydata = [data_by_panel[p][1] for p in panels if p in data_by_panel]
    if len(out) < len(panel_ydata):
        out.extend(
            [
                (0.0, float(ylim_factor) * _positive_ymax(d))
                for d in panel_ydata[len(out):]
            ]
        )
    for i, item in enumerate(out):
        try:
            if len(item) != 2:
                raise ValueError("Incorrect span value")
        except TypeError:
            ymax = _positive_ymax(panel_ydata[i])
            out[i] = (
                ymax * float(item),
                ymax,
            )
    return out


def render_dynesty_runplot(fig, df, style):
    """Render dynesty runplot-style panels from a Jarvis-HEP dynesty CSV."""
    style = dict(style or {})
    axes_names, axes = _resolve_axes(fig, style)
    panels = _as_list(style.get("panels"), ("nlive", "likelihood", "importance", "evidence"))
    if len(axes) >= 5 and len(panels) == 4:
        panels.append("iters")
    panels = panels[: len(axes)]

    logplot = bool(style.get("logplot", False))
    kde = bool(style.get("kde", True))
    nkde = int(style.get("nkde", 1000))
    lnz_error = bool(style.get("lnz_error", True))
    lnz_error_levels = _as_error_levels(style.get("lnz_error_levels", None))
    max_x_ticks = int(style.get("max_x_ticks", 8))
    max_y_ticks = int(style.get("max_y_ticks", 3))
    ylim_factor = float(style.get("ylim_factor", 1.05))
    use_math_text = bool(style.get("use_math_text", True))
    scatter_only_panels = _as_panel_set(style.get("scatter_panels", ()))
    if style.get("scatter", False):
        scatter_only_panels = set(panels)
    overlay_scatter_panels = _as_panel_set(
        style.get("overlay_scatter_panels", style.get("overlay_scatter", False)),
        default=panels,
    )
    line_kwargs = _line_kwargs(style)
    color = line_kwargs.pop("color", line_kwargs.pop("c", "blue"))
    fill_kwargs = dict(
        style.get("fill_kwargs", {})
        if isinstance(style.get("fill_kwargs"), Mapping)
        else {}
    )
    fill_kwargs.setdefault("alpha", 0.2)
    evidence_summary = bool(style.get("evidence_summary", False))
    evidence_summary_line_kwargs = dict(
        style.get("evidence_summary_line_kwargs", {})
        if isinstance(style.get("evidence_summary_line_kwargs"), Mapping)
        else {}
    )
    evidence_summary_line_kwargs.setdefault("color", "#8a8a8a")
    evidence_summary_line_kwargs.setdefault("linewidth", 0.8)
    evidence_summary_line_kwargs.setdefault("alpha", 0.55)
    evidence_summary_line_kwargs.setdefault("zorder", 2)
    evidence_summary_text_kwargs = dict(
        style.get("evidence_summary_text_kwargs", {})
        if isinstance(style.get("evidence_summary_text_kwargs"), Mapping)
        else {}
    )
    evidence_text_x = float(evidence_summary_text_kwargs.pop("x", 0.04))
    evidence_text_offset = evidence_summary_text_kwargs.pop("xytext", (0, -8))
    evidence_summary_text_kwargs.setdefault("ha", "left")
    evidence_summary_text_kwargs.setdefault("va", "top")
    evidence_summary_text_kwargs.setdefault("fontsize", 7)
    evidence_summary_text_kwargs.setdefault("fontfamily", "STIXGeneral")
    evidence_summary_text_kwargs.setdefault("color", "black")
    truth_kwargs = dict(
        style.get("truth_kwargs", {})
        if isinstance(style.get("truth_kwargs"), Mapping)
        else {}
    )
    truth_kwargs.setdefault("linestyle", "solid")
    truth_kwargs.setdefault("linewidth", 3)

    logvol, logl_raw, logwt_raw, logz, logzerr, nlive, iters = _prepare_arrays(df, style)
    logl = logl_raw - np.nanmax(logl_raw)
    logwt = logwt_raw - logz[-1]
    weights = np.exp(logwt)
    x = -logvol

    importance_x = x
    importance_y = weights
    if kde and len(logvol) > 1 and np.count_nonzero(weights > 0) > 1:
        rstate = _rng(style.get("seed", None))
        equal_samples = _resample_equal(-logvol, weights, rstate=rstate)
        wt_kde = gaussian_kde(equal_samples)
        logvol_new = np.linspace(logvol[0], logvol[-1], nkde)
        importance_x = -logvol_new
        importance_y = wt_kde.pdf(-logvol_new)

    data_by_panel = {
        "nlive": (x, nlive, "Live Points"),
        "likelihood": (x, np.exp(logl), "Likelihood\n(normalized)"),
        "importance": (
            importance_x,
            importance_y,
            "Importance\nWeight PDF" if kde else "Importance\nWeight",
        ),
        "evidence": (
            x,
            logz if logplot else np.exp(logz),
            "log(Evidence)" if logplot else "Evidence",
        ),
        "iters": (x, iters, "Iters"),
    }
    importance_scatter_y = weights
    importance_scatter_normalize = str(
        style.get("importance_scatter_normalize", "")
    ).strip().lower()
    if importance_scatter_normalize in {"max", "peak", "ymax", "pdf_peak"}:
        importance_scatter_y = np.exp(logwt_raw - np.nanmax(logwt_raw))
        if importance_scatter_normalize == "pdf_peak":
            pdf_peak = np.nanmax(importance_y)
            if np.isfinite(pdf_peak) and pdf_peak > 0:
                importance_scatter_y = importance_scatter_y * pdf_peak
    raw_data_by_panel = {
        "nlive": (x, nlive),
        "likelihood": (x, np.exp(logl)),
        "importance": (x, importance_scatter_y),
        "evidence": (x, logz if logplot else np.exp(logz)),
        "iters": (x, iters),
    }

    span = _span_from_panels(
        panels,
        data_by_panel,
        style.get("span", None),
        logplot=logplot,
        logz=logz,
        logzerr=logzerr,
        lnz_error=lnz_error and "evidence" in panels,
        ylim_factor=ylim_factor,
    )

    xmax = float(np.nanmax(x))
    artists = []
    for i, panel in enumerate(panels):
        if panel not in data_by_panel:
            continue
        ax = axes[i]
        px, py, ylabel = data_by_panel[panel]

        ax.set_xlim([min(0.0, ax.get_xlim()[0]), max(xmax, ax.get_xlim()[1])])
        if i < len(span):
            ymin, ymax = span[i]
            if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
                ax.set_ylim([ymin, ymax])

        if max_x_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_x_ticks))
        if max_y_ticks == 0:
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.yaxis.set_major_locator(MaxNLocator(max_y_ticks))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=use_math_text))
        if style.get("set_labels", True):
            ax.set_xlabel(r"$-\ln X$", **(style.get("label_kwargs", {}) or {}))
            ax.set_ylabel(ylabel, **(style.get("label_kwargs", {}) or {}))

        c = (
            color[i]
            if not isinstance(color, str) and isinstance(color, Sequence)
            else color
        )
        if panel in scatter_only_panels:
            raw_x, raw_y = raw_data_by_panel[panel]
            scatter_kwargs = _scatter_kwargs(style)
            scatter_kwargs.setdefault("color", c)
            artists.append(ax.scatter(raw_x, raw_y, **scatter_kwargs))
        else:
            artists.extend(ax.plot(px, py, color=c, **line_kwargs))
            if panel in overlay_scatter_panels:
                raw_x, raw_y = raw_data_by_panel[panel]
                scatter_kwargs = _overlay_scatter_kwargs(style)
                scatter_kwargs.setdefault("color", c)
                artists.append(ax.scatter(raw_x, raw_y, **scatter_kwargs))

        if panel == "evidence" and lnz_error:
            if logplot:
                mask = logz >= ax.get_ylim()[0] - 10
                for s in lnz_error_levels:
                    artists.append(
                        ax.fill_between(
                            x[mask],
                            (logz + s * logzerr)[mask],
                            (logz - s * logzerr)[mask],
                            color=c,
                            **fill_kwargs,
                        )
                    )
            else:
                for s in lnz_error_levels:
                    artists.append(
                        ax.fill_between(
                            x,
                            np.exp(logz + s * logzerr),
                            np.exp(logz - s * logzerr),
                            color=c,
                            **fill_kwargs,
                        )
                    )

        if panel == "evidence":
            if evidence_summary:
                y_summary, y_summary_err = _evidence_value_and_error(
                    logz[-1],
                    logzerr[-1],
                    logplot=logplot,
                )
                artists.append(ax.axhline(y_summary, **evidence_summary_line_kwargs))
                label = _format_evidence_summary(
                    y_summary,
                    y_summary_err,
                    style.get("evidence_summary_fmt", None),
                )
                artists.append(
                    ax.annotate(
                        label,
                        xy=(evidence_text_x, y_summary),
                        xycoords=blended_transform_factory(ax.transAxes, ax.transData),
                        xytext=evidence_text_offset,
                        textcoords="offset points",
                        **evidence_summary_text_kwargs,
                    )
                )

            lnz_truth = style.get("lnz_truth", None)
            if lnz_truth is not None:
                ytruth = float(lnz_truth) if logplot else float(np.exp(lnz_truth))
                artists.append(
                    ax.axhline(
                        ytruth,
                        color=style.get("truth_color", "red"),
                        **truth_kwargs,
                    )
                )

    if bool(style.get("mark_final_live", False)):
        live_idx = None
        finite_nlive = nlive[np.isfinite(nlive)]
        if finite_nlive.size:
            live_candidates = np.where(nlive <= np.nanmin(finite_nlive))[0]
            if live_candidates.size:
                live_idx = int(live_candidates[0])
        if live_idx is not None and 0 <= live_idx < len(x):
            final_live_kwargs = dict(
                style.get("final_live_kwargs", {})
                if isinstance(style.get("final_live_kwargs"), Mapping)
                else {}
            )
            final_live_kwargs.setdefault("linestyle", "dashed")
            final_live_kwargs.setdefault("linewidth", 2)
            final_live_kwargs.setdefault("alpha", line_kwargs.get("alpha", 0.7))
            for ax in axes[: len(panels)]:
                artists.append(
                    ax.axvline(
                        x[live_idx],
                        color=color if isinstance(color, str) else color[0],
                        **final_live_kwargs,
                    )
                )

    if getattr(fig, "logger", None):
        fig.logger.debug(
            "dynesty_runplot rendered panels {} on axes {}".format(
                panels, axes_names[: len(panels)]
            )
        )
    return artists

#!/usr/bin/env python3

"""Render-health rules: turn layer/transform observations into ``JP-VIZ-*``.

These checks answer "is the figure wrong?" without reading pixels. They run on
observations collected during dryrun (row ledger + coordinate samples) or full
render hooks when those are wired.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

from .diagnostics import DiagnosticBag

__all__ = [
    "LayerObservation",
    "TransformStepObs",
    "evaluate_health",
    "observe_layer_dataframe",
    "report_to_dict",
]

_MESH_METHODS = frozenset(
    {
        "pcolormesh",
        "pcolor",
        "contourf",
        "imshow",
        "jpfield",
        "jpcontourf",
        "tripcolor",
        "tricontourf",
        "voronoif",
    }
)


@dataclass
class TransformStepObs:
    name: str
    detail: str = ""
    rows_in: int = 0
    rows_out: int = 0

    @property
    def emptied(self) -> bool:
        return self.rows_in > 0 and self.rows_out == 0


@dataclass
class LayerObservation:
    figure: str
    layer: str
    method: str = ""
    source: str = ""
    axes: str = "ax"
    n_points: int = 0
    finite_ratio: float = 1.0
    nan_ratio: float = 0.0
    data_bbox: Optional[tuple[float, float, float, float]] = None  # xmin,xmax,ymin,ymax
    axes_lim: Optional[dict[str, list[float]]] = None  # {"x": [lo,hi], "y": [...]}
    xscale: str = "linear"
    yscale: str = "linear"
    zorder: float = 0.0
    # colour channel / colorbar (JP-VIZ-004)
    c_min: Optional[float] = None
    c_max: Optional[float] = None
    colorbar_vmin: Optional[float] = None
    colorbar_vmax: Optional[float] = None
    # grid / interp (JP-VIZ-007)
    grid_nan_ratio: Optional[float] = None
    # legend (JP-VIZ-009)
    style_label: Optional[str] = None
    legend_labels: Optional[list[str]] = None
    steps: list[TransformStepObs] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    twin_path: Optional[str] = None
    #: True when dryrun skipped a heavy transform this layer needs — empty
    #: ledger is then *incomplete coverage*, not a proven empty plot.
    incomplete: bool = False


def report_to_dict(
    observations: Sequence[LayerObservation],
    *,
    diagnostics: DiagnosticBag | None = None,
    datasets: dict[str, Any] | None = None,
    twins: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "datasets": datasets or {},
        "layers": [_layer_dict(o) for o in observations],
        "twins": twins or {},
        "diagnostics": diagnostics.to_list() if diagnostics is not None else [],
    }


def _layer_dict(obs: LayerObservation) -> dict[str, Any]:
    return asdict(obs)


def evaluate_health(
    observations: Sequence[LayerObservation],
    *,
    bag: DiagnosticBag | None = None,
) -> DiagnosticBag:
    """Apply JP-VIZ-001…009 rules supported by pure observations."""
    bag = bag if bag is not None else DiagnosticBag()
    for obs in observations:
        path = f"$.Figures[name={obs.figure}].layers[name={obs.layer}]"
        _viz_001_empty(obs, path, bag)
        _viz_002_clipped(obs, path, bag)
        _viz_003_transform_empty(obs, path, bag)
        _viz_004_colorbar_saturation(obs, path, bag)
        _viz_005_log_nonpositive(obs, path, bag)
        _viz_007_grid_nan(obs, path, bag)
        _viz_008_collapsed(obs, path, bag)
        _viz_009_legend(obs, path, bag)
    _viz_006_occlusion(observations, bag)
    return bag


def observe_layer_dataframe(
    *,
    figure: str,
    layer: Mapping[str, Any] | dict[str, Any],
    df: Any,
    frame_cfg: Mapping[str, Any] | dict[str, Any] | None = None,
    source: str = "",
    steps: Sequence[TransformStepObs] | None = None,
    incomplete: bool = False,
    notes: Sequence[str] | None = None,
    twin_path: str | None = None,
) -> LayerObservation:
    """Build a :class:`LayerObservation` from a post-transform dataframe.

    Shared hook for dryrun light checks and Figure render paths. Observations
    describe the table available after the steps that path actually ran
    (light transforms only in doctor/dryrun; full pipeline on render).
    """
    import numpy as np

    from .utils.expression import eval_dataframe_expression

    layer_name = str(layer.get("name") or "layer")
    method = str(layer.get("method") or "")
    frame_cfg = frame_cfg if isinstance(frame_cfg, Mapping) else {}
    n_points = 0
    finite_ratio = 1.0
    nan_ratio = 0.0
    data_bbox = None
    c_min = c_max = None
    grid_nan_ratio = None

    coords = layer.get("coordinates") if isinstance(layer.get("coordinates"), dict) else {}
    x_vals = _eval_axis_expr(df, coords, "x", eval_dataframe_expression)
    y_vals = _eval_axis_expr(df, coords, "y", eval_dataframe_expression)
    if x_vals is None and "left" in coords:
        x_vals = _eval_axis_expr(df, coords, "left", eval_dataframe_expression)
    if y_vals is None and "right" in coords:
        y_vals = _eval_axis_expr(df, coords, "right", eval_dataframe_expression)
    z_vals = _eval_axis_expr(df, coords, "z", eval_dataframe_expression)
    c_vals = _eval_axis_expr(df, coords, "c", eval_dataframe_expression)

    try:
        n_points = int(len(df)) if df is not None else 0
    except Exception:
        n_points = 0

    if x_vals is not None and y_vals is not None:
        x = np.asarray(x_vals, dtype=float).reshape(-1)
        y = np.asarray(y_vals, dtype=float).reshape(-1)
        n = min(x.size, y.size)
        x, y = x[:n], y[:n]
        finite = np.isfinite(x) & np.isfinite(y)
        n_points = int(finite.sum())
        finite_ratio = float(finite.mean()) if n else 0.0
        nan_ratio = 1.0 - finite_ratio
        if n_points:
            xf, yf = x[finite], y[finite]
            data_bbox = (
                float(xf.min()),
                float(xf.max()),
                float(yf.min()),
                float(yf.max()),
            )

    for name, arr in (("z", z_vals), ("c", c_vals)):
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float).reshape(-1)
        finite = np.isfinite(a)
        if name == "z" and a.size:
            grid_nan_ratio = float(1.0 - finite.mean())
        if name in {"c", "z"} and finite.any():
            lo, hi = float(a[finite].min()), float(a[finite].max())
            if c_min is None:
                c_min, c_max = lo, hi
            else:
                c_min = min(c_min, lo)
                c_max = max(c_max, hi)

    axes_name = str(layer.get("axes") or "ax")
    ax_frame = frame_cfg.get(axes_name) if isinstance(frame_cfg.get(axes_name), dict) else {}
    axes_lim = None
    if isinstance(ax_frame, dict):
        xlim = ax_frame.get("xlim")
        ylim = ax_frame.get("ylim")
        if isinstance(xlim, (list, tuple)) and isinstance(ylim, (list, tuple)) and len(xlim) == 2 and len(ylim) == 2:
            try:
                axes_lim = {
                    "x": [float(xlim[0]), float(xlim[1])],
                    "y": [float(ylim[0]), float(ylim[1])],
                }
            except Exception:
                axes_lim = None
    xscale = str(ax_frame.get("xscale") or "linear") if isinstance(ax_frame, dict) else "linear"
    yscale = str(ax_frame.get("yscale") or "linear") if isinstance(ax_frame, dict) else "linear"

    cb_name = str(layer.get("colorbar") or "axc")
    cb_frame = frame_cfg.get(cb_name) if isinstance(frame_cfg.get(cb_name), dict) else {}
    color_cfg = {}
    if isinstance(cb_frame, dict):
        color_cfg = cb_frame.get("color") if isinstance(cb_frame.get("color"), dict) else {}
    cb_vmin = _as_float_opt(color_cfg.get("vmin"))
    cb_vmax = _as_float_opt(color_cfg.get("vmax"))

    zorder = 0.0
    style_label = None
    style = layer.get("style")
    if isinstance(style, dict):
        if "zorder" in style:
            try:
                zorder = float(style["zorder"])
            except Exception:
                pass
        if style.get("label") is not None:
            style_label = str(style["label"])

    legend_labels = None
    if isinstance(ax_frame, dict) and isinstance(ax_frame.get("legend"), dict):
        raw = ax_frame["legend"].get("labels")
        if isinstance(raw, list):
            legend_labels = [str(x) for x in raw]
        elif raw is None and "labels" in ax_frame["legend"]:
            legend_labels = []

    return LayerObservation(
        figure=figure,
        layer=layer_name,
        method=method,
        source=source,
        axes=axes_name,
        n_points=n_points,
        finite_ratio=finite_ratio,
        nan_ratio=nan_ratio,
        data_bbox=data_bbox,
        axes_lim=axes_lim,
        xscale=xscale,
        yscale=yscale,
        zorder=zorder,
        c_min=c_min,
        c_max=c_max,
        colorbar_vmin=cb_vmin,
        colorbar_vmax=cb_vmax,
        grid_nan_ratio=grid_nan_ratio,
        style_label=style_label,
        legend_labels=legend_labels,
        steps=list(steps or ()),
        twin_path=twin_path,
        incomplete=incomplete,
        notes=list(notes or ()),
    )


def _eval_axis_expr(df, coordinates: Mapping[str, Any], name: str, eval_fn) -> Any:
    if name not in coordinates:
        return None
    spec = coordinates[name]
    expr = spec.get("expr") if isinstance(spec, Mapping) else spec
    if expr is None and isinstance(spec, Mapping):
        expr = spec.get("name")
    if expr is None:
        return None
    try:
        return eval_fn(df, expr)
    except Exception:
        return None


def _as_float_opt(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _viz_001_empty(obs: LayerObservation, path: str, bag: DiagnosticBag) -> None:
    if obs.n_points <= 0:
        if obs.incomplete:
            bag.info(
                "JP-VIZ-001",
                path,
                f"layer {obs.layer!r} has zero points in dryrun, but coverage is "
                f"incomplete (heavy transform skipped; method={obs.method or '?'})",
                suggestion=(
                    "This is not a config failure. dryrun skips profile/density/"
                    "interp steps. Render with `jplot <file>`, or treat doctor "
                    "status=partial as 'structure ok, full mesh not simulated'."
                ),
                context={
                    "figure": obs.figure,
                    "layer": obs.layer,
                    "n_points": 0,
                    "incomplete": True,
                },
            )
            return
        bag.error(
            "JP-VIZ-001",
            path,
            f"layer {obs.layer!r} has zero points after transforms "
            f"(method={obs.method or '?'})",
            suggestion=(
                "Check the row ledger: a filter or density step may have emptied "
                "the table. Try jplot data head / eval on the source."
            ),
            context={"figure": obs.figure, "layer": obs.layer, "n_points": 0},
        )


def _viz_002_clipped(obs: LayerObservation, path: str, bag: DiagnosticBag) -> None:
    if not obs.data_bbox or not obs.axes_lim or obs.n_points <= 0:
        return
    xmin, xmax, ymin, ymax = obs.data_bbox
    xlim = obs.axes_lim.get("x")
    ylim = obs.axes_lim.get("y")
    if not (isinstance(xlim, (list, tuple)) and len(xlim) == 2):
        return
    if not (isinstance(ylim, (list, tuple)) and len(ylim) == 2):
        return
    x0, x1 = float(xlim[0]), float(xlim[1])
    y0, y1 = float(ylim[0]), float(ylim[1])
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    fully_out = xmax < x0 or xmin > x1 or ymax < y0 or ymin > y1
    if fully_out:
        bag.error(
            "JP-VIZ-002",
            path,
            f"layer {obs.layer!r} data bbox lies entirely outside axes lim "
            f"(data=[{xmin:.4g},{xmax:.4g}]×[{ymin:.4g},{ymax:.4g}], "
            f"lim=[{x0:.4g},{x1:.4g}]×[{y0:.4g},{y1:.4g}])",
            suggestion="Widen frame.<axes>.xlim/ylim or fix coordinate expressions.",
            context={"figure": obs.figure, "layer": obs.layer, "clip_fraction_est": 1.0},
        )
        return

    ox0, ox1 = max(xmin, x0), min(xmax, x1)
    oy0, oy1 = max(ymin, y0), min(ymax, y1)
    data_area = max(xmax - xmin, 0.0) * max(ymax - ymin, 0.0)
    overlap = max(ox1 - ox0, 0.0) * max(oy1 - oy0, 0.0)
    if data_area <= 0:
        return
    outside = 1.0 - (overlap / data_area)
    if outside >= 0.9:
        bag.error(
            "JP-VIZ-002",
            path,
            f"layer {obs.layer!r}: ~{outside:.0%} of data extent is outside axes lim",
            suggestion="Expand axis limits so most of the data is visible.",
            context={"figure": obs.figure, "layer": obs.layer, "clip_fraction_est": outside},
        )
    elif outside >= 0.5:
        bag.warning(
            "JP-VIZ-002",
            path,
            f"layer {obs.layer!r}: ~{outside:.0%} of data extent is outside axes lim",
            suggestion="Consider expanding xlim/ylim; more than half the data extent is clipped.",
            context={"figure": obs.figure, "layer": obs.layer, "clip_fraction_est": outside},
        )


def _viz_003_transform_empty(obs: LayerObservation, path: str, bag: DiagnosticBag) -> None:
    for step in obs.steps:
        if not step.emptied:
            continue
        if "skipped" in (step.detail or "").lower():
            # Heavy/unknown steps are skipped, not emptied.
            continue
        if obs.incomplete:
            bag.info(
                "JP-VIZ-003",
                path,
                f"transform step {step.name!r} looks empty under incomplete dryrun "
                f"({step.rows_in} → {step.rows_out})",
                suggestion="Not a config failure while coverage is partial.",
                context={"figure": obs.figure, "layer": obs.layer, "step": step.name},
            )
            continue
        bag.error(
            "JP-VIZ-003",
            path,
            f"transform {step.name!r} emptied the table "
            f"({step.rows_in} → {step.rows_out})"
            + (f": {step.detail}" if step.detail else ""),
            suggestion=(
                "Relax the filter / density cut, or verify the expression with "
                "jplot data eval against the source file."
            ),
            context={
                "figure": obs.figure,
                "layer": obs.layer,
                "step": step.name,
                "rows_in": step.rows_in,
                "rows_out": step.rows_out,
                "detail": step.detail,
            },
        )


def _viz_004_colorbar_saturation(obs: LayerObservation, path: str, bag: DiagnosticBag) -> None:
    if obs.c_max is None or obs.colorbar_vmax is None:
        return
    cmax = float(obs.c_max)
    vmax = float(obs.colorbar_vmax)
    vmin = float(obs.colorbar_vmin) if obs.colorbar_vmin is not None else (
        float(obs.c_min) if obs.c_min is not None else 0.0
    )
    if not (cmax > vmax):
        return
    span = max(cmax - vmin, 1e-30)
    overflow = (cmax - vmax) / span
    if overflow < 0.05:
        return
    level = "error" if overflow >= 0.3 else "warning"
    msg = (
        f"colorbar vmax={vmax:g} is below data max={cmax:g} "
        f"(~{overflow:.0%} of the colour span saturates)"
    )
    kwargs = dict(
        code="JP-VIZ-004",
        path=path,
        message=msg,
        suggestion=(
            "Raise frame.axc.color.vmax (or layer colorbar limits) to cover the data, "
            "or clip/filter the colour channel."
        ),
        context={
            "figure": obs.figure,
            "layer": obs.layer,
            "c_min": obs.c_min,
            "c_max": cmax,
            "colorbar_vmin": obs.colorbar_vmin,
            "colorbar_vmax": vmax,
            "overflow_fraction": overflow,
        },
    )
    if level == "error":
        bag.error(**kwargs)
    else:
        bag.warning(**kwargs)


def _viz_005_log_nonpositive(obs: LayerObservation, path: str, bag: DiagnosticBag) -> None:
    if obs.n_points <= 0 or not obs.data_bbox:
        return
    xmin, xmax, ymin, ymax = obs.data_bbox
    if str(obs.xscale).lower() == "log" and xmin <= 0:
        bag.warning(
            "JP-VIZ-005",
            path,
            f"xscale is log but data x reaches {xmin:.4g} (non-positive values "
            "are dropped silently by matplotlib)",
            suggestion="Filter x>0, use a linear scale, or shift the data.",
            context={"figure": obs.figure, "layer": obs.layer, "axis": "x"},
        )
    if str(obs.yscale).lower() == "log" and ymin <= 0:
        bag.warning(
            "JP-VIZ-005",
            path,
            f"yscale is log but data y reaches {ymin:.4g} (non-positive values "
            "are dropped silently by matplotlib)",
            suggestion="Filter y>0, use a linear scale, or shift the data.",
            context={"figure": obs.figure, "layer": obs.layer, "axis": "y"},
        )


def _viz_006_occlusion(
    observations: Sequence[LayerObservation], bag: DiagnosticBag
) -> None:
    """Higher-zorder mesh fully covering a lower layer's bbox → likely occluded."""
    by_key: dict[tuple[str, str], list[LayerObservation]] = {}
    for obs in observations:
        by_key.setdefault((obs.figure, obs.axes), []).append(obs)

    for (figure, axes), layers in by_key.items():
        ordered = sorted(layers, key=lambda o: o.zorder)
        for i, lower in enumerate(ordered):
            if lower.n_points <= 0 or not lower.data_bbox:
                continue
            for upper in ordered[i + 1 :]:
                if upper.zorder <= lower.zorder:
                    continue
                if str(upper.method).lower() not in _MESH_METHODS:
                    continue
                if upper.n_points <= 0 or not upper.data_bbox:
                    continue
                if _bbox_covers(upper.data_bbox, lower.data_bbox):
                    path = f"$.Figures[name={figure}].layers[name={lower.layer}]"
                    bag.warning(
                        "JP-VIZ-006",
                        path,
                        f"layer {lower.layer!r} (zorder={lower.zorder:g}) may be fully "
                        f"occluded by {upper.layer!r} (zorder={upper.zorder:g}, "
                        f"method={upper.method})",
                        suggestion=(
                            "Raise the lower layer's zorder, lower the mesh layer's "
                            "zorder, or reduce mesh opacity/coverage."
                        ),
                        context={
                            "figure": figure,
                            "axes": axes,
                            "lower": lower.layer,
                            "upper": upper.layer,
                            "lower_zorder": lower.zorder,
                            "upper_zorder": upper.zorder,
                        },
                    )


def _bbox_covers(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
    *,
    tol: float = 1e-9,
) -> bool:
    ox0, ox1, oy0, oy1 = outer
    ix0, ix1, iy0, iy1 = inner
    return (
        ox0 <= ix0 + tol
        and ox1 >= ix1 - tol
        and oy0 <= iy0 + tol
        and oy1 >= iy1 - tol
    )


def _viz_007_grid_nan(obs: LayerObservation, path: str, bag: DiagnosticBag) -> None:
    ratio = obs.grid_nan_ratio
    if ratio is None:
        # fall back to coordinate nan_ratio for mesh-like methods
        if str(obs.method).lower() not in _MESH_METHODS:
            return
        ratio = obs.nan_ratio
    if ratio is None or ratio < 0.5:
        return
    level = "error" if ratio >= 0.85 else "warning"
    msg = (
        f"layer {obs.layer!r} grid/field is ~{ratio:.0%} non-finite "
        "(often outside the convex hull of support points)"
    )
    kwargs = dict(
        code="JP-VIZ-007",
        path=path,
        message=msg,
        suggestion=(
            "Shrink the interpolation domain, densify support points, "
            "or mask exterior cells explicitly."
        ),
        context={"figure": obs.figure, "layer": obs.layer, "nan_ratio": ratio},
    )
    if level == "error":
        bag.error(**kwargs)
    else:
        bag.warning(**kwargs)


def _viz_008_collapsed(obs: LayerObservation, path: str, bag: DiagnosticBag) -> None:
    if obs.n_points <= 0 or not obs.data_bbox or not obs.axes_lim:
        return
    xmin, xmax, ymin, ymax = obs.data_bbox
    xlim = obs.axes_lim.get("x")
    ylim = obs.axes_lim.get("y")
    if not (isinstance(xlim, (list, tuple)) and isinstance(ylim, (list, tuple))):
        return
    if len(xlim) != 2 or len(ylim) != 2:
        return
    ax_w = abs(float(xlim[1]) - float(xlim[0]))
    ax_h = abs(float(ylim[1]) - float(ylim[0]))
    data_w = abs(xmax - xmin)
    data_h = abs(ymax - ymin)
    if ax_w <= 0 or ax_h <= 0:
        return
    frac = (data_w * data_h) / (ax_w * ax_h)
    if frac < 0.01 and data_w > 0 and data_h > 0:
        bag.warning(
            "JP-VIZ-008",
            path,
            f"layer {obs.layer!r} data occupies ~{frac:.2%} of the axes area "
            "(likely wrong lim scale)",
            suggestion="Tighten xlim/ylim around the data, or check coordinate units.",
            context={"figure": obs.figure, "layer": obs.layer, "area_fraction": frac},
        )


def _viz_009_legend(obs: LayerObservation, path: str, bag: DiagnosticBag) -> None:
    labels = obs.legend_labels
    style_label = obs.style_label
    if not labels:
        return
    # legend configured but this layer has no label and is the only candidate — soft
    if style_label:
        # if legend labels are explicit list and style label not among them
        if style_label not in labels and labels != ["auto"]:
            bag.warning(
                "JP-VIZ-009",
                path,
                f"layer label {style_label!r} is not listed in frame legend labels "
                f"{labels!r}",
                suggestion="Add the label to frame.<axes>.legend or fix style.label.",
                context={
                    "figure": obs.figure,
                    "layer": obs.layer,
                    "style_label": style_label,
                    "legend_labels": labels,
                },
            )
    # empty legend label list with handles expected
    if labels == []:
        bag.warning(
            "JP-VIZ-009",
            path,
            f"legend on axes {obs.axes!r} has an empty label list",
            suggestion="Provide legend labels or remove the legend block.",
            context={"figure": obs.figure, "axes": obs.axes},
        )

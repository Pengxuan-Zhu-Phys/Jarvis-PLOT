#!/usr/bin/env python3

"""Render-health rules: turn layer/transform observations into ``JP-VIZ-*``.

These checks answer "is the figure wrong?" without reading pixels. They run on
observations collected during dryrun (row ledger + coordinate samples) or,
later, full render hooks (E1 adapters).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional, Sequence

from .diagnostics import Diagnostic, DiagnosticBag

__all__ = [
    "LayerObservation",
    "TransformStepObs",
    "evaluate_health",
    "report_to_dict",
]


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
    n_points: int = 0
    finite_ratio: float = 1.0
    nan_ratio: float = 0.0
    data_bbox: Optional[tuple[float, float, float, float]] = None  # xmin,xmax,ymin,ymax
    axes_lim: Optional[dict[str, list[float]]] = None  # {"x": [lo,hi], "y": [...]}
    xscale: str = "linear"
    yscale: str = "linear"
    zorder: float = 0.0
    steps: list[TransformStepObs] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def report_to_dict(
    observations: Sequence[LayerObservation],
    *,
    diagnostics: DiagnosticBag | None = None,
    datasets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "datasets": datasets or {},
        "layers": [_layer_dict(o) for o in observations],
        "diagnostics": diagnostics.to_list() if diagnostics is not None else [],
    }


def _layer_dict(obs: LayerObservation) -> dict[str, Any]:
    payload = asdict(obs)
    return payload


def evaluate_health(
    observations: Sequence[LayerObservation],
    *,
    bag: DiagnosticBag | None = None,
) -> DiagnosticBag:
    """Apply JP-VIZ-001…008 rules that pure observations can support."""
    bag = bag if bag is not None else DiagnosticBag()
    for obs in observations:
        path = f"$.Figures[name={obs.figure}].layers[name={obs.layer}]"
        _viz_001_empty(obs, path, bag)
        _viz_002_clipped(obs, path, bag)
        _viz_003_transform_empty(obs, path, bag)
        _viz_005_log_nonpositive(obs, path, bag)
        _viz_008_collapsed(obs, path, bag)
    return bag


def _viz_001_empty(obs: LayerObservation, path: str, bag: DiagnosticBag) -> None:
    if obs.n_points <= 0:
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
    if not obs.data_bbox or not obs.axes_lim:
        return
    if obs.n_points <= 0:
        return
    xmin, xmax, ymin, ymax = obs.data_bbox
    xlim = obs.axes_lim.get("x")
    ylim = obs.axes_lim.get("y")
    if not (isinstance(xlim, (list, tuple)) and len(xlim) == 2):
        return
    if not (isinstance(ylim, (list, tuple)) and len(ylim) == 2):
        return
    # Approximate clipped fraction via bbox vs lim overlap of the data extent.
    # Full point-wise fraction needs the arrays; dryrun stores bbox + n_points.
    # Use "data bbox completely outside lim" as hard clip, partial via area.
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

    # Overlap area vs data bbox area as a rough clip proxy when extents differ a lot.
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

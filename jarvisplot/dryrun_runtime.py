#!/usr/bin/env python3

"""Load config + data + transforms without drawing (agent dryrun / doctor).

Builds a row ledger and coordinate samples so :mod:`render_health` can emit
``JP-VIZ-*`` findings. Deliberately avoids importing matplotlib.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .diagnostics import DiagnosticBag
from .render_health import LayerObservation, TransformStepObs, evaluate_health, report_to_dict
from .utils.pathing import resolve_project_path

__all__ = ["dryrun_config", "dryrun_file"]


def dryrun_file(
    path: str,
    *,
    with_data: bool = False,
    out_dir: str | None = None,
) -> tuple[dict[str, Any], DiagnosticBag]:
    import yaml

    resolved = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(resolved):
        bag = DiagnosticBag()
        bag.error(
            "JP-YML-000",
            "$",
            f"config file not found: {resolved}",
        )
        return {"file": resolved, "ok": False}, bag
    with open(resolved, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    report, bag = dryrun_config(
        config,
        base_dir=os.path.dirname(resolved),
        with_data=with_data,
        out_dir=out_dir,
    )
    report["file"] = resolved
    return report, bag


def dryrun_config(
    config: Any,
    *,
    base_dir: str | None = None,
    with_data: bool = False,
    out_dir: str | None = None,
) -> tuple[dict[str, Any], DiagnosticBag]:
    bag = DiagnosticBag()
    if not isinstance(config, dict):
        bag.error("JP-YML-003", "$", f"config root must be a mapping, got {type(config).__name__}")
        return {"ok": False, "layers": [], "datasets": {}, "coverage": "failed"}, bag

    # Expand type: macros in-memory (same engine as render) so dryrun/doctor are
    # full coverage, not "structure ok / transforms skipped".
    from copy import deepcopy

    from .Figure.figure_types import expand_typed_figures

    work = deepcopy(config)
    expanded_names: list[str] = []
    try:
        expanded_names = expand_typed_figures(work, raise_on_error=True)
    except Exception as exc:
        bag.error(
            "JP-VIZ-000",
            "$.Figures",
            f"type: expansion failed: {exc}",
            suggestion=(
                "Fix type macro fields (x/y/z/weight/data) or run "
                "`jplot config expand <yaml> --diff` to inspect the expanded layers."
            ),
        )
        return {
            "ok": False,
            "layers": [],
            "datasets": {},
            "coverage": "failed",
            "type_expanded": [],
        }, bag

    datasets_meta, frames = _load_datasets(work, base_dir=base_dir, bag=bag)
    observations: list[LayerObservation] = []
    twins: dict[str, Any] = {}
    heavy_skipped: list[str] = []
    # share_data names that were not materialised because their producer
    # transform is heavy and skipped in dryrun.
    incomplete_sources: set[str] = set()

    twin_root = None
    if with_data:
        twin_root = Path(
            out_dir
            or os.path.join(base_dir or ".", ".cache", "agent_twins")
        )
        twin_root.mkdir(parents=True, exist_ok=True)

    for figure_index, figure in enumerate(work.get("Figures") or ()):
        if not isinstance(figure, dict):
            continue
        fig_name = str(figure.get("name") or f"figure_{figure_index}")
        if "type" in figure:
            # Unknown type left unexpanded → hard error (not silent skip).
            bag.error(
                "JP-VIZ-000",
                f"$.Figures[{figure_index}]",
                f"figure {fig_name!r} still has type: {figure.get('type')!r} after expansion",
                suggestion="Use a known type (jplot cap types) or write layers: by hand.",
            )
            continue
        frame = figure.get("frame") if isinstance(figure.get("frame"), dict) else {}
        for layer_index, layer in enumerate(figure.get("layers") or ()):
            if not isinstance(layer, dict):
                continue
            obs, twin_meta, skipped = _observe_layer(
                figure_name=fig_name,
                layer=layer,
                layer_index=layer_index,
                frames=frames,
                frame_cfg=frame,
                bag=bag,
                with_data=with_data,
                twin_root=twin_root,
                incomplete_sources=incomplete_sources,
            )
            if skipped:
                for step_name in skipped:
                    heavy_skipped.append(f"{fig_name}/{obs.layer if obs else layer_index}:{step_name}")
                share = layer.get("share_data")
                if isinstance(share, str) and share.strip():
                    incomplete_sources.add(share.strip())
            if obs is not None:
                observations.append(obs)
            if twin_meta:
                twins[f"{fig_name}/{obs.layer if obs else layer_index}"] = twin_meta

        # When heavy transforms are skipped, layer coords are often mesh columns
        # (x,y after density) that do not exist on the raw source — so JP-VIZ-002
        # never fires on the product default type: path. Check frame lims against
        # pre-transform input axes (type stash or profile/density transform x/y).
        # Only on heavy-skip figures: full-coverage layers already get JP-VIZ via
        # evaluate_health on post-transform observations.
        fig_heavy = any(h.startswith(f"{fig_name}/") for h in heavy_skipped)
        if fig_heavy:
            _pretransform_lim_check(
                figure=figure,
                figure_index=figure_index,
                fig_name=fig_name,
                frames=frames,
                bag=bag,
            )

    evaluate_health(observations, bag=bag)
    # Tri-state ok + coverage (partial ≠ failed).
    if not bag.ok:
        verdict_ok: bool | None = False
        coverage = "failed"
        status = "failed"
    elif heavy_skipped:
        # Incomplete dryrun ledger ≠ config failure. Config is expected to render.
        verdict_ok = None
        coverage = "partial"
        status = "partial_renderable"
        bag.info(
            "JP-VIZ-010",
            "$.Figures",
            f"dryrun skipped {len(heavy_skipped)} heavy transform step(s); "
            "layer ledgers for density/profile/interp are incomplete "
            "(status=partial_renderable — config is OK to render)",
            suggestion=(
                "Not a failed config. Structure/columns were checked; full mesh/"
                "density only runs on `jplot <file>`. Proceed to render or "
                "agent_output — do not rewrite YAML solely because of this."
            ),
            context={
                "heavy_skipped": heavy_skipped[:20],
                "renderable": True,
                "status": "partial_renderable",
            },
        )
    else:
        verdict_ok = True
        coverage = "full"
        status = "ok"

    report = report_to_dict(
        observations, diagnostics=bag, datasets=datasets_meta, twins=twins
    )
    report["type_expanded"] = list(expanded_names)
    report["heavy_skipped"] = list(heavy_skipped)
    report["ok"] = verdict_ok
    report["coverage"] = coverage
    report["status"] = status
    report["renderable"] = status in {"ok", "partial_renderable"}
    if status == "partial_renderable":
        report["status_note"] = (
            "heavy transforms skipped in dryrun; YAML is expected to render successfully"
        )
    return report, bag


def _load_datasets(
    config: dict[str, Any],
    *,
    base_dir: str | None,
    bag: DiagnosticBag,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (metadata, name->dataframe)."""
    import pandas as pd

    from .data_access import detect_type, load_dataframe

    meta: dict[str, Any] = {}
    frames: dict[str, Any] = {}
    for index, entry in enumerate(config.get("DataSet") or ()):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        raw_path = entry.get("path")
        if not isinstance(raw_path, str):
            bag.warning(
                "JP-DAT-004",
                f"$.DataSet[{index}].path",
                f"dataset {name!r} has no single string path; dryrun skips it",
            )
            continue
        resolved = resolve_project_path(raw_path, base_dir)
        if not Path(resolved).exists():
            bag.error(
                "JP-DAT-004",
                f"$.DataSet[{index}].path",
                f"data file not found: {resolved}",
            )
            meta[name] = {"path": str(resolved), "rows": None, "error": "missing"}
            continue
        kind = str(entry.get("type") or detect_type(str(resolved), "auto"))
        try:
            df = load_dataframe(
                str(resolved),
                kind=kind if kind in {"csv", "parquet", "hdf5"} else "csv",
                group=entry.get("dataset") if isinstance(entry.get("dataset"), str) else None,
            )
            # dataset-level transform (simple steps only)
            ds_transform = entry.get("transform")
            if isinstance(ds_transform, list) and ds_transform:
                df, _steps = _apply_simple_transforms(df, ds_transform)
            rows = int(len(df))
            frames[name] = df
            meta[name] = {
                "path": str(resolved),
                "type": kind,
                "rows": rows,
                "columns": [str(c) for c in df.columns],
            }
        except Exception as exc:
            bag.warning(
                "JP-COL-900",
                f"$.DataSet[{index}]",
                f"could not load dataset {name!r}: {exc}",
            )
            meta[name] = {"path": str(resolved), "rows": None, "error": str(exc)}
    return meta, frames


_HEAVY_TRANSFORM_KEYS = frozenset(
    {
        "profile",
        "make_density_core",
        "posterior_density",
        "make_interp_2d",
        "to_csv",
        "to_parquet",
    }
)


def _pretransform_lim_check(
    *,
    figure: dict[str, Any],
    figure_index: int,
    fig_name: str,
    frames: dict[str, Any],
    bag: DiagnosticBag,
) -> None:
    """JP-VIZ-002 on input axes when heavy transforms leave dryrun blind.

    Uses type-expand stash (``agent_output._digest_axes``) or the first heavy
    step's ``coordinates.x/y`` / top-level x/y against ``frame.ax`` lims.
    Marked ``basis: pre-transform`` so agents know this is not post-mesh.
    """
    frame = figure.get("frame") if isinstance(figure.get("frame"), dict) else {}
    ax = frame.get("ax") if isinstance(frame.get("ax"), dict) else {}
    xlim = ax.get("xlim")
    ylim = ax.get("ylim")
    if not (isinstance(xlim, (list, tuple)) and len(xlim) == 2):
        return
    if not (isinstance(ylim, (list, tuple)) and len(ylim) == 2):
        return

    source, x_expr, y_expr = _figure_input_xy(figure)
    if not source or not x_expr or not y_expr:
        return
    df = frames.get(source)
    if df is None:
        return
    try:
        from .utils.expression import eval_dataframe_expression

        xv = eval_dataframe_expression(df, x_expr)
        yv = eval_dataframe_expression(df, y_expr)
        import numpy as np

        x = np.asarray(xv, dtype=float).reshape(-1)
        y = np.asarray(yv, dtype=float).reshape(-1)
        n = min(x.size, y.size)
        if n <= 0:
            return
        x, y = x[:n], y[:n]
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            return
        xf, yf = x[finite], y[finite]
        xmin, xmax = float(xf.min()), float(xf.max())
        ymin, ymax = float(yf.min()), float(yf.max())
    except Exception:
        return

    x0, x1 = float(xlim[0]), float(xlim[1])
    y0, y1 = float(ylim[0]), float(ylim[1])
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    path = f"$.Figures[{figure_index}]"
    fully_out = xmax < x0 or xmin > x1 or ymax < y0 or ymin > y1
    if fully_out:
        bag.error(
            "JP-VIZ-002",
            path,
            f"figure {fig_name!r}: pre-transform data bbox lies entirely outside "
            f"frame.ax lim (data=[{xmin:.4g},{xmax:.4g}]×[{ymin:.4g},{ymax:.4g}], "
            f"lim=[{x0:.4g},{x1:.4g}]×[{y0:.4g},{y1:.4g}]; basis=pre-transform)",
            suggestion=(
                "Widen frame.ax.xlim/ylim (or type x/y lim). Checked on raw source "
                "columns because dryrun skipped heavy transforms."
            ),
            context={
                "figure": fig_name,
                "basis": "pre-transform",
                "clip_fraction_est": 1.0,
                "x_expr": x_expr,
                "y_expr": y_expr,
            },
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
            f"figure {fig_name!r}: ~{outside:.0%} of pre-transform data extent "
            f"is outside frame.ax lim (basis=pre-transform)",
            suggestion="Expand xlim/ylim so most of the source data is visible.",
            context={
                "figure": fig_name,
                "basis": "pre-transform",
                "clip_fraction_est": outside,
            },
        )
    elif outside >= 0.5:
        bag.warning(
            "JP-VIZ-002",
            path,
            f"figure {fig_name!r}: ~{outside:.0%} of pre-transform data extent "
            f"is outside frame.ax lim (basis=pre-transform)",
            suggestion="Consider expanding xlim/ylim; more than half the source extent is clipped.",
            context={
                "figure": fig_name,
                "basis": "pre-transform",
                "clip_fraction_est": outside,
            },
        )


def _figure_input_xy(figure: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    """Return (source_name, x_expr, y_expr) for pre-transform checks."""

    def _expr(field: Any) -> str | None:
        if field is None:
            return None
        if isinstance(field, Mapping):
            e = field.get("expr", field.get("name"))
            return str(e) if e is not None else None
        return str(field)

    ao = figure.get("agent_output") if isinstance(figure.get("agent_output"), Mapping) else {}
    stash = ao.get("_digest_axes") if isinstance(ao, Mapping) else None
    if isinstance(stash, Mapping):
        src = stash.get("data")
        if isinstance(src, list):
            src = src[0] if src else None
        if isinstance(src, str) and src.strip():
            xe, ye = _expr(stash.get("x")), _expr(stash.get("y"))
            if xe and ye:
                return src.strip(), xe, ye

    # Expanded type figures keep layers with heavy transforms on the raw source.
    for layer in figure.get("layers") or ():
        if not isinstance(layer, Mapping):
            continue
        for block in layer.get("data") or ():
            if not isinstance(block, Mapping):
                continue
            src = block.get("source")
            if isinstance(src, list):
                src = src[0] if src else None
            if not isinstance(src, str) or not src.strip():
                continue
            for step in block.get("transform") or ():
                if not isinstance(step, Mapping):
                    continue
                for heavy in ("profile", "posterior_density", "make_density_core"):
                    if heavy not in step:
                        continue
                    cfg = step.get(heavy)
                    if not isinstance(cfg, Mapping):
                        continue
                    # top-level x/y or coordinates.x/y
                    xe = _expr(cfg.get("x"))
                    ye = _expr(cfg.get("y"))
                    coords = cfg.get("coordinates") if isinstance(cfg.get("coordinates"), Mapping) else {}
                    if xe is None:
                        xe = _expr(coords.get("x"))
                    if ye is None:
                        ye = _expr(coords.get("y"))
                    if xe and ye:
                        return src.strip(), xe, ye
    return None, None, None


def _observe_layer(
    *,
    figure_name: str,
    layer: dict[str, Any],
    layer_index: int,
    frames: dict[str, Any],
    frame_cfg: dict[str, Any],
    bag: DiagnosticBag,
    with_data: bool = False,
    twin_root: Path | None = None,
    incomplete_sources: set[str] | None = None,
) -> tuple[LayerObservation | None, dict[str, Any] | None, list[str]]:
    """Return ``(obs, twin_meta, heavy_step_names_skipped)``."""
    layer_name = str(layer.get("name") or f"layer_{layer_index}")
    method = str(layer.get("method") or "")
    incomplete_sources = incomplete_sources or set()
    data_blocks = layer.get("data") or []
    if not isinstance(data_blocks, list) or not data_blocks:
        return (
            LayerObservation(
                figure=figure_name,
                layer=layer_name,
                method=method,
                n_points=0,
                notes=["no data blocks"],
            ),
            None,
            [],
        )

    # Use the first data block (primary series); multi-source layers get a note.
    block = data_blocks[0] if isinstance(data_blocks[0], dict) else {}
    source = block.get("source")
    if isinstance(source, list):
        source = source[0] if source else None
    if not isinstance(source, str) or source not in frames:
        # Missing share_data after a skipped heavy producer → incomplete, not failed.
        incomplete = isinstance(source, str) and source in incomplete_sources
        note = f"source {source!r} not loaded"
        if incomplete:
            note += " (upstream heavy transform skipped in dryrun)"
        return (
            LayerObservation(
                figure=figure_name,
                layer=layer_name,
                method=method,
                source=str(source or ""),
                n_points=0,
                notes=[note],
                incomplete=incomplete,
            ),
            None,
            [],
        )

    df = frames[source]
    steps: list[TransformStepObs] = []
    heavy_skipped: list[str] = []
    # layer-level transform under data block
    transform = block.get("transform")
    if isinstance(transform, list) and transform:
        df, steps, heavy_skipped = _apply_simple_transforms(df, transform)

    # coordinate samples for bbox + colour channel
    n_points = int(len(df))
    finite_ratio = 1.0
    nan_ratio = 0.0
    data_bbox = None
    c_min = c_max = None
    grid_nan_ratio = None
    twin_cols: dict[str, Any] = {}
    coords = layer.get("coordinates") if isinstance(layer.get("coordinates"), dict) else {}
    x_vals, y_vals = _eval_xy(df, coords)
    z_vals = _eval_axis(df, coords, "z")
    c_vals = _eval_axis(df, coords, "c")
    if x_vals is not None and y_vals is not None and len(x_vals) and len(y_vals):
        import numpy as np

        x = np.asarray(x_vals, dtype=float)
        y = np.asarray(y_vals, dtype=float)
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
        twin_cols["x"] = x
        twin_cols["y"] = y

    for name, arr in (("z", z_vals), ("c", c_vals)):
        if arr is None:
            continue
        import numpy as np

        a = np.asarray(arr, dtype=float).reshape(-1)
        twin_cols[name] = a
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
        if isinstance(xlim, (list, tuple)) and isinstance(ylim, (list, tuple)):
            axes_lim = {
                "x": [float(xlim[0]), float(xlim[1])],
                "y": [float(ylim[0]), float(ylim[1])],
            }
    xscale = str(ax_frame.get("xscale") or "linear")
    yscale = str(ax_frame.get("yscale") or "linear")

    # colorbar limits from frame.axc / named colorbar axes
    cb_name = str(layer.get("colorbar") or "axc")
    cb_frame = frame_cfg.get(cb_name) if isinstance(frame_cfg.get(cb_name), dict) else {}
    color_cfg = {}
    if isinstance(cb_frame, dict):
        color_cfg = cb_frame.get("color") if isinstance(cb_frame.get("color"), dict) else {}
    cb_vmin = _as_float(color_cfg.get("vmin"))
    cb_vmax = _as_float(color_cfg.get("vmax"))

    zorder = 0.0
    style_label = None
    style = layer.get("style")
    if isinstance(style, dict):
        if "zorder" in style:
            try:
                zorder = float(style["zorder"])
            except Exception:
                pass
        if "label" in style and style["label"] is not None:
            style_label = str(style["label"])

    legend_labels = None
    if isinstance(ax_frame, dict) and isinstance(ax_frame.get("legend"), dict):
        leg = ax_frame["legend"]
        if "labels" in leg:
            raw = leg.get("labels")
            if isinstance(raw, list):
                legend_labels = [str(x) for x in raw]
            elif raw is None:
                legend_labels = []

    twin_path = None
    twin_meta = None
    if with_data and twin_root is not None and twin_cols:
        twin_path, twin_meta = _write_twin(
            twin_root, figure_name, layer_name, twin_cols, n_points=n_points
        )

    incomplete = bool(heavy_skipped) or (
        isinstance(source, str) and source in incomplete_sources
    )
    # Coordinates often name columns produced by the skipped heavy step (x/y/z
    # mesh); zero finite points then means incomplete, not empty data.
    if heavy_skipped and n_points <= 0:
        incomplete = True

    obs = LayerObservation(
        figure=figure_name,
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
        steps=steps,
        twin_path=twin_path,
        incomplete=incomplete,
        notes=(["heavy transform skipped in dryrun"] if heavy_skipped else []),
    )
    return obs, twin_meta, heavy_skipped


def _apply_simple_transforms(
    df,
    transform: Sequence[Mapping[str, Any]],
) -> tuple[Any, list[TransformStepObs], list[str]]:
    """Apply the cheap transform subset; skip density/profile/interp (noted).

    Returns ``(df, steps, heavy_step_names_skipped)``.
    """
    from .Figure.preprocessor_runtime import (
        add_column,
        drop_columns,
        filter_df,
        keep_columns,
        sort_by,
    )

    steps: list[TransformStepObs] = []
    heavy_skipped: list[str] = []
    work = df
    for step in transform:
        if not isinstance(step, Mapping):
            continue
        rows_in = int(len(work))
        name = next(iter(step.keys()), "unknown")
        detail = ""
        try:
            if "filter" in step:
                detail = str(step.get("filter"))
                work = filter_df(work, step["filter"], logger=None)
            elif "sortby" in step:
                detail = str(step.get("sortby"))
                work = sort_by(work, step["sortby"], logger=None)
            elif "add_column" in step:
                detail = str((step.get("add_column") or {}).get("name", ""))
                work = add_column(work, step["add_column"], logger=None)
            elif "keep_columns" in step:
                detail = str(step.get("keep_columns"))
                work = keep_columns(work, step.get("keep_columns"), logger=None)
            elif "drop_columns" in step:
                detail = str(step.get("drop_columns"))
                work = drop_columns(work, step.get("drop_columns"), logger=None)
            elif any(k in step for k in _HEAVY_TRANSFORM_KEYS):
                heavy_name = str(name)
                heavy_skipped.append(heavy_name)
                steps.append(
                    TransformStepObs(
                        name=heavy_name,
                        detail="skipped in dryrun (heavy step)",
                        rows_in=rows_in,
                        rows_out=rows_in,
                    )
                )
                continue
            else:
                steps.append(
                    TransformStepObs(
                        name=str(name),
                        detail="unknown step skipped",
                        rows_in=rows_in,
                        rows_out=rows_in,
                    )
                )
                continue
        except Exception as exc:
            steps.append(
                TransformStepObs(
                    name=str(name),
                    detail=f"failed: {exc}",
                    rows_in=rows_in,
                    rows_out=0,
                )
            )
            # leave work unchanged on failure
            continue
        steps.append(
            TransformStepObs(
                name=str(name),
                detail=detail,
                rows_in=rows_in,
                rows_out=int(len(work)),
            )
        )
    return work, steps, heavy_skipped


def _eval_axis(df, coordinates: Mapping[str, Any], name: str):
    from .utils.expression import eval_dataframe_expression

    if name not in coordinates:
        return None
    spec = coordinates[name]
    expr = spec.get("expr") if isinstance(spec, Mapping) else spec
    if expr is None:
        return None
    try:
        return eval_dataframe_expression(df, expr)
    except Exception:
        return None


def _eval_xy(df, coordinates: Mapping[str, Any]):
    x = _eval_axis(df, coordinates, "x")
    y = _eval_axis(df, coordinates, "y")
    if x is None and "left" in coordinates:
        x = _eval_axis(df, coordinates, "left")
    if y is None and "right" in coordinates:
        y = _eval_axis(df, coordinates, "right")
    return x, y


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _write_twin(
    twin_root: Path,
    figure: str,
    layer: str,
    columns: dict[str, Any],
    *,
    n_points: int,
) -> tuple[str, dict[str, Any]]:
    import numpy as np
    import pandas as pd

    safe_fig = "".join(c if c.isalnum() or c in "-_" else "_" for c in figure)
    safe_ly = "".join(c if c.isalnum() or c in "-_" else "_" for c in layer)
    # Prefer parquet when an engine is available; else CSV (still a numeric twin).
    base = twin_root / f"{safe_fig}__{safe_ly}"
    lengths = [np.asarray(v).reshape(-1).size for v in columns.values()]
    n = min(lengths) if lengths else 0
    frame = {k: np.asarray(v).reshape(-1)[:n] for k, v in columns.items()}
    df = pd.DataFrame(frame)
    fmt = "parquet"
    path = base.with_suffix(".parquet")
    try:
        df.to_parquet(path, index=False)
    except Exception:
        fmt = "csv"
        path = base.with_suffix(".csv")
        df.to_csv(path, index=False)
    meta = {
        "path": str(path),
        "format": fmt,
        "figure": figure,
        "layer": layer,
        "rows": int(n),
        "n_points": int(n_points),
        "columns": list(frame.keys()),
    }
    return str(path), meta

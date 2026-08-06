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


def dryrun_file(path: str) -> tuple[dict[str, Any], DiagnosticBag]:
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
    report, bag = dryrun_config(config, base_dir=os.path.dirname(resolved))
    report["file"] = resolved
    return report, bag


def dryrun_config(
    config: Any,
    *,
    base_dir: str | None = None,
) -> tuple[dict[str, Any], DiagnosticBag]:
    bag = DiagnosticBag()
    if not isinstance(config, dict):
        bag.error("JP-YML-003", "$", f"config root must be a mapping, got {type(config).__name__}")
        return {"ok": False, "layers": [], "datasets": {}}, bag

    datasets_meta, frames = _load_datasets(config, base_dir=base_dir, bag=bag)
    observations: list[LayerObservation] = []

    for figure_index, figure in enumerate(config.get("Figures") or ()):
        if not isinstance(figure, dict):
            continue
        fig_name = str(figure.get("name") or f"figure_{figure_index}")
        if "type" in figure:
            # type: macros expand only on the render path today
            bag.info(
                "JP-VIZ-000",
                f"$.Figures[{figure_index}]",
                f"figure {fig_name!r} uses type: {figure.get('type')!r}; "
                "dryrun skips type expansion (use expanded layers for full ledger)",
            )
            continue
        frame = figure.get("frame") if isinstance(figure.get("frame"), dict) else {}
        for layer_index, layer in enumerate(figure.get("layers") or ()):
            if not isinstance(layer, dict):
                continue
            obs = _observe_layer(
                figure_name=fig_name,
                layer=layer,
                layer_index=layer_index,
                frames=frames,
                frame_cfg=frame,
                bag=bag,
            )
            if obs is not None:
                observations.append(obs)

    evaluate_health(observations, bag=bag)
    report = report_to_dict(observations, diagnostics=bag, datasets=datasets_meta)
    report["ok"] = bag.ok
    return report, bag


def _load_datasets(
    config: dict[str, Any],
    *,
    base_dir: str | None,
    bag: DiagnosticBag,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (metadata, name->dataframe)."""
    import pandas as pd

    from .verbs.data import _detect_type, _load_dataframe

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
        kind = str(entry.get("type") or _detect_type(str(resolved), "auto"))
        try:
            df = _load_dataframe(
                str(resolved),
                kind=kind if kind in {"csv", "parquet", "hdf5"} else "csv",
                group=entry.get("dataset"),
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


def _observe_layer(
    *,
    figure_name: str,
    layer: dict[str, Any],
    layer_index: int,
    frames: dict[str, Any],
    frame_cfg: dict[str, Any],
    bag: DiagnosticBag,
) -> LayerObservation | None:
    layer_name = str(layer.get("name") or f"layer_{layer_index}")
    method = str(layer.get("method") or "")
    data_blocks = layer.get("data") or []
    if not isinstance(data_blocks, list) or not data_blocks:
        return LayerObservation(
            figure=figure_name,
            layer=layer_name,
            method=method,
            n_points=0,
            notes=["no data blocks"],
        )

    # Use the first data block (primary series); multi-source layers get a note.
    block = data_blocks[0] if isinstance(data_blocks[0], dict) else {}
    source = block.get("source")
    if isinstance(source, list):
        source = source[0] if source else None
    if not isinstance(source, str) or source not in frames:
        return LayerObservation(
            figure=figure_name,
            layer=layer_name,
            method=method,
            source=str(source or ""),
            n_points=0,
            notes=[f"source {source!r} not loaded"],
        )

    df = frames[source]
    steps: list[TransformStepObs] = []
    # layer-level transform under data block
    transform = block.get("transform")
    if isinstance(transform, list) and transform:
        df, steps = _apply_simple_transforms(df, transform)

    # coordinate samples for bbox
    n_points = int(len(df))
    finite_ratio = 1.0
    nan_ratio = 0.0
    data_bbox = None
    coords = layer.get("coordinates") if isinstance(layer.get("coordinates"), dict) else {}
    x_vals, y_vals = _eval_xy(df, coords)
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

    axes_name = str(layer.get("axes") or "ax")
    ax_frame = frame_cfg.get(axes_name) if isinstance(frame_cfg.get(axes_name), dict) else {}
    axes_lim = None
    if isinstance(ax_frame, dict):
        xlim = ax_frame.get("xlim")
        ylim = ax_frame.get("ylim")
        if isinstance(xlim, (list, tuple)) and isinstance(ylim, (list, tuple)):
            axes_lim = {"x": [float(xlim[0]), float(xlim[1])], "y": [float(ylim[0]), float(ylim[1])]}
    xscale = str(ax_frame.get("xscale") or "linear")
    yscale = str(ax_frame.get("yscale") or "linear")
    zorder = 0.0
    style = layer.get("style")
    if isinstance(style, dict) and "zorder" in style:
        try:
            zorder = float(style["zorder"])
        except Exception:
            pass

    return LayerObservation(
        figure=figure_name,
        layer=layer_name,
        method=method,
        source=source,
        n_points=n_points,
        finite_ratio=finite_ratio,
        nan_ratio=nan_ratio,
        data_bbox=data_bbox,
        axes_lim=axes_lim,
        xscale=xscale,
        yscale=yscale,
        zorder=zorder,
        steps=steps,
    )


def _apply_simple_transforms(
    df,
    transform: Sequence[Mapping[str, Any]],
) -> tuple[Any, list[TransformStepObs]]:
    """Apply the cheap transform subset; skip density/profile/interp (noted)."""
    from .Figure.preprocessor_runtime import (
        add_column,
        drop_columns,
        filter_df,
        keep_columns,
        sort_by,
    )

    steps: list[TransformStepObs] = []
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
            elif any(
                k in step
                for k in (
                    "profile",
                    "make_density_core",
                    "posterior_density",
                    "make_interp_2d",
                    "to_csv",
                    "to_parquet",
                )
            ):
                steps.append(
                    TransformStepObs(
                        name=name,
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
    return work, steps


def _eval_xy(df, coordinates: Mapping[str, Any]):
    from .utils.expression import eval_dataframe_expression

    def _axis(name: str):
        if name not in coordinates:
            # ternary aliases
            return None
        spec = coordinates[name]
        expr = spec.get("expr") if isinstance(spec, Mapping) else spec
        if expr is None:
            return None
        try:
            return eval_dataframe_expression(df, expr)
        except Exception:
            return None

    x = _axis("x")
    y = _axis("y")
    if x is None and "left" in coordinates:
        x = _axis("left")
    if y is None and "right" in coordinates:
        y = _axis("right")
    return x, y

#!/usr/bin/env python3

"""YAML-declared agent digests (lossy Voronoi cell summaries).

CLI does not own this. Figures declare ``agent_output``; ``doctor`` plans
exports; ``jplot <yaml>`` writes JSON after the real data path runs.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .diagnostics import DiagnosticBag

__all__ = [
    "AGENT_DIGEST_VERSION",
    "AgentOutputSpec",
    "build_voronoi_digest",
    "maybe_write_figure_digest",
    "parse_agent_output",
    "plan_agent_exports",
    "resolve_export_path",
    "write_digest_json",
]

AGENT_DIGEST_VERSION = "1.0.0"
_MAX_CELLS_HARD = 50_000
_MAX_CELLS_DEFAULT = 1024


@dataclass
class AgentOutputSpec:
    enable: bool
    format: str
    method: str
    path: str  # auto or path string
    max_cells: int
    seed: int
    geometry: str
    weight: Any  # auto | expr string | mapping
    include: list[str]
    strict: bool
    figure_name: str
    figure_index: int
    raw: dict[str, Any]


def parse_agent_output(
    figure: Mapping[str, Any],
    *,
    figure_index: int = 0,
    root_output: Mapping[str, Any] | None = None,
) -> AgentOutputSpec | None:
    """Merge root ``output.agent_output`` with figure block; None if disabled."""
    root = {}
    if isinstance(root_output, Mapping):
        root = root_output.get("agent_output") if isinstance(root_output.get("agent_output"), Mapping) else {}
    fig_block = figure.get("agent_output")
    if fig_block is False:
        return None
    if fig_block is None and not root:
        return None
    if not isinstance(fig_block, Mapping):
        fig_block = {}
    if not isinstance(root, Mapping):
        root = {}
    merged = {**dict(root), **dict(fig_block)}
    if merged.get("enable", True) is False:
        return None

    max_cells = merged.get("max_cells")
    if max_cells is None and ("xbin" in merged or "ybin" in merged):
        xb = int(merged.get("xbin") or 1)
        yb = int(merged.get("ybin") or 1)
        max_cells = max(1, xb) * max(1, yb)
    if max_cells is None:
        max_cells = _MAX_CELLS_DEFAULT
    max_cells = int(max_cells)

    name = figure.get("name")
    fig_name = str(name).strip() if isinstance(name, str) and name.strip() else f"figure_{figure_index}"

    include = merged.get("include")
    if not isinstance(include, list) or not include:
        include = ["quantiles", "top_cells", "tails", "nan_stats", "provenance"]

    return AgentOutputSpec(
        enable=True,
        format=str(merged.get("format") or "json").strip().lower() or "json",
        method=str(merged.get("method") or "voronoi").strip().lower() or "voronoi",
        path=str(merged.get("path") or "auto").strip() or "auto",
        max_cells=max_cells,
        seed=int(merged.get("seed") or 0),
        geometry=str(merged.get("geometry") or "none").strip().lower() or "none",
        weight=merged.get("weight", "auto"),
        include=[str(x) for x in include],
        strict=bool(merged.get("strict", False)),
        figure_name=fig_name,
        figure_index=figure_index,
        raw=dict(merged),
    )


def resolve_export_path(
    spec: AgentOutputSpec,
    *,
    output_dir: str | Path,
    base_dir: str | Path | None = None,
) -> Path:
    out_dir = Path(output_dir).expanduser()
    raw = spec.path
    if raw in {"", "auto", "none", "null"}:
        return out_dir / f"{spec.figure_name}.agent.json"
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    # bare filename → under output.dir; nested relative → under base_dir or output.dir
    if path.parent == Path("."):
        return out_dir / path.name
    base = Path(base_dir).expanduser() if base_dir else out_dir
    return (base / path).resolve()


def plan_agent_exports(
    config: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
    bag: DiagnosticBag | None = None,
) -> list[dict[str, Any]]:
    """Doctor-facing planned exports (never writes files)."""
    bag = bag if bag is not None else DiagnosticBag()
    if not isinstance(config, Mapping):
        return []
    output = config.get("output") if isinstance(config.get("output"), Mapping) else {}
    out_dir = output.get("dir") or "./plots"
    if base_dir and not Path(str(out_dir)).is_absolute():
        out_dir_path = Path(base_dir) / str(out_dir)
    else:
        out_dir_path = Path(str(out_dir))

    exports: list[dict[str, Any]] = []
    for index, figure in enumerate(config.get("Figures") or ()):
        if not isinstance(figure, Mapping):
            continue
        if figure.get("enable", True) is False:
            continue
        try:
            spec = parse_agent_output(figure, figure_index=index, root_output=output)
        except Exception as exc:
            bag.error(
                "JP-AGT-002",
                f"$.Figures[{index}].agent_output",
                f"invalid agent_output: {exc}",
            )
            continue
        if spec is None:
            continue
        status = "planned"
        notes: list[str] = []
        if spec.format != "json":
            status = "invalid"
            bag.error(
                "JP-AGT-002",
                f"$.Figures[{index}].agent_output.format",
                f"unsupported agent_output.format {spec.format!r} (v1 only json)",
            )
        if spec.method not in {"voronoi", "bridson"}:
            status = "invalid"
            bag.error(
                "JP-AGT-002",
                f"$.Figures[{index}].agent_output.method",
                f"unknown agent_output.method {spec.method!r}; use voronoi (or bridson alias)",
            )
        if spec.max_cells < 4:
            status = "invalid"
            bag.error(
                "JP-AGT-001",
                f"$.Figures[{index}].agent_output.max_cells",
                f"max_cells={spec.max_cells} too small (min 4)",
            )
        elif spec.max_cells > _MAX_CELLS_HARD:
            status = "invalid"
            bag.error(
                "JP-AGT-001",
                f"$.Figures[{index}].agent_output.max_cells",
                f"max_cells={spec.max_cells} exceeds hard cap {_MAX_CELLS_HARD}",
            )
        axes_ok, axis_note = _figure_has_2d_axes(figure)
        if not axes_ok:
            status = "invalid"
            bag.error(
                "JP-AGT-004",
                f"$.Figures[{index}]",
                axis_note or "figure has no usable 2D x/y for agent digest",
            )
        path = resolve_export_path(spec, output_dir=out_dir_path, base_dir=base_dir)
        exports.append(
            {
                "figure": spec.figure_name,
                "format": spec.format,
                "method": "voronoi" if spec.method == "bridson" else spec.method,
                "max_cells": spec.max_cells,
                "path": str(path),
                "status": status,
                "notes": notes,
                "seed": spec.seed,
                "geometry": spec.geometry,
            }
        )
    return exports


def _figure_has_2d_axes(figure: Mapping[str, Any]) -> tuple[bool, str]:
    ao = figure.get("agent_output") if isinstance(figure.get("agent_output"), Mapping) else {}
    stash = ao.get("_digest_axes") if isinstance(ao, Mapping) else None
    if isinstance(stash, Mapping) and stash.get("x") is not None and stash.get("y") is not None:
        return True, ""
    if figure.get("x") is not None and figure.get("y") is not None:
        return True, ""
    if "type" in figure:
        return False, "type figure needs x and y for agent_output"
    layers = figure.get("layers") or []
    if not isinstance(layers, list) or not layers:
        return False, "layers figure needs at least one layer with x/y coordinates"
    for layer in layers:
        if not isinstance(layer, Mapping):
            continue
        coords = layer.get("coordinates") if isinstance(layer.get("coordinates"), Mapping) else {}
        if "x" in coords and "y" in coords:
            return True, ""
    return False, "no layer provides coordinates.x and coordinates.y"


def build_voronoi_digest(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    max_cells: int = _MAX_CELLS_DEFAULT,
    seed: int = 0,
    axes_meta: Mapping[str, Any] | None = None,
    figure_name: str = "figure",
    source_rows: int | None = None,
    source_hash: str | None = None,
    yaml_path: str | None = None,
    include: Sequence[str] | None = None,
    jplot_version: str = "0.0.0",
) -> dict[str, Any]:
    """Compress (x,y[,z,w]) into a lossy nearest-site (Voronoi generator) digest."""
    include = list(include or ["quantiles", "top_cells", "tails", "nan_stats", "provenance"])
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = int(min(x.size, y.size))
    x, y = x[:n], y[:n]
    if z is not None:
        z = np.asarray(z, dtype=float).reshape(-1)[:n]
    if weight is not None:
        weight = np.asarray(weight, dtype=float).reshape(-1)[:n]

    finite = np.isfinite(x) & np.isfinite(y)
    if z is not None:
        # z may be optional for mass-only digests
        z_fin = np.isfinite(z)
    else:
        z_fin = np.ones(n, dtype=bool)
    if weight is not None:
        w_fin = np.isfinite(weight) & (weight >= 0)
    else:
        w_fin = np.ones(n, dtype=bool)

    usable = finite & w_fin
    nan_rows = int((~finite).sum())
    # inf counted in nan_rows-ish; track separately
    inf_rows = int(((~np.isfinite(x)) | (~np.isfinite(y))).sum() - int(np.isnan(x).sum() + np.isnan(y).sum()) // 2)
    inf_rows = max(0, int((np.isinf(x) | np.isinf(y)).sum()))

    xu, yu = x[usable], y[usable]
    zu = z[usable] if z is not None else None
    wu = weight[usable] if weight is not None else np.ones(xu.size, dtype=float)
    finite_rows = int(xu.size)
    source_rows = int(source_rows if source_rows is not None else n)

    max_cells = max(4, min(int(max_cells), _MAX_CELLS_HARD))
    rng = np.random.default_rng(int(seed))

    cells: list[dict[str, Any]] = []
    if finite_rows == 0:
        actual_cells = 0
        sites = np.zeros((0, 2), dtype=float)
    else:
        n_sites = min(max_cells, finite_rows)
        # Weighted sampling of generators without replacement (approximate).
        if wu.sum() <= 0:
            probs = None
        else:
            probs = wu / wu.sum()
        if probs is None:
            pick = rng.choice(finite_rows, size=n_sites, replace=False)
        else:
            # choice with p requires replace=True if n large; use Gumbel top-k for w/o replace
            g = np.log(np.maximum(probs, 1e-300)) + rng.gumbel(size=finite_rows)
            pick = np.argpartition(-g, n_sites - 1)[:n_sites]
        sites = np.column_stack([xu[pick], yu[pick]])
        # Assign every usable sample to nearest site
        # chunked for memory
        owners = np.empty(finite_rows, dtype=np.int32)
        chunk = 50_000
        for start in range(0, finite_rows, chunk):
            stop = min(start + chunk, finite_rows)
            pts = np.column_stack([xu[start:stop], yu[start:stop]])
            # squared distances to sites
            d2 = ((pts[:, None, :] - sites[None, :, :]) ** 2).sum(axis=2)
            owners[start:stop] = np.argmin(d2, axis=1)

        for i in range(n_sites):
            mask = owners == i
            cnt = int(mask.sum())
            if cnt == 0:
                cells.append(
                    {
                        "i": i,
                        "site": [float(sites[i, 0]), float(sites[i, 1])],
                        "bbox": [
                            float(sites[i, 0]),
                            float(sites[i, 0]),
                            float(sites[i, 1]),
                            float(sites[i, 1]),
                        ],
                        "count": 0,
                        "mass": 0.0,
                        "weight_mean": 0.0,
                        "z_mean": None,
                        "z_min": None,
                        "z_max": None,
                        "density": 0.0,
                        "flags": ["empty"],
                    }
                )
                continue
            xs, ys = xu[mask], yu[mask]
            ws = wu[mask]
            mass = float(ws.sum())
            w_mean = float(mass / cnt) if cnt else 0.0
            bbox = [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())]
            width = bbox[1] - bbox[0]
            height = bbox[3] - bbox[2]
            # Degenerate (point/line) cells: do NOT invent a microscopic area —
            # that made density ~1e60 and poisoned top_density rankings.
            degenerate = (cnt < 3) or (width <= 0.0) or (height <= 0.0)
            flags: list[str] = []
            if degenerate:
                flags.append("degenerate")
                dens = 0.0
                area = 0.0
            else:
                area = width * height
                dens = float(mass / area) if mass > 0 and area > 0 else 0.0
            rec: dict[str, Any] = {
                "i": i,
                "site": [float(sites[i, 0]), float(sites[i, 1])],
                "bbox": bbox,
                "count": cnt,
                "mass": mass,
                "weight_mean": w_mean,
                "density": dens,
                "area": area,
                "flags": flags,
            }
            if zu is not None:
                zz = zu[mask]
                zf = zz[np.isfinite(zz)]
                if zf.size:
                    rec["z_mean"] = float(zf.mean())
                    rec["z_min"] = float(zf.min())
                    rec["z_max"] = float(zf.max())
                else:
                    rec["z_mean"] = rec["z_min"] = rec["z_max"] = None
            else:
                rec["z_mean"] = rec["z_min"] = rec["z_max"] = None
            cells.append(rec)

        # normalize mass
        total_mass = sum(c["mass"] for c in cells) or 1.0
        for c in cells:
            c["mass"] = float(c["mass"] / total_mass)

        # flag tails among non-empty, non-degenerate cells with positive density
        rankable = [
            c
            for c in cells
            if c["count"] > 0
            and "degenerate" not in (c.get("flags") or [])
            and (c.get("density") or 0.0) > 0
        ]
        if rankable:
            dens_vals = np.array([c["density"] for c in rankable], dtype=float)
            thr = float(np.quantile(dens_vals, 0.1)) if dens_vals.size else 0.0
            for c in rankable:
                if c["density"] <= thr:
                    c["flags"].append("tail")
        actual_cells = n_sites

    # global stats
    global_block: dict[str, Any] = {}
    if "quantiles" in include and finite_rows:
        global_block["quantiles"] = {
            "x": _quantiles(xu),
            "y": _quantiles(yu),
        }
        if zu is not None:
            global_block["quantiles"]["z"] = _quantiles(zu[np.isfinite(zu)])
        global_block["quantiles"]["weight"] = _quantiles(wu)
    if finite_rows and wu.sum() > 0:
        global_block["weighted_centroid"] = [
            float(np.average(xu, weights=wu)),
            float(np.average(yu, weights=wu)),
        ]
        # Kish ESS
        wsum = float(wu.sum())
        w2 = float(np.square(wu).sum())
        global_block["ess"] = float((wsum * wsum) / w2) if w2 > 0 else 0.0
    if "nan_stats" in include:
        global_block["nan_rows"] = nan_rows
        global_block["inf_rows"] = inf_rows

    highlights: dict[str, Any] = {}
    if "top_cells" in include and cells:
        # Never rank degenerate cells by density (area≈0 would dominate).
        dens_pool = [
            c
            for c in cells
            if c.get("count", 0) > 0
            and "degenerate" not in (c.get("flags") or [])
            and (c.get("density") or 0.0) > 0
        ]
        mass_pool = [c for c in cells if c.get("count", 0) > 0]
        by_dens = sorted(dens_pool, key=lambda c: c.get("density") or 0.0, reverse=True)
        by_mass = sorted(mass_pool, key=lambda c: c.get("mass") or 0.0, reverse=True)
        highlights["top_density"] = [
            {"cell_index": c["i"], "density": c["density"], "count": c["count"]}
            for c in by_dens[:10]
        ]
        highlights["top_mass"] = [
            {"cell_index": c["i"], "mass": c["mass"], "count": c["count"]}
            for c in by_mass[:10]
        ]
    if "tails" in include and cells:
        highlights["tails"] = [
            {"cell_index": c["i"], "flags": list(c.get("flags") or [])}
            for c in cells
            if "tail" in (c.get("flags") or [])
        ][:20]

    n_degenerate = sum(1 for c in cells if "degenerate" in (c.get("flags") or []))
    payload = {
        "schema_version": 1,
        "kind": "agent_digest",
        "figure": figure_name,
        "lossy": True,
        "algorithm": {
            "method": "voronoi",
            "version": AGENT_DIGEST_VERSION,
            "seed": int(seed),
            "max_cells": int(max_cells),
            "actual_cells": int(actual_cells),
            "partition": "nearest_site",
            "area_mode": "bbox",
            # Cells excluded from density ranking (point/line or count<3).
            "excluded_cells": int(n_degenerate),
        },
        "provenance": {
            "source_rows": source_rows,
            "finite_rows": finite_rows,
            "nan_rows": nan_rows,
            "inf_rows": inf_rows,
            "source_hash": source_hash,
            "yaml_path": yaml_path,
            "jarvisplot_version": jplot_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "axes": dict(axes_meta or {}),
        "global": global_block,
        "highlights": highlights,
        "cells": cells,
    }
    return payload


def _quantiles(arr: np.ndarray) -> dict[str, float]:
    if arr is None or arr.size == 0:
        return {}
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {}
    qs = np.quantile(a, [0.05, 0.5, 0.95])
    return {"q05": float(qs[0]), "q50": float(qs[1]), "q95": float(qs[2])}


def write_digest_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def maybe_write_figure_digest(
    *,
    figure_cfg: Mapping[str, Any],
    config: Mapping[str, Any],
    dataframe: Any,
    base_dir: str | Path | None = None,
    yaml_path: str | None = None,
    logger=None,
    jplot_version: str = "0.0.0",
    source_hash: str | None = None,
) -> Path | None:
    """Write digest for one figure if agent_output is set. Returns path or None."""
    from .utils.expression import eval_dataframe_expression

    output = config.get("output") if isinstance(config.get("output"), Mapping) else {}
    try:
        spec = parse_agent_output(figure_cfg, root_output=output)
    except Exception as exc:
        if logger:
            logger.warning(f"agent_output parse failed: {exc}")
        if False:  # strict handled below
            pass
        return None
    if spec is None:
        return None
    if spec.format != "json" or spec.method not in {"voronoi", "bridson"}:
        if logger:
            logger.warning(
                f"agent_output skipped for {spec.figure_name}: "
                f"format={spec.format} method={spec.method}"
            )
        if spec.strict:
            raise ValueError(f"invalid agent_output on figure {spec.figure_name}")
        return None

    out_dir = output.get("dir") or "./plots"
    if base_dir and not Path(str(out_dir)).is_absolute():
        out_dir = Path(base_dir) / str(out_dir)
    path = resolve_export_path(spec, output_dir=out_dir, base_dir=base_dir)

    axes_meta, exprs = _resolve_axes_and_exprs(figure_cfg, spec)
    try:
        import pandas as pd

        df = dataframe
        if df is None:
            raise ValueError("no dataframe for agent digest")
        if not isinstance(df, pd.DataFrame):
            from .utils.dataframes import polars_to_pandas

            df = polars_to_pandas(df)
        x = np.asarray(eval_dataframe_expression(df, exprs["x"]), dtype=float)
        y = np.asarray(eval_dataframe_expression(df, exprs["y"]), dtype=float)
        z = None
        if exprs.get("z"):
            z = np.asarray(eval_dataframe_expression(df, exprs["z"]), dtype=float)
        w = None
        if exprs.get("weight"):
            w = np.asarray(eval_dataframe_expression(df, exprs["weight"]), dtype=float)
    except Exception as exc:
        msg = f"agent digest data eval failed for {spec.figure_name}: {exc}"
        if logger:
            logger.error(msg) if spec.strict else logger.warning(msg)
        if spec.strict:
            raise
        return None

    if source_hash is None:
        source_hash = _hash_arrays(x, y, z, w)

    payload = build_voronoi_digest(
        x=x,
        y=y,
        z=z,
        weight=w,
        max_cells=spec.max_cells,
        seed=spec.seed,
        axes_meta=axes_meta,
        figure_name=spec.figure_name,
        source_rows=int(len(x)),
        source_hash=source_hash,
        yaml_path=yaml_path,
        include=spec.include,
        jplot_version=jplot_version,
    )
    try:
        written = write_digest_json(payload, path)
    except Exception as exc:
        msg = f"agent digest write failed for {spec.figure_name}: {exc}"
        if logger:
            logger.error(msg) if spec.strict else logger.warning(msg)
        if spec.strict:
            raise
        return None
    if logger:
        logger.warning(
            f"Agent digest written -> {written} "
            f"(cells={payload['algorithm']['actual_cells']}/{payload['algorithm']['max_cells']})"
        )
    return written


def _resolve_axes_and_exprs(
    figure: Mapping[str, Any], spec: AgentOutputSpec
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return (axes metadata, expr strings for x/y/z/weight)."""

    def _expr_of(field: Any, fallback: str | None = None) -> str | None:
        if field is None:
            return fallback
        if isinstance(field, Mapping):
            e = field.get("expr", field.get("name"))
            return str(e) if e is not None else fallback
        return str(field)

    def _meta_of(field: Any, name: str) -> dict[str, Any]:
        out: dict[str, Any] = {"name": name}
        if isinstance(field, Mapping):
            if "expr" in field:
                out["expr"] = field.get("expr")
            if "lim" in field:
                out["lim"] = field.get("lim")
            elif "limits" in field:
                out["lim"] = field.get("limits")
            if "scale" in field:
                out["scale"] = field.get("scale")
            if "label" in field:
                out["label"] = field.get("label")
        elif field is not None:
            out["expr"] = field
        return out

    axes: dict[str, Any] = {}
    exprs: dict[str, str] = {}

    # Stashed by type expand (x/y/weight removed from figure body).
    ao = figure.get("agent_output") if isinstance(figure.get("agent_output"), Mapping) else {}
    stash = ao.get("_digest_axes") if isinstance(ao, Mapping) else None
    if isinstance(stash, Mapping) and stash.get("x") is not None and stash.get("y") is not None:
        axes["x"] = _meta_of(stash.get("x"), "x")
        axes["y"] = _meta_of(stash.get("y"), "y")
        exprs["x"] = _expr_of(stash.get("x")) or "x"
        exprs["y"] = _expr_of(stash.get("y")) or "y"
        if stash.get("z") is not None:
            axes["z"] = _meta_of(stash.get("z"), "z")
            exprs["z"] = _expr_of(stash.get("z")) or "z"
        wfield = stash.get("weight") if spec.weight in (None, "auto") else spec.weight
        if wfield is not None and wfield != "auto":
            axes["weight"] = _meta_of(wfield, "weight")
            exprs["weight"] = _expr_of(wfield) or "1"
        return axes, exprs

    # Type macros or any figure that still carries top-level x/y (pre-expand / digest helpers).
    if figure.get("x") is not None and figure.get("y") is not None:
        axes["x"] = _meta_of(figure.get("x"), "x")
        axes["y"] = _meta_of(figure.get("y"), "y")
        exprs["x"] = _expr_of(figure.get("x")) or "x"
        exprs["y"] = _expr_of(figure.get("y")) or "y"
        if figure.get("z") is not None:
            axes["z"] = _meta_of(figure.get("z"), "z")
            exprs["z"] = _expr_of(figure.get("z")) or "z"
        wfield = figure.get("weight") if spec.weight in (None, "auto") else spec.weight
        if wfield is not None and wfield != "auto":
            axes["weight"] = _meta_of(wfield, "weight")
            exprs["weight"] = _expr_of(wfield) or "1"
        elif figure.get("weight") is not None:
            axes["weight"] = _meta_of(figure.get("weight"), "weight")
            exprs["weight"] = _expr_of(figure.get("weight")) or "1"
        return axes, exprs

    # layers form: first layer with x/y
    for layer in figure.get("layers") or ():
        if not isinstance(layer, Mapping):
            continue
        coords = layer.get("coordinates") if isinstance(layer.get("coordinates"), Mapping) else {}
        if "x" not in coords or "y" not in coords:
            continue
        axes["x"] = _meta_of(coords.get("x"), "x")
        axes["y"] = _meta_of(coords.get("y"), "y")
        exprs["x"] = _expr_of(coords.get("x")) or "x"
        exprs["y"] = _expr_of(coords.get("y")) or "y"
        if "z" in coords:
            axes["z"] = _meta_of(coords.get("z"), "z")
            exprs["z"] = _expr_of(coords.get("z")) or "z"
        if "c" in coords and "z" not in exprs:
            axes["z"] = _meta_of(coords.get("c"), "c")
            exprs["z"] = _expr_of(coords.get("c")) or "c"
        if spec.weight not in (None, "auto"):
            axes["weight"] = _meta_of(spec.weight, "weight")
            exprs["weight"] = _expr_of(spec.weight) or "1"
        break
    return axes, exprs


def _hash_arrays(*arrays: np.ndarray | None) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        if arr is None:
            h.update(b"none")
            continue
        a = np.asarray(arr)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        # sample for speed
        flat = a.reshape(-1)
        if flat.size:
            step = max(1, flat.size // 1000)
            sample = flat[::step]
            h.update(sample.tobytes())
    return "sha256:" + h.hexdigest()[:16]


def load_figure_source_dataframe(core, figure_cfg: Mapping[str, Any]):
    """Best-effort load of the primary dataset for a figure from Core context."""
    source = figure_cfg.get("data")
    ao = figure_cfg.get("agent_output") if isinstance(figure_cfg.get("agent_output"), Mapping) else {}
    stash = ao.get("_digest_axes") if isinstance(ao, Mapping) else None
    if source is None and isinstance(stash, Mapping):
        source = stash.get("data")
    if source is None:
        layers = figure_cfg.get("layers") or []
        if layers and isinstance(layers[0], Mapping):
            data = layers[0].get("data")
            if isinstance(data, list) and data and isinstance(data[0], Mapping):
                source = data[0].get("source")
            elif isinstance(data, Mapping):
                source = data.get("source")
    if isinstance(source, list):
        source = source[0] if source else None
    if not isinstance(source, str) or not source.strip():
        return None
    name = source.strip()
    # Prefer dataset registry / ctx
    reg = getattr(core, "dataset_registry", None) or {}
    if name in reg:
        ds = reg[name]
        if hasattr(ds, "get_data"):
            return ds.get_data()
    ctx = getattr(core, "ctx", None)
    if ctx is not None and hasattr(ctx, "get"):
        try:
            return ctx.get(name)
        except Exception:
            pass
    return None

#!/usr/bin/env python3

"""One-dimensional weighted binning as a transform step.

``bin_stat`` collapses an event table into one row per bin.  It is deliberately
generic -- it knows about a coordinate, a weight and a normalisation, and
nothing about what the numbers mean -- so the physics stays in the YAML where
it can be read and changed, rather than inside a plotting method.

Output columns::

    bin_index  x_lo  x_hi  x_center  <out>  [<scale_to.name>]

``<out>`` defaults to ``density``.  Which normalisation that name refers to is
spelled out rather than assumed, because the word is used for two conventions
that differ by a bin width -- and confusing them scales a downstream
significance by the square root of the bin count, not by a rounding error.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from ..utils.expression import eval_dataframe_expression

__all__ = [
    "is_bin_stat_transform",
    "bin_stat_config",
    "bin_stat",
    "bin_stat_input_columns",
    "bin_stat_output_columns",
]

#: ``sum`` makes the bins add to 1, so ``scale_to.value * density`` is directly
#: the expected count in that bin.  ``integral`` is matplotlib's ``density=True``
#: (the bins integrate to 1).  ``none`` leaves the weighted sum alone.
NORMALISE_MODES = ("sum", "integral", "none")

_FIXED_OUTPUTS = ("bin_index", "x_lo", "x_hi", "x_center")


def is_bin_stat_transform(step: Any) -> bool:
    if not isinstance(step, Mapping):
        return False
    if "bin_stat" in step:
        return True
    return str(step.get("type", "")).strip().lower() == "bin_stat"


def bin_stat_config(step: Mapping[str, Any]) -> dict:
    if "bin_stat" in step:
        cfg = step.get("bin_stat")
        return dict(cfg) if isinstance(cfg, Mapping) else {}
    cfg = dict(step)
    cfg.pop("type", None)
    return cfg


def _scale_spec(cfg: Mapping[str, Any]) -> Optional[dict]:
    spec = cfg.get("scale_to")
    if not isinstance(spec, Mapping):
        return None
    name = spec.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("bin_stat scale_to needs a 'name'")
    if "value" not in spec:
        raise ValueError("bin_stat scale_to needs a 'value'")
    return {"name": name.strip(), "value": spec.get("value")}


def bin_stat_input_columns(cfg: Mapping[str, Any]) -> set:
    """Columns this step reads, so the column projection keeps them."""
    from ..expr_names import expr_identifiers

    out: set = set()
    for key in ("x", "weights"):
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            out.update(expr_identifiers(value))
    try:
        spec = _scale_spec(cfg)
    except ValueError:
        spec = None
    if spec is not None and isinstance(spec["value"], str):
        out.update(expr_identifiers(spec["value"]))
    return out


def bin_stat_output_columns(cfg: Mapping[str, Any]) -> set:
    """Columns this step writes; without these the projection prunes them."""
    out = set(_FIXED_OUTPUTS)
    out.add(str(cfg.get("out", "density")).strip() or "density")
    try:
        spec = _scale_spec(cfg)
    except ValueError:
        spec = None
    if spec is not None:
        out.add(spec["name"])
    return out


def _edges(cfg: Mapping[str, Any]) -> np.ndarray:
    bins = cfg.get("bins")
    if bins is None:
        raise ValueError("bin_stat needs 'bins'")
    if isinstance(bins, (list, tuple, np.ndarray)):
        edges = np.asarray(bins, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("bin_stat bins list must hold at least two edges")
        if not np.all(np.diff(edges) > 0):
            raise ValueError("bin_stat bins list must increase")
        return edges
    count = int(bins)
    if count < 1:
        raise ValueError("bin_stat bins must be a positive integer")
    rng = cfg.get("range")
    if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
        raise ValueError("bin_stat needs 'range' when 'bins' is a count")
    lo, hi = float(rng[0]), float(rng[1])
    if not hi > lo:
        raise ValueError(f"bin_stat range must increase, got [{lo}, {hi}]")
    return np.linspace(lo, hi, count + 1)


def _series(df: pd.DataFrame, expr: Any, what: str, logger=None) -> np.ndarray:
    if expr is None:
        raise ValueError(f"bin_stat needs '{what}'")
    values = eval_dataframe_expression(df, expr, logger=logger, allow_column=True)
    return np.asarray(values, dtype=float).reshape(-1)


def _resolve_scale(df: pd.DataFrame, value: Any, logger=None) -> float:
    """A literal, or a column that has to agree with itself across the block."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    series = eval_dataframe_expression(df, value, logger=logger, allow_column=True)
    arr = np.asarray(series, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"bin_stat scale_to value {value!r} has no finite rows")
    first = float(arr[0])
    if not np.allclose(arr, first, rtol=0, atol=0):
        raise ValueError(
            f"bin_stat scale_to value {value!r} is not constant over the block "
            f"(min={arr.min():.6g}, max={arr.max():.6g}); filter first, or pass a number"
        )
    return first


def bin_stat(df, cfg: Mapping[str, Any], logger=None) -> pd.DataFrame:
    """Collapse ``df`` into one row per bin of ``cfg['x']``."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("bin_stat needs a pandas table")

    edges = _edges(cfg)
    x = _series(df, cfg.get("x"), "x", logger=logger)
    weights_expr = cfg.get("weights")
    if weights_expr is None:
        w = np.ones_like(x, dtype=float)
    else:
        w = _series(df, weights_expr, "weights", logger=logger)
        if w.size != x.size:
            raise ValueError("bin_stat x and weights must have the same length")

    mode = str(cfg.get("normalise", "sum")).strip().lower()
    if mode not in NORMALISE_MODES:
        raise ValueError(
            f"bin_stat normalise must be one of {', '.join(NORMALISE_MODES)}; got {mode!r}"
        )

    finite = np.isfinite(x) & np.isfinite(w)
    totals, _ = np.histogram(x[finite], bins=edges, weights=w[finite])

    widths = np.diff(edges)
    values = totals.astype(float, copy=True)
    if mode != "none":
        # Only what landed in range takes part, so the normalisation matches the
        # table this step actually returns rather than the rows it was handed.
        grand = float(totals.sum())
        if grand == 0.0:
            raise ValueError(
                "bin_stat cannot normalise: the binned weights sum to zero "
                "(signed weights can cancel -- use normalise: none to keep raw sums)"
            )
        values = values / grand
        if mode == "integral":
            values = values / widths

    out_name = str(cfg.get("out", "density")).strip() or "density"
    table = pd.DataFrame(
        {
            "bin_index": np.arange(len(widths), dtype=int),
            "x_lo": edges[:-1],
            "x_hi": edges[1:],
            "x_center": 0.5 * (edges[:-1] + edges[1:]),
            out_name: values,
        }
    )

    spec = _scale_spec(cfg)
    if spec is not None:
        scale = _resolve_scale(df, spec["value"], logger=logger)
        table[spec["name"]] = scale * table[out_name]

    if logger:
        try:
            logger.debug(
                "bin_stat -> {} bins over [{:.6g}, {:.6g}], normalise={}, rows {} -> {}".format(
                    len(widths), edges[0], edges[-1], mode, len(df), len(table)
                )
            )
        except Exception:
            pass
    return table

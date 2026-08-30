#!/usr/bin/env python3

"""Pearson correlation tables from event-level columns.

``correlation`` is deliberately an unweighted transform.  It turns a wide
event table into a long table, one row per requested variable pair, so the
result can either be written with ``to_csv`` or drawn directly by a later
JarvisPLOT layer.  The default listwise finite-row policy reproduces the
correlation calculation used for the RJR BDT input pruning.

Output columns::

    var_x  var_y  x_index  y_index  rho  abs_rho  n

``x_index`` and ``y_index`` are the positions of the two variables in
``columns``, so a layer draws the matrix by putting them on the two axes.
The step also publishes the private ``__grid_*`` columns the grid-consuming
methods read, because a long table alone does not say how big its matrix is:
without them ``imshow`` has no shape at all and ``pcolormesh`` falls back to
guessing one from the row count, which a ``triangle`` selection makes wrong
without any complaint.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "is_correlation_transform",
    "correlation_config",
    "correlation",
    "correlation_input_columns",
    "correlation_output_columns",
    "correlation_selects_dynamically",
    "resolve_correlation_columns",
    "pearson_matrix",
    "pearson_pvalues",
]


_TRIANGLES = ("full", "upper", "lower")
_MISSING_POLICIES = ("listwise", "pairwise")
_OUTPUT_COLUMNS = {
    "var_x",
    "var_y",
    "x_index",
    "y_index",
    "rho",
    "abs_rho",
    "n",
}

#: Grid metadata the drawing side reads to recover the matrix.  It is the
#: same private protocol ``profile: {method: grid}`` already speaks, so
#: ``imshow``, ``pcolormesh`` and ``contour`` pick it up with no new plotting
#: code.  The extent is the index convention matplotlib uses for an image:
#: cell ``k`` is centred on ``k`` and spans ``[k - 0.5, k + 0.5]``.
_GRID_COLUMNS = {
    "__grid_ix__",
    "__grid_iy__",
    "__grid_nx__",
    "__grid_ny__",
    "__grid_xmin__",
    "__grid_xmax__",
    "__grid_ymin__",
    "__grid_ymax__",
}


def is_correlation_transform(step: Any) -> bool:
    if not isinstance(step, Mapping):
        return False
    if "correlation" in step:
        return True
    return str(step.get("type", "")).strip().lower() == "correlation"


def correlation_config(step: Mapping[str, Any]) -> dict:
    if "correlation" in step:
        cfg = step.get("correlation")
        return dict(cfg) if isinstance(cfg, Mapping) else {}
    cfg = dict(step)
    cfg.pop("type", None)
    return cfg


def _name_list(value: Any, what: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"correlation {what} must be a string or a list of strings")
    names = [str(name).strip() for name in value]
    if any(not name for name in names):
        raise ValueError(f"correlation {what} must not contain empty names")
    return names


def _explicit_columns(cfg: Mapping[str, Any]) -> list[str]:
    """The literal ``columns`` list, validated; empty when the step selects."""
    raw = cfg.get("columns")
    if raw is None:
        return []
    columns = _name_list(raw, "columns")
    duplicates = sorted({name for name in columns if columns.count(name) > 1})
    if duplicates:
        raise ValueError(f"correlation columns must be unique; repeated {duplicates}")
    return columns


def _selector(cfg: Mapping[str, Any]) -> dict:
    """Validate how this step picks its inputs, without needing the table."""
    columns = _explicit_columns(cfg)
    pattern = cfg.get("regex")
    if pattern is not None and not isinstance(pattern, str):
        raise ValueError("correlation regex must be a string")
    if pattern is not None:
        pattern = pattern.strip()
        if not pattern:
            raise ValueError("correlation regex must not be empty")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"correlation regex {cfg.get('regex')!r} is not valid: {exc}") from exc
    if columns and pattern is not None:
        raise ValueError(
            "correlation takes either 'columns' or 'regex', not both -- they are "
            "two ways of naming the same set. Use 'exclude' to trim either one."
        )
    return {
        "columns": columns,
        "regex": pattern,
        "exclude": _name_list(cfg.get("exclude"), "exclude"),
    }


def correlation_selects_dynamically(cfg: Mapping[str, Any]) -> bool:
    """Whether the input columns are only known once the table is in hand.

    A step that says ``regex`` or nothing at all cannot be answered from the
    YAML, so no static column projection over its source can be safe: pruning
    to the names the config happens to mention would hand it a table with the
    features already cut out.
    """
    try:
        return not _selector(cfg)["columns"]
    except ValueError:
        # A malformed selector is the runtime's error to report, with the real
        # message. Until then, assume the widest reading.
        return True


def _is_correlatable(series) -> bool:
    """Numeric and not a flag: bools are labels by construction, not features."""
    return bool(
        pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    )


def resolve_correlation_columns(
    names: Any, cfg: Mapping[str, Any], *, correlatable=None
) -> list[str]:
    """Turn the selector into the ordered column list, from names alone.

    Explicit ``columns`` keep the order they were written in; a ``regex`` or
    the bare default keeps the table's own column order.  Either way the order
    is what ``x_index`` counts, so it has to be stated rather than incidental.

    ``correlatable`` is a predicate on the column name.  The renderer answers it
    from the real dtypes; a caller that only has metadata answers it from a
    sampled schema, or passes ``None`` to accept every non-private name.  The
    selection lives here rather than in each caller because the axis labels are
    resolved before the figure is built and the cells are computed during the
    render: if those two ever disagree about which columns are in, the labels
    stop naming the cells they sit next to and nothing says so.
    """
    names = [str(name) for name in names]
    spec = _selector(cfg)
    exclude = set(spec["exclude"])

    if spec["columns"]:
        chosen = spec["columns"]
        tried = "columns"
    else:
        # The private bookkeeping columns are never features, and neither is a
        # non-numeric one; a table arriving with __grid_* already on it would
        # otherwise be correlated against its own metadata.
        auto = [
            name
            for name in names
            if not name.startswith("__") and (correlatable is None or correlatable(name))
        ]
        if spec["regex"] is not None:
            matcher = re.compile(spec["regex"])
            chosen = [name for name in auto if matcher.search(name)]
            tried = f"regex {spec['regex']!r}"
        else:
            chosen = auto
            tried = "every numeric column"

    kept = [name for name in chosen if name not in exclude]
    if len(kept) < 2:
        detail = f"{tried} selected {len(chosen)} column(s)"
        if exclude:
            detail += f", {len(chosen) - len(kept)} of them removed by 'exclude'"
        raise ValueError(
            f"correlation needs at least two columns to correlate, but {detail}. "
            f"The table has: {', '.join(names[:12])}"
            + (" ..." if len(names) > 12 else "")
        )
    return kept


def _resolve_columns(df: pd.DataFrame, cfg: Mapping[str, Any]) -> list[str]:
    """The render-time selection, answered from the real dtypes."""
    return resolve_correlation_columns(
        df.columns, cfg, correlatable=lambda name: _is_correlatable(df[name])
    )


def correlation_input_columns(cfg: Mapping[str, Any]) -> set:
    """Columns this transform reads, for projection and preflight checks.

    Only meaningful when the step names them; see
    :func:`correlation_selects_dynamically` for the case where it does not.
    """
    try:
        return set(_selector(cfg)["columns"])
    except ValueError:
        return set()


def correlation_output_columns(cfg: Mapping[str, Any]) -> set:
    """Stable long-table column names emitted by :func:`correlation`."""
    return set(_OUTPUT_COLUMNS) | set(_GRID_COLUMNS)


def _prepared_values(df, columns, *, missing: str = "listwise"):
    """The numeric block Pearson is computed on, under one missing policy.

    Coercion intentionally makes non-numeric entries missing instead of giving
    a surprising object-dtype result.  Inf is likewise not a valid Pearson
    input.
    """
    values = df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    values = values.where(np.isfinite(values))
    if str(missing).strip().lower() == "listwise":
        values = values.dropna(axis=0, how="any")
    return values


def pearson_matrix(
    df, columns, *, missing: str = "listwise", min_periods: int = 2
) -> pd.DataFrame:
    """The square correlation matrix, in ``columns`` order.

    :func:`correlation` emits a long table because that is what a layer draws;
    ordering a matrix needs it square.  Both go through here so an ``order:
    hclust`` resolved before the figure is built cannot be computed from a
    different matrix than the one that ends up on the page.
    """
    values = _prepared_values(df, columns, missing=missing)
    return values.corr(method="pearson", min_periods=int(min_periods))


def pearson_pvalues(rho, n):
    """Two-sided p-values for Pearson r, from the coefficient and its count.

    The long table already carries ``n`` per pair, so significance needs no
    second input -- which is the one place this departs from R, where marking
    insignificant cells means passing a separate ``p.mat``.  Pairs with fewer
    than three usable rows have no test and come back as NaN rather than as a
    confidently significant zero.
    """
    rho = np.asarray(rho, dtype=float)
    n = np.asarray(n, dtype=float)
    out = np.full(rho.shape, np.nan, dtype=float)
    dof = n - 2.0
    ok = np.isfinite(rho) & np.isfinite(dof) & (dof > 0) & (np.abs(rho) < 1.0)
    if not np.any(ok):
        return out
    try:
        from scipy import stats
    except Exception:
        return out
    r = rho[ok]
    d = dof[ok]
    t = r * np.sqrt(d / (1.0 - r * r))
    out[ok] = 2.0 * stats.t.sf(np.abs(t), d)
    # |r| == 1 with a real sample is exact, not untested.
    exact = np.isfinite(rho) & np.isfinite(dof) & (dof > 0) & (np.abs(rho) >= 1.0)
    out[exact] = 0.0
    return out


def _pair_indices(size: int, triangle: str, include_diagonal: bool):
    """Cells to emit, named by their position on the drawn matrix.

    ``i`` is ``x_index`` and ``j`` is ``y_index``, so ``upper`` keeps
    ``y_index >= x_index`` -- the half above the diagonal once the matrix is
    drawn with y increasing upward, which is what both ``imshow(origin=
    "lower")`` and ``pcolormesh`` do.  Read as numpy row/column indexing the
    same half is the lower one; the name follows the picture, not the array.

    Returned as two arrays in row-major order (``i`` outer, ``j`` inner), which
    is the order the long table is written in and the order a reader of the CSV
    expects: the first row of the matrix, then the second.
    """
    i, j = np.divmod(np.arange(size * size, dtype=np.int64), size)
    keep = np.ones(i.shape, dtype=bool)
    if triangle == "upper":
        keep &= j >= i
    elif triangle == "lower":
        keep &= j <= i
    if not include_diagonal:
        keep &= i != j
    return i[keep], j[keep]


def _attach_grid_metadata(out: pd.DataFrame, size: int) -> None:
    """Tell the drawing side the matrix shape instead of letting it guess.

    A ``triangle`` selection leaves holes, so the row count no longer implies
    the side length.  Saying it outright is what keeps a 6-variable upper
    triangle from being rendered on a 4x4 mesh -- silently, and reported as a
    successful figure.
    """
    out["__grid_ix__"] = np.asarray(out["x_index"], dtype=np.int32)
    out["__grid_iy__"] = np.asarray(out["y_index"], dtype=np.int32)
    out["__grid_nx__"] = np.int32(size)
    out["__grid_ny__"] = np.int32(size)
    out["__grid_xmin__"] = -0.5
    out["__grid_xmax__"] = float(size) - 0.5
    out["__grid_ymin__"] = -0.5
    out["__grid_ymax__"] = float(size) - 0.5


def correlation(df, cfg: Mapping[str, Any], logger=None) -> pd.DataFrame:
    """Return unweighted Pearson correlations in a long, plot-ready table."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("correlation needs a pandas table")

    columns = _resolve_columns(df, cfg)
    missing_columns = [name for name in columns if name not in df.columns]
    if missing_columns:
        raise ValueError(f"correlation cannot find columns {missing_columns}")

    policy = str(cfg.get("missing", "listwise")).strip().lower()
    if policy not in _MISSING_POLICIES:
        raise ValueError(
            "correlation missing must be one of {}; got {!r}".format(
                ", ".join(_MISSING_POLICIES), policy
            )
        )
    triangle = str(cfg.get("triangle", "full")).strip().lower()
    if triangle not in _TRIANGLES:
        raise ValueError(
            "correlation triangle must be one of {}; got {!r}".format(
                ", ".join(_TRIANGLES), triangle
            )
        )
    include_diagonal = bool(cfg.get("include_diagonal", True))
    min_periods = int(cfg.get("min_periods", 2))
    if min_periods < 1:
        raise ValueError("correlation min_periods must be positive")

    values = _prepared_values(df, columns, missing=policy)

    if logger and len(values) < min_periods:
        try:
            logger.warning(
                "correlation has {} usable row(s) after the {} finite-value cut, "
                "below min_periods={}; every rho will be NaN and the matrix will "
                "draw empty".format(len(values), policy, min_periods)
            )
        except Exception:
            pass

    matrix = values.corr(method="pearson", min_periods=min_periods)
    names = np.asarray(columns, dtype=object)
    rho_matrix = matrix.loc[columns, columns].to_numpy(dtype=float)
    ix, iy = _pair_indices(len(columns), triangle, include_diagonal)

    # The per-pair usable count, for every pair at once.  Under listwise the
    # policy already dropped every row with a hole in it, so the count is the
    # same number for all of them.  Under pairwise it is the number of rows
    # where both columns are present, which is exactly `mask.T @ mask` -- one
    # matrix product instead of n^2 column-wise ANDs, and at 100 variables
    # that loop was the most expensive thing this transform did.
    if policy == "listwise":
        counts = np.full(ix.shape, len(values), dtype=np.int64)
    else:
        mask = values.notna().to_numpy(dtype=np.int64)
        counts = (mask.T @ mask)[ix, iy]

    rho = rho_matrix[ix, iy]
    out = pd.DataFrame(
        {
            "var_x": names[ix],
            "var_y": names[iy],
            "x_index": ix,
            "y_index": iy,
            "rho": rho,
            "abs_rho": np.abs(rho),
            "n": counts,
        }
    )
    _attach_grid_metadata(out, len(columns))
    if logger:
        try:
            logger.debug(
                "correlation -> {} variables ({}), {} pairs, missing={}, rows {} -> {}".format(
                    len(columns),
                    ", ".join(columns[:8]) + (" ..." if len(columns) > 8 else ""),
                    len(out),
                    policy,
                    len(df),
                    len(values),
                )
            )
        except Exception:
            pass
    return out

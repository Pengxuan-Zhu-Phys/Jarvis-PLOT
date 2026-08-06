#!/usr/bin/env python3

"""``jplot data …`` -- look at a data file the way the renderer will.

``describe`` is the physical exit for the anti-hallucination rule
"column names only come from the data". ``head`` shows real rows;
``eval`` checks an expression before it is written into YAML.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from ..agent_io import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, envelope, error_payload
from ..diagnostics import did_you_mean
from ..expr_names import EXPR_IDENTIFIER_IGNORE

__all__ = [
    "SUBCOMMANDS",
    "build_parser",
    "describe_file",
    "eval_on_file",
    "head_file",
    "run",
    "suggest_axes",
]

SUBCOMMANDS = ("describe", "head", "eval", "suggest-axes")

_DESCRIBE_CACHE_VERSION = 1


def build_parser(prog: str = "jplot data") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Inspect a data file the way Jarvis-PLOT will load it.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    def _common_source(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--type",
            dest="file_type",
            choices=["csv", "parquet", "hdf5", "auto"],
            default="auto",
            help="force a loader (default: guess from extension)",
        )
        p.add_argument(
            "--group",
            default=None,
            help="HDF5 group to open (default: file root)",
        )
        p.add_argument(
            "--json",
            action="store_true",
            help="emit one JSON envelope on stdout",
        )

    describe = sub.add_parser(
        "describe",
        help="column names, dtypes, ranges (and HDF5 tree when applicable)",
    )
    describe.add_argument("file", help="path to csv / parquet / hdf5")
    _common_source(describe)
    describe.add_argument(
        "--no-rows",
        action="store_true",
        help="header-only: skip min/max/quantile stats (still lists columns)",
    )
    describe.add_argument(
        "--no-cache",
        action="store_true",
        help="bypass the workdir .cache/summary for this describe",
    )

    head = sub.add_parser(
        "head",
        help="first N real sample rows",
    )
    head.add_argument("file", help="path to csv / parquet / hdf5")
    _common_source(head)
    head.add_argument(
        "-n",
        "--n",
        dest="n_rows",
        type=int,
        default=5,
        help="number of rows (default: 5, hard cap 100)",
    )
    head.add_argument(
        "--cols",
        default=None,
        help="comma-separated column subset (default: all)",
    )

    evaluate = sub.add_parser(
        "eval",
        help="evaluate an expression against the file (sandbox before YAML)",
    )
    evaluate.add_argument("expr", help='expression, e.g. "exp(LogL)"')
    evaluate.add_argument(
        "--data",
        required=True,
        dest="file",
        help="path to csv / parquet / hdf5",
    )
    _common_source(evaluate)
    evaluate.add_argument(
        "--sample",
        type=int,
        default=5,
        help="how many sample values to return (default: 5)",
    )

    axes = sub.add_parser(
        "suggest-axes",
        help="per-column scale/lim suggestions for frame.ax",
    )
    axes.add_argument("file", help="path to csv / parquet / hdf5")
    _common_source(axes)
    axes.add_argument(
        "--cols",
        default=None,
        help="comma-separated column subset (default: all numeric)",
    )
    return parser


def run(argv: Sequence[str], *, prog: str = "jplot data") -> int:
    parser = build_parser(prog)
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        return int(exc.code or EXIT_USAGE)

    action = str(getattr(args, "action", "") or "")
    as_json = bool(getattr(args, "json", False)) or not sys.stdout.isatty()

    handlers = {
        "describe": _run_describe,
        "head": _run_head,
        "eval": _run_eval,
        "suggest-axes": _run_suggest_axes,
    }
    handler = handlers.get(action)
    if handler is None:
        env = envelope(
            f"data.{action or 'unknown'}",
            False,
            error=error_payload(
                "UsageError",
                f"data action {action!r} is not implemented yet "
                f"(available: describe, head, eval, suggest-axes)",
            ),
        )
        return emit(env) if as_json else _usage_err(env)

    return handler(args, as_json=as_json, prog=prog)


# --------------------------------------------------------------------------- #
# Subcommand runners
# --------------------------------------------------------------------------- #


def _run_describe(args, *, as_json: bool, prog: str) -> int:
    try:
        data = describe_file(
            args.file,
            file_type=args.file_type,
            group=args.group,
            stats=not args.no_rows,
            use_cache=not args.no_cache,
        )
    except Exception as exc:
        return _emit_failure("data.describe", args.file, exc, as_json=as_json)

    env = envelope("data.describe", True, data=data)
    if as_json:
        return emit(env)
    _print_describe(data)
    return EXIT_OK


def _run_head(args, *, as_json: bool, prog: str) -> int:
    try:
        data = head_file(
            args.file,
            n=int(args.n_rows),
            cols=args.cols,
            file_type=args.file_type,
            group=args.group,
        )
    except Exception as exc:
        return _emit_failure("data.head", args.file, exc, as_json=as_json)

    env = envelope("data.head", True, data=data)
    if as_json:
        return emit(env)
    _print_head(data)
    return EXIT_OK


def _run_suggest_axes(args, *, as_json: bool, prog: str) -> int:
    try:
        data = suggest_axes(
            args.file,
            cols=args.cols,
            file_type=args.file_type,
            group=args.group,
        )
    except Exception as exc:
        return _emit_failure("data.suggest_axes", args.file, exc, as_json=as_json)

    env = envelope("data.suggest_axes", True, data=data)
    if as_json:
        return emit(env)
    _print_suggest_axes(data)
    return EXIT_OK


def _run_eval(args, *, as_json: bool, prog: str) -> int:
    try:
        data = eval_on_file(
            args.expr,
            args.file,
            file_type=args.file_type,
            group=args.group,
            sample=int(args.sample),
        )
    except _EvalFailed as exc:
        env = envelope(
            "data.eval",
            False,
            data=exc.data,
            error=error_payload(exc.code, exc.message),
            diagnostics=[
                {
                    "code": exc.code,
                    "level": "error",
                    "path": "$.expr",
                    "message": exc.message,
                    "suggestion": exc.suggestion,
                    "context": {
                        k: v
                        for k, v in (exc.data or {}).items()
                        if k
                        in {
                            "available_columns",
                            "did_you_mean",
                            "available_functions",
                            "symbols_unresolved",
                        }
                    },
                }
            ],
        )
        if as_json:
            return emit(env)
        print(exc.message, file=sys.stderr)
        if exc.data.get("did_you_mean"):
            print(f"  did_you_mean: {exc.data['did_you_mean']}", file=sys.stderr)
        return EXIT_FAILED
    except Exception as exc:
        return _emit_failure("data.eval", getattr(args, "file", ""), exc, as_json=as_json)

    env = envelope("data.eval", True, data=data)
    if as_json:
        return emit(env)
    _print_eval(data)
    return EXIT_OK


def _emit_failure(kind: str, path: str, exc: BaseException, *, as_json: bool) -> int:
    env = envelope(kind, False, data={"path": str(path)}, error=error_payload(exc))
    if as_json:
        return emit(env)
    return _fail(env)


class _EvalFailed(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        data: dict[str, Any] | None = None,
        suggestion: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data or {}
        self.suggestion = suggestion


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def describe_file(
    path: str,
    *,
    file_type: str = "auto",
    group: str | None = None,
    stats: bool = True,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Return a machine-readable summary of one data file."""
    resolved = _resolve_path(path)
    kind = _detect_type(resolved, file_type)

    if use_cache:
        cached = _cache_get_describe(resolved, kind=kind, group=group, stats=stats)
        if cached is not None:
            cached = dict(cached)
            cached["cache"] = "hit"
            return cached

    if kind == "hdf5":
        payload = _describe_hdf5(resolved, group=group, stats=stats)
    elif kind in {"parquet", "csv"}:
        payload = _describe_tabular(resolved, kind=kind, stats=stats)
    else:
        raise ValueError(
            f"cannot detect file type for {resolved!r}; pass --type csv|parquet|hdf5"
        )

    payload["cache"] = "miss"
    if use_cache:
        _cache_put_describe(resolved, kind=kind, group=group, stats=stats, payload=payload)
    return payload


def head_file(
    path: str,
    *,
    n: int = 5,
    cols: str | None = None,
    file_type: str = "auto",
    group: str | None = None,
) -> dict[str, Any]:
    """Return the first ``n`` rows as JSON-serialisable records."""
    resolved = _resolve_path(path)
    kind = _detect_type(resolved, file_type)
    n = max(1, min(int(n), 100))
    col_list = _parse_cols(cols)

    df = _load_dataframe(
        resolved,
        kind=kind,
        group=group,
        nrows=n if kind == "csv" else None,
    )
    if col_list:
        missing = [c for c in col_list if c not in df.columns]
        if missing:
            available = [str(c) for c in df.columns]
            raise KeyError(
                f"unknown columns {missing}; available: {available}"
            )
        df = df.loc[:, col_list]
    head = df.head(n)
    rows = json.loads(head.to_json(orient="records", date_format="iso", default_handler=str))
    return {
        "path": resolved,
        "type": kind,
        "n": len(rows),
        "columns": [str(c) for c in head.columns],
        "rows": rows,
    }


def suggest_axes(
    path: str,
    *,
    cols: str | None = None,
    file_type: str = "auto",
    group: str | None = None,
) -> dict[str, Any]:
    """Suggest frame-style ``scale`` / ``lim`` for numeric columns."""
    import numpy as np
    from pandas.api import types as pdt

    resolved = _resolve_path(path)
    kind = _detect_type(resolved, file_type)
    df = _load_dataframe(resolved, kind=kind, group=group)
    wanted = _parse_cols(cols)
    axes: list[dict[str, Any]] = []
    for col in df.columns:
        name = str(col)
        if wanted is not None and name not in wanted:
            continue
        series = df[col]
        if not pdt.is_numeric_dtype(series.dtype) or pdt.is_bool_dtype(series.dtype):
            continue
        values = series.dropna().to_numpy(dtype=float, copy=False)
        if values.size == 0:
            axes.append(
                {
                    "col": name,
                    "scale": "linear",
                    "lim": None,
                    "reason": "no finite values",
                }
            )
            continue
        positive = bool(np.all(values > 0))
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        q_lo, q_hi = np.quantile(values, [0.005, 0.995])
        decades = float(np.log10(vmax / vmin)) if positive and vmin > 0 else 0.0
        if positive and decades >= 2.0:
            scale = "log"
            lo = _nice_log_bound(float(q_lo), direction="down")
            hi = _nice_log_bound(float(q_hi), direction="up")
            reason = (
                f"all positive; spans {decades:.2f} decades; "
                f"lim from q0.5%–q99.5% rounded outward on a log scale"
            )
        else:
            scale = "linear"
            lo, hi = _nice_linear_bounds(float(q_lo), float(q_hi))
            reason = (
                f"spans {decades:.2f} decades"
                if positive
                else "not strictly positive"
            ) + "; lim from q0.5%–q99.5% rounded outward"
        axes.append(
            {
                "col": name,
                "scale": scale,
                "lim": [lo, hi],
                "reason": reason,
                "stats": {
                    "min": vmin,
                    "max": vmax,
                    "q005": float(q_lo),
                    "q995": float(q_hi),
                    "positive": positive,
                    "decades": decades if positive else None,
                },
            }
        )
    return {"path": resolved, "type": kind, "axes": axes}


def _nice_linear_bounds(lo: float, hi: float) -> tuple[float, float]:
    import math

    if not math.isfinite(lo) or not math.isfinite(hi):
        return lo, hi
    if lo == hi:
        pad = abs(lo) * 0.1 if lo else 1.0
        return lo - pad, hi + pad
    span = hi - lo
    step = 10 ** math.floor(math.log10(span)) if span > 0 else 1.0
    # expand slightly so data is not flush with the frame
    lo2 = math.floor(lo / step) * step
    hi2 = math.ceil(hi / step) * step
    if lo2 == hi2:
        hi2 = lo2 + step
    return float(lo2), float(hi2)


def _nice_log_bound(value: float, *, direction: str) -> float:
    import math

    if value <= 0 or not math.isfinite(value):
        return value
    exp = math.floor(math.log10(value))
    base = 10.0**exp
    candidates = [m * base for m in (1.0, 2.0, 5.0, 10.0)]
    if direction == "down":
        below = [c for c in candidates if c <= value]
        return float(below[-1] if below else base / 10.0)
    above = [c for c in candidates if c >= value]
    return float(above[0] if above else base * 10.0)


def eval_on_file(
    expr: str,
    path: str,
    *,
    file_type: str = "auto",
    group: str | None = None,
    sample: int = 5,
) -> dict[str, Any]:
    """Evaluate ``expr`` on the file; raise :class:`_EvalFailed` on bad symbols."""
    import numpy as np

    from ..utils.expression import build_eval_globals, eval_dataframe_expression

    resolved = _resolve_path(path)
    kind = _detect_type(resolved, file_type)
    text = str(expr or "").strip()
    if not text:
        raise _EvalFailed(
            "JP-EXP-000",
            "expression is empty",
            suggestion="Pass a non-empty expression, e.g. exp(LogL).",
        )

    df = _load_dataframe(resolved, kind=kind, group=group)
    columns = [str(c) for c in df.columns]
    symbols = _expr_symbols(text)
    unresolved = sorted(
        s
        for s in symbols
        if s not in columns and s not in EXPR_IDENTIFIER_IGNORE
    )
    # allow namespace attributes like np.log10 — tokens include log10 which is ignored
    if unresolved:
        hints: list[str] = []
        for name in unresolved:
            hints.extend(did_you_mean(name, columns))
        # unique preserve order
        seen: set[str] = set()
        unique_hints = []
        for h in hints:
            if h not in seen:
                seen.add(h)
                unique_hints.append(h)
        funcs = sorted(
            n
            for n, v in build_eval_globals().items()
            if not n.startswith("__") and callable(v)
        )[:40]
        raise _EvalFailed(
            "JP-EXP-002",
            (
                f"Symbol {unresolved[0]!r} is not a column of this dataset "
                "and not a known function."
                if len(unresolved) == 1
                else f"Symbols {unresolved} are not columns and not known functions."
            ),
            data={
                "expr": text,
                "path": resolved,
                "type": kind,
                "symbols_used": sorted(symbols),
                "symbols_unresolved": unresolved,
                "available_columns": columns,
                "did_you_mean": unique_hints[:5],
                "available_functions": funcs,
            },
            suggestion=(
                f"Use a column from available_columns"
                + (f", e.g. {unique_hints[0]!r}" if unique_hints else "")
                + ", or a function from jplot cap funcs."
            ),
        )

    try:
        arr = eval_dataframe_expression(df, text)
    except Exception as exc:
        funcs = sorted(
            n
            for n, v in build_eval_globals().items()
            if not n.startswith("__") and callable(v)
        )[:40]
        raise _EvalFailed(
            "JP-EXP-001",
            f"expression failed: {exc}",
            data={
                "expr": text,
                "path": resolved,
                "type": kind,
                "symbols_used": sorted(symbols),
                "symbols_unresolved": [],
                "available_columns": columns,
                "available_functions": funcs,
            },
            suggestion="Check operators and parentheses; run jplot cap funcs for callables.",
        ) from exc

    values = np.asarray(arr).reshape(-1)
    n = int(values.size)
    finite_mask = np.isfinite(values.astype(float, copy=False)) if n else np.array([], dtype=bool)
    try:
        finite_vals = values[finite_mask].astype(float, copy=False)
    except (TypeError, ValueError):
        finite_vals = np.array([], dtype=float)
        finite_mask = np.zeros(n, dtype=bool)

    n_finite = int(finite_mask.sum()) if n else 0
    sample_n = max(0, min(int(sample), 20, n_finite or n))
    sample_vals: list[Any] = []
    if sample_n and n:
        take = values[:sample_n]
        sample_vals = [_jsonable(v) for v in take.tolist()]

    out: dict[str, Any] = {
        "expr": text,
        "path": resolved,
        "type": kind,
        "dtype": str(getattr(values, "dtype", "object")),
        "n": n,
        "n_finite": n_finite,
        "n_nan": int(n - n_finite),
        "sample": sample_vals,
        "symbols_used": sorted(s for s in symbols if s in columns),
        "symbols_unresolved": [],
    }
    if n_finite:
        out["min"] = float(np.min(finite_vals))
        out["max"] = float(np.max(finite_vals))
        out["n_nonpositive"] = int(np.sum(finite_vals <= 0))
    return out


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _resolve_path(path: str) -> str:
    resolved = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.exists(resolved):
        raise FileNotFoundError(resolved)
    if not os.path.isfile(resolved):
        raise IsADirectoryError(resolved) if os.path.isdir(resolved) else OSError(
            f"not a regular file: {resolved}"
        )
    return resolved


def _detect_type(path: str, forced: str) -> str:
    if forced and forced != "auto":
        return forced
    suffix = Path(path).suffix.lower()
    if suffix in {".h5", ".hdf5", ".hdf"}:
        return "hdf5"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    if suffix in {".csv", ".tsv", ".txt"}:
        return "csv"
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
        if head.startswith(b"\x89HDF"):
            return "hdf5"
        if head.startswith(b"PAR1"):
            return "parquet"
    except OSError:
        pass
    return "csv"


def _load_dataframe(
    path: str,
    *,
    kind: str,
    group: str | None = None,
    nrows: int | None = None,
):
    import pandas as pd

    if kind == "csv":
        return pd.read_csv(path, nrows=nrows)
    if kind == "parquet":
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            df = table.to_pandas()
        except Exception:
            df = pd.read_parquet(path)
        return df.head(nrows) if nrows is not None else df
    if kind == "hdf5":
        df = _hdf5_to_dataframe(path, group=group)
        return df.head(nrows) if nrows is not None else df
    raise ValueError(f"unsupported type {kind!r}")


def _hdf5_to_dataframe(path: str, *, group: str | None):
    import h5py
    import pandas as pd

    with h5py.File(path, "r") as handle:
        target = handle[group] if group else handle
        columns_data: dict[str, Any] = {}

        def _walk(node, prefix: str = "") -> None:
            for key in node.keys():
                child = node[key]
                name = f"{prefix}/{key}" if prefix else key
                if isinstance(child, h5py.Group):
                    _walk(child, name)
                elif isinstance(child, h5py.Dataset):
                    try:
                        arr = child[()]
                    except Exception:
                        continue
                    if getattr(arr, "ndim", 0) == 1:
                        columns_data[name] = arr
                    elif getattr(arr, "ndim", 0) == 0:
                        columns_data[name] = [arr]

        if isinstance(target, h5py.Dataset):
            try:
                arr = target[()]
                if getattr(arr, "ndim", 0) == 1:
                    columns_data[group or target.name] = arr
            except Exception:
                pass
        else:
            _walk(target)

    if not columns_data:
        return pd.DataFrame()
    df = pd.DataFrame(columns_data)
    short = {c: c.rsplit("/", 1)[-1] for c in df.columns}
    if len(set(short.values())) == len(short):
        df = df.rename(columns=short)
    return df


def _describe_tabular(path: str, *, kind: str, stats: bool) -> dict[str, Any]:
    df = _load_dataframe(path, kind=kind)
    return {
        "path": path,
        "type": kind,
        "rows": int(len(df)),
        "columns": _column_records(df, stats=stats),
    }


def _describe_hdf5(path: str, *, group: str | None, stats: bool) -> dict[str, Any]:
    import h5py

    tree_lines: list[str] = []
    try:
        tree_text = _hdf5_tree_text(path)
        if tree_text:
            tree_lines = tree_text.splitlines()
    except Exception:
        tree_lines = []

    groups: list[str] = []
    with h5py.File(path, "r") as handle:
        target = handle[group] if group else handle

        def _list_groups(node, prefix: str = "") -> None:
            for key in getattr(node, "keys", lambda: [])():
                child = node[key]
                name = f"{prefix}/{key}" if prefix else key
                if isinstance(child, h5py.Group):
                    groups.append(name)
                    _list_groups(child, name)

        if not isinstance(target, h5py.Dataset):
            _list_groups(target)

    df = _hdf5_to_dataframe(path, group=group)
    if df.empty:
        return {
            "path": path,
            "type": "hdf5",
            "group": group,
            "rows": 0,
            "groups": groups,
            "columns": [],
            "tree": "\n".join(tree_lines) if tree_lines else None,
            "note": "no 1-D datasets found to summarise as columns",
        }
    return {
        "path": path,
        "type": "hdf5",
        "group": group,
        "rows": int(len(df)),
        "groups": groups,
        "columns": _column_records(df, stats=stats),
        "tree": "\n".join(tree_lines) if tree_lines else None,
    }


def _hdf5_tree_text(path: str) -> str:
    import io
    from contextlib import redirect_stdout

    import h5py

    from ..data_loader_summary import print_hdf5_tree_ascii

    buf = io.StringIO()
    try:
        with h5py.File(path, "r") as handle, redirect_stdout(buf):
            print_hdf5_tree_ascii(handle, root_name=os.path.basename(path))
    except Exception:
        return ""
    return buf.getvalue().strip()


# --------------------------------------------------------------------------- #
# Describe cache (C3) — reuses ProjectCache summary store
# --------------------------------------------------------------------------- #


def _cache_key_payload(path: str, *, kind: str, group: str | None, stats: bool) -> dict[str, Any]:
    from ..cache_store import ProjectCache

    workdir = str(Path(path).resolve().parent)
    cache = ProjectCache(workdir)
    fp = cache.source_fingerprint(
        path,
        extra={
            "verb": "data.describe",
            "version": _DESCRIBE_CACHE_VERSION,
            "kind": kind,
            "group": group,
            "stats": bool(stats),
        },
    )
    return {"fp": fp, "workdir": workdir}


def _cache_get_describe(
    path: str, *, kind: str, group: str | None, stats: bool
) -> dict[str, Any] | None:
    try:
        from ..cache_store import ProjectCache

        meta = _cache_key_payload(path, kind=kind, group=group, stats=stats)
        cache = ProjectCache(meta["workdir"])
        text = cache.get_summary(meta["fp"])
        if not text:
            return None
        data = json.loads(text)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def _cache_put_describe(
    path: str,
    *,
    kind: str,
    group: str | None,
    stats: bool,
    payload: dict[str, Any],
) -> None:
    try:
        from ..cache_store import ProjectCache

        meta = _cache_key_payload(path, kind=kind, group=group, stats=stats)
        cache = ProjectCache(meta["workdir"])
        to_store = {k: v for k, v in payload.items() if k != "cache"}
        cache.put_summary(meta["fp"], json.dumps(to_store, ensure_ascii=False, default=str))
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Column stats / role hints
# --------------------------------------------------------------------------- #


def _column_records(df, *, stats: bool) -> list[dict[str, Any]]:
    import numpy as np
    from pandas.api import types as pdt

    records: list[dict[str, Any]] = []
    n = len(df)
    for col in df.columns:
        series = df[col]
        rec: dict[str, Any] = {
            "name": str(col),
            "dtype": str(series.dtype),
            "nonnull": float(series.notna().sum()) / float(n) if n else 0.0,
        }
        if not stats:
            records.append(rec)
            continue
        if pdt.is_numeric_dtype(series.dtype) and not pdt.is_bool_dtype(series.dtype):
            nonnull = series.dropna()
            if len(nonnull):
                values = nonnull.to_numpy(dtype=float, copy=False)
                rec["min"] = float(np.min(values))
                rec["max"] = float(np.max(values))
                try:
                    qs = np.quantile(values, [0.01, 0.25, 0.5, 0.75, 0.99])
                    rec["q"] = {
                        "01": float(qs[0]),
                        "25": float(qs[1]),
                        "50": float(qs[2]),
                        "75": float(qs[3]),
                        "99": float(qs[4]),
                    }
                except Exception:
                    pass
                positive = bool(np.all(values > 0))
                rec["positive"] = positive
                if positive and rec["min"] > 0 and rec["max"] > 0:
                    rec["decades"] = float(np.log10(rec["max"] / rec["min"]))
            rec["role_hint"] = _role_hint(str(col), rec)
        else:
            try:
                rec["n_unique"] = int(series.nunique(dropna=True))
            except Exception:
                rec["n_unique"] = None
            rec["role_hint"] = _role_hint(str(col), rec)
        records.append(rec)
    return records


def _role_hint(name: str, rec: dict[str, Any]) -> str | None:
    lower = name.lower().rsplit("/", 1)[-1]
    if lower in {"logl", "loglike", "log_likelihood", "lnl", "loglikelihood"}:
        return "log_likelihood"
    if lower in {"chi2", "chisq", "chi_square", "chi_squared"}:
        return "chi2"
    if lower in {"weight", "weights", "w", "posterior_weight"}:
        return "weight"
    if lower in {"isvalid", "valid", "flag", "passed"}:
        return "flag"
    if "logl" in lower or lower.endswith("_logl"):
        return "log_likelihood"
    if rec.get("positive") and rec.get("decades", 0) and rec["decades"] >= 2:
        return "parameter"
    if rec.get("min") is not None:
        return "parameter"
    return None


def _expr_symbols(expr: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))


def _parse_cols(cols: str | None) -> list[str] | None:
    if cols is None or str(cols).strip() == "":
        return None
    return [c.strip() for c in str(cols).split(",") if c.strip()]


def _jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, (np.floating, float)):
        f = float(value)
        if f != f:  # NaN
            return None
        return f
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value if value is None or isinstance(value, (str, list, dict)) else str(value)


# --------------------------------------------------------------------------- #
# Human printers
# --------------------------------------------------------------------------- #


def _print_describe(data: dict[str, Any]) -> None:
    print(
        f"{data.get('type')}  {data.get('path')}  rows={data.get('rows')}  "
        f"cache={data.get('cache', '-')}",
        file=sys.stderr,
    )
    for col in data.get("columns") or []:
        bits = [col["name"], col.get("dtype", "")]
        if "min" in col:
            bits.append(f"[{col['min']}, {col['max']}]")
        if col.get("role_hint"):
            bits.append(f"role={col['role_hint']}")
        print("  " + "  ".join(str(b) for b in bits if b), file=sys.stderr)
    if data.get("tree"):
        print(data["tree"], file=sys.stderr)


def _print_head(data: dict[str, Any]) -> None:
    print(
        f"{data.get('type')}  {data.get('path')}  n={data.get('n')}  "
        f"columns={data.get('columns')}",
        file=sys.stderr,
    )
    for row in data.get("rows") or []:
        print(f"  {row}", file=sys.stderr)


def _print_eval(data: dict[str, Any]) -> None:
    print(
        f"expr={data.get('expr')!r}  n={data.get('n')}  "
        f"finite={data.get('n_finite')}  "
        f"range=[{data.get('min')}, {data.get('max')}]",
        file=sys.stderr,
    )
    print(f"  sample: {data.get('sample')}", file=sys.stderr)


def _print_suggest_axes(data: dict[str, Any]) -> None:
    print(f"{data.get('type')}  {data.get('path')}", file=sys.stderr)
    for axis in data.get("axes") or []:
        print(
            f"  {axis['col']}: scale={axis.get('scale')}  lim={axis.get('lim')}  "
            f"({axis.get('reason')})",
            file=sys.stderr,
        )


def _usage_err(env: dict) -> int:
    print(env.get("error", {}).get("message", "usage error"), file=sys.stderr)
    return EXIT_USAGE


def _fail(env: dict) -> int:
    print(env.get("error", {}).get("message", "failed"), file=sys.stderr)
    return EXIT_FAILED

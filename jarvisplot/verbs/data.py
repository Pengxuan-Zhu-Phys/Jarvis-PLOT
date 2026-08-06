#!/usr/bin/env python3

"""``jplot data …`` -- look at a data file the way the renderer will.

``describe`` is the physical exit for the anti-hallucination rule
"column names only come from the data". It must never invent columns.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from ..agent_io import EXIT_FAILED, EXIT_OK, EXIT_USAGE, emit, envelope, error_payload

__all__ = ["SUBCOMMANDS", "build_parser", "run"]

SUBCOMMANDS = ("describe", "head", "eval", "suggest-axes")


def build_parser(prog: str = "jplot data") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Inspect a data file the way Jarvis-PLOT will load it.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    describe = sub.add_parser(
        "describe",
        help="column names, dtypes, ranges (and HDF5 tree when applicable)",
    )
    describe.add_argument("file", help="path to csv / parquet / hdf5")
    describe.add_argument(
        "--type",
        dest="file_type",
        choices=["csv", "parquet", "hdf5", "auto"],
        default="auto",
        help="force a loader (default: guess from extension)",
    )
    describe.add_argument(
        "--group",
        default=None,
        help="HDF5 group to open (default: first usable / file root)",
    )
    describe.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON envelope on stdout",
    )
    describe.add_argument(
        "--no-rows",
        action="store_true",
        help="header-only: skip min/max/quantile stats (still lists columns)",
    )
    return parser


def run(argv: Sequence[str], *, prog: str = "jplot data") -> int:
    parser = build_parser(prog)
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        return int(exc.code or EXIT_USAGE)

    action = str(getattr(args, "action", "") or "")
    if action != "describe":
        env = envelope(
            f"data.{action or 'unknown'}",
            False,
            error=error_payload(
                "UsageError",
                f"data action {action!r} is not implemented yet "
                f"(available: describe; planned: head, eval, suggest-axes)",
            ),
        )
        return emit(env) if getattr(args, "json", False) else _usage_err(env)

    as_json = bool(args.json) or not sys.stdout.isatty()
    try:
        data = describe_file(
            args.file,
            file_type=args.file_type,
            group=args.group,
            stats=not args.no_rows,
        )
    except FileNotFoundError as exc:
        env = envelope(
            "data.describe",
            False,
            data={"path": str(args.file)},
            error=error_payload(exc),
        )
        return emit(env) if as_json else _fail(env)
    except Exception as exc:
        env = envelope(
            "data.describe",
            False,
            data={"path": str(args.file)},
            error=error_payload(exc),
        )
        return emit(env) if as_json else _fail(env)

    env = envelope("data.describe", True, data=data)
    if as_json:
        return emit(env)
    _print_describe(data)
    return EXIT_OK


def describe_file(
    path: str,
    *,
    file_type: str = "auto",
    group: str | None = None,
    stats: bool = True,
) -> dict[str, Any]:
    """Return a machine-readable summary of one data file."""
    resolved = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.exists(resolved):
        raise FileNotFoundError(resolved)
    if not os.path.isfile(resolved):
        raise IsADirectoryError(resolved) if os.path.isdir(resolved) else OSError(
            f"not a regular file: {resolved}"
        )

    kind = _detect_type(resolved, file_type)
    if kind == "hdf5":
        return _describe_hdf5(resolved, group=group, stats=stats)
    if kind == "parquet":
        return _describe_tabular(resolved, kind="parquet", stats=stats)
    if kind == "csv":
        return _describe_tabular(resolved, kind="csv", stats=stats)
    raise ValueError(
        f"cannot detect file type for {resolved!r}; pass --type csv|parquet|hdf5"
    )


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
    # sniff magic
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


def _describe_tabular(path: str, *, kind: str, stats: bool) -> dict[str, Any]:
    import pandas as pd

    if kind == "parquet":
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            df = table.to_pandas()
        except Exception:
            df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    return {
        "path": path,
        "type": kind,
        "rows": int(len(df)),
        "columns": _column_records(df, stats=stats),
    }


def _describe_hdf5(path: str, *, group: str | None, stats: bool) -> dict[str, Any]:
    import h5py
    import pandas as pd

    tree_lines: list[str] = []
    try:
        tree_text = _hdf5_tree_text(path)
        if tree_text:
            tree_lines = tree_text.splitlines()
    except Exception:
        tree_lines = []

    with h5py.File(path, "r") as handle:
        target = handle[group] if group else handle
        # Flatten leaf datasets into a frame of columns when they are 1-D.
        columns_data: dict[str, Any] = {}
        groups: list[str] = []

        def _walk(node, prefix: str = "") -> None:
            for key in node.keys():
                child = node[key]
                name = f"{prefix}/{key}" if prefix else key
                if isinstance(child, h5py.Group):
                    groups.append(name)
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

    df = pd.DataFrame(columns_data)
    # Prefer short names when unique
    short = {c: c.rsplit("/", 1)[-1] for c in df.columns}
    if len(set(short.values())) == len(short):
        df = df.rename(columns=short)

    payload = {
        "path": path,
        "type": "hdf5",
        "group": group,
        "rows": int(len(df)),
        "groups": groups,
        "columns": _column_records(df, stats=stats),
        "tree": "\n".join(tree_lines) if tree_lines else None,
    }
    return payload


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


def _column_records(df, *, stats: bool) -> list[dict[str, Any]]:
    import numpy as np
    import pandas as pd
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
    """Cheap name-based role guess (C2, co-located so describe is useful now)."""
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


def _print_describe(data: dict[str, Any]) -> None:
    print(
        f"{data.get('type')}  {data.get('path')}  rows={data.get('rows')}",
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


def _usage_err(env: dict) -> int:
    print(env.get("error", {}).get("message", "usage error"), file=sys.stderr)
    return EXIT_USAGE


def _fail(env: dict) -> int:
    print(env.get("error", {}).get("message", "failed"), file=sys.stderr)
    return EXIT_FAILED

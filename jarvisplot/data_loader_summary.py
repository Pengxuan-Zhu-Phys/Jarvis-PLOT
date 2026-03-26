from __future__ import annotations

from typing import Optional

import h5py
import numpy as np
import pandas as pd
from pandas.api import types as pdt


def _format_scalar(value) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except Exception:
        pass
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def format_table(headers, rows, aligns=None, separator: str = "  ") -> str:
    headers = [str(h) for h in headers]
    row_strings = [[_format_scalar(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in row_strings:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    aligns = list(aligns or ["left"] * len(headers))
    if len(aligns) < len(headers):
        aligns.extend(["left"] * (len(headers) - len(aligns)))

    def _fmt(cell: str, idx: int) -> str:
        width = widths[idx]
        if aligns[idx] == "right":
            return f"{cell:>{width}}"
        if aligns[idx] == "center":
            return f"{cell:^{width}}"
        return f"{cell:<{width}}"

    table_width = sum(widths) + len(separator) * (len(headers) - 1)
    lines = ["-" * table_width]
    lines.append(separator.join(_fmt(headers[idx], idx) for idx in range(len(headers))))
    lines.append("-" * table_width)
    for row in row_strings:
        lines.append(separator.join(_fmt(row[idx], idx) for idx in range(len(headers))))
    lines.append("-" * table_width)
    return "\n".join(lines)


def dataframe_summary_rows(df: pd.DataFrame):
    rows = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        try:
            nonnull_pct = (float(series.notna().sum()) / float(len(series)) * 100.0) if len(series) else 0.0
        except Exception:
            nonnull_pct = 0.0
        nonnull_text = f"{nonnull_pct:.1f}%"

        if pdt.is_numeric_dtype(series.dtype) and not pdt.is_bool_dtype(series.dtype):
            nonnull = series.dropna()
            if len(nonnull):
                min_text = _format_scalar(nonnull.min())
                max_text = _format_scalar(nonnull.max())
            else:
                min_text = "NA"
                max_text = "NA"
        else:
            try:
                uniq = int(series.nunique(dropna=True))
            except Exception:
                uniq = 0
            min_text = f"uniq={uniq}"
            max_text = "NA"

        rows.append((str(col), dtype, nonnull_text, min_text, max_text))
    return rows


def dataframe_summary(df: pd.DataFrame, name: str = "") -> str:
    """Pretty, compact multi-line summary for a DataFrame."""
    title = name.strip("\n") if name else " DataFrame loaded!"
    rows, cols = df.shape
    lines = [title, f"DataFrame shape\t-> {rows}\t rows × {cols} \tcols"]
    try:
        mem_bytes = int(df.memory_usage(deep=True).sum())
        lines.append(f"Approx memory usage\t-> {dataframe_summary_human_bytes(mem_bytes)}")
    except Exception:
        pass

    if cols <= 0:
        return "\n".join(lines)

    lines.append("=== DataFrame Summary Table ===")
    lines.append(
        format_table(
            ["name", "dtype", "nonnull%", "[min]", "[max]"],
            dataframe_summary_rows(df),
            aligns=["left", "left", "right", "right", "right"],
        )
    )
    return "\n".join(lines)


def dataframe_summary_human_bytes(num_bytes: int) -> str:
    """Format bytes in a human-readable short form."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(max(int(num_bytes), 0))
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{int(value)} B"


def print_hdf5_tree_ascii(hobj, root_name: str = "/", logger=None, max_depth: Optional[int] = None):
    """Print a compact ASCII tree for a HDF5 group/file."""
    lines = [f"{root_name}"]

    def emit(msg):
        if logger:
            logger.debug(msg)
        else:
            print(msg)

    def walk(node, prefix="", depth=0):
        if max_depth is not None and depth >= max_depth:
            return
        items = list(node.items())
        for i, (name, child) in enumerate(items):
            last = i == len(items) - 1
            branch = "└── " if last else "├── "
            next_prefix = prefix + ("    " if last else "│   ")
            if isinstance(child, h5py.Dataset):
                shape = getattr(child, "shape", ())
                dtype = getattr(child, "dtype", None)
                if getattr(dtype, "names", None):
                    dtype_desc = f"structured[{', '.join(dtype.names)}]"
                else:
                    dtype_desc = str(dtype)
                lines.append(f"{prefix}{branch}{name} [dataset] shape={shape} dtype={dtype_desc}")
            elif isinstance(child, h5py.Group):
                lines.append(f"{prefix}{branch}{name} [group]")
                walk(child, next_prefix, depth + 1)
            else:
                lines.append(f"{prefix}{branch}{name} [{type(child).__name__}]")

    if isinstance(hobj, h5py.File):
        root = hobj[root_name] if root_name in hobj else hobj
    else:
        root = hobj

    walk(root, prefix="")
    emit("\n".join(lines))

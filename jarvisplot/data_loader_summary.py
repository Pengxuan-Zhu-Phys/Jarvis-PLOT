from __future__ import annotations

from typing import Optional

import h5py
import pandas as pd


def dataframe_summary(df: pd.DataFrame, name: str = "") -> str:
    """Pretty, compact multi-line summary for a DataFrame."""
    if name:
        title = name.strip("\n")
    else:
        title = " DataFrame loaded!"

    rows, cols = df.shape
    lines = [title, f"shape -> ({rows}, {cols})"]

    if cols > 0:
        preview_cols = list(df.columns[: min(cols, 6)])
        lines.append("columns -> " + ", ".join(map(str, preview_cols)))

    if rows > 0 and cols > 0:
        sample = df.iloc[: min(rows, 3), : min(cols, 3)]
        lines.append("head ->")
        lines.extend("  " + str(line) for line in sample.to_string(index=False).splitlines())

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


#!/usr/bin/env python3 

from __future__ import annotations
from pathlib import Path
from typing import Optional, Any, Dict, List
import yaml
import os, sys 
import pandas as pd 
import h5py
import numpy as np

class DataSet():
    def __init__(self):
        self._file: Optional[str]   = None
        self.path:  Optional[str]   = None 
        self._type:  Optional[str]  = None
        self.base:  Optional[str]   = None
        self.keys:  Optional[List[str]]  = None 
        self._logger                = None 
        self.data                   = None
        self.group                  = None
        self.is_gambit              = False
       
    def setinfo(self, dtinfo, rootpath):
        self.file = os.path.join(rootpath, dtinfo['path'])
        self.name = dtinfo['name']
        self.type = dtinfo['type'].lower()
        if self.type == "csv":
            self.load_csv()
        if self.type == "hdf5" and dtinfo.get('dataset'): 
            self.group = dtinfo['dataset']
            self.is_gambit = dtinfo.get('is_gambit', False)
            self.columnmap = dtinfo.get('columnmap', {})
            self.load_hdf5()
            
        
       
     
    @property 
    def file(self) -> Optional[str]:
        return self._file 
    
    @property
    def type(self) -> Optional[str]:
        return self._type 
    
    @property
    def logger(self): 
        return self._logger
    
    @logger.setter
    def logger(self, logger) -> None: 
        if logger is None: 
            self._logger = None
        self._logger = logger
    
    @file.setter 
    def file(self, value: Optional[str]) -> None: 
        if value is None: 
            self._file  = None
            self.path   = None
            self.base   = None
            
        p = Path(value).expanduser().resolve()
        self._file  = str(p)
        self.path   = os.path.abspath(p)
        self.base   = os.path.basename(p)
        
    @type.setter 
    def type(self, value: Optional[str]) -> None: 
        if value is None: 
            self._type = None
            
        self._type = str(value).lower()
        self.logger.debug("Dataset -> {} is assigned as \n\t-> {}\ttype".format(self.base, self.type))
    
    def load_csv(self):
        if self.type == "csv":
            if self.logger:
                self.logger.debug("Loading CSV from {}".format(self.path))

            self.data = pd.read_csv(self.path)
            self.keys = list(self.data.columns)

            # Emit the same pretty summary used for HDF5 datasets
            summary_name = f" CSV loaded!\n\t name  -> {self.name}\n\t path  -> {self.path}"
            try:
                summary_msg = dataframe_summary(self.data, name=summary_name)
            except Exception:
                # Fallback minimal summary if something goes wrong
                summary_msg = f"CSV loaded  {summary_name}\nDataFrame shape: {self.data.shape}"

            if self.logger:
                self.logger.warning("\n" + summary_msg)
            else:
                print(summary_msg)
    
    def load_hdf5(self):
            def _iter_datasets(hobj, prefix=""):
                for k, v in hobj.items():
                    path = f"{prefix}/{k}" if prefix else k
                    if isinstance(v, h5py.Dataset):
                        yield path, v
                    elif isinstance(v, h5py.Group):
                        yield from _iter_datasets(v, path)

            def _pick_dataset(hfile: h5py.File):
                # Heuristic: prefer structured arrays, then 2D arrays
                best = None
                for path, ds in _iter_datasets(hfile):
                    shape = getattr(ds, "shape", ())
                    dt = getattr(ds, "dtype", None)
                    score = 0
                    if dt is not None and getattr(dt, "names", None):
                        score += 10  # structured array → good for DataFrame
                    if len(shape) == 2:
                        score += 5
                        if shape[1] >= 2:
                            score += 1
                    if best is None or score > best[0]:
                        best = (score, path, ds)
                if best is None:
                    raise RuntimeError("No datasets found in HDF5 file.")
                _, path, ds = best
                return path, ds[()]

            def _to_dataframe(arr, name=""):
                if isinstance(arr, np.ndarray) and getattr(arr.dtype, "names", None):
                    df = pd.DataFrame.from_records(arr)
                    # prefix columns to keep dataset origin
                    if name:
                        df.columns = [f"{name}:{c}" for c in df.columns]
                    return df
                elif hasattr(arr, "ndim") and arr.ndim == 2:
                    cols = [f"col{i}" for i in range(arr.shape[1])]
                    if name:
                        cols = [f"{name}:{c}" for c in cols]
                    return pd.DataFrame(arr, columns=cols)
                else:
                    col = name if name else "value"
                    return pd.DataFrame({col: np.ravel(arr)})

            def _collect_group_datasets(g: h5py.Group, prefix: str=""):
                """Recursively collect (path, ndarray) for all datasets under a group."""
                items = []
                for k, v in g.items():
                    path = f"{prefix}/{k}" if prefix else k
                    if isinstance(v, h5py.Dataset):
                        items.append((path, v[()]))
                    elif isinstance(v, h5py.Group):
                        items.extend(_collect_group_datasets(v, path))
                return items

            with h5py.File(self.path, "r") as f1:
                # Log top-level keys to help the user
                print_hdf5_tree_ascii(f1[self.group], root_name=self.group, logger=self.logger)

                if self.group in f1 and isinstance(f1[self.group], h5py.Group):
                    group = f1[self.group]
                    self.logger.debug("Loading HDF5 group '{}' from {}".format(self.group, self.path))
                    if self.is_gambit: 
                        self.logger.debug("GAMBIT Standard Output")

                    # Collect all datasets under the group (recursively)
                    items = _collect_group_datasets(group, prefix=self.group)
                    if not items:
                        raise RuntimeError(f"HDF5 group '{self.group}' contains no datasets.")

                    # If there is only one dataset, behave like before
                    kkeys = []
                    if len(items) == 1:
                        path, arr = items[0]
                        dfs = [(path, _to_dataframe(arr, name=path))]
                        kkeys.append(path)
                    else:
                        # Build a dataframe per dataset
                        dfs = [(p, _to_dataframe(arr, name=p)) for p, arr in items]
                        kkeys = [p for p, arr in items]
                    
                    # Try to concatenate along columns; all datasets must have identical row counts
                    lengths = {len(df) for _, df in dfs}
                    if len(lengths) == 1:
                        # safe to concat by columns → single merged DataFrame only
                        self.data = pd.concat([df for _, df in dfs], axis=1)
                        
                        self.keys = list(self.data.columns)
                        
                        # Deal GAMBIT filtering 
                        if self.is_gambit:
                            self.gambit_filtering(kkeys)
                            if self.columnmap.get("list", False):
                                self.logger.warning("{}: Loading Column Maps".format(self.name))
                                cmap = {}
                                for item in self.columnmap.get("list", False): 
                                    cmap[item['source_name']] = item['new_name']
                                self.rename_columns(cmap)
                                
                        # Emit a pretty summary BEFORE returning
                        summary_name = f" HDF5 loaded!\n\t name  -> {self.name}\n\t group -> {self.group}\n\t path  -> {self.path}"
                        summary_msg = dataframe_summary(self.data, name=summary_name)
                        if self.logger:
                            self.logger.warning("\n" + summary_msg)
                        else:
                            print(summary_msg)

                        return  # IMPORTANT: stop here; avoid falling through to single-dataset path
                    else:
                        # Not mergeable → print tree for diagnostics and raise a hard error
                        try:
                            print_hdf5_tree_ascii(group, root_name=self.group, logger=self.logger)
                        except Exception:
                            pass
                        shapes = {p: df.shape for p, df in dfs}
                        raise ValueError(
                            "HDF5 group '{grp}' is invalid for merging: datasets have different row counts. "
                            "Please fix the input or choose a different dataset/group. Details: {details}".format(
                                grp=self.group,
                                details=shapes,
                            )
                        )
                else:
                    path, arr = _pick_dataset(f1)
    
    def gambit_filtering(self, kkeys): 
        isvalids = []
        for kk in kkeys: 
            if "_isvalid" == kk[-8:] and kk[:-8] in self.keys: 
                isvalids.append(kk)
        self.logger.warning("Filtering Invalid Data from GAMBIT Output")
        sps = self.data.shape
        mask = self.data[isvalids].all(axis=1)
        self.data = self.data[mask].drop(columns=isvalids)
        self.logger.warning("DataSet Shape: \n\t Before filtering -> {}\n\t  After filtering -> {}".format(sps, self.data.shape))
        self.keys = list(self.data.columns)
                
    def rename_columns(self, vdict):
        self.data = self.data.rename(columns=vdict)
        self.keys = list(self.data.columns)
        
                   
def dataframe_summary(df: pd.DataFrame, name: str = "") -> str:
    """Pretty, compact multi-line summary for a DataFrame.

    Sections:
      • header: dataset path (if any) and shape
      • columns table (first max_cols): name | dtype | non-null% | unique (for small card.) | min..max (numeric)
      • tiny preview of first rows/cols
    """
    import pandas as _pd
    import numpy as _np
    import shutil

    def term_width(default=120):
        try:
            return shutil.get_terminal_size().columns
        except Exception:
            return default

    def trunc(s: str, width: int) -> str:
        if len(s) <= width:
            return s
        # keep both ends
        head = max(0, width // 2 - 2)
        tail = max(0, width - head - 3)
        return s[:head] + "..." + s[-tail:]

    nrows, ncols = df.shape
    cols = list(df.columns)
    show_cols = cols[:]

    # Compute per-column stats for the shown columns
    dtypes = df[show_cols].dtypes.astype(str)
    non_null_pct = (df[show_cols].notna().sum() / max(1, nrows) * 100.0).round(1)

    # numeric min/max; categorical unique count (cap at 20)
    is_num = [_pd.api.types.is_numeric_dtype(df[c]) for c in show_cols]
    num_cols = [c for c, ok in zip(show_cols, is_num) if ok]
    cat_cols = [c for c, ok in zip(show_cols, is_num) if not ok]

    num_min = {}
    num_max = {}
    if num_cols:
        try:
            desc = df[num_cols].agg(["min", "max"]).T
            for c in num_cols:
                mn = desc.loc[c, "min"]
                mx = desc.loc[c, "max"]
                num_min[c] = mn
                num_max[c] = mx
        except Exception:
            pass

    uniques = {}
    if cat_cols:
        for c in cat_cols:
            try:
                u = df[c].nunique(dropna=True)
                uniques[c] = int(u)
            except Exception:
                pass

    # Build a compact table
    tw = term_width()
    name_w = 34 if tw < 120 else 48
    dtype_w = 10
    nn_w = 8
    stat_w = max(12, tw - (name_w + dtype_w + nn_w + 8))  # 8 for separators/padding

    def fmt_stat(c: str) -> str:
        if c in num_min and c in num_max:
            try:
                mn = num_min[c]
                mx = num_max[c]
                return f"{mn:>10.4g} .. {mx:>10.4g}"
            except Exception:
                return f"{str(num_min[c]):>10} .. {str(num_max[c]):>10}"
        if c in uniques:
            return f"uniq={uniques[c]}"
        return ""

    head_lines = []
    if name:
        head_lines.append(f"Selected dataset:{name}")
    head_lines.append(f"DataFrame shape:\n\t {nrows}\t rows × {ncols} \tcols\n")
    head_lines.append("=== DataFrame Summary Table ===")

    # Column table header
    rows = []
    header = f"{'name':<{name_w}}  {'dtype':<{dtype_w}}  {'nonnull%':>{nn_w}}  {'     [min] ..      [max]':<{stat_w}}"
    rows.append("-" * len(header))
    rows.append(header)
    rows.append("-" * len(header))

    for c in show_cols:
        c_name = trunc(str(c), name_w)
        c_dtype = trunc(dtypes[c], dtype_w)
        c_nn = f"{non_null_pct[c]:.1f}%" if nrows else "n/a"
        c_stat = trunc(fmt_stat(c), stat_w)
        rows.append(f"{c_name:<{name_w}}  {c_dtype:<{dtype_w}}  {c_nn:>{nn_w}}  {c_stat:<{stat_w}}")
    rows.append("-" * len(header))


    parts = []
    parts.extend(head_lines)
    if show_cols:
        parts.extend(rows)

    return "\n".join(parts)
                
                
def print_hdf5_tree_ascii(hobj, root_name='/', logger=None, max_depth=None):
    """
    Pretty-print an ASCII tree of an h5py.File or Group.

    Example output:
    /
    ├── data (Group)
    │   ├── samples (Dataset, shape=(1000, 3), dtype=float64)
    │   └── extra (Group)
    │       ├── X (Dataset, shape=(..., ...), dtype=...)
    │       └── Y (Dataset, shape=(..., ...), dtype=...)
    └── metadata (Group)
        └── attrs (Dataset, shape=(...,), dtype=...)

    Parameters
    ----------
    hobj : h5py.File or h5py.Group
    root_name : str
        Name shown at the root.
    logger : logging-like object (optional)
        If provided, uses logger.debug(...) instead of print.
    max_depth : int or None
        Limit recursion depth (0=only root). None = unlimited.
    """
    try:
        import h5py  # noqa: F401
    except Exception:
        raise RuntimeError("h5py is required for HDF5 tree printing.")

    def emit(msg):
        if logger is None:
            print(msg)
        else:
            try:
                logger.debug(msg)
            except Exception:
                print(msg)

    def is_dataset(x):
        import h5py
        return isinstance(x, h5py.Dataset)

    def is_group(x):
        import h5py
        return isinstance(x, h5py.Group)

    def fmt_leaf(name, obj):
        # maxlen = 60
        def shorten(n):
            if len(n) > 50:
                return f"{n[:15]}...{n[-30:]}"
            else:
                return "{:48}".format(n)
            # return n
        if is_dataset(obj):
            shp = getattr(obj, "shape", None)
            # dt  = getattr(obj, "dtype", None)
            extra = []
            if shp is not None:
                extra.append(f"shape -> {shp}")
            # if dt is not None:
            #     extra.append(f"dtype={dt}")
            suffix = f"(Dataset), {', '.join(extra)}" if extra else "(Dataset)"
            return f"{shorten(name)}{suffix:>40}"
        elif is_group(obj):
            return f"{shorten(name)}          (Group)"
        return shorten(name)

    def walk(group, prefix="", depth=0, last=True):
        lines = []
        if depth == 0:
            lines.append("│ {}          (Group)".format(root_name))
        if max_depth is not None and depth >= max_depth:
            return

        keys = list(group.keys())
        keys.sort()
        n = len(keys)
        for i, key in enumerate(keys):
            child = group[key]
            is_last = (i == n - 1)
            connector = "└── " if is_last else "├── "
            line = prefix + connector + fmt_leaf(key, child)
            lines.append(line)

            if is_group(child):
                extension = "    " if is_last else "│   "
                walk(child, prefix + extension, depth + 1, is_last)
        emit("\n".join(lines))

    walk(hobj, "", 0, True)
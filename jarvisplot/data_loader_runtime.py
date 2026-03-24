from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np
import pandas as pd

try:
    import polars as pl
except Exception:
    pl = None

from .memtrace import memtrace_checkpoint, memtrace_enabled
from .data_loader_hdf5 import (
    available_memory_bytes,
    build_hdf5_whitelist,
    canonical_dataset_path,
    columns_dict,
    materialized_cache_key,
    materialized_source_fingerprint,
    materialized_summary,
    path_aliases,
    polars_schema_names,
    rename_source_by_alias,
    shape_token,
    sql_bool_ops,
)
from .Figure.load_data import addcolumn, filter as filter_df, grid_profiling, profiling, sortby

JP_ROW_IDX = "__jp_row_idx__"

def apply_dataset_transform(dataset, stage: str = "dataset") -> None:
    """
    Apply DataSet-level transforms using the same operators as layer transforms.
    For HDF5 this is called after is_valid policy + columns rename.
    """
    if dataset.data is None:
        return
    if dataset.transform is None:
        return
    if not isinstance(dataset.transform, list):
        if dataset.logger:
            dataset.logger.warning(
                "Dataset '{}' transform ignored: list required, got {}.".format(
                    dataset.name, type(dataset.transform)
                )
            )
        return

    from .Figure.load_data import addcolumn, filter as filter_df, grid_profiling, profiling, sortby

    push_mode = str(os.getenv("JP_DATASET_POLARS_TRANSFORM", "auto")).strip().lower()
    use_polars_pushdown = push_mode not in {"0", "false", "no", "off", "disable", "disabled"}
    pushdown_keep_lazy = push_mode in {"lazy", "keep-lazy"}
    if use_polars_pushdown and pl is not None and isinstance(dataset.data, (pl.LazyFrame, pl.DataFrame)):
        if dataset._apply_dataset_transform_polars(
            stage=stage,
            materialize_to_pandas=(not pushdown_keep_lazy),
        ):
            return
        if dataset.logger:
            dataset.logger.warning(
                "Dataset '{}' transform pushdown fallback to pandas.".format(dataset.name)
            )

    if pl is not None and isinstance(dataset.data, pl.LazyFrame):
        dataset.logger.warning(f"Materializing polars LazyFrame for dataset transform -> {dataset.name}")
        dataset.data = polars_to_pandas_compat(dataset.data, logger=dataset.logger, stage=f"dataset:{dataset.name}")
        dataset._data_backend = "pandas"
    elif pl is not None and isinstance(dataset.data, pl.DataFrame):
        dataset.logger.warning(f"Materializing polars DataFrame for dataset transform -> {dataset.name}")
        dataset.data = polars_to_pandas_compat(dataset.data, logger=dataset.logger, stage=f"dataset:{dataset.name}")
        dataset._data_backend = "pandas"

    df = dataset.data
    memtrace_checkpoint(
        dataset.logger,
        "dataset.transform.before",
        df,
        extra={"dataset": dataset.name, "stage": stage},
    )
    before_shape = dataset._shape_token(df)
    for trans in dataset.transform:
        if not isinstance(trans, dict):
            if dataset.logger:
                dataset.logger.warning(f"Dataset '{dataset.name}' invalid transform step skipped -> {trans}")
            continue
        prev_df = df

        if "filter" in trans:
            df = filter_df(df, trans["filter"], dataset.logger)
        elif "profile" in trans:
            df = profiling(df, trans["profile"], dataset.logger)
        elif "grid_profile" in trans:
            cfg = trans.get("grid_profile", {})
            if isinstance(cfg, dict):
                cfg = cfg.copy()
                cfg.setdefault("method", "grid")
            else:
                cfg = {"method": "grid"}
            df = grid_profiling(df, cfg, dataset.logger)
        elif "sortby" in trans:
            df = sortby(df, trans["sortby"], dataset.logger)
        elif "add_column" in trans:
            df = addcolumn(df, trans["add_column"], dataset.logger)

        if prev_df is not df and isinstance(prev_df, pd.DataFrame):
            collect_prev = int(prev_df.shape[0]) >= 100_000
            prev_df = None
            if collect_prev:
                gc.collect()

    dataset.data = df
    if isinstance(dataset.data, pd.DataFrame):
        if JP_ROW_IDX not in dataset.data.columns:
            dataset.data.insert(0, JP_ROW_IDX, np.arange(int(dataset.data.shape[0]), dtype=np.int64))
        if isinstance(dataset.retained_columns, set) and dataset.retained_columns:
            retained = [c for c in dataset.data.columns if c in dataset.retained_columns or c == JP_ROW_IDX]
            if retained and len(retained) < int(dataset.data.shape[1]):
                dataset.data = dataset.data.loc[:, retained].copy(deep=False)
    memtrace_checkpoint(
        dataset.logger,
        "dataset.transform.after",
        dataset.data,
        extra={"dataset": dataset.name, "stage": stage},
    )
    if isinstance(dataset.data, pd.DataFrame):
        dataset.keys = list(dataset.data.columns)
    if dataset.logger:
        dataset.logger.warning(
            "DataSet transform done:\n\t name \t-> {}\n\t stage \t-> {}\n\t rows \t-> {} -> {}".format(
                dataset.name,
                stage,
                before_shape,
                dataset._shape_token(dataset.data),
            )
        )

@staticmethod
def _sql_bool_ops(expr: Any) -> str:
    return str(expr).replace("&&", " AND ").replace("||", " OR ")

@staticmethod
def _polars_schema_names(lf) -> List[str]:
    try:
        schema = lf.collect_schema()
        if hasattr(schema, "names"):
            names = schema.names()
            return [str(x) for x in names]
        if isinstance(schema, dict):
            return [str(x) for x in schema.keys()]
    except Exception:
        pass
    return []

def _apply_dataset_transform_polars(
    self,
    stage: str = "dataset",
    materialize_to_pandas: bool = True,
) -> bool:
    if pl is None:
        return False
    if not isinstance(dataset.data, (pl.LazyFrame, pl.DataFrame)):
        return False
    if not isinstance(dataset.transform, list):
        return False

    lf = dataset.data if isinstance(dataset.data, pl.LazyFrame) else dataset.data.lazy()
    before_shape = dataset._shape_token(dataset.data)
    memtrace_checkpoint(
        dataset.logger,
        "dataset.transform.before",
        dataset.data,
        extra={"dataset": dataset.name, "stage": stage, "engine": "polars"},
    )
    try:
        for trans in dataset.transform:
            if not isinstance(trans, dict):
                continue

            if "filter" in trans:
                condition = trans.get("filter")
                if isinstance(condition, bool):
                    if not condition:
                        lf = lf.filter(pl.lit(False))
                elif isinstance(condition, (int, float)) and condition in (0, 1):
                    if int(condition) == 0:
                        lf = lf.filter(pl.lit(False))
                elif isinstance(condition, str):
                    s = str(condition).strip()
                    low = s.lower()
                    if low in {"true", "t", "yes", "y"}:
                        pass
                    elif low in {"false", "f", "no", "n"}:
                        lf = lf.filter(pl.lit(False))
                    else:
                        lf = lf.filter(pl.sql_expr(dataset._sql_bool_ops(s)))
                else:
                    raise TypeError(
                        "unsupported filter condition type for polars pushdown: {}".format(type(condition))
                    )
            elif "sortby" in trans:
                expr = str(trans.get("sortby", "")).strip()
                if not expr:
                    continue
                cols = dataset._polars_schema_names(lf)
                if expr in cols:
                    lf = lf.sort(expr)
                else:
                    skey = "__jp_sortkey__"
                    lf = lf.with_columns(pl.sql_expr(dataset._sql_bool_ops(expr)).alias(skey)).sort(skey).drop(skey)
            elif "add_column" in trans:
                adds = trans.get("add_column", {})
                if not isinstance(adds, dict):
                    raise TypeError("add_column step must be dict")
                name = str(adds.get("name", "")).strip()
                expr = str(adds.get("expr", "")).strip()
                if not name or not expr:
                    raise ValueError("add_column requires non-empty name and expr")
                lf = lf.with_columns(pl.sql_expr(dataset._sql_bool_ops(expr)).alias(name))
            elif "profile" in trans or "grid_profile" in trans:
                return False
            else:
                return False
    except Exception as e:
        if dataset.logger:
            dataset.logger.warning(
                "Dataset '{}' polars transform pushdown failed: {}.".format(dataset.name, e)
            )
        return False

    lf_with_idx = lf
    cols_after_transform = dataset._polars_schema_names(lf_with_idx)
    if JP_ROW_IDX not in cols_after_transform:
        lf_with_idx = lf_with_idx.with_row_index(JP_ROW_IDX)
        cols_after_transform = dataset._polars_schema_names(lf_with_idx)
    dataset._full_lazy_frame = lf_with_idx

    keep_cols = list(cols_after_transform)
    if isinstance(dataset.retained_columns, set) and dataset.retained_columns:
        retained = set(dataset.retained_columns)
        retained.add(JP_ROW_IDX)
        keep_cols = [c for c in cols_after_transform if c in retained]

    if materialize_to_pandas:
        manifest_rows = None
        manifest_bytes = None
        if isinstance(dataset._materialized_manifest, dict):
            manifest_rows = dataset._materialized_manifest.get("rows")
            manifest_bytes = dataset._materialized_manifest.get("bytes_total")
        est_bytes = None
        try:
            if manifest_bytes is not None and cols_after_transform:
                est_bytes = int(float(manifest_bytes) * (float(len(keep_cols)) / float(len(cols_after_transform))))
        except Exception:
            est_bytes = None
        if memtrace_enabled():
            memtrace_checkpoint(
                dataset.logger,
                f"dataset:{dataset.name}.pushdown.collect_plan",
                None,
                extra={
                    "dataset": dataset.name,
                    "rows": manifest_rows,
                    "columns": len(cols_after_transform),
                    "required_columns": len(keep_cols),
                    "estimated_size": est_bytes,
                    "column_list": "|".join(cols_after_transform),
                    "required_list": "|".join(keep_cols),
                },
            )
        if keep_cols and len(keep_cols) < len(cols_after_transform) and dataset.logger:
            dataset.logger.warning(
                "Dataset '{}' pre-collect column projection -> {} -> {}".format(
                    dataset.name,
                    len(cols_after_transform),
                    len(keep_cols),
                )
            )
        narrowed = lf_with_idx.select(keep_cols) if keep_cols else lf_with_idx
        dataset.data = polars_to_pandas_compat(narrowed, logger=dataset.logger, stage=f"dataset:{dataset.name}.pushdown")
        dataset._data_backend = "pandas"
        if isinstance(dataset.data, pd.DataFrame):
            dataset.keys = list(dataset.data.columns)
        else:
            dataset.keys = dataset._polars_schema_names(narrowed)
        engine_name = "polars->pandas"
    else:
        dataset.data = lf_with_idx.select(keep_cols) if keep_cols else lf_with_idx
        dataset._data_backend = "polars_lazy"
        dataset.keys = dataset._polars_schema_names(dataset.data)
        engine_name = "polars"
    memtrace_checkpoint(
        dataset.logger,
        "dataset.transform.after",
        dataset.data,
        extra={"dataset": dataset.name, "stage": stage, "engine": engine_name},
    )
    if dataset.logger:
        dataset.logger.warning(
            "DataSet transform done:\n\t name \t-> {}\n\t stage \t-> {}\n\t engine \t-> {}\n\t rows \t-> {} -> {}".format(
                dataset.name,
                stage,
                engine_name,
                before_shape,
                dataset._shape_token(dataset.data),
            )
        )
    return True


def activate_materialized_manifest(dataset, cache_key: str, manifest: Dict[str, Any]) -> None:
    if pl is None or dataset.cache is None:
        raise RuntimeError("polars materialized backend is unavailable.")
    slot = dataset.cache.materialized_slot(cache_key)
    part_files = manifest.get("part_files", [])
    paths = []
    for part in part_files:
        p = Path(str(part))
        if not p.is_absolute():
            p = slot / p
        if p.exists():
            paths.append(str(p))
    if not paths:
        raise RuntimeError(f"Materialized parquet parts missing for cache key '{cache_key}'.")
    dataset.data = pl.scan_parquet(paths)
    dataset.keys = list(manifest.get("columns", []))
    dataset._materialized_manifest = dict(manifest)
    dataset._data_backend = "polars_lazy"
    memtrace_checkpoint(
        dataset.logger,
        "hdf5.materialized.ready",
        dataset.data,
        extra={
            "dataset": dataset.name,
            "rows": manifest.get("rows"),
            "cols": manifest.get("cols"),
            "parts": manifest.get("parts"),
            "bytes_total": manifest.get("bytes_total"),
            "cache_key": cache_key[:16] if isinstance(cache_key, str) else cache_key,
        },
    )

def load_hdf5_materialized(dataset) -> None:
    if pl is None or dataset.cache is None:
        raise RuntimeError("polars materialized backend is unavailable.")

    cache_key = dataset._materialized_cache_key()
    if cache_key is None:
        raise RuntimeError("materialized cache key unavailable.")

    manifest = dataset.cache.get_materialized_manifest(cache_key)
    if isinstance(manifest, dict):
        dataset._activate_materialized_manifest(cache_key, manifest)
        memtrace_checkpoint(
            dataset.logger,
            "hdf5.materialized.cache_hit",
            dataset.data,
            extra={"dataset": dataset.name, "cache_key": cache_key[:16]},
        )
        return

    def _iter_dataset_paths(hobj, prefix="", whitelist=None):
        for k, v in hobj.items():
            path = f"{prefix}/{k}" if prefix else k
            if isinstance(v, h5py.Dataset):
                if whitelist is None or path in whitelist:
                    yield path
            elif isinstance(v, h5py.Group):
                yield from _iter_dataset_paths(v, path, whitelist=whitelist)

    def _rename_map_from_entries() -> Dict[str, str]:
        rename_entries = dataset._columns_dict().get("rename", [])
        rename_map = {}
        if isinstance(rename_entries, list):
            for item in rename_entries:
                if not isinstance(item, dict):
                    continue
                source = str(item.get("source", "")).strip()
                target = str(item.get("target", "")).strip()
                if not source or not target:
                    continue
                source_canon = dataset._canonical_dataset_path(source)
                rename_map[source_canon] = target
                rename_map[f"{source_canon}_isvalid"] = f"{target}_isvalid"
        return rename_map

    def _dataset_nrows(ds: h5py.Dataset) -> int:
        shape = getattr(ds, "shape", ())
        if len(shape) == 0:
            return int(np.ravel(ds[()]).shape[0])
        return int(shape[0])

    def _estimate_bytes_per_row(ds: h5py.Dataset) -> int:
        dt = getattr(ds, "dtype", None)
        shape = getattr(ds, "shape", ())
        if dt is not None and getattr(dt, "names", None):
            return int(sum(int(dt.fields[name][0].itemsize) for name in dt.names))
        if len(shape) >= 2:
            return int(dt.itemsize) * int(shape[1])
        return int(getattr(dt, "itemsize", 8) or 8)

    def _slice_to_columns(path: str, ds: h5py.Dataset, start: int, end: int, row_mask=None, rename_map=None):
        arr = ds[start:end] if getattr(ds, "ndim", 0) > 0 else ds[()]

        def _apply_mask(vec):
            if row_mask is None:
                return vec
            if len(vec) != len(row_mask):
                raise ValueError(
                    "HDF5 row mismatch while applying chunk mask: "
                    f"dataset='{path}', rows={len(vec)}, mask_rows={len(row_mask)}"
                )
            return vec[row_mask]

        def _map_name(name: str) -> str:
            if isinstance(rename_map, dict):
                return rename_map.get(name, name)
            return name

        if isinstance(arr, np.ndarray) and getattr(arr.dtype, "names", None):
            out = {}
            for fname in arr.dtype.names:
                src = f"{path}:{fname}"
                dst = _map_name(src)
                out[dst] = _apply_mask(np.asarray(arr[fname]).reshape(-1))
            return out

        if hasattr(arr, "ndim") and getattr(arr, "ndim", 0) == 2:
            out = {}
            for i in range(int(arr.shape[1])):
                src = f"{path}:col{i}"
                dst = _map_name(src)
                out[dst] = _apply_mask(np.asarray(arr[:, i]).reshape(-1))
            return out

        src = path if path else "value"
        dst = _map_name(src)
        return {dst: _apply_mask(np.ravel(arr))}

    slot = dataset.cache.clear_materialized_slot(cache_key)
    rename_map = {}
    manifest = None

    with h5py.File(dataset.path, "r") as f1:
        if not (dataset.group in f1 and isinstance(f1[dataset.group], h5py.Group)):
            raise RuntimeError(
                f"HDF5 group '{dataset.group}' is required for polars materialization."
            )

        group = f1[dataset.group]
        whitelist = None
        if not bool(getattr(dataset, "full_load", False)):
            whitelist = dataset._build_hdf5_whitelist()

        dataset_paths = list(_iter_dataset_paths(group, prefix=dataset.group, whitelist=whitelist))
        if not dataset_paths:
            raise RuntimeError(f"HDF5 group '{dataset.group}' contains no datasets.")

        dataset_set = set(dataset_paths)
        base_paths = [p for p in dataset_paths if not p.endswith("_isvalid")]
        isvalid_paths = [p for p in dataset_paths if p.endswith("_isvalid")]
        rename_map = _rename_map_from_entries()

        required_base = dataset._whitelist_base_paths if isinstance(dataset._whitelist_base_paths, set) else None
        filter_isvalid_paths = []
        if dataset.isvalid_policy == "clean":
            if required_base is not None:
                missing_base = sorted([c for c in required_base if c not in dataset_set])
                missing_isvalid = sorted(
                    [f"{c}_isvalid" for c in required_base if f"{c}_isvalid" not in dataset_set]
                )
                if missing_base or missing_isvalid:
                    dataset.logger.warning(
                        "Skip is_valid policy for materialization: not all whitelist columns have companion _isvalid columns. "
                        "missing_base={}, missing_isvalid={}".format(missing_base, missing_isvalid)
                    )
                else:
                    filter_isvalid_paths = [f"{c}_isvalid" for c in sorted(required_base)]
            else:
                filter_isvalid_paths = [p for p in isvalid_paths if p[:-8] in dataset_set]

        read_paths = list(base_paths)
        if dataset.isvalid_policy == "raw":
            read_paths.extend(isvalid_paths)
        elif required_base is not None and not filter_isvalid_paths:
            read_paths.extend(isvalid_paths)

        if not read_paths:
            raise RuntimeError(f"HDF5 group '{dataset.group}' has no readable datasets.")

        expected_rows = None
        bytes_per_row = 0
        for path in read_paths:
            ds = f1[path]
            nrows = _dataset_nrows(ds)
            if expected_rows is None:
                expected_rows = nrows
            elif nrows != expected_rows:
                raise ValueError(
                    f"HDF5 group '{dataset.group}' is invalid for materialization: row mismatch at '{path}'."
                )
            bytes_per_row += _estimate_bytes_per_row(ds)
        for path in filter_isvalid_paths:
            nrows = _dataset_nrows(f1[path])
            if expected_rows is None:
                expected_rows = nrows
            elif nrows != expected_rows:
                raise ValueError(
                    f"HDF5 group '{dataset.group}' is invalid for materialization: row mismatch at '{path}'."
                )
            bytes_per_row += 1

        total_rows = int(expected_rows or 0)
        avail = dataset._available_memory_bytes()
        target_bytes = max(64 * 1024 * 1024, int((avail or (512 * 1024 * 1024)) * 0.15))
        bytes_per_row = max(int(bytes_per_row), 1)
        chunk_rows = max(10_000, min(total_rows or 10_000, target_bytes // bytes_per_row))
        if chunk_rows <= 0:
            chunk_rows = 10_000

        if dataset.logger:
            dataset.logger.warning(
                "HDF5 materialization START:\n\t name \t-> {}\n\t backend \t-> polars/parquet\n\t rows \t-> {}\n\t chunk_rows \t-> {}\n\t selected \t-> {}".format(
                    dataset.name,
                    total_rows,
                    chunk_rows,
                    len(read_paths),
                )
            )

        rows_out = 0
        parts = []
        bytes_total = 0
        columns_out: List[str] = []
        wrote_any_part = False

        for start in range(0, total_rows or 0, chunk_rows):
            end = min(total_rows, start + chunk_rows)
            row_mask = None
            if filter_isvalid_paths:
                for vp in filter_isvalid_paths:
                    vec = np.ravel(f1[vp][start:end]).astype(bool, copy=False)
                    row_mask = vec if row_mask is None else (row_mask & vec)

            chunk_cols = {}
            for path in read_paths:
                chunk_cols.update(
                    _slice_to_columns(path, f1[path], start, end, row_mask=row_mask, rename_map=rename_map)
                )

            if not chunk_cols:
                continue
            chunk_df = pl.DataFrame(chunk_cols)
            if not columns_out:
                columns_out = list(chunk_df.columns)

            if chunk_df.height == 0 and wrote_any_part:
                continue

            part_name = f"part-{len(parts):05d}.parquet"
            part_path = slot / part_name
            chunk_df.write_parquet(part_path)
            parts.append(part_name)
            rows_out += int(chunk_df.height)
            try:
                bytes_total += int(part_path.stat().st_size)
            except Exception:
                pass
            wrote_any_part = True

        if not parts:
            empty_df = pl.DataFrame({})
            empty_path = slot / "part-00000.parquet"
            empty_df.write_parquet(empty_path)
            parts = [empty_path.name]

        manifest = {
            "schema": "jp-materialized-v1",
            "backend": "polars_parquet",
            "rows": rows_out,
            "cols": len(columns_out),
            "columns": columns_out,
            "parts": len(parts),
            "part_files": parts,
            "bytes_total": bytes_total,
            "source": dataset._materialized_source_fingerprint(),
            "group": dataset.group,
            "path": dataset.path,
        }

    dataset.cache.put_materialized_manifest(cache_key, manifest)
    dataset._activate_materialized_manifest(cache_key, manifest)
    if dataset.logger:
        dataset.logger.warning(
            "HDF5 materialization DONE:\n\t name \t-> {}\n\t rows_out \t-> {}\n\t parts \t-> {}".format(
                dataset.name,
                manifest.get("rows", "NA"),
                manifest.get("parts", 0),
            )
        )

def load_hdf5(dataset):
    from .data_loader_summary import dataframe_summary, print_hdf5_tree_ascii

    if pl is not None and dataset.cache is not None:
        try:
            dataset._load_hdf5_materialized()
            dataset._apply_dataset_transform(stage="hdf5")
            summary_msg = None
            source_fp = dataset.fingerprint()
            if dataset.cache is not None:
                summary_msg = dataset.cache.get_summary(source_fp)
            if summary_msg is None:
                if isinstance(dataset.data, pd.DataFrame):
                    summary_msg = dataframe_summary(dataset.data, name=dataset._summary_name())
                elif isinstance(dataset._materialized_manifest, dict):
                    summary_msg = dataset._materialized_summary(dataset._materialized_manifest)
                if dataset.cache is not None and summary_msg is not None:
                    dataset.cache.put_summary(source_fp, summary_msg)
            if summary_msg is not None:
                dataset._emit_summary_text(summary_msg)
            return
        except Exception as e:
            if dataset.logger:
                dataset.logger.warning(f"Polars HDF5 materialization fallback to pandas: {e}")
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

    def _collect_group_dataset_paths(g: h5py.Group, prefix: str="", whitelist=None):
        """Recursively collect dataset paths under a group."""
        paths = []
        for k, v in g.items():
            path = f"{prefix}/{k}" if prefix else k
            if isinstance(v, h5py.Dataset):
                if whitelist is None or path in whitelist:
                    paths.append(path)
            elif isinstance(v, h5py.Group):
                paths.extend(_collect_group_dataset_paths(v, path, whitelist=whitelist))
        return paths

    def _dataset_to_columns(path: str, ds: h5py.Dataset, row_mask=None, rename_map=None):
        """
        Convert one HDF5 dataset to DataFrame-ready columns.
        Returns: (columns_dict, nrows, shape_tuple)
        """
        arr = ds[()]

        def _apply_mask(vec):
            if row_mask is None:
                return vec
            if len(vec) != len(row_mask):
                raise ValueError(
                    "HDF5 row mismatch while applying is_valid mask: "
                    f"dataset='{path}', rows={len(vec)}, mask_rows={len(row_mask)}"
                )
            return vec[row_mask]

        def _map_name(name: str) -> str:
            if isinstance(rename_map, dict):
                return rename_map.get(name, name)
            return name

        if isinstance(arr, np.ndarray) and getattr(arr.dtype, "names", None):
            # Structured array: one output column per field
            out = {}
            for fname in arr.dtype.names:
                src = f"{path}:{fname}"
                dst = _map_name(src)
                vec = np.asarray(arr[fname]).reshape(-1)
                out[dst] = _apply_mask(vec)
            nrows = len(next(iter(out.values()))) if out else 0
            return out, nrows, (nrows, len(out))

        if hasattr(arr, "ndim") and getattr(arr, "ndim", 0) == 2:
            # 2D array: one output column per axis-1 entry
            out = {}
            nrows = int(arr.shape[0])
            ncols = int(arr.shape[1])
            for i in range(ncols):
                src = f"{path}:col{i}"
                dst = _map_name(src)
                out[dst] = _apply_mask(arr[:, i])
            out_rows = len(next(iter(out.values()))) if out else 0
            return out, out_rows, (nrows, ncols)

        # Scalar / 1D / anything else: flatten to one column
        flat = np.ravel(arr)
        src = path if path else "value"
        dst = _map_name(src)
        out = _apply_mask(flat)
        return {dst: out}, len(out), (len(flat), 1)

    with h5py.File(dataset.path, "r") as f1:
        # Tree traversal is expensive on large files; keep it behind debug mode.
        if dataset._debug_enabled() and dataset.group and dataset.group in f1:
            print_hdf5_tree_ascii(f1[dataset.group], root_name=dataset.group, logger=dataset.logger)

        if dataset.group and dataset.group in f1 and isinstance(f1[dataset.group], h5py.Group):
            group = f1[dataset.group]
            dataset.logger.debug("Loading HDF5 group '{}' from {}".format(dataset.group, dataset.path))

            whitelist = None
            if not bool(getattr(dataset, "full_load", False)):
                whitelist = dataset._build_hdf5_whitelist()
                if whitelist is not None:
                    dataset.logger.debug(
                        "HDF5 load_whitelist enabled -> {} paths".format(len(whitelist))
                    )

            # Collect dataset paths only (defer heavy reads to reduce peak memory)
            dataset_paths = _collect_group_dataset_paths(group, prefix=dataset.group, whitelist=whitelist)
            if not dataset_paths:
                if whitelist is not None:
                    raise RuntimeError(
                        "HDF5 group '{}' has no datasets after applying columns.load_whitelist.".format(
                            dataset.group
                        )
                    )
                raise RuntimeError(f"HDF5 group '{dataset.group}' contains no datasets.")

            dataset_set = set(dataset_paths)
            base_paths = [p for p in dataset_paths if not p.endswith("_isvalid")]
            isvalid_paths = [p for p in dataset_paths if p.endswith("_isvalid")]

            # Step 1: parse columns config and finalize rename mapping first.
            rename_entries = dataset._columns_dict().get("rename", [])
            rename_map = {}
            if isinstance(rename_entries, list) and rename_entries:
                dataset.logger.warning("{}: Loading Columns Rename Map".format(dataset.name))
                for item in rename_entries:
                    if not isinstance(item, dict):
                        continue
                    source = str(item.get("source", "")).strip()
                    target = str(item.get("target", "")).strip()
                    if not source or not target:
                        continue
                    source_canon = dataset._canonical_dataset_path(source)
                    rename_map[source_canon] = target
                    rename_map[f"{source_canon}_isvalid"] = f"{target}_isvalid"

            # Step 2: build row mask by isvalid policy, before main table assembly.
            row_mask = None
            filter_rows_before = None
            filter_rows_after = None
            required_base = dataset._whitelist_base_paths if isinstance(dataset._whitelist_base_paths, set) else None

            if dataset.isvalid_policy == "clean":
                filter_isvalid_paths = []
                if required_base is not None:
                    missing_base = sorted([c for c in required_base if c not in dataset_set])
                    missing_isvalid = sorted([f"{c}_isvalid" for c in required_base if f"{c}_isvalid" not in dataset_set])
                    if missing_base or missing_isvalid:
                        dataset.logger.warning(
                            "Skip is_valid policy: not all whitelist columns have companion _isvalid columns. "
                            "missing_base={}, missing_isvalid={}".format(missing_base, missing_isvalid)
                        )
                    else:
                        filter_isvalid_paths = [f"{c}_isvalid" for c in sorted(required_base)]
                else:
                    # No whitelist: use all available *_isvalid columns with base companions.
                    filter_isvalid_paths = [p for p in isvalid_paths if p[:-8] in dataset_set]

                if filter_isvalid_paths:
                    mask = None
                    nrows_ref = None
                    for vp in filter_isvalid_paths:
                        vec = np.ravel(f1[vp][()]).astype(bool, copy=False)
                        if nrows_ref is None:
                            nrows_ref = len(vec)
                            mask = vec.copy()
                        else:
                            if len(vec) != nrows_ref:
                                raise ValueError(
                                    "HDF5 is_valid datasets have inconsistent row counts: '{}' has {}, expected {}.".format(
                                        vp, len(vec), nrows_ref
                                    )
                                )
                            mask &= vec
                    row_mask = mask
                    filter_rows_before = int(nrows_ref or 0)
                    filter_rows_after = int(mask.sum()) if mask is not None else filter_rows_before

            # Step 3: decide final output columns, then assemble DataFrame once.
            # clean mode drops *_isvalid by default; if whitelist is incomplete, keep them (existing behavior).
            read_paths = list(base_paths)
            if dataset.isvalid_policy == "raw":
                read_paths.extend(isvalid_paths)
            elif required_base is not None and row_mask is None:
                read_paths.extend(isvalid_paths)

            merged_cols = {}
            shapes = {}
            expected_rows = None
            row_mismatch = False
            for path in read_paths:
                ds = f1[path]
                cols_dict, nrows, shape_token = _dataset_to_columns(path, ds, row_mask=row_mask, rename_map=rename_map)
                shapes[path] = shape_token

                if expected_rows is None:
                    expected_rows = nrows
                elif nrows != expected_rows:
                    row_mismatch = True

                for cname, cval in cols_dict.items():
                    merged_cols[cname] = cval

            if not row_mismatch:
                # safe to concat by columns → single merged DataFrame only
                dataset.data = pd.DataFrame(merged_cols, copy=False)

                dataset.keys = list(dataset.data.columns)

                if row_mask is not None and dataset.isvalid_policy == "clean":
                    dataset.logger.warning("Filtering Invalid Data from HDF5 Output")
                    dataset.logger.warning(
                        "DataSet Shape: \n\t Before filtering -> ({}, {})\n\t  After filtering -> ({}, {})".format(
                            filter_rows_before if filter_rows_before is not None else dataset.data.shape[0],
                            dataset.data.shape[1],
                            filter_rows_after if filter_rows_after is not None else dataset.data.shape[0],
                            dataset.data.shape[1],
                        )
                    )

                dataset._apply_dataset_transform(stage="hdf5")

                # Emit a pretty summary BEFORE returning
                summary_name = dataset._summary_name()
                source_fp = dataset.fingerprint()
                summary_msg = None
                if dataset.cache is not None:
                    summary_msg = dataset.cache.get_summary(source_fp)
                if summary_msg is None:
                    summary_msg = dataframe_summary(dataset.data, name=summary_name)
                    if dataset.cache is not None:
                        dataset.cache.put_summary(source_fp, summary_msg)
                dataset._emit_summary_text(summary_msg)

                return  # IMPORTANT: stop here; avoid falling through to single-dataset path
            else:
                # Not mergeable → print tree for diagnostics and raise a hard error
                try:
                    print_hdf5_tree_ascii(group, root_name=dataset.group, logger=dataset.logger)
                except Exception:
                    pass
                raise ValueError(
                    "HDF5 group '{grp}' is invalid for merging: datasets have different row counts. "
                    "Please fix the input or choose a different dataset/group. Details: {details}".format(
                        grp=dataset.group,
                        details=shapes,
                    )
                )
        else:
            path, arr = _pick_dataset(f1)
            if not dataset.group:
                dataset.group = path
            if isinstance(arr, np.ndarray):
                if arr.ndim == 0:
                    dataset.data = pd.DataFrame({str(path).split("/")[-1] or "value": [arr.item()]})
                elif getattr(arr.dtype, "names", None):
                    dataset.data = pd.DataFrame.from_records(arr)
                elif arr.ndim == 1:
                    dataset.data = pd.DataFrame({str(path).split("/")[-1] or "value": arr})
                else:
                    dataset.data = pd.DataFrame(arr)
            else:
                dataset.data = pd.DataFrame(np.asarray(arr))
            dataset.keys = list(dataset.data.columns)
            dataset._apply_dataset_transform(stage="hdf5")
            summary_name = dataset._summary_name()
            source_fp = dataset.fingerprint()
            summary_msg = None
            if dataset.cache is not None:
                summary_msg = dataset.cache.get_summary(source_fp)
            if summary_msg is None:
                summary_msg = dataframe_summary(dataset.data, name=summary_name)
                if dataset.cache is not None:
                    dataset.cache.put_summary(source_fp, summary_msg)
            dataset._emit_summary_text(summary_msg)
            return

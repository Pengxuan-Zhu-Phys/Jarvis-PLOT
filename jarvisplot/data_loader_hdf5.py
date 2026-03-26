from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import h5py

from .data_loader_summary import dataframe_summary_human_bytes, format_table

try:
    import polars as pl
except Exception:
    pl = None


def columns_dict(dataset) -> Dict[str, Any]:
    if isinstance(getattr(dataset, "columns", None), dict):
        return dataset.columns
    return {}


def canonical_dataset_path(dataset, value: str) -> str:
    sval = str(value).strip()
    if not sval:
        return ""
    group = getattr(dataset, "group", None)
    if group and not sval.startswith(f"{group}/"):
        return f"{group}/{sval}"
    return sval


def path_aliases(dataset, value: str) -> set[str]:
    sval = str(value).strip()
    if not sval:
        return set()
    aliases = {sval}
    group = getattr(dataset, "group", None)
    if group:
        prefix = f"{group}/"
        if sval.startswith(prefix):
            aliases.add(sval[len(prefix):])
        else:
            aliases.add(prefix + sval)
    return aliases


def rename_source_by_alias(dataset) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    rename_list = columns_dict(dataset).get("rename", [])
    if not isinstance(rename_list, list):
        return alias_map

    for item in rename_list:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        if not source:
            continue
        source_canon = canonical_dataset_path(dataset, source)
        for alias in path_aliases(dataset, source):
            alias_map[alias] = source_canon
        if target:
            alias_map[target] = source_canon
    return alias_map


def build_hdf5_whitelist(dataset) -> Optional[set[str]]:
    cfg = columns_dict(dataset)
    raw_whitelist = cfg.get("load_whitelist", None)
    if raw_whitelist is None:
        dataset._whitelist_base_paths = None
        return None

    use_only_in_list = False
    requested_tokens: List[str] = []

    if isinstance(raw_whitelist, list):
        for item in raw_whitelist:
            if item is None:
                continue
            sval = str(item).strip()
            if sval:
                requested_tokens.append(sval)
    elif isinstance(raw_whitelist, str):
        sval = raw_whitelist.strip()
        if not sval:
            return None
        if sval == "only_in_list":
            use_only_in_list = True
        else:
            raise ValueError("columns.load_whitelist only supports a list or the string 'only_in_list'.")
    else:
        raise TypeError("columns.load_whitelist must be a list or the string 'only_in_list'.")

    if use_only_in_list:
        rename_list = cfg.get("rename", [])
        if not isinstance(rename_list, list):
            raise TypeError("columns.load_whitelist='only_in_list' requires columns.rename to be a list.")
        for item in rename_list:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            if source:
                requested_tokens.append(source)

    alias_to_source = rename_source_by_alias(dataset)
    selected_paths: set[str] = set()
    for token in requested_tokens:
        source = alias_to_source.get(token, token)
        source_canon = canonical_dataset_path(dataset, source)
        if source_canon:
            selected_paths.add(source_canon)
    dataset._whitelist_base_paths = {p for p in selected_paths if not p.endswith("_isvalid")}

    with_isvalid = set(selected_paths)
    for path in list(selected_paths):
        if not path.endswith("_isvalid"):
            with_isvalid.add(f"{path}_isvalid")
    return with_isvalid


def scan_hdf5_leaf_metadata(path: str, group: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return lightweight metadata for leaf datasets without reading dataset bodies."""

    def _walk(node, prefix: str = "") -> None:
        for name, child in node.items():
            child_path = f"{prefix}/{name}" if prefix else name
            if isinstance(child, h5py.Dataset):
                metadata.append(
                    {
                        "path": child_path,
                        "shape": tuple(getattr(child, "shape", ()) or ()),
                        "dtype": str(getattr(child, "dtype", "NA")),
                    }
                )
            elif isinstance(child, h5py.Group):
                _walk(child, child_path)

    group_name = str(group).strip() if group is not None else ""
    metadata: List[Dict[str, Any]] = []

    try:
        with h5py.File(path, "r") as h5f:
            if group_name:
                if group_name not in h5f:
                    raise RuntimeError(f"HDF5 group '{group_name}' not found in '{path}'.")
                root = h5f[group_name]
                if not isinstance(root, h5py.Group):
                    raise RuntimeError(f"HDF5 group '{group_name}' in '{path}' is not a group.")
                _walk(root, group_name)
            else:
                _walk(h5f, "")
    except FileNotFoundError as e:
        raise RuntimeError(f"HDF5 file not found: {path}") from e
    except OSError as e:
        raise RuntimeError(f"Unable to open HDF5 file '{path}': {e}") from e

    if not metadata:
        if group_name:
            raise RuntimeError(f"HDF5 group '{group_name}' in '{path}' contains no leaf datasets.")
        raise RuntimeError(f"No leaf datasets found in HDF5 file '{path}'.")

    return metadata


def shape_token(df) -> str:
    import pandas as pd

    if isinstance(df, pd.DataFrame):
        return f"{df.shape}"
    if pl is not None and isinstance(df, pl.DataFrame):
        return f"({df.height}, {df.width})"
    if pl is not None and isinstance(df, pl.LazyFrame):
        return "lazy"
    return "NA"


def available_memory_bytes() -> Optional[int]:
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except Exception:
        return None


def materialized_source_fingerprint(dataset):
    extra = {
        "name": dataset.name,
        "type": dataset.type,
        "group": dataset.group,
        "columns": dataset.columns,
        "full_load": bool(getattr(dataset, "full_load", False)),
    }
    if dataset.cache is not None:
        return dataset.cache.source_fingerprint(dataset.path, extra=extra)
    try:
        st = os.stat(dataset.path)
        return {
            "path": dataset.path,
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
            "md5": None,
            **extra,
        }
    except Exception:
        return {"path": dataset.path, "size": None, "mtime_ns": None, "md5": None, **extra}


def materialized_cache_key(dataset):
    if dataset.cache is None:
        return None
    source_fp = materialized_source_fingerprint(dataset)
    return dataset.cache.cache_key({"kind": "hdf5-polars-materialized", "source": source_fp})


def materialized_summary(dataset, manifest: Dict[str, Any], stats: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    rows = manifest.get("rows", "NA")
    cols = manifest.get("cols", len(manifest.get("columns", [])))
    parts = manifest.get("parts", 0)
    bytes_total = manifest.get("bytes_total", 0)
    lines = [
        f"Selected dataset:{dataset._summary_name()}",
        f"\t Materialized backend\t-> polars/parquet",
        f"\t DataFrame shape\t-> {rows}\t rows × {cols} \tcols",
        f"\t Parquet parts\t-> {parts}",
    ]
    try:
        lines.append(f"\t Materialized size\t-> {dataframe_summary_human_bytes(int(bytes_total))}")
    except Exception:
        pass
    if manifest.get("columns"):
        lines.append("=== Materialized Columns ===")
        lines.append(", ".join(str(c) for c in manifest.get("columns", [])))
    if isinstance(stats, dict) and stats:
        lines.append("=== Materialized Stats ===")
        stat_rows = []
        df = getattr(dataset, "data", None)
        schema_names = None
        schema_dtypes = {}
        if pl is not None and isinstance(df, pl.LazyFrame):
            try:
                schema = df.collect_schema()
                if hasattr(schema, "names"):
                    schema_names = set(str(x) for x in schema.names())
                elif isinstance(schema, dict):
                    schema_names = set(str(x) for x in schema.keys())
                    schema_dtypes = {str(k): str(v) for k, v in schema.items()}
                elif hasattr(schema, "items"):
                    schema_names = set(str(x) for x in schema.keys())
                    schema_dtypes = {str(k): str(v) for k, v in schema.items()}
            except Exception:
                schema_names = set()
        elif pl is not None and isinstance(df, pl.DataFrame):
            try:
                schema_names = set(str(x) for x in df.columns)
            except Exception:
                schema_names = set()
        elif hasattr(df, "columns"):
            try:
                schema_names = set(str(x) for x in getattr(df, "columns", []))
            except Exception:
                schema_names = set()
        for col in manifest.get("columns", []):
            if col not in stats:
                continue
            bounds = stats.get(col, {})
            dtype = "NA"
            nonnull_text = "NA"
            if pl is not None and isinstance(df, pl.LazyFrame):
                dtype = schema_dtypes.get(col, "NA")
            elif schema_names is not None and col in schema_names and hasattr(df, "__getitem__"):
                series = df[col]
                dtype = str(series.dtype)
                try:
                    nonnull_pct = (float(series.notna().sum()) / float(len(series)) * 100.0) if len(series) else 0.0
                except Exception:
                    nonnull_pct = 0.0
                nonnull_text = f"{nonnull_pct:.1f}%"
            else:
                dtype = str(type(bounds.get("min", None)).__name__)
            stat_rows.append(
                (
                    str(col),
                    dtype,
                    nonnull_text,
                    bounds.get("min", None),
                    bounds.get("max", None),
                )
            )

        if stat_rows:
            lines.append(
                format_table(
                    ["name", "dtype", "nonnull%", "[min]", "[max]"],
                    stat_rows,
                    aligns=["left", "left", "right", "right", "right"],
                )
            )
    return "\n".join(lines)
def sql_bool_ops(expr: Any) -> str:
    return str(expr).replace("&&", " AND ").replace("||", " OR ")


def polars_schema_names(lf) -> List[str]:
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

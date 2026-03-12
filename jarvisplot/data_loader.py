#!/usr/bin/env python3 

from __future__ import annotations
from pathlib import Path
from typing import Optional, Any, Dict, List
import gc
import os
import pandas as pd 
import h5py
import numpy as np
from .memtrace import memtrace_checkpoint, memtrace_enabled, memtrace_object_inventory

try:
    import polars as pl
except Exception:
    pl = None


JP_ROW_IDX = "__jp_row_idx__"


def polars_to_pandas_compat(frame, logger=None, stage: str = "dataset"):
    if pl is None:
        return frame
    if isinstance(frame, pl.LazyFrame):
        lazy_frame = frame
        memtrace_checkpoint(logger, f"{stage}.polars_collect.before", frame)
        frame = frame.collect()
        memtrace_checkpoint(logger, f"{stage}.polars_collect.after", frame)
        memtrace_object_inventory(
            logger,
            f"{stage}.polars_collect.inventory",
            {"lazy_frame": lazy_frame, "collected": frame},
            roles={
                "lazy_frame": "lazy source plan",
                "collected": "collected polars dataframe",
            },
            min_bytes=64 * 1024 * 1024,
        )
    if isinstance(frame, pl.DataFrame):
        memtrace_checkpoint(logger, f"{stage}.pandas_convert.before", frame)
        try:
            out = frame.to_pandas()
        except ModuleNotFoundError:
            out = pd.DataFrame(frame.to_dict(as_series=False))
        memtrace_checkpoint(logger, f"{stage}.pandas_convert.after", out)
        memtrace_object_inventory(
            logger,
            f"{stage}.pandas_convert.inventory",
            {"polars_df": frame, "pandas_df": out},
            roles={
                "polars_df": "polars dataframe before conversion",
                "pandas_df": "converted pandas dataframe",
            },
            min_bytes=64 * 1024 * 1024,
        )
        return out
    return frame

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
        self.columns                = {}
        self.isvalid_policy         = "clean"
        self.full_load              = False
        self.transform              = None
        self.required_columns: Optional[set[str]] = None
        self.retained_columns: Optional[set[str]] = None
        self._whitelist_base_paths  = None
        self.cache                  = None
        self._loaded                = False
        self._summary_emitted       = False
        self._data_backend          = "pandas"
        self._materialized_manifest = None
        self._full_lazy_frame       = None
       
    def setinfo(self, dtinfo, rootpath, eager: bool = False, cache=None):
        self.cache = cache
        raw_path = str(dtinfo['path'])
        p = Path(raw_path).expanduser()
        if p.is_absolute():
            resolved = p
        else:
            base = Path(rootpath or ".").expanduser().resolve()
            resolved = (base / p).resolve()
        self.file = str(resolved)
        self.name = dtinfo['name']
        self.type = dtinfo['type'].lower()
        self.transform = dtinfo.get("transform", None)
        self.required_columns = None
        self.retained_columns = None
        self._full_lazy_frame = None
        if self.type == "hdf5":
            self.group = dtinfo.get("dataset", None)
            self.columns = dtinfo.get("columns", {})
            if not isinstance(self.columns, dict):
                self.columns = {}
            policy_raw = str(self.columns.get("isvalid_policy", "clean")).strip().lower()
            if policy_raw not in {"clean", "raw"}:
                raise ValueError(
                    "columns.isvalid_policy must be one of: clean, raw. got: {}".format(
                        self.columns.get("isvalid_policy")
                    )
                )
            self.isvalid_policy = policy_raw

        if eager:
            self.load(force=True)
        else:
            self._prepare_lazy_metadata()

    def set_required_columns(self, columns: Optional[set[str]], retained: Optional[set[str]] = None) -> None:
        try:
            cleaned = {str(c).strip() for c in (columns or set()) if str(c).strip()}
        except Exception:
            cleaned = set()
        try:
            kept = {str(c).strip() for c in (retained or set()) if str(c).strip()}
        except Exception:
            kept = set()
        self.required_columns = cleaned if cleaned else None
        if retained is None:
            self.retained_columns = self.required_columns
        else:
            self.retained_columns = kept if kept else None

    def _prepare_lazy_metadata(self):
        """Fetch cheap metadata only; avoid full table load."""
        if self.type == "csv":
            try:
                head = pd.read_csv(self.path, nrows=0)
                self.keys = list(head.columns)
                if self.logger:
                    self.logger.debug(
                        f"Dataset '{self.name}' registered in lazy mode (csv columns={len(self.keys)})."
                    )
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Dataset '{self.name}' lazy metadata failed: {e}")
        else:
            if self.logger:
                self.logger.debug(f"Dataset '{self.name}' registered in lazy mode.")

    def fingerprint(self, cache=None):
        cache = cache or self.cache
        extra = {
            "name": self.name,
            "type": self.type,
            "group": self.group,
            "transform": self.transform,
            "columns": self.columns,
            "full_load": bool(getattr(self, "full_load", False)),
        }
        if cache is not None:
            return cache.source_fingerprint(self.path, extra=extra)
        try:
            st = os.stat(self.path)
            return {
                "path": self.path,
                "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
                "md5": None,
                **extra,
            }
        except Exception:
            return {"path": self.path, "size": None, "mtime_ns": None, "md5": None, **extra}

    def load(self, force: bool = False):
        if self._loaded and self.data is not None and not force:
            return self.data
        if self.type == "csv":
            self.load_csv()
        elif self.type == "hdf5":
            self.load_hdf5()
        else:
            raise ValueError(f"Unsupported dataset type: {self.type}")
        self._loaded = True
        return self.data

    def get_data(self):
        return self.load(force=False)

    def release(self):
        self.data = None
        self._loaded = False

    def fetch_rows_columns(self, row_ids, columns, row_key: str = JP_ROW_IDX):
        cols = [str(c).strip() for c in (columns or []) if str(c).strip() and str(c).strip() != row_key]
        try:
            n_rows = len(row_ids)
        except Exception:
            n_rows = 0
        if not cols:
            return pd.DataFrame(index=np.arange(n_rows))

        try:
            order = np.asarray(row_ids, dtype=np.int64)
        except Exception:
            order = np.asarray(list(row_ids or []), dtype=np.int64)
        out = pd.DataFrame(index=np.arange(int(order.shape[0])))

        if isinstance(self.data, pd.DataFrame) and row_key in self.data.columns:
            avail = [c for c in cols if c in self.data.columns]
            if avail:
                base = self.data[[row_key] + avail].drop_duplicates(subset=[row_key], keep="last").set_index(row_key)
                pulled = base.reindex(order)
                pulled.index = np.arange(int(pulled.shape[0]))
                for col in avail:
                    out[col] = pulled[col].to_numpy(copy=False)

        missing = [c for c in cols if c not in out.columns]
        if missing and pl is not None and isinstance(self._full_lazy_frame, pl.LazyFrame):
            schema = self._polars_schema_names(self._full_lazy_frame)
            miss_avail = [c for c in missing if c in schema]
            if miss_avail:
                lf = self._full_lazy_frame.select([row_key] + miss_avail)
                uniq = np.unique(order)
                total_rows = 0
                if isinstance(self._materialized_manifest, dict):
                    try:
                        total_rows = int(self._materialized_manifest.get("rows", 0) or 0)
                    except Exception:
                        total_rows = 0
                if uniq.size > 0 and (total_rows <= 0 or uniq.size < total_rows):
                    lf = lf.filter(pl.col(row_key).is_in(pl.Series(name=row_key, values=uniq.tolist())))
                pulled = polars_to_pandas_compat(lf, logger=self.logger, stage=f"dataset:{self.name}.lookup")
                if isinstance(pulled, pd.DataFrame) and row_key in pulled.columns:
                    pulled = pulled.drop_duplicates(subset=[row_key], keep="last").set_index(row_key).reindex(order)
                    pulled.index = np.arange(int(pulled.shape[0]))
                    for col in miss_avail:
                        out[col] = pulled[col].to_numpy(copy=False)

        return out

    def _summary_name(self) -> str:
        if self.type == "hdf5":
            return f" HDF5 loaded!\n\t name  -> {self.name}\n\t group -> {self.group}\n\t path  -> {self.path}"
        return f" CSV loaded!\n\t name  -> {self.name}\n\t path  -> {self.path}"

    def _debug_enabled(self) -> bool:
        """Best-effort check for loguru debug level without assuming logger internals."""
        try:
            core = getattr(self.logger, "_core", None)
            min_level = getattr(core, "min_level", 100)
            return int(min_level) <= 10
        except Exception:
            return False

    def _emit_summary_text(self, summary_msg: str) -> None:
        if self.logger:
            self.logger.warning("\n" + str(summary_msg))
        else:
            print(summary_msg)
        self._summary_emitted = True

    def emit_summary(self, force_load: bool = True) -> bool:
        """Emit source summary even when data pipeline hits cache.

        Priority:
        1) cached summary text in .cache/summary
        2) build from already loaded dataframe
        3) optional fallback: load source and emit summary from loader
        """
        if self._summary_emitted:
            return True

        source_fp = self.fingerprint()
        if self.cache is not None:
            cached = self.cache.get_summary(source_fp)
            if cached is not None:
                self._emit_summary_text(cached)
                return True

        if self.data is not None:
            try:
                if isinstance(self._materialized_manifest, dict):
                    msg = self._materialized_summary(self._materialized_manifest)
                else:
                    msg = dataframe_summary(self.data, name=self._summary_name())
            except Exception:
                if isinstance(self._materialized_manifest, dict):
                    msg = self._materialized_summary(self._materialized_manifest)
                else:
                    msg = f"Data backend: {type(self.data).__name__}"
            if self.cache is not None:
                self.cache.put_summary(source_fp, msg)
            self._emit_summary_text(msg)
            return True

        if force_load:
            self.load(force=False)
            self._summary_emitted = True
            return True

        return False

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
        if self.logger:
            self.logger.debug("Dataset -> {} is assigned as \n\t-> {}\ttype".format(self.base, self.type))

    def _columns_dict(self) -> Dict[str, Any]:
        if isinstance(self.columns, dict):
            return self.columns
        return {}

    def _canonical_dataset_path(self, value: str) -> str:
        sval = str(value).strip()
        if not sval:
            return ""
        if self.group and not sval.startswith(f"{self.group}/"):
            return f"{self.group}/{sval}"
        return sval

    def _path_aliases(self, value: str) -> set[str]:
        sval = str(value).strip()
        if not sval:
            return set()
        aliases = {sval}
        if self.group:
            prefix = f"{self.group}/"
            if sval.startswith(prefix):
                aliases.add(sval[len(prefix):])
            else:
                aliases.add(prefix + sval)
        return aliases

    def _rename_source_by_alias(self) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}
        rename_list = self._columns_dict().get("rename", [])
        if not isinstance(rename_list, list):
            return alias_map

        for item in rename_list:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            if not source:
                continue
            source_canon = self._canonical_dataset_path(source)
            for alias in self._path_aliases(source):
                alias_map[alias] = source_canon
            if target:
                alias_map[target] = source_canon
        return alias_map

    def _build_hdf5_whitelist(self) -> Optional[set[str]]:
        cfg = self._columns_dict()
        raw_whitelist = cfg.get("load_whitelist", None)
        if raw_whitelist is None:
            self._whitelist_base_paths = None
            return None

        use_only_in_list = False
        requested_tokens: List[str] = []

        if isinstance(raw_whitelist, list):
            for item in raw_whitelist:
                if item is None:
                    continue
                sval = str(item).strip()
                if not sval:
                    continue
                requested_tokens.append(sval)
        elif isinstance(raw_whitelist, str):
            sval = raw_whitelist.strip()
            if not sval:
                return None
            if sval == "only_in_list":
                use_only_in_list = True
            else:
                raise ValueError(
                    "columns.load_whitelist only supports a list or the string 'only_in_list'."
                )
        else:
            raise TypeError(
                "columns.load_whitelist must be a list or the string 'only_in_list'."
            )

        if use_only_in_list:
            rename_list = cfg.get("rename", [])
            if not isinstance(rename_list, list):
                raise TypeError(
                    "columns.load_whitelist='only_in_list' requires columns.rename to be a list."
                )
            for item in rename_list:
                if not isinstance(item, dict):
                    continue
                source = str(item.get("source", "")).strip()
                if source:
                    requested_tokens.append(source)

        alias_to_source = self._rename_source_by_alias()
        selected_paths: set[str] = set()
        for token in requested_tokens:
            source = alias_to_source.get(token, token)
            source_canon = self._canonical_dataset_path(source)
            if source_canon:
                selected_paths.add(source_canon)
        self._whitelist_base_paths = {p for p in selected_paths if not p.endswith("_isvalid")}

        # Always add companion validity columns so filtering/keeping works consistently.
        with_isvalid = set(selected_paths)
        for path in list(selected_paths):
            if not path.endswith("_isvalid"):
                with_isvalid.add(f"{path}_isvalid")
        return with_isvalid

    @staticmethod
    def _shape_token(df) -> str:
        if isinstance(df, pd.DataFrame):
            return f"{df.shape}"
        if pl is not None and isinstance(df, pl.DataFrame):
            return f"({df.height}, {df.width})"
        if pl is not None and isinstance(df, pl.LazyFrame):
            return "lazy"
        return "NA"

    @staticmethod
    def _available_memory_bytes() -> Optional[int]:
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages) * int(page_size)
        except Exception:
            return None

    def _materialized_cache_key(self):
        if self.cache is None:
            return None
        source_fp = self._materialized_source_fingerprint()
        return self.cache.cache_key({"kind": "hdf5-polars-materialized", "source": source_fp})

    def _materialized_source_fingerprint(self):
        extra = {
            "name": self.name,
            "type": self.type,
            "group": self.group,
            "columns": self.columns,
            "full_load": bool(getattr(self, "full_load", False)),
        }
        if self.cache is not None:
            return self.cache.source_fingerprint(self.path, extra=extra)
        try:
            st = os.stat(self.path)
            return {
                "path": self.path,
                "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
                "md5": None,
                **extra,
            }
        except Exception:
            return {"path": self.path, "size": None, "mtime_ns": None, "md5": None, **extra}

    def _materialized_summary(self, manifest: Dict[str, Any]) -> str:
        rows = manifest.get("rows", "NA")
        cols = manifest.get("cols", len(manifest.get("columns", [])))
        parts = manifest.get("parts", 0)
        bytes_total = manifest.get("bytes_total", 0)
        lines = [
            f"Selected dataset:{self._summary_name()}",
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
        return "\n".join(lines)

    def _apply_dataset_transform(self, stage: str = "dataset") -> None:
        """
        Apply DataSet-level transforms using the same operators as layer transforms.
        For HDF5 this is called after is_valid policy + columns rename.
        """
        if self.data is None:
            return
        if self.transform is None:
            return
        if not isinstance(self.transform, list):
            if self.logger:
                self.logger.warning(
                    "Dataset '{}' transform ignored: list required, got {}.".format(
                        self.name, type(self.transform)
                    )
                )
            return

        from .Figure.load_data import addcolumn, filter as filter_df, grid_profiling, profiling, sortby

        push_mode = str(os.getenv("JP_DATASET_POLARS_TRANSFORM", "auto")).strip().lower()
        use_polars_pushdown = push_mode not in {"0", "false", "no", "off", "disable", "disabled"}
        pushdown_keep_lazy = push_mode in {"lazy", "keep-lazy"}
        if use_polars_pushdown and pl is not None and isinstance(self.data, (pl.LazyFrame, pl.DataFrame)):
            if self._apply_dataset_transform_polars(
                stage=stage,
                materialize_to_pandas=(not pushdown_keep_lazy),
            ):
                return
            if self.logger:
                self.logger.warning(
                    "Dataset '{}' transform pushdown fallback to pandas.".format(self.name)
                )

        if pl is not None and isinstance(self.data, pl.LazyFrame):
            self.logger.warning(f"Materializing polars LazyFrame for dataset transform -> {self.name}")
            self.data = polars_to_pandas_compat(self.data, logger=self.logger, stage=f"dataset:{self.name}")
            self._data_backend = "pandas"
        elif pl is not None and isinstance(self.data, pl.DataFrame):
            self.logger.warning(f"Materializing polars DataFrame for dataset transform -> {self.name}")
            self.data = polars_to_pandas_compat(self.data, logger=self.logger, stage=f"dataset:{self.name}")
            self._data_backend = "pandas"

        df = self.data
        memtrace_checkpoint(
            self.logger,
            "dataset.transform.before",
            df,
            extra={"dataset": self.name, "stage": stage},
        )
        before_shape = self._shape_token(df)
        for trans in self.transform:
            if not isinstance(trans, dict):
                if self.logger:
                    self.logger.warning(f"Dataset '{self.name}' invalid transform step skipped -> {trans}")
                continue
            prev_df = df

            if "filter" in trans:
                df = filter_df(df, trans["filter"], self.logger)
            elif "profile" in trans:
                df = profiling(df, trans["profile"], self.logger)
            elif "grid_profile" in trans:
                cfg = trans.get("grid_profile", {})
                if isinstance(cfg, dict):
                    cfg = cfg.copy()
                    cfg.setdefault("method", "grid")
                else:
                    cfg = {"method": "grid"}
                df = grid_profiling(df, cfg, self.logger)
            elif "sortby" in trans:
                df = sortby(df, trans["sortby"], self.logger)
            elif "add_column" in trans:
                df = addcolumn(df, trans["add_column"], self.logger)

            if prev_df is not df and isinstance(prev_df, pd.DataFrame):
                collect_prev = int(prev_df.shape[0]) >= 100_000
                prev_df = None
                if collect_prev:
                    gc.collect()

        self.data = df
        if isinstance(self.data, pd.DataFrame):
            if JP_ROW_IDX not in self.data.columns:
                self.data.insert(0, JP_ROW_IDX, np.arange(int(self.data.shape[0]), dtype=np.int64))
            if isinstance(self.retained_columns, set) and self.retained_columns:
                retained = [c for c in self.data.columns if c in self.retained_columns or c == JP_ROW_IDX]
                if retained and len(retained) < int(self.data.shape[1]):
                    self.data = self.data.loc[:, retained].copy(deep=False)
        memtrace_checkpoint(
            self.logger,
            "dataset.transform.after",
            self.data,
            extra={"dataset": self.name, "stage": stage},
        )
        if isinstance(self.data, pd.DataFrame):
            self.keys = list(self.data.columns)
        if self.logger:
            self.logger.warning(
                "DataSet transform done:\n\t name \t-> {}\n\t stage \t-> {}\n\t rows \t-> {} -> {}".format(
                    self.name,
                    stage,
                    before_shape,
                    self._shape_token(self.data),
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
        if not isinstance(self.data, (pl.LazyFrame, pl.DataFrame)):
            return False
        if not isinstance(self.transform, list):
            return False

        lf = self.data if isinstance(self.data, pl.LazyFrame) else self.data.lazy()
        before_shape = self._shape_token(self.data)
        memtrace_checkpoint(
            self.logger,
            "dataset.transform.before",
            self.data,
            extra={"dataset": self.name, "stage": stage, "engine": "polars"},
        )
        try:
            for trans in self.transform:
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
                            lf = lf.filter(pl.sql_expr(self._sql_bool_ops(s)))
                    else:
                        raise TypeError(
                            "unsupported filter condition type for polars pushdown: {}".format(type(condition))
                        )
                elif "sortby" in trans:
                    expr = str(trans.get("sortby", "")).strip()
                    if not expr:
                        continue
                    cols = self._polars_schema_names(lf)
                    if expr in cols:
                        lf = lf.sort(expr)
                    else:
                        skey = "__jp_sortkey__"
                        lf = lf.with_columns(pl.sql_expr(self._sql_bool_ops(expr)).alias(skey)).sort(skey).drop(skey)
                elif "add_column" in trans:
                    adds = trans.get("add_column", {})
                    if not isinstance(adds, dict):
                        raise TypeError("add_column step must be dict")
                    name = str(adds.get("name", "")).strip()
                    expr = str(adds.get("expr", "")).strip()
                    if not name or not expr:
                        raise ValueError("add_column requires non-empty name and expr")
                    lf = lf.with_columns(pl.sql_expr(self._sql_bool_ops(expr)).alias(name))
                elif "profile" in trans or "grid_profile" in trans:
                    return False
                else:
                    return False
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    "Dataset '{}' polars transform pushdown failed: {}.".format(self.name, e)
                )
            return False

        lf_with_idx = lf
        cols_after_transform = self._polars_schema_names(lf_with_idx)
        if JP_ROW_IDX not in cols_after_transform:
            lf_with_idx = lf_with_idx.with_row_index(JP_ROW_IDX)
            cols_after_transform = self._polars_schema_names(lf_with_idx)
        self._full_lazy_frame = lf_with_idx

        keep_cols = list(cols_after_transform)
        if isinstance(self.retained_columns, set) and self.retained_columns:
            retained = set(self.retained_columns)
            retained.add(JP_ROW_IDX)
            keep_cols = [c for c in cols_after_transform if c in retained]

        if materialize_to_pandas:
            manifest_rows = None
            manifest_bytes = None
            if isinstance(self._materialized_manifest, dict):
                manifest_rows = self._materialized_manifest.get("rows")
                manifest_bytes = self._materialized_manifest.get("bytes_total")
            est_bytes = None
            try:
                if manifest_bytes is not None and cols_after_transform:
                    est_bytes = int(float(manifest_bytes) * (float(len(keep_cols)) / float(len(cols_after_transform))))
            except Exception:
                est_bytes = None
            if memtrace_enabled():
                memtrace_checkpoint(
                    self.logger,
                    f"dataset:{self.name}.pushdown.collect_plan",
                    None,
                    extra={
                        "dataset": self.name,
                        "rows": manifest_rows,
                        "columns": len(cols_after_transform),
                        "required_columns": len(keep_cols),
                        "estimated_size": est_bytes,
                        "column_list": "|".join(cols_after_transform),
                        "required_list": "|".join(keep_cols),
                    },
                )
            if keep_cols and len(keep_cols) < len(cols_after_transform) and self.logger:
                self.logger.warning(
                    "Dataset '{}' pre-collect column projection -> {} -> {}".format(
                        self.name,
                        len(cols_after_transform),
                        len(keep_cols),
                    )
                )
            narrowed = lf_with_idx.select(keep_cols) if keep_cols else lf_with_idx
            self.data = polars_to_pandas_compat(narrowed, logger=self.logger, stage=f"dataset:{self.name}.pushdown")
            self._data_backend = "pandas"
            if isinstance(self.data, pd.DataFrame):
                self.keys = list(self.data.columns)
            else:
                self.keys = self._polars_schema_names(narrowed)
            engine_name = "polars->pandas"
        else:
            self.data = lf_with_idx.select(keep_cols) if keep_cols else lf_with_idx
            self._data_backend = "polars_lazy"
            self.keys = self._polars_schema_names(self.data)
            engine_name = "polars"
        memtrace_checkpoint(
            self.logger,
            "dataset.transform.after",
            self.data,
            extra={"dataset": self.name, "stage": stage, "engine": engine_name},
        )
        if self.logger:
            self.logger.warning(
                "DataSet transform done:\n\t name \t-> {}\n\t stage \t-> {}\n\t engine \t-> {}\n\t rows \t-> {} -> {}".format(
                    self.name,
                    stage,
                    engine_name,
                    before_shape,
                    self._shape_token(self.data),
                )
            )
        return True
    
    def load_csv(self):
        if self.type == "csv":
            if self.logger:
                self.logger.debug("Loading CSV from {}".format(self.path))

            self.data = pd.read_csv(self.path)
            self.keys = list(self.data.columns)
            self._apply_dataset_transform(stage="csv")

            # Emit the same pretty summary used for HDF5 datasets
            summary_name = self._summary_name()
            summary_msg = None
            source_fp = self.fingerprint()
            if self.cache is not None:
                summary_msg = self.cache.get_summary(source_fp)

            if summary_msg is None:
                try:
                    summary_msg = dataframe_summary(self.data, name=summary_name)
                except Exception:
                    # Fallback minimal summary if something goes wrong
                    summary_msg = f"CSV loaded  {summary_name}\nDataFrame shape: {self.data.shape}"
                if self.cache is not None:
                    self.cache.put_summary(source_fp, summary_msg)

            self._emit_summary_text(summary_msg)

    def _activate_materialized_manifest(self, cache_key: str, manifest: Dict[str, Any]) -> None:
        if pl is None or self.cache is None:
            raise RuntimeError("polars materialized backend is unavailable.")
        slot = self.cache.materialized_slot(cache_key)
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
        self.data = pl.scan_parquet(paths)
        self.keys = list(manifest.get("columns", []))
        self._materialized_manifest = dict(manifest)
        self._data_backend = "polars_lazy"
        memtrace_checkpoint(
            self.logger,
            "hdf5.materialized.ready",
            self.data,
            extra={
                "dataset": self.name,
                "rows": manifest.get("rows"),
                "cols": manifest.get("cols"),
                "parts": manifest.get("parts"),
                "bytes_total": manifest.get("bytes_total"),
                "cache_key": cache_key[:16] if isinstance(cache_key, str) else cache_key,
            },
        )

    def _load_hdf5_materialized(self) -> None:
        if pl is None or self.cache is None:
            raise RuntimeError("polars materialized backend is unavailable.")

        cache_key = self._materialized_cache_key()
        if cache_key is None:
            raise RuntimeError("materialized cache key unavailable.")

        manifest = self.cache.get_materialized_manifest(cache_key)
        if isinstance(manifest, dict):
            self._activate_materialized_manifest(cache_key, manifest)
            memtrace_checkpoint(
                self.logger,
                "hdf5.materialized.cache_hit",
                self.data,
                extra={"dataset": self.name, "cache_key": cache_key[:16]},
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
            rename_entries = self._columns_dict().get("rename", [])
            rename_map = {}
            if isinstance(rename_entries, list):
                for item in rename_entries:
                    if not isinstance(item, dict):
                        continue
                    source = str(item.get("source", "")).strip()
                    target = str(item.get("target", "")).strip()
                    if not source or not target:
                        continue
                    source_canon = self._canonical_dataset_path(source)
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

        slot = self.cache.clear_materialized_slot(cache_key)
        rename_map = {}
        manifest = None

        with h5py.File(self.path, "r") as f1:
            if not (self.group in f1 and isinstance(f1[self.group], h5py.Group)):
                raise RuntimeError(
                    f"HDF5 group '{self.group}' is required for polars materialization."
                )

            group = f1[self.group]
            whitelist = None
            if not bool(getattr(self, "full_load", False)):
                whitelist = self._build_hdf5_whitelist()

            dataset_paths = list(_iter_dataset_paths(group, prefix=self.group, whitelist=whitelist))
            if not dataset_paths:
                raise RuntimeError(f"HDF5 group '{self.group}' contains no datasets.")

            dataset_set = set(dataset_paths)
            base_paths = [p for p in dataset_paths if not p.endswith("_isvalid")]
            isvalid_paths = [p for p in dataset_paths if p.endswith("_isvalid")]
            rename_map = _rename_map_from_entries()

            required_base = self._whitelist_base_paths if isinstance(self._whitelist_base_paths, set) else None
            filter_isvalid_paths = []
            if self.isvalid_policy == "clean":
                if required_base is not None:
                    missing_base = sorted([c for c in required_base if c not in dataset_set])
                    missing_isvalid = sorted(
                        [f"{c}_isvalid" for c in required_base if f"{c}_isvalid" not in dataset_set]
                    )
                    if missing_base or missing_isvalid:
                        self.logger.warning(
                            "Skip is_valid policy for materialization: not all whitelist columns have companion _isvalid columns. "
                            "missing_base={}, missing_isvalid={}".format(missing_base, missing_isvalid)
                        )
                    else:
                        filter_isvalid_paths = [f"{c}_isvalid" for c in sorted(required_base)]
                else:
                    filter_isvalid_paths = [p for p in isvalid_paths if p[:-8] in dataset_set]

            read_paths = list(base_paths)
            if self.isvalid_policy == "raw":
                read_paths.extend(isvalid_paths)
            elif required_base is not None and not filter_isvalid_paths:
                read_paths.extend(isvalid_paths)

            if not read_paths:
                raise RuntimeError(f"HDF5 group '{self.group}' has no readable datasets.")

            expected_rows = None
            bytes_per_row = 0
            for path in read_paths:
                ds = f1[path]
                nrows = _dataset_nrows(ds)
                if expected_rows is None:
                    expected_rows = nrows
                elif nrows != expected_rows:
                    raise ValueError(
                        f"HDF5 group '{self.group}' is invalid for materialization: row mismatch at '{path}'."
                    )
                bytes_per_row += _estimate_bytes_per_row(ds)
            for path in filter_isvalid_paths:
                nrows = _dataset_nrows(f1[path])
                if expected_rows is None:
                    expected_rows = nrows
                elif nrows != expected_rows:
                    raise ValueError(
                        f"HDF5 group '{self.group}' is invalid for materialization: row mismatch at '{path}'."
                    )
                bytes_per_row += 1

            total_rows = int(expected_rows or 0)
            avail = self._available_memory_bytes()
            target_bytes = max(64 * 1024 * 1024, int((avail or (512 * 1024 * 1024)) * 0.15))
            bytes_per_row = max(int(bytes_per_row), 1)
            chunk_rows = max(10_000, min(total_rows or 10_000, target_bytes // bytes_per_row))
            if chunk_rows <= 0:
                chunk_rows = 10_000

            if self.logger:
                self.logger.warning(
                    "HDF5 materialization START:\n\t name \t-> {}\n\t backend \t-> polars/parquet\n\t rows \t-> {}\n\t chunk_rows \t-> {}\n\t selected \t-> {}".format(
                        self.name,
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
                "source": self._materialized_source_fingerprint(),
                "group": self.group,
                "path": self.path,
            }

        self.cache.put_materialized_manifest(cache_key, manifest)
        self._activate_materialized_manifest(cache_key, manifest)
        if self.logger:
            self.logger.warning(
                "HDF5 materialization DONE:\n\t name \t-> {}\n\t rows_out \t-> {}\n\t parts \t-> {}".format(
                    self.name,
                    manifest.get("rows", "NA"),
                    manifest.get("parts", 0),
                )
            )
    
    def load_hdf5(self):
            if pl is not None and self.cache is not None:
                try:
                    self._load_hdf5_materialized()
                    self._apply_dataset_transform(stage="hdf5")
                    summary_msg = None
                    source_fp = self.fingerprint()
                    if self.cache is not None:
                        summary_msg = self.cache.get_summary(source_fp)
                    if summary_msg is None:
                        if isinstance(self.data, pd.DataFrame):
                            summary_msg = dataframe_summary(self.data, name=self._summary_name())
                        elif isinstance(self._materialized_manifest, dict):
                            summary_msg = self._materialized_summary(self._materialized_manifest)
                        if self.cache is not None and summary_msg is not None:
                            self.cache.put_summary(source_fp, summary_msg)
                    if summary_msg is not None:
                        self._emit_summary_text(summary_msg)
                    return
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Polars HDF5 materialization fallback to pandas: {e}")
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

            with h5py.File(self.path, "r") as f1:
                # Tree traversal is expensive on large files; keep it behind debug mode.
                if self._debug_enabled() and self.group in f1:
                    print_hdf5_tree_ascii(f1[self.group], root_name=self.group, logger=self.logger)

                if self.group in f1 and isinstance(f1[self.group], h5py.Group):
                    group = f1[self.group]
                    self.logger.debug("Loading HDF5 group '{}' from {}".format(self.group, self.path))

                    whitelist = None
                    if not bool(getattr(self, "full_load", False)):
                        whitelist = self._build_hdf5_whitelist()
                        if whitelist is not None:
                            self.logger.debug(
                                "HDF5 load_whitelist enabled -> {} paths".format(len(whitelist))
                            )

                    # Collect dataset paths only (defer heavy reads to reduce peak memory)
                    dataset_paths = _collect_group_dataset_paths(group, prefix=self.group, whitelist=whitelist)
                    if not dataset_paths:
                        if whitelist is not None:
                            raise RuntimeError(
                                "HDF5 group '{}' has no datasets after applying columns.load_whitelist.".format(
                                    self.group
                                )
                            )
                        raise RuntimeError(f"HDF5 group '{self.group}' contains no datasets.")

                    dataset_set = set(dataset_paths)
                    base_paths = [p for p in dataset_paths if not p.endswith("_isvalid")]
                    isvalid_paths = [p for p in dataset_paths if p.endswith("_isvalid")]

                    # Step 1: parse columns config and finalize rename mapping first.
                    rename_entries = self._columns_dict().get("rename", [])
                    rename_map = {}
                    if isinstance(rename_entries, list) and rename_entries:
                        self.logger.warning("{}: Loading Columns Rename Map".format(self.name))
                        for item in rename_entries:
                            if not isinstance(item, dict):
                                continue
                            source = str(item.get("source", "")).strip()
                            target = str(item.get("target", "")).strip()
                            if not source or not target:
                                continue
                            source_canon = self._canonical_dataset_path(source)
                            rename_map[source_canon] = target
                            rename_map[f"{source_canon}_isvalid"] = f"{target}_isvalid"

                    # Step 2: build row mask by isvalid policy, before main table assembly.
                    row_mask = None
                    filter_rows_before = None
                    filter_rows_after = None
                    required_base = self._whitelist_base_paths if isinstance(self._whitelist_base_paths, set) else None

                    if self.isvalid_policy == "clean":
                        filter_isvalid_paths = []
                        if required_base is not None:
                            missing_base = sorted([c for c in required_base if c not in dataset_set])
                            missing_isvalid = sorted([f"{c}_isvalid" for c in required_base if f"{c}_isvalid" not in dataset_set])
                            if missing_base or missing_isvalid:
                                self.logger.warning(
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
                    if self.isvalid_policy == "raw":
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
                        self.data = pd.DataFrame(merged_cols, copy=False)
                        
                        self.keys = list(self.data.columns)

                        if row_mask is not None and self.isvalid_policy == "clean":
                            self.logger.warning("Filtering Invalid Data from HDF5 Output")
                            self.logger.warning(
                                "DataSet Shape: \n\t Before filtering -> ({}, {})\n\t  After filtering -> ({}, {})".format(
                                    filter_rows_before if filter_rows_before is not None else self.data.shape[0],
                                    self.data.shape[1],
                                    filter_rows_after if filter_rows_after is not None else self.data.shape[0],
                                    self.data.shape[1],
                                )
                            )

                        self._apply_dataset_transform(stage="hdf5")
                                
                        # Emit a pretty summary BEFORE returning
                        summary_name = self._summary_name()
                        source_fp = self.fingerprint()
                        summary_msg = None
                        if self.cache is not None:
                            summary_msg = self.cache.get_summary(source_fp)
                        if summary_msg is None:
                            summary_msg = dataframe_summary(self.data, name=summary_name)
                            if self.cache is not None:
                                self.cache.put_summary(source_fp, summary_msg)
                        self._emit_summary_text(summary_msg)

                        return  # IMPORTANT: stop here; avoid falling through to single-dataset path
                    else:
                        # Not mergeable → print tree for diagnostics and raise a hard error
                        try:
                            print_hdf5_tree_ascii(group, root_name=self.group, logger=self.logger)
                        except Exception:
                            pass
                        raise ValueError(
                            "HDF5 group '{grp}' is invalid for merging: datasets have different row counts. "
                            "Please fix the input or choose a different dataset/group. Details: {details}".format(
                                grp=self.group,
                                details=shapes,
                            )
                        )
                else:
                    path, arr = _pick_dataset(f1)
    
    def apply_is_valid_policy(self, kkeys): 
        if self.data is None:
            self.load(force=False)

        source_cols = list(self.keys or [])
        source_col_set = set(source_cols)
        all_isvalid_cols = [col for col in source_cols if isinstance(col, str) and col.endswith("_isvalid")]

        isvalids = [
            kk for kk in kkeys
            if isinstance(kk, str) and kk.endswith("_isvalid") and kk[:-8] in source_col_set
        ]

        # Fallback: if mapping by kkeys fails, still use loaded *_isvalid columns.
        if not isvalids:
            isvalids = all_isvalid_cols[:]

        required_base = self._whitelist_base_paths if isinstance(self._whitelist_base_paths, set) else None
        if required_base is not None:
            missing_base = sorted([c for c in required_base if c not in source_col_set])
            missing_isvalid = sorted([f"{c}_isvalid" for c in required_base if f"{c}_isvalid" not in source_col_set])
            if missing_base or missing_isvalid:
                self.logger.warning(
                    "Skip is_valid policy: not all whitelist columns have companion _isvalid columns. "
                    "missing_base={}, missing_isvalid={}".format(missing_base, missing_isvalid)
                )
                self.keys = list(self.data.columns)
                return

        if self.isvalid_policy == "raw":
            self.logger.warning(
                "isvalid_policy=raw -> skip is_valid filtering and keep {} is_valid columns".format(
                    len(all_isvalid_cols)
                )
            )
            self.keys = list(self.data.columns)
            return

        if isvalids:
            self.logger.warning("Filtering Invalid Data from HDF5 Output")
            sps = self.data.shape
            mask = self.data[isvalids].all(axis=1)
            self.data = self.data[mask]
            self.logger.warning(
                "DataSet Shape: \n\t Before filtering -> {}\n\t  After filtering -> {}".format(
                    sps, self.data.shape
                )
            )

        if all_isvalid_cols:
            self.data = self.data.drop(columns=all_isvalid_cols, errors="ignore")
        self.keys = list(self.data.columns)
                
    def rename_columns(self, vdict):
        if self.data is None:
            self.load(force=False)
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

    def human_bytes(num_bytes: int) -> str:
        return dataframe_summary_human_bytes(num_bytes)

    nrows, ncols = df.shape
    cols = list(df.columns)
    show_cols = cols[:]
    try:
        # Fast estimate (deep=False) to avoid expensive scans on large object columns.
        mem_bytes = int(df.memory_usage(index=True, deep=False).sum())
    except Exception:
        mem_bytes = 0

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
    head_lines.append(f"\t DataFrame shape\t-> {nrows}\t rows × {ncols} \tcols")
    head_lines.append(f"\t Approx memory usage\t-> {human_bytes(mem_bytes)}")
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


def dataframe_summary_human_bytes(num_bytes: int) -> str:
    size = float(max(0, num_bytes))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
                
                
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

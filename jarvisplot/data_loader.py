#!/usr/bin/env python3 

from __future__ import annotations
from pathlib import Path
from typing import Optional, Any, Dict, List
import gc
import os
import pandas as pd 
import numpy as np
from .memtrace import memtrace_checkpoint, memtrace_enabled, memtrace_object_inventory
from .utils.dataframes import polars_to_pandas
from .utils.pathing import resolve_project_path
from . import data_loader_hdf5 as hdf5
from . import data_loader_summary as summary
from . import data_loader_runtime as runtime

try:
    import polars as pl
except Exception:
    pl = None

JP_ROW_IDX = "__jp_row_idx__"

class DataSet():
    def __init__(self):
        self._file: Optional[str]   = None
        self.path:  Optional[str]   = None 
        self.rootpath: Optional[str] = None
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
        self.rootpath = str(rootpath) if rootpath is not None else None
        raw_path = str(dtinfo['path'])
        resolved = resolve_project_path(raw_path, base_dir=rootpath or ".")
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
        elif self.type == "parquet":
            if pl is not None:
                try:
                    scan = pl.scan_parquet(self.path)
                    self.keys = list(scan.collect_schema().names())
                    if self.logger:
                        self.logger.debug(
                            f"Dataset '{self.name}' registered in lazy mode (parquet columns={len(self.keys)})."
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
        elif self.type == "parquet":
            self.load_parquet()
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
            schema = hdf5.polars_schema_names(self._full_lazy_frame)
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
                pulled = polars_to_pandas(lf, logger=self.logger, stage=f"dataset:{self.name}.lookup")
                if isinstance(pulled, pd.DataFrame) and row_key in pulled.columns:
                    pulled = pulled.drop_duplicates(subset=[row_key], keep="last").set_index(row_key).reindex(order)
                    pulled.index = np.arange(int(pulled.shape[0]))
                    for col in miss_avail:
                        out[col] = pulled[col].to_numpy(copy=False)

        return out

    def _summary_name(self) -> str:
        if self.type == "hdf5":
            return f" HDF5 loaded!\n\t name  -> {self.name}\n\t group -> {self.group}\n\t path  -> {self.path}"
        if self.type == "parquet":
            return f" Parquet loaded!\n\t name  -> {self.name}\n\t path  -> {self.path}"
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
                    msg = hdf5.materialized_summary(
                        self,
                        self._materialized_manifest,
                        stats=runtime.materialized_numeric_bounds(self),
                    )
                else:
                    msg = summary.dataframe_summary(self.data, name=self._summary_name())
            except Exception:
                if isinstance(self._materialized_manifest, dict):
                    msg = hdf5.materialized_summary(
                        self,
                        self._materialized_manifest,
                        stats=runtime.materialized_numeric_bounds(self),
                    )
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
            return
            
        p = Path(value).expanduser().resolve()
        self._file  = str(p)
        self.path   = os.path.abspath(p)
        self.base   = os.path.basename(p)
        
    @type.setter 
    def type(self, value: Optional[str]) -> None: 
        if value is None: 
            self._type = None
            return
            
        self._type = str(value).lower()
        if self.logger:
            self.logger.debug("Dataset -> {} is assigned as \n\t-> {}\ttype".format(self.base, self.type))

    def load_csv(self):
        if self.type == "csv":
            if self.logger:
                self.logger.debug("Loading CSV from {}".format(self.path))

            self.data = pd.read_csv(self.path)
            self.keys = list(self.data.columns)
            runtime.apply_dataset_transform(self, stage="csv")

            # Emit the same pretty summary used for HDF5 datasets
            summary_name = self._summary_name()
            summary_msg = None
            source_fp = self.fingerprint()
            if self.cache is not None:
                summary_msg = self.cache.get_summary(source_fp)

            if summary_msg is None:
                try:
                    summary_msg = summary.dataframe_summary(self.data, name=summary_name)
                except Exception:
                    # Fallback minimal summary if something goes wrong
                    summary_msg = f"CSV loaded  {summary_name}\nDataFrame shape: {self.data.shape}"
                if self.cache is not None:
                    self.cache.put_summary(source_fp, summary_msg)

            self._emit_summary_text(summary_msg)

    def load_hdf5(self):
        return runtime.load_hdf5(self)

    def load_parquet(self):
        return runtime.load_parquet(self)

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
            self.logger.info(
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

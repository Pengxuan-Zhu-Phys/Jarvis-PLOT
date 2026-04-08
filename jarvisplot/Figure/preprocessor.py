#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
import gc
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import hashlib
import json

import numpy as np
import pandas as pd

try:
    import polars as pl
except Exception:
    pl = None

from ..data_loader import JP_ROW_IDX
from .profile_runtime import _preprofiling
from . import preprocessor_runtime as runtime
from ..memtrace import memtrace_checkpoint, memtrace_object_inventory
from ..utils.dataframes import polars_to_pandas


class DataPreprocessor:
    def __init__(
        self,
        context,
        cache=None,
        dataset_registry: Optional[Dict[str, Any]] = None,
        logger=None,
        base_dir: Optional[Any] = None,
    ):
        self.context = context
        self.cache = cache
        self.dataset_registry = dataset_registry or {}
        self.logger = logger
        self.base_dir = base_dir
        self._emitted_sources = set()
        self._preprofile_alias_meta: Dict[str, Dict[str, Any]] = {}
        self._named_share_signatures: Dict[str, str] = {}
        self._named_share_sources: Dict[str, Any] = {}

    def _debug(self, msg: str) -> None:
        if self.logger:
            try:
                self.logger.debug(msg)
            except Exception:
                pass

    def _warn(self, msg: str) -> None:
        if self.logger:
            try:
                self.logger.warning(msg)
            except Exception:
                pass

    def _info(self, msg: str) -> None:
        if self.logger:
            try:
                self.logger.info(msg)
            except Exception:
                pass

    @staticmethod
    def _canonical_json(payload: Any) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)

    @classmethod
    def _stable_hash(cls, payload: Any) -> str:
        return hashlib.sha1(cls._canonical_json(payload).encode("utf-8")).hexdigest()

    @staticmethod
    def _expr_symbols(expr: Any) -> List[str]:
        if expr is None or isinstance(expr, (int, float, bool)):
            return []
        toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(expr)))
        ignore = {
            "np",
            "math",
            "True",
            "False",
            "None",
            "and",
            "or",
            "not",
            "in",
            "if",
            "else",
            "for",
            "lambda",
            "abs",
            "min",
            "max",
            "sum",
            "len",
            "int",
            "float",
            "str",
            "bool",
            "round",
            "sin",
            "cos",
            "tan",
            "exp",
            "log",
            "sqrt",
            "pi",
            "e",
        }
        return sorted(t for t in toks if t not in ignore)

    def _profile_cfg_input_columns(self, cfg: Any) -> List[str]:
        out: set[str] = set()
        if not isinstance(cfg, Mapping):
            return []
        coors = cfg.get("coordinates", {})
        if not isinstance(coors, Mapping):
            return []
        for axis_key, axis_cfg in coors.items():
            axis = str(axis_key).strip()
            if isinstance(axis_cfg, Mapping):
                out.update(self._expr_symbols(axis_cfg.get("expr")))
                name = axis_cfg.get("name")
                if isinstance(name, str) and name.strip():
                    out.add(name.strip())
                elif axis in {"x", "y", "z", "left", "right", "bottom"}:
                    out.add(axis)
            elif isinstance(axis_cfg, str):
                out.update(self._expr_symbols(axis_cfg))
                if axis in {"x", "y", "z", "left", "right", "bottom"}:
                    out.add(axis)
        return sorted(out)

    def _profile_cfg_output_columns(self, cfg: Any) -> List[str]:
        out: set[str] = set()
        if not isinstance(cfg, Mapping):
            return []
        coors = cfg.get("coordinates", {})
        if not isinstance(coors, Mapping):
            return []
        for axis_key, axis_cfg in coors.items():
            axis = str(axis_key).strip()
            if isinstance(axis_cfg, Mapping):
                name = axis_cfg.get("name")
                if isinstance(name, str) and name.strip():
                    out.add(name.strip())
                elif axis in {"x", "y", "z", "left", "right", "bottom"}:
                    out.add(axis)
        return sorted(out)

    def _transform_input_columns(self, transform: Any) -> List[str]:
        out: set[str] = set()
        if not isinstance(transform, list):
            return []
        for step in transform:
            if not isinstance(step, Mapping):
                continue
            if "filter" in step:
                out.update(self._expr_symbols(step.get("filter")))
            if "sortby" in step:
                out.update(self._expr_symbols(step.get("sortby")))
            if "add_column" in step:
                add_cfg = step.get("add_column", {})
                if isinstance(add_cfg, Mapping):
                    out.update(self._expr_symbols(add_cfg.get("expr")))
            if "profile" in step:
                out.update(self._profile_cfg_input_columns(step.get("profile", {})))
            if "grid_profile" in step:
                out.update(self._profile_cfg_input_columns(step.get("grid_profile", {})))
        return sorted(out)

    def _transform_output_columns(self, transform: Any) -> List[str]:
        out: set[str] = set()
        if not isinstance(transform, list):
            return []
        for step in transform:
            if not isinstance(step, Mapping):
                continue
            if "add_column" in step:
                add_cfg = step.get("add_column", {})
                if isinstance(add_cfg, Mapping):
                    name = add_cfg.get("name")
                    if isinstance(name, str) and name.strip():
                        out.add(name.strip())
            if "profile" in step:
                out.update(self._profile_cfg_output_columns(step.get("profile", {})))
            if "grid_profile" in step:
                out.update(self._profile_cfg_output_columns(step.get("grid_profile", {})))
                out.update(
                    {
                        "__grid_ix__",
                        "__grid_iy__",
                        "__grid_bin__",
                        "__grid_xmin__",
                        "__grid_xmax__",
                        "__grid_ymin__",
                        "__grid_ymax__",
                        "__grid_xscale__",
                        "__grid_yscale__",
                        "__grid_objective__",
                        "__grid_empty_value__",
                    }
                )
        return sorted(out)

    @staticmethod
    def _transform_requests_csv_export(transform: Any) -> bool:
        if not isinstance(transform, list):
            return False
        for step in transform:
            if not isinstance(step, Mapping):
                continue
            if any(key in step for key in ("tocsv", "to_csv")):
                return True
        return False

    def _layer_requests_csv_export(self, layer: Any) -> bool:
        if not isinstance(layer, Mapping):
            return False
        entries = layer.get("data", [])
        if not isinstance(entries, list):
            return False
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            if self._transform_requests_csv_export(entry.get("transform")):
                return True
        return False

    def _layer_expr_columns(self, obj: Any) -> List[str]:
        out: set[str] = set()

        def _walk(node: Any) -> None:
            if isinstance(node, Mapping):
                for kk, vv in node.items():
                    if str(kk).strip().lower() == "expr":
                        out.update(self._expr_symbols(vv))
                    else:
                        _walk(vv)
            elif isinstance(node, (list, tuple)):
                for item in node:
                    _walk(item)

        _walk(obj)
        return sorted(out)

    def layer_demand_columns(self, layer: Mapping[str, Any]) -> List[str]:
        out: set[str] = set()
        if not isinstance(layer, Mapping):
            return []
        out.update(self._layer_expr_columns(layer.get("coordinates", {})))
        out.update(self._layer_expr_columns(layer.get("style", {})))
        return sorted(out)

    @staticmethod
    def _projection_list(columns: Optional[Sequence[str]]) -> Optional[List[str]]:
        if not columns:
            return None
        out = sorted({str(c).strip() for c in columns if str(c).strip()})
        return out if out else None

    def _runtime_projection(self, transform: Any, demand_columns: Optional[Sequence[str]]) -> Optional[List[str]]:
        out: set[str] = {JP_ROW_IDX}
        out.update(self._transform_input_columns(transform))
        out.update(self._transform_output_columns(transform))
        if demand_columns:
            out.update(str(c).strip() for c in demand_columns if str(c).strip())
        return self._projection_list(sorted(out))

    def _runtime_cache_columns(self, transform: Any, demand_columns: Optional[Sequence[str]]) -> Optional[List[str]]:
        out: set[str] = {JP_ROW_IDX}
        out.update(self._transform_output_columns(transform))
        if demand_columns:
            out.update(str(c).strip() for c in demand_columns if str(c).strip())
        return self._projection_list(sorted(out))

    def _preprofile_base_projection(self, prefix_transform: Any, profile_cfg: Any) -> Optional[List[str]]:
        out: set[str] = {JP_ROW_IDX}
        out.update(self._transform_input_columns(prefix_transform))
        out.update(self._transform_output_columns(prefix_transform))
        out.update(self._profile_cfg_input_columns(profile_cfg))
        return self._projection_list(sorted(out))

    def _select_columns(self, df: Any, columns: Optional[Sequence[str]]):
        proj = self._projection_list(columns)
        if not proj:
            return df
        if isinstance(df, pd.DataFrame):
            keep = [c for c in proj if c in df.columns]
            if not keep or len(keep) == int(df.shape[1]):
                return df
            return df.loc[:, keep].copy(deep=False)
        if pl is not None and isinstance(df, pl.DataFrame):
            keep = [c for c in proj if c in df.columns]
            if not keep or len(keep) == int(df.width):
                return df
            return df.select(keep)
        if pl is not None and isinstance(df, pl.LazyFrame):
            keep = [c for c in proj if c in self._polars_schema_names(df)]
            if not keep:
                return df
            return df.select(keep)
        return df

    @staticmethod
    def _single_layer_source(layer: Mapping[str, Any]) -> Any:
        entries = layer.get("data", [])
        if not isinstance(entries, list) or len(entries) != 1:
            return None
        entry = entries[0]
        if not isinstance(entry, Mapping):
            return None
        return deepcopy(entry.get("source"))

    def _resolve_base_dataset_source(self, source: Any):
        current = source
        seen: set[str] = set()
        if isinstance(current, (list, tuple)) and len(current) == 1 and isinstance(current[0], str):
            current = current[0]
        while isinstance(current, str):
            if current in seen:
                return None
            seen.add(current)
            if current in self.dataset_registry:
                return current
            if current in self._preprofile_alias_meta:
                meta = self._preprofile_alias_meta.get(current, {})
                loader = meta.get("loader", {}) if isinstance(meta, Mapping) else {}
                current = loader.get("source") if isinstance(loader, Mapping) else None
                continue
            if current in self._named_share_sources:
                current = self._named_share_sources.get(current)
                continue
            return None
        return None

    def _enrich_for_demand(self, df: Any, source: Any, demand_columns: Optional[Sequence[str]]):
        if not isinstance(df, pd.DataFrame):
            return df
        if JP_ROW_IDX not in df.columns:
            return df
        demand = self._projection_list(demand_columns)
        if not demand:
            return df
        missing = [c for c in demand if c not in df.columns]
        if not missing:
            return df
        base_source = self._resolve_base_dataset_source(source)
        if not isinstance(base_source, str):
            return df
        dataset = self.dataset_registry.get(base_source)
        if dataset is None or not hasattr(dataset, "fetch_rows_columns"):
            return df
        try:
            row_idx = pd.to_numeric(df[JP_ROW_IDX], errors="coerce")
        except Exception:
            return df
        valid_mask = row_idx.notna().to_numpy(copy=False)
        if not bool(np.any(valid_mask)):
            return df
        try:
            row_ids = row_idx[valid_mask].astype(np.int64).to_numpy(copy=False)
        except Exception:
            return df
        extra = dataset.fetch_rows_columns(row_ids, missing, row_key=JP_ROW_IDX)
        if not isinstance(extra, pd.DataFrame) or extra.empty:
            return df
        out = self._clone_df(df)
        valid_index = out.index[valid_mask]
        for col in [c for c in extra.columns if c in missing]:
            values = extra[col].to_numpy(copy=False)
            if int(values.shape[0]) != int(valid_index.shape[0]):
                continue
            if bool(np.all(valid_mask)):
                out[col] = values
            else:
                out.loc[valid_index, col] = values
        return out

    @staticmethod
    def _clone_df(df: pd.DataFrame) -> pd.DataFrame:
        try:
            return df.copy(deep=False)
        except Exception:
            return deepcopy(df)

    @staticmethod
    def _should_collect_dataframe(df: Any) -> bool:
        if not isinstance(df, pd.DataFrame):
            return False
        try:
            return int(df.shape[0]) >= 100_000
        except Exception:
            return True

    @staticmethod
    def _safe_nrows(df: Any) -> Optional[int]:
        if isinstance(df, pd.DataFrame):
            return int(df.shape[0])
        if pl is not None and isinstance(df, pl.DataFrame):
            return int(df.height)
        return None

    @staticmethod
    def _is_polars_frame(df: Any) -> bool:
        if pl is None:
            return False
        return isinstance(df, (pl.DataFrame, pl.LazyFrame))

    @staticmethod
    def _polars_schema_names(lf) -> List[str]:
        if pl is None:
            return []
        try:
            if isinstance(lf, pl.DataFrame):
                return list(lf.columns)
            if isinstance(lf, pl.LazyFrame):
                return list(lf.collect_schema().names())
        except Exception:
            return []
        return []

    def _polars_to_pandas(self, df: Any, reason: str = "runtime"):
        if pl is None:
            return df
        if isinstance(df, (pl.LazyFrame, pl.DataFrame)):
            if isinstance(df, pl.LazyFrame):
                self._info(f"Collecting polars LazyFrame -> {reason}")
            else:
                self._info(f"Converting polars DataFrame to pandas -> {reason}")
            return polars_to_pandas(df, logger=self.logger, stage=f"pipeline:{reason}")
        return df

    def ensure_pandas(self, df: Any, reason: str = "runtime"):
        if pl is None:
            return df
        if isinstance(df, (pl.LazyFrame, pl.DataFrame)):
            return self._polars_to_pandas(df, reason=reason)
        if isinstance(df, dict):
            return {kk: self.ensure_pandas(vv, reason=reason) for kk, vv in df.items()}
        return df

    def _runtime_source_label(self, source: Any) -> str:
        if isinstance(source, str):
            return source
        if isinstance(source, (list, tuple)):
            return "[" + ", ".join(str(x) for x in source) + "]"
        return str(source)

    def _dataset_transform(self, source: Any):
        if not isinstance(source, str):
            return None
        dts = self.dataset_registry.get(source)
        if dts is None:
            return None
        tf = getattr(dts, "transform", None)
        if isinstance(tf, list):
            return tf
        return None

    def _effective_transform(self, source: Any, transform: Any):
        """Drop duplicated transform prefix when dataset already applies the same steps."""
        if transform is None or not isinstance(transform, list):
            return transform
        base = self._dataset_transform(source)
        if not isinstance(base, list) or len(base) == 0:
            return transform

        nmatch = 0
        nmax = min(len(base), len(transform))
        while nmatch < nmax and transform[nmatch] == base[nmatch]:
            nmatch += 1
        if nmatch <= 0:
            return transform

        trimmed = transform[nmatch:]
        self._debug(
            "Pipeline transform prefix dedup:\n\t source \t-> {}\n\t stripped \t-> {}\n\t remained \t-> {}".format(
                self._runtime_source_label(source),
                nmatch,
                len(trimmed),
            )
        )
        return trimmed if trimmed else None

    def _source_token(self, source: Any, combine: str = "concat") -> Dict[str, Any]:
        if isinstance(source, str):
            dts = self.dataset_registry.get(source)
            if dts is not None and hasattr(dts, "fingerprint"):
                fp = dts.fingerprint(self.cache)
                return {"source": source, "fingerprint": fp}
            named_sig = self._named_share_signatures.get(source)
            if named_sig:
                return {
                    "source": source,
                    "fingerprint": {
                        "kind": "named_share",
                        "signature": str(named_sig),
                    },
                }
            return {"source": source, "fingerprint": {"kind": "shared_only"}}

        if isinstance(source, (list, tuple)):
            items = []
            for ss in source:
                items.append(self._source_token(str(ss), combine=combine))
            return {"source_list": items, "combine": str(combine)}

        return {"source": str(source), "fingerprint": {"kind": "unknown"}}

    def _pipeline_key(
        self,
        source: Any,
        transform: Any,
        combine: str = "concat",
        mode: str = "runtime",
        projection: Optional[Sequence[str]] = None,
    ) -> str:
        eff_transform = self._effective_transform(source, transform)
        payload = {
            "kind": "pipeline",
            "algo": "pregrid-v6",
            "source": self._source_token(source, combine=combine),
            "transform": eff_transform,
            "combine": str(combine),
            "mode": str(mode),
            "projection": self._projection_list(projection),
        }
        if self.cache is not None:
            return self.cache.cache_key(payload)
        return self._stable_hash(payload)

    def _demand_payload(
        self,
        source: Any,
        transform: Any,
        combine: str = "concat",
        mode: str = "runtime",
        projection: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        eff_transform = self._effective_transform(source, transform)
        return {
            "schema": "jp-demand-v7",
            "source": self._source_token(source, combine=combine),
            "transform": eff_transform,
            "combine": str(combine),
            "mode": str(mode),
            "projection": self._projection_list(projection),
        }

    def _demand_fingerprint(
        self,
        source: Any,
        transform: Any,
        combine: str = "concat",
        mode: str = "runtime",
        projection: Optional[Sequence[str]] = None,
    ) -> str:
        return self._stable_hash(
            self._demand_payload(source, transform, combine=combine, mode=mode, projection=projection)
        )

    def _layer_signature(self, layer: Mapping[str, Any]) -> str:
        ds = layer.get("data", [])
        tokens = []
        for item in ds:
            if isinstance(item, Mapping):
                src = item.get("source")
                tf = self._effective_transform(src, item.get("transform"))
                tokens.append(
                    {
                        "source": self._source_token(src),
                        "transform": tf,
                    }
                )
        payload = {
            "name": layer.get("name"),
            "axes": layer.get("axes"),
            "method": layer.get("method"),
            "data": tokens,
            "combine": layer.get("combine", "concat"),
        }
        return self._stable_hash(payload)

    def _runtime_profile_tokens(self, transform: Any) -> List[Dict[str, Any]]:
        tokens: List[Dict[str, Any]] = []
        if not isinstance(transform, list):
            return tokens
        for step in transform:
            if not isinstance(step, Mapping):
                continue
            if "profile" in step:
                cfg = step.get("profile", {})
                if isinstance(cfg, Mapping):
                    cfg = deepcopy(cfg)
                else:
                    cfg = {"value": cfg}
                tokens.append({"kind": "profile", "cfg": cfg})
            elif "grid_profile" in step:
                cfg = step.get("grid_profile", {})
                if isinstance(cfg, Mapping):
                    cfg = deepcopy(cfg)
                else:
                    cfg = {}
                cfg.setdefault("method", "grid")
                tokens.append({"kind": "grid_profile", "cfg": cfg})
        return tokens

    def _runtime_profile_signature(self, transform: Any) -> Optional[str]:
        tokens = self._runtime_profile_tokens(transform)
        if not tokens:
            return None
        return self._stable_hash(tokens)

    @staticmethod
    def _preprofile_alias_name(key: str) -> str:
        return f"__jp_preprofile_{str(key)[:16]}"

    def _preprofile_cache_file(self, key: str) -> str:
        if self.cache is None:
            return "<memory-only>"
        try:
            return str((self.cache.data_dir / f"{key}.pkl").resolve())
        except Exception:
            return "<memory-only>"

    @staticmethod
    def _preprofile_loader_spec(task: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "source": deepcopy(task.get("source")),
            "combine": str(task.get("combine", "concat")),
            "pre_transform": deepcopy(task.get("pre_transform")),
            "prefix_transform": deepcopy(task.get("prefix_transform")),
            "profile_cfg": deepcopy(task.get("profile_cfg", {})),
            "base_projection": deepcopy(task.get("base_projection")),
            "cache_flag": bool(task.get("cache_flag", True)),
        }

    def _register_preprofile_alias(
        self,
        alias: str,
        task: Mapping[str, Any],
        origin: str,
        rows: Optional[int] = None,
        cache_ready: bool = False,
    ) -> None:
        key = str(task.get("pre_key", ""))
        cache_file = self._preprofile_cache_file(key) if cache_ready else "<lazy-recompute>"
        self._preprofile_alias_meta[str(alias)] = {
            "pre_key": key,
            "origin": str(origin),
            "cache_file": cache_file,
            "source": deepcopy(task.get("source")),
            "rows": rows,
            "cache_ready": bool(cache_ready),
            "loader": self._preprofile_loader_spec(task),
        }
        self.context.register(
            str(alias),
            lambda _shared, _alias=str(alias): self._load_preprofile_alias(_alias),
        )

    def _load_preprofile_alias(self, alias: str):
        meta = self._preprofile_alias_meta.get(str(alias))
        if not isinstance(meta, Mapping):
            self._warn(f"Lazy preprofile alias metadata missing -> {alias}")
            return None

        key = str(meta.get("pre_key", ""))
        if bool(meta.get("cache_ready")) and self.cache is not None and key:
            cached = self.cache.get_dataframe(key)
            if cached is not None:
                if meta.get("rows") is None:
                    meta["rows"] = self._safe_nrows(cached)
                self._debug(f"Lazy preprofile alias cache LOAD -> {alias}")
                return cached

        loader = meta.get("loader")
        if not isinstance(loader, Mapping):
            self._warn(f"Lazy preprofile alias loader missing -> {alias}")
            return None

        self._info(
            "Lazy preprofile alias BUILD:\n\t alias \t-> {},\n\t source \t-> {},\n\t key \t-> {}".format(
                alias,
                self._runtime_source_label(loader.get("source")),
                key[:16] if key else "<none>",
            )
        )

        base_df, _, _ = self.run_pipeline(
            source=loader.get("source"),
            transform=loader.get("prefix_transform"),
            combine=loader.get("combine", "concat"),
            use_cache=bool(loader.get("cache_flag", True)),
            mode="preprofile-base",
            projection=loader.get("base_projection"),
        )
        if base_df is None:
            return None

        try:
            out = _preprofiling(base_df, loader.get("profile_cfg", {}), self.logger)
        finally:
            base_df = None
            gc.collect()

        if out is None:
            return None

        out = self._select_columns(out, loader.get("base_projection"))
        rows = self._safe_nrows(out)
        if bool(loader.get("cache_flag", True)) and self.cache is not None and isinstance(out, pd.DataFrame) and key:
            pre_demand_fp = self._demand_fingerprint(
                loader.get("source"),
                loader.get("pre_transform"),
                combine=loader.get("combine", "concat"),
                mode="preprofile",
                projection=loader.get("base_projection"),
            )
            self.cache.put_dataframe(
                key,
                out,
                meta={
                    "source": self._source_token(loader.get("source"), combine=loader.get("combine", "concat")),
                    "combine": loader.get("combine", "concat"),
                    "transform": loader.get("pre_transform"),
                    "mode": "preprofile",
                    "demand_fingerprint": pre_demand_fp,
                    "projection": loader.get("base_projection"),
                    "rows": rows,
                },
            )
            meta["cache_ready"] = True
            meta["cache_file"] = self._preprofile_cache_file(key)

        meta["rows"] = rows
        self._preprofile_alias_meta[str(alias)] = dict(meta)
        return out

    def _is_runtime_profile_cache_compatible(
        self,
        transform: Any,
        meta: Optional[Mapping[str, Any]],
        mode: str,
    ) -> bool:
        if str(mode).lower() != "runtime":
            return True
        current_sig = self._runtime_profile_signature(transform)
        if current_sig is None:
            return True
        if not isinstance(meta, Mapping):
            return False
        cached_sig = meta.get("runtime_profile_signature")
        if not cached_sig:
            return False
        if str(cached_sig) != str(current_sig):
            return False
        current_transform_sig = self._stable_hash(transform)
        cached_transform_sig = meta.get("runtime_transform_signature")
        if not cached_transform_sig:
            return False
        return str(cached_transform_sig) == str(current_transform_sig)

    def _is_dataframe_cache_compatible(
        self,
        source: Any,
        transform: Any,
        combine: str,
        mode: str,
        key: str,
        meta: Optional[Mapping[str, Any]],
        projection: Optional[Sequence[str]] = None,
    ) -> Tuple[bool, str]:
        if not isinstance(meta, Mapping):
            return False, "meta-missing"

        if self._transform_requests_csv_export(transform):
            return False, "csv-export-step"

        expected_demand = self._demand_fingerprint(
            source,
            transform,
            combine=combine,
            mode=mode,
            projection=projection,
        )
        cached_demand = meta.get("demand_fingerprint")
        if not cached_demand:
            return False, "demand-fingerprint-missing"
        if str(cached_demand) != str(expected_demand):
            return False, "demand-fingerprint-mismatch"

        if not self._is_runtime_profile_cache_compatible(transform, meta, mode):
            return False, "runtime-profile-signature-mismatch"

        if self.cache is not None and hasattr(self.cache, "is_dataframe_meta_consistent"):
            try:
                ok_file = bool(self.cache.is_dataframe_meta_consistent(key, dict(meta)))
            except Exception:
                ok_file = False
            if not ok_file:
                return False, "cache-file-fingerprint-mismatch"

        return True, "ok"

    def load_named_layer(
        self,
        name: Optional[str],
        layer: Mapping[str, Any],
        demand_columns: Optional[Sequence[str]] = None,
    ):
        if not name or self.cache is None:
            return None
        if self._layer_requests_csv_export(layer):
            self._debug(f"Named share_data cache bypassed for CSV export layer -> {name}")
            return None
        signature = self._layer_signature(layer)
        return self.load_named_layer_by_signature(name, signature, demand_columns=demand_columns)

    def load_named_layer_by_signature(
        self,
        name: Optional[str],
        signature: str,
        demand_columns: Optional[Sequence[str]] = None,
    ):
        if not name or self.cache is None:
            return None
        self._named_share_signatures[str(name)] = str(signature)
        named = self.cache.get_named(name, signature)
        if isinstance(named, pd.DataFrame):
            self._warn(
                "Named share_data cache HIT:\n\t name \t\t-> {}, \n\t signature \t-> {}, \n\t rows \t\t-> {}.".format(
                    name,
                    signature,
                    self._safe_nrows(named) if self._safe_nrows(named) is not None else "NA",
                )
            )
        return self._enrich_for_demand(named, str(name), demand_columns)

    def register_named_layer(self, name: Optional[str], layer: Mapping[str, Any]) -> bool:
        if not name or self.cache is None:
            return False
        if self._layer_requests_csv_export(layer):
            self._debug(f"Named share_data registration bypassed for CSV export layer -> {name}")
            return False
        signature = self._layer_signature(layer)
        self._named_share_signatures[str(name)] = str(signature)
        self._named_share_sources[str(name)] = self._single_layer_source(layer)
        self.context.register(
            str(name),
            lambda _shared, _name=str(name), _sig=str(signature): self.load_named_layer_by_signature(_name, _sig),
        )
        return True

    def should_release_between_uses(self, source: Any) -> bool:
        if not isinstance(source, str):
            return False
        if source in self.dataset_registry:
            return False
        if source in self._preprofile_alias_meta:
            return True
        if source in self._named_share_signatures:
            return True
        return False

    def persist_named_layer(
        self,
        name: Optional[str],
        layer: Mapping[str, Any],
        data,
        cache_ref: Optional[str] = None,
    ) -> None:
        if not name or self.cache is None:
            return
        if not isinstance(data, pd.DataFrame):
            return
        if self._layer_requests_csv_export(layer):
            self._debug(f"Skipping named share_data persistence for CSV export layer -> {name}")
            return
        signature = self._layer_signature(layer)
        self._named_share_signatures[str(name)] = str(signature)
        self._named_share_sources[str(name)] = self._single_layer_source(layer)
        ref_key = str(cache_ref or "").strip()
        if ref_key:
            try:
                self.cache.put_named_reference(name, signature, ref_key)
                self._debug(f"Stored named dataset '{name}' as data-cache reference -> {ref_key[:16]}")
                return
            except Exception as e:
                self._debug(f"Named dataset reference fallback to dataframe store for '{name}': {e}")
        self.cache.put_named(name, signature, data)
        self._debug(f"Stored named dataset '{name}' into cache.")

    def apply_transforms(self, df, transform: Optional[Sequence[Mapping[str, Any]]]):
        return runtime.apply_transforms(self, df, transform)

    def apply_runtime_transforms(
        self,
        df,
        transform: Optional[Sequence[Mapping[str, Any]]],
        source_label: Optional[str] = None,
    ):
        return runtime.apply_runtime_transforms(self, df, transform, source_label=source_label)

    def run_pipeline(
        self,
        source: Any,
        transform: Optional[Sequence[Mapping[str, Any]]],
        combine: str = "concat",
        use_cache: bool = True,
        mode: str = "runtime",
        demand_columns: Optional[Sequence[str]] = None,
        projection: Optional[Sequence[str]] = None,
    ) -> Tuple[Optional[pd.DataFrame], str, bool]:
        return runtime.run_pipeline(
            self,
            source,
            transform,
            combine=combine,
            use_cache=use_cache,
            mode=mode,
            demand_columns=demand_columns,
            projection=projection,
        )

    @staticmethod
    def _contains_profile(transform: Any) -> bool:
        if not isinstance(transform, list):
            return False
        for step in transform:
            if isinstance(step, Mapping) and "profile" in step:
                return True
        return False

    @staticmethod
    def _first_profile_index(transform: Any) -> int:
        if not isinstance(transform, list):
            return -1
        for idx, step in enumerate(transform):
            if isinstance(step, Mapping) and "profile" in step:
                return idx
        return -1

    @staticmethod
    def _preprofile_profile_cfg(profile_cfg: Any) -> Any:
        # Preprofile cache identity is intentionally coordinates-only:
        # runtime profile method/bin/objective changes must reuse same preprofile.
        if not isinstance(profile_cfg, Mapping):
            return {}
        slim: Dict[str, Any] = {}
        if "coordinates" in profile_cfg:
            slim["coordinates"] = deepcopy(profile_cfg["coordinates"])
        return slim

    def _split_prebuild_transform(self, transform: Any):
        if not isinstance(transform, list):
            return None, None
        idx = self._first_profile_index(transform)
        if idx < 0:
            return None, None

        pre_transform: List[Mapping[str, Any]] = []
        for step in transform[:idx]:
            if isinstance(step, Mapping):
                pre_transform.append(deepcopy(step))

        first = transform[idx]
        if not isinstance(first, Mapping) or "profile" not in first:
            return None, None

        profile_cfg: Any = first.get("profile", {})

        pre_transform.append(
            {"profile": self._preprofile_profile_cfg(profile_cfg)}
        )

        runtime_transform = deepcopy(transform[idx:])
        return pre_transform, runtime_transform

    def prebuild_profiles(self, config: Mapping[str, Any]) -> Dict[str, int]:
        """Prebuild preprofile pipelines in one traversal and rewrite source to alias.

        Strategy:
          1) collect unique preprofile tasks (unique by preprofile cache key)
          2) fast-path cache hits for each task
          3) group cache misses by identical input data (source + transform prefix)
          4) for each input group, build base dataframe once and fan out multiple preprofiles
        """
        figures = config.get("Figures", [])
        if not isinstance(figures, list):
            return {"tasks": 0, "hits": 0, "miss": 0}

        prepared: Dict[str, str] = {}
        hits = 0
        miss = 0
        tasks: Dict[str, Dict[str, Any]] = {}
        self._preprofile_alias_meta = {}

        for fig in figures:
            if not isinstance(fig, Mapping):
                continue
            if fig.get("enable", True) is False:
                continue
            layers = fig.get("layers", [])
            if not isinstance(layers, list):
                continue

            for layer in layers:
                if not isinstance(layer, Mapping):
                    continue
                entries = layer.get("data", [])
                if not isinstance(entries, list):
                    continue

                for ds in entries:
                    if not isinstance(ds, Mapping):
                        continue
                    transform = ds.get("transform")
                    pre_transform, runtime_transform = self._split_prebuild_transform(transform)
                    if pre_transform is None:
                        continue

                    src = ds.get("source")
                    cache_flag = bool(ds.get("cache", True))
                    combine = "concat"

                    prefix_transform = pre_transform[:-1]
                    profile_cfg = {}
                    if pre_transform and isinstance(pre_transform[-1], Mapping):
                        profile_cfg = deepcopy(pre_transform[-1].get("profile", {}))
                    base_projection = self._preprofile_base_projection(prefix_transform, profile_cfg)
                    key = self._pipeline_key(
                        src,
                        pre_transform,
                        combine=combine,
                        mode="preprofile",
                        projection=base_projection,
                    )
                    task = tasks.get(key)
                    if task is None:
                        task = {
                            "pre_key": key,
                            "source": src,
                            "combine": combine,
                            "pre_transform": pre_transform,
                            "prefix_transform": prefix_transform,
                            "profile_cfg": profile_cfg,
                            "base_projection": base_projection,
                            "cache_flag": cache_flag,
                            "targets": [],
                        }
                        tasks[key] = task
                    else:
                        task["cache_flag"] = bool(task["cache_flag"] or cache_flag)

                    fig_name = fig.get("name", "<noname>")
                    layer_name = layer.get("name", "<noname>")
                    task["targets"].append(
                        {
                            "ds": ds,
                            "runtime_transform": runtime_transform,
                            "figure_name": fig_name,
                            "layer_name": layer_name,
                        }
                    )

        # phase-1: preprofile cache fast-path
        pending: List[Dict[str, Any]] = []
        for key, task in tasks.items():
            alias = None
            if bool(task.get("cache_flag", True)) and self.cache is not None:
                meta = None
                try:
                    if hasattr(self.cache, "get_dataframe_meta"):
                        meta = self.cache.get_dataframe_meta(key)
                except Exception:
                    meta = None

                ok_cache, reason = self._is_dataframe_cache_compatible(
                    source=task.get("source"),
                    transform=task.get("pre_transform"),
                    combine=task.get("combine", "concat"),
                    mode="preprofile",
                    key=key,
                    meta=meta,
                    projection=task.get("base_projection"),
                )
                if ok_cache:
                    alias = self._preprofile_alias_name(key)
                    rows = meta.get("rows") if isinstance(meta, Mapping) else None
                    self._register_preprofile_alias(alias, task, origin="cache-hit", rows=rows, cache_ready=True)
                    prepared[key] = alias
                    runtime.emit_source_summary(self, task.get("source"))
                    logged_pairs = set()
                    for tgt in task.get("targets", []):
                        fig_name = str(tgt.get("figure_name", "<noname>"))
                        layer_name = str(tgt.get("layer_name", "<noname>"))
                        pair = (fig_name, layer_name)
                        if pair in logged_pairs:
                            continue
                        logged_pairs.add(pair)
                        self._info(
                            "Preprofile cache HIT:\n\t figure -> {},\n\t layer -> {},\n\t source -> {},\n\t key -> {}".format(
                                fig_name,
                                layer_name,
                                task.get("source"),
                                key[:16],
                            )
                        )
                    hits += 1
                elif reason != "meta-missing":
                    self._debug(f"Preprofile cache INVALID ({reason}) -> {key}")

            if alias is None:
                if bool(task.get("cache_flag", True)) and self.cache is not None:
                    pending.append(task)
                else:
                    alias = self._preprofile_alias_name(key)
                    self._register_preprofile_alias(alias, task, origin="lazy-recompute", rows=None, cache_ready=False)
                    prepared[key] = alias
                    miss += 1

        # phase-2: batch by identical input data
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for task in pending:
            base_key = self._pipeline_key(
                task.get("source"),
                task.get("prefix_transform"),
                combine=task.get("combine", "concat"),
                mode="preprofile-base",
            )
            groups.setdefault(base_key, []).append(task)

        for base_key, grp in groups.items():
            if not grp:
                continue

            sample = grp[0]
            base_df, _, _ = self.run_pipeline(
                source=sample.get("source"),
                transform=sample.get("prefix_transform"),
                combine=sample.get("combine", "concat"),
                use_cache=any(bool(t.get("cache_flag", True)) for t in grp),
                mode="preprofile-base",
                projection=sorted(
                    {
                        str(col).strip()
                        for task in grp
                        for col in (task.get("base_projection") or [])
                        if str(col).strip()
                    }
                ),
            )
            if base_df is None:
                continue

            for task in grp:
                # _preprofiling is implemented as read-mostly on input dataframe.
                # Reusing base_df here avoids cloning a large table for every
                # task in the same preprofile group.
                out = _preprofiling(base_df, task.get("profile_cfg", {}), self.logger)
                if out is None:
                    continue

                out = self._select_columns(out, task.get("base_projection"))
                key = task["pre_key"]
                rows = self._safe_nrows(out)
                cache_ready = (
                    bool(task.get("cache_flag", True))
                    and self.cache is not None
                    and isinstance(out, pd.DataFrame)
                    and not self._transform_requests_csv_export(task.get("pre_transform"))
                )
                if cache_ready:
                    pre_demand_fp = self._demand_fingerprint(
                        task.get("source"),
                        task.get("pre_transform"),
                        combine=task.get("combine", "concat"),
                        mode="preprofile",
                        projection=task.get("base_projection"),
                    )
                    self.cache.put_dataframe(
                        key,
                        out,
                        meta={
                            "source": self._source_token(task.get("source"), combine=task.get("combine", "concat")),
                            "combine": task.get("combine", "concat"),
                            "transform": task.get("pre_transform"),
                            "mode": "preprofile",
                            "demand_fingerprint": pre_demand_fp,
                            "projection": task.get("base_projection"),
                            "rows": rows,
                        },
                    )

                alias = self._preprofile_alias_name(key)
                self._register_preprofile_alias(alias, task, origin="rebuilt", rows=rows, cache_ready=cache_ready)
                prepared[key] = alias
                miss += 1
                out = None
                if rows is not None and int(rows) >= 100_000:
                    gc.collect()

            base_df = None
            gc.collect()

        # phase-3: rewrite dataset sources to prepared aliases
        for key, task in tasks.items():
            alias = prepared.get(key)
            if alias is None:
                continue
            for tgt in task.get("targets", []):
                ds = tgt.get("ds")
                if not isinstance(ds, Mapping):
                    continue
                ds["source"] = alias
                ds["transform"] = tgt.get("runtime_transform")

        return {"tasks": len(prepared), "hits": hits, "miss": miss}

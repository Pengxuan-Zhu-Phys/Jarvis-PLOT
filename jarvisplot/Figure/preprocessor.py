#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import hashlib
import json

import pandas as pd

from .load_data import _preprofiling, addcolumn, filter as filter_df, grid_profiling, profiling, sortby


class DataPreprocessor:
    def __init__(self, context, cache=None, dataset_registry: Optional[Dict[str, Any]] = None, logger=None):
        self.context = context
        self.cache = cache
        self.dataset_registry = dataset_registry or {}
        self.logger = logger
        self._emitted_sources = set()
        self._preprofile_alias_meta: Dict[str, Dict[str, Any]] = {}
        self._named_share_signatures: Dict[str, str] = {}

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

    @staticmethod
    def _canonical_json(payload: Any) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)

    @classmethod
    def _stable_hash(cls, payload: Any) -> str:
        return hashlib.sha1(cls._canonical_json(payload).encode("utf-8")).hexdigest()

    @staticmethod
    def _clone_df(df: pd.DataFrame) -> pd.DataFrame:
        try:
            return df.copy(deep=False)
        except Exception:
            return deepcopy(df)

    @staticmethod
    def _safe_nrows(df: Any) -> Optional[int]:
        if isinstance(df, pd.DataFrame):
            return int(df.shape[0])
        return None

    def _runtime_source_label(self, source: Any) -> str:
        if isinstance(source, str):
            return source
        if isinstance(source, (list, tuple)):
            return "[" + ", ".join(str(x) for x in source) + "]"
        return str(source)

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
    ) -> str:
        payload = {
            "kind": "pipeline",
            "algo": "pregrid-v6",
            "source": self._source_token(source, combine=combine),
            "transform": transform,
            "combine": str(combine),
            "mode": str(mode),
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
    ) -> Dict[str, Any]:
        return {
            "schema": "jp-demand-v7",
            "source": self._source_token(source, combine=combine),
            "transform": transform,
            "combine": str(combine),
            "mode": str(mode),
        }

    def _demand_fingerprint(
        self,
        source: Any,
        transform: Any,
        combine: str = "concat",
        mode: str = "runtime",
    ) -> str:
        return self._stable_hash(self._demand_payload(source, transform, combine=combine, mode=mode))

    def _layer_signature(self, layer: Mapping[str, Any]) -> str:
        ds = layer.get("data", [])
        tokens = []
        for item in ds:
            if isinstance(item, Mapping):
                tokens.append(
                    {
                        "source": self._source_token(item.get("source")),
                        "transform": item.get("transform"),
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
    ) -> Tuple[bool, str]:
        if not isinstance(meta, Mapping):
            return False, "meta-missing"

        expected_demand = self._demand_fingerprint(source, transform, combine=combine, mode=mode)
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

    def load_named_layer(self, name: Optional[str], layer: Mapping[str, Any]):
        if not name or self.cache is None:
            return None
        signature = self._layer_signature(layer)
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
        return named

    def persist_named_layer(self, name: Optional[str], layer: Mapping[str, Any], data) -> None:
        if not name or self.cache is None:
            return
        if not isinstance(data, pd.DataFrame):
            return
        signature = self._layer_signature(layer)
        self._named_share_signatures[str(name)] = str(signature)
        self.cache.put_named(name, signature, data)
        self._debug(f"Stored named dataset '{name}' into cache.")

    def _resolve_source_data(self, source: Any, combine: str = "concat"):
        if isinstance(source, str):
            return self.context.get(source)

        if isinstance(source, (list, tuple)):
            frames: List[pd.DataFrame] = []
            source_rows: List[str] = []
            rows_before = 0
            for ss in source:
                dt = self.context.get(str(ss))
                if dt is None:
                    self._warn(f"Source '{ss}' not found in context.")
                    continue
                if not isinstance(dt, pd.DataFrame):
                    self._warn(f"Source '{ss}' is not a DataFrame (type={type(dt)}), skipped in concat.")
                    continue
                nrow = int(dt.shape[0])
                rows_before += nrow
                source_rows.append(f"{ss}:{nrow}")
                frames.append(dt)
            if not frames:
                return None
            mode = str(combine or "concat").lower()
            if mode != "concat":
                self._warn(f"Unsupported source-list combine mode '{combine}', fallback to 'concat'.")
            out = pd.concat(frames, ignore_index=False)
            rows_after = int(out.shape[0]) if isinstance(out, pd.DataFrame) else "NA"
            self._warn(
                "Source concat rows:\n\t sources -> {}\n\t rows_before -> {}\n\t rows_after -> {}.".format(
                    ", ".join(source_rows) if source_rows else "<none>",
                    rows_before,
                    rows_after,
                )
            )
            return out

        self._warn(f"Unsupported source type in pipeline: {type(source)}")
        return None

    def _emit_source_summary(self, source: Any) -> None:
        names: List[str] = []
        if isinstance(source, str):
            names = [source]
        elif isinstance(source, (list, tuple)):
            names = [str(x) for x in source]

        for name in names:
            if name in self._emitted_sources:
                continue
            dts = self.dataset_registry.get(name)
            if dts is None:
                self._emitted_sources.add(name)
                continue
            try:
                if hasattr(dts, "emit_summary"):
                    dts.emit_summary(force_load=True)
            except Exception as e:
                self._warn(f"Emit summary failed for source '{name}': {e}")
            self._emitted_sources.add(name)

    def _apply_transforms(
        self,
        df,
        transform: Optional[Sequence[Mapping[str, Any]]],
        profile_mode: str = "runtime",
        source_label: Optional[str] = None,
    ):
        if transform is None:
            return df
        if not isinstance(transform, list):
            self._warn(f"Illegal transform format, list required -> {transform}")
            return df

        for trans in transform:
            if not isinstance(trans, Mapping):
                self._warn(f"Invalid transform step skipped -> {trans}")
                continue

            if "filter" in trans:
                df = filter_df(df, trans["filter"], self.logger)
            elif "profile" in trans:
                profile_cfg = trans.get("profile", {})
                if str(profile_mode).lower() == "preprofile":
                    df = _preprofiling(df, profile_cfg, self.logger)
                else:
                    before_rows = self._safe_nrows(df)
                    method = "bridson"
                    binv = "default"
                    if isinstance(profile_cfg, Mapping):
                        method = str(profile_cfg.get("method", "bridson")).lower()
                        if "bin" in profile_cfg:
                            binv = profile_cfg.get("bin")
                    self._warn(
                        "Runtime profile START:\n\t source \t-> {}\n\t step \t\t-> profile, \n\t method \t-> {}\n\t bin \t\t-> {}\n\t rows_before \t-> {}".format(
                            source_label or "<unknown>",
                            method,
                            binv,
                            before_rows if before_rows is not None else "NA",
                        )
                    )
                    df = profiling(df, profile_cfg, self.logger)
                    after_rows = self._safe_nrows(df)
                    delta = "NA"
                    if before_rows is not None and after_rows is not None:
                        delta = after_rows - before_rows
                    self._warn(
                        "Runtime profile DONE: \n\t source \t-> {}\n\t step \t\t-> profile \n\t method \t-> {}\n\t bin \t\t-> {}\n\t rows_after \t-> {}\n\t delta \t\t-> {}".format(
                            source_label or "<unknown>",
                            method,
                            binv,
                            after_rows if after_rows is not None else "NA",
                            delta,
                        )
                    )
            elif "grid_profile" in trans:
                profile_cfg = trans.get("grid_profile", {})
                if isinstance(profile_cfg, Mapping):
                    profile_cfg = deepcopy(profile_cfg)
                    profile_cfg.setdefault("method", "grid")
                else:
                    profile_cfg = {"method": "grid"}
                before_rows = self._safe_nrows(df)
                binv = profile_cfg.get("bin", "default")
                self._warn(
                    "Runtime profile START: source \t-> {}\n\t step \t-> 'grid_profile,\n\t method \t-> 'grid',\n\t bin \t-> {},\n\t rows_before \t-> {}".format(
                        source_label or "<unknown>",
                        binv,
                        before_rows if before_rows is not None else "NA",
                    )
                )
                df = grid_profiling(df, profile_cfg, self.logger)
                after_rows = self._safe_nrows(df)
                delta = "NA"
                if before_rows is not None and after_rows is not None:
                    delta = after_rows - before_rows
                self._warn(
                    "Runtime profile DONE: \n\t source \t-> {}\n\tstep \t-> 'grid_profile,\n\t method \t-> 'grid',\n\t bin \t\t-> {}\n\t rows_after \t-> {},\n\t delta \t->".format(
                        source_label or "<unknown>",
                        binv,
                        after_rows if after_rows is not None else "NA",
                        delta,
                    )
                )
            elif "sortby" in trans:
                df = sortby(df, trans["sortby"], self.logger)
            elif "add_column" in trans:
                df = addcolumn(df, trans["add_column"], self.logger)

        return df

    def apply_transforms(self, df, transform: Optional[Sequence[Mapping[str, Any]]]):
        """Prebuild pass: execute profile step as lightweight _preprofiling."""
        return self._apply_transforms(df, transform, profile_mode="preprofile")

    def apply_runtime_transforms(
        self,
        df,
        transform: Optional[Sequence[Mapping[str, Any]]],
        source_label: Optional[str] = None,
    ):
        """Runtime pass: keep original profiling behavior."""
        return self._apply_transforms(df, transform, profile_mode="runtime", source_label=source_label)

    def run_pipeline(
        self,
        source: Any,
        transform: Optional[Sequence[Mapping[str, Any]]],
        combine: str = "concat",
        use_cache: bool = True,
        mode: str = "runtime",
    ) -> Tuple[Optional[pd.DataFrame], str, bool]:
        key = self._pipeline_key(source, transform, combine=combine, mode=mode)
        runtime_mode = str(mode).lower() == "runtime"
        runtime_sig = self._runtime_profile_signature(transform) if runtime_mode else None
        demand_fp = self._demand_fingerprint(source, transform, combine=combine, mode=mode)

        if use_cache and self.cache is not None:
            meta = None
            try:
                if hasattr(self.cache, "get_dataframe_meta"):
                    meta = self.cache.get_dataframe_meta(key)
            except Exception:
                meta = None

            compatible, reason = self._is_dataframe_cache_compatible(
                source=source,
                transform=transform,
                combine=combine,
                mode=mode,
                key=key,
                meta=meta,
            )
            if compatible:
                cached = self.cache.get_dataframe(key)
                if cached is not None:
                    if runtime_mode and runtime_sig is not None:
                        cache_file = "<unknown>"
                        try:
                            cache_file = str((self.cache.data_dir / f"{key}.pkl").resolve())
                        except Exception:
                            pass
                        self._warn(
                            "Runtime profile cache HIT:\n\t source \t-> {},\n\t key \t\t-> {},\n\t fingerprint \t-> {},\n\t cache_file \t-> {},\n\t rows \t\t-> {}.".format(
                                self._runtime_source_label(source),
                                key,
                                demand_fp,
                                cache_file,
                                self._safe_nrows(cached) if self._safe_nrows(cached) is not None else "NA",
                            )
                        )
                    self._emit_source_summary(source)
                    self._debug(f"Pipeline cache HIT -> {key}")
                    return self._clone_df(cached), key, True
                reason = "cache-read-failed"

            if runtime_mode and runtime_sig is not None:
                if reason in {"meta-missing", "demand-fingerprint-missing"}:
                    self._warn(
                        "Runtime profile cache MISS:\n\t source \t-> {},\n\t key \t\t-> {},\n\t fingerprint \t-> {}".format(
                            self._runtime_source_label(source),
                            key,
                            demand_fp,
                        )
                    )
                else:
                    cached_sig = None
                    cached_transform_sig = None
                    cached_demand = None
                    if isinstance(meta, Mapping):
                        cached_sig = meta.get("runtime_profile_signature")
                        cached_transform_sig = meta.get("runtime_transform_signature")
                        cached_demand = meta.get("demand_fingerprint")
                    self._warn(
                        "Runtime profile cache INVALID:\n\t source \t-> {},\n\t key \t-> {},\n\t reason \t-> {},\n\t expected_demand \t-> {},\n\t cached_demand \t-> {},\n\t expected_profile_sig \t-> {},\n\t cached_profile_sig \t-> {},\n\t expected_transform_sig \t-> {},\n\t cached_transform_sig \t-> {}".format(
                            self._runtime_source_label(source),
                            key,
                            reason,
                            demand_fp,
                            str(cached_demand) if cached_demand else "<none>",
                            runtime_sig,
                            str(cached_sig) if cached_sig else "<none>",
                            self._stable_hash(transform),
                            str(cached_transform_sig) if cached_transform_sig else "<none>",
                        )
                    )
            else:
                if reason != "meta-missing":
                    self._debug(f"Pipeline cache INVALID ({reason}) -> {key}")

        raw = self._resolve_source_data(source, combine=combine)
        if raw is None:
            return None, key, False

        if str(mode).lower() == "runtime":
            src_label = self._runtime_source_label(source)
            if isinstance(source, str) and source in self._preprofile_alias_meta:
                meta = self._preprofile_alias_meta.get(source, {})
                self._warn(
                    "Runtime profile input:\n\t source \t-> {},\n uses preprofile alias:\n\t key \t\t-> {},\n\t origin \t-> {},\n\t cache_file \t-> {},\n\t rows_in \t-> {}.".format(
                        src_label,
                        meta.get("pre_key", "<unknown>")[:16] if isinstance(meta.get("pre_key"), str) else "<unknown>",
                        meta.get("origin", "<unknown>"),
                        meta.get("cache_file", "<memory-only>"),
                        self._safe_nrows(raw) if self._safe_nrows(raw) is not None else "NA",
                    )
                )

        if isinstance(raw, pd.DataFrame):
            work = raw.copy(deep=True)
        else:
            work = deepcopy(raw)
        if str(mode).lower() == "preprofile":
            work = self.apply_transforms(work, transform)
        else:
            work = self.apply_runtime_transforms(
                work,
                transform,
                source_label=self._runtime_source_label(source),
            )

        if use_cache and self.cache is not None and isinstance(work, pd.DataFrame):
            meta = {
                "source": self._source_token(source, combine=combine),
                "combine": combine,
                "transform": transform,
                "mode": mode,
                "demand_fingerprint": demand_fp,
            }
            runtime_profile_sig = runtime_sig if runtime_mode else None
            if runtime_mode and runtime_profile_sig is not None:
                meta["runtime_profile_signature"] = runtime_profile_sig
                meta["runtime_transform_signature"] = self._stable_hash(transform)
            self.cache.put_dataframe(
                key,
                work,
                meta=meta,
            )
            if runtime_mode and runtime_profile_sig is not None:
                cache_file = "<unknown>"
                try:
                    cache_file = str((self.cache.data_dir / f"{key}.pkl").resolve())
                except Exception:
                    pass
                self._warn(
                    "Runtime profile cache STORE:\n\t source \t-> {},\n\t key \t\t-> {},\n\t cache_file \t-> {},\n\t rows \t\t-> {}.".format(
                        self._runtime_source_label(source),
                        key[:16],
                        cache_file,
                        self._safe_nrows(work) if self._safe_nrows(work) is not None else "NA",
                    )
                )
            self._debug(f"Pipeline cache STORE -> {key}")

        if isinstance(work, pd.DataFrame):
            work = self._clone_df(work)
        return work, key, False

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

                    key = self._pipeline_key(src, pre_transform, combine=combine, mode="preprofile")
                    task = tasks.get(key)
                    if task is None:
                        prefix_transform = pre_transform[:-1]
                        profile_cfg = {}
                        if pre_transform and isinstance(pre_transform[-1], Mapping):
                            profile_cfg = deepcopy(pre_transform[-1].get("profile", {}))
                        task = {
                            "pre_key": key,
                            "source": src,
                            "combine": combine,
                            "pre_transform": pre_transform,
                            "prefix_transform": prefix_transform,
                            "profile_cfg": profile_cfg,
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
                )
                if ok_cache:
                    cached = self.cache.get_dataframe(key)
                    if cached is not None:
                        alias = f"__jp_preprofile_{key[:16]}"
                        self.context.update(alias, self._clone_df(cached))
                        prepared[key] = alias
                        self._emit_source_summary(task.get("source"))
                        cache_file = "<memory-only>"
                        if self.cache is not None:
                            try:
                                cache_file = str((self.cache.data_dir / f"{key}.pkl").resolve())
                            except Exception:
                                cache_file = "<memory-only>"
                        self._preprofile_alias_meta[alias] = {
                            "pre_key": key,
                            "origin": "cache-hit",
                            "cache_file": cache_file,
                            "source": task.get("source"),
                            "rows": self._safe_nrows(cached),
                        }
                        logged_pairs = set()
                        for tgt in task.get("targets", []):
                            fig_name = str(tgt.get("figure_name", "<noname>"))
                            layer_name = str(tgt.get("layer_name", "<noname>"))
                            pair = (fig_name, layer_name)
                            if pair in logged_pairs:
                                continue
                            logged_pairs.add(pair)
                            self._warn(
                                "Preprofile cache HIT:\n\t figure -> {},\n\t layer -> {},\n\t source -> {},\n\t key -> {}".format(
                                    fig_name,
                                    layer_name,
                                    task.get("source"),
                                    key[:16],
                                )
                            )
                        hits += 1
                    else:
                        self._debug(f"Preprofile cache INVALID (cache-read-failed) -> {key}")
                elif reason != "meta-missing":
                    self._debug(f"Preprofile cache INVALID ({reason}) -> {key}")

            if alias is None:
                pending.append(task)

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
            )
            if base_df is None:
                continue

            for task in grp:
                out = _preprofiling(self._clone_df(base_df), task.get("profile_cfg", {}), self.logger)
                if out is None:
                    continue

                key = task["pre_key"]
                if bool(task.get("cache_flag", True)) and self.cache is not None and isinstance(out, pd.DataFrame):
                    pre_demand_fp = self._demand_fingerprint(
                        task.get("source"),
                        task.get("pre_transform"),
                        combine=task.get("combine", "concat"),
                        mode="preprofile",
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
                        },
                    )

                alias = f"__jp_preprofile_{key[:16]}"
                self.context.update(alias, self._clone_df(out))
                cache_file = "<memory-only>"
                if bool(task.get("cache_flag", True)) and self.cache is not None:
                    try:
                        cache_file = str((self.cache.data_dir / f"{key}.pkl").resolve())
                    except Exception:
                        cache_file = "<memory-only>"
                self._preprofile_alias_meta[alias] = {
                    "pre_key": key,
                    "origin": "rebuilt",
                    "cache_file": cache_file,
                    "source": task.get("source"),
                    "rows": self._safe_nrows(out),
                }
                prepared[key] = alias
                miss += 1

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

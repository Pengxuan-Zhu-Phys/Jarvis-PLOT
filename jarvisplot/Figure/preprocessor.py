#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import hashlib
import json

import pandas as pd

from .load_data import _preprofiling, addcolumn, filter as filter_df, profiling, sortby


class DataPreprocessor:
    def __init__(self, context, cache=None, dataset_registry: Optional[Dict[str, Any]] = None, logger=None):
        self.context = context
        self.cache = cache
        self.dataset_registry = dataset_registry or {}
        self.logger = logger
        self._emitted_sources = set()

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

    def _source_token(self, source: Any, combine: str = "concat") -> Dict[str, Any]:
        if isinstance(source, str):
            dts = self.dataset_registry.get(source)
            if dts is not None and hasattr(dts, "fingerprint"):
                fp = dts.fingerprint(self.cache)
                return {"source": source, "fingerprint": fp}
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
            "algo": "pregrid-v1",
            "source": self._source_token(source, combine=combine),
            "transform": transform,
            "combine": str(combine),
            "mode": str(mode),
        }
        if self.cache is not None:
            return self.cache.cache_key(payload)
        return self._stable_hash(payload)

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

    def load_named_layer(self, name: Optional[str], layer: Mapping[str, Any]):
        if not name or self.cache is None:
            return None
        signature = self._layer_signature(layer)
        return self.cache.get_named(name, signature)

    def persist_named_layer(self, name: Optional[str], layer: Mapping[str, Any], data) -> None:
        if not name or self.cache is None:
            return
        if not isinstance(data, pd.DataFrame):
            return
        signature = self._layer_signature(layer)
        self.cache.put_named(name, signature, data)
        self._debug(f"Stored named dataset '{name}' into cache.")

    def _resolve_source_data(self, source: Any, combine: str = "concat"):
        if isinstance(source, str):
            return self.context.get(source)

        if isinstance(source, (list, tuple)):
            frames: List[pd.DataFrame] = []
            for ss in source:
                dt = self.context.get(str(ss))
                if dt is None:
                    self._warn(f"Source '{ss}' not found in context.")
                    continue
                frames.append(dt)
            if not frames:
                return None
            mode = str(combine or "concat").lower()
            if mode != "concat":
                self._warn(f"Unsupported source-list combine mode '{combine}', fallback to 'concat'.")
            return pd.concat(frames, ignore_index=False)

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
                if str(profile_mode).lower() == "preprofile":
                    df = _preprofiling(df, trans["profile"], self.logger)
                else:
                    df = profiling(df, trans["profile"], self.logger)
            elif "sortby" in trans:
                df = sortby(df, trans["sortby"], self.logger)
            elif "add_column" in trans:
                df = addcolumn(df, trans["add_column"], self.logger)

        return df

    def apply_transforms(self, df, transform: Optional[Sequence[Mapping[str, Any]]]):
        """Prebuild pass: execute profile step as lightweight _preprofiling."""
        return self._apply_transforms(df, transform, profile_mode="preprofile")

    def apply_runtime_transforms(self, df, transform: Optional[Sequence[Mapping[str, Any]]]):
        """Runtime pass: keep original profiling behavior."""
        return self._apply_transforms(df, transform, profile_mode="runtime")

    def run_pipeline(
        self,
        source: Any,
        transform: Optional[Sequence[Mapping[str, Any]]],
        combine: str = "concat",
        use_cache: bool = True,
        mode: str = "runtime",
    ) -> Tuple[Optional[pd.DataFrame], str, bool]:
        key = self._pipeline_key(source, transform, combine=combine, mode=mode)

        if use_cache and self.cache is not None:
            cached = self.cache.get_dataframe(key)
            if cached is not None:
                self._emit_source_summary(source)
                self._debug(f"Pipeline cache HIT -> {key}")
                return self._clone_df(cached), key, True

        raw = self._resolve_source_data(source, combine=combine)
        if raw is None:
            return None, key, False

        if isinstance(raw, pd.DataFrame):
            work = raw.copy(deep=True)
        else:
            work = deepcopy(raw)
        if str(mode).lower() == "preprofile":
            work = self.apply_transforms(work, transform)
        else:
            work = self.apply_runtime_transforms(work, transform)

        if use_cache and self.cache is not None and isinstance(work, pd.DataFrame):
            self.cache.put_dataframe(
                key,
                work,
                meta={
                    "source": self._source_token(source, combine=combine),
                    "combine": combine,
                    "transform": transform,
                    "mode": mode,
                },
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
        if not isinstance(profile_cfg, Mapping):
            return profile_cfg
        slim: Dict[str, Any] = {}
        if "coordinates" in profile_cfg:
            slim["coordinates"] = deepcopy(profile_cfg["coordinates"])
        if "objective" in profile_cfg:
            slim["objective"] = profile_cfg["objective"]
        if "grid_points" in profile_cfg:
            slim["grid_points"] = profile_cfg["grid_points"]
        if "pregrid" in profile_cfg:
            slim["pregrid"] = deepcopy(profile_cfg["pregrid"])
        if "pregrid_bin" in profile_cfg:
            slim["pregrid_bin"] = profile_cfg["pregrid_bin"]
        if "coordinates" not in slim and "coordinates" in profile_cfg:
            slim["coordinates"] = deepcopy(profile_cfg["coordinates"])
        if not slim:
            return deepcopy(profile_cfg)
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
        pre_transform.append(
            {"profile": self._preprofile_profile_cfg(first.get("profile", {}))}
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

                    task["targets"].append({"ds": ds, "runtime_transform": runtime_transform})

        # phase-1: preprofile cache fast-path
        pending: List[Dict[str, Any]] = []
        for key, task in tasks.items():
            alias = None
            if bool(task.get("cache_flag", True)) and self.cache is not None:
                cached = self.cache.get_dataframe(key)
                if cached is not None:
                    alias = f"__jp_preprofile_{key[:16]}"
                    self.context.update(alias, self._clone_df(cached))
                    prepared[key] = alias
                    self._emit_source_summary(task.get("source"))
                    hits += 1

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
                    self.cache.put_dataframe(
                        key,
                        out,
                        meta={
                            "source": self._source_token(task.get("source"), combine=task.get("combine", "concat")),
                            "combine": task.get("combine", "concat"),
                            "transform": task.get("pre_transform"),
                            "mode": "preprofile",
                        },
                    )

                alias = f"__jp_preprofile_{key[:16]}"
                self.context.update(alias, self._clone_df(out))
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

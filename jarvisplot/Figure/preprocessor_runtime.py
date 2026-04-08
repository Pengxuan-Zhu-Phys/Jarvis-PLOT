from __future__ import annotations

from copy import deepcopy
import gc
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import polars as pl
except Exception:
    pl = None

from ..memtrace import memtrace_checkpoint, memtrace_object_inventory
from ..utils.pathing import resolve_project_path
from .load_data import _preprofiling, addcolumn, drop_columns, filter as filter_df, grid_profiling, keep_columns, profiling, sortby


def _csv_export_target(target: Any) -> Any:
    if isinstance(target, Mapping):
        return target.get("path", target.get("file", target.get("target", target.get("value", ""))))
    return target


def _resolve_csv_export_path(preprocessor, target: Any) -> Path:
    raw = _csv_export_target(target)
    path = str(raw).strip()
    if not path:
        raise ValueError("tocsv requires a non-empty path")
    return resolve_project_path(path, base_dir=getattr(preprocessor, "base_dir", None))


def _save_dataframe_csv(preprocessor, df, target: Any, *, stage: str, source_label: Optional[str] = None) -> Path:
    out_path = _resolve_csv_export_path(preprocessor, target)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = preprocessor._safe_nrows(df) if hasattr(preprocessor, "_safe_nrows") else None
    memtrace_checkpoint(
        preprocessor.logger,
        "pipeline.csv_export.before",
        df,
        extra={
            "source": source_label or "<unknown>",
            "stage": stage,
            "path": str(out_path),
            "rows": rows if rows is not None else "NA",
        },
    )
    memtrace_object_inventory(
        preprocessor.logger,
        "pipeline.csv_export.inventory",
        {"df": df},
        roles={"df": "csv export dataframe"},
        min_bytes=64 * 1024 * 1024,
    )
    if isinstance(df, pd.DataFrame):
        df.to_csv(out_path, index=False)
    else:
        pd.DataFrame(df).to_csv(out_path, index=False)
    if preprocessor.logger:
        preprocessor._info(
            "Saved transformed dataframe to CSV:\n\t source \t-> {}\n\t stage \t-> {}\n\t path \t-> {}".format(
                source_label or "<unknown>",
                stage,
                out_path,
            )
        )
    memtrace_checkpoint(
        preprocessor.logger,
        "pipeline.csv_export.after",
        df,
        extra={
            "source": source_label or "<unknown>",
            "stage": stage,
            "path": str(out_path),
            "rows": rows if rows is not None else "NA",
        },
    )
    return out_path

def resolve_source_data(preprocessor, source: Any, combine: str = "concat"):
    if isinstance(source, str):
        return preprocessor.context.get(source)

    if isinstance(source, (list, tuple)):
        frames: List[Any] = []
        source_rows: List[str] = []
        rows_before_total = 0
        for ss in source:
            dt = preprocessor.context.get(str(ss))
            if dt is None:
                preprocessor._warn(f"Source '{ss}' not found in context.")
                continue
            nrow = preprocessor._safe_nrows(dt)
            if nrow is not None:
                rows_before_total += int(nrow)
            source_rows.append(f"{ss}:{nrow if nrow is not None else 'NA'}")
            frames.append(dt)
        if not frames:
            return None
        mode = str(combine or "concat").lower()
        if mode != "concat":
            preprocessor._warn(f"Unsupported source-list combine mode '{combine}', fallback to 'concat'.")

        if pl is not None and all(preprocessor._is_polars_frame(frame) for frame in frames):
            lazy_frames = [
                frame if isinstance(frame, pl.LazyFrame) else frame.lazy()
                for frame in frames
            ]
            out = pl.concat(lazy_frames, how="vertical_relaxed")
            rows_after = "lazy"
        else:
            pandas_frames = [preprocessor.ensure_pandas(frame, reason="concat-source-list") for frame in frames]
            out = pd.concat(pandas_frames, ignore_index=False)
            rows_after = int(out.shape[0]) if isinstance(out, pd.DataFrame) else "NA"
        preprocessor._warn(
            "Source concat rows:\n\t sources -> {}\n\t rows_before -> {}\n\t rows_after -> {}.".format(
                ", ".join(source_rows) if source_rows else "<none>",
                rows_before_total if rows_before_total else "NA",
                rows_after,
            )
        )
        return out

    preprocessor._warn(f"Unsupported source type in pipeline: {type(source)}")
    return None

def emit_source_summary(preprocessor, source: Any) -> None:
    names: List[str] = []
    if isinstance(source, str):
        names = [source]
    elif isinstance(source, (list, tuple)):
        names = [str(x) for x in source]

    for name in names:
        if name in preprocessor._emitted_sources:
            continue
        dts = preprocessor.dataset_registry.get(name)
        if dts is None:
            preprocessor._emitted_sources.add(name)
            continue
        try:
            if hasattr(dts, "emit_summary"):
                dts.emit_summary(force_load=True)
        except Exception as e:
            preprocessor._warn(f"Emit summary failed for source '{name}': {e}")
        preprocessor._emitted_sources.add(name)

def apply_transforms_impl(
    preprocessor,
    df,
    transform: Optional[Sequence[Mapping[str, Any]]],
    profile_mode: str = "runtime",
    source_label: Optional[str] = None,
):
    if transform is None:
        return df
    if not isinstance(transform, list):
        preprocessor._warn(f"Illegal transform format, list required -> {transform}")
        return df

    df = preprocessor.ensure_pandas(df, reason=f"{profile_mode}-transform")

    for trans in transform:
        if not isinstance(trans, Mapping):
            preprocessor._warn(f"Invalid transform step skipped -> {trans}")
            continue
        prev_df = df

        if "filter" in trans:
            df = filter_df(df, trans["filter"], preprocessor.logger)
        elif "profile" in trans:
            profile_cfg = trans.get("profile", {})
            if str(profile_mode).lower() == "preprofile":
                memtrace_checkpoint(
                    preprocessor.logger,
                    "pipeline.profile.before",
                    df,
                    extra={"source": source_label or "<preprofile>", "mode": profile_mode},
                )
                df = _preprofiling(df, profile_cfg, preprocessor.logger)
                memtrace_checkpoint(
                    preprocessor.logger,
                    "pipeline.profile.after",
                    df,
                    extra={"source": source_label or "<preprofile>", "mode": profile_mode},
                )
            else:
                before_rows = preprocessor._safe_nrows(df)
                method = "bridson"
                binv = "default"
                if isinstance(profile_cfg, Mapping):
                    method = str(profile_cfg.get("method", "bridson")).lower()
                    if "bin" in profile_cfg:
                        binv = profile_cfg.get("bin")
                preprocessor._info(
                    "Runtime profile START:\n\t source \t-> {}\n\t step \t\t-> profile, \n\t method \t-> {}\n\t bin \t\t-> {}\n\t rows_before \t-> {}".format(
                        source_label or "<unknown>",
                        method,
                        binv,
                        before_rows if before_rows is not None else "NA",
                    )
                )
                memtrace_checkpoint(
                    preprocessor.logger,
                    "pipeline.profile.before",
                    df,
                    extra={
                        "source": source_label or "<unknown>",
                        "mode": profile_mode,
                        "method": method,
                        "bin": binv,
                    },
                )
                df = profiling(df, profile_cfg, preprocessor.logger)
                after_rows = preprocessor._safe_nrows(df)
                delta = "NA"
                if before_rows is not None and after_rows is not None:
                    delta = after_rows - before_rows
                preprocessor._warn(
                    "Runtime profile DONE: \n\t source \t-> {}\n\t step \t\t-> profile \n\t method \t-> {}\n\t bin \t\t-> {}\n\t rows_after \t-> {}\n\t delta \t\t-> {}".format(
                        source_label or "<unknown>",
                        method,
                        binv,
                        after_rows if after_rows is not None else "NA",
                        delta,
                    )
                )
                memtrace_checkpoint(
                    preprocessor.logger,
                    "pipeline.profile.after",
                    df,
                    extra={
                        "source": source_label or "<unknown>",
                        "mode": profile_mode,
                        "method": method,
                        "bin": binv,
                    },
                )
        elif "grid_profile" in trans:
            profile_cfg = trans.get("grid_profile", {})
            if isinstance(profile_cfg, Mapping):
                profile_cfg = deepcopy(profile_cfg)
                profile_cfg.setdefault("method", "grid")
            else:
                profile_cfg = {"method": "grid"}
            before_rows = preprocessor._safe_nrows(df)
            binv = profile_cfg.get("bin", "default")
            preprocessor._info(
                "Runtime profile START: source \t-> {}\n\t step \t-> 'grid_profile,\n\t method \t-> 'grid',\n\t bin \t-> {},\n\t rows_before \t-> {}".format(
                    source_label or "<unknown>",
                    binv,
                    before_rows if before_rows is not None else "NA",
                )
            )
            memtrace_checkpoint(
                preprocessor.logger,
                "pipeline.grid_profile.before",
                df,
                extra={
                    "source": source_label or "<unknown>",
                    "mode": profile_mode,
                    "bin": binv,
                },
            )
            df = grid_profiling(df, profile_cfg, preprocessor.logger)
            after_rows = preprocessor._safe_nrows(df)
            delta = "NA"
            if before_rows is not None and after_rows is not None:
                delta = after_rows - before_rows
            preprocessor._warn(
                "Runtime profile DONE: \n\t source \t-> {}\n\tstep \t-> 'grid_profile,\n\t method \t-> 'grid',\n\t bin \t\t-> {}\n\t rows_after \t-> {},\n\t delta \t->".format(
                    source_label or "<unknown>",
                    binv,
                    after_rows if after_rows is not None else "NA",
                    delta,
                )
            )
            memtrace_checkpoint(
                preprocessor.logger,
                "pipeline.grid_profile.after",
                df,
                extra={
                    "source": source_label or "<unknown>",
                    "mode": profile_mode,
                    "bin": binv,
                },
            )
        elif "sortby" in trans:
            df = sortby(df, trans["sortby"], preprocessor.logger)
        elif "add_column" in trans:
            df = addcolumn(df, trans["add_column"], preprocessor.logger)
        elif "keep_columns" in trans:
            df = keep_columns(df, trans.get("keep_columns"), preprocessor.logger)
        elif "drop_columns" in trans:
            df = drop_columns(df, trans.get("drop_columns"), preprocessor.logger)
        elif "tocsv" in trans or "to_csv" in trans:
            _save_dataframe_csv(
                preprocessor,
                df,
                trans.get("tocsv", trans.get("to_csv")),
                stage=profile_mode,
                source_label=source_label,
            )

        if prev_df is not df:
            collect_prev = preprocessor._should_collect_dataframe(prev_df)
            try:
                del prev_df
            except Exception:
                prev_df = None
            if collect_prev:
                gc.collect()

    return df

def apply_transforms(preprocessor, df, transform: Optional[Sequence[Mapping[str, Any]]]):
    """Prebuild pass: execute profile step as lightweight _preprofiling."""
    return preprocessor._apply_transforms(df, transform, profile_mode="preprofile")

def apply_runtime_transforms(preprocessor,
    df,
    transform: Optional[Sequence[Mapping[str, Any]]],
    source_label: Optional[str] = None,
):
    """Runtime pass: keep original profiling behavior."""
    return preprocessor._apply_transforms(df, transform, profile_mode="runtime", source_label=source_label)

def run_pipeline(preprocessor,
    source: Any,
    transform: Optional[Sequence[Mapping[str, Any]]],
    combine: str = "concat",
    use_cache: bool = True,
    mode: str = "runtime",
    demand_columns: Optional[Sequence[str]] = None,
    projection: Optional[Sequence[str]] = None,
) -> Tuple[Optional[pd.DataFrame], str, bool]:
    mode_lower = str(mode).lower()
    effective_transform = preprocessor._effective_transform(source, transform)
    if projection is None and mode_lower == "runtime":
        projection = preprocessor._runtime_projection(effective_transform, demand_columns)
    projection = preprocessor._projection_list(projection)
    key = preprocessor._pipeline_key(source, effective_transform, combine=combine, mode=mode, projection=projection)
    runtime_mode = mode_lower == "runtime"
    export_requested = preprocessor._transform_requests_csv_export(effective_transform)
    cache_enabled = bool(use_cache) and mode_lower != "preprofile-base" and not export_requested
    if export_requested and bool(use_cache):
        preprocessor._debug(f"Pipeline cache disabled for CSV export transform -> {key}")
    runtime_sig = preprocessor._runtime_profile_signature(effective_transform) if runtime_mode else None
    demand_fp = preprocessor._demand_fingerprint(
        source,
        effective_transform,
        combine=combine,
        mode=mode,
        projection=projection,
    )

    if cache_enabled and preprocessor.cache is not None:
        meta = None
        try:
            if hasattr(preprocessor.cache, "get_dataframe_meta"):
                meta = preprocessor.cache.get_dataframe_meta(key)
        except Exception:
            meta = None

        compatible, reason = preprocessor._is_dataframe_cache_compatible(
            source=source,
            transform=effective_transform,
            combine=combine,
            mode=mode,
            key=key,
            meta=meta,
            projection=projection,
        )
        if compatible:
            cached = preprocessor.cache.get_dataframe(key)
            if cached is not None:
                if runtime_mode and runtime_sig is not None:
                    cache_file = "<unknown>"
                    try:
                        cache_file = str((preprocessor.cache.data_dir / f"{key}.pkl").resolve())
                    except Exception:
                        pass
                    preprocessor._info(
                        "Runtime profile cache HIT:\n\t source \t-> {},\n\t key \t\t-> {},\n\t fingerprint \t-> {},\n\t cache_file \t-> {},\n\t rows \t\t-> {}.".format(
                            preprocessor._runtime_source_label(source),
                            key,
                            demand_fp,
                            cache_file,
                            preprocessor._safe_nrows(cached) if preprocessor._safe_nrows(cached) is not None else "NA",
                        )
                    )
                preprocessor._emit_source_summary(source)
                preprocessor._debug(f"Pipeline cache HIT -> {key}")
                memtrace_checkpoint(
                    preprocessor.logger,
                    "pipeline.cache_hit",
                    cached,
                    extra={"source": preprocessor._runtime_source_label(source), "mode": mode},
                )
                cached = preprocessor._enrich_for_demand(cached, source, demand_columns)
                return preprocessor._clone_df(cached), key, True
            reason = "cache-read-failed"

        if runtime_mode and runtime_sig is not None:
            if reason in {"meta-missing", "demand-fingerprint-missing"}:
                preprocessor._info(
                    "Runtime profile cache MISS:\n\t source \t-> {},\n\t key \t\t-> {},\n\t fingerprint \t-> {}".format(
                        preprocessor._runtime_source_label(source),
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
                preprocessor._warn(
                    "Runtime profile cache INVALID:\n\t source \t-> {},\n\t key \t-> {},\n\t reason \t-> {},\n\t expected_demand \t-> {},\n\t cached_demand \t-> {},\n\t expected_profile_sig \t-> {},\n\t cached_profile_sig \t-> {},\n\t expected_transform_sig \t-> {},\n\t cached_transform_sig \t-> {}".format(
                        preprocessor._runtime_source_label(source),
                        key,
                        reason,
                        demand_fp,
                        str(cached_demand) if cached_demand else "<none>",
                        runtime_sig,
                        str(cached_sig) if cached_sig else "<none>",
                        preprocessor._stable_hash(effective_transform),
                        str(cached_transform_sig) if cached_transform_sig else "<none>",
                    )
                )
        else:
            if reason != "meta-missing":
                preprocessor._debug(f"Pipeline cache INVALID ({reason}) -> {key}")

    raw = preprocessor._resolve_source_data(source, combine=combine)
    if raw is None:
        return None, key, False
    raw = preprocessor._select_columns(raw, projection)
    memtrace_checkpoint(
        preprocessor.logger,
        "pipeline.source_resolved",
        raw,
        extra={
            "source": preprocessor._runtime_source_label(source),
            "mode": mode,
            "combine": combine,
        },
    )
    memtrace_object_inventory(
        preprocessor.logger,
        "pipeline.source_resolved.inventory",
        {"raw": raw},
        roles={"raw": "source dataframe"},
        min_bytes=64 * 1024 * 1024,
    )

    if mode_lower == "runtime":
        src_label = preprocessor._runtime_source_label(source)
        if isinstance(source, str) and source in preprocessor._preprofile_alias_meta:
            meta = preprocessor._preprofile_alias_meta.get(source, {})
            preprocessor._info(
                "Runtime profile input:\n\t source \t-> {},\n uses preprofile alias:\n\t key \t\t-> {},\n\t origin \t-> {},\n\t cache_file \t-> {},\n\t rows_in \t-> {}.".format(
                    src_label,
                    meta.get("pre_key", "<unknown>")[:16] if isinstance(meta.get("pre_key"), str) else "<unknown>",
                    meta.get("origin", "<unknown>"),
                    meta.get("cache_file", "<memory-only>"),
                    preprocessor._safe_nrows(raw) if preprocessor._safe_nrows(raw) is not None else "NA",
                )
            )

    must_pandas = mode_lower != "runtime"
    if effective_transform is None:
        work = preprocessor.ensure_pandas(raw, reason=f"{mode}-pipeline") if must_pandas else raw
    elif isinstance(raw, pd.DataFrame):
        if mode_lower in {"preprofile-base", "preprofile"}:
            work = raw
        else:
            work = preprocessor._clone_df(raw)
    else:
        work = preprocessor.ensure_pandas(raw, reason=f"{mode}-pipeline")

    if mode_lower == "preprofile":
        work = preprocessor.apply_transforms(work, effective_transform)
    elif effective_transform is not None:
        work = preprocessor.apply_runtime_transforms(
            work,
            effective_transform,
            source_label=preprocessor._runtime_source_label(source),
        )
    if mode_lower == "runtime":
        work = preprocessor._select_columns(work, preprocessor._runtime_cache_columns(effective_transform, demand_columns))
    memtrace_checkpoint(
        preprocessor.logger,
        "pipeline.transform_done",
        work,
        extra={"source": preprocessor._runtime_source_label(source), "mode": mode},
    )
    memtrace_object_inventory(
        preprocessor.logger,
        "pipeline.transform_done.inventory",
        {"raw": raw, "work": work},
        roles={
            "raw": "source dataframe",
            "work": "transform output",
        },
        min_bytes=64 * 1024 * 1024,
    )
    raw = None

    if cache_enabled and preprocessor.cache is not None and isinstance(work, pd.DataFrame):
        meta = {
            "source": preprocessor._source_token(source, combine=combine),
            "combine": combine,
            "transform": effective_transform,
            "mode": mode,
            "demand_fingerprint": demand_fp,
            "projection": projection,
        }
        runtime_profile_sig = runtime_sig if runtime_mode else None
        if runtime_mode and runtime_profile_sig is not None:
            meta["runtime_profile_signature"] = runtime_profile_sig
            meta["runtime_transform_signature"] = preprocessor._stable_hash(effective_transform)
        preprocessor.cache.put_dataframe(
            key,
            work,
            meta=meta,
        )
        if runtime_mode and runtime_profile_sig is not None:
            cache_file = "<unknown>"
            try:
                cache_file = str((preprocessor.cache.data_dir / f"{key}.pkl").resolve())
            except Exception:
                pass
            preprocessor._info(
                "Runtime profile cache STORE:\n\t source \t-> {},\n\t key \t\t-> {},\n\t cache_file \t-> {},\n\t rows \t\t-> {}.".format(
                    preprocessor._runtime_source_label(source),
                    key[:16],
                    cache_file,
                    preprocessor._safe_nrows(work) if preprocessor._safe_nrows(work) is not None else "NA",
                )
            )
        preprocessor._debug(f"Pipeline cache STORE -> {key}")

    work = preprocessor._enrich_for_demand(work, source, demand_columns)
    if isinstance(work, pd.DataFrame) and mode_lower == "runtime":
        work = preprocessor._clone_df(work)
    memtrace_checkpoint(
        preprocessor.logger,
        "pipeline.return",
        work,
        extra={"source": preprocessor._runtime_source_label(source), "mode": mode},
    )
    return work, key, False

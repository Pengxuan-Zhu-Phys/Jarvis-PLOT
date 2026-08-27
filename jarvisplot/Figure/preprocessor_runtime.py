from __future__ import annotations

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
from ..utils.expression import eval_dataframe_expression
from ..utils.pathing import resolve_project_path
from .bin_stat_runtime import bin_stat, bin_stat_config, is_bin_stat_transform
from .significance_runtime import (
    is_significance_transform,
    significance,
    significance_config,
)
from .density_cell_runtime import (
    density_cell,
    density_cell_config,
    is_density_cell_transform,
)
from .interp_2d_runtime import (
    interp_2d_config,
    is_interp_2d_transform,
    make_interp_2d,
)
from .posterior_density_runtime import (
    is_posterior_density_transform,
    posterior_density,
    posterior_density_config,
)
from .profile_runtime import _preprofiling, eval_series, profiling


def _normalize_column_list(spec):
    if spec is None:
        return []
    if isinstance(spec, dict):
        for key in ("columns", "keep", "drop", "select", "retain", "value"):
            if key in spec:
                return _normalize_column_list(spec.get(key))
        return []
    if isinstance(spec, str):
        text = spec.strip()
        return [text] if text else []
    if isinstance(spec, (list, tuple, set)):
        out = []
        for item in spec:
            sval = str(item).strip()
            if sval:
                out.append(sval)
        return out
    sval = str(spec).strip()
    return [sval] if sval else []


#: Steps doctor/dryrun may execute. Everything else is heavy or unknown and is
#: skipped in the check phase (mesh/profile only on ``jplot <yaml>``).
LIGHT_TRANSFORM_KEYS = frozenset(
    {"filter", "sortby", "add_column", "keep_columns", "drop_columns"}
)

HEAVY_TRANSFORM_KEYS = frozenset(
    {
        "profile",
        "make_density_core",
        "posterior_density",
        "make_interp_2d",
        "type",
        # Not heavy to compute, but cross-block: what a ``to_df`` name resolves
        # to, and the fact that a producing block hands the layer nothing, only
        # exist once the whole layer runs.  dryrun models one block at a time,
        # so it must report itself blind here rather than guess a row count.
        "to_df",
        "to_ds",
        # Reads tables other blocks published, so it is cross-block for the same
        # reason to_df is.
        "significance",
    }
)


class _NullLogger:
    def debug(self, *a, **k):
        return None

    def info(self, *a, **k):
        return None

    def warning(self, *a, **k):
        return None

    def error(self, *a, **k):
        return None


def apply_light_transforms(df, transform, logger=None):
    """Run only light pipeline steps; skip heavy mesh/profile steps.

    Shared by dryrun/doctor (check phase) so light logic is not forked from
    the render runtime helpers. Returns ``(df, step_records, heavy_skipped)``
    where each step_record is ``{name, detail, rows_in, rows_out}``.
    """
    log = logger if logger is not None else _NullLogger()
    if transform is None:
        return df, [], []
    if not isinstance(transform, list):
        try:
            log.warning(f"Illegal transform format, list required -> {transform}")
        except Exception:
            pass
        return df, [], []

    steps: list[dict] = []
    heavy_skipped: list[str] = []
    work = df
    for step in transform:
        if not isinstance(step, Mapping):
            continue
        try:
            rows_in = int(len(work))
        except Exception:
            rows_in = 0
        name = next(iter(step.keys()), "unknown")
        detail = ""
        try:
            if "duplicate" in step:
                detail = str(step.get("duplicate"))
                if step.get("duplicate") is not False:
                    try:
                        work = work.copy(deep=False)
                    except Exception:
                        pass
            elif "filter" in step:
                detail = str(step.get("filter"))
                work = filter_df(work, step["filter"], log)
            elif "sortby" in step:
                detail = str(step.get("sortby"))
                work = sort_by(work, step["sortby"], log)
            elif "add_column" in step:
                detail = str((step.get("add_column") or {}).get("name", ""))
                work = add_column(work, step["add_column"], log)
            elif "keep_columns" in step:
                detail = str(step.get("keep_columns"))
                work = keep_columns(work, step.get("keep_columns"), log)
            elif "drop_columns" in step:
                detail = str(step.get("drop_columns"))
                work = drop_columns(work, step.get("drop_columns"), log)
            elif is_bin_stat_transform(step):
                cfg = bin_stat_config(step)
                detail = str(cfg.get("x", ""))
                work = bin_stat(work, cfg, log)
            elif any(k in step for k in HEAVY_TRANSFORM_KEYS) or (
                "type" in step
                and str(step.get("type") or "")
                in {
                    "make_density_core",
                    "posterior_density",
                    "make_interp_2d",
                }
            ):
                heavy_name = str(name)
                heavy_skipped.append(heavy_name)
                steps.append(
                    {
                        "name": heavy_name,
                        "detail": "skipped in check phase (heavy step)",
                        "rows_in": rows_in,
                        "rows_out": rows_in,
                    }
                )
                continue
            else:
                steps.append(
                    {
                        "name": str(name),
                        "detail": "unknown step skipped",
                        "rows_in": rows_in,
                        "rows_out": rows_in,
                    }
                )
                continue
        except Exception as exc:
            steps.append(
                {
                    "name": str(name),
                    "detail": f"failed: {exc}",
                    "rows_in": rows_in,
                    "rows_out": 0,
                }
            )
            continue
        try:
            rows_out = int(len(work))
        except Exception:
            rows_out = 0
        steps.append(
            {
                "name": str(name),
                "detail": detail,
                "rows_in": rows_in,
                "rows_out": rows_out,
            }
        )
    return work, steps, heavy_skipped


def filter_df(df, condition, logger):
    try:
        if isinstance(condition, bool):
            return df.copy(deep=False) if condition else df.iloc[0:0].copy()
        if isinstance(condition, (int, float)) and condition in (0, 1):
            return df.copy(deep=False) if int(condition) == 1 else df.iloc[0:0].copy()

        if isinstance(condition, str):
            s = condition.strip()
            low = s.lower()
            if low in {"true", "t", "yes", "y"}:
                return df.copy(deep=False)
            if low in {"false", "f", "no", "n"}:
                return df.iloc[0:0].copy()
            s = s.replace("&&", " & ").replace("||", " | ")
            condition = s
        else:
            raise TypeError(f"Unsupported condition type: {type(condition)}")

        mask = eval_dataframe_expression(df, condition, logger=logger, allow_column=True)

        if isinstance(mask, (bool, np.bool_, int, float)):
            return df.copy(deep=False) if bool(mask) else df.iloc[0:0].copy()
        if not isinstance(mask, pd.Series):
            mask = pd.Series(mask, index=df.index)
        mask = mask.astype(bool)
        if bool(mask.all()):
            return df.copy(deep=False)
        return df[mask].copy()
    except Exception as e:
        logger.error(f"Errors when evaluating condition -> {condition}:\n\t{e}")
        return pd.DataFrame(index=df.index).iloc[0:0].copy()


def add_column(df, adds, logger):
    try:
        name = adds.get("name", False)
        expr = adds.get("expr", False)
        if not (name and expr):
            logger.error("Error in loading add_column -> {}".format(adds))
        value = eval_dataframe_expression(df, expr, logger=logger, allow_column=True)
        df[name] = value
        return df
    except Exception as e:
        logger.error(
            "Errors when add new column -> {}:\n\t{}: {}".format(
                adds, e.__class__.__name__, e
            )
        )
        return df


def keep_columns(df, spec, logger):
    try:
        cols = _normalize_column_list(spec)
        if not cols:
            return df
        if isinstance(df, pd.DataFrame):
            keep = [c for c in cols if c in df.columns]
            missing = [c for c in cols if c not in df.columns]
            if missing and logger:
                logger.warning(f"keep_columns missing columns ignored -> {missing}")
            if not keep:
                return df.iloc[0:0].copy()
            return df.loc[:, keep].copy(deep=False)
        if hasattr(df, "select"):
            try:
                return df.select(cols)
            except Exception:
                return df
        return df
    except Exception as e:
        if logger:
            logger.warning(f"keep_columns failed for spec={spec}: {e}")
        return df


def drop_columns(df, spec, logger):
    try:
        cols = _normalize_column_list(spec)
        if not cols:
            return df
        if isinstance(df, pd.DataFrame):
            existing = [c for c in cols if c in df.columns]
            if not existing:
                return df
            return df.drop(columns=existing, errors="ignore")
        if hasattr(df, "drop"):
            try:
                return df.drop(cols)
            except Exception:
                return df
        return df
    except Exception as e:
        if logger:
            logger.warning(f"drop_columns failed for spec={spec}: {e}")
        return df


def sort_by(df, expr, logger):
    try:
        return sort_df_by_expr(df, expr, logger=logger)
    except Exception as e:
        logger.warning(f"sort_by failed for expr={expr}: {e}")
        return df


def sort_df_by_expr(df: pd.DataFrame, expr: str, logger) -> pd.DataFrame:
    """
    Sort the dataframe by evaluating the given expression.
    The expression can be a column name or a valid expression understood by eval_series.
    Returns a new DataFrame sorted ascending by the evaluated values.
    """
    if df is None or expr is None:
        return df
    try:
        key = str(expr).strip()
        if key in df.columns:
            values = np.asarray(df[key].to_numpy(copy=False))
            if values.ndim != 1 or values.shape[0] != int(df.shape[0]):
                raise ValueError(
                    "sort key length mismatch: "
                    f"rows={int(df.shape[0])}, key_shape={getattr(values, 'shape', None)}"
                )
            order = np.argsort(values, kind="quicksort")
            return df.iloc[order]

        values = np.asarray(eval_series(df, {"expr": expr}, logger))
        if values.ndim != 1 or values.shape[0] != int(df.shape[0]):
            raise ValueError(
                "sort expression output length mismatch: "
                f"rows={int(df.shape[0])}, values={getattr(values, 'shape', None)}"
            )
        order = np.argsort(values, kind="quicksort")
        return df.iloc[order]
    except Exception as e:
        logger.warning(f"LB: sort_by failed for expr={expr}: {e}")
        return df


def _csv_export_target(target: Any) -> Any:
    if isinstance(target, Mapping):
        return target.get("path", target.get("file", target.get("target", target.get("value", ""))))
    return target


def _resolve_csv_export_path(preprocessor, target: Any) -> Path:
    raw = _csv_export_target(target)
    path = str(raw).strip()
    if not path:
        raise ValueError("to_csv requires a non-empty path")
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
        published = preprocessor.named_table(source)
        if published is not None:
            return published
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

        if "duplicate" in trans:
            # Detach from a shared table before anything downstream edits it.
            if trans.get("duplicate") is not False:
                df = preprocessor._clone_df(df)
        elif "filter" in trans:
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
        elif is_significance_transform(trans):
            before_rows = preprocessor._safe_nrows(df)
            df = significance(
                df,
                significance_config(trans),
                preprocessor.logger,
                tables=preprocessor._named_tables,
            )
            preprocessor._debug(
                "significance:\n\t source \t-> {}\n\t rows \t\t-> {} -> {}".format(
                    source_label or "<unknown>",
                    before_rows if before_rows is not None else "NA",
                    preprocessor._safe_nrows(df),
                )
            )
        elif is_bin_stat_transform(trans):
            before_rows = preprocessor._safe_nrows(df)
            df = bin_stat(df, bin_stat_config(trans), preprocessor.logger)
            preprocessor._debug(
                "bin_stat:\n\t source \t-> {}\n\t rows \t\t-> {} -> {}".format(
                    source_label or "<unknown>",
                    before_rows if before_rows is not None else "NA",
                    preprocessor._safe_nrows(df),
                )
            )
        elif is_density_cell_transform(trans):
            cfg = dict(density_cell_config(trans))
            cfg.setdefault("_base_dir", getattr(preprocessor, "base_dir", None))
            before_rows = preprocessor._safe_nrows(df)
            preprocessor._info(
                "Density cell START:\n\t source \t-> {}\n\t rows_before \t-> {}".format(
                    source_label or "<unknown>",
                    before_rows if before_rows is not None else "NA",
                )
            )
            df = density_cell(df, cfg, preprocessor.logger)
            after_rows = preprocessor._safe_nrows(df)
            preprocessor._warn(
                "Density cell DONE:\n\t source \t-> {}\n\t rows_after \t-> {}".format(
                    source_label or "<unknown>",
                    after_rows if after_rows is not None else "NA",
                )
            )
        elif is_posterior_density_transform(trans):
            cfg = dict(posterior_density_config(trans))
            cfg.setdefault("_base_dir", getattr(preprocessor, "base_dir", None))
            before_rows = preprocessor._safe_nrows(df)
            preprocessor._info(
                "Posterior density START:\n\t source \t-> {}\n\t rows_before \t-> {}".format(
                    source_label or "<unknown>",
                    before_rows if before_rows is not None else "NA",
                )
            )
            df = posterior_density(df, cfg, preprocessor.logger)
            after_rows = preprocessor._safe_nrows(df)
            preprocessor._warn(
                "Posterior density DONE:\n\t source \t-> {}\n\t rows_after \t-> {}".format(
                    source_label or "<unknown>",
                    after_rows if after_rows is not None else "NA",
                )
            )
        elif is_interp_2d_transform(trans):
            cfg = interp_2d_config(trans)
            before_rows = preprocessor._safe_nrows(df)
            preprocessor._info(
                "2D interpolation START:\n\t source \t-> {}\n\t rows_before \t-> {}".format(
                    source_label or "<unknown>",
                    before_rows if before_rows is not None else "NA",
                )
            )
            df = make_interp_2d(df, cfg, preprocessor.logger)
            after_rows = preprocessor._safe_nrows(df)
            preprocessor._warn(
                "2D interpolation DONE:\n\t source \t-> {}\n\t rows_after \t-> {}".format(
                    source_label or "<unknown>",
                    after_rows if after_rows is not None else "NA",
                )
            )
        elif "sortby" in trans:
            df = sort_by(df, trans["sortby"], preprocessor.logger)
        elif "add_column" in trans:
            df = add_column(df, trans["add_column"], preprocessor.logger)
        elif "keep_columns" in trans:
            df = keep_columns(df, trans.get("keep_columns"), preprocessor.logger)
        elif "drop_columns" in trans:
            df = drop_columns(df, trans.get("drop_columns"), preprocessor.logger)
        elif "to_csv" in trans:
            _save_dataframe_csv(
                preprocessor,
                df,
                trans.get("to_csv"),
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

def transform_publish_target(transform: Any) -> Optional[Dict[str, Any]]:
    """The ``to_df`` / ``to_ds`` step that ends this chain, if there is one.

    Both are terminal by construction: they name what the finished table *is*,
    so allowing them mid-chain would only invite a half-built table to be
    published under a settled name.  A misplaced one raises rather than being
    silently skipped, unlike the rest of the if/elif chain.
    """
    if not isinstance(transform, list) or not transform:
        return None
    target = None
    for index, step in enumerate(transform):
        if not isinstance(step, Mapping):
            continue
        last = index == len(transform) - 1
        if "to_df" in step:
            if not last:
                raise ValueError("transform step 'to_df' must be the last step of its block")
            spec = step.get("to_df")
            if isinstance(spec, Mapping):
                name, keep = spec.get("name"), bool(spec.get("keep", False))
            else:
                name, keep = spec, False
            if not isinstance(name, str) or not name.strip():
                raise ValueError("transform step 'to_df' needs a table name")
            target = {"kind": "to_df", "name": name.strip(), "keep": keep}
        elif "to_ds" in step:
            if not last:
                raise ValueError("transform step 'to_ds' must be the last step of its block")
            if step.get("to_ds") is False:
                continue
            target = {"kind": "to_ds", "name": None, "keep": True}
    return target


def finish_pipeline(preprocessor, work, source, transform, key, hit):
    """Register whatever this block publishes, then decide what it hands back."""
    target = transform_publish_target(transform)
    if target is None:
        return work, key, hit

    name = target["name"] or (source if isinstance(source, str) else None)
    if not name:
        preprocessor._warn("to_ds needs the data block to name a single source; skipped.")
        return work, key, hit

    signature = preprocessor._stable_hash(
        {
            "schema": "jp-named-table-v1",
            "source": preprocessor._source_token(source),
            "transform": transform,
        }
    )
    if target["kind"] == "to_ds":
        # The finished table takes over the block's own (in-memory) dataset, and
        # the scratch tables that built it go away.
        for scoped in preprocessor.scoped_table_names():
            if scoped != name:
                preprocessor.drop_named_table(scoped)
        preprocessor._named_tables[str(name)] = work
        preprocessor._named_table_signatures[str(name)] = signature
        preprocessor.scope_table(str(name))
        preprocessor._debug(f"to_ds stored the finished table under -> {name}")
        return work, key, hit

    preprocessor.publish_named_table(name, work, signature)
    if target["keep"]:
        return work, key, hit
    # A producer block draws nothing: load_layer_data skips a None result, so the
    # concat downstream sees only the blocks that actually carry rows.
    return None, key, hit


def apply_transforms(preprocessor, df, transform: Optional[Sequence[Mapping[str, Any]]]):
    """Prebuild pass: execute profile step as lightweight _preprofiling."""
    return apply_transforms_impl(preprocessor, df, transform, profile_mode="preprofile")

def apply_runtime_transforms(preprocessor,
    df,
    transform: Optional[Sequence[Mapping[str, Any]]],
    source_label: Optional[str] = None,
):
    """Runtime pass: keep original profiling behavior."""
    return apply_transforms_impl(preprocessor, df, transform, profile_mode="runtime", source_label=source_label)

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
                emit_source_summary(preprocessor, source)
                preprocessor._debug(f"Pipeline cache HIT -> {key}")
                memtrace_checkpoint(
                    preprocessor.logger,
                    "pipeline.cache_hit",
                    cached,
                    extra={"source": preprocessor._runtime_source_label(source), "mode": mode},
                )
                cached = preprocessor._enrich_for_demand(cached, source, demand_columns)
                # A cache hit skips the transform loop, so publishing has to
                # happen here as well -- otherwise the second run of a config
                # would leave the named table undefined.
                return finish_pipeline(
                    preprocessor,
                    preprocessor._clone_df(cached),
                    source,
                    effective_transform,
                    key,
                    True,
                )
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

    raw = resolve_source_data(preprocessor, source, combine=combine)
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
    return finish_pipeline(preprocessor, work, source, effective_transform, key, False)

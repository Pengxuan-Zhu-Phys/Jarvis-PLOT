from __future__ import annotations

import pandas as pd

try:
    import polars as pl
except Exception:
    pl = None

from ..memtrace import memtrace_checkpoint, memtrace_object_inventory


def polars_to_pandas(frame, logger=None, stage: str = "dataset"):
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
            if logger:
                try:
                    logger.warning(
                        f"pyarrow unavailable during polars->pandas conversion; using dict fallback -> {stage}"
                    )
                except Exception:
                    pass
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

#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional
import os
import time
import resource

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional at import time
    pd = None

try:
    import polars as pl
except Exception:  # pragma: no cover - optional at import time
    pl = None

try:
    import numpy as np
except Exception:  # pragma: no cover - optional at import time
    np = None


_STATE = {
    "pid": None,
    "seq": 0,
    "last_rss_mb": None,
}


def memtrace_enabled() -> bool:
    flag = str(os.getenv("JP_MEM_TRACE", "")).strip().lower()
    return flag in {"1", "true", "yes", "on", "debug"}


def _emit(logger, msg: str) -> None:
    if logger is not None:
        for meth in ("warning", "info", "debug"):
            fn = getattr(logger, meth, None)
            if callable(fn):
                try:
                    fn(msg)
                    return
                except Exception:
                    continue
    print(msg)


def _current_rss_mb() -> Optional[float]:
    try:
        import psutil  # type: ignore

        rss = int(psutil.Process(os.getpid()).memory_info().rss)
        return float(rss) / (1024.0 * 1024.0)
    except Exception:
        pass

    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        raw = float(ru.ru_maxrss)
        # macOS reports bytes, Linux reports KB.
        if raw > (1024.0 * 1024.0 * 64.0):
            return raw / (1024.0 * 1024.0)
        return raw / 1024.0
    except Exception:
        pass

    return None


def _shape_token(obj: Any) -> Mapping[str, Any]:
    if obj is None:
        return {}

    if pd is not None and isinstance(obj, pd.DataFrame):
        return {
            "backend": "pandas",
            "rows": int(obj.shape[0]),
            "cols": int(obj.shape[1]),
        }

    if pl is not None and isinstance(obj, pl.DataFrame):
        return {
            "backend": "polars_df",
            "rows": int(obj.height),
            "cols": int(obj.width),
        }

    if pl is not None and isinstance(obj, pl.LazyFrame):
        cols = None
        try:
            cols = len(obj.collect_schema())
        except Exception:
            cols = None
        payload = {"backend": "polars_lazy", "rows": "lazy"}
        if cols is not None:
            payload["cols"] = int(cols)
        return payload

    if isinstance(obj, dict):
        return {"backend": "dict", "items": len(obj)}

    if isinstance(obj, (list, tuple)):
        return {"backend": type(obj).__name__.lower(), "items": len(obj)}

    return {"backend": type(obj).__name__}


def _human_bytes(size: Any) -> str:
    try:
        val = float(size)
    except Exception:
        return str(size)
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while val >= 1024.0 and idx < len(units) - 1:
        val /= 1024.0
        idx += 1
    return f"{val:.2f}{units[idx]}"


def _estimate_bytes(obj: Any) -> Optional[int]:
    if obj is None:
        return None
    try:
        if pd is not None and isinstance(obj, pd.DataFrame):
            return int(obj.memory_usage(index=True, deep=True).sum())
    except Exception:
        pass
    try:
        if pl is not None and isinstance(obj, pl.DataFrame):
            return int(obj.estimated_size())
    except Exception:
        pass
    try:
        if np is not None and isinstance(obj, np.ndarray):
            return int(obj.nbytes)
    except Exception:
        pass
    return None


def memtrace_checkpoint(
    logger,
    stage: str,
    obj: Any = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    if not memtrace_enabled():
        return

    pid = os.getpid()
    if _STATE.get("pid") != pid:
        _STATE["pid"] = pid
        _STATE["seq"] = 0
        _STATE["last_rss_mb"] = None

    rss_mb = _current_rss_mb()
    last = _STATE.get("last_rss_mb")
    delta_mb = None
    if rss_mb is not None and isinstance(last, float):
        delta_mb = rss_mb - last

    _STATE["seq"] = int(_STATE.get("seq", 0)) + 1
    if rss_mb is not None:
        _STATE["last_rss_mb"] = rss_mb

    parts = [
        f"[MEMTRACE] seq={_STATE['seq']}",
        f"ts={time.time():.3f}",
        f"stage={stage}",
    ]
    if rss_mb is not None:
        parts.append(f"rss_mb={rss_mb:.2f}")
    else:
        parts.append("rss_mb=NA")
    if delta_mb is not None:
        parts.append(f"delta_mb={delta_mb:+.2f}")

    info = dict(_shape_token(obj))
    if extra:
        for k, v in extra.items():
            if v is None:
                continue
            if str(k).lower().endswith("bytes") or str(k).lower().endswith("size"):
                info[k] = _human_bytes(v)
            else:
                info[k] = v

    for k in sorted(info.keys()):
        parts.append(f"{k}={info[k]}")

    _emit(logger, " ".join(parts))


def memtrace_object_inventory(
    logger,
    stage: str,
    objects: Mapping[str, Any],
    roles: Optional[Mapping[str, str]] = None,
    min_bytes: int = 64 * 1024 * 1024,
) -> None:
    if not memtrace_enabled():
        return
    if not isinstance(objects, Mapping):
        return
    role_map = roles if isinstance(roles, Mapping) else {}
    for name, obj in objects.items():
        if obj is None:
            continue
        est = _estimate_bytes(obj)
        if est is not None and int(est) < int(min_bytes):
            continue
        info = {
            "var": str(name),
            "type": type(obj).__name__,
        }
        if est is not None:
            info["estimated_size"] = int(est)
        role = role_map.get(name)
        if role:
            info["role"] = str(role)
        memtrace_checkpoint(logger, stage, obj=obj, extra=info)


def memtrace_file_checkpoint(logger, stage: str, path: Any, extra: Optional[Mapping[str, Any]] = None) -> None:
    if not memtrace_enabled():
        return
    size = None
    try:
        size = int(Path(path).expanduser().resolve().stat().st_size)
    except Exception:
        size = None
    payload = dict(extra or {})
    if size is not None:
        payload["file_size"] = size
    payload["file_path"] = str(path)
    memtrace_checkpoint(logger, stage, obj=None, extra=payload)

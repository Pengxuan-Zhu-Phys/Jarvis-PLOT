from __future__ import annotations

from typing import Any, Mapping, Optional
import math

import numpy as np
import pandas as pd

from ..inner_func import update_funcs


def build_eval_globals(extra: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Build the shared eval globals used by dataframe-expression helpers."""
    allowed = update_funcs({"np": np, "math": math})
    allowed["__builtins__"] = {}
    if extra:
        allowed.update(dict(extra))
    return allowed


def _coerce_result(result: Any, fillna: Any = None) -> np.ndarray:
    arr = np.asarray(result)
    if fillna is None:
        return arr

    try:
        if np.issubdtype(arr.dtype, np.number):
            mask = np.isnan(arr)
        else:
            mask = pd.isna(arr)
        if np.asarray(mask).any():
            arr = np.where(mask, fillna, arr)
    except Exception:
        pass
    return np.asarray(arr)


def eval_dataframe_expression(
    df: pd.DataFrame,
    expr: Any,
    logger=None,
    *,
    fillna: Any = None,
    allow_column: bool = True,
) -> np.ndarray:
    """Evaluate a column name or trusted YAML expression against a dataframe."""
    if expr is None:
        raise ValueError("expr must not be None")

    text = str(expr).strip()
    if not text:
        raise ValueError("expr must not be empty")

    if allow_column and text in df.columns:
        arr = df[text].to_numpy(copy=False)
    else:
        if logger:
            try:
                logger.debug(f"Loading variable expression -> {text}")
            except Exception:
                pass
        local_vars = df.to_dict("series")
        allowed_globals = build_eval_globals()
        arr = eval(text, allowed_globals, local_vars)

    return _coerce_result(arr, fillna=fillna)

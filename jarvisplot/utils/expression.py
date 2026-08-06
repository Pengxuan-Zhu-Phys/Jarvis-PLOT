from __future__ import annotations

from typing import Any, Mapping, Optional
import math

import numpy as np
import pandas as pd

from ..expr_names import EXPR_IDENTIFIER_IGNORE
from ..inner_func import update_funcs

__all__ = [
    "EXPR_IDENTIFIER_IGNORE",
    "build_eval_globals",
    "eval_dataframe_expression",
    "eval_scalar_expression",
]


def build_eval_globals(extra: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Build the shared eval globals used by dataframe-expression helpers."""
    allowed = update_funcs({"np": np, "math": math})
    allowed.update(
        {
            "exp": np.exp,
            "log": np.log,
            "ln": np.log,
            "log10": np.log10,
            "sqrt": np.sqrt,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "abs": np.abs,
        }
    )
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


def eval_scalar_expression(
    expr: Any,
    local_vars: Optional[Mapping[str, Any]] = None,
    logger=None,
    *,
    extra_globals: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Evaluate a trusted scalar expression with the shared eval globals.

    This is the non-dataframe companion to :func:`eval_dataframe_expression`.
    Callers such as layer ``clip_expr`` must not open a second bare ``eval``
    surface; route through here so globals and the trusted-input assumption stay
    centralized.

    Trusted-input assumption: expression text comes from project YAML / style
    cards controlled by the local user, not from untrusted remote payloads.
    """
    if expr is None:
        raise ValueError("expr must not be None")

    text = str(expr).strip()
    if not text:
        raise ValueError("expr must not be empty")

    if logger:
        try:
            logger.debug(f"Evaluating scalar expression -> {text}")
        except Exception:
            pass

    allowed_globals = build_eval_globals(extra_globals)
    return eval(text, allowed_globals, dict(local_vars or {}))


def eval_dataframe_expression(
    df: pd.DataFrame,
    expr: Any,
    logger=None,
    *,
    fillna: Any = None,
    allow_column: bool = True,
) -> np.ndarray:
    """Evaluate a column name or trusted YAML expression against a dataframe.

    Trusted-input assumption: expression text comes from project YAML / style
    cards controlled by the local user, not from untrusted remote payloads.
    """
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

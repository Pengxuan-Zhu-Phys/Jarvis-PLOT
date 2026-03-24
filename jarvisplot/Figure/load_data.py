#!/usr/bin/env python3 

import numpy as np 
import pandas as pd 
from ..utils.expression import eval_dataframe_expression
from .profile_runtime import eval_series, grid_profiling, profiling, _preprofiling


def filter(df, condition, logger):
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

def addcolumn(df, adds, logger):
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
        
def sortby(df, expr, logger):
    try:
        return sort_df_by_expr(df, expr, logger=logger)
    except Exception as e:
        logger.warning(f"sortby failed for expr={expr}: {e}")
        return df

def sort_df_by_expr(df: pd.DataFrame, expr: str, logger) -> pd.DataFrame:
    """
    Sort the dataframe by evaluating the given expression.
    The expression can be a column name or a valid expression understood by _eval_series.
    Returns a new DataFrame sorted ascending by the evaluated values.
    """
    if df is None or expr is None:
        return df
    try:
        key = str(expr).strip()
        # Fast path: direct column sort via positional argsort to avoid
        # pandas block-manager heavy sort/copy behavior on wide tables.
        if key in df.columns:
            values = np.asarray(df[key].to_numpy(copy=False))
            if values.ndim != 1 or values.shape[0] != int(df.shape[0]):
                raise ValueError(
                    "sort key length mismatch: "
                    f"rows={int(df.shape[0])}, key_shape={getattr(values, 'shape', None)}"
                )
            order = np.argsort(values, kind="quicksort")
            return df.iloc[order]

        # Expression path: evaluate once and reorder by positional argsort,
        # avoiding assignment of a temporary "__sortkey__" column.
        values = np.asarray(eval_series(df, {"expr": expr}, logger))
        if values.ndim != 1 or values.shape[0] != int(df.shape[0]):
            raise ValueError(
                "sort expression output length mismatch: "
                f"rows={int(df.shape[0])}, values={getattr(values, 'shape', None)}"
            )
        order = np.argsort(values, kind="quicksort")
        return df.iloc[order]
    except Exception as e:
        logger.warning(f"LB: sortby failed for expr={expr}: {e}")
        return df

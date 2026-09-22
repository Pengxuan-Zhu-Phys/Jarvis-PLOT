"""Declarative, in-memory data generation for virtual plotting tables.

``generate`` is deliberately a small, inspectable vocabulary rather than a
second Python surface in YAML.  Expressions use Jarvis-PLOT's established
trusted-config expression runtime, and can reference columns produced earlier
in the same ``columns`` mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .utils.expression import eval_scalar_expression

GENERATED_OPERATIONS = ("linspace", "logspace", "arange", "full", "values", "expr")

__all__ = ["GENERATED_OPERATIONS", "generate_dataframe"]


def _as_vector(value: Any, *, column: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr
    if arr.ndim != 1:
        raise ValueError(f"generated column {column!r} must be a scalar or one-dimensional array")
    return arr


def _args(value: Any, *, operation: str, column: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"generated column {column!r}: {operation} expects a YAML list of arguments")
    return list(value)


def _make_column(spec: Any, columns: Mapping[str, np.ndarray], *, name: str) -> np.ndarray:
    """Evaluate one generated-column specification."""
    if not isinstance(spec, Mapping):
        return _as_vector(spec, column=name)

    generators = [key for key in GENERATED_OPERATIONS if key in spec]
    if len(generators) != 1:
        raise ValueError(
            f"generated column {name!r} must declare exactly one of "
            "linspace, logspace, arange, full, values, or expr"
        )
    operation = generators[0]
    value = spec[operation]

    try:
        if operation == "linspace":
            args = _args(value, operation=operation, column=name)
            if len(args) not in {2, 3}:
                raise ValueError("expects [start, stop] or [start, stop, num]")
            return _as_vector(np.linspace(*args), column=name)
        if operation == "logspace":
            args = _args(value, operation=operation, column=name)
            if len(args) not in {2, 3}:
                raise ValueError("expects [start, stop] or [start, stop, num]")
            return _as_vector(np.logspace(*args), column=name)
        if operation == "arange":
            args = _args(value, operation=operation, column=name)
            if not 1 <= len(args) <= 3:
                raise ValueError("expects [stop], [start, stop], or [start, stop, step]")
            return _as_vector(np.arange(*args), column=name)
        if operation == "full":
            args = _args(value, operation=operation, column=name)
            if len(args) != 2:
                raise ValueError("expects [size, value]")
            return _as_vector(np.full(int(args[0]), args[1]), column=name)
        if operation == "values":
            if not isinstance(value, (list, tuple)):
                raise ValueError("expects a YAML list")
            return _as_vector(value, column=name)
        # Generated expressions only see earlier generated columns.  This
        # makes order meaningful and avoids accidental reads from a DataSet.
        if not isinstance(value, str) or not value.strip():
            raise ValueError("expr expects a non-empty string")
        return _as_vector(eval_scalar_expression(value, local_vars=columns), column=name)
    except ValueError as exc:
        message = str(exc)
        if message.startswith("generated column"):
            raise
        raise ValueError(f"generated column {name!r}: {operation} {message}") from exc
    except Exception as exc:
        raise ValueError(f"generated column {name!r}: could not evaluate {operation}: {exc}") from exc


def generate_dataframe(spec: Any) -> pd.DataFrame:
    """Materialise a ``generate`` mapping as a pandas DataFrame.

    Scalar column values broadcast to the length of the first vector.  Empty
    vectors are legal and produce a zero-row table; conflicting non-scalar
    lengths are rejected before pandas can silently align them.
    """
    if not isinstance(spec, Mapping):
        raise ValueError("generate must be a mapping with a columns mapping")
    raw_columns = spec.get("columns")
    if not isinstance(raw_columns, Mapping) or not raw_columns:
        raise ValueError("generate.columns must be a non-empty mapping")

    evaluated: dict[str, np.ndarray] = {}
    vector_length: int | None = None
    for raw_name, column_spec in raw_columns.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("generate.columns has an empty column name")
        if name in evaluated:
            raise ValueError(f"generate.columns declares {name!r} more than once")
        value = _make_column(column_spec, evaluated, name=name)
        if value.ndim == 1:
            size = int(value.size)
            if vector_length is None:
                vector_length = size
            elif size != vector_length:
                raise ValueError(
                    f"generated column {name!r} has {size} rows; expected {vector_length} rows"
                )
        evaluated[name] = value

    nrows = 1 if vector_length is None else vector_length
    materialized: dict[str, np.ndarray] = {}
    for name, value in evaluated.items():
        if value.ndim == 0:
            materialized[name] = np.full(nrows, value.item())
        else:
            materialized[name] = value
    return pd.DataFrame(materialized, copy=False)

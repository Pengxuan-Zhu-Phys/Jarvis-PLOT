#!/usr/bin/env python3
import os
import sys
from types import SimpleNamespace
from sympy.core import numbers as SCNum
import sympy 
import numpy as np

_AllSCNum = (
    SCNum.Float,
    SCNum.Number,
    SCNum.Rational,
    SCNum.Integer,
    SCNum.Infinity,
    SCNum.AlgebraicNumber,
    SCNum.RealNumber,
    SCNum.Zero,
    SCNum.One,
    SCNum.NegativeOne,
    SCNum.NegativeInfinity,
    SCNum.Exp1,
    SCNum.Pi,
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64
)
def _reciprocal(fn):
    """``1 / fn(x)`` for sec / csc / cot and their hyperbolic siblings."""
    def wrapped(x):
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / fn(x)
    return wrapped


def _of_reciprocal(fn):
    """``fn(1 / x)`` for asec / acsc / acot and their hyperbolic siblings."""
    def wrapped(x):
        with np.errstate(divide="ignore", invalid="ignore"):
            return fn(1.0 / np.asarray(x, dtype=float))
    return wrapped


def _sinc(x):
    # sin(x) / x, as sympy writes it; np.sinc is the normalised sin(pi x) / (pi x).
    return np.sinc(np.asarray(x, dtype=float) / np.pi)


def _root(x, n):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.power(x, 1.0 / n)


def _min(*args):
    """Elementwise minimum of several columns; the smallest value of a single one."""
    if len(args) == 1:
        return np.min(np.asarray(args[0]))
    return np.minimum.reduce(np.broadcast_arrays(*args))


def _max(*args):
    """Elementwise maximum of several columns; the largest value of a single one."""
    if len(args) == 1:
        return np.max(np.asarray(args[0]))
    return np.maximum.reduce(np.broadcast_arrays(*args))


def _heaviside(x, h0=0.5):
    return np.heaviside(x, h0)


# Expressions are evaluated over whole columns, so every function here takes a
# pandas Series / numpy array as readily as a scalar.  sympy's versions raise
# "'Series' object has no attribute 'is_Number'" on a column, which is why
# these are numpy rather than ``sympy.<name>``.
_Inner_FCs = {
        # Natural Logarithm
        "log":  np.log,
        "exp":  np.exp,
        "ln":   np.log,
        # Triangle Function
        "sin":  np.sin,
        "cos":  np.cos,
        "tan":  np.tan,
        "sec":  _reciprocal(np.cos),
        "csc":  _reciprocal(np.sin),
        "cot":  _reciprocal(np.tan),
        "sinc": _sinc,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "asec": _of_reciprocal(np.arccos),
        "acsc": _of_reciprocal(np.arcsin),
        "acot": _of_reciprocal(np.arctan),
        "atan2":np.arctan2,
        # Hyperbolic Function 
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "sech": _reciprocal(np.cosh),
        "csch": _reciprocal(np.sinh),
        "coth": _reciprocal(np.tanh),
        "asinh":    np.arcsinh,
        "acosh":    np.arccosh,
        "atanh":    np.arctanh,
        "acoth":    _of_reciprocal(np.arctanh),
        "asech":    _of_reciprocal(np.arccosh),
        "acsch":    _of_reciprocal(np.arcsinh),
        # General Math
        "sqrt": np.sqrt,
        "Min":  _min,
        "Max":  _max,
        "min":  _min,
        "max":  _max,
        "root": _root,
        "Abs":  np.abs,
    }

_Constant = {
    "Pi":   np.pi,
    "pi":   np.pi,
    "E":    np.e,
    "Inf":  np.inf
}

# External function hooks (e.g. user-defined / lazy-loaded interpolators)
# These are injected into the expression runtime via `update_funcs`.
#
# Usage:
# - core/context code can call `set_external_funcs({...})` once after building ctx.
# - or provide a getter with `set_external_funcs_getter(lambda: {...})` if the set may change.
# - values must be callables (LazyCallable is OK).
_EXTERNAL_FCS = {}
_EXTERNAL_FCS_GETTER = None


def set_external_funcs(funcs: dict) -> None:
    """Register external functions to be injected by `update_funcs`.

    Parameters
    ----------
    funcs:
        Mapping of name -> callable (LazyCallable is acceptable).
    """
    global _EXTERNAL_FCS
    _EXTERNAL_FCS = dict(funcs) if funcs is not None else {}


def set_external_funcs_getter(getter) -> None:
    """Register a callable that returns a dict of external functions.

    The getter should return a `dict[str, callable]`.
    """
    global _EXTERNAL_FCS_GETTER
    _EXTERNAL_FCS_GETTER = getter


def clear_external_funcs() -> None:
    """Clear any registered external functions/getter."""
    global _EXTERNAL_FCS, _EXTERNAL_FCS_GETTER
    _EXTERNAL_FCS = {}
    _EXTERNAL_FCS_GETTER = None

def Gauss(xx, mean, err):
    prob = np.exp(-0.5 * ((xx - mean) / err)**2)
    return prob 


# def Gauss(xx, mean, err):
#     from math import sqrt, pi, exp
#     # prob = 1./ (err * sqrt(2 * pi)) * exp(-0.5*((xx - mean)/err)**2)
#     prob = exp(-0.5*((xx - mean)/err)**2)
#     return prob

def Normal(xx, mean, err):
    # from math import sqrt, pi, exp
    prob = 1./ (err * np.sqrt(2 * np.pi)) * np.exp(-0.5*((xx - mean)/err)**2)
    return prob

def LogGauss(xx, mean, err):
    prob = -0.5*((xx - mean)/err)**2
    return prob


def _extract_operas_full_name_map():
    """Build `namespace.function -> callable` map from Jarvis-Operas registry."""
    try:
        from jarvis_operas import build_register_dicts

        return build_register_dicts()
    except Exception:
        return {}


def _build_operas_eval_namespaces(full_name_map):
    """Build `namespace -> SimpleNamespace(function=callable)` for plain eval."""
    grouped = {}
    for full_name, fn in (full_name_map or {}).items():
        if not callable(fn) or "." not in full_name:
            continue
        namespace, short_name = full_name.split(".", 1)
        if not namespace or not short_name:
            continue
        grouped.setdefault(namespace, {})[short_name] = fn
    return {namespace: SimpleNamespace(**attrs) for namespace, attrs in grouped.items()}


def update_funcs(funcs):
    if funcs is None:
        funcs = {}
    funcs['sympy'] = sympy
    funcs['Gauss'] = Gauss
    funcs['LogGauss'] = LogGauss
    funcs['Normal'] = Normal
    funcs['Heaviside'] = _heaviside

    # Built-in functions and constants
    funcs.update(_Inner_FCs)
    funcs.update(_Constant)

    # Jarvis-Operas registered functions.
    # Keep both symbolic and numeric views, and provide namespace-style
    # objects so plain `eval` can call `namespace.func(x)`.
    try:
        from jarvis_operas import build_sympy_dicts

        operas_full_name_map = _extract_operas_full_name_map()
        func_locals, numeric_funcs = build_sympy_dicts(
            operas_full_name_map,
            include_all=True,
        )
        funcs.update(numeric_funcs)
        funcs.update(func_locals)
        funcs.update(operas_full_name_map)
        funcs.update(_build_operas_eval_namespaces(operas_full_name_map))
    except Exception:
        pass

    # External functions (e.g. lazy-loaded interpolators). External takes priority.
    try:
        if _EXTERNAL_FCS_GETTER is not None:
            ext = _EXTERNAL_FCS_GETTER() or {}
            funcs.update(ext)
        elif _EXTERNAL_FCS:
            funcs.update(_EXTERNAL_FCS)
    except Exception:
        # Never fail expression evaluation because external injection failed.
        pass

    return funcs

def update_const(vars):
    vars.update(_Constant)
    return vars

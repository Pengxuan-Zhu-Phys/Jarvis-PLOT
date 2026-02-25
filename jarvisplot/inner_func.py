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
_Inner_FCs = { 
        # Natural Logarithm
        "log":  sympy.log,
        "exp":  sympy.exp,
        "ln":   sympy.ln,
        # Triangle Function
        "sin":  sympy.sin,
        "cos":  sympy.cos,
        "tan":  sympy.tan,
        "sec":  sympy.sec,
        "csc":  sympy.csc,
        "cot":  sympy.cot,
        "sinc": sympy.sinc,
        "asin": sympy.asin,
        "acos": sympy.acos,
        "atan": sympy.atan,
        "asec": sympy.asec,
        "acsc": sympy.acsc,
        "acot": sympy.acot,
        "atan2":sympy.atan2,
        # Hyperbolic Function 
        "sinh": sympy.sinh,
        "cosh": sympy.cosh,
        "tanh": sympy.tanh,
        "sech": sympy.sech,
        "csch": sympy.csch,
        "coth": sympy.coth,
        "asinh":    sympy.asinh,
        "acosh":    sympy.acosh,
        "atanh":    sympy.atanh,
        "acoth":    sympy.acoth,
        "asech":    sympy.asech,
        "acsch":    sympy.acsch,
        # General Math
        "sqrt": sympy.sqrt,
        "Min":  sympy.Min,
        "Max":  sympy.Max,
        "root": sympy.root,
        "Abs":  sympy.Abs,
    }

_Constant = {
    "Pi":   sympy.pi,
    "E":    sympy.E,
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
    prob = sympy.exp(-0.5 * ((xx - mean) / err)**2)
    return prob 


# def Gauss(xx, mean, err):
#     from math import sqrt, pi, exp
#     # prob = 1./ (err * sqrt(2 * pi)) * exp(-0.5*((xx - mean)/err)**2)
#     prob = exp(-0.5*((xx - mean)/err)**2)
#     return prob

def Normal(xx, mean, err):
    # from math import sqrt, pi, exp
    prob = 1./ (err * sympy.sqrt(2 * sympy.pi)) * sympy.exp(-0.5*((xx - mean)/err)**2)
    return prob

def LogGauss(xx, mean, err):
    prob = -0.5*((xx - mean)/err)**2
    return prob


def _extract_operas_full_name_map(func_locals, numeric_funcs):
    """Build `namespace.function -> callable` map from Jarvis-Operas dicts."""
    full_name_map = {}
    for namespace, ns_obj in (func_locals or {}).items():
        if not isinstance(namespace, str):
            continue
        attrs = getattr(ns_obj, "__dict__", None)
        if not isinstance(attrs, dict):
            continue
        for short_name, symbol_fn in attrs.items():
            if not isinstance(short_name, str):
                continue
            symbolic_name = str(symbol_fn)
            fn = numeric_funcs.get(symbolic_name)
            if callable(fn):
                full_name_map[f"{namespace}.{short_name}"] = fn
    return full_name_map


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
    funcs['Heaviside'] = sympy.Heaviside

    # Built-in functions
    funcs.update(_Inner_FCs)

    # Jarvis-Operas registered functions.
    # Keep both symbolic/numeric views for compatibility, and provide
    # namespace-style objects so plain `eval` can call `namespace.func(x)`.
    try:
        from jarvis_operas import func_locals, numeric_funcs

        funcs.update(numeric_funcs)
        funcs.update(func_locals)
        operas_full_name_map = _extract_operas_full_name_map(func_locals, numeric_funcs)
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

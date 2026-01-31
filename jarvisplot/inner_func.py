#!/usr/bin/env python3
import os
import sys
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

def update_funcs(funcs):
    funcs['sympy'] = sympy
    funcs['Gauss'] = Gauss
    funcs['LogGauss'] = LogGauss
    funcs['Normal'] = Normal
    funcs['Heaviside'] = sympy.Heaviside

    # Built-in functions
    funcs.update(_Inner_FCs)

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

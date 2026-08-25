#!/usr/bin/env python3

"""Identifier tokens that are never column names in Jarvis-PLOT expressions.

Stdlib-only on purpose: :mod:`jarvisplot.column_demand` (used by
``jplot validate``) and the eval surface must share one table without forcing
numpy / sympy into the pure-shape validation path.
"""

from __future__ import annotations

import re

__all__ = ["EXPR_IDENTIFIER_IGNORE", "expr_identifiers"]

#: Keep in lockstep with :func:`jarvisplot.utils.expression.build_eval_globals`
#: and :data:`jarvisplot.inner_func._Inner_FCs` / ``_Constant``. Agents type
#: these bare or after ``np.``; treating them as columns invents false
#: ``JP-COL-001`` diagnostics.
EXPR_IDENTIFIER_IGNORE: frozenset[str] = frozenset(
    {
        # namespaces
        "np",
        "math",
        # Python keywords / literals
        "True",
        "False",
        "None",
        "and",
        "or",
        "not",
        "in",
        "if",
        "else",
        "for",
        "lambda",
        # builtins commonly written bare
        "abs",
        "min",
        "max",
        "sum",
        "len",
        "int",
        "float",
        "str",
        "bool",
        "round",
        "pi",
        "e",
        # expression.py aliases
        "exp",
        "log",
        "ln",
        "log10",
        "sqrt",
        "sin",
        "cos",
        "tan",
        # inner_func sympy surface
        "sec",
        "csc",
        "cot",
        "sinc",
        "asin",
        "acos",
        "atan",
        "asec",
        "acsc",
        "acot",
        "atan2",
        "sinh",
        "cosh",
        "tanh",
        "sech",
        "csch",
        "coth",
        "asinh",
        "acosh",
        "atanh",
        "acoth",
        "asech",
        "acsch",
        "Min",
        "Max",
        "root",
        "Abs",
        "Pi",
        "E",
        "Inf",
        # common numpy method tokens after `np.`
        "log2",
        "log1p",
        "expm1",
        "power",
        "clip",
        "where",
        "isnan",
        "isfinite",
        "isinf",
        "sign",
        "floor",
        "ceil",
        "maximum",
        "minimum",
        "mean",
        "median",
        "std",
        "var",
        "percentile",
        "quantile",
        "arange",
        "linspace",
        "ones",
        "zeros",
        "full",
        "array",
        "asarray",
        "nan",
        "inf",
    }
)


#: Quoted spans: ``.transform('sum')`` names a pandas reduction, not a column.
_STRING_LITERAL_RE = re.compile(r"'[^']*'|\"[^\"]*\"")

#: A bare identifier.  The lookbehind rejects two things that are never column
#: references: the tail of a longer identifier, and an attribute name -- the
#: ``digitize`` of ``np.digitize`` or the ``groupby`` of ``weight.groupby``.
#: Without it every method an expression calls has to be denylisted by hand,
#: and one that is not yet listed becomes a phantom ``JP-COL-001``.
_IDENTIFIER_RE = re.compile(r"(?<![\w.])[A-Za-z_][A-Za-z0-9_]*")


def expr_identifiers(text: str) -> set[str]:
    """Return the identifiers in an expression that could name a column.

    Attribute names, quoted literals and the :data:`EXPR_IDENTIFIER_IGNORE`
    table are all dropped, so what comes back is only what the caller should
    look for among the data source's real columns.
    """
    stripped = _STRING_LITERAL_RE.sub(" ", str(text))
    return {
        t for t in _IDENTIFIER_RE.findall(stripped)
        if t not in EXPR_IDENTIFIER_IGNORE
    }

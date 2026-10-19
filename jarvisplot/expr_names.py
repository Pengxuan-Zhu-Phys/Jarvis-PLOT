#!/usr/bin/env python3

"""Identifier tokens that are never column names in Jarvis-PLOT expressions.

Stdlib-only on purpose: :mod:`jarvisplot.column_demand` (used by
``jplot validate``) and the eval surface must share one table without forcing
numpy / sympy into the pure-shape validation path.
"""

from __future__ import annotations

import re

__all__ = ["EXPR_IDENTIFIER_IGNORE", "expr_identifiers", "expr_namespace_names"]

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
        "Gauss",
        "LogGauss",
        "Normal",
        "Heaviside",
        "sympy",
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


#: An identifier, and whether an attribute access (``.name``) follows it.
_IDENTIFIER_OR_HEAD_RE = re.compile(r"(?<![\w.])([A-Za-z_][A-Za-z0-9_]*)(\s*\.\s*[A-Za-z_])?")


def expr_namespace_names(texts) -> set[str]:
    """Names that only ever head an attribute access, across all ``texts``.

    ``pdg`` in ``pdg.alphaEM`` is a Jarvis-Operas namespace, not a column; the
    registered namespaces cannot be listed without importing Operas, which
    pulls in the render stack this module exists to avoid. A name used both as
    ``w.groupby(...)`` and as a bare ``w`` is a column and stays out of the
    result. A column that is only ever used as ``name.method()`` and is missing
    from the data is the price: the check cannot tell it from a namespace, and
    the render reports it instead.
    """
    heads: set[str] = set()
    bare: set[str] = set()
    for text in texts:
        stripped = _STRING_LITERAL_RE.sub(" ", str(text))
        for name, attribute in _IDENTIFIER_OR_HEAD_RE.findall(stripped):
            (heads if attribute else bare).add(name)
    return heads - bare

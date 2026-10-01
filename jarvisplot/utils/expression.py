"""One expression language for every table expression in a config.

``filter`` / ``add_column`` / ``sortby`` and every ``{expr: ...}`` are Python
expressions evaluated over whole columns. Python's own operators are the
syntax, with three changes that make them mean the same thing per row as they
do on scalars:

* ``and`` / ``or`` / ``not`` -- and their spellings ``&&`` / ``||`` -- are
  elementwise. Being boolean operators they bind *looser* than comparisons,
  so ``x > 1 && y > 1`` needs no parentheses.
* A chained comparison ``0 < x < 1`` is elementwise too.
* ``&`` / ``|`` / ``~`` keep Python's meaning. They bind *tighter* than a
  comparison, so ``x > 1 & y > 1`` is ``x > (1 & y) > 1``; that spelling is
  reported once as a likely mistake.

The text is parsed once with :mod:`ast` (cached per expression) and compiled.
The pandas pipeline evaluates the compiled code directly; the polars pushdown
hands the very same code the column arrays through ``map_batches``. Both
engines therefore share one parser and one evaluator, so a filter cannot keep
different rows depending on which engine happened to run it.
"""

from __future__ import annotations

import ast
import io
import tokenize
import warnings
from dataclasses import dataclass
from functools import lru_cache
from types import CodeType
from typing import Any, Mapping, Optional, Sequence
import math

import numpy as np
import pandas as pd

from ..expr_names import EXPR_IDENTIFIER_IGNORE
from ..inner_func import update_funcs

__all__ = [
    "CompiledExpression",
    "EXPR_IDENTIFIER_IGNORE",
    "build_eval_globals",
    "compile_dataframe_expression",
    "eval_dataframe_expression",
    "eval_scalar_expression",
    "polars_expression",
]

_AND = "__jp_and__"
_OR = "__jp_or__"
_NOT = "__jp_not__"
_LOGIC_GLOBALS = {_AND: np.logical_and, _OR: np.logical_or, _NOT: np.logical_not}

_PRECEDENCE_HINT = (
    "`&` and `|` bind tighter than comparisons, so {segment!r} is evaluated as "
    "{reading!r}. To combine conditions write `(a > 1) & (b > 1)`, "
    "`a > 1 and b > 1` or `a > 1 && b > 1`."
)
_precedence_reported: set[str] = set()


@dataclass(frozen=True)
class CompiledExpression:
    """A parsed table expression: compiled code plus the names it reads."""

    text: str
    code: CodeType
    names: frozenset
    precedence_hint: Optional[str] = None


def _normalize_logic_tokens(text: str) -> str:
    """Spell ``&&`` / ``||`` as ``and`` / ``or``, leaving string literals alone."""
    if "&&" not in text and "||" not in text:
        return text
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError):
        return text

    line_starts = [0]
    for line in text.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))

    def _offset(pos) -> int:
        return line_starts[pos[0] - 1] + pos[1]

    spans = []
    previous = None
    for tok in tokens:
        if (
            previous is not None
            and tok.type == tokenize.OP
            and tok.string in ("&", "|")
            and previous.type == tokenize.OP
            and previous.string == tok.string
            and previous.end == tok.start
        ):
            word = " and " if tok.string == "&" else " or "
            spans.append((_offset(previous.start), _offset(tok.end), word))
            previous = None
            continue
        previous = tok

    for start, end, word in reversed(spans):
        text = text[:start] + word + text[end:]
    return text


def _source_around(source: str, node: ast.AST) -> tuple[str, str]:
    """Source text before and after ``node`` (ast offsets are UTF-8 bytes)."""
    lines = source.splitlines(keepends=True) or [""]
    first = lines[node.lineno - 1].encode("utf-8")
    last = lines[node.end_lineno - 1].encode("utf-8")
    before = "".join(lines[: node.lineno - 1]) + first[: node.col_offset].decode("utf-8", "replace")
    after = last[node.end_col_offset :].decode("utf-8", "replace") + "".join(lines[node.end_lineno :])
    return before, after


def _precedence_hint(source: str, tree: ast.AST) -> Optional[str]:
    """Flag an unparenthesised ``&`` / ``|`` used as a comparison operand."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for operand in (node.left, *node.comparators):
            if not (isinstance(operand, ast.BinOp) and isinstance(operand.op, (ast.BitAnd, ast.BitOr))):
                continue
            before, after = _source_around(source, operand)
            if before.rstrip().endswith("(") and after.lstrip().startswith(")"):
                continue
            segment = ast.get_source_segment(source, node) or source
            inner = ast.get_source_segment(source, operand) or ""
            reading = segment.replace(inner, f"({inner})", 1) if inner else segment
            return _PRECEDENCE_HINT.format(segment=segment, reading=reading)
    return None


class _ElementwiseLogic(ast.NodeTransformer):
    """Rewrite Python's scalar-only logic into its elementwise equivalent."""

    @staticmethod
    def _call(name: str, *args: ast.expr) -> ast.Call:
        return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=list(args), keywords=[])

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        name = _AND if isinstance(node.op, ast.And) else _OR
        result = node.values[0]
        for value in node.values[1:]:
            result = self._call(name, result, value)
        return ast.copy_location(result, node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.copy_location(self._call(_NOT, node.operand), node)
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) == 1:
            return node
        operands = [node.left, *node.comparators]
        result = None
        for index, op in enumerate(node.ops):
            pair = ast.Compare(left=operands[index], ops=[op], comparators=[operands[index + 1]])
            result = pair if result is None else self._call(_AND, result, pair)
        return ast.copy_location(result, node)


@lru_cache(maxsize=1024)
def compile_dataframe_expression(text: str) -> CompiledExpression:
    """Parse and compile one table expression (cached per text).

    Raises :class:`SyntaxError` for text that is not a Python expression once
    ``&&`` / ``||`` are spelled out.
    """
    source = _normalize_logic_tokens(str(text).strip())
    tree = ast.parse(source, mode="eval")
    names = frozenset(node.id for node in ast.walk(tree) if isinstance(node, ast.Name))
    hint = _precedence_hint(source, tree)
    tree = ast.fix_missing_locations(_ElementwiseLogic().visit(tree))
    code = compile(tree, "<jarvisplot expression>", "eval")
    return CompiledExpression(text=str(text).strip(), code=code, names=names, precedence_hint=hint)


def _report_precedence(compiled: CompiledExpression, logger) -> None:
    if not compiled.precedence_hint or compiled.text in _precedence_reported:
        return
    _precedence_reported.add(compiled.text)
    if logger:
        try:
            logger.warning(f"Expression {compiled.text!r}: {compiled.precedence_hint}")
        except Exception:
            pass


def _expression_locals(df: pd.DataFrame, names) -> dict[str, Any]:
    """Only the columns the expression reads, as Series."""
    if not df.columns.is_unique:
        return df.to_dict("series")
    return {name: df[name] for name in names if name in df.columns}


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
    allowed.update(_LOGIC_GLOBALS)
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
        compiled = compile_dataframe_expression(text)
        _report_precedence(compiled, logger)
        local_vars = _expression_locals(df, compiled.names)
        arr = eval(compiled.code, build_eval_globals(), local_vars)

    return _coerce_result(arr, fillna=fillna)


# --------------------------------------------------------------------------- #
# polars pushdown: the same compiled code, fed by map_batches
# --------------------------------------------------------------------------- #


def _numpy_dtype_for(pl_dtype) -> np.dtype:
    import polars as pl

    return pl.Series("probe", [], dtype=pl_dtype).to_numpy().dtype


def _polars_dtype_for(np_dtype):
    """polars dtype of a numpy result, or ``None`` when there is no faithful one."""
    import polars as pl

    np_dtype = np.dtype(np_dtype)
    if np_dtype.kind not in "biuf":
        return None
    return pl.Series("probe", np.empty(0, dtype=np_dtype)).dtype


def _probe_return_dtype(compiled: CompiledExpression, used: Sequence[str], schema: Mapping[str, Any], allowed_globals):
    """Result dtype from one row of ones; numpy's dtype rules do not depend on values."""
    sample = {}
    for name in used:
        np_dtype = _numpy_dtype_for(schema[name])
        sample[name] = pd.Series(np.ones(1, dtype=np_dtype), name=name)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = eval(compiled.code, allowed_globals, sample)
    return np.asarray(result).dtype


def polars_expression(
    expr: Any,
    schema: Mapping[str, Any],
    *,
    as_mask: bool = False,
    logger=None,
):
    """A polars expression that evaluates ``expr`` with the pandas evaluator.

    ``schema`` maps the frame's column names to polars dtypes. The returned
    expression feeds the referenced columns, as pandas Series, to the very same
    compiled code :func:`eval_dataframe_expression` runs, so polars and pandas
    cannot disagree about what an expression means (NaN comparisons included:
    polars orders NaN above every number, numpy compares it as False).

    ``as_mask`` builds a filter mask, cast to bool the same way ``filter`` does
    on pandas. Raises when the expression cannot be pushed down faithfully --
    an unknown name, a result without a numeric/bool dtype -- so the caller can
    fall back to the pandas path, which then reports the error itself.
    """
    import polars as pl

    text = str(expr).strip()
    if not text:
        raise ValueError("expr must not be empty")
    if text in schema:
        if not as_mask:
            return pl.col(text)
        # numpy's truthiness, not polars' cast: NaN counts as True on both paths.
        return pl.map_batches(
            [pl.col(text)],
            lambda series: pl.Series(series[0].to_numpy().astype(bool)),
            return_dtype=pl.Boolean,
        )

    compiled = compile_dataframe_expression(text)
    _report_precedence(compiled, logger)
    allowed_globals = build_eval_globals()
    used = sorted(name for name in compiled.names if name in schema)
    unknown = sorted(name for name in compiled.names if name not in schema and name not in allowed_globals)
    if unknown:
        raise NameError(f"name {unknown[0]!r} is not defined")

    if not used:
        value = np.asarray(eval(compiled.code, allowed_globals, {}))
        if value.ndim != 0:
            raise ValueError("an expression that reads no column must be a scalar")
        value = bool(value) if as_mask else value.item()
        return pl.lit(value)

    if as_mask:
        return_dtype = pl.Boolean
    else:
        return_dtype = _polars_dtype_for(_probe_return_dtype(compiled, used, schema, allowed_globals))
        if return_dtype is None:
            raise TypeError("expression result has no numeric or boolean dtype")

    def _evaluate(series):
        local_vars = {
            name: pd.Series(values.to_numpy(), name=name)
            for name, values in zip(used, series)
        }
        rows = len(series[0])
        out = np.asarray(eval(compiled.code, allowed_globals, local_vars))
        if out.ndim == 0:
            out = np.full(rows, out.item())
        if as_mask:
            out = out.astype(bool)
        result = pl.Series(out)
        if result.dtype != return_dtype:
            raise TypeError(
                f"expression {text!r} produced {result.dtype}, expected {return_dtype}"
            )
        return result

    return pl.map_batches([pl.col(name) for name in used], _evaluate, return_dtype=return_dtype)

#!/usr/bin/env python3

"""Combine a binned signal table and a binned background table into Z per bin.

The step takes two tables that ``bin_stat`` already produced -- one per class,
each published under its own name -- and aligns them on the bin index.  Signal
and background therefore keep living in separate tables, which is how they
usually arrive, and the layer that draws the result only ever sees two columns
of numbers.

Which figure of merit is meant is a key, not a convention: ``S/sqrt(S+B)`` and
the Asimov significance disagree by tens of percent in the regime where these
plots are read, and a background uncertainty changes the answer again.  The
default is the one the event-level method it replaces used, so a config can be
ported without changing the curve.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from ..utils.expression import eval_dataframe_expression

__all__ = [
    "is_significance_transform",
    "significance_config",
    "significance",
    "significance_source_names",
    "significance_output_columns",
    "FORMULAS",
]

#: ``*_syst`` variants need the matching ``sigma`` on the background side.
FORMULAS = (
    "s_over_sqrt_sb",
    "s_over_sqrt_b",
    "s_over_sqrt_b_syst",
    "asimov",
    "asimov_syst",
)

_CARRY = ("x_lo", "x_hi", "x_center")


def is_significance_transform(step: Any) -> bool:
    if not isinstance(step, Mapping):
        return False
    if "significance" in step:
        return True
    return str(step.get("type", "")).strip().lower() == "significance"


def significance_config(step: Mapping[str, Any]) -> dict:
    if "significance" in step:
        cfg = step.get("significance")
        return dict(cfg) if isinstance(cfg, Mapping) else {}
    cfg = dict(step)
    cfg.pop("type", None)
    return cfg


def _side(cfg: Mapping[str, Any], key: str, value_name: str, sigma_name: str) -> dict:
    spec = cfg.get(key)
    if not isinstance(spec, Mapping):
        raise ValueError(f"significance needs a '{key}' block")
    source = spec.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"significance {key} needs a 'source' table name")
    return {
        "source": source.strip(),
        "value": spec.get(value_name, value_name),
        "sigma": spec.get(sigma_name),
        "value_name": value_name,
        "sigma_name": sigma_name,
    }


def _sides(cfg: Mapping[str, Any]) -> tuple:
    return (
        _side(cfg, "dfS", "Sn", "sigmaSn"),
        _side(cfg, "dfB", "Bn", "sigmaBn"),
    )


def significance_source_names(cfg: Mapping[str, Any]) -> list:
    """Tables this step reads that its own data block never names.

    The cache has to fold these in, or a change upstream of one of them would
    leave this step's key untouched and a stale result would be served.
    """
    out = []
    for key in ("dfS", "dfB"):
        spec = cfg.get(key)
        if isinstance(spec, Mapping):
            source = spec.get("source")
            if isinstance(source, str) and source.strip():
                out.append(source.strip())
    return out


def significance_output_columns(cfg: Mapping[str, Any]) -> set:
    out = {"bin_index", "zn", *_CARRY}
    for key, value_name, sigma_name in (("dfS", "Sn", "sigmaSn"), ("dfB", "Bn", "sigmaBn")):
        out.add(value_name)
        spec = cfg.get(key)
        if isinstance(spec, Mapping) and spec.get(sigma_name) is not None:
            out.add(sigma_name)
    return out


def _column(table: pd.DataFrame, expr: Any, what: str, logger=None) -> np.ndarray:
    values = eval_dataframe_expression(table, expr, logger=logger, allow_column=True)
    return np.asarray(values, dtype=float).reshape(-1)


def _z(name: str, S, B, sigma_B) -> np.ndarray:
    """Z per bin, with invalid bins left as NaN rather than dropped."""
    S = np.asarray(S, dtype=float)
    B = np.asarray(B, dtype=float)
    out = np.full(S.shape, np.nan, dtype=float)

    if name == "s_over_sqrt_sb":
        total = S + B
        ok = (B > 0) & (total > 0)
        out[ok] = S[ok] / np.sqrt(total[ok])
        return out

    if name == "s_over_sqrt_b":
        ok = B > 0
        out[ok] = S[ok] / np.sqrt(B[ok])
        return out

    if name == "s_over_sqrt_b_syst":
        var = B + np.asarray(sigma_B, dtype=float) ** 2
        ok = var > 0
        out[ok] = S[ok] / np.sqrt(var[ok])
        return out

    if name == "asimov":
        ok = (B > 0) & (S + B > 0)
        s, b = S[ok], B[ok]
        out[ok] = np.sqrt(np.maximum(2.0 * ((s + b) * np.log1p(s / b) - s), 0.0))
        return out

    if name == "asimov_syst":
        # Cowan, Cranmer, Gross & Vitells (2011), the discovery significance
        # with a background uncertainty.  Reduces to `asimov` as sigma -> 0,
        # which the tests check rather than take on trust.
        sig = np.asarray(sigma_B, dtype=float)
        var = sig ** 2
        ok = (B > 0) & (S + B > 0) & (var > 0)
        s, b, v = S[ok], B[ok], var[ok]
        first = (s + b) * np.log(((s + b) * (b + v)) / (b * b + (s + b) * v))
        second = (b * b / v) * np.log1p((v * s) / (b * (b + v)))
        out[ok] = np.sqrt(np.maximum(2.0 * (first - second), 0.0))
        # Where the uncertainty vanishes the systematic form is undefined but
        # the limit is not, so fall back rather than emitting a hole.
        plain = (B > 0) & (S + B > 0) & ~(var > 0)
        if plain.any():
            s0, b0 = S[plain], B[plain]
            out[plain] = np.sqrt(np.maximum(2.0 * ((s0 + b0) * np.log1p(s0 / b0) - s0), 0.0))
        return out

    raise ValueError(f"significance formula must be one of {', '.join(FORMULAS)}; got {name!r}")


def significance(df, cfg: Mapping[str, Any], logger=None, tables=None) -> pd.DataFrame:
    """Align the two binned tables and evaluate the figure of merit."""
    sig_side, bkg_side = _sides(cfg)
    key = str(cfg.get("key", "bin_index")).strip() or "bin_index"
    formula = str(cfg.get("formula", "s_over_sqrt_sb")).strip() or "s_over_sqrt_sb"
    if formula not in FORMULAS:
        raise ValueError(
            f"significance formula must be one of {', '.join(FORMULAS)}; got {formula!r}"
        )

    lookup = tables or {}
    frames = {}
    for side in (sig_side, bkg_side):
        table = lookup.get(side["source"])
        if table is None:
            raise ValueError(
                f"significance cannot find table {side['source']!r}; "
                "an earlier data block must publish it with to_df"
            )
        if key not in getattr(table, "columns", ()):
            raise ValueError(
                f"significance table {side['source']!r} has no key column {key!r}"
            )
        frames[side["source"]] = table

    sig_tbl = frames[sig_side["source"]]
    bkg_tbl = frames[bkg_side["source"]]

    # Align on the integer bin index: two independent np.histogram calls agree
    # bit-for-bit on their edges, but keying on a float centre would still be
    # betting on that.
    sig_keys = np.asarray(sig_tbl[key])
    bkg_keys = np.asarray(bkg_tbl[key])
    common = np.intersect1d(sig_keys, bkg_keys)
    if common.size == 0:
        raise ValueError(
            f"significance found no shared {key!r} between "
            f"{sig_side['source']!r} and {bkg_side['source']!r}; "
            "the two bin_stat steps must use the same bins"
        )
    sig_rows = sig_tbl.set_index(key).loc[common]
    bkg_rows = bkg_tbl.set_index(key).loc[common]

    S = _column(sig_rows, sig_side["value"], "Sn", logger=logger)
    B = _column(bkg_rows, bkg_side["value"], "Bn", logger=logger)
    sigma_S = (
        _column(sig_rows, sig_side["sigma"], "sigmaSn", logger=logger)
        if sig_side["sigma"] is not None
        else None
    )
    sigma_B = (
        _column(bkg_rows, bkg_side["sigma"], "sigmaBn", logger=logger)
        if bkg_side["sigma"] is not None
        else None
    )
    if formula.endswith("_syst") and sigma_B is None:
        raise ValueError(f"significance formula {formula!r} needs dfB.sigmaBn")

    if bool(cfg.get("cumulative", False)):
        # Z above each threshold rather than inside each bin: the per-bin form
        # is a shape diagnostic, the cumulative one is what a cut would buy.
        S = np.cumsum(S[::-1])[::-1]
        B = np.cumsum(B[::-1])[::-1]
        if sigma_S is not None:
            sigma_S = np.sqrt(np.cumsum((sigma_S[::-1]) ** 2)[::-1])
        if sigma_B is not None:
            sigma_B = np.sqrt(np.cumsum((sigma_B[::-1]) ** 2)[::-1])

    out = pd.DataFrame({key: common, "Sn": S, "Bn": B})
    if sigma_S is not None:
        out["sigmaSn"] = sigma_S
    if sigma_B is not None:
        out["sigmaBn"] = sigma_B
    for column in _CARRY:
        if column in sig_rows.columns:
            out[column] = np.asarray(sig_rows[column])
        elif column in bkg_rows.columns:
            out[column] = np.asarray(bkg_rows[column])
    out["zn"] = _z(formula, S, B, sigma_B)

    if bool(cfg.get("drop_invalid", False)):
        out = out[np.isfinite(out["zn"])].reset_index(drop=True)

    if logger:
        try:
            logger.debug(
                "significance -> {} bins, formula={}, cumulative={}, finite Z in {} bins".format(
                    len(out), formula, bool(cfg.get("cumulative", False)),
                    int(np.isfinite(out["zn"]).sum()),
                )
            )
        except Exception:
            pass
    return out

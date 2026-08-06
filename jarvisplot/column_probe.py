#!/usr/bin/env python3

"""Read the *names* of a data file's columns, and nothing else.

Everything here is a header/metadata read: no row is ever materialized, so
``jplot validate`` can answer "is there a column called ``aa``?" for a 200k-row
scan without paying for it.

Imports are deliberately function-local. Loading this module costs nothing;
only a config that actually declares a CSV pulls in pandas.

Precision over coverage: a spurious "column not found" would teach an agent to
ignore the check, so any format or naming scheme this module cannot resolve
unambiguously reports ``supported=False`` instead of guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = ["ColumnProbe", "probe_dataset_columns"]

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class ColumnProbe:
    """What one data source offers, as far as names go."""

    names: set[str] = field(default_factory=set)
    supported: bool = True
    #: Set when the file exists but could not be read; validation reports it as
    #: a warning rather than pretending the column check succeeded.
    error: str | None = None
    #: Why the check was skipped, when ``supported`` is False.
    reason: str | None = None

    @classmethod
    def unsupported(cls, reason: str) -> "ColumnProbe":
        return cls(supported=False, reason=reason)

    def resolves(self, symbol: str) -> bool:
        """True if ``symbol`` could name -- or be part of -- a real column.

        The demand analyser tokenises expressions with an identifier regex, so a
        column called ``pVa.E`` arrives as the two symbols ``pVa`` and ``E``.
        Accepting fragments of real column names keeps that from becoming a
        confident, wrong "no such column".
        """
        return symbol in self.names or symbol in self._atoms

    @property
    def _atoms(self) -> set[str]:
        if self.__dict__.get("_atom_cache") is None:
            self.__dict__["_atom_cache"] = {
                token for name in self.names for token in _IDENTIFIER.findall(name)
            }
        return self.__dict__["_atom_cache"]


def probe_dataset_columns(entry: Mapping[str, Any], resolved_path: str) -> ColumnProbe:
    """Names an expression can legally use against this ``DataSet`` entry."""
    kind = str(entry.get("type", "")).strip().lower()
    if kind == "csv":
        return _probe_csv(resolved_path)
    if kind == "parquet":
        return _probe_parquet(resolved_path)
    if kind == "hdf5":
        return _probe_hdf5(entry, resolved_path)
    return ColumnProbe.unsupported(f"unknown dataset type {kind!r}")


def _probe_csv(path: str) -> ColumnProbe:
    try:
        import pandas as pd

        head = pd.read_csv(path, nrows=0)
    except Exception as exc:
        return ColumnProbe(error=f"could not read CSV header: {exc}")
    return ColumnProbe(names={str(c) for c in head.columns})


def _probe_parquet(path: str) -> ColumnProbe:
    try:
        import pyarrow.parquet as pq

        return ColumnProbe(names=set(pq.ParquetFile(path).schema.names))
    except Exception:
        pass
    try:
        import pandas as pd

        return ColumnProbe(names={str(c) for c in pd.read_parquet(path).columns})
    except Exception as exc:
        return ColumnProbe(error=f"could not read parquet schema: {exc}")


def _probe_hdf5(entry: Mapping[str, Any], path: str) -> ColumnProbe:
    """Leaf paths plus every alias the loader would accept for them.

    HDF5 columns are addressable several ways -- full leaf path, path relative to
    the declared group, and any ``columns.rename[].target`` alias -- so the probe
    reports the union. A name matching none of them cannot resolve at render time
    either.
    """
    try:
        from .data_loader_hdf5 import scan_hdf5_leaf_metadata

        leaves = scan_hdf5_leaf_metadata(path, entry.get("dataset"))
    except Exception as exc:
        return ColumnProbe(error=f"could not scan HDF5 tree: {exc}")

    group = str(entry.get("dataset") or "").strip()
    names: set[str] = set()
    for leaf in leaves:
        leaf_path = str(leaf.get("path", "")).strip()
        if not leaf_path:
            continue
        names.add(leaf_path)
        if group:
            prefix = f"{group}/"
            names.add(leaf_path[len(prefix):] if leaf_path.startswith(prefix) else prefix + leaf_path)
        names.add(leaf_path.rsplit("/", 1)[-1])

    columns = entry.get("columns")
    if isinstance(columns, Mapping):
        for item in columns.get("rename") or ():
            if isinstance(item, Mapping):
                target = str(item.get("target", "")).strip()
                if target:
                    names.add(target)

    if not names:
        return ColumnProbe.unsupported("HDF5 file exposed no leaf datasets")
    return ColumnProbe(names=names)

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _default_hdf5_path() -> Path:
    return Path(__file__).resolve().parent.parent / "Workshop" / "datas" / "MSSM7.hdf5"


def _leaf_datasets(hobj, prefix: str = ""):
    for key, value in hobj.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, h5py.Dataset):
            yield path, value
        elif isinstance(value, h5py.Group):
            yield from _leaf_datasets(value, path)


def _find_dataset(h5f: h5py.File, token: str):
    token = str(token).strip()
    if not token:
        raise ValueError("column token must be non-empty")

    exact_matches = []
    fuzzy_matches = []
    low_token = token.lower()

    for path, ds in _leaf_datasets(h5f):
        base = path.rsplit("/", 1)[-1]
        if path == token or base == token:
            exact_matches.append((path, ds))
        elif low_token in path.lower() or low_token in base.lower():
            fuzzy_matches.append((path, ds))

    if exact_matches:
        return exact_matches[0], fuzzy_matches
    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0], []
    if not fuzzy_matches:
        raise RuntimeError(f"No HDF5 leaf dataset matched token '{token}'.")
    return None, fuzzy_matches


def _find_isvalid_dataset(h5f: h5py.File, path: str):
    parent, leaf = path.rsplit("/", 1) if "/" in path else ("", path)
    candidates = []
    sibling_names = [
        f"{leaf}_isvalid",
        leaf.replace("LogLike", "LogLike_isvalid"),
        leaf.replace("LogL", "LogL_isvalid"),
    ]
    for candidate in sibling_names:
        full_path = f"{parent}/{candidate}" if parent else candidate
        obj = h5f.get(full_path, None)
        if isinstance(obj, h5py.Dataset):
            candidates.append((full_path, obj))
    if candidates:
        return candidates[0]
    return None, None


def _as_1d_float_array(ds: h5py.Dataset) -> np.ndarray:
    arr = np.asarray(ds[()]).reshape(-1)
    if arr.size == 0:
        return arr.astype(np.float64, copy=False)
    return arr.astype(np.float64, copy=False)


def _as_1d_bool_array(ds: h5py.Dataset) -> np.ndarray:
    arr = np.asarray(ds[()]).reshape(-1)
    if arr.size == 0:
        return arr.astype(bool, copy=False)
    return arr.astype(bool, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect and sort the MSSM7 HDF5 log-likelihood column in descending order."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(_default_hdf5_path()),
        help="Path to MSSM7.hdf5 (default: Workshop/datas/MSSM7.hdf5)",
    )
    parser.add_argument(
        "--column",
        default="LogLike",
        help="Leaf dataset name or path token to inspect (default: LogLike)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Number of top values to print in descending order (default: 50)",
    )
    parser.add_argument(
        "--full-sort",
        action="store_true",
        help="Sort the full column in descending order before printing.",
    )
    args = parser.parse_args()

    hdf5_path = Path(args.path).expanduser().resolve()
    if not hdf5_path.exists():
        raise SystemExit(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as h5f:
        found, fuzzy_matches = _find_dataset(h5f, args.column)
        if found is None:
            print(f"Matched multiple HDF5 leaves for token '{args.column}':")
            for path, ds in fuzzy_matches:
                print(f"  - {path} shape={getattr(ds, 'shape', None)} dtype={getattr(ds, 'dtype', None)}")
            return 2

        path, ds = found
        values = _as_1d_float_array(ds)
        if values.size == 0:
            print(f"{path}: empty dataset")
            return 0
        row_idx = np.arange(values.size, dtype=np.int64)

        isvalid_path, isvalid_ds = _find_isvalid_dataset(h5f, path)
        if isvalid_ds is not None:
            mask = _as_1d_bool_array(isvalid_ds)
            if mask.size != values.size:
                print(
                    f"Warning: {isvalid_path} has {mask.size} rows, expected {values.size}; ignoring is-valid mask."
                )
                mask = None
            else:
                row_idx = row_idx[mask]
                values = values[mask]
                if values.size == 0:
                    print(f"{path}: no valid rows after applying {isvalid_path}")
                    return 0

        top_n = max(int(args.top), 1)
        if args.full_sort or top_n >= values.size:
            order = np.argsort(values)[::-1]
        else:
            top_idx = np.argpartition(values, -top_n)[-top_n:]
            order = top_idx[np.argsort(values[top_idx])[::-1]]

        sorted_values = values[order]
        print(f"File: {hdf5_path}")
        print(f"Dataset: {path}")
        if isvalid_ds is not None:
            print(f"is-valid: {isvalid_path}")
        print(f"Rows: {values.size}")
        print(f"Min: {np.nanmin(values):.6g}")
        print(f"Max: {np.nanmax(values):.6g}")
        print(f"Top {min(top_n, values.size)} values (descending):")
        for idx, value in zip(row_idx[order[:top_n]], sorted_values[:top_n]):
            print(f"{idx:>12}  {value:.12g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

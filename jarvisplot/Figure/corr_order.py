#!/usr/bin/env python3

"""Variable ordering for the correlation matrix -- R's ``corrMatOrder``.

Ordering looks like a drawing option and is not one.  The position a variable
takes is what ``x_index`` counts, what the tick labels are written against and
where ``addrect`` puts its boxes, and in Jarvis-PLOT the tick labels are
resolved *before* the figure is built.  So the order has to be settled at the
same moment: once at config time, written into the transform's ``columns``,
and then simply obeyed by everything downstream.  Nothing reorders at render.

That is why this module takes a square matrix and returns names, and imports
neither matplotlib nor the Figure runtime.  Given the same matrix it gives the
same answer, which is the only property that keeps a label attached to the
cell it names.

The four data-dependent orders reproduce ``corrplot::corrMatOrder``:

``AOE``
    Angular order of the first two eigenvectors -- variables placed by the
    angle of their loading, so correlated groups come out adjacent.
``FPC``
    First principal component; a plain 1D version of the same idea.
``hclust``
    Hierarchical clustering on ``1 - rho``, in dendrogram leaf order.  This is
    the one ``addrect`` needs, because only a tree defines blocks to box.
``alphabet``
    Sorted by name.  Not data-dependent, but it belongs with the others.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "ORDERS",
    "HCLUST_METHODS",
    "order_columns",
]

#: The vocabulary R uses, case-insensitive here because YAML is not R.
ORDERS = ("original", "AOE", "FPC", "hclust", "alphabet")

#: scipy's linkage methods, plus the R spellings people arrive with.
HCLUST_METHODS = {
    "single": "single",
    "complete": "complete",
    "average": "average",
    "weighted": "weighted",
    "mcquitty": "weighted",
    "centroid": "centroid",
    "median": "median",
    "ward": "ward",
    "ward.d": "ward",
    "ward.d2": "ward",
}


def _square(matrix, names: Sequence[str]) -> np.ndarray:
    """The matrix as a clean symmetric array, with holes closed.

    A column that never varies correlates with nothing and arrives as a row of
    NaN.  Treating those as zero keeps one degenerate variable from taking the
    whole ordering down with it; it lands wherever "uncorrelated with
    everything" lands, which is the honest place for it.
    """
    arr = np.asarray(matrix.loc[list(names), list(names)], dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = 0.5 * (arr + arr.T)
    np.fill_diagonal(arr, 1.0)
    return np.clip(arr, -1.0, 1.0)


def _leading_vectors(arr: np.ndarray, count: int) -> np.ndarray:
    """The ``count`` eigenvectors of the largest eigenvalues, sign-fixed.

    An eigenvector is only defined up to sign, and LAPACK is free to hand back
    either one.  Since the sign here decides which end of the figure a group of
    variables lands on, it is pinned: the largest-magnitude entry is made
    positive, so the same matrix always produces the same picture.
    """
    values, vectors = np.linalg.eigh(arr)          # ascending eigenvalues
    lead = vectors[:, np.argsort(values)[::-1][:count]]
    for column in range(lead.shape[1]):
        vector = lead[:, column]
        if vector[int(np.argmax(np.abs(vector)))] < 0:
            lead[:, column] = -vector
    return lead


def _aoe(arr: np.ndarray) -> np.ndarray:
    lead = _leading_vectors(arr, 2)
    e1, e2 = lead[:, 0], lead[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        angle = np.arctan(e2 / e1)
    # e1 == 0 is a right angle, not a missing one.
    angle = np.where(np.isfinite(angle), angle, np.sign(e2) * np.pi / 2.0)
    angle = np.where(e1 > 0, angle, angle + np.pi)
    return np.argsort(angle, kind="stable")


def _fpc(arr: np.ndarray) -> np.ndarray:
    return np.argsort(_leading_vectors(arr, 1)[:, 0], kind="stable")


def _linkage(arr: np.ndarray, method: str):
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    distance = 1.0 - arr
    np.fill_diagonal(distance, 0.0)
    distance[distance < 0.0] = 0.0
    return linkage(squareform(distance, checks=False), method=method)


def _leaves(tree) -> np.ndarray:
    from scipy.cluster.hierarchy import leaves_list

    return np.asarray(leaves_list(tree), dtype=int)


def _blocks(tree, leaves: np.ndarray, k: int) -> list[list[int]]:
    """Inclusive ``[start, end]`` index ranges for ``addrect`` boxes.

    Cut from the *same* linkage the order came from -- passed in rather than
    rebuilt -- so a block is always a run of adjacent positions, which is the
    only reason a rectangle can enclose one.
    """
    from scipy.cluster.hierarchy import fcluster

    k = max(1, min(int(k), len(leaves)))
    labels = np.asarray(fcluster(tree, k, criterion="maxclust"), dtype=int)
    ordered = labels[leaves]
    out: list[list[int]] = []
    start = 0
    for pos in range(1, len(ordered) + 1):
        if pos == len(ordered) or ordered[pos] != ordered[start]:
            out.append([start, pos - 1])
            start = pos
    return out


def order_columns(
    matrix,
    names: Sequence[str],
    order: Any = "original",
    *,
    hclust_method: str = "complete",
    addrect: Any = None,
) -> tuple[list[str], list[list[int]] | None]:
    """Return ``(ordered names, addrect blocks)``.

    ``blocks`` is ``None`` unless ``addrect`` asked for boxes, and boxes are
    only defined under ``hclust``: they are cuts of that tree, so under any
    other order there is nothing to cut.  Raises on an unknown order rather
    than falling back, because a silently ignored ``order`` produces a figure
    that is wrong in exactly the way nobody checks for.
    """
    names = [str(name) for name in names]
    key = str(order or "original").strip()
    match = {value.lower(): value for value in ORDERS}.get(key.lower())
    if match is None:
        raise ValueError(
            "corrplot order must be one of {}; got {!r}".format(", ".join(ORDERS), order)
        )

    # Kept from the ordering pass so `addrect` cuts the tree the order was read
    # off, rather than a second one built from the same matrix: identical today,
    # and one fewer thing that has to stay identical tomorrow.
    tree = None
    index = None
    if match == "original":
        chosen = list(names)
    elif match == "alphabet":
        chosen = sorted(names)
    else:
        arr = _square(matrix, names)
        if match == "AOE":
            index = _aoe(arr)
        elif match == "FPC":
            index = _fpc(arr)
        else:
            method = HCLUST_METHODS.get(str(hclust_method).strip().lower())
            if method is None:
                raise ValueError(
                    "corrplot hclust.method must be one of {}; got {!r}".format(
                        ", ".join(sorted(HCLUST_METHODS)), hclust_method
                    )
                )
            tree = _linkage(arr, method)
            index = _leaves(tree)
        chosen = [names[i] for i in index]

    blocks = None
    if addrect is not None:
        try:
            k = int(addrect)
        except (TypeError, ValueError):
            raise ValueError(f"corrplot addrect must be an integer; got {addrect!r}")
        if k > 0:
            if match != "hclust":
                raise ValueError(
                    "corrplot addrect draws the boxes cut from the hclust tree, so "
                    "it needs order: hclust (got order: {}). Drop addrect, or "
                    "cluster the matrix.".format(match)
                )
            blocks = _blocks(tree, index, k)

    return chosen, blocks

#!/usr/bin/env python3

"""The DataSet ``type`` vocabulary, with no imports attached.

``column_probe`` promises that importing it costs nothing, so the one fact both
it and the pandas-backed loader need lives here rather than in ``data_loader``.
"""

from __future__ import annotations

__all__ = ["IN_MEMORY_DATASET_TYPES", "FILE_DATASET_TYPES", "is_in_memory_type"]

#: Types that declare an empty in-memory table instead of naming a file:
#: ``pd.DataFrame`` is an empty frame, ``pd.Series`` an empty single column.
#: Transforms fill them in (see the ``to_df`` / ``to_ds`` steps).  Stored
#: lower-cased, matching ``DataSet.type``.
IN_MEMORY_DATASET_TYPES = frozenset({"pd.dataframe", "pd.series"})

FILE_DATASET_TYPES = frozenset({"csv", "hdf5", "parquet"})


def is_in_memory_type(value) -> bool:
    """True for a DataSet ``type`` that carries no path, at any casing."""
    return str(value or "").strip().lower() in IN_MEMORY_DATASET_TYPES

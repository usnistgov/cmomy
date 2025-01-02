"""Submodule to work with grouped data."""

from ._factorize import block_by, factor_by, factor_by_to_index
from ._reduction import reduce_data_grouped, reduce_data_indexed, resample_data_indexed

__all__ = [
    "block_by",
    "factor_by",
    "factor_by_to_index",
    "reduce_data_grouped",
    "reduce_data_indexed",
    "resample_data_indexed",
]

"""
Submodule to work with grouped data (:mod:`~cmomy.grouped`)
===========================================================
"""

from ._factorize import block_by, factor_by, factor_by_to_index
from ._reduction import (
    reduce_data_grouped,
    reduce_data_indexed,
    reduce_vals_grouped,
    reduce_vals_indexed,
)

__all__ = [
    "block_by",
    "factor_by",
    "factor_by_to_index",
    "reduce_data_grouped",
    "reduce_data_indexed",
    "reduce_vals_grouped",
    "reduce_vals_indexed",
]

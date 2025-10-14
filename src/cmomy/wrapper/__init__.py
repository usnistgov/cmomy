"""Central moments wrappers."""

from ._constructors import (
    wrap,
    wrap_raw,
    wrap_reduce_vals,
    wrap_resample_vals,
    zeros_like,
)
from ._wrapper import CentralMomentsArray, CentralMomentsData
from .typing import CentralMomentsDataArray, CentralMomentsDataset

#: Alias to :class:`cmomy.CentralMomentsArray`
CentralMoments = CentralMomentsArray
#: Alias to :class:`cmomy.CentralMomentsData`
xCentralMoments = CentralMomentsData  # noqa: N816


__all__ = [
    "CentralMoments",
    "CentralMomentsArray",
    "CentralMomentsData",
    "CentralMomentsDataArray",
    "CentralMomentsDataset",
    "wrap",
    "wrap_raw",
    "wrap_reduce_vals",
    "wrap_resample_vals",
    "xCentralMoments",
    "zeros_like",
]

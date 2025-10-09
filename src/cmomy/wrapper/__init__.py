"""Central moments wrappers."""

from typing import TYPE_CHECKING

from .constructors import (
    wrap,
    wrap_raw,
    wrap_reduce_vals,
    wrap_resample_vals,
    zeros_like,
)
from .wrap_np import CentralMomentsArray
from .wrap_xr import CentralMomentsData

if TYPE_CHECKING:
    import xarray as xr  # noqa: F401

#: Alias to :class:`cmomy.CentralMomentsArray`
CentralMoments = CentralMomentsArray
#: Alias to :class:`cmomy.CentralMomentsData`
xCentralMoments = CentralMomentsData  # noqa: N816

#: :class:`~.CentralMomentsData` wrapping :class:`~xarray.DataArray`
CentralMomentsDataArray = CentralMomentsData["xr.DataArray"]
#: :class:`~.CentralMomentsData` wrapping :class:`~xarray.Dataset`
CentralMomentsDataset = CentralMomentsData["xr.Dataset"]

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

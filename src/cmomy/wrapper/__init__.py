"""Central moments wrappers."""

from .constructors import (
    wrap,
    wrap_raw,
    wrap_reduce_vals,
    wrap_resample_vals,
    zeros_like,
)
from .wrap_np import CentralMomentsArray
from .wrap_xr import CentralMomentsXArray

#: Alias to :class:`cmomy.CentralMomentsArray`
CentralMoments = CentralMomentsArray
#: Alias to :class:`cmomy.CentralMomentsXArray`
xCentralMoments = CentralMomentsXArray  # noqa: N816

__all__ = [
    "CentralMoments",
    "CentralMomentsArray",
    "CentralMomentsXArray",
    "wrap",
    "wrap_raw",
    "wrap_reduce_vals",
    "wrap_resample_vals",
    "xCentralMoments",
    "zeros_like",
]

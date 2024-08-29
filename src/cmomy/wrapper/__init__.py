"""Central moments wrappers."""

from .np_array import CentralMomentsArray
from .xr_array import CentralMomentsXArray

#: Alias to :class:`cmomy.CentralMomentsArray`
CentralMoments = CentralMomentsArray
#: Alias to :class:`cmomy.CentralMomentsXArray`
xCentralMoments = CentralMomentsXArray  # noqa: N816

__all__ = [
    "CentralMoments",
    "CentralMomentsArray",
    "CentralMomentsXArray",
    "xCentralMoments",
]

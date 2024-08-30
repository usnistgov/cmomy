"""Central moments wrappers."""

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
    "xCentralMoments",
]

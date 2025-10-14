"""
Typing aliases (:mod:`cmomy.wrapper.typing`)
============================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import xarray as xr

from ._wrapper import CentralMomentsArray, CentralMomentsData

if TYPE_CHECKING:
    from cmomy.core.typing_compat import TypeAlias

CentralMomentsArrayAny: TypeAlias = CentralMomentsArray[Any]
CentralMomentsDataAny: TypeAlias = CentralMomentsData[Any]

#: :class:`~.CentralMomentsData` wrapping :class:`~xarray.DataArray`
CentralMomentsDataArray: TypeAlias = CentralMomentsData[xr.DataArray]
#: :class:`~.CentralMomentsData` wrapping :class:`~xarray.Dataset`
CentralMomentsDataset: TypeAlias = CentralMomentsData[xr.Dataset]

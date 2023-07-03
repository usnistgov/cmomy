from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr

else:
    from lazy_import import lazy_module

    np = lazy_module("numpy")
    xr = lazy_module("xarray")

__all__ = ["np", "xr"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr

else:
    import lazy_loader as lazy

    np = lazy.load("numpy")
    xr = lazy.load("xarray")

__all__ = ["np", "xr"]

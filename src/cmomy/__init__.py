"""

.. currentmodule:: cmomy


Central (co)moments object creation
===================================

From :class:`numpy.ndarray`
----------------------------
.. autosummary::
    :toctree: generated/

    central_moments
    CentralMoments


From :class:`xarray.DataArray`
------------------------------
.. autosummary::
    :toctree: generated/

    xcentral_moments
    xCentralMoments


Utility modules
===============

.. autosummary::
    :toctree: generated/

    resample
    convert


"""

from .central import CentralMoments, central_moments
from .resample import (
    bootstrap_confidence_interval,
    randsamp_freq,
    xbootstrap_confidence_interval,
)
from .xcentral import xcentral_moments, xCentralMoments

# updated versioning scheme
try:
    from importlib.metadata import version as _version
except ImportError:
    # if the fallback library is missing, we are doomed.
    from importlib_metadata import version as _version  # type: ignore[no-redef]

try:
    __version__ = _version("cmomy")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"


__author__ = """William P. Krekelberg"""
__email__ = "wpk@nist.gov"

__all__ = [
    "CentralMoments",
    "central_moments",
    "xCentralMoments",
    "xcentral_moments",
    "bootstrap_confidence_interval",
    "randsamp_freq",
    "xbootstrap_confidence_interval",
    "__version__",
]

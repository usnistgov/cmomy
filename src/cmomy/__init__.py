"""
Top level API (:mod:`cmomy`)
============================
"""

from .central import CentralMoments, central_moments

# from .resample import (
#     bootstrap_confidence_interval,
#     randsamp_freq,
#     xbootstrap_confidence_interval,
# )
from .xcentral import xcentral_moments, xCentralMoments

# updated versioning scheme
try:
    from ._version import __version__
except Exception:
    __version__ = "999"


__author__ = """William P. Krekelberg"""
__email__ = "wpk@nist.gov"

__all__ = [
    "CentralMoments",
    "central_moments",
    "xCentralMoments",
    "xcentral_moments",
    # "bootstrap_confidence_interval",
    # "randsamp_freq",
    # "xbootstrap_confidence_interval",
    "__version__",
]

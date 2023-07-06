"""
Top level API (:mod:`cmomy`)
============================
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Need this to play nice with IDE/pyright
    from .central import CentralMoments, central_moments  # noqa: TCH004
    from .xcentral import xcentral_moments, xCentralMoments  # noqa: TCH004
else:
    import lazy_loader as lazy

    __getattr__, __dir__, _ = lazy.attach(
        __name__,
        submod_attrs={
            "central": ["CentralMoments", "central_moments"],
            "xcentral": ["xCentralMoments", "xcentral_moments"],
        },
    )

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
    "__version__",
]

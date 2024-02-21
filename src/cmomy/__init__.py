"""
Top level API (:mod:`cmomy`)
============================
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Need this to play nice with IDE/pyright
    from . import convert, random, resample  # noqa: TCH004
    from .central import CentralMoments, central_moments  # noqa: TCH004
    from .xcentral import xcentral_moments, xCentralMoments  # noqa: TCH004
else:
    import lazy_loader as lazy

    __getattr__, __dir__, _ = lazy.attach(
        __name__,
        submodules=[
            "convert",
            "random",
            "resample",
        ],
        submod_attrs={
            "central": ["CentralMoments", "central_moments"],
            "xcentral": ["xCentralMoments", "xcentral_moments"],
        },
    )

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("cmomy")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "999"

__author__ = """William P. Krekelberg"""
__email__ = "wpk@nist.gov"

__all__ = [
    "CentralMoments",
    "__version__",
    "central_moments",
    "convert",
    "random",
    "resample",
    "xCentralMoments",
    "xcentral_moments",
]

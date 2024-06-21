"""Public api for :mod:`cmomy`"""
# Top level API (:mod:`cmomy`)
# ============================
# """

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Need this to play nice with IDE/pyright
    # submodules
    from . import convert, random, reduction, resample  # noqa: TCH004
    from ._central_dataarray import xCentralMoments  # noqa: TCH004
    from ._central_numpy import CentralMoments  # noqa: TCH004
    from .convert import concat  # noqa: TCH004
    from .reduction import reduce_data, reduce_data_grouped, reduce_vals  # noqa: TCH004
    from .resample import (
        indices_to_freq,  # noqa: TCH004
        random_freq,  # noqa: TCH004
        random_indices,  # noqa: TCH004
        randsamp_freq,  # noqa: TCH004
        resample_data,  # noqa: TCH004
        resample_vals,  # noqa: TCH004
    )


else:
    import lazy_loader as lazy

    __getattr__, __dir__, _ = lazy.attach(
        __name__,
        submodules=[
            "random",
            "reduction",
            "resample",
            "convert",
        ],
        submod_attrs={
            "convert": ["concat"],
            "_central_numpy": ["CentralMoments"],
            "_central_dataarray": ["xCentralMoments"],
            "reduction": ["reduce_data", "reduce_data_grouped", "reduce_vals"],
            "resample": [
                "indices_to_freq",
                "random_freq",
                "random_indices",
                "randsamp_freq",
                "resample_data",
                "resample_vals",
            ],
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
    "concat",
    "convert",
    "indices_to_freq",
    "random",
    "random_freq",
    "random_indices",
    "randsamp_freq",
    "reduce_data",
    "reduce_data_grouped",
    "reduce_vals",
    "reduction",
    "resample",
    "resample_data",
    "resample_vals",
    "xCentralMoments",
]

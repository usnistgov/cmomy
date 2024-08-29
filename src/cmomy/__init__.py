"""Public api for :mod:`cmomy`"""
# Top level API (:mod:`cmomy`)
# ============================
# """

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Need this to play nice with IDE/pyright
    # submodules
    from . import convert, random, reduction, resample, rolling, utils  # noqa: TCH004
    from .confidence_interval import bootstrap_confidence_interval  # noqa: TCH004
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
    from .utils import assign_moment, moveaxis, select_moment  # noqa: TCH004
    from .wrapper import (  # noqa: TCH004
        CentralMoments,
        CentralMomentsArray,
        CentralMomentsXArray,
        xCentralMoments,
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
            "rolling",
            "utils",
        ],
        submod_attrs={
            "convert": ["concat"],
            "wrapper": [
                "CentralMomentsArray",
                "CentralMomentsXArray",
                "CentralMoments",
                "xCentralMoments",
            ],
            "confidence_interval": ["bootstrap_confidence_interval"],
            "reduction": ["reduce_data", "reduce_data_grouped", "reduce_vals"],
            "resample": [
                "indices_to_freq",
                "random_freq",
                "random_indices",
                "randsamp_freq",
                "resample_data",
                "resample_vals",
            ],
            "utils": ["moveaxis", "select_moment", "assign_moment"],
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
    "CentralMomentsArray",
    "CentralMomentsXArray",
    "__version__",
    "assign_moment",
    "bootstrap_confidence_interval",
    "concat",
    "convert",
    "indices_to_freq",
    "moveaxis",
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
    "rolling",
    "select_moment",
    "utils",
    "xCentralMoments",
]

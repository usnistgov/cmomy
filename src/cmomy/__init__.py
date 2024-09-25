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
    from .random import default_rng  # noqa: TCH004
    from .reduction import reduce_data, reduce_data_grouped, reduce_vals  # noqa: TCH004
    from .resample import (  # noqa: TCH004
        IndexSampler,
        factory_sampler,
        random_freq,
        random_indices,
        resample_data,
        resample_vals,
    )
    from .utils import (  # noqa: TCH004
        assign_moment,
        moveaxis,
        select_moment,
        vals_to_data,
    )
    from .wrapper import (  # noqa: TCH004
        CentralMoments,
        CentralMomentsArray,
        CentralMomentsData,
        wrap,
        wrap_raw,
        wrap_reduce_vals,
        wrap_resample_vals,
        xCentralMoments,
        zeros_like,
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
            "confidence_interval": ["bootstrap_confidence_interval"],
            "random": ["default_rng"],
            "reduction": ["reduce_data", "reduce_data_grouped", "reduce_vals"],
            "resample": [
                "IndexSampler",
                "factory_sampler",
                "random_indices",
                "random_freq",
                "resample_data",
                "resample_vals",
            ],
            "utils": [
                "moveaxis",
                "select_moment",
                "assign_moment",
                "vals_to_data",
            ],
            "wrapper": [
                "CentralMomentsArray",
                "CentralMomentsData",
                "CentralMoments",
                "zeros_like",
                "wrap_raw",
                "wrap_resample_vals",
                "wrap_reduce_vals",
                "wrap",
                "xCentralMoments",
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
    "CentralMomentsArray",
    "CentralMomentsData",
    "IndexSampler",
    "__version__",
    "assign_moment",
    "bootstrap_confidence_interval",
    "concat",
    "convert",
    "default_rng",
    "factory_sampler",
    "moveaxis",
    "random",
    "random_freq",
    "random_indices",
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
    "vals_to_data",
    "wrap",
    "wrap_raw",
    "wrap_reduce_vals",
    "wrap_resample_vals",
    "xCentralMoments",
    "zeros_like",
]

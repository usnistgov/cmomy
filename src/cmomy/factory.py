"""Interface to routines in this submodule."""
# pylint: disable=missing-class-docstring

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import TYPE_CHECKING, NamedTuple, cast

from cmomy.options import OPTIONS

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType
    from typing import Any, Protocol

    from numpy.typing import NDArray

    from cmomy.core.typing import (
        ConvertStyle,
        FloatT,
        MomNDim,
        NDArrayInt,
    )

    # Resample signature
    # These don't play well with pyright and overloading...
    class FreqToIndices(Protocol):
        def __call__(
            self,
            freq: NDArrayInt,
            indices: NDArrayInt,
            /,
            **kwargs: Any,
        ) -> tuple[()]: ...

    class IndicesToFreq(Protocol):
        def __call__(
            self,
            indices: NDArrayInt,
            freq: NDArrayInt,
            /,
            **kwargs: Any,
        ) -> tuple[()]: ...

    class ResampleVals(Protocol):
        def __call__(
            self,
            out: NDArray[FloatT],
            freq: NDArrayInt,
            x: NDArray[FloatT],
            w: NDArray[FloatT],
            /,
            *y: NDArray[FloatT],
            **kwargs: Any,
        ) -> tuple[()]: ...

    class ResampleData(Protocol):
        def __call__(
            self,
            freq: NDArrayInt,
            data: NDArray[FloatT],
            /,
            out: NDArray[FloatT] | None = None,
            **kwargs: Any,
        ) -> NDArray[FloatT]: ...

    # Jackknife resample
    class JackknifeVals(Protocol):
        def __call__(
            self,
            data_reduced: NDArray[FloatT],
            x: NDArray[FloatT],
            w: NDArray[FloatT],
            /,
            *y: NDArray[FloatT],
            out: NDArray[FloatT] | None = None,
            **kwargs: Any,
        ) -> NDArray[FloatT]: ...

    class JackknifeData(Protocol):
        def __call__(
            self,
            data_reduced: NDArray[FloatT],
            data: NDArray[FloatT],
            /,
            out: NDArray[FloatT] | None = None,
            **kwargs: Any,
        ) -> NDArray[FloatT]: ...

    # Reduce
    class ReduceVals(Protocol):
        def __call__(
            self,
            out: NDArray[FloatT],
            x: NDArray[FloatT],
            w: NDArray[FloatT],
            /,
            *y: NDArray[FloatT],
            **kwargs: Any,
        ) -> tuple[()]: ...

    class ReduceData(Protocol):
        def __call__(
            self,
            data: NDArray[FloatT],
            /,
            out: NDArray[FloatT] | None = None,
            **kwargs: Any,
        ) -> NDArray[FloatT]: ...

    # Grouped
    class ReduceDataGrouped(Protocol):
        def __call__(
            self,
            data: NDArray[FloatT],
            group_idx: NDArrayInt,
            out: NDArray[FloatT],
            /,
            **kwargs: Any,
        ) -> tuple[()]: ...

    class ReduceValsGrouped(Protocol):
        def __call__(
            self,
            out: NDArray[FloatT],
            group_idx: NDArrayInt,
            x: NDArray[FloatT],
            w: NDArray[FloatT],
            /,
            *y: NDArray[FloatT],
            **kwargs: Any,
        ) -> tuple[()]: ...

    # Indexed
    class ReduceDataIndexed(Protocol):
        def __call__(
            self,
            data: NDArray[FloatT],
            index: NDArrayInt,
            group_start: NDArrayInt,
            group_end: NDArrayInt,
            scale: NDArray[FloatT],
            /,
            out: NDArray[FloatT] | None = None,
            **kwargs: Any,
        ) -> NDArray[FloatT]: ...

    class ReduceValsIndexed(Protocol):
        def __call__(
            self,
            out: NDArray[FloatT],
            index: NDArrayInt,
            group_start: NDArrayInt,
            group_end: NDArrayInt,
            scale: NDArray[FloatT],
            x: NDArray[FloatT],
            w: NDArray[FloatT],
            /,
            *y: NDArray[FloatT],
            **kwargs: Any,
        ) -> tuple[()]: ...

    # convert
    class Convert(Protocol):
        def __call__(
            self,
            values_in: NDArray[FloatT],
            /,
            out: NDArray[FloatT] | None = None,
            **kwargs: Any,
        ) -> NDArray[FloatT]: ...

    # Rolling
    class RollingVals(Protocol):
        def __call__(
            self,
            out: NDArray[FloatT],
            window: int,
            min_count: int,
            x: NDArray[FloatT],
            w: NDArray[FloatT],
            /,
            *y: NDArray[FloatT],
            **kwargs: Any,
        ) -> tuple[()]: ...

    class RollingData(Protocol):
        def __call__(
            self,
            window: int,
            min_count: int,
            data: NDArray[FloatT],
            /,
            out: NDArray[FloatT] | None = None,
            **kwargs: Any,
        ) -> NDArray[FloatT]: ...

    class RollingExpVals(Protocol):
        def __call__(
            self,
            out: NDArray[FloatT],
            alpha: NDArray[FloatT],
            adjust: bool,
            min_count: int,
            x: NDArray[FloatT],
            w: NDArray[FloatT],
            /,
            *y: NDArray[FloatT],
            **kwargs: Any,
        ) -> tuple[()]: ...

    class RollingExpData(Protocol):
        def __call__(
            self,
            alpha: NDArray[FloatT],
            adjust: bool,
            min_count: int,
            data: NDArray[FloatT],
            /,
            out: NDArray[FloatT] | None = None,
            **kwargs: Any,
        ) -> NDArray[FloatT]: ...


# * Threading safety.  Taken from https://github.com/numbagg/numbagg/blob/main/numbagg/decorators.py


@lru_cache
def _safe_threadpool() -> bool:
    from ._lib.decorators import is_in_unsafe_thread_pool

    return not is_in_unsafe_thread_pool()


def supports_parallel() -> bool:
    """
    Checks if system supports parallel numba functions.

    If an unsafe thread pool is detected, return ``False``.

    Returns
    -------
    bool :
        ``True`` if supports parallel.  ``False`` otherwise.
    """
    return OPTIONS["parallel"] and _safe_threadpool()


# * Heuristic for parallel
def parallel_heuristic(
    parallel: bool | None,
    size: int | None = None,
    cutoff: int = 10_000,
) -> bool:
    """Default parallel."""
    if parallel is not None:
        return parallel and supports_parallel()
    if size is None or not supports_parallel():
        return False
    return size > cutoff


class Pusher(NamedTuple):
    """Collection of pusher functions."""

    val: Callable[..., None]
    vals: Callable[..., None]
    data: Callable[..., None]
    data_scale: Callable[..., None]
    datas: Callable[..., None]


@lru_cache
def _import_library_module_cached(
    submodule_base_name: str,
    mom_ndim: int = 1,
    parallel: bool = False,
    package: str | None = None,
) -> ModuleType:
    submodule = submodule_base_name
    if mom_ndim == 2:
        submodule += "_cov"

    if parallel:
        submodule += "_parallel"
    return import_module(f"cmomy._lib.{submodule}", package=package)


# NOTE(wpk): Use call to `_import_library_module_cached` so always recalculate supports_parallel
def _import_library_module(
    submodule_base_name: str,
    mom_ndim: int = 1,
    parallel: bool = False,
    package: str | None = None,
) -> ModuleType:
    return _import_library_module_cached(
        submodule_base_name,
        mom_ndim=mom_ndim,
        parallel=parallel and supports_parallel(),
        package=package,
    )


def factory_pusher(
    mom_ndim: MomNDim = 1,
    parallel: bool = True,
) -> Pusher:
    """Factory method to get pusher functions."""
    push_mod = _import_library_module("push", mom_ndim=mom_ndim, parallel=parallel)

    return Pusher(
        val=push_mod.push_val,
        vals=push_mod.reduce_vals,
        data=push_mod.push_data,
        data_scale=push_mod.push_data_scale,
        datas=push_mod.reduce_data,
    )


# * Resample
def factory_freq_to_indices(parallel: bool = True) -> FreqToIndices:
    return cast(
        "FreqToIndices",
        _import_library_module("resample", parallel=parallel).freq_to_indices,
    )


def factory_indices_to_freq(parallel: bool = True) -> IndicesToFreq:
    return cast(
        "IndicesToFreq",
        _import_library_module("resample", parallel=parallel).indices_to_freq,
    )


def factory_resample_vals(
    mom_ndim: MomNDim = 1,
    parallel: bool = True,
) -> ResampleVals:
    """Resample values."""
    return cast(
        "ResampleVals",
        _import_library_module(
            "resample", parallel=parallel, mom_ndim=mom_ndim
        ).resample_vals,
    )


def factory_resample_data(
    mom_ndim: MomNDim = 1,
    parallel: bool = True,
) -> ResampleData:
    """Resample data."""
    return cast(
        "ResampleData",
        _import_library_module(
            "resample", parallel=parallel, mom_ndim=mom_ndim
        ).resample_data_fromzero,
    )


# * Jackknife
def factory_jackknife_vals(
    mom_ndim: MomNDim = 1,
    parallel: bool = True,
) -> JackknifeVals:
    return cast(
        "JackknifeVals",
        _import_library_module(
            "resample", parallel=parallel, mom_ndim=mom_ndim
        ).jackknife_vals_fromzero,
    )


def factory_jackknife_data(
    mom_ndim: MomNDim = 1,
    parallel: bool = True,
) -> JackknifeData:
    return cast(
        "JackknifeData",
        _import_library_module(
            "resample", parallel=parallel, mom_ndim=mom_ndim
        ).jackknife_data_fromzero,
    )


# * Reduce
def factory_reduce_vals(
    mom_ndim: MomNDim = 1,
    parallel: bool = True,
) -> ReduceVals:
    return cast(
        "ReduceVals",
        _import_library_module(
            "push", parallel=parallel, mom_ndim=mom_ndim
        ).reduce_vals,
    )


def factory_reduce_data(
    mom_ndim: MomNDim = 1,
    parallel: bool = True,
) -> ReduceData:
    return cast(
        "ReduceData",
        _import_library_module(
            "push", parallel=parallel, mom_ndim=mom_ndim
        ).reduce_data_fromzero,
    )


def factory_reduce_data_grouped(
    mom_ndim: MomNDim = 1,
    parallel: bool = True,
) -> ReduceDataGrouped:
    return cast(
        "ReduceDataGrouped",
        _import_library_module(
            "grouped", parallel=parallel, mom_ndim=mom_ndim
        ).reduce_data_grouped,
    )


def factory_reduce_vals_grouped(
    mom_ndim: MomNDim = 1,
    parallel: bool = True,
) -> ReduceValsGrouped:
    return cast(
        "ReduceValsGrouped",
        _import_library_module(
            "grouped", parallel=parallel, mom_ndim=mom_ndim
        ).reduce_vals_grouped,
    )


def factory_reduce_data_indexed(
    mom_ndim: MomNDim = 1,
    parallel: bool = True,
) -> ReduceDataIndexed:
    return cast(
        "ReduceDataIndexed",
        _import_library_module(
            "grouped", parallel=parallel, mom_ndim=mom_ndim
        ).reduce_data_indexed_fromzero,
    )


def factory_reduce_vals_indexed(
    mom_ndim: MomNDim = 1,
    parallel: bool = True,
) -> ReduceValsIndexed:
    return cast(
        "ReduceValsIndexed",
        _import_library_module(
            "grouped", parallel=parallel, mom_ndim=mom_ndim
        ).reduce_vals_indexed_fromzero,
    )


# * Convert
def factory_convert(mom_ndim: MomNDim = 1, to: ConvertStyle = "central") -> Convert:
    m = _import_library_module("convert", mom_ndim=mom_ndim)
    name = "raw_to_central" if to == "central" else "central_to_raw"
    return cast("Convert", getattr(m, name))


# * CumReduce
def factory_cumulative(
    mom_ndim: MomNDim = 1,
    parallel: bool = True,
    inverse: bool = False,
) -> Convert:
    m = _import_library_module("push", parallel=parallel, mom_ndim=mom_ndim)
    name = "cumulative_inverse" if inverse else "cumulative"
    return cast("Convert", getattr(m, name))


# * Rolling
def factory_rolling_vals(
    mom_ndim: MomNDim = 1,
    parallel: bool = True,
) -> RollingVals:
    return cast(
        "RollingVals",
        _import_library_module(
            "rolling", parallel=parallel, mom_ndim=mom_ndim
        ).rolling_vals,
    )


def factory_rolling_data(
    mom_ndim: MomNDim = 1,
    parallel: bool = True,
) -> RollingData:
    return cast(
        "RollingData",
        _import_library_module(
            "rolling", parallel=parallel, mom_ndim=mom_ndim
        ).rolling_data,
    )


def factory_rolling_exp_vals(
    mom_ndim: MomNDim = 1,
    parallel: bool = True,
) -> RollingExpVals:
    return cast(
        "RollingExpVals",
        _import_library_module(
            "rolling", parallel=parallel, mom_ndim=mom_ndim
        ).rolling_exp_vals,
    )


def factory_rolling_exp_data(
    mom_ndim: MomNDim = 1,
    parallel: bool = True,
) -> RollingExpData:
    return cast(
        "RollingExpData",
        _import_library_module(
            "rolling", parallel=parallel, mom_ndim=mom_ndim
        ).rolling_exp_data,
    )

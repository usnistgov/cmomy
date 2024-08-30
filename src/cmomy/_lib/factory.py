"""Interface to routines in this submodule."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple, cast

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Any, Callable, Protocol

    from numpy.typing import NDArray

    from cmomy.core.typing import (
        ConvertStyle,
        FloatT,
        Mom_NDim,
        NDArrayInt,
    )

    # Resample signature
    # These don't play well with pyright and overloading...
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
            data_tmp: NDArray[FloatT],
            window: int,
            min_count: int,
            x: NDArray[FloatT],
            w: NDArray[FloatT],
            /,
            *y: NDArray[FloatT],
            out: NDArray[FloatT] | None = None,
            **kwargs: Any,
        ) -> NDArray[FloatT]: ...

    class RollingData(Protocol):
        def __call__(
            self,
            data_tmp: NDArray[FloatT],
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
            data_tmp: NDArray[FloatT],
            alpha: NDArray[FloatT],
            adjust: bool,
            min_count: int,
            x: NDArray[FloatT],
            w: NDArray[FloatT],
            /,
            *y: NDArray[FloatT],
            out: NDArray[FloatT] | None = None,
            **kwargs: Any,
        ) -> NDArray[FloatT]: ...

    class RollingExpData(Protocol):
        def __call__(
            self,
            data_tmp: NDArray[FloatT],
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
def supports_parallel() -> bool:
    """
    Checks if system supports parallel numba functions.

    If an unsafe thread pool is detected, return ``False``.

    Returns
    -------
    bool :
        ``True`` if supports parallel.  ``False`` otherwise.
    """
    from .decorators import is_in_unsafe_thread_pool

    return not is_in_unsafe_thread_pool()


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
    datas: Callable[..., None]


@lru_cache
def factory_pusher(
    mom_ndim: Mom_NDim = 1,
    parallel: bool = True,
) -> Pusher:
    """Factory method to get pusher functions."""
    parallel = parallel and supports_parallel()

    _push_mod: ModuleType
    if mom_ndim == 1 and parallel:
        from . import push_parallel as _push_mod
    elif mom_ndim == 1:
        from . import push as _push_mod
    elif mom_ndim == 2 and parallel:
        from . import push_cov_parallel as _push_mod
    else:
        from . import push_cov as _push_mod

    return Pusher(
        val=_push_mod.push_val,
        vals=_push_mod.reduce_vals,
        data=_push_mod.push_data,
        datas=_push_mod.reduce_data,
    )


# * Resample
@lru_cache
def factory_resample_vals(
    mom_ndim: Mom_NDim = 1,
    parallel: bool = True,
) -> ResampleVals:
    parallel = parallel and supports_parallel()

    if mom_ndim == 1:
        if parallel:
            from .resample_parallel import resample_vals as _resample
        else:
            from .resample import resample_vals as _resample
        return cast("ResampleVals", _resample)

    if parallel:
        from .resample_cov_parallel import resample_vals as _resample_cov
    else:
        from .resample_cov import resample_vals as _resample_cov
    return cast("ResampleVals", _resample_cov)


@lru_cache
def factory_resample_data(
    mom_ndim: Mom_NDim = 1,
    parallel: bool = True,
) -> ResampleData:
    parallel = parallel and supports_parallel()
    if mom_ndim == 1 and parallel:
        from .resample_parallel import resample_data_fromzero
    elif mom_ndim == 1:
        from .resample import resample_data_fromzero
    elif parallel:
        from .resample_cov_parallel import resample_data_fromzero
    else:
        from .resample_cov import resample_data_fromzero
    return cast("ResampleData", resample_data_fromzero)


# * Jackknife
@lru_cache
def factory_jackknife_vals(
    mom_ndim: Mom_NDim = 1,
    parallel: bool = True,
) -> JackknifeVals:
    parallel = parallel and supports_parallel()

    if mom_ndim == 1:
        if parallel:
            from .resample_parallel import jackknife_vals_fromzero as _jackknife
        else:
            from .resample import jackknife_vals_fromzero as _jackknife
        return cast("JackknifeVals", _jackknife)

    if parallel:
        from .resample_cov_parallel import jackknife_vals_fromzero as _jackknife_cov
    else:
        from .resample_cov import jackknife_vals_fromzero as _jackknife_cov
    return cast("JackknifeVals", _jackknife_cov)


@lru_cache
def factory_jackknife_data(
    mom_ndim: Mom_NDim = 1,
    parallel: bool = True,
) -> JackknifeData:
    parallel = parallel and supports_parallel()
    if mom_ndim == 1 and parallel:
        from .resample_parallel import jackknife_data_fromzero
    elif mom_ndim == 1:
        from .resample import jackknife_data_fromzero
    elif parallel:
        from .resample_cov_parallel import jackknife_data_fromzero
    else:
        from .resample_cov import jackknife_data_fromzero
    return cast("JackknifeData", jackknife_data_fromzero)


# * Reduce
@lru_cache
def factory_reduce_vals(
    mom_ndim: Mom_NDim = 1,
    parallel: bool = True,
) -> ReduceVals:
    parallel = parallel and supports_parallel()
    if mom_ndim == 1:
        if parallel:
            from .push_parallel import reduce_vals as _reduce
        else:
            from .push import reduce_vals as _reduce
        return cast("ReduceVals", _reduce)

    if parallel:
        from .push_cov_parallel import reduce_vals as _reduce_cov
    else:
        from .push_cov import reduce_vals as _reduce_cov
    return cast("ReduceVals", _reduce_cov)


@lru_cache
def factory_reduce_data(
    mom_ndim: Mom_NDim = 1,
    parallel: bool = True,
) -> ReduceData:
    parallel = parallel and supports_parallel()

    if mom_ndim == 1 and parallel:
        from .push_parallel import reduce_data_fromzero
    elif mom_ndim == 1:
        from .push import reduce_data_fromzero
    elif parallel:
        from .push_cov_parallel import reduce_data_fromzero
    else:
        from .push_cov import reduce_data_fromzero

    return cast("ReduceData", reduce_data_fromzero)


@lru_cache
def factory_reduce_data_grouped(
    mom_ndim: Mom_NDim = 1,
    parallel: bool = True,
) -> ReduceDataGrouped:
    parallel = parallel and supports_parallel()
    if mom_ndim == 1 and parallel:
        from .indexed_parallel import reduce_data_grouped
    elif mom_ndim == 1:
        from .indexed import reduce_data_grouped
    elif parallel:
        from .indexed_cov_parallel import reduce_data_grouped
    else:
        from .indexed_cov import reduce_data_grouped

    return cast("ReduceDataGrouped", reduce_data_grouped)


@lru_cache
def factory_reduce_data_indexed(
    mom_ndim: Mom_NDim = 1,
    parallel: bool = True,
) -> ReduceDataIndexed:
    parallel = parallel and supports_parallel()

    if mom_ndim == 1 and parallel:
        from .indexed_parallel import reduce_data_indexed_fromzero
    elif mom_ndim == 1:
        from .indexed import reduce_data_indexed_fromzero
    elif parallel:
        from .indexed_cov_parallel import reduce_data_indexed_fromzero
    else:
        from .indexed_cov import reduce_data_indexed_fromzero

    # cast because guvectorized with optional out
    return cast("ReduceDataIndexed", reduce_data_indexed_fromzero)


# * Convert
@lru_cache
def factory_convert(mom_ndim: Mom_NDim = 1, to: ConvertStyle = "central") -> Convert:
    if to == "central":
        # raw to central
        if mom_ndim == 1:
            from .convert import raw_to_central
        else:
            from .convert_cov import raw_to_central
        return cast("Convert", raw_to_central)

    # central to raw
    if mom_ndim == 1:
        from .convert import central_to_raw
    else:
        from .convert_cov import central_to_raw
    return cast("Convert", central_to_raw)


# * CumReduce
@lru_cache
def factory_cumulative(
    mom_ndim: Mom_NDim = 1,
    parallel: bool = True,
    inverse: bool = False,
) -> Convert:
    parallel = parallel and supports_parallel()
    if inverse:
        if mom_ndim == 1 and parallel:
            from .push_parallel import cumulative_inverse as func
        elif mom_ndim == 1:
            from .push import cumulative_inverse as func
        elif parallel:
            from .push_cov_parallel import cumulative_inverse as func
        else:
            from .push_cov import cumulative_inverse as func

    elif mom_ndim == 1 and parallel:
        from .push_parallel import cumulative as func
    elif mom_ndim == 1:
        from .push import cumulative as func
    elif parallel:
        from .push_cov_parallel import cumulative as func
    else:
        from .push_cov import cumulative as func

    return cast("Convert", func)


# * Rolling
@lru_cache
def factory_rolling_vals(
    mom_ndim: Mom_NDim = 1,
    parallel: bool = True,
) -> RollingVals:
    parallel = parallel and supports_parallel()

    if mom_ndim == 1:
        if parallel:
            from .rolling_parallel import rolling_vals
        else:
            from .rolling import rolling_vals
        return cast("RollingVals", rolling_vals)

    if parallel:
        from .rolling_cov_parallel import rolling_vals as _rolling_cov
    else:
        from .rolling_cov import rolling_vals as _rolling_cov
    return cast("RollingVals", _rolling_cov)


@lru_cache
def factory_rolling_data(
    mom_ndim: Mom_NDim = 1,
    parallel: bool = True,
) -> RollingData:
    parallel = parallel and supports_parallel()

    if mom_ndim == 1 and parallel:
        from .rolling_parallel import rolling_data
    elif mom_ndim == 1:
        from .rolling import rolling_data
    elif parallel:
        from .rolling_cov_parallel import rolling_data
    else:
        from .rolling_cov import rolling_data

    return cast("RollingData", rolling_data)


@lru_cache
def factory_rolling_exp_vals(
    mom_ndim: Mom_NDim = 1,
    parallel: bool = True,
) -> RollingExpVals:
    parallel = parallel and supports_parallel()

    if mom_ndim == 1:
        if parallel:
            from .rolling_parallel import rolling_exp_vals
        else:
            from .rolling import rolling_exp_vals
        return cast("RollingExpVals", rolling_exp_vals)

    if parallel:
        from .rolling_cov_parallel import rolling_exp_vals as _rolling_cov
    else:
        from .rolling_cov import rolling_exp_vals as _rolling_cov
    return cast("RollingExpVals", _rolling_cov)


@lru_cache
def factory_rolling_exp_data(
    mom_ndim: Mom_NDim = 1,
    parallel: bool = True,
) -> RollingExpData:
    parallel = parallel and supports_parallel()

    if mom_ndim == 1 and parallel:
        from .rolling_parallel import rolling_exp_data
    elif mom_ndim == 1:
        from .rolling import rolling_exp_data
    elif parallel:
        from .rolling_cov_parallel import rolling_exp_data
    else:
        from .rolling_cov import rolling_exp_data

    return cast("RollingExpData", rolling_exp_data)

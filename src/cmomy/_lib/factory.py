"""Interface to routines in this submodule."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple, cast, overload

from numpy.typing import NDArray

from .utils import supports_parallel

if TYPE_CHECKING:
    from typing import Callable, Literal, Protocol

    from numpy.typing import NDArray

    from ..typing import ConvertStyle, FloatT, Mom_NDim, NDArrayInt

    # Resample signature
    # These don't play well with pyright and overloading...
    class ResampleVals(Protocol):
        def __call__(
            self,
            x: NDArray[FloatT],
            w: NDArray[FloatT],
            freq: NDArrayInt,
            out: NDArray[FloatT],
            /,
        ) -> tuple[()]: ...

    class ResampleValsCov(Protocol):
        def __call__(
            self,
            x0: NDArray[FloatT],
            x1: NDArray[FloatT],
            w: NDArray[FloatT],
            freq: NDArrayInt,
            out: NDArray[FloatT],
            /,
        ) -> tuple[()]: ...

    class ResampleData(Protocol):
        def __call__(
            self,
            data: NDArray[FloatT],
            freq: NDArrayInt,
            out: NDArray[FloatT] | None = None,
            /,
        ) -> NDArray[FloatT]: ...

    # Reduce
    class ReduceVals(Protocol):
        def __call__(
            self, x: NDArray[FloatT], w: NDArray[FloatT], out: NDArray[FloatT], /
        ) -> tuple[()]: ...

    class ReduceValsCov(Protocol):
        def __call__(
            self,
            x0: NDArray[FloatT],
            x1: NDArray[FloatT],
            w: NDArray[FloatT],
            out: NDArray[FloatT],
            /,
        ) -> tuple[()]: ...

    class ReduceData(Protocol):
        def __call__(
            self, data: NDArray[FloatT], out: NDArray[FloatT] | None = None, /
        ) -> NDArray[FloatT]: ...

    # Grouped
    class ReduceDataGrouped(Protocol):
        def __call__(
            self,
            data: NDArray[FloatT],
            group_idx: NDArrayInt,
            out: NDArray[FloatT],
            /,
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
            out: NDArray[FloatT] | None = None,
            /,
        ) -> NDArray[FloatT]: ...

    # convert
    class Convert(Protocol):
        def __call__(
            self, values_in: NDArray[FloatT], out: NDArray[FloatT] | None = None, /
        ) -> NDArray[FloatT]: ...


class Pusher(NamedTuple):
    """Collection of pusher functions."""

    val: Callable[..., None]
    vals: Callable[..., None]
    data: Callable[..., None]
    datas: Callable[..., None]
    stat: Callable[..., None] | None = None
    stats: Callable[..., None] | None = None


@lru_cache
def factory_pusher(mom_ndim: Mom_NDim = 1, parallel: bool = True) -> Pusher:
    """Factory method to get pusher functions."""
    parallel = parallel and supports_parallel()

    if mom_ndim == 1 and parallel:
        from . import push_parallel

        return Pusher(
            val=push_parallel.push_val,
            vals=push_parallel.reduce_vals,
            data=push_parallel.push_data,
            datas=push_parallel.reduce_data,
            stat=push_parallel.push_stat,
            stats=push_parallel.reduce_stats,
        )

    if mom_ndim == 1:
        from . import push

        return Pusher(
            val=push.push_val,
            vals=push.reduce_vals,
            data=push.push_data,
            datas=push.reduce_data,
            stat=push.push_stat,
            stats=push.reduce_stats,
        )

    if mom_ndim == 2 and parallel:
        from . import push_cov_parallel

        return Pusher(
            val=push_cov_parallel.push_val,
            vals=push_cov_parallel.reduce_vals,
            data=push_cov_parallel.push_data,
            datas=push_cov_parallel.reduce_data,
        )

    from . import push_cov

    return Pusher(
        val=push_cov.push_val,
        vals=push_cov.reduce_vals,
        data=push_cov.push_data,
        datas=push_cov.reduce_data,
    )


# * Resample
@overload
def factory_resample_vals(
    mom_ndim: Literal[1] = ...,
    parallel: bool = ...,
) -> ResampleVals: ...


@overload
def factory_resample_vals(
    mom_ndim: Literal[2],
    parallel: bool = ...,
) -> ResampleValsCov: ...


@lru_cache
def factory_resample_vals(
    mom_ndim: Mom_NDim = 1, parallel: bool = True
) -> ResampleVals | ResampleValsCov:
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
    return cast("ResampleValsCov", _resample_cov)


@lru_cache
def factory_resample_data(
    mom_ndim: Mom_NDim = 1, parallel: bool = True
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
    return cast("ResampleData", resample_data_fromzero)  # pyright: ignore[reportReturnType]


# * Reduce
@overload
def factory_reduce_vals(
    mom_ndim: Literal[1] = ...,
    parallel: bool = ...,
) -> ReduceVals: ...


@overload
def factory_reduce_vals(
    mom_ndim: Literal[2],
    parallel: bool = ...,
) -> ReduceValsCov: ...


@overload
def factory_reduce_vals(
    mom_ndim: Mom_NDim = ...,
    parallel: bool = ...,
) -> ReduceVals | ReduceValsCov: ...


@lru_cache
def factory_reduce_vals(
    mom_ndim: Mom_NDim = 1, parallel: bool = True
) -> ReduceVals | ReduceValsCov:
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
    return cast("ReduceValsCov", _reduce_cov)


@lru_cache
def factory_reduce_data(mom_ndim: Mom_NDim = 1, parallel: bool = True) -> ReduceData:
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
    mom_ndim: Mom_NDim = 1, parallel: bool = True
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
    mom_ndim: Mom_NDim = 1, parallel: bool = True
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

"""Interface to routines in this submodule."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple, TypedDict, cast, overload

from numpy.typing import NDArray

from cmomy.new._lib.utils import supports_parallel

if TYPE_CHECKING:
    from typing import Callable, Literal, Protocol

    from numpy.typing import NDArray

    from ..typing import ConvertStyle, LongIntDType, Mom_NDim
    from ..typing import T_FloatDType as T_Float

    # Resample signature
    # These don't play well with pyright and overloading...
    class ResampleVals(Protocol):
        def __call__(
            self,
            x: NDArray[T_Float],
            w: NDArray[T_Float],
            freq: NDArray[LongIntDType],
            out: NDArray[T_Float],
            /,
        ) -> tuple[()]: ...

    class ResampleValsCov(Protocol):
        def __call__(
            self,
            x0: NDArray[T_Float],
            x1: NDArray[T_Float],
            w: NDArray[T_Float],
            freq: NDArray[LongIntDType],
            out: NDArray[T_Float],
            /,
        ) -> tuple[()]: ...

    # ResampleVals = Callable[
    #     [
    #         NDArray[T_Float], # x
    #         NDArray[T_Float], # w
    #         NDArray[LongIntDType], # freq
    #         NDArray[T_Float], # out
    #     ],
    #     None
    # ]
    # ResampleValsCov = Callable[
    #     [
    #         NDArray[T_Float], # x0
    #         NDArray[T_Float], # x1
    #         NDArray[T_Float], # w
    #         NDArray[LongIntDType], # freq
    #         NDArray[T_Float], # out
    #     ],
    #     None
    # ]

    class ResampleData(Protocol):
        def __call__(
            self,
            data: NDArray[T_Float],
            freq: NDArray[LongIntDType],
            out: NDArray[T_Float] | None = None,
            /,
        ) -> NDArray[T_Float]: ...

    # Reduce
    class ReduceVals(Protocol):
        def __call__(
            self, x: NDArray[T_Float], w: NDArray[T_Float], out: NDArray[T_Float], /
        ) -> tuple[()]: ...

    class ReduceValsCov(Protocol):
        def __call__(
            self,
            x0: NDArray[T_Float],
            x1: NDArray[T_Float],
            w: NDArray[T_Float],
            out: NDArray[T_Float],
            /,
        ) -> tuple[()]: ...

    # ReduceVals = Callable[
    #     [
    #         NDArray[T_Float], # x
    #         NDArray[T_Float], # w
    #         NDArray[T_Float], # out
    #     ],
    #     tuple[()]
    # ]
    # ReduceValsCov = Callable[
    #     [
    #         NDArray[T_Float], # x0
    #         NDArray[T_Float], # x1
    #         NDArray[T_Float], # w
    #         NDArray[T_Float], # out
    #     ],
    #     tuple[()]
    # ]

    class ReduceData(Protocol):
        def __call__(
            self, data: NDArray[T_Float], out: NDArray[T_Float] | None = None, /
        ) -> NDArray[T_Float]: ...

    # Grouped
    class ReduceDataGrouped(Protocol):
        def __call__(
            self,
            data: NDArray[T_Float],
            group_idx: NDArray[LongIntDType],
            out: NDArray[T_Float],
            /,
        ) -> tuple[()]: ...

    # Indexed
    class ReduceDataIndexed(Protocol):
        def __call__(
            self,
            data: NDArray[T_Float],
            index: NDArray[LongIntDType],
            group_start: NDArray[LongIntDType],
            group_end: NDArray[LongIntDType],
            scale: NDArray[T_Float],
            out: NDArray[T_Float] | None = None,
            /,
        ) -> NDArray[T_Float]: ...

    # convert
    class Convert(Protocol):
        def __call__(
            self, values_in: NDArray[T_Float], out: NDArray[T_Float] | None = None, /
        ) -> NDArray[T_Float]: ...


class Pusher(NamedTuple):
    """Collection of pusher functions."""

    val: Callable[..., None]
    vals: Callable[..., None]
    data: Callable[..., None]
    datas: Callable[..., None]
    stat: Callable[..., None] | None = None
    stats: Callable[..., None] | None = None


class PusherTotal(TypedDict):
    """Collection of pushers."""

    serial: Pusher
    parallel: Pusher


@lru_cache
def factory_pushers(mom_ndim: Mom_NDim = 1, parallel: bool = True) -> PusherTotal:
    """Factory method to get pusher functions."""
    parallel = parallel and supports_parallel()

    if mom_ndim == 1:
        from . import push

        out_serial = Pusher(
            val=push.push_val,
            vals=push.reduce_vals,
            data=push.push_data,
            datas=push.reduce_data,
            stat=push.push_stat,
            stats=push.reduce_stats,
        )
    else:
        from . import push_cov

        out_serial = Pusher(
            val=push_cov.push_val,
            vals=push_cov.reduce_vals,
            data=push_cov.push_data,
            datas=push_cov.reduce_data,
        )

    if parallel and mom_ndim == 1:
        from . import push_parallel

        out_parallel = Pusher(
            val=push_parallel.push_val,
            vals=push_parallel.reduce_vals,
            data=push_parallel.push_data,
            datas=push_parallel.reduce_data,
            stat=push_parallel.push_stat,
            stats=push_parallel.reduce_stats,
        )

    elif parallel:
        from . import push_cov_parallel

        out_parallel = Pusher(
            val=push_cov_parallel.push_val,
            vals=push_cov_parallel.reduce_vals,
            data=push_cov_parallel.push_data,
            datas=push_cov_parallel.reduce_data,
        )

    else:
        out_parallel = out_serial

    return PusherTotal(serial=out_serial, parallel=out_parallel)


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

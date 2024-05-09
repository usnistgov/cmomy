"""
Routines to perform central moments reduction (:mod:`~cmomy.reduce)
===================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray

from ._lib.factory import (
    factory_reduce_data,
    factory_reduce_data_grouped,
    factory_reduce_data_indexed,
    factory_reduce_vals,
)
from .docstrings import docfiller
from .utils import (
    parallel_heuristic,
    prepare_data_for_reduction,
    prepare_values_for_reduction,
    raise_if_wrong_shape,
    validate_mom_and_mom_ndim,
    validate_mom_dims,
    validate_mom_ndim,
    xprepare_data_for_reduction,
    xprepare_values_for_reduction,
)

if TYPE_CHECKING:
    from typing import Hashable, Sequence

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from cmomy.typing import MomentsStrict

    from .typing import ArrayOrder, LongIntDType, Mom_NDim, MomDims, Moments, NDArrayAny
    from .typing import T_FloatDType as T_Float


# * Base reducers -------------------------------------------------------------
#   These assume input data in correct form (core dimensions last)
def _reduce_vals(
    x0: NDArray[T_Float],
    w: NDArray[T_Float],
    *x1: NDArray[T_Float],
    mom: MomentsStrict,
    parallel: bool | None = None,
    out: NDArray[T_Float] | None = None,
) -> NDArray[T_Float]:
    val_shape: tuple[int, ...] = np.broadcast_shapes(*(_.shape for _ in (x0, *x1, w)))[
        :-1
    ]
    mom_shape: tuple[int, ...] = tuple(m + 1 for m in mom)
    out_shape: tuple[int, ...] = (*val_shape, *mom_shape)

    if out is None:
        out = np.zeros(out_shape, dtype=x0.dtype)
    else:
        raise_if_wrong_shape(out, out_shape)
        out.fill(0.0)

    factory_reduce_vals(  # type: ignore[call-overload]
        mom_ndim=len(mom),
        parallel=parallel_heuristic(parallel, x0.size * len(mom)),
    )(x0, *x1, w, out)  # pyright: ignore[reportCallIssue]
    return out


def _reduce_data(
    data: NDArray[T_Float],
    *,
    mom_ndim: Mom_NDim,
    parallel: bool | None = None,
    out: NDArrayAny | None = None,
) -> NDArray[T_Float]:
    _reduce = factory_reduce_data(
        mom_ndim=mom_ndim,
        parallel=parallel_heuristic(parallel, data.size * mom_ndim),
    )
    if out:
        return _reduce(data, out)
    return _reduce(data)


def _reduce_data_grouped(
    data: NDArray[T_Float],
    by: NDArray[LongIntDType],
    *,
    mom_ndim: Mom_NDim,
    parallel: bool | None = None,
    out: NDArray[T_Float] | None = None,
) -> NDArray[T_Float]:
    if len(by) != data.shape[-(mom_ndim + 1)]:
        msg = f"{len(by)=} != data.shape[axis]={data.shape[-(mom_ndim + 1)]}"
        raise ValueError(msg)

    ngroup = by.max() + 1
    out_shape = (*data.shape[: -(mom_ndim + 1)], ngroup, *data.shape[-mom_ndim:])
    if out is None:
        out = np.zeros(out_shape, dtype=data.dtype)
    else:
        raise_if_wrong_shape(out, out_shape)
        out.fill(0.0)

    factory_reduce_data_grouped(
        mom_ndim=mom_ndim,
        parallel=parallel_heuristic(parallel, data.size * mom_ndim),
    )(data, by, out)

    return out


def _reduce_data_indexed(
    data: NDArray[T_Float],
    *,
    mom_ndim: Mom_NDim,
    index: NDArray[LongIntDType],
    group_start: NDArray[LongIntDType],
    group_end: NDArray[LongIntDType],
    scale: NDArray[T_Float] | None = None,
    parallel: bool | None = None,
    out: NDArray[T_Float] | None = None,
) -> NDArray[T_Float]:
    if scale is None:
        scale = np.ones_like(index, dtype=data.dtype)

    _reduce = factory_reduce_data_indexed(
        mom_ndim=mom_ndim,
        parallel=parallel_heuristic(parallel, data.size * mom_ndim),
    )

    if out is None:
        return _reduce(data, index, group_start, group_end, scale)
    return _reduce(data, index, group_start, group_end, scale, out)


# * Array api -----------------------------------------------------------------
# ** reduce vals
@overload
def reduce_vals(
    x: xr.DataArray,
    *y: ArrayLike | xr.DataArray,
    mom: Moments,
    weight: ArrayLike | xr.DataArray | None = ...,
    axis: int | None = ...,
    dim: Hashable | None = ...,
    mom_dims: MomDims | None = ...,
    order: ArrayOrder = ...,
    parallel: bool | None = ...,
    out: NDArray[T_Float] | None = ...,
) -> xr.DataArray: ...


@overload
def reduce_vals(
    x: NDArray[T_Float],
    *y: ArrayLike | xr.DataArray,
    mom: Moments,
    weight: ArrayLike | xr.DataArray | None = ...,
    axis: int | None = ...,
    dim: Hashable | None = ...,
    mom_dims: MomDims | None = ...,
    order: ArrayOrder = ...,
    parallel: bool | None = ...,
    out: NDArray[T_Float] | None = ...,
) -> NDArray[T_Float]: ...


@docfiller.decorate
def reduce_vals(
    x: NDArray[T_Float] | xr.DataArray,
    *y: ArrayLike | xr.DataArray,
    mom: Moments,
    weight: ArrayLike | xr.DataArray | None = None,
    axis: int | None = None,
    dim: Hashable | None = None,
    mom_dims: MomDims | None = None,
    order: ArrayOrder = None,
    parallel: bool | None = None,
    out: NDArray[T_Float] | None = None,
) -> NDArray[T_Float] | xr.DataArray:
    """
    Reduce values to central (co)moments.

    Parameters
    ----------
    x : ndarray or DataArray
        Values to analyze.
    *y : array-like or DataArray
        Seconda value.  Must specify if ``len(mom) == 2.``
    {mom}
    weight : scalar or array-like or DataArray
        Weights for each point.
    {axis}
    {dim}
    {mom_dims}
    {order}
    {parallel}
    {out}

    Returns
    -------
    out : ndarray or DataArray
        Central moments array of same type as ``x``.
        ``out.shape = (...,shape[axis-1], shape[axis+1], ..., mom0, ...)``
        where ``shape = args[0].shape``.
    """
    mom_validated, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
    weight = 1.0 if weight is None else weight

    if isinstance(x, xr.DataArray):
        input_core_dims, (x0, w, *x1) = xprepare_values_for_reduction(
            x, weight, *y, axis=axis, dim=dim, order=order, narrays=mom_ndim + 1
        )
        mom_dims_strict = validate_mom_dims(mom_dims=mom_dims, mom_ndim=mom_ndim)

        return xr.apply_ufunc(  # type: ignore[no-any-return]
            _reduce_vals,
            x0,
            w,
            *x1,
            input_core_dims=input_core_dims,
            output_core_dims=[mom_dims_strict],
            kwargs={"mom": mom_validated, "parallel": parallel, "out": out},
        )

    _x0, _w, *_x1 = prepare_values_for_reduction(
        x, weight, *y, axis=axis, order=order, narrays=mom_ndim + 1
    )
    return _reduce_vals(_x0, _w, *_x1, mom=mom_validated, parallel=parallel, out=out)  # type: ignore[has-type]


# ** reduce data
@overload
def reduce_data(
    data: NDArray[T_Float],
    *,
    mom_ndim: Mom_NDim,
    dim: Hashable | None = ...,
    axis: int | None = ...,
    order: ArrayOrder = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
) -> NDArray[T_Float]: ...


@overload
def reduce_data(
    data: xr.DataArray,
    *,
    mom_ndim: Mom_NDim,
    dim: Hashable | None = ...,
    axis: int | None = ...,
    order: ArrayOrder = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
) -> xr.DataArray: ...


# NOTE: might be easier to use T_Array, but
# then have to change other functions...
@docfiller.decorate
def reduce_data(
    data: NDArray[T_Float] | xr.DataArray,
    *,
    mom_ndim: Mom_NDim,
    dim: Hashable | None = None,
    axis: int | None = None,
    order: ArrayOrder = None,
    parallel: bool | None = None,
    dtype: DTypeLike = None,
    out: NDArrayAny | None = None,
) -> NDArray[T_Float] | xr.DataArray:
    """
    Reduce central moments array along axis.


    Parameters
    ----------
    {data_numpy_or_dataarray}
    {mom_ndim}
    {axis_data}
    {dim}
    {order}
    {parallel}
    {out}

    Returns
    -------
    out : ndarray or DataArray.
        Reduced data array with shape ``data.shape`` with ``axis`` removed.
        Same type as input ``data``.
    """
    mom_ndim = validate_mom_ndim(mom_ndim)
    if isinstance(data, xr.DataArray):
        dim, data = xprepare_data_for_reduction(
            data, axis=axis, dim=dim, mom_ndim=mom_ndim, order=order, dtype=dtype
        )
        return xr.apply_ufunc(  # type: ignore[no-any-return]
            _reduce_data,
            data,
            input_core_dims=[data.dims[-(mom_ndim + 1) :]],
            output_core_dims=[data.dims[-mom_ndim:]],
            kwargs={"mom_ndim": mom_ndim, "parallel": parallel, "out": out},
        )

    data = prepare_data_for_reduction(data, axis=axis, mom_ndim=mom_ndim, order=order)
    return _reduce_data(data, mom_ndim=mom_ndim, parallel=parallel, out=out)


# ** grouped
def factor_by(
    by: Sequence[int | None], sort: bool = False
) -> tuple[NDArray[LongIntDType], NDArray[LongIntDType]]:
    """Factor by to group_idx and groups."""
    from pandas import factorize  # pyright: ignore[reportUnknownVariableType]

    # filter None and negative -> None
    by = [None if x is None or x < 0 else x for x in by]

    # convert to codes
    codes, groups = factorize(np.array(by, dtype=object), sort=sort)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue]
    return codes.astype(np.int64), groups.astype(np.int64)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


# @docfiller.decorate
# def reduce_data_grouped(
#     data: NDArray[T_Float] | xr.DataArray,
#     *,
#     mom_ndim: Mom_NDim,
#     by: ArrayLike,
#     axis: int | None = None,
#     dim: Hashable | None = None,
#     order: ArrayOrder = None,
#     parallel: bool | None = None,
#     out: NDArray[T_Float] | None = None,  # TODO(wpk): check that I can have T_Float on this.
#     # xarray specific
#     dtype: DTypeLike | None = None,
#     coords_policy: CoordsPolicy | Literal["group"] = "first",
#     group_name: Hashable | None = None,
#     rename_dim: Hashable | None = None,
#     keep_attrs: KeepAttrs = None,
# ) -> NDArray[T_Float]:
#     """
#     Reduce data by group.


#     Parameters
#     ----------
#     {data_numpy_or_dataarray}
#     {mom_ndim}
#     {by}
#     {axis_data}
#     {dim}
#     {order}
#     {parallel}
#     {out}

#     Returns
#     -------
#     out : ndarray or DataArray.
#         Reduced data of same type as input ``data``. The last dimensions are
#         "group", followed by moments. ``out.shape = (..., shape[axis-1],
#         shape[axis+1], ..., ngroup, mom0, ...)`` where ``shape = data.shape``.
#     """
#     by = np.asarray(by, dtype=np.int64)
#     mom_ndim = validate_mom_ndim(mom_ndim)

#     if isinstance(data, xr.DataArray):
#         dim, data = xprepare_data_for_reduction(data, axis=axis, dim=dim, mom_ndim=mom_ndim, order=order, dtype=dtype)

#         core_dims = list(data.dims[-(mom_ndim + 1)])
#         return xr.apply_ufunc(  # type: ignore[no-any-return]
#             _reduce_data_grouped,
#             data, by,
#             input_core_dims=[core_dims, [dim]],
#             output_core_dims=[core_dims],
#             exclude_dims={dim},
#             kwargs={"mom_ndim": mom_ndim, "parallel": parallel, "out": out}
#         )


#         if coords_policy in {"first", "last"}:
#             dim_select = index[
#                 group_end - 1 if coords_policy == "last" else group_start
#             ]
#             # fix up coordinates
#             out = self._replace_coords_isel(
#                 out,
#                 {dim: dim_select},
#             )
#         elif coords_policy == "group":
#             group_name = dim

#         if group_name:
#             out = out.assign_coords({group_name: (dim, groups)})  # pyright: ignore[reportUnknownMemberType]

#         if rename_dim:
#             out = out.rename({dim: rename_dim})


#     data = prepare_data_for_reduction(
#         data=data, axis=axis, mom_ndim=mom_ndim, order=order
#     )
#     return _reduce_data_grouped(
#         data, mom_ndim=mom_ndim, by=by, parallel=parallel, out=out
#     )


@docfiller.decorate
def reduce_data_indexed(
    data: NDArray[T_Float],
    *,
    mom_ndim: Mom_NDim,
    index: NDArray[LongIntDType],
    group_start: NDArray[LongIntDType],
    group_end: NDArray[LongIntDType],
    scale: NDArray[T_Float] | None = None,
    axis: int = -1,
    order: ArrayOrder = None,
    parallel: bool | None = None,
    out: NDArray[T_Float] | None = None,
) -> NDArray[T_Float]:
    """
    Reduce data by index

    Parameters
    ----------
    {data_numpy}
    {mom_ndim}
    index : ndarray
        Index into `data.shape[axis]`.
    group_start, group_end : ndarray
        Start, end of index for a group.
        ``index[group_start[group]:group_end[group]]`` are the indices for
        group ``group``.
    scale : ndarray, optional
        Weights of same size as ``index``.
    {axis_data}
    {order}
    {parallel}
    {out}

    Returns
    -------
    out : ndarray
        Reduced data. The last dimensions are `group` and `moments`.
        ``out.shape = (..., shape[axis-1], shape[axis+1], ..., ngroup, mom0,
        ...)``, where ``shape = data.shape`` and ``ngroup = len(group_start)``.
    """
    mom_ndim = validate_mom_ndim(mom_ndim)
    data = prepare_data_for_reduction(
        data=data, axis=axis, mom_ndim=mom_ndim, order=order
    )
    return _reduce_data_indexed(
        data,
        mom_ndim=mom_ndim,
        index=index,
        group_start=group_start,
        group_end=group_end,
        scale=scale,
        parallel=parallel,
        out=out,
    )

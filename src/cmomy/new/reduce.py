"""
Routines to perform central moments reduction (:mod:`~cmomy.reduce)
===================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .docstrings import docfiller
from .utils import (
    parallel_heuristic,
    prepare_data_for_reduction,
    prepare_values_for_reduction,
    raise_if_wrong_shape,
    validate_mom_and_mom_ndim,
    validate_mom_ndim,
)

if TYPE_CHECKING:
    from typing import Sequence

    from numpy.typing import ArrayLike, NDArray

    from .typing import ArrayOrder, LongIntDType, Mom_NDim, Moments
    from .typing import T_FloatDType as T_Float


@docfiller.decorate
def reduce_vals(
    *args: NDArray[T_Float],
    mom: Moments,
    weight: ArrayLike | None = None,
    axis: int = -1,
    order: ArrayOrder = None,
    parallel: bool | None = None,
    out: NDArray[T_Float] | None = None,
) -> NDArray[T_Float]:
    """
    Reduce values to central (co)moments.

    Parameters
    ----------
    *args : ndarray
        Values to analyze.
    {mom}
    {weight}
    {axis}
    {order}
    {parallel}
    {out}

    Returns
    -------
    out : ndarray
        Central moments array. ``out.shape = (...,shape[axis-1], shape[axis+1],
        ..., mom0, ...)`` where ``shape = args[0].shape``.

    """
    mom_validated, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)

    if len(args) != mom_ndim:
        msg = f"Number of arrays {len(args)} != {mom_ndim=}"
        raise ValueError(msg)

    weight = 1.0 if weight is None else weight
    x0, *x1, w = prepare_values_for_reduction(*args, weight, axis=axis, order=order)

    out_shape: tuple[int, ...] = (*x0.shape[:-1], *(m + 1 for m in mom_validated))  # type: ignore[has-type]

    if out is None:
        out = np.zeros(out_shape, dtype=x0.dtype)
    else:
        raise_if_wrong_shape(out, out_shape)
        out.fill(0.0)

    from ._lib.factory import factory_reduce_vals

    factory_reduce_vals(  # type: ignore[call-arg, type-var]
        mom_ndim=mom_ndim,
        parallel=parallel_heuristic(parallel, x0.size * mom_ndim),
    )(x0, *x1, w, out)  # pyright: ignore[reportCallIssue]

    return out


@docfiller.decorate
def reduce_data(
    data: NDArray[T_Float],
    *,
    mom_ndim: Mom_NDim,
    axis: int = -1,
    order: ArrayOrder = None,
    parallel: bool | None = None,
    out: NDArray[T_Float] | None = None,
) -> NDArray[T_Float]:
    """
    Reduce central moments array along axis.


    Parameters
    ----------
    {data_numpy}
    {mom_ndim}
    {axis_data}
    {order}
    {parallel}
    {out}

    Returns
    -------
    out : ndarray
        Reduced data array with shape ``data.shape`` with ``axis`` removed.
    """
    mom_ndim = validate_mom_ndim(mom_ndim)
    data = prepare_data_for_reduction(data, axis=axis, mom_ndim=mom_ndim, order=order)

    from ._lib.factory import factory_reduce_data

    _reduce = factory_reduce_data(
        mom_ndim=mom_ndim,
        parallel=parallel_heuristic(parallel, data.size * mom_ndim),
    )
    if out:
        return _reduce(data, out)
    return _reduce(data)


def factor_by(
    by: Sequence[int | None], sort: bool = False
) -> tuple[NDArray[LongIntDType], NDArray[LongIntDType]]:
    """Factor by to group_idx and groups."""
    from pandas import factorize

    # filter None and negative -> None
    by = [None if x is None or x < 0 else x for x in by]

    # convert to codes
    codes, groups = factorize(np.array(by, dtype=object), sort=sort)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue]
    return codes.astype(np.int64), groups.astype(np.int64)


@docfiller.decorate
def reduce_data_grouped(
    data: NDArray[T_Float],
    *,
    mom_ndim: Mom_NDim,
    by: NDArray[LongIntDType],
    axis: int = -1,
    order: ArrayOrder = None,
    parallel: bool | None = None,
    out: NDArray[T_Float] | None = None,
) -> NDArray[T_Float]:
    """
    Reduce data by group.


    Parameters
    ----------
    {data_numpy}
    {by}
    {axis_data}
    {mom_ndim}
    {order}
    {parallel}
    {out}

    Returns
    -------
    out : ndarray
        Reduced data. The last dimensions are "group", followed by moments.
        ``out.shape = (..., shape[axis-1], shape[axis+1], ..., ngroup, mom0,
        ...)`` where ``shape = data.shape``.
    """
    mom_ndim = validate_mom_ndim(mom_ndim)
    data = prepare_data_for_reduction(
        data=data, axis=axis, mom_ndim=mom_ndim, order=order
    )

    ngroup = by.max() + 1

    out_shape = (*data.shape[: -(mom_ndim + 1)], ngroup, *data.shape[-mom_ndim:])
    if out is None:
        out = np.zeros(out_shape, dtype=data.dtype)
    else:
        raise_if_wrong_shape(out, out_shape)
        out.fill(0.0)

    # print(data.shape, out.shape)

    from ._lib.factory import factory_reduce_data_grouped

    factory_reduce_data_grouped(
        mom_ndim=mom_ndim,
        parallel=parallel_heuristic(parallel, data.size * mom_ndim),
    )(data, by, out)

    return out


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

    if scale is None:
        scale = np.ones_like(index, dtype=data.dtype)

    from ._lib.factory import factory_reduce_data_indexed

    _reduce = factory_reduce_data_indexed(
        mom_ndim=mom_ndim,
        parallel=parallel_heuristic(parallel, data.size * mom_ndim),
    )

    if out is None:
        return _reduce(data, index, group_start, group_end, scale)
    return _reduce(data, index, group_start, group_end, scale, out)

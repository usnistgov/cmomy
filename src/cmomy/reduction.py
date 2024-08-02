"""
Routines to perform central moments reduction (:mod:`~cmomy.reduction`)
=======================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import pandas as pd
import xarray as xr

from ._lib.factory import (
    factory_reduce_data,
    factory_reduce_data_grouped,
    factory_reduce_data_indexed,
    factory_reduce_vals,
)
from .core.docstrings import docfiller
from .core.missing import MISSING
from .core.prepare import (
    prepare_data_for_reduction,
    prepare_out_from_values,
    prepare_values_for_reduction,
    xprepare_out_for_resample_data,
    xprepare_values_for_reduction,
)
from .core.utils import (
    axes_data_reduction,
    get_axes_from_values,
    normalize_axis_tuple,
    optional_keepdims,
    raise_if_wrong_shape,
    replace_coords_from_isel,
    select_axis_dim,
    select_axis_dim_mult,
    select_dtype,
)
from .core.validate import (
    validate_axis_mult,
    validate_mom_and_mom_ndim,
    validate_mom_dims,
    validate_mom_ndim,
)

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from .core.typing import (
        ArrayLikeArg,
        ArrayT,
        AxesGUFunc,
        AxisReduce,
        AxisReduceMult,
        CoordsPolicy,
        DimsReduce,
        DimsReduceMult,
        DTypeLikeArg,
        FloatT,
        Groups,
        IndexAny,
        KeepAttrs,
        MissingType,
        Mom_NDim,
        MomDims,
        Moments,
        MomentsStrict,
        NDArrayAny,
        NDArrayInt,
    )


# * Reduce vals ---------------------------------------------------------------
# ** overloads
@overload
def reduce_vals(  # pyright: ignore[reportOverlappingOverload]
    x: xr.DataArray,
    *y: ArrayLike | xr.DataArray,
    mom: Moments,
    weight: ArrayLike | xr.DataArray | None = ...,
    axis: AxisReduce | MissingType = ...,
    keepdims: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> xr.DataArray: ...
# array
@overload
def reduce_vals(
    x: ArrayLikeArg[FloatT],
    *y: ArrayLike,
    mom: Moments,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    keepdims: bool = ...,
    parallel: bool | None = ...,
    dtype: None = ...,
    out: None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# out
@overload
def reduce_vals(
    x: Any,
    *y: ArrayLike,
    mom: Moments,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    keepdims: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArray[FloatT],
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def reduce_vals(
    x: Any,
    *y: ArrayLike,
    mom: Moments,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    keepdims: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLikeArg[FloatT],
    out: None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# fallback
@overload
def reduce_vals(
    x: Any,
    *y: ArrayLike,
    mom: Moments,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    keepdims: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArrayAny: ...


# TODO(wpk): add tests for keepdims...
@docfiller.decorate
def reduce_vals(  # pyright: ignore[reportOverlappingOverload]
    x: ArrayLike | xr.DataArray,
    *y: ArrayLike | xr.DataArray,
    mom: Moments,
    weight: ArrayLike | xr.DataArray | None = None,
    axis: AxisReduce | MissingType = MISSING,
    keepdims: bool = False,
    parallel: bool | None = None,
    dtype: DTypeLike = None,
    out: NDArrayAny | None = None,
    # xarray specific
    dim: DimsReduce | MissingType = MISSING,
    mom_dims: MomDims | None = None,
    keep_attrs: KeepAttrs = None,
) -> NDArrayAny | xr.DataArray:
    """
    Reduce values to central (co)moments.

    Parameters
    ----------
    x : ndarray or DataArray
        Values to analyze.
    *y : array-like or DataArray
        Seconda value. Must specify if ``len(mom) == 2.`` Should either be able
        to broadcast to ``x`` or be 1d array with length ``x.shape[axis]``.
    {mom}
    weight : scalar or array-like or DataArray
        Weights for each point. Should either be able to broadcast to ``x`` or
        be `d array of length ``x.shape[axis]``.
    {axis}
    {keepdims}
    {parallel}
    {dtype}
    {out}
    {dim}
    {mom_dims}
    {keep_attrs}

    Returns
    -------
    out : ndarray or DataArray
        Central moments array of same type as ``x``. ``out.shape = shape +
        (mom0, ...)`` where ``shape = np.broadcast_shapes(*(a.shape for a in
        (x_, *y_, weight_)))[:-1]`` and ``x_``, ``y_`` and ``weight_`` are the
        input arrays with ``axis`` moved to the last axis.



    See Also
    --------
    numpy.broadcast_shapes
    """
    mom_validated, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
    dtype = select_dtype(x, out=out, dtype=dtype)
    weight = 1.0 if weight is None else weight

    if isinstance(x, xr.DataArray):
        input_core_dims, xargs = xprepare_values_for_reduction(
            x,
            weight,
            *y,
            axis=axis,
            dim=dim,
            narrays=mom_ndim + 1,
            dtype=dtype,
        )

        mom_dims_strict = validate_mom_dims(mom_dims=mom_dims, mom_ndim=mom_ndim)
        dim = input_core_dims[0][0]

        xout: xr.DataArray = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _reduce_vals,
            *xargs,
            input_core_dims=input_core_dims,
            output_core_dims=[(dim, *mom_dims_strict)],
            exclude_dims={dim},
            kwargs={
                "mom": mom_validated,
                "mom_ndim": mom_ndim,
                "parallel": parallel,
                "axis_neg": -1,
                "keepdims": True,
                "out": out,
            },
            keep_attrs=keep_attrs,
        ).transpose(..., *x.dims, *mom_dims_strict)

        if not keepdims:
            xout = xout.squeeze(dim)
        return xout

    axis_neg, args = prepare_values_for_reduction(
        x,
        weight,
        *y,
        axis=axis,
        dtype=dtype,
        narrays=mom_ndim + 1,
        move_axis_to_end=False,
    )

    return _reduce_vals(
        *args,
        mom=mom_validated,
        mom_ndim=mom_ndim,
        axis_neg=axis_neg,
        parallel=parallel,
        out=out,
        keepdims=keepdims,
    )


# Low level
def _reduce_vals(
    x0: NDArray[FloatT],
    w: NDArray[FloatT],
    *x1: NDArray[FloatT],
    mom: MomentsStrict,
    mom_ndim: Mom_NDim,
    axis_neg: int,
    parallel: bool | None = None,
    out: NDArray[FloatT] | None = None,
    keepdims: bool = False,
) -> NDArray[FloatT]:
    args: tuple[NDArray[FloatT], ...] = (x0, w, *x1)  # type: ignore[arg-type]
    out = prepare_out_from_values(out, *args, mom=mom, axis_neg=axis_neg)

    axes: AxesGUFunc = [
        # out
        tuple(range(-mom_ndim, 0)),
        # x, weight, *y
        *get_axes_from_values(*args, axis_neg=axis_neg),
    ]
    factory_reduce_vals(
        mom_ndim=mom_ndim,
        parallel=parallel,
        size=args[0].size,
    )(out, *args, axes=axes)

    return optional_keepdims(
        out,
        axis=axis_neg - mom_ndim,
        keepdims=keepdims,
    )


# * Reduce data ---------------------------------------------------------------
# ** overload
@overload
def reduce_data(
    data: xr.DataArray,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduceMult | MissingType = ...,
    keepdims: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
    dim: DimsReduceMult | MissingType = ...,
    keep_attrs: bool | None = ...,
) -> xr.DataArray: ...
# array no output
@overload
def reduce_data(
    data: ArrayLikeArg[FloatT],
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduceMult | MissingType = ...,
    keepdims: bool = ...,
    parallel: bool | None = ...,
    dtype: None = ...,
    out: None = ...,
    dim: DimsReduceMult | MissingType = ...,
    keep_attrs: bool | None = ...,
) -> NDArray[FloatT]: ...
# out
@overload
def reduce_data(
    data: Any,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduceMult | MissingType = ...,
    keepdims: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArray[FloatT],
    dim: DimsReduceMult | MissingType = ...,
    keep_attrs: bool | None = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def reduce_data(
    data: Any,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduceMult | MissingType = ...,
    keepdims: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLikeArg[FloatT],
    out: None = ...,
    dim: DimsReduceMult | MissingType = ...,
    keep_attrs: bool | None = ...,
) -> NDArray[FloatT]: ...
# fallback
@overload
def reduce_data(
    data: Any,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduceMult | MissingType = ...,
    keepdims: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    dim: DimsReduceMult | MissingType = ...,
    keep_attrs: bool | None = ...,
) -> NDArrayAny: ...


# ** public
@docfiller.decorate
def reduce_data(
    data: xr.DataArray | ArrayLike,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduceMult | MissingType = MISSING,
    keepdims: bool = False,
    parallel: bool | None = None,
    dtype: DTypeLike = None,
    out: NDArrayAny | None = None,
    # xarray specific
    dim: DimsReduceMult | MissingType = MISSING,
    keep_attrs: bool | None = None,
) -> xr.DataArray | NDArrayAny:
    """
    Reduce central moments array along axis.


    Parameters
    ----------
    {data_numpy_or_dataarray}
    {mom_ndim}
    {axis_data_mult}
    {keepdims}
    {parallel}
    {dtype}
    {out}
    {dim_mult}
    {keep_attrs}

    Returns
    -------
    out : ndarray or DataArray
        Reduced data array with shape ``data.shape`` with ``axis`` removed.
        Same type as input ``data``.
    """
    if isinstance(data, xr.DataArray):
        axis, dim = select_axis_dim_mult(data, axis=axis, dim=dim, mom_ndim=mom_ndim)
        return data.reduce(
            reduce_data,
            dim=dim,
            keep_attrs=keep_attrs,
            keepdims=keepdims,
            mom_ndim=mom_ndim,
            parallel=parallel,
            dtype=dtype,
            out=out,
        )

    mom_ndim = validate_mom_ndim(mom_ndim)
    dtype = select_dtype(data, out=out, dtype=dtype)

    # special to support multiple reduction dimensions...
    data = np.asarray(data, dtype=dtype)
    ndim = data.ndim - mom_ndim
    axis_tuple = normalize_axis_tuple(
        validate_axis_mult(axis),
        ndim,
        msg_prefix="reduce_data",
    )

    # move axis to end and reshape
    data = np.moveaxis(data, axis_tuple, range(ndim - len(axis_tuple), ndim))
    new_shape = (
        *data.shape[: -(len(axis_tuple) + mom_ndim)],
        -1,
        *data.shape[-mom_ndim:],
    )
    data = data.reshape(new_shape)

    out = factory_reduce_data(
        mom_ndim=mom_ndim,
        parallel=parallel,
        size=data.size,
    )(data, out=out, dtype=dtype)

    return optional_keepdims(
        out,
        axis=axis_tuple,
        keepdims=keepdims,
    )


# * Grouped -------------------------------------------------------------------
# ** utilities
def factor_by(
    by: Groups,
    sort: bool = True,
) -> tuple[list[Any] | IndexAny | pd.MultiIndex, NDArrayInt]:
    """
    Factor by to codes and groups.

    Parameters
    ----------
    by : sequence
        Values to group by. Negative or ``None`` values indicate to skip this
        value. Note that if ``by`` is a pandas :class:`pandas.Index` object,
        missing values should be marked with ``None`` only.
    sort : bool, default=True
        If ``True`` (default), sort ``groups``.
        If ``False``, return groups in order of first appearance.

    Returns
    -------
    groups : list or :class:`pandas.Index`
        Unique group names (excluding negative or ``None`` Values.)
    codes : ndarray of int
        Indexer into ``groups``.


    Examples
    --------
    >>> by = [1, 1, 0, -1, 0, 2, 2]
    >>> groups, codes = factor_by(by, sort=False)
    >>> groups
    [1, 0, 2]
    >>> codes
    array([ 0,  0,  1, -1,  1,  2,  2])

    Note that with sort=False, groups are in order of first appearance.

    >>> groups, codes = factor_by(by)
    >>> groups
    [0, 1, 2]
    >>> codes
    array([ 1,  1,  0, -1,  0,  2,  2])

    This also works for sequences of non-intengers.

    >>> by = ["a", "a", None, "c", "c", -1]
    >>> groups, codes = factor_by(by)
    >>> groups
    ['a', 'c']
    >>> codes
    array([ 0,  0, -1,  1,  1, -1])

    And for :class:`pandas.Index` objects

    >>> import pandas as pd
    >>> by = pd.Index(["a", "a", None, "c", "c", None])
    >>> groups, codes = factor_by(by)
    >>> groups
    Index(['a', 'c'], dtype='object')
    >>> codes
    array([ 0,  0, -1,  1,  1, -1])

    """
    from pandas import factorize  # pyright: ignore[reportUnknownVariableType]

    # filter None and negative -> None
    _by: Groups
    if isinstance(by, pd.Index):
        _by = by
    else:
        _by = np.fromiter(
            (None if isinstance(x, (int, np.integer)) and x < 0 else x for x in by),  # pyright: ignore[reportUnknownArgumentType]
            dtype=object,
        )

    codes, groups = factorize(_by, sort=sort)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]

    codes = codes.astype(np.int64)
    if isinstance(_by, (pd.Index, pd.MultiIndex)):
        if not isinstance(groups, (pd.Index, pd.MultiIndex)):  # pragma: no cover
            msg = f"{type(groups)=} should be instance of pd.Index"  # pyright: ignore[reportUnknownArgumentType]
            raise TypeError(msg)
        groups.names = _by.names
        return groups, codes  # pyright: ignore[reportUnknownVariableType]

    return list(groups), codes  # pyright: ignore[reportUnknownArgumentType]


# ** overload
@overload
def reduce_data_grouped(
    data: xr.DataArray,
    by: ArrayLike,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    group_dim: str | None = ...,
    groups: Groups | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> xr.DataArray: ...
# Array no output or dtype
@overload
def reduce_data_grouped(
    data: ArrayLikeArg[FloatT],
    by: ArrayLike,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    out: None = ...,
    dtype: None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    group_dim: str | None = ...,
    groups: Groups | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# out
@overload
def reduce_data_grouped(
    data: Any,
    by: ArrayLike,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    group_dim: str | None = ...,
    groups: Groups | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def reduce_data_grouped(
    data: Any,
    by: ArrayLike,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    group_dim: str | None = ...,
    groups: Groups | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# fallback
@overload
def reduce_data_grouped(
    data: Any,
    by: ArrayLike,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    out: None = ...,
    dtype: DTypeLike = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    group_dim: str | None = ...,
    groups: Groups | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArrayAny: ...


# ** public
@docfiller.decorate
def reduce_data_grouped(
    data: xr.DataArray | ArrayLike,
    by: ArrayLike,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = MISSING,
    move_axis_to_end: bool = False,
    parallel: bool | None = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    # xarray specific
    dim: DimsReduce | MissingType = MISSING,
    group_dim: str | None = None,
    groups: Groups | None = None,
    keep_attrs: KeepAttrs = None,
) -> xr.DataArray | NDArrayAny:
    """
    Reduce data by group.


    Parameters
    ----------
    {data_numpy_or_dataarray}
    {mom_ndim}
    {by}
    {axis_data}
    {move_axis_to_end}
    {parallel}
    {out}
    {dtype}
    {dim}
    {group_dim}
    {groups}
    {keep_attrs}

    Returns
    -------
    out : ndarray or DataArray
        Reduced data of same type as input ``data``, with shape
        ``out.shape = (..., shape[axis-1], ngroup, shape[axis+1], ..., mom0, ...)``
        where ``shape = data.shape`` and ngroups = ``by.max() + 1``.


    See Also
    --------
    factor_by

    Examples
    --------
    >>> data = np.ones((5, 3))
    >>> by = [0, 0, -1, 1, -1]
    >>> reduce_data_grouped(data, mom_ndim=1, axis=0, by=by)
    array([[2., 1., 1.],
           [1., 1., 1.]])

    This also works for :class:`~xarray.DataArray` objects.  In this case,
    the groups are added as coordinates to ``group_dim``

    >>> xout = xr.DataArray(data, dims=["rec", "mom"])
    >>> reduce_data_grouped(xout, mom_ndim=1, dim="rec", by=by, group_dim="group")
    <xarray.DataArray (group: 2, mom: 3)> Size: 48B
    array([[2., 1., 1.],
           [1., 1., 1.]])
    Dimensions without coordinates: group, mom

    Note that if ``by`` skips some groups, they will still be included in
    The output.  For example the following ``by`` skips the value `0`.

    >>> by = [1, 1, -1, 2, 2]
    >>> reduce_data_grouped(xout, mom_ndim=1, dim="rec", by=by)
    <xarray.DataArray (rec: 3, mom: 3)> Size: 72B
    array([[0., 0., 0.],
           [2., 1., 1.],
           [2., 1., 1.]])
    Dimensions without coordinates: rec, mom

    If you want to ensure that only included groups are used, use :func:`factor_by`.
    This has the added benefit of working with non integer groups as well

    >>> by = ["a", "a", None, "b", "b"]
    >>> groups, codes = factor_by(by)
    >>> reduce_data_grouped(xout, mom_ndim=1, dim="rec", by=codes, groups=groups)
    <xarray.DataArray (rec: 2, mom: 3)> Size: 48B
    array([[2., 1., 1.],
           [2., 1., 1.]])
    Coordinates:
      * rec      (rec) <U1 8B 'a' 'b'
    Dimensions without coordinates: mom


    """
    if isinstance(data, xr.DataArray):
        axis, dim = select_axis_dim(data, axis=axis, dim=dim, mom_ndim=mom_ndim)
        core_dims = (dim, *data.dims[-mom_ndim:])

        xout: xr.DataArray = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            reduce_data_grouped,
            data,
            by,
            input_core_dims=[core_dims, [dim]],
            output_core_dims=[core_dims],
            exclude_dims={dim},
            kwargs={
                "mom_ndim": mom_ndim,
                "parallel": parallel,
                "out": xprepare_out_for_resample_data(
                    out,
                    mom_ndim=mom_ndim,
                    axis=axis,
                    move_axis_to_end=move_axis_to_end,
                ),
                "dtype": dtype,
                "axis": -1,
            },
            keep_attrs=keep_attrs,
        )

        if not move_axis_to_end:
            xout = xout.transpose(*data.dims)
        if groups is not None:
            xout = xout.assign_coords({dim: (dim, groups)})  # pyright: ignore[reportUnknownMemberType]
        if group_dim:
            xout = xout.rename({dim: group_dim})
        return xout

    mom_ndim = validate_mom_ndim(mom_ndim)
    dtype = select_dtype(data, out=out, dtype=dtype)

    axis, data = prepare_data_for_reduction(  # pyright: ignore[reportArgumentType]
        data=data,
        axis=axis,
        mom_ndim=mom_ndim,
        dtype=dtype,
        move_axis_to_end=move_axis_to_end,
    )

    # include inner core dims for by
    axes = axes_data_reduction((-1,), mom_ndim=mom_ndim, axis=axis, out_has_axis=True)

    # check by
    by = np.asarray(by, dtype=np.int64)
    if len(by) != data.shape[axis]:
        msg = f"{len(by)=} != {data.shape[axis]=}"
        raise ValueError(msg)

    ngroup = by.max() + 1
    out_shape = (*data.shape[:axis], ngroup, *data.shape[axis + 1 :])
    if out is None:
        out = np.zeros(out_shape, dtype=data.dtype)
    else:
        raise_if_wrong_shape(out, out_shape)
        out.fill(0.0)

    factory_reduce_data_grouped(
        mom_ndim=mom_ndim,
        parallel=parallel,
        size=data.size,
    )(data, by, out, axes=axes)

    return out


# * Indexed -------------------------------------------------------------------
# ** utils
@docfiller.decorate
def factor_by_to_index(
    by: Groups,
) -> tuple[list[Any] | IndexAny | pd.MultiIndex, NDArrayInt, NDArrayInt, NDArrayInt]:
    """
    Transform group_idx to quantities to be used with :func:`reduce_data_indexed`.

    Parameters
    ----------
    by: array-like
        Values to factor.
    exclude_missing : bool, default=True
        If ``True`` (default), filter Negative and ``None`` values from ``group_idx``.

    Returns
    -------
    groups : list or :class:`pandas.Index`
        Unique groups in `group_idx` (excluding Negative or ``None`` values in
        ``group_idx`` if ``exclude_negative`` is ``True``).
    index : ndarray
        Indexing array. ``index[start[k]:end[k]]`` are the index with group
        ``groups[k]``.
    start : ndarray
        See ``index``
    end : ndarray
        See ``index``.

    See Also
    --------
    reduce_data_indexed
    factor_by

    Examples
    --------
    >>> factor_by_to_index([0, 1, 0, 1])
    ([0, 1], array([0, 2, 1, 3]), array([0, 2]), array([2, 4]))

    >>> factor_by_to_index(["a", "b", "a", "b"])
    (['a', 'b'], array([0, 2, 1, 3]), array([0, 2]), array([2, 4]))

    Also, missing values (None or negative) are excluded:

    >>> factor_by_to_index([None, "a", None, "b"])
    (['a', 'b'], array([1, 3]), array([0, 1]), array([1, 2]))

    You can also pass :class:`pandas.Index` objects:

    >>> factor_by_to_index(pd.Index([None, "a", None, "b"], name="my_index"))
    (Index(['a', 'b'], dtype='object', name='my_index'), array([1, 3]), array([0, 1]), array([1, 2]))

    """
    # factorize by to groups and codes
    groups, codes = factor_by(by, sort=True)

    # exclude missing
    keep = codes >= 0
    if not np.all(keep):
        index = np.where(keep)[0]
        codes = codes[keep]
    else:
        index = None

    indexes_sorted = np.argsort(codes)
    group_idx_sorted = codes[indexes_sorted]
    _groups, n_start, count = np.unique(
        group_idx_sorted, return_index=True, return_counts=True
    )
    n_end = n_start + count

    if index is not None:
        indexes_sorted = index[indexes_sorted]

    return groups, indexes_sorted, n_start, n_end


def _validate_index(
    ndat: int,
    index: ArrayLike,
    group_start: ArrayLike,
    group_end: ArrayLike,
) -> tuple[NDArrayInt, NDArrayInt, NDArrayInt]:
    def _validate(name: str, x: NDArrayInt, upper: int) -> None:
        min_: int = x.min()
        max_: int = x.max()
        if min_ < 0:
            msg = f"min({name}) = {min_} < 0"
            raise ValueError(msg)
        if max_ >= upper:
            msg = f"max({name}) = {max_} > {upper}"
            raise ValueError(msg)

    index = np.asarray(index, dtype=np.int64)
    group_start = np.asarray(group_start, dtype=np.int64)
    group_end = np.asarray(group_end, dtype=np.int64)

    # TODO(wpk): if nindex == 0, should just get out of here?
    nindex = len(index)
    if nindex == 0:
        if (group_start != 0).any() or (group_end != 0).any():
            msg = "With zero length index, group_start = group_stop = 0"
            raise ValueError(msg)
    else:
        _validate("index", index, ndat)
        _validate("group_start", group_start, nindex)
        _validate("group_end", group_end, nindex + 1)

    if len(group_start) != len(group_end):
        msg = "len(start) != len(end)"
        raise ValueError(msg)

    if (group_end < group_start).any():
        msg = "Found end < start"
        raise ValueError(msg)

    return index, group_start, group_end


# ** overload
@overload
def reduce_data_indexed(
    data: xr.DataArray,
    *,
    mom_ndim: Mom_NDim,
    index: ArrayLike,
    group_start: ArrayLike,
    group_end: ArrayLike,
    scale: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    # xarray specific...
    dim: DimsReduce | MissingType = ...,
    coords_policy: CoordsPolicy = ...,
    group_dim: str | None = ...,
    groups: Groups | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> xr.DataArray: ...
# Array no out or dtype
@overload
def reduce_data_indexed(
    data: ArrayLikeArg[FloatT],
    *,
    mom_ndim: Mom_NDim,
    index: ArrayLike,
    group_start: ArrayLike,
    group_end: ArrayLike,
    scale: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    out: None = ...,
    dtype: None = ...,
    # xarray specific...
    dim: DimsReduce | MissingType = ...,
    coords_policy: CoordsPolicy = ...,
    group_dim: str | None = ...,
    groups: Groups | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# out
@overload
def reduce_data_indexed(
    data: Any,
    *,
    mom_ndim: Mom_NDim,
    index: ArrayLike,
    group_start: ArrayLike,
    group_end: ArrayLike,
    scale: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    # xarray specific...
    dim: DimsReduce | MissingType = ...,
    coords_policy: CoordsPolicy = ...,
    group_dim: str | None = ...,
    groups: Groups | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def reduce_data_indexed(
    data: Any,
    *,
    mom_ndim: Mom_NDim,
    index: ArrayLike,
    group_start: ArrayLike,
    group_end: ArrayLike,
    scale: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    # xarray specific...
    dim: DimsReduce | MissingType = ...,
    coords_policy: CoordsPolicy = ...,
    group_dim: str | None = ...,
    groups: Groups | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# fallback
@overload
def reduce_data_indexed(
    data: Any,
    *,
    mom_ndim: Mom_NDim,
    index: ArrayLike,
    group_start: ArrayLike,
    group_end: ArrayLike,
    scale: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    out: None = ...,
    dtype: DTypeLike = ...,
    # xarray specific...
    dim: DimsReduce | MissingType = ...,
    coords_policy: CoordsPolicy = ...,
    group_dim: str | None = ...,
    groups: Groups | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArrayAny: ...


# ** public
@docfiller.decorate
def reduce_data_indexed(  # noqa: PLR0913
    data: xr.DataArray | ArrayLike,
    *,
    mom_ndim: Mom_NDim,
    index: ArrayLike,
    group_start: ArrayLike,
    group_end: ArrayLike,
    scale: ArrayLike | None = None,
    axis: AxisReduce | MissingType = MISSING,
    move_axis_to_end: bool = False,
    parallel: bool | None = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    # xarray specific...
    dim: DimsReduce | MissingType = MISSING,
    coords_policy: CoordsPolicy = "first",
    group_dim: str | None = None,
    groups: Groups | None = None,
    keep_attrs: KeepAttrs = None,
) -> xr.DataArray | NDArrayAny:
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
    {move_axis_to_end}
    {parallel}
    {out}
    {dtype}
    {dim}
    {coords_policy}
    {group_dim}
    {groups}
    {keep_attrs}

    Returns
    -------
    out : ndarray or DataArray
        Reduced data of same type as input ``data``, with shape
        ``out.shape = (..., shape[axis-1], ngroup, shape[axis+1], ..., mom0, ...)``,
        where ``shape = data.shape`` and ``ngroup = len(group_start)``.

    See Also
    --------
    factor_by_to_index


    Examples
    --------
    This is a more general reduction than :func:`reduce_data_grouped`, but
    it can be used similarly.

    >>> data = np.ones((5, 3))
    >>> by = ["a", "a", "b", "b", "c"]
    >>> groups, index, start, end = factor_by_to_index(by)
    >>> reduce_data_indexed(
    ...     data, mom_ndim=1, axis=0, index=index, group_start=start, group_end=end
    ... )
    array([[2., 1., 1.],
           [2., 1., 1.],
           [1., 1., 1.]])

    This also works for :class:`~xarray.DataArray` objects

    >>> xout = xr.DataArray(data, dims=["rec", "mom"])
    >>> reduce_data_indexed(
    ...     xout,
    ...     mom_ndim=1,
    ...     dim="rec",
    ...     index=index,
    ...     group_start=start,
    ...     group_end=end,
    ...     group_dim="group",
    ...     groups=groups,
    ...     coords_policy="group",
    ... )
    <xarray.DataArray (group: 3, mom: 3)> Size: 72B
    array([[2., 1., 1.],
           [2., 1., 1.],
           [1., 1., 1.]])
    Coordinates:
      * group    (group) <U1 12B 'a' 'b' 'c'
    Dimensions without coordinates: mom
    """
    if isinstance(data, xr.DataArray):
        axis, dim = select_axis_dim(data, axis=axis, dim=dim, mom_ndim=mom_ndim)
        core_dims = (dim, *data.dims[-mom_ndim:])

        xout: xr.DataArray = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            reduce_data_indexed,
            data,
            input_core_dims=[core_dims],
            output_core_dims=[core_dims],
            exclude_dims={dim},
            kwargs={
                "mom_ndim": mom_ndim,
                "index": index,
                "group_start": group_start,
                "group_end": group_end,
                "scale": scale,
                "parallel": parallel,
                "out": xprepare_out_for_resample_data(
                    out,
                    mom_ndim=mom_ndim,
                    axis=axis,
                    move_axis_to_end=move_axis_to_end,
                ),
                "dtype": dtype,
                "axis": -1,
            },
            keep_attrs=keep_attrs,
        )

        if not move_axis_to_end:
            xout = xout.transpose(*data.dims)

        if coords_policy in {"first", "last"}:
            # in case we passed in index, group_start, group_end as non-arrays
            # these will be processed correctly in the numpy call
            # but just make sure here....
            index, group_start, group_end = (
                np.asarray(_, dtype=np.int64) for _ in (index, group_start, group_end)
            )
            dim_select = index[
                group_end - 1 if coords_policy == "last" else group_start
            ]

            xout = replace_coords_from_isel(
                da_original=data,
                da_selected=xout,
                indexers={dim: dim_select},
                drop=False,
            )
        elif coords_policy == "group" and groups is not None:
            xout = xout.assign_coords({dim: groups})  # pyright: ignore[reportUnknownMemberType]
        if group_dim:
            xout = xout.rename({dim: group_dim})

        return xout

    mom_ndim = validate_mom_ndim(mom_ndim)
    dtype = select_dtype(data, out=out, dtype=dtype)

    axis, data = prepare_data_for_reduction(  # pyright: ignore[reportAssignmentType]
        data=data,
        axis=axis,
        mom_ndim=mom_ndim,
        dtype=dtype,
        move_axis_to_end=move_axis_to_end,
    )

    index, group_start, group_end = _validate_index(
        ndat=data.shape[axis],
        index=index,
        group_start=group_start,
        group_end=group_end,
    )
    return _reduce_data_indexed(
        data=data,
        axis=axis,
        mom_ndim=mom_ndim,
        index=index,
        group_start=group_start,
        group_end=group_end,
        scale=scale,
        parallel=parallel,
        out=out,
    )


def _reduce_data_indexed(
    data: NDArray[FloatT],
    *,
    axis: int,
    mom_ndim: Mom_NDim,
    index: NDArrayInt,
    group_start: NDArrayInt,
    group_end: NDArrayInt,
    scale: ArrayLike | None,
    parallel: bool | None,
    out: NDArrayAny | None,
) -> NDArray[FloatT]:
    """Get reduced_data, index, start, end."""
    # validate index
    if scale is None:
        scale = np.ones(len(index), dtype=data.dtype)
    else:
        scale = np.asarray(scale, dtype=data.dtype)
        if len(scale) != len(index):
            msg = f"{len(scale)=} != {len(index)=}"
            raise ValueError(msg)

    # include inner dims for index, start, end, scale
    axes = axes_data_reduction(
        *(-1,) * 4, mom_ndim=mom_ndim, axis=axis, out_has_axis=True
    )

    return factory_reduce_data_indexed(
        mom_ndim=mom_ndim,
        parallel=parallel,
        size=data.size,
    )(data, index, group_start, group_end, scale, axes=axes, out=out)


# * For testing purposes
def resample_data_indexed(
    data: ArrayT,
    freq: NDArrayAny,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = MISSING,
    move_axis_to_end: bool = False,
    parallel: bool = True,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    # xarray specific
    dim: DimsReduce | MissingType = MISSING,
    coords_policy: CoordsPolicy = "first",
    group_dim: str | None = None,
    groups: Groups | None = None,
    keep_attrs: KeepAttrs = None,
) -> ArrayT:
    """Resample using indexed reduction."""
    from ._lib.utils import freq_to_index_start_end_scales

    index, start, end, scales = freq_to_index_start_end_scales(freq)

    return reduce_data_indexed(  # pyright: ignore[reportUnknownVariableType, reportCallIssue]
        data=data,
        mom_ndim=mom_ndim,
        index=index,
        group_start=start,
        group_end=end,
        scale=scales,
        axis=axis,
        move_axis_to_end=move_axis_to_end,
        parallel=parallel,
        out=out,  # pyright: ignore[reportArgumentType]
        dtype=dtype,
        dim=dim,
        coords_policy=coords_policy,
        group_dim=group_dim,
        groups=groups,
        keep_attrs=keep_attrs,
    )

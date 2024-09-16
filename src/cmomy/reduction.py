"""
Routines to perform central moments reduction (:mod:`~cmomy.reduction`)
=======================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import pandas as pd
import xarray as xr

from .core.array_utils import (
    asarray_maybe_recast,
    axes_data_reduction,
    get_axes_from_values,
    normalize_axis_tuple,
    optional_keepdims,
    select_dtype,
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
from .core.utils import mom_to_mom_shape
from .core.validate import (
    is_dataarray,
    is_dataset,
    is_xarray,
    raise_if_wrong_value,
    validate_axis_mult,
    validate_mom_and_mom_ndim,
    validate_mom_dims,
    validate_mom_ndim,
)
from .core.xr_utils import (
    get_apply_ufunc_kwargs,
    replace_coords_from_isel,
    select_axis_dim,
    select_axis_dim_mult,
)
from .factory import (
    factory_reduce_data,
    factory_reduce_data_grouped,
    factory_reduce_data_indexed,
    factory_reduce_vals,
    parallel_heuristic,
)

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from .core.typing import (
        ApplyUFuncKwargs,
        ArrayLikeArg,
        ArrayOrder,
        ArrayOrderCF,
        ArrayT,
        AxesGUFunc,
        AxisReduce,
        AxisReduceMult,
        BlockByModes,
        Casting,
        CoordsPolicy,
        DataT,
        DimsReduce,
        DimsReduceMult,
        DTypeLikeArg,
        FloatT,
        Groups,
        IndexAny,
        KeepAttrs,
        MissingCoreDimOptions,
        MissingType,
        Mom_NDim,
        MomDims,
        Moments,
        MomentsStrict,
        NDArrayAny,
        NDArrayInt,
        ReduceDataGroupedKwargs,
        ReduceDataIndexedKwargs,
        ReduceDataKwargs,
        ReduceValsKwargs,
    )
    from .core.typing_compat import Unpack


# * Reduce vals ---------------------------------------------------------------
# ** overloads
@overload
def reduce_vals(
    x: DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    weight: ArrayLike | xr.DataArray | DataT | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceValsKwargs],
) -> DataT: ...
# array
@overload
def reduce_vals(
    x: ArrayLikeArg[FloatT],
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[ReduceValsKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def reduce_vals(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceValsKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def reduce_vals(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[ReduceValsKwargs],
) -> NDArray[FloatT]: ...
# fallback Array
@overload
def reduce_vals(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceValsKwargs],
) -> NDArrayAny: ...


@docfiller.decorate
def reduce_vals(
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    mom: Moments,
    weight: ArrayLike | xr.DataArray | DataT | None = None,
    axis: AxisReduce | MissingType = MISSING,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderCF = None,
    keepdims: bool = False,
    parallel: bool | None = None,
    dim: DimsReduce | MissingType = MISSING,
    mom_dims: MomDims | None = None,
    keep_attrs: KeepAttrs = None,
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Reduce values to central (co)moments.

    Parameters
    ----------
    {x_genarray}
    {y_genarray}
    {mom}
    {weight_genarray}
    {axis}
    {out}
    {dtype}
    {casting}
    {order_cf}
    {keepdims}
    {parallel}
    {dim}
    {mom_dims}
    {keep_attrs}
    {on_missing_core_dim}
    {apply_ufunc_kwargs}

    Returns
    -------
    out : ndarray or DataArray or Dataset
        Central moments array of same type as ``x``. ``out.shape = shape +
        (mom0, ...)`` where ``shape = np.broadcast_shapes(*(a.shape for a in
        (x_, *y_, weight_)))[:-1]`` and ``x_``, ``y_`` and ``weight_`` are the
        input arrays with ``axis`` moved to the last axis.



    See Also
    --------
    numpy.broadcast_shapes
    """
    mom, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
    weight = 1.0 if weight is None else weight
    dtype = select_dtype(x, out=out, dtype=dtype)
    if is_xarray(x):
        mom_dims = validate_mom_dims(mom_dims=mom_dims, mom_ndim=mom_ndim)
        dim, input_core_dims, xargs = xprepare_values_for_reduction(
            x,
            weight,
            *y,
            axis=axis,
            dim=dim,
            narrays=mom_ndim + 1,
            dtype=dtype,
            recast=False,
        )

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _reduce_vals,
            *xargs,
            input_core_dims=input_core_dims,
            output_core_dims=[[dim, *mom_dims]],  # type: ignore[misc]  # no clue...
            exclude_dims={dim},
            kwargs={
                "mom": mom,
                "mom_ndim": mom_ndim,
                "parallel": parallel,
                "axis_neg": -1,
                "keepdims": True,
                "out": None if is_dataset(x) else out,
                "dtype": dtype,
                "casting": casting,
                "order": order,
                "fastpath": is_dataarray(x),
            },
            keep_attrs=keep_attrs,
            **get_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                on_missing_core_dim=on_missing_core_dim,
                dask="parallelized",
                output_sizes={dim: 1, **dict(zip(mom_dims, mom_to_mom_shape(mom)))},
                output_dtypes=dtype or np.float64,
            ),
        )

        if is_dataarray(x):
            xout = xout.transpose(..., *x.dims, *mom_dims)
        if not keepdims:
            xout = xout.squeeze(dim)
        return xout

    # Numpy
    axis_neg, args = prepare_values_for_reduction(
        x,
        weight,
        *y,
        axis=axis,
        dtype=dtype,
        recast=False,
        narrays=mom_ndim + 1,
        move_axis_to_end=False,
    )

    return _reduce_vals(
        *args,
        mom=mom,
        mom_ndim=mom_ndim,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        axis_neg=axis_neg,
        parallel=parallel,
        keepdims=keepdims,
        fastpath=True,
    )


def _reduce_vals(
    # x, w, *y
    *args: NDArrayAny,
    mom: MomentsStrict,
    mom_ndim: Mom_NDim,
    axis_neg: int,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrderCF,
    parallel: bool | None,
    keepdims: bool,
    fastpath: bool = False,
) -> NDArrayAny:
    if not fastpath:
        dtype = select_dtype(args[0], out=out, dtype=dtype)

    out = prepare_out_from_values(
        out,
        *args,
        mom=mom,
        axis_neg=axis_neg,
        dtype=dtype,
        order=order,
    )

    axes: AxesGUFunc = [
        # out
        tuple(range(-mom_ndim, 0)),
        # x, weight, *y
        *get_axes_from_values(*args, axis_neg=axis_neg),
    ]

    factory_reduce_vals(
        mom_ndim=mom_ndim,
        parallel=parallel_heuristic(parallel, size=args[0].size * mom_ndim),
    )(
        out,
        *args,
        axes=axes,
        casting=casting,
        order=order,
        signature=(dtype,) * (len(args) + 1),
    )

    return optional_keepdims(
        out,
        axis=axis_neg - mom_ndim,
        keepdims=keepdims,
    )


# * Reduce data ---------------------------------------------------------------
# ** overload
@overload
def reduce_data(
    data: DataT,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceDataKwargs],
) -> DataT: ...
@overload
def reduce_data(
    data: ArrayLikeArg[FloatT],
    *,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[ReduceDataKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def reduce_data(
    data: ArrayLike,
    *,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceDataKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def reduce_data(
    data: ArrayLike,
    *,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[ReduceDataKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def reduce_data(
    data: ArrayLike,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceDataKwargs],
) -> NDArrayAny: ...


# ** public
@docfiller.decorate
def reduce_data(
    data: ArrayLike | DataT,
    *,
    mom_ndim: Mom_NDim = 1,
    axis: AxisReduceMult | MissingType = MISSING,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrder = None,
    keepdims: bool = False,
    parallel: bool | None = None,
    use_reduce: bool = True,
    dim: DimsReduceMult | MissingType = MISSING,
    mom_dims: MomDims | None = None,
    keep_attrs: KeepAttrs = None,
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Reduce central moments array along axis.


    Parameters
    ----------
    {data_numpy_or_dataarray_or_dataset}
    {mom_ndim}
    {axis_data_mult}
    {out}
    {dtype}
    {casting}
    {order}
    {keepdims}
    {parallel}
    use_reduce : bool
        If ``True``, use ``data.reduce(reduce_data, ....)`` for
        :class:`~xarray.DataArray` or :class:`~xarray.Dataset` ``data``.
        Otherwise, use :func:`~xarray.apply_ufunc` for reduction. The later
        will preserve dask based arrays while the former will greedily convert
        dask data to :class:`~numpy.ndarray` arrays. Also, not using reduce Can
        be useful if reducing a dataset which contains arrays that do not
        contain ``mom_dims`` that should be dropped.
    {dim_mult}
    {mom_dims_data}
    {keep_attrs}
    {on_missing_core_dim}
    {apply_ufunc_kwargs}

    Returns
    -------
    out : ndarray or DataArray or Dataset
        Reduced data array with shape ``data.shape`` with ``axis`` removed.
        Same type as input ``data``.
    """
    mom_ndim = validate_mom_ndim(mom_ndim)
    dtype = select_dtype(data, out=out, dtype=dtype)

    if is_xarray(data):
        axis, dim = select_axis_dim_mult(data, axis=axis, dim=dim, mom_ndim=mom_ndim)
        xout: DataT
        if use_reduce:
            xout = data.reduce(  # type: ignore[assignment]
                _reduce_data,
                dim=dim,
                keep_attrs=bool(keep_attrs),
                keepdims=keepdims,
                mom_ndim=mom_ndim,
                parallel=parallel,
                out=out,
                dtype=dtype,
                casting=casting,
                order=order,
            )

        else:
            mom_dims = validate_mom_dims(mom_dims, mom_ndim, data)
            xout = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
                _reduce_data,
                data,
                input_core_dims=[[*dim, *mom_dims]],  # type: ignore[misc]
                output_core_dims=[mom_dims],
                kwargs={
                    "mom_ndim": mom_ndim,
                    "axis": range(-len(dim), 0),
                    "out": None if is_dataset(data) else out,
                    "dtype": dtype,
                    "parallel": parallel,
                    "keepdims": False,
                    "fastpath": is_dataarray(data),
                    "casting": casting,
                    "order": order,
                },
                keep_attrs=keep_attrs,
                **get_apply_ufunc_kwargs(
                    apply_ufunc_kwargs,
                    on_missing_core_dim=on_missing_core_dim,
                    dask="parallelized",
                    output_dtypes=dtype or np.float64,
                ),
            )

        return xout

    # Numpy
    return _reduce_data(
        asarray_maybe_recast(data, dtype=dtype, recast=False),
        mom_ndim=mom_ndim,
        axis=validate_axis_mult(axis),
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        parallel=parallel,
        keepdims=keepdims,
        fastpath=True,
    )


def _reduce_data(
    data: NDArrayAny,
    mom_ndim: Mom_NDim,
    axis: AxisReduceMult,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrder,
    parallel: bool | None,
    keepdims: bool = False,
    fastpath: bool = False,
) -> NDArrayAny:
    if not fastpath:
        dtype = select_dtype(data, out=out, dtype=dtype)

    # special to support multiple reduction dimensions...
    ndim = data.ndim - mom_ndim
    axis_tuple = normalize_axis_tuple(
        axis,
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
        parallel=parallel_heuristic(parallel, size=data.size),
    )(
        data,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
    )

    return optional_keepdims(
        out,
        axis=axis_tuple,
        keepdims=keepdims,
    )


# * Grouped -------------------------------------------------------------------
# ** utilities
def block_by(
    ndat: int,
    block: int,
    mode: BlockByModes = "drop_last",
) -> NDArrayInt:
    """
    Get groupby array for block reduction.

    Parameters
    ----------
    ndat : int
        Size of ``by``.
    block : int
        Block size. Negative values is a single block.
    mode : {drop_first, drop_last, expand_first, expand_last}
        What to do if ndat does not divide equally by ``block``.

        - "drop_first" : drop first samples
        - "drop_last" : drop last samples
        - "expand_first": expand first block size
        - "expand_last": expand last block size

    Returns
    -------
    by : ndarray
        Group array for block reduction.

    See Also
    --------
    reduce_data_grouped

    Examples
    --------
    >>> block_by(5, 2)
    array([ 0,  0,  1,  1, -1])

    >>> block_by(5, 2, mode="drop_first")
    array([-1,  0,  0,  1,  1])

    >>> block_by(5, 2, mode="expand_first")
    array([0, 0, 0, 1, 1])

    >>> block_by(5, 2, mode="expand_last")
    array([0, 0, 1, 1, 1])

    """
    if block <= 0 or block == ndat:
        return np.broadcast_to(np.int64(0), ndat)

    if block > ndat:
        msg = f"{block=} > {ndat=}."
        raise ValueError(msg)

    if mode not in {"drop_first", "drop_last", "expand_first", "expand_last"}:
        msg = f"Unknown {mode=}"
        raise ValueError(msg)

    nblock = ndat // block
    by = np.arange(nblock).repeat(block).astype(np.int64, copy=False)
    if len(by) == ndat:
        return by

    shift = ndat - len(by)
    pad_width = (shift, 0) if mode.endswith("_first") else (0, shift)
    if mode.startswith("drop"):
        return np.pad(by, pad_width, mode="constant", constant_values=-1)
    return np.pad(by, pad_width, mode="edge")


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

    codes, groups = factorize(_by, sort=sort)  # pyright: ignore[reportUnknownVariableType]

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
    data: DataT,
    by: ArrayLike,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceDataGroupedKwargs],
) -> DataT: ...
# Array no output or dtype
@overload
def reduce_data_grouped(
    data: ArrayLikeArg[FloatT],
    by: ArrayLike,
    *,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[ReduceDataGroupedKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def reduce_data_grouped(
    data: ArrayLike,
    by: ArrayLike,
    *,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceDataGroupedKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def reduce_data_grouped(
    data: ArrayLike,
    by: ArrayLike,
    *,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[ReduceDataGroupedKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def reduce_data_grouped(
    data: ArrayLike,
    by: ArrayLike,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceDataGroupedKwargs],
) -> NDArrayAny: ...


# ** public
@docfiller.decorate
def reduce_data_grouped(  # noqa: PLR0913
    data: ArrayLike | DataT,
    by: ArrayLike,
    *,
    mom_ndim: Mom_NDim = 1,
    axis: AxisReduce | MissingType = MISSING,
    move_axis_to_end: bool = False,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderCF = None,
    parallel: bool | None = None,
    dim: DimsReduce | MissingType = MISSING,
    mom_dims: MomDims | None = None,
    group_dim: str | None = None,
    groups: Groups | None = None,
    keep_attrs: KeepAttrs = None,
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Reduce data by group.


    Parameters
    ----------
    {data_numpy_or_dataarray_or_dataset}
    {mom_ndim}
    {by}
    {axis_data}
    {move_axis_to_end}
    {parallel}
    {out}
    {dtype}
    {casting}
    {order_cf}
    {dim}
    {mom_dims_data}
    {group_dim}
    {groups}
    {keep_attrs}
    {on_missing_core_dim}
    {apply_ufunc_kwargs}

    Returns
    -------
    out : ndarray or DataArray or Dataset
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
    mom_ndim = validate_mom_ndim(mom_ndim)
    dtype = select_dtype(data, out=out, dtype=dtype)
    by = np.asarray(by, dtype=np.int64)

    if is_xarray(data):
        axis, dim = select_axis_dim(data, axis=axis, dim=dim, mom_ndim=mom_ndim)
        core_dims = (dim, *validate_mom_dims(mom_dims, mom_ndim, data))

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _reduce_data_grouped,
            data,
            by,
            input_core_dims=[core_dims, [dim]],
            output_core_dims=[core_dims],
            exclude_dims={dim},
            kwargs={
                "mom_ndim": mom_ndim,
                # Need total axis here...
                "axis": -(mom_ndim + 1),
                "dtype": dtype,
                "out": xprepare_out_for_resample_data(
                    out,
                    mom_ndim=mom_ndim,
                    axis=axis,
                    move_axis_to_end=move_axis_to_end,
                    data=data,
                ),
                "casting": casting,
                "order": order,
                "parallel": parallel,
                "fastpath": is_dataarray(data),
            },
            keep_attrs=keep_attrs,
            **get_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                on_missing_core_dim=on_missing_core_dim,
                dask="parallelized",
                output_sizes={dim: np.max(by) + 1},
                output_dtypes=dtype or np.float64,
            ),
        )

        if not move_axis_to_end and is_dataarray(data):
            xout = xout.transpose(*data.dims)
        if groups is not None:
            xout = xout.assign_coords({dim: (dim, groups)})  # pyright: ignore[reportUnknownMemberType]
        if group_dim:
            xout = xout.rename({dim: group_dim})
        return xout

    # Numpy
    axis, data = prepare_data_for_reduction(
        data=data,
        axis=axis,
        mom_ndim=mom_ndim,
        dtype=dtype,
        recast=False,
        move_axis_to_end=move_axis_to_end,
    )
    return _reduce_data_grouped(
        data,
        by=by,
        mom_ndim=mom_ndim,
        axis=axis,
        dtype=dtype,
        out=out,
        casting=casting,
        order=order,
        parallel=parallel,
        fastpath=True,
    )


def _reduce_data_grouped(
    data: NDArrayAny,
    by: NDArrayInt,
    *,
    mom_ndim: Mom_NDim,
    axis: int,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrderCF,
    out: NDArrayAny | None,
    parallel: bool | None,
    fastpath: bool,
) -> NDArrayAny:
    if not fastpath:
        dtype = select_dtype(data, out=out, dtype=dtype)

    # include inner core dims for by
    axes = axes_data_reduction((-1,), mom_ndim=mom_ndim, axis=axis, out_has_axis=True)
    raise_if_wrong_value(len(by), data.shape[axis], "Wrong length of `by`.")

    if out is None:
        ngroup = by.max() + 1
        out_shape = (*data.shape[:axis], ngroup, *data.shape[axis + 1 :])
        out = np.zeros(out_shape, dtype=dtype, order=order)
    else:
        out.fill(0.0)

    factory_reduce_data_grouped(
        mom_ndim=mom_ndim,
        parallel=parallel_heuristic(parallel, size=data.size),
    )(
        data,
        by,
        out,
        axes=axes,
        casting=casting,
        signature=(dtype, np.int64, dtype),
    )
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

    raise_if_wrong_value(
        len(group_start),
        len(group_end),
        "`group_start` and `group_end` must have same length",
    )

    if (group_end < group_start).any():
        msg = "Found end < start"
        raise ValueError(msg)

    return index, group_start, group_end


# ** overload
@overload
def reduce_data_indexed(
    data: DataT,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceDataIndexedKwargs],
) -> DataT: ...
# Array no out or dtype
@overload
def reduce_data_indexed(
    data: ArrayLikeArg[FloatT],
    *,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[ReduceDataIndexedKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def reduce_data_indexed(
    data: ArrayLike,
    *,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceDataIndexedKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def reduce_data_indexed(
    data: ArrayLike,
    *,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[ReduceDataIndexedKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def reduce_data_indexed(
    data: ArrayLike,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceDataIndexedKwargs],
) -> NDArrayAny: ...


# ** public
@docfiller.decorate
def reduce_data_indexed(  # noqa: PLR0913
    data: ArrayLike | DataT,
    *,
    mom_ndim: Mom_NDim = 1,
    index: ArrayLike,
    group_start: ArrayLike,
    group_end: ArrayLike,
    scale: ArrayLike | None = None,
    axis: AxisReduce | MissingType = MISSING,
    move_axis_to_end: bool = False,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrder = None,
    parallel: bool | None = None,
    dim: DimsReduce | MissingType = MISSING,
    mom_dims: MomDims | None = None,
    coords_policy: CoordsPolicy = "first",
    group_dim: str | None = None,
    groups: Groups | None = None,
    keep_attrs: KeepAttrs = None,
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Reduce data by index

    Parameters
    ----------
    {data_numpy_or_dataarray_or_dataset}
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
    {out}
    {dtype}
    {casting}
    {order}
    {parallel}
    {dim}
    {mom_dims_data}
    {coords_policy}
    {group_dim}
    {groups}
    {keep_attrs}
    {on_missing_core_dim}
    {apply_ufunc_kwargs}

    Returns
    -------
    out : ndarray or DataArray or Dataset
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
    mom_ndim = validate_mom_ndim(mom_ndim)
    dtype = select_dtype(data, out=out, dtype=dtype)

    if is_xarray(data):
        axis, dim = select_axis_dim(data, axis=axis, dim=dim, mom_ndim=mom_ndim)
        core_dims = (dim, *validate_mom_dims(mom_dims, mom_ndim, data))

        # Yes, doing this here and in nmpy section.
        index, group_start, group_end = _validate_index(
            ndat=data.sizes[dim],
            index=index,
            group_start=group_start,
            group_end=group_end,
        )

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _reduce_data_indexed,
            data,
            input_core_dims=[core_dims],
            output_core_dims=[core_dims],
            exclude_dims={dim},
            kwargs={
                "axis": -(mom_ndim + 1),
                "mom_ndim": mom_ndim,
                "index": index,
                "group_start": group_start,
                "group_end": group_end,
                "scale": scale,
                "out": xprepare_out_for_resample_data(
                    out,
                    mom_ndim=mom_ndim,
                    axis=axis,
                    move_axis_to_end=move_axis_to_end,
                    data=data,
                ),
                "dtype": dtype,
                "casting": casting,
                "order": order,
                "parallel": parallel,
                "fastpath": is_dataarray(data),
            },
            keep_attrs=keep_attrs,
            **get_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                on_missing_core_dim=on_missing_core_dim,
                dask="parallelized",
                output_sizes={dim: len(group_start)},
                output_dtypes=dtype or np.float64,
            ),
        )

        if not move_axis_to_end and is_dataarray(xout):
            xout = xout.transpose(*data.dims)

        if coords_policy in {"first", "last"} and is_dataarray(data):
            # in case we passed in index, group_start, group_end as non-arrays
            # these will be processed correctly in the numpy call
            # but just make sure here....
            index, group_start, group_end = (
                np.asarray(_, dtype=np.int64) for _ in (index, group_start, group_end)
            )
            dim_select = index[
                group_end - 1 if coords_policy == "last" else group_start
            ]

            xout = replace_coords_from_isel(  # type: ignore[assignment]
                da_original=data,
                da_selected=xout,  # type: ignore[arg-type]
                indexers={dim: dim_select},
                drop=False,
            )
        elif coords_policy == "group" and groups is not None:
            xout = xout.assign_coords({dim: groups})  # pyright: ignore[reportUnknownMemberType]
        if group_dim:
            xout = xout.rename({dim: group_dim})

        return xout

    # Numpy
    axis, data = prepare_data_for_reduction(
        data=data,
        axis=axis,
        mom_ndim=mom_ndim,
        dtype=dtype,
        recast=False,
        move_axis_to_end=move_axis_to_end,
    )
    index, group_start, group_end = _validate_index(
        ndat=data.shape[axis],
        index=index,
        group_start=group_start,
        group_end=group_end,
    )

    return _reduce_data_indexed(
        data,
        axis=axis,
        mom_ndim=mom_ndim,
        index=index,
        group_start=group_start,
        group_end=group_end,
        scale=scale,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        parallel=parallel,
        fastpath=True,
    )


def _reduce_data_indexed(
    data: NDArrayAny,
    *,
    axis: int,
    mom_ndim: Mom_NDim,
    index: NDArrayAny,
    group_start: NDArrayAny,
    group_end: NDArrayAny,
    scale: ArrayLike | None,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrder,
    parallel: bool | None,
    fastpath: bool = False,
) -> NDArrayAny:
    if not fastpath:
        dtype = select_dtype(data, out=out, dtype=dtype)

    if scale is None:
        scale = np.broadcast_to(np.dtype(dtype).type(1), index.shape)
    else:
        scale = asarray_maybe_recast(scale, dtype=dtype, recast=False)
        raise_if_wrong_value(
            len(scale), len(index), "`scale` and `index` must have same length."
        )

    # include inner dims for index, start, end, scale
    axes = axes_data_reduction(
        *(-1,) * 4, mom_ndim=mom_ndim, axis=axis, out_has_axis=True
    )

    return factory_reduce_data_indexed(
        mom_ndim=mom_ndim,
        parallel=parallel_heuristic(parallel, size=data.size),
    )(
        data,
        index,
        group_start,
        group_end,
        scale,
        out=out,
        axes=axes,
        casting=casting,
        order=order,
        signature=(dtype, np.int64, np.int64, np.int64, dtype, dtype),
    )


# * For testing purposes
def resample_data_indexed(  # noqa: PLR0913
    data: ArrayT,
    freq: NDArrayAny,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = MISSING,
    move_axis_to_end: bool = False,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrder = None,
    parallel: bool = True,
    # xarray specific
    dim: DimsReduce | MissingType = MISSING,
    mom_dims: MomDims | None = None,
    coords_policy: CoordsPolicy = "first",
    group_dim: str | None = None,
    groups: Groups | None = None,
    keep_attrs: KeepAttrs = None,
) -> ArrayT:
    """Resample using indexed reduction."""
    from ._lib.utils import freq_to_index_start_end_scales

    index, start, end, scales = freq_to_index_start_end_scales(freq)

    return reduce_data_indexed(  # pyright: ignore[reportReturnType]
        data=data,
        mom_ndim=mom_ndim,
        index=index,
        group_start=start,
        group_end=end,
        scale=scales,
        axis=axis,
        move_axis_to_end=move_axis_to_end,
        parallel=parallel,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        dim=dim,
        mom_dims=mom_dims,
        coords_policy=coords_policy,
        group_dim=group_dim,
        groups=groups,
        keep_attrs=keep_attrs,
    )

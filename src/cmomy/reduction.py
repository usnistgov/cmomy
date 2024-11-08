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
    arrayorder_to_arrayorder_cf,
    asarray_maybe_recast,
    get_axes_from_values,
    moveaxis_order,
    optional_keepdims,
    select_dtype,
)
from .core.docstrings import docfiller
from .core.missing import MISSING
from .core.moment_params import (
    MomParamsArray,
    MomParamsXArray,
)
from .core.prepare import (
    optional_prepare_out_for_resample_data,
    prepare_data_for_reduction,
    prepare_out_for_reduce_data_grouped,
    prepare_out_from_values,
    prepare_values_for_reduction,
    xprepare_out_for_reduce_data,
    xprepare_out_for_resample_data,
    xprepare_values_for_reduction,
)
from .core.utils import mom_to_mom_shape
from .core.validate import (
    is_dataarray,
    is_dataset,
    is_xarray_typevar,
    raise_if_wrong_value,
    validate_axis_mult,
)
from .core.xr_utils import (
    contains_dims,
    factory_apply_ufunc_kwargs,
    replace_coords_from_isel,
    transpose_like,
)
from .factory import (
    factory_reduce_data,
    factory_reduce_data_grouped,
    factory_reduce_data_indexed,
    factory_reduce_vals,
    parallel_heuristic,
)

if TYPE_CHECKING:
    from collections.abc import Hashable
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from .core.typing import (
        ApplyUFuncKwargs,
        ArrayLikeArg,
        ArrayOrder,
        ArrayOrderCF,
        ArrayT,
        AxesGUFunc,
        AxisReduceMultWrap,
        AxisReduceWrap,
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
        MissingType,
        MomAxes,
        MomDims,
        Moments,
        MomentsStrict,
        MomNDim,
        MomParamsInput,
        NDArrayAny,
        NDArrayInt,
        ReduceDataGroupedKwargs,
        ReduceDataIndexedKwargs,
        ReduceDataKwargs,
        ReduceValsKwargs,
        Sampler,
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


@docfiller.decorate  # type: ignore[arg-type, unused-ignore]
def reduce_vals(
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    mom: Moments,
    axis: AxisReduceWrap | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    weight: ArrayLike | xr.DataArray | DataT | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsInput = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderCF = None,
    parallel: bool | None = None,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Reduce values to central (co)moments.

    Parameters
    ----------
    {x_genarray}
    {y_genarray}
    {mom}
    {axis}
    {dim}
    {weight_genarray}
    {mom_dims}
    {mom_params}
    {out}
    {dtype}
    {casting}
    {order_cf}
    {parallel}
    {keep_attrs}
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
    weight = 1.0 if weight is None else weight
    dtype = select_dtype(x, out=out, dtype=dtype)
    if is_xarray_typevar(x):
        mom, mom_params = MomParamsXArray.factory_mom(
            mom_params=mom_params, mom=mom, dims=mom_dims
        )
        dim, input_core_dims, xargs = xprepare_values_for_reduction(
            x,
            weight,
            *y,
            axis=axis,
            dim=dim,
            narrays=mom_params.ndim + 1,
            dtype=dtype,
            recast=False,
        )

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _reduce_vals,
            *xargs,
            input_core_dims=input_core_dims,
            output_core_dims=[mom_params.dims],
            kwargs={
                "mom": mom,
                "mom_params": mom_params.to_array(),
                "parallel": parallel,
                "axis_neg": -1,
                "out": None if is_dataset(x) else out,
                "dtype": dtype,
                "casting": casting,
                "order": order,
                "fastpath": is_dataarray(x),
            },
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_sizes={
                    **dict(zip(mom_params.dims, mom_to_mom_shape(mom))),
                },
                output_dtypes=dtype or np.float64,
            ),
        )

        return xout

    # Numpy
    mom, mom_params = MomParamsArray.factory_mom(mom=mom, mom_params=mom_params)
    axis_neg, args = prepare_values_for_reduction(
        x,
        weight,
        *y,
        axis=axis,
        dtype=dtype,
        recast=False,
        narrays=mom_params.ndim + 1,
        axes_to_end=False,
    )

    return _reduce_vals(
        *args,
        mom=mom,
        mom_params=mom_params,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        axis_neg=axis_neg,
        parallel=parallel,
        fastpath=True,
    )


def _reduce_vals(
    # x, w, *y
    *args: NDArrayAny,
    mom: MomentsStrict,
    mom_params: MomParamsArray,
    axis_neg: int,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrderCF,
    parallel: bool | None,
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
    out.fill(0.0)

    axes: AxesGUFunc = [
        # out
        mom_params.axes,
        # x, weight, *y
        *get_axes_from_values(*args, axis_neg=axis_neg),
    ]

    factory_reduce_vals(
        mom_ndim=mom_params.ndim,
        parallel=parallel_heuristic(parallel, size=args[0].size * mom_params.ndim),
    )(
        out,
        *args,
        axes=axes,
        casting=casting,
        order=order,
        signature=(dtype,) * (len(args) + 1),
    )

    return out


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
@docfiller.decorate  # type: ignore[arg-type, unused-ignore]
def reduce_data(  # noqa: PLR0913
    data: ArrayLike | DataT,
    *,
    axis: AxisReduceMultWrap | MissingType = MISSING,
    dim: DimsReduceMult | MissingType = MISSING,
    mom_ndim: MomNDim | None = None,
    mom_dims: MomDims | None = None,
    mom_axes: MomAxes | None = None,
    mom_params: MomParamsInput = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrder = None,
    keepdims: bool = False,
    parallel: bool | None = None,
    axes_to_end: bool = False,
    use_map: bool | None = None,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Reduce central moments array along axis.


    Parameters
    ----------
    {data_numpy_or_dataarray_or_dataset}
    {axis_data_mult}
    {dim_mult}
    {mom_ndim_data}
    {mom_axes}
    {mom_dims_data}
    {mom_params}
    {out}
    {dtype}
    {casting}
    {order}
    {keepdims}
    {parallel}
    use_map : bool, optional
        If not ``False``, use ``data.map`` if ``data`` is a
        :class:`~xarray.Dataset` and ``dim`` is not a single scalar. This will
        properly handle cases where ``dim`` is ``None`` or has multiple
        dimensions. Note that with this option, variables that do not contain
        ``dim`` or ``mom_dims`` will be left in the result unchanged.
    {keep_attrs}
    {apply_ufunc_kwargs}

    Returns
    -------
    out : ndarray or DataArray or Dataset
        Reduced data array with shape ``data.shape`` with ``axis`` removed.
        Same type as input ``data``.
    """
    dtype = select_dtype(data, out=out, dtype=dtype)
    if is_xarray_typevar(data):
        mom_params = MomParamsXArray.factory(
            mom_params=mom_params,
            ndim=mom_ndim,
            axes=mom_axes,
            dims=mom_dims,
            data=data,
            default_ndim=1,
        )

        # Special case for dataset with multiple dimensions.
        # Can't do this with xr.apply_ufunc if variables have different
        # dimensions.  Use map in this case
        if is_dataset(data):
            _dims_check: tuple[Hashable, ...] = mom_params.dims
            if dim is not None:
                dim = mom_params.select_axis_dim_mult(data, axis=axis, dim=dim)[1]
                _dims_check = (*dim, *_dims_check)  # type: ignore[misc, unused-ignore]  # unused in python3.12

            if not contains_dims(data, *_dims_check):
                msg = f"Dimensions {dim} and {mom_params.dims} not found in {tuple(data.dims)}"
                raise ValueError(msg)

            if (use_map is None or use_map) and (dim is None or len(dim) > 1):
                return data.map(  # pyright: ignore[reportUnknownMemberType]
                    reduce_data,
                    keep_attrs=keep_attrs if keep_attrs is None else bool(keep_attrs),
                    mom_params=mom_params,
                    dim=dim,
                    dtype=dtype,
                    casting=casting,
                    order=order,
                    keepdims=keepdims,
                    parallel=parallel,
                    axes_to_end=axes_to_end,
                    use_map=True,
                    apply_ufunc_kwargs=apply_ufunc_kwargs,
                )

        if use_map:
            if not contains_dims(data, *mom_params.dims):
                return data  # type: ignore[return-value, unused-ignore]  # used error in python3.12
            # if specified dims, only keep those in current dataarray
            if dim not in {None, MISSING}:
                dim = (dim,) if isinstance(dim, str) else dim

                def _filter_func(d: Hashable) -> bool:
                    return contains_dims(data, d)

                dim = tuple(filter(_filter_func, dim))  # type: ignore[arg-type]
                if len(dim) == 0:
                    return data  # type: ignore[return-value , unused-ignore] # used error in python3.12

        axis, dim = mom_params.select_axis_dim_mult(
            data,
            axis=axis,
            dim=dim,
        )

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _reduce_data,
            data,
            input_core_dims=[mom_params.core_dims(*dim)],
            output_core_dims=[
                mom_params.core_dims(*dim) if keepdims else mom_params.dims
            ],
            exclude_dims=set(dim),
            kwargs={
                "mom_params": mom_params.to_array(),
                "axis": tuple(range(-len(dim) - mom_params.ndim, -mom_params.ndim)),
                "out": xprepare_out_for_reduce_data(
                    target=data,
                    out=out,
                    dim=dim,
                    mom_params=mom_params,
                    keepdims=keepdims,
                    axes_to_end=axes_to_end,
                ),
                "dtype": dtype,
                "parallel": parallel,
                "keepdims": keepdims,
                "fastpath": is_dataarray(data),
                "casting": casting,
                "order": order,
                "axes_to_end": False,
            },
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_dtypes=dtype or np.float64,
                output_sizes=dict.fromkeys(dim, 1) if keepdims else None,
            ),
        )

        if not axes_to_end:
            xout = transpose_like(
                xout,
                template=data,
                remove=None if keepdims else dim,
            )
        else:
            _order: tuple[Hashable, ...] = (
                mom_params.core_dims(*dim) if keepdims else mom_params.dims
            )
            xout = xout.transpose(..., *_order, missing_dims="ignore")  # pyright: ignore[reportUnknownArgumentType]

        return xout

    # Numpy
    mom_params = MomParamsArray.factory(
        mom_params=mom_params, ndim=mom_ndim, axes=mom_axes, default_ndim=1
    )
    return _reduce_data(
        asarray_maybe_recast(data, dtype=dtype, recast=False),
        mom_params=mom_params,
        axis=validate_axis_mult(axis),
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        parallel=parallel,
        keepdims=keepdims,
        fastpath=True,
        axes_to_end=axes_to_end,
    )


def _reduce_data(
    data: NDArrayAny,
    *,
    mom_params: MomParamsArray,
    axis: AxisReduceMultWrap,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrder,
    parallel: bool | None,
    keepdims: bool = False,
    fastpath: bool = False,
    axes_to_end: bool,
) -> NDArrayAny:
    if not fastpath:
        dtype = select_dtype(data, out=out, dtype=dtype)

    mom_params = mom_params.normalize_axes(data.ndim)

    axis_tuple: tuple[int, ...]
    if axis is None:
        axis_tuple = tuple(a for a in range(data.ndim) if a not in mom_params.axes)
    else:
        axis_tuple = mom_params.normalize_axis_tuple(
            axis, data.ndim, msg_prefix="reduce_data"
        )

    if axis_tuple == ():
        return data

    # move reduction dimensions to last positions and reshape
    _order = moveaxis_order(data.ndim, axis_tuple, range(-len(axis_tuple), 0))
    data = data.transpose(*_order)
    data = data.reshape(*data.shape[: -len(axis_tuple)], -1)
    # transform _mom_axes to new positions
    _mom_axes = tuple(_order.index(a) for a in mom_params.axes)

    if out is not None:
        if axes_to_end:
            # easier to move axes in case keep dims
            if keepdims:
                # get rid of reduction axes and move moment axes
                out = np.squeeze(
                    out,
                    axis=tuple(
                        range(-(len(axis_tuple) + mom_params.ndim), -mom_params.ndim)
                    ),
                )
            out = np.moveaxis(out, mom_params.axes_last, _mom_axes)
        elif keepdims:
            out = np.squeeze(out, axis=axis_tuple)

    elif (_order_cf := arrayorder_to_arrayorder_cf(order)) is not None:
        # make the output have correct order if passed ``order`` flag.
        out = np.empty(data.shape[:-1], dtype=dtype, order=_order_cf)

    out = factory_reduce_data(
        mom_ndim=mom_params.ndim,
        parallel=parallel_heuristic(parallel, size=data.size),
    )(
        data,
        out=out,
        # data, out
        axes=[(-1, *_mom_axes), _mom_axes],
        dtype=dtype,
        casting=casting,
        order=order,
    )
    out = optional_keepdims(
        out,
        axis=axis_tuple,
        keepdims=keepdims,
    )

    if axes_to_end:
        if keepdims:
            order0 = (*axis_tuple, *mom_params.axes)
            order1 = tuple(range(-mom_params.ndim - len(axis_tuple), 0))
        else:
            order0 = _mom_axes
            order1 = mom_params.axes_to_end().axes

        out = np.moveaxis(out, order0, order1)

    return out


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
@docfiller.decorate  # type: ignore[arg-type, unused-ignore]
def reduce_data_grouped(  # noqa: PLR0913
    data: ArrayLike | DataT,
    by: ArrayLike,
    *,
    axis: AxisReduceWrap | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: MomNDim | None = None,
    mom_axes: MomAxes | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsInput = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderCF = None,
    parallel: bool | None = None,
    axes_to_end: bool = False,
    group_dim: str | None = None,
    groups: Groups | None = None,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Reduce data by group.


    Parameters
    ----------
    {data_numpy_or_dataarray_or_dataset}
    {by}
    {axis_data}
    {dim}
    {mom_ndim_data}
    {mom_axes}
    {mom_dims_data}
    {mom_params}
    {out}
    {dtype}
    {casting}
    {order_cf}
    {axes_to_end}
    {parallel}
    {group_dim}
    {groups}
    {keep_attrs}
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
    dtype = select_dtype(data, out=out, dtype=dtype)
    by = np.asarray(by, dtype=np.int64)
    if is_xarray_typevar(data):
        mom_params = MomParamsXArray.factory(
            mom_params=mom_params,
            ndim=mom_ndim,
            dims=mom_dims,
            axes=mom_axes,
            data=data,
            default_ndim=1,
        )
        axis, dim = mom_params.select_axis_dim(data, axis=axis, dim=dim)
        core_dims = mom_params.core_dims(dim)

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _reduce_data_grouped,
            data,
            by,
            input_core_dims=[core_dims, [dim]],
            output_core_dims=[core_dims],
            exclude_dims={dim},
            kwargs={
                "mom_params": mom_params.to_array(),
                # Need total axis here...
                "axis": -(mom_params.ndim + 1),
                "dtype": dtype,
                "out": xprepare_out_for_resample_data(
                    out,
                    mom_params=mom_params,
                    axis=axis,
                    axes_to_end=axes_to_end,
                    data=data,
                ),
                "casting": casting,
                "order": order,
                "parallel": parallel,
                "fastpath": is_dataarray(data),
            },
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_sizes={dim: by.max() + 1},
                output_dtypes=dtype or np.float64,
            ),
        )

        if not axes_to_end:
            xout = transpose_like(
                xout,
                template=data,
            )
        elif is_dataset(xout):
            xout = xout.transpose(..., dim, *mom_params.dims, missing_dims="ignore")  # pyright: ignore[reportUnknownArgumentType]

        if groups is not None:
            xout = xout.assign_coords({dim: (dim, groups)})  # pyright: ignore[reportUnknownMemberType]
        if group_dim:
            xout = xout.rename({dim: group_dim})

        return xout

    # Numpy
    axis, mom_params, data = prepare_data_for_reduction(
        data=data,
        axis=axis,
        mom_params=MomParamsArray.factory(
            mom_params=mom_params, ndim=mom_ndim, axes=mom_axes, default_ndim=1
        ),
        dtype=dtype,
        recast=False,
        axes_to_end=axes_to_end,
    )
    return _reduce_data_grouped(
        data,
        by=by,
        mom_params=mom_params,
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
    mom_params: MomParamsArray,
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
    axes = mom_params.axes_data_reduction(
        (-1,),
        axis=axis,
        out_has_axis=True,
    )
    raise_if_wrong_value(len(by), data.shape[axis], "Wrong length of `by`.")

    if out is None:
        out = prepare_out_for_reduce_data_grouped(
            data,
            mom_params=mom_params,
            axis=axis,
            axis_new_size=by.max() + 1,
            order=order,
            dtype=dtype,
        )
    out.fill(0.0)

    factory_reduce_data_grouped(
        mom_ndim=mom_params.ndim,
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
    **kwargs: Any,
) -> tuple[list[Any] | IndexAny | pd.MultiIndex, NDArrayInt, NDArrayInt, NDArrayInt]:
    """
    Transform group_idx to quantities to be used with :func:`reduce_data_indexed`.

    Parameters
    ----------
    by: array-like
        Values to factor.
    exclude_missing : bool, default=True
        If ``True`` (default), filter Negative and ``None`` values from ``group_idx``.

    **kwargs
        Extra arguments to :func:`numpy.argsort`

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

    indexes_sorted = np.argsort(codes, **kwargs)
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
@docfiller.decorate  # type: ignore[arg-type, unused-ignore]
def reduce_data_indexed(  # noqa: PLR0913
    data: ArrayLike | DataT,
    *,
    index: ArrayLike,
    group_start: ArrayLike,
    group_end: ArrayLike,
    scale: ArrayLike | None = None,
    axis: AxisReduceWrap | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: MomNDim | None = None,
    mom_axes: MomAxes | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsInput = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrder = None,
    parallel: bool | None = None,
    axes_to_end: bool = False,
    coords_policy: CoordsPolicy = "first",
    group_dim: str | None = None,
    groups: Groups | None = None,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Reduce data by index

    Parameters
    ----------
    {data_numpy_or_dataarray_or_dataset}
    index : ndarray
        Index into `data.shape[axis]`.
    group_start, group_end : ndarray
        Start, end of index for a group.
        ``index[group_start[group]:group_end[group]]`` are the indices for
        group ``group``.
    scale : ndarray, optional
        Weights of same size as ``index``.
    {axis_data}
    {dim}
    {mom_ndim_data}
    {mom_axes}
    {mom_dims_data}
    {mom_params}
    {out}
    {dtype}
    {casting}
    {order}
    {parallel}
    {axes_to_end}
    {coords_policy}
    {group_dim}
    {groups}
    {keep_attrs}
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
    dtype = select_dtype(data, out=out, dtype=dtype)

    if is_xarray_typevar(data):
        mom_params = MomParamsXArray.factory(
            mom_params=mom_params,
            ndim=mom_ndim,
            dims=mom_dims,
            axes=mom_axes,
            data=data,
            default_ndim=1,
        )
        axis, dim = mom_params.select_axis_dim(data, axis=axis, dim=dim)
        core_dims = mom_params.core_dims(dim)

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
                "axis": -(mom_params.ndim + 1),
                "mom_params": mom_params.to_array(),
                "index": index,
                "group_start": group_start,
                "group_end": group_end,
                "scale": scale,
                "out": xprepare_out_for_resample_data(
                    out,
                    mom_params=mom_params,
                    axis=axis,
                    axes_to_end=axes_to_end,
                    data=data,
                ),
                "dtype": dtype,
                "casting": casting,
                "order": order,
                "parallel": parallel,
                "fastpath": is_dataarray(data),
            },
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_sizes={dim: len(group_start)},
                output_dtypes=dtype or np.float64,
            ),
        )

        if not axes_to_end:
            xout = transpose_like(
                xout,
                template=data,
            )
        elif is_dataset(xout):
            xout = xout.transpose(..., dim, *mom_params.dims, missing_dims="ignore")  # pyright: ignore[reportUnknownArgumentType]

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

            xout = replace_coords_from_isel(  # type: ignore[assignment, unused-ignore]  # error with python3.12
                da_original=data,
                da_selected=xout,  # type: ignore[arg-type, unused-ignore]  # error python3.12 and pyright
                indexers={dim: dim_select},
                drop=False,
            )
        elif coords_policy == "group" and groups is not None:
            xout = xout.assign_coords({dim: groups})  # pyright: ignore[reportUnknownMemberType]
        if group_dim:
            xout = xout.rename({dim: group_dim})

        return xout

    # Numpy
    axis, mom_params, data = prepare_data_for_reduction(
        data=data,
        axis=axis,
        mom_params=MomParamsArray.factory(
            mom_params=mom_params, ndim=mom_ndim, axes=mom_axes, default_ndim=1
        ),
        dtype=dtype,
        recast=False,
        axes_to_end=axes_to_end,
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
        mom_params=mom_params,
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
    mom_params: MomParamsArray,
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
    axes = mom_params.axes_data_reduction(
        *(-1,) * 4,
        axis=axis,
        out_has_axis=True,
    )

    # optional out with correct ordering
    out = optional_prepare_out_for_resample_data(
        out=out,
        data=data,
        axis=axis,
        axis_new_size=len(group_start),
        order=order,
        dtype=dtype,
    )

    return factory_reduce_data_indexed(
        mom_ndim=mom_params.ndim,
        parallel=parallel_heuristic(parallel, size=data.size),
    )(
        data,
        index,
        group_start,
        group_end,
        scale,  # pyright: ignore[reportArgumentType]
        out=out,
        axes=axes,
        casting=casting,
        order=order,
        signature=(dtype, np.int64, np.int64, np.int64, dtype, dtype),
    )


# * For testing purposes
def resample_data_indexed(  # noqa: PLR0913
    data: ArrayT,
    sampler: Sampler,
    *,
    mom_ndim: MomNDim | None = None,
    axis: AxisReduceWrap | MissingType = MISSING,
    mom_axes: MomAxes | None = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrder = None,
    parallel: bool = True,
    axes_to_end: bool = False,
    # xarray specific
    dim: DimsReduce | MissingType = MISSING,
    mom_dims: MomDims | None = None,
    coords_policy: CoordsPolicy = "first",
    rep_dim: str = "rep",
    groups: Groups | None = None,
    keep_attrs: KeepAttrs = None,
) -> ArrayT:
    """Resample using indexed reduction."""
    from .resample import factory_sampler

    sampler = factory_sampler(
        sampler,
        data=data,
        axis=axis,
        mom_axes=mom_axes,
        dim=dim,
        mom_ndim=mom_ndim,
        mom_dims=mom_dims,
        rep_dim=rep_dim,
        parallel=parallel,
    )

    from ._lib.utils import freq_to_index_start_end_scales

    index, start, end, scales = freq_to_index_start_end_scales(sampler.freq)

    return reduce_data_indexed(  # pyright: ignore[reportReturnType]
        data=data,
        mom_ndim=mom_ndim,
        index=index,
        group_start=start,
        group_end=end,
        scale=scales,
        axis=axis,
        mom_axes=mom_axes,
        axes_to_end=axes_to_end,
        parallel=parallel,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        dim=dim,
        mom_dims=mom_dims,
        coords_policy=coords_policy,
        group_dim=rep_dim,
        groups=groups,
        keep_attrs=keep_attrs,
    )

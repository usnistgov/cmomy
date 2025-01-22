"""
Routines to perform central moments reduction (:mod:`~cmomy.reduction`)
=======================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
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
    prepare_out_from_values,
    prepare_values_for_reduction,
    xprepare_out_for_reduce_data,
    xprepare_values_for_reduction,
)
from .core.utils import mom_to_mom_shape
from .core.validate import (
    is_dataarray,
    is_dataset,
    is_xarray_typevar,
    validate_axis_mult,
)
from .core.xr_utils import (
    contains_dims,
    factory_apply_ufunc_kwargs,
    transpose_like,
)
from .factory import (
    factory_reduce_data,
    factory_reduce_vals,
    parallel_heuristic,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from .core.typing import (
        ApplyUFuncKwargs,
        ArrayLikeArg,
        ArrayOrder,
        ArrayOrderCF,
        AxesGUFunc,
        AxisReduceMultWrap,
        AxisReduceWrap,
        Casting,
        DataT,
        DimsReduce,
        DimsReduceMult,
        DTypeLikeArg,
        FloatT,
        KeepAttrs,
        MissingType,
        MomAxes,
        MomDims,
        Moments,
        MomentsStrict,
        MomNDim,
        MomParamsInput,
        NDArrayAny,
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
            dims_check: tuple[Hashable, ...] = mom_params.dims
            if dim is not None:
                dim = mom_params.select_axis_dim_mult(data, axis=axis, dim=dim)[1]
                dims_check = (*dim, *dims_check)  # type: ignore[misc, unused-ignore]  # unused in python3.12

            if not contains_dims(data, *dims_check):
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
                if not (dim := tuple(d for d in dim if contains_dims(data, d))):  # type: ignore[union-attr]
                    return data  # type: ignore[return-value , unused-ignore] # used error in python3.12

        axis, dim = mom_params.select_axis_dim_mult(
            data,
            axis=axis,
            dim=dim,  # pyright: ignore[reportUnknownArgumentType]
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
            order_: tuple[Hashable, ...] = (
                mom_params.core_dims(*dim) if keepdims else mom_params.dims
            )
            xout = xout.transpose(..., *order_, missing_dims="ignore")  # pyright: ignore[reportUnknownArgumentType]

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
    if axis is None:  # pylint: disable=consider-ternary-expression
        axis_tuple = tuple(a for a in range(data.ndim) if a not in mom_params.axes)
    else:
        axis_tuple = mom_params.normalize_axis_tuple(
            axis, data.ndim, msg_prefix="reduce_data"
        )

    if axis_tuple == ():
        return data

    # move reduction dimensions to last positions and reshape
    order_ = moveaxis_order(data.ndim, axis_tuple, range(-len(axis_tuple), 0))
    data = data.transpose(*order_)
    data = data.reshape(*data.shape[: -len(axis_tuple)], -1)
    # transform _mom_axes to new positions
    mom_axes = tuple(order_.index(a) for a in mom_params.axes)

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
            out = np.moveaxis(out, mom_params.axes_last, mom_axes)
        elif keepdims:
            out = np.squeeze(out, axis=axis_tuple)

    elif (_order_cf := arrayorder_to_arrayorder_cf(order)) is not None:  # pylint: disable=confusing-consecutive-elif
        # make the output have correct order if passed ``order`` flag.
        out = np.empty(data.shape[:-1], dtype=dtype, order=_order_cf)

    # pylint: disable=unexpected-keyword-arg
    out = factory_reduce_data(
        mom_ndim=mom_params.ndim,
        parallel=parallel_heuristic(parallel, size=data.size),
    )(
        data,
        out=out,
        # data, out
        axes=[(-1, *mom_axes), mom_axes],
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
            order0 = mom_axes
            order1 = mom_params.axes_to_end().axes

        out = np.moveaxis(out, order0, order1)

    return out

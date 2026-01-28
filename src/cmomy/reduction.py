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
    get_axes_from_values,
    optional_keepdims,
    reorder,
    select_dtype,
)
from .core.docstrings import docfiller
from .core.missing import MISSING
from .core.prepare import (
    PrepareDataArray,
    PrepareDataXArray,
    PrepareValsArray,
    PrepareValsXArray,
)
from .core.utils import mom_to_mom_shape
from .core.validate import (
    is_dataarray,
    is_dataset,
    is_xarray_typevar,
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
    from collections.abc import (
        Hashable,
    )

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from .core._typing_kwargs import (
        ApplyUFuncKwargs,
        ReduceDataKwargs,
        ReduceValsKwargs,
    )
    from .core.moment_params import MomParamsType
    from .core.typing import (
        ArrayLikeArg,
        ArrayOrderCF,
        ArrayOrderKACF,
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
        NDArrayAny,
    )
    from .core.typing_compat import Unpack


# * Reduce vals ---------------------------------------------------------------
# ** overloads
@overload
def reduce_vals(  # pyright: ignore[reportOverlappingOverload]
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
# arraylike or DataT
@overload
def reduce_vals(
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    weight: ArrayLike | xr.DataArray | DataT | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceValsKwargs],
) -> NDArrayAny | DataT: ...


@docfiller.decorate  # type: ignore[arg-type, unused-ignore]
def reduce_vals(  # noqa: PLR0913
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    mom: Moments,
    axis: AxisReduceWrap | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    weight: ArrayLike | xr.DataArray | DataT | None = None,
    mom_dims: MomDims | None = None,
    mom_axes: MomAxes | None = None,
    mom_params: MomParamsType = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderCF = None,
    parallel: bool | None = None,
    axes_to_end: bool = False,
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
    {mom_axes}
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
    if is_xarray_typevar["DataT"].check(x):
        prep, mom = PrepareValsXArray.factory_mom(
            mom_params=mom_params,
            mom=mom,
            dims=mom_dims,
            recast=False,
        )
        dim, input_core_dims, xargs = prep.values_for_reduction(
            x,
            weight,
            *y,
            axis=axis,
            dim=dim,
            narrays=prep.mom_params.ndim + 1,
            dtype=dtype,
        )

        out, mom_params_axes = prep.optional_out_from_values(
            out,
            *xargs,
            target=x,
            dim=dim,
            mom=mom,
            axes_to_end=axes_to_end,
            order=order,
            dtype=dtype,
            mom_axes=mom_axes,
            mom_params=prep.mom_params,
        )

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _reduce_vals,
            *xargs,
            input_core_dims=input_core_dims,
            output_core_dims=[prep.mom_params.dims],
            kwargs={
                "mom": mom,
                "prep": prep.prepare_array,
                "parallel": parallel,
                "axis_neg": -1,
                "out": out,
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
                    **dict(
                        zip(prep.mom_params.dims, mom_to_mom_shape(mom), strict=True)
                    ),
                },
                output_dtypes=dtype if dtype is not None else np.float64,  # type: ignore[redundant-expr]
            ),
        )

        return mom_params_axes.maybe_reorder_dataarray(xout)

    # Numpy
    prep, mom = PrepareValsArray.factory_mom(
        mom=mom, axes=mom_axes, mom_params=mom_params, recast=False
    )
    prep, axis_neg, args = prep.values_for_reduction(
        x,
        weight,
        *y,
        axis=axis,
        narrays=prep.mom_params.ndim + 1,
        axes_to_end=axes_to_end,
        dtype=dtype,
    )

    return _reduce_vals(
        *args,
        mom=mom,
        prep=prep,
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
    prep: PrepareValsArray,
    axis_neg: int,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrderCF,
    parallel: bool | None,
    fastpath: bool = False,
) -> NDArrayAny:
    dtype = select_dtype(args[0], out=out, dtype=dtype, fastpath=fastpath)

    out, _ = prep.out_from_values(
        out,
        val_shape=prep.get_val_shape(*args),
        mom=mom,
        axis_neg=axis_neg,
        dtype=dtype,
        order=order,
    )
    out.fill(0.0)

    axes: AxesGUFunc = [
        # out
        prep.mom_params.axes,
        # x, weight, *y
        *get_axes_from_values(*args, axis_neg=axis_neg),
    ]

    _ = factory_reduce_vals(
        mom_ndim=prep.mom_params.ndim,
        parallel=parallel_heuristic(parallel, size=args[0].size * prep.mom_params.ndim),
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
def reduce_data(  # pyright: ignore[reportOverlappingOverload]
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
# Arraylike or DataT
@overload
def reduce_data(
    data: ArrayLike | DataT,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceDataKwargs],
) -> NDArrayAny | DataT: ...


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
    mom_params: MomParamsType = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderKACF = None,
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
    if is_xarray_typevar["DataT"].check(data):
        prep = PrepareDataXArray.factory(
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
            dims_check: tuple[Hashable, ...] = prep.mom_params.dims
            if dim is not None:
                dim = prep.mom_params.select_axis_dim_mult(data, axis=axis, dim=dim)[1]
                dims_check = (*dim, *dims_check)  # type: ignore[misc, unused-ignore]  # unused in python3.12

            if not contains_dims(data, *dims_check):
                msg = f"Dimensions {dim} and {prep.mom_params.dims} not found in {tuple(data.dims)}"
                raise ValueError(msg)

            if (use_map is None or use_map) and (dim is None or len(dim) > 1):
                return data.map(  # pyright: ignore[reportUnknownMemberType]
                    reduce_data,
                    keep_attrs=keep_attrs if keep_attrs is None else bool(keep_attrs),
                    mom_params=prep.mom_params,
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
            if not contains_dims(data, *prep.mom_params.dims):
                return data  # type: ignore[return-value, unused-ignore]  # used error in python3.12
            # if specified dims, only keep those in current dataarray
            if dim not in {None, MISSING}:
                dim = (dim,) if isinstance(dim, str) else dim  # type: ignore[redundant-expr, unused-ignore]
                if not (dim := tuple(d for d in dim if contains_dims(data, d))):  # type: ignore[union-attr]  # pyright: ignore[reportGeneralTypeIssues, reportOptionalIterable, reportUnknownVariableType, reportUnknownArgumentType]
                    return data  # type: ignore[return-value , unused-ignore] # used error in python3.12

        axis, dim = prep.mom_params.select_axis_dim_mult(
            data,
            axis=axis,
            dim=dim,  # pyright: ignore[reportUnknownArgumentType]
        )

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _reduce_data,
            data,
            input_core_dims=[prep.mom_params.core_dims(*dim)],
            output_core_dims=[
                prep.mom_params.core_dims(*dim) if keepdims else prep.mom_params.dims
            ],
            exclude_dims=set(dim),
            kwargs={
                "prep": prep.prepare_array,
                "axis": tuple(
                    range(-len(dim) - prep.mom_params.ndim, -prep.mom_params.ndim)
                ),
                "out": prep.optional_out_reduce(
                    target=data,
                    out=out,
                    dim=dim,
                    keepdims=keepdims,
                    axes_to_end=axes_to_end,
                    order=order,
                    dtype=dtype,
                ),
                "dtype": dtype,
                "parallel": parallel,
                "keepdims": keepdims,
                "fastpath": is_dataarray(data),
                "casting": casting,
                "order": order,
            },
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_dtypes=dtype if dtype is not None else np.float64,  # type: ignore[redundant-expr]
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
                prep.mom_params.core_dims(*dim) if keepdims else prep.mom_params.dims
            )
            xout = xout.transpose(..., *order_, missing_dims="ignore")

        return xout

    # Numpy
    prep, axis, data = PrepareDataArray.factory(
        mom_params=mom_params, ndim=mom_ndim, axes=mom_axes, default_ndim=1
    ).data_for_reduction_multiple(
        data,
        axis=axis,
        axes_to_end=axes_to_end,
        dtype=dtype,
    )

    return _reduce_data(
        data,
        prep=prep,
        axis=axis,
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
    *,
    prep: PrepareDataArray,
    axis: tuple[int, ...],
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrderKACF,
    parallel: bool | None,
    keepdims: bool = False,
    fastpath: bool = False,
) -> NDArrayAny:
    dtype = select_dtype(data, out=out, dtype=dtype, fastpath=fastpath)

    mom_params = prep.mom_params.normalize_axes(data.ndim)
    if axis == ():
        return data

    # move reduction dimensions to last positions and reshape
    order_ = reorder(data.ndim, axis, range(-len(axis), 0))
    data = data.transpose(*order_)
    if len(axis) > 1:
        data = data.reshape(*data.shape[: -len(axis)], -1)
    # transform _mom_axes to new positions
    mom_axes = tuple(order_.index(a) for a in mom_params.axes)

    if out is None:
        if (order_cf := arrayorder_to_arrayorder_cf(order)) is not None:
            out = np.empty(data.shape[:-1], dtype=dtype, order=order_cf)
    elif keepdims:  # pylint: disable=confusing-consecutive-elif)
        out = np.squeeze(out, axis=axis)

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
    return optional_keepdims(
        out,
        axis=axis,
        keepdims=keepdims,
    )

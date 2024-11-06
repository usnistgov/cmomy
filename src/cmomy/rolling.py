"""
Rolling and rolling exponential averages (:mod:`~cmomy.rolling`)
================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr

from .core.array_utils import (
    asarray_maybe_recast,
    get_axes_from_values,
    normalize_axis_index,
    positive_to_negative_index,
    select_dtype,
)
from .core.docstrings import docfiller
from .core.missing import MISSING
from .core.moment_params import (
    MomParamsArray,
    MomParamsXArray,
    MomParamsXArrayOptional,
)
from .core.prepare import (
    optional_prepare_out_for_resample_data,
    prepare_data_for_reduction,
    prepare_out_from_values,
    prepare_values_for_reduction,
    xprepare_out_for_resample_data,
    xprepare_out_for_resample_vals,
    xprepare_values_for_reduction,
)
from .core.utils import (
    mom_to_mom_shape,
)
from .core.validate import (
    is_dataarray,
    is_dataset,
    is_xarray,
    is_xarray_typevar,
    validate_axis,
)
from .core.xr_utils import (
    factory_apply_ufunc_kwargs,
    transpose_like,
)
from .factory import (
    factory_rolling_data,
    factory_rolling_exp_data,
    factory_rolling_exp_vals,
    factory_rolling_vals,
    parallel_heuristic,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Any

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
        RollingDataKwargs,
        RollingExpDataKwargs,
        RollingExpValsKwargs,
        RollingValsKwargs,
        ScalarT,
    )
    from .core.typing_compat import Unpack


# * Moving average
@overload
def construct_rolling_window_array(
    x: DataT,
    window: int | Sequence[int],
    *,
    axis: AxisReduceMultWrap | MissingType = ...,
    dim: DimsReduceMult | MissingType = ...,
    center: bool | Sequence[bool] = ...,
    stride: int | Sequence[int] = ...,
    fill_value: ArrayLike = ...,
    mom_ndim: MomNDim | None = ...,
    mom_axes: MomAxes | None = ...,
    mom_dims: MomDims | None = ...,
    mom_params: MomParamsInput = ...,
    # xarray specific
    window_dim: str | Sequence[str] | None = ...,
    keep_attrs: bool | None = ...,
    **kwargs: Any,
) -> DataT: ...
@overload
def construct_rolling_window_array(
    x: NDArray[FloatT],
    window: int | Sequence[int],
    *,
    axis: AxisReduceMultWrap | MissingType = ...,
    dim: DimsReduceMult | MissingType = ...,
    center: bool | Sequence[bool] = ...,
    stride: int | Sequence[int] = ...,
    fill_value: ArrayLike = ...,
    mom_ndim: MomNDim | None = ...,
    mom_axes: MomAxes | None = ...,
    mom_dims: MomDims | None = ...,
    mom_params: MomParamsInput = ...,
    # xarray specific
    window_dim: str | Sequence[str] | None = ...,
    keep_attrs: bool | None = ...,
    **kwargs: Any,
) -> NDArray[FloatT]: ...
@overload
def construct_rolling_window_array(
    x: NDArrayAny,
    window: int | Sequence[int],
    *,
    axis: AxisReduceMultWrap | MissingType = ...,
    dim: DimsReduceMult | MissingType = ...,
    center: bool | Sequence[bool] = ...,
    stride: int | Sequence[int] = ...,
    fill_value: ArrayLike = ...,
    mom_ndim: MomNDim | None = ...,
    mom_axes: MomAxes | None = ...,
    mom_dims: MomDims | None = ...,
    mom_params: MomParamsInput = ...,
    # xarray specific
    window_dim: str | Sequence[str] | None = ...,
    keep_attrs: bool | None = ...,
    **kwargs: Any,
) -> NDArrayAny: ...


@docfiller.decorate
def construct_rolling_window_array(
    x: NDArrayAny | DataT,
    window: int | Sequence[int],
    *,
    axis: AxisReduceMultWrap | MissingType = MISSING,
    dim: DimsReduceMult | MissingType = MISSING,
    center: bool | Sequence[bool] = False,
    stride: int | Sequence[int] = 1,
    fill_value: ArrayLike = np.nan,
    mom_ndim: MomNDim | None = None,
    mom_axes: MomAxes | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsInput = None,
    # xarray specific
    window_dim: str | Sequence[str] | None = None,
    keep_attrs: bool | None = None,
    **kwargs: Any,
) -> NDArrayAny | DataT:
    """
    Convert an array to one with rolling windows.

    Parameters
    ----------
    x : array or DataArray or Dataset
        Input array.
    axis : int or iterable of int
        To sample along.
    dim : str or sequence of hashable
    window : int or sequence of int
        Window size.
    center : bool
        If ``True``, center windows.
    stride : int
        Size of strides in rolling window.
    fill_value : scalar
        Fill value for missing values.
    {mom_ndim_optional}
    {mom_axes}
    {mom_dims_data}
    {mom_params}
    window_dim : str or Sequence of str, optional
        Names of output window dimension(s). Default to ``rolling_{dim}`` where
        ``{dim}`` is replaced by ``dim`` names.
    {keep_attrs}
    **kwargs
        Extra arguments to :meth:`xarray.DataArray.rolling` or :meth:`xarray.Dataset.rolling`.


    Returns
    -------
    output : array
        Array of shape ``(window, *shape)``.

    Notes
    -----
    This function uses different syntax compared to
    :meth:`xarray.DataArray.rolling`. Instead of mappings for ``center``,
    etc, here you pass scalar or sequence values corresponding to axis/dim.
    Corresponding mappings are created from, for example ``center=dict(zip(dim,
    center))``.


    Examples
    --------
    >>> x = np.arange(5).astype(float)
    >>> construct_rolling_window_array(x, window=4, axis=0)
    array([[nan, nan, nan,  0.,  1.],
           [nan, nan,  0.,  1.,  2.],
           [nan,  0.,  1.,  2.,  3.],
           [ 0.,  1.,  2.,  3.,  4.]])

    >>> dx = xr.DataArray(x)
    >>> construct_rolling_window_array(dx, window=4, center=True, dim="dim_0")
    <xarray.DataArray (rolling_dim_0: 4, dim_0: 5)> Size: 160B
    array([[nan, nan,  0.,  1.,  2.],
           [nan,  0.,  1.,  2.,  3.],
           [ 0.,  1.,  2.,  3.,  4.],
           [ 1.,  2.,  3.,  4., nan]])
    Dimensions without coordinates: rolling_dim_0, dim_0

    See Also
    --------
    xarray.DataArray.rolling
    xarray.core.rolling.DataArrayRolling.construct
    """
    if is_xarray_typevar(x):
        mom_params = MomParamsXArrayOptional.factory(
            mom_params, ndim=mom_ndim, dims=mom_dims, axes=mom_axes, data=x
        )
        axis, dim = mom_params.select_axis_dim_mult(
            x,
            axis=axis,
            dim=dim,
        )

        nroll = len(dim)
        window = (window,) * nroll if isinstance(window, int) else window
        center = (center,) * nroll if isinstance(center, bool) else center
        stride = (stride,) * nroll if isinstance(stride, int) else stride

        window_dim = (
            tuple(f"rolling_{d}" for d in dim)
            if window_dim is None
            else (window_dim,)
            if isinstance(window_dim, str)
            else window_dim
        )

        if any(len(x) != nroll for x in (window, center, stride, window_dim)):
            msg = f"{axis=}, {window=}, {center=}, {stride=}, {window_dim=} must have same length"
            raise ValueError(msg)

        xout = x.rolling(
            dict(zip(dim, window)),
            center=dict(zip(dim, center)),
            **kwargs,
        ).construct(
            window_dim=dict(zip(dim, window_dim)),
            stride=dict(zip(dim, stride)),
            fill_value=fill_value,
            keep_attrs=keep_attrs,
        )

        # for safety, move window_dim to front...
        # this avoids it being placed after any moment dimensions...
        return xout.transpose(*window_dim, ...)  # pyright: ignore[reportUnknownArgumentType]  # python3.9

    return construct_rolling_window_array(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        x=xr.DataArray(x),
        window=window,
        axis=axis,
        center=center,
        stride=stride,
        fill_value=fill_value,
        mom_ndim=mom_ndim,
        mom_axes=mom_axes,
        **kwargs,
    ).to_numpy()


# * Moving
def _pad_along_axis(
    data: NDArray[ScalarT], axis: int, shift: int, fill_value: float
) -> NDArray[ScalarT]:
    pads = [(0, 0)] * data.ndim
    pads[axis] = (0, -shift)
    return np.pad(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
        data,
        pads,  # pyright: ignore[reportArgumentType]
        mode="constant",
        constant_values=fill_value,
    )


def _optional_zero_missing_weight(
    data: NDArray[ScalarT], mom_axes: Iterable[int], zero_missing_weights: bool
) -> NDArray[ScalarT]:
    """Note that this modifies ``data`` inplace.  Pass in a copy if this is not what you want."""
    if zero_missing_weights:
        s: list[Any] = [slice(None)] * data.ndim
        for m in mom_axes:
            s[m] = 0
        w = data[tuple(s)]
        w[np.isnan(w)] = 0.0
    return data


# ** Data
@overload
def rolling_data(
    data: DataT,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[RollingDataKwargs],
) -> DataT: ...
# Array
@overload
def rolling_data(
    data: ArrayLikeArg[FloatT],
    *,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[RollingDataKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def rolling_data(
    data: ArrayLike,
    *,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[RollingDataKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def rolling_data(
    data: ArrayLike,
    *,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[RollingDataKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def rolling_data(
    data: ArrayLike,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[RollingDataKwargs],
) -> NDArrayAny: ...


@docfiller.decorate  # type: ignore[arg-type, unused-ignore]
def rolling_data(  # noqa: PLR0913
    data: ArrayLike | DataT,
    *,
    window: int,
    axis: AxisReduceWrap | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: MomNDim | None = None,
    mom_axes: MomAxes | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsInput = None,
    min_periods: int | None = None,
    center: bool = False,
    zero_missing_weights: bool = True,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrder = None,
    parallel: bool | None = None,
    axes_to_end: bool = False,
    # xarray specific
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Moving average of central moments array.

    Parameters
    ----------
    data : array-like or DataArray or Dataset
    {window}
    {axis}
    {dim}
    {mom_ndim_data}
    {mom_axes}
    {mom_dims_data}
    {mom_params}
    {min_periods}
    {center}
    {zero_missing_weights}
    {out}
    {dtype}
    {casting}
    {order}
    {parallel}
    {axes_to_end}
    {keep_attrs}
    {apply_ufunc_kwargs}

    Returns
    -------
    out : ndarray or DataArray or Dataset
        Moving average data, of same shape and type as ``data``.
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
        axis, dim = mom_params.select_axis_dim(data, axis=axis, dim=dim)
        core_dims = [mom_params.core_dims(dim)]

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _rolling_data,
            data,
            input_core_dims=core_dims,
            output_core_dims=core_dims,
            kwargs={
                "window": window,
                "axis": -(mom_params.ndim + 1),
                "mom_params": mom_params.to_array(),
                "min_periods": min_periods,
                "center": center,
                "zero_missing_weights": zero_missing_weights,
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
                output_dtypes=dtype or np.float64,
            ),
        )

        if not axes_to_end:
            xout = transpose_like(
                xout,
                template=data,
            )
        elif is_dataset(data):
            xout = xout.transpose(..., dim, *mom_params.dims, missing_dims="ignore")  # pyright: ignore[reportUnknownArgumentType]

        return xout

    # Numpy
    axis, mom_params, data = prepare_data_for_reduction(
        data,
        axis=axis,
        mom_params=MomParamsArray.factory(
            mom_params=mom_params, ndim=mom_ndim, axes=mom_axes, default_ndim=1
        ),
        dtype=dtype,
        recast=False,
        axes_to_end=axes_to_end,
    )

    return _rolling_data(
        data,
        window=window,
        axis=axis,
        mom_params=mom_params,
        min_periods=min_periods,
        center=center,
        zero_missing_weights=zero_missing_weights,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        parallel=parallel,
        fastpath=True,
    )


def _rolling_data(
    data: NDArrayAny,
    *,
    window: int,
    axis: int,
    mom_params: MomParamsArray,
    min_periods: int | None,
    center: bool,
    zero_missing_weights: bool,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrder,
    parallel: bool | None,
    fastpath: bool = False,
) -> NDArrayAny:
    if not fastpath:
        dtype = select_dtype(data, out=out, dtype=dtype)

    shift = (-window // 2) + 1 if center else None
    if shift is not None:
        data = _pad_along_axis(data, axis=axis, shift=shift, fill_value=0.0)

    out = optional_prepare_out_for_resample_data(
        data=data,
        out=out,
        axis=axis,
        axis_new_size=data.shape[axis],
        order=order,
        dtype=dtype,
    )

    axes = [
        # add in window, count
        (),
        (),
        *mom_params.axes_data_reduction(
            axis=axis,
            out_has_axis=True,
        ),
    ]

    min_periods = window if min_periods is None else min_periods

    out = factory_rolling_data(
        mom_ndim=mom_params.ndim,
        parallel=parallel_heuristic(parallel, size=data.size),
    )(
        window,
        min_periods,
        data,
        out=out,
        axes=axes,
        casting=casting,
        order=order,
        signature=(np.int64, np.int64, dtype, dtype),
    )

    if shift is not None:
        valid = [slice(None)] * data.ndim
        valid[axis] = slice(-shift, None)
        out = out[tuple(valid)]

    return _optional_zero_missing_weight(out, mom_params.axes, zero_missing_weights)


# * Vals
@overload
def rolling_vals(
    x: DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    weight: ArrayLike | xr.DataArray | DataT | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[RollingValsKwargs],
) -> DataT: ...
# array
@overload
def rolling_vals(
    x: ArrayLikeArg[FloatT],
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[RollingValsKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def rolling_vals(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[RollingValsKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def rolling_vals(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[RollingValsKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def rolling_vals(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[RollingValsKwargs],
) -> NDArrayAny: ...


@docfiller.decorate  # type: ignore[arg-type, unused-ignore]
def rolling_vals(  # noqa: PLR0913
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    mom: Moments,
    window: int,
    axis: AxisReduceWrap | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    weight: ArrayLike | xr.DataArray | DataT | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsInput = None,
    min_periods: int | None = None,
    center: bool = False,
    zero_missing_weights: bool = True,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderCF = None,
    parallel: bool | None = None,
    axes_to_end: bool = True,
    # xarray specific
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Moving average of central moments generated by values.

    Parameters
    ----------
    {x_genarray}
    {y_genarray}
    {mom}
    {window}
    {axis}
    {dim}
    {weight_genarray}
    {mom_dims}
    {mom_params}
    {min_periods}
    {center}
    {zero_missing_weights}
    {out}
    {dtype}
    {casting}
    {order_cf}
    {parallel}
    {axes_to_end}
    {keep_attrs}
    {apply_ufunc_kwargs}

    Returns
    -------
    out : ndarray or DataArray
        Central moments array of same type as ``x``. ``out.shape = shape +
        (mom0, ...)`` where ``shape = np.broadcast_shapes(*(a.shape for a in
        (x_, *y_, weight_)))`` and ``x_``, ``y_`` and ``weight_`` are the input
        arrays with ``axis`` moved to the last axis. That is, the last
        dimensions are the moving average axis ``axis`` and the moments
        dimensions.


    See Also
    --------
    numpy.broadcast_shapes
    """
    weight = 1.0 if weight is None else weight
    dtype = select_dtype(x, out=out, dtype=dtype)

    if is_xarray_typevar(x):
        mom, mom_params = MomParamsXArray.factory_mom(
            mom, mom_params=mom_params, dims=mom_dims
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
            _rolling_vals,
            *xargs,
            input_core_dims=input_core_dims,
            output_core_dims=[mom_params.core_dims(dim)],
            kwargs={
                "mom": mom,
                "mom_params": mom_params.to_array(),
                "axis_neg": -1,
                "window": window,
                "min_periods": min_periods,
                "center": center,
                "zero_missing_weights": zero_missing_weights,
                "out": xprepare_out_for_resample_vals(
                    target=x,
                    out=out,
                    dim=dim,
                    mom_ndim=mom_params.ndim,
                    axes_to_end=axes_to_end,
                ),
                "dtype": dtype,
                "casting": casting,
                "order": order,
                "parallel": parallel,
                "fastpath": is_dataarray(x),
            },
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_sizes=dict(zip(mom_params.dims, mom_to_mom_shape(mom))),
                output_dtypes=dtype or np.float64,
            ),
        )
        if not axes_to_end:
            xout = transpose_like(
                xout,
                template=x,
                append=mom_params.dims,
            )
        elif is_dataset(x):
            xout = xout.transpose(..., dim, *mom_params.dims, missing_dims="ignore")  # pyright: ignore[reportUnknownArgumentType]
        return xout

    # Numpy
    mom, mom_params = MomParamsArray.factory_mom(mom=mom, mom_params=mom_params)
    axis_neg, args = prepare_values_for_reduction(
        x,
        weight,
        *y,
        axis=axis,
        narrays=mom_params.ndim + 1,
        dtype=dtype,
        recast=False,
        axes_to_end=axes_to_end,
    )

    return _rolling_vals(
        *args,
        mom=mom,
        mom_params=mom_params,
        axis_neg=axis_neg,
        window=window,
        min_periods=min_periods,
        center=center,
        zero_missing_weights=zero_missing_weights,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        parallel=parallel,
        fastpath=True,
    )


def _rolling_vals(
    # x, weight, *y
    *args: NDArrayAny,
    mom: MomentsStrict,
    mom_params: MomParamsArray,
    window: int,
    axis_neg: int,
    min_periods: int | None,
    center: bool,
    zero_missing_weights: bool,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrderCF,
    parallel: bool | None,
    fastpath: bool = False,
) -> NDArrayAny:
    # Numpy
    if not fastpath:
        dtype = select_dtype(args[0], out=out, dtype=dtype)

    axes_args: AxesGUFunc = get_axes_from_values(*args, axis_neg=axis_neg)

    shift = (-window // 2) + 1 if center else None
    if shift is not None:
        args = tuple(
            _pad_along_axis(arg, axis=axes[0], shift=shift, fill_value=0.0)
            for arg, axes in zip(args, axes_args)
        )

    out = prepare_out_from_values(
        out,
        *args,
        mom=mom,
        axis_neg=axis_neg,
        axis_new_size=args[0].shape[axis_neg],
        dtype=dtype,
        order=order,
    )

    axes: AxesGUFunc = [
        # out
        (axis_neg - mom_params.ndim, *mom_params.axes),
        # window, min_periods
        (),
        (),
        # args
        *axes_args,
    ]

    factory_rolling_vals(
        mom_ndim=mom_params.ndim,
        parallel=parallel_heuristic(parallel, args[0].size * mom_params.ndim),
    )(
        out,
        window,
        window if min_periods is None else min_periods,
        *args,
        axes=axes,
        casting=casting,
        order=order,
        signature=(dtype, np.int64, np.int64, *(dtype,) * len(args)),
    )

    if shift is not None:
        valid = [slice(None)] * out.ndim
        valid[axis_neg - mom_params.ndim] = slice(-shift, None)
        out = out[tuple(valid)]

    return _optional_zero_missing_weight(out, mom_params.axes, zero_missing_weights)


# * Move Exponential
# ** Data
@overload
def rolling_exp_data(
    data: DataT,
    alpha: ArrayLike | xr.DataArray | xr.Dataset,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[RollingExpDataKwargs],
) -> DataT: ...
# array
@overload
def rolling_exp_data(
    data: ArrayLikeArg[FloatT],
    alpha: ArrayLike,
    *,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[RollingExpDataKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def rolling_exp_data(
    data: ArrayLike,
    alpha: ArrayLike,
    *,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[RollingExpDataKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def rolling_exp_data(
    data: ArrayLike,
    alpha: ArrayLike,
    *,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[RollingExpDataKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def rolling_exp_data(
    data: ArrayLike,
    alpha: ArrayLike,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[RollingExpDataKwargs],
) -> NDArray[FloatT]: ...


@docfiller.decorate  # type: ignore[arg-type, unused-ignore]
def rolling_exp_data(  # noqa: PLR0913
    data: ArrayLike | DataT,
    alpha: ArrayLike | xr.DataArray | xr.Dataset,
    *,
    axis: AxisReduceWrap | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: MomNDim | None = None,
    mom_axes: MomAxes | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsInput = None,
    min_periods: int | None = None,
    alpha_axis: AxisReduceWrap | MissingType = MISSING,
    adjust: bool = True,
    zero_missing_weights: bool = True,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrder = None,
    parallel: bool | None = None,
    axes_to_end: bool = False,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Moving average of central moments array.

    Parameters
    ----------
    data : array-like
    alpha : array-like
        `alpha` values.
    {axis}
    {dim}
    {mom_ndim_data}
    {mom_axes}
    {mom_dims_data}
    {mom_params}
    {min_periods}
    alpha_axis : int
        Axis of ``alpha`` corresponding to ``axis`` for ``data``.
    adjust : bool, default=True
        Same as ``adjust`` parameter of :meth:`pandas.DataFrame.ewm`
    {zero_missing_weights}
    {out}
    {dtype}
    {casting}
    {order}
    {parallel}
    {axes_to_end}
    {keep_attrs}
    {apply_ufunc_kwargs}

    Returns
    -------
    out : ndarray or DataArray
        Exponential moving average of same shape and type of ``data``.

    See Also
    --------
    xarray.DataArray.rolling_exp
    xarray.core.rolling_exp.RollingExp
    pandas.DataFrame.ewm

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

        if not is_xarray(alpha):
            # prepare array alpha
            alpha_axis, alpha = _prepare_alpha_array(
                alpha,
                alpha_axis,
                ndat=data.sizes[dim],
                dtype=dtype,
                axes_to_end=True,
            )
        else:
            alpha_axis = -1

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _rolling_exp_data,
            data,
            alpha,
            input_core_dims=[core_dims, [dim]],
            output_core_dims=[core_dims],
            kwargs={
                "axis": -(mom_params.ndim + 1),
                "alpha_axis": alpha_axis,
                "mom_params": mom_params.to_array(),
                "min_periods": min_periods,
                "adjust": adjust,
                "zero_missing_weights": zero_missing_weights,
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
                output_dtypes=dtype or np.float64,
            ),
        )

        if not axes_to_end:
            xout = transpose_like(
                xout,
                template=data,
            )
        elif is_dataset(data):
            xout = xout.transpose(..., dim, *mom_params.dims, missing_dims="ignore")  # pyright: ignore[reportUnknownArgumentType]

        return xout

    # save the original axis for alpha_axis...
    axis, mom_params, data = prepare_data_for_reduction(
        data,
        axis=axis,
        mom_params=MomParamsArray.factory(
            mom_params=mom_params, ndim=mom_ndim, axes=mom_axes, default_ndim=1
        ),
        dtype=dtype,
        recast=False,
        axes_to_end=axes_to_end,
    )

    # prepare alpha
    assert not is_xarray(alpha)  # noqa: S101
    alpha_axis, alpha = _prepare_alpha_array(
        alpha,
        alpha_axis,
        ndat=data.shape[axis],
        dtype=dtype,
        axes_to_end=False,
    )

    return _rolling_exp_data(
        data,
        alpha,
        axis=axis,
        alpha_axis=alpha_axis,
        mom_params=mom_params,
        min_periods=min_periods,
        adjust=adjust,
        zero_missing_weights=zero_missing_weights,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        parallel=parallel,
        fastpath=True,
    )


def _prepare_alpha_array(
    alpha: ArrayLike,
    alpha_axis: AxisReduceWrap | MissingType,
    ndat: int,
    dtype: DTypeLike,
    axes_to_end: bool,
) -> tuple[int, NDArrayAny]:
    """Should only be called with array-like alpha"""
    alpha = asarray_maybe_recast(alpha, dtype, recast=False)
    if alpha.ndim == 0:
        alpha = np.broadcast_to(alpha, ndat)
        alpha_axis = -1
    elif alpha.ndim == 1 or alpha_axis is MISSING:
        alpha_axis = -1
    else:
        alpha_axis = normalize_axis_index(validate_axis(alpha_axis), alpha.ndim)
        alpha_axis = positive_to_negative_index(alpha_axis, alpha.ndim)
        if axes_to_end and alpha_axis != -1:
            alpha = np.moveaxis(alpha, alpha_axis, -1)
            alpha_axis = -1

    return alpha_axis, alpha


def _rolling_exp_data(
    data: NDArrayAny,
    alpha: NDArrayAny,
    *,
    axis: int,
    alpha_axis: int,
    mom_params: MomParamsArray,
    min_periods: int | None,
    adjust: bool,
    zero_missing_weights: bool,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrder,
    parallel: bool | None,
    fastpath: bool,
) -> NDArrayAny:
    if not fastpath:
        dtype = select_dtype(data, out=out, dtype=dtype)

    out = optional_prepare_out_for_resample_data(
        data=data,
        out=out,
        axis=axis,
        axis_new_size=data.shape[axis],
        order=order,
        dtype=dtype,
    )

    # axes for reduction
    axes = [
        # add alpha, adjust, min_periods
        (alpha_axis,),
        (),
        (),
        *mom_params.axes_data_reduction(
            axis=axis,
            out_has_axis=True,
        ),
    ]

    min_periods = 1 if min_periods is None else max(1, min_periods)
    out = factory_rolling_exp_data(
        mom_ndim=mom_params.ndim,
        parallel=parallel_heuristic(parallel, data.size),
    )(
        alpha,
        adjust,
        min_periods,
        data,
        out=out,
        axes=axes,
        casting=casting,
        order=order,
        signature=(dtype, np.bool_, np.int64, dtype, dtype),
    )

    return _optional_zero_missing_weight(out, mom_params.axes, zero_missing_weights)


# ** Vals
@overload
def rolling_exp_vals(
    x: DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    alpha: ArrayLike | xr.DataArray | DataT,
    weight: ArrayLike | xr.DataArray | DataT | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[RollingExpValsKwargs],
) -> DataT: ...
# Array
@overload
def rolling_exp_vals(
    x: ArrayLikeArg[FloatT],
    *y: ArrayLike,
    alpha: ArrayLike,
    weight: ArrayLike | None = ...,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[RollingExpValsKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def rolling_exp_vals(
    x: ArrayLike,
    *y: ArrayLike,
    alpha: ArrayLike,
    weight: ArrayLike | None = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[RollingExpValsKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def rolling_exp_vals(
    x: ArrayLike,
    *y: ArrayLike,
    alpha: ArrayLike,
    weight: ArrayLike | None = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[RollingExpValsKwargs],
) -> NDArray[FloatT]: ...
# Fallback
@overload
def rolling_exp_vals(
    x: ArrayLike,
    *y: ArrayLike,
    alpha: ArrayLike,
    weight: ArrayLike | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[RollingExpValsKwargs],
) -> NDArray[FloatT]: ...


@docfiller.decorate  # type: ignore[arg-type, unused-ignore]
def rolling_exp_vals(  # noqa: PLR0913
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    alpha: ArrayLike | xr.DataArray | DataT,
    mom: Moments,
    axis: AxisReduceWrap | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    weight: ArrayLike | xr.DataArray | DataT | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsInput = None,
    min_periods: int | None = None,
    adjust: bool = True,
    zero_missing_weights: bool = True,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderCF = None,
    parallel: bool | None = None,
    axes_to_end: bool = True,
    # xarray specific
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Moving average of central moments generated by values.

    Parameters
    ----------
    {x_genarray}
    {y_genarray}
    alpha : array-like
        `alpha` values.
    {mom}
    {axis}
    {dim}
    {weight_genarray}
    {mom_dims}
    {mom_params}
    {min_periods}
    adjust : bool, default=True
        Same as ``adjust`` parameter of :meth:`pandas.DataFrame.ewm`
    {zero_missing_weights}
    {out}
    {dtype}
    {casting}
    {order_cf}
    {parallel}
    {axes_to_end}
    {keep_attrs}
    {apply_ufunc_kwargs}

    Returns
    -------
    out : ndarray or DataArray
        Central moments array of same type as ``x``. ``out.shape = shape +
        (mom0, ...)`` where ``shape = np.broadcast_shapes(*(a.shape for a in
        (x_, *y_, weight_)))`` and ``x_``, ``y_`` and ``weight_`` are the input
        arrays with ``axis`` moved to the last axis. That is, the last
        dimensions are the moving average axis ``axis`` and the moments
        dimensions.


    See Also
    --------
    numpy.broadcast_shapes
    """
    weight = 1.0 if weight is None else weight
    dtype = select_dtype(x, out=out, dtype=dtype)
    if is_xarray_typevar(x):
        mom, mom_params = MomParamsXArray.factory_mom(
            mom=mom, mom_params=mom_params, dims=mom_dims
        )
        dim, input_core_dims, xargs = xprepare_values_for_reduction(
            x,
            weight,
            alpha,
            *y,
            axis=axis,
            dim=dim,
            narrays=mom_params.ndim + 2,
            dtype=dtype,
            recast=False,
        )

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _rolling_exp_vals,
            *xargs,
            input_core_dims=input_core_dims,
            output_core_dims=[mom_params.core_dims(dim)],
            kwargs={
                "mom": mom,
                "mom_params": mom_params.to_array(),
                "axis_neg": -1,
                "adjust": adjust,
                "min_periods": min_periods,
                "zero_missing_weights": zero_missing_weights,
                "out": xprepare_out_for_resample_vals(
                    target=x,
                    out=out,
                    dim=dim,
                    mom_ndim=mom_params.ndim,
                    axes_to_end=axes_to_end,
                ),
                "dtype": dtype,
                "casting": casting,
                "order": order,
                "parallel": parallel,
                "fastpath": is_dataarray(x),
            },
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_sizes=dict(zip(mom_params.dims, mom_to_mom_shape(mom))),
                output_dtypes=dtype or np.float64,
            ),
        )

        if not axes_to_end:
            xout = transpose_like(
                xout,
                template=x,
                append=mom_params.dims,
            )
        elif is_dataset(x):
            xout = xout.transpose(..., dim, *mom_params.dims, missing_dims="ignore")  # pyright: ignore[reportUnknownArgumentType]

        return xout

    mom, mom_params = MomParamsArray.factory_mom(mom=mom, mom_params=mom_params)
    axis_neg, args = prepare_values_for_reduction(
        x,
        weight,
        alpha,
        *y,
        axis=axis,
        narrays=mom_params.ndim + 2,
        dtype=dtype,
        recast=False,
        axes_to_end=axes_to_end,
    )

    return _rolling_exp_vals(
        *args,
        mom=mom,
        mom_params=mom_params,
        axis_neg=axis_neg,
        adjust=adjust,
        min_periods=min_periods,
        zero_missing_weights=zero_missing_weights,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        parallel=parallel,
        fastpath=True,
    )


def _rolling_exp_vals(
    x: NDArrayAny,
    w: NDArrayAny,
    alpha: NDArrayAny,
    *y: NDArrayAny,
    mom: MomentsStrict,
    mom_params: MomParamsArray,
    axis_neg: int,
    adjust: bool,
    min_periods: int | None,
    zero_missing_weights: bool,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrderCF,
    parallel: bool | None,
    fastpath: bool = False,
) -> NDArrayAny:
    args: list[NDArrayAny] = [x, w, *y]
    if not fastpath:
        # reapply dtype in case calling with `apply_ufunc` with a dataset...
        dtype = select_dtype(x, out=out, dtype=dtype)

    out = prepare_out_from_values(
        out,
        *args,
        mom=mom,
        axis_neg=axis_neg,
        axis_new_size=args[0].shape[axis_neg],
        dtype=dtype,
        order=order,
    )

    axes_alpha, *axes_args = get_axes_from_values(alpha, *args, axis_neg=axis_neg)
    axes = [
        # out
        (axis_neg - mom_params.ndim, *mom_params.axes),
        # alpha
        axes_alpha,
        # adjust, min_periods
        (),
        (),
        # x, w, *y
        *axes_args,
    ]

    min_periods = 1 if min_periods is None else max(1, min_periods)
    factory_rolling_exp_vals(
        mom_ndim=mom_params.ndim,
        parallel=parallel_heuristic(parallel, x.size * mom_params.ndim),
    )(
        out,
        alpha,
        adjust,
        min_periods,
        *args,
        axes=axes,
        casting=casting,
        order=order,
        signature=(dtype, dtype, np.bool_, np.int64, *(dtype,) * len(args)),
    )

    return _optional_zero_missing_weight(out, mom_params.axes, zero_missing_weights)

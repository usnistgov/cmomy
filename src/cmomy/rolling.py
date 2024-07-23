"""
Moving averages (:mod:`~cmomy.moving`)
======================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr

from ._utils import (
    MISSING,
    axes_data_reduction,
    mom_to_mom_shape,
    normalize_axis_index,
    parallel_heuristic,
    prepare_values_for_reduction,
    select_axis_dim,
    select_axis_dim_mult,
    select_dtype,
    validate_axis,
    validate_mom_and_mom_ndim,
    validate_mom_dims,
    validate_mom_ndim,
    xprepare_values_for_reduction,
)
from .docstrings import docfiller

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from .typing import (
        ArrayLikeArg,
        AxisReduce,
        AxisReduceMult,
        DimsReduce,
        DimsReduceMult,
        DTypeLikeArg,
        FloatT,
        KeepAttrs,
        MissingType,
        Mom_NDim,
        MomDims,
        Moments,
        MomentsStrict,
        NDArrayAny,
        ScalarT,
    )


# * Moving average
@overload
def construct_rolling_window_array(
    x: xr.DataArray,
    window: int | Sequence[int],
    axis: AxisReduceMult | MissingType = ...,
    center: bool | Sequence[bool] = ...,
    stride: int | Sequence[int] = ...,
    fill_value: ArrayLike = ...,
    mom_ndim: Mom_NDim | None = ...,
    # xarray specific
    dim: DimsReduceMult | MissingType = ...,
    window_dim: str | Sequence[str] | None = ...,
    keep_attrs: bool | None = ...,
    **kwargs: Any,
) -> xr.DataArray: ...
@overload
def construct_rolling_window_array(
    x: NDArray[FloatT],
    window: int | Sequence[int],
    axis: AxisReduceMult | MissingType = ...,
    center: bool | Sequence[bool] = ...,
    stride: int | Sequence[int] = ...,
    fill_value: ArrayLike = ...,
    mom_ndim: Mom_NDim | None = ...,
    # xarray specific
    dim: DimsReduceMult | MissingType = ...,
    window_dim: str | Sequence[str] | None = ...,
    keep_attrs: bool | None = ...,
    **kwargs: Any,
) -> NDArray[FloatT]: ...
@overload
def construct_rolling_window_array(
    x: NDArrayAny,
    window: int | Sequence[int],
    axis: AxisReduceMult | MissingType = ...,
    center: bool | Sequence[bool] = ...,
    stride: int | Sequence[int] = ...,
    fill_value: ArrayLike = ...,
    mom_ndim: Mom_NDim | None = ...,
    # xarray specific
    dim: DimsReduceMult | MissingType = ...,
    window_dim: str | Sequence[str] | None = ...,
    keep_attrs: bool | None = ...,
    **kwargs: Any,
) -> NDArrayAny: ...


@docfiller.decorate
def construct_rolling_window_array(
    x: NDArrayAny | xr.DataArray,
    window: int | Sequence[int],
    axis: AxisReduceMult | MissingType = MISSING,
    center: bool | Sequence[bool] = False,
    stride: int | Sequence[int] = 1,
    fill_value: ArrayLike = np.nan,
    mom_ndim: Mom_NDim | None = None,
    # xarray specific
    dim: DimsReduceMult | MissingType = MISSING,
    window_dim: str | Sequence[str] | None = None,
    keep_attrs: bool | None = None,
    **kwargs: Any,
) -> NDArrayAny | xr.DataArray:
    """
    Convert an array to one with rolling windows.

    Parameters
    ----------
    x : array
    axis : int or iterable of int
    window : int or sequence of int
    center : bool
    fill_value : scalar

    Returns
    -------
    output : array
        Array of shape ``(*shape, window)`` if ``mom_ndim = None`` or
        ``(*shape[:-mom_ndim], window, *shape[-mom_ndim:])`` if ``mom_ndim is
        not None``. That is, the new window dimension is placed at the end, but
        before any moment dimensions if they are specified.

    Notes
    -----
    This function uses different syntax compared to
    :meth:`~xarray.DataArray.rolling`. Instead of mappings ffor ``center``,
    etc, here you pass scalar or sequence values corresponding to axis/dim.
    Corresponding mappings are created from, for example ``center=dict(zip(dim,
    center))``.


    Examples
    --------
    >>> x = np.arange(5).astype(float)
    >>> construct_rolling_window_array(x, window=4, axis=0)
    array([[nan, nan, nan,  0.],
           [nan, nan,  0.,  1.],
           [nan,  0.,  1.,  2.],
           [ 0.,  1.,  2.,  3.],
           [ 1.,  2.,  3.,  4.]])

    >>> dx = xr.DataArray(x)
    >>> construct_rolling_window_array(dx, window=4, center=True, dim="dim_0")
    <xarray.DataArray (dim_0: 5, rolling_dim_0: 4)> Size: 160B
    array([[nan, nan,  0.,  1.],
           [nan,  0.,  1.,  2.],
           [ 0.,  1.,  2.,  3.],
           [ 1.,  2.,  3.,  4.],
           [ 2.,  3.,  4., nan]])
    Dimensions without coordinates: dim_0, rolling_dim_0

    See Also
    --------
    xarray.DataArray.rolling
    xarray.core.rolling.DataArrayRolling.construct
    """
    if isinstance(x, xr.DataArray):
        mom_ndim = validate_mom_ndim(mom_ndim) if mom_ndim is not None else mom_ndim
        axis, dim = select_axis_dim_mult(x, axis=axis, dim=dim, mom_ndim=mom_ndim)

        nroll = len(axis)
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

        if mom_ndim is not None:
            xout = xout.transpose(..., *x.dims[-mom_ndim:])

        return xout

    return construct_rolling_window_array(
        x=xr.DataArray(x),
        window=window,
        axis=axis,
        center=center,
        stride=stride,
        fill_value=fill_value,
        mom_ndim=mom_ndim,
        **kwargs,
    ).to_numpy()


# * Moving
def _pad_along_axis(
    data: NDArray[ScalarT], axis: int, shift: int, fill_value: float
) -> NDArray[ScalarT]:
    pads = [(0, 0)] * data.ndim
    pads[axis] = (0, -shift)
    return np.pad(
        data,
        pads,
        mode="constant",
        constant_values=fill_value,
    )


def _optional_zero_missing_weight(
    data: NDArray[ScalarT], mom_ndim: Mom_NDim, zero_missing_weights: bool
) -> NDArray[ScalarT]:
    """Note that this modifies ``data`` inplace.  Pass in a copy if this is not what you want."""
    if zero_missing_weights:
        w = data[(..., *(0,) * mom_ndim)]  # type: ignore[arg-type]
        w[np.isnan(w)] = 0.0
    return data


# ** Data
@overload
def rolling_data(  # type: ignore[overload-overlap]
    data: xr.DataArray,
    *,
    window: int,
    axis: AxisReduce | MissingType = ...,
    mom_ndim: Mom_NDim = ...,
    min_periods: int | None = ...,
    center: bool = ...,
    zero_missing_weights: bool = ...,
    parallel: bool | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
) -> xr.DataArray: ...
# Array
@overload
def rolling_data(
    data: ArrayLikeArg[FloatT],
    *,
    window: int,
    axis: AxisReduce | MissingType = ...,
    mom_ndim: Mom_NDim = ...,
    min_periods: int | None = ...,
    center: bool = ...,
    zero_missing_weights: bool = ...,
    parallel: bool | None = ...,
    out: None = ...,
    dtype: None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
) -> NDArray[FloatT]: ...
# out
@overload
def rolling_data(
    data: Any,
    *,
    window: int,
    axis: AxisReduce | MissingType = ...,
    mom_ndim: Mom_NDim = ...,
    min_periods: int | None = ...,
    center: bool = ...,
    zero_missing_weights: bool = ...,
    parallel: bool | None = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def rolling_data(
    data: Any,
    *,
    window: int,
    axis: AxisReduce | MissingType = ...,
    mom_ndim: Mom_NDim = ...,
    min_periods: int | None = ...,
    center: bool = ...,
    zero_missing_weights: bool = ...,
    parallel: bool | None = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    # xarray specific
    dim: DimsReduce | MissingType = ...,
) -> NDArray[FloatT]: ...
# fallback
@overload
def rolling_data(
    data: Any,
    *,
    window: int,
    axis: AxisReduce | MissingType = ...,
    mom_ndim: Mom_NDim = ...,
    min_periods: int | None = ...,
    center: bool = ...,
    zero_missing_weights: bool = ...,
    parallel: bool | None = ...,
    out: Any = ...,
    dtype: Any = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
) -> NDArrayAny: ...


@docfiller.decorate
def rolling_data(  # pyright: ignore[reportOverlappingOverload]
    data: ArrayLike | xr.DataArray,
    *,
    window: int,
    axis: AxisReduce | MissingType = MISSING,
    mom_ndim: Mom_NDim = 1,
    min_periods: int | None = None,
    center: bool = False,
    zero_missing_weights: bool = True,
    parallel: bool | None = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    # xarray specific
    dim: DimsReduce | MissingType = MISSING,
) -> NDArrayAny | xr.DataArray:
    """
    Moving average of central moments array.

    Parameters
    ----------
    data : array-like or xarray.DataArray
    {window}
    {axis}
    {mom_ndim}
    {min_periods}
    {center}
    {zero_missing_weights}
    {parallel}
    {out}
    {dtype}
    {dim}

    Returns
    -------
    out : ndarray or DataArray
        Moving average data, of same shape and type as ``data``.
    """
    if isinstance(data, xr.DataArray):
        axis, dim = select_axis_dim(data, axis=axis, dim=dim, mom_ndim=mom_ndim)
        return data.copy(
            data=rolling_data(
                data.to_numpy(),
                window=window,
                axis=axis,
                mom_ndim=mom_ndim,
                min_periods=min_periods,
                center=center,
                parallel=parallel,
                zero_missing_weights=zero_missing_weights,
                dtype=dtype,
                out=out,
            )
        )

    mom_ndim = validate_mom_ndim(mom_ndim)
    dtype = select_dtype(data, out=out, dtype=dtype)

    # special to support multiple reduction dimensions...
    data = np.asarray(data, dtype=dtype)
    axis = normalize_axis_index(
        validate_axis(axis), data.ndim, mom_ndim, "rolling_data"
    )

    shift = (-window // 2) + 1 if center else None
    if shift is not None:
        data = _pad_along_axis(data, axis=axis, shift=shift, fill_value=0.0)

    axes = [
        # add in data_tmp, window, count
        tuple(range(-mom_ndim, 0)),
        (),
        (),
        *axes_data_reduction(
            mom_ndim=mom_ndim,
            axis=axis,
            out_has_axis=True,
        ),
    ]

    from ._lib.factory import factory_rolling_data

    min_periods = window if min_periods is None else min_periods
    data_tmp = np.zeros(data.shape[-mom_ndim:], dtype=dtype)

    out = factory_rolling_data(
        mom_ndim=mom_ndim,
        parallel=parallel_heuristic(parallel, data.size * mom_ndim),
    )(data_tmp, window, min_periods, data, out=out, axes=axes)

    if shift is not None:
        valid = [slice(None)] * data.ndim
        valid[axis] = slice(-shift, None)
        out = out[tuple(valid)]

    return _optional_zero_missing_weight(out, mom_ndim, zero_missing_weights)


# * Vals
@overload
def rolling_vals(  # type: ignore[overload-overlap]
    x: xr.DataArray,
    *y: ArrayLike | xr.DataArray,
    mom: Moments,
    window: int,
    weight: ArrayLike | xr.DataArray | None = ...,
    axis: AxisReduce | MissingType = ...,
    min_periods: int | None = ...,
    center: bool = ...,
    zero_missing_weights: bool = ...,
    parallel: bool | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> xr.DataArray: ...
# array
@overload
def rolling_vals(
    x: ArrayLikeArg[FloatT],
    *y: ArrayLike,
    mom: Moments,
    window: int,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    min_periods: int | None = ...,
    center: bool = ...,
    zero_missing_weights: bool = ...,
    parallel: bool | None = ...,
    out: None = ...,
    dtype: None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# out
@overload
def rolling_vals(
    x: Any,
    *y: ArrayLike,
    mom: Moments,
    window: int,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    min_periods: int | None = ...,
    center: bool = ...,
    zero_missing_weights: bool = ...,
    parallel: bool | None = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# out
@overload
def rolling_vals(
    x: Any,
    *y: ArrayLike,
    mom: Moments,
    window: int,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    min_periods: int | None = ...,
    center: bool = ...,
    zero_missing_weights: bool = ...,
    parallel: bool | None = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# fallback
@overload
def rolling_vals(
    x: Any,
    *y: ArrayLike,
    mom: Moments,
    window: int,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    min_periods: int | None = ...,
    center: bool = ...,
    zero_missing_weights: bool = ...,
    parallel: bool | None = ...,
    out: Any = ...,
    dtype: Any = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArrayAny: ...


@docfiller.decorate
def rolling_vals(  # pyright: ignore[reportOverlappingOverload]
    x: ArrayLike | xr.DataArray,
    *y: ArrayLike | xr.DataArray,
    mom: Moments,
    window: int,
    weight: ArrayLike | xr.DataArray | None = None,
    axis: AxisReduce | MissingType = MISSING,
    min_periods: int | None = None,
    center: bool = False,
    zero_missing_weights: bool = True,
    parallel: bool | None = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    # xarray specific
    dim: DimsReduce | MissingType = MISSING,
    mom_dims: MomDims | None = None,
    keep_attrs: KeepAttrs = None,
) -> NDArrayAny | xr.DataArray:
    """
    Moving average of central moments generated by values.

    Parameters
    ----------
    x : ndarray or DataArray
        Values to analyze.
    *y : array-like or DataArray
        Seconda value. Must specify if ``len(mom) == 2.`` Should either be able
        to broadcast to ``x`` or be 1d array with length ``x.shape[axis]``.
    {mom}
    {window}
    weight : scalar or array-like or DataArray
        Weights for each point. Should either be able to broadcast to ``x`` or
        be `d array of length ``x.shape[axis]``.
    {axis}
    {min_periods}
    {center}
    {zero_missing_weights}
    {parallel}
    {out}
    {dtype}
    {dim}
    {mom_dims}
    {keep_attrs}

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
    mom, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
    dtype = select_dtype(x, out=out, dtype=dtype)
    weight = 1.0 if weight is None else weight

    if isinstance(x, xr.DataArray):
        input_core_dims, args = xprepare_values_for_reduction(
            x,
            weight,
            *y,
            axis=axis,
            dim=dim,
            narrays=mom_ndim + 1,
            dtype=dtype,
        )

        mom_dims_strict = validate_mom_dims(mom_dims=mom_dims, mom_ndim=mom_ndim)

        return xr.apply_ufunc(  # type: ignore[no-any-return]
            _rolling_vals,
            *args,
            input_core_dims=input_core_dims,
            output_core_dims=[[input_core_dims[0][0], *mom_dims_strict]],
            kwargs={
                "mom": mom,
                "mom_ndim": mom_ndim,
                "parallel": parallel,
                "out": out,
                "center": center,
                "window": window,
                "min_periods": min_periods,
                "zero_missing_weights": zero_missing_weights,
            },
            keep_attrs=keep_attrs,
        )

    axis, args = prepare_values_for_reduction(
        x,
        weight,
        *y,
        axis=axis,
        narrays=mom_ndim + 1,
        dtype=dtype,
    )

    return _rolling_vals(
        *args,
        mom=mom,
        mom_ndim=mom_ndim,
        parallel=parallel,
        out=out,
        center=center,
        window=window,
        min_periods=min_periods,
        zero_missing_weights=zero_missing_weights,
    )


def _rolling_vals(
    x: NDArray[FloatT],
    w: NDArray[FloatT],
    *y: NDArray[FloatT],
    mom: MomentsStrict,
    mom_ndim: Mom_NDim,
    parallel: bool | None,
    out: NDArray[FloatT] | None,
    center: bool,
    window: int,
    min_periods: int | None,
    zero_missing_weights: bool,
) -> NDArray[FloatT]:
    shift = (-window // 2) + 1 if center else None
    if shift is not None:
        (x, w, *y) = tuple(  # type: ignore[assignment]
            _pad_along_axis(_, axis=-1, shift=shift, fill_value=0.0) for _ in (x, w, *y)
        )

    min_periods = window if min_periods is None else min_periods
    data_tmp = np.zeros(mom_to_mom_shape(mom), dtype=x.dtype)

    from ._lib.factory import factory_rolling_vals

    out = factory_rolling_vals(
        mom_ndim=mom_ndim, parallel=parallel_heuristic(parallel, x.size * mom_ndim)
    )(data_tmp, window, min_periods, x, w, *y, out=out)

    if shift is not None:
        valid = [slice(None)] * out.ndim
        valid[-(mom_ndim + 1)] = slice(-shift, None)
        out = out[tuple(valid)]

    return _optional_zero_missing_weight(out, mom_ndim, zero_missing_weights)  # pyright: ignore[reportReturnType, reportArgumentType]


# * Move Exponential
# ** Data
@overload
def rolling_exp_data(  # type: ignore[overload-overlap]
    data: xr.DataArray,
    alpha: ArrayLike,
    *,
    axis: AxisReduce | MissingType = ...,
    mom_ndim: Mom_NDim = ...,
    min_periods: int | None = ...,
    adjust: bool = ...,
    zero_missing_weights: bool = ...,
    parallel: bool | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    dim: DimsReduce | MissingType = ...,
) -> xr.DataArray: ...
# array
@overload
def rolling_exp_data(
    data: ArrayLikeArg[FloatT],
    alpha: ArrayLike,
    *,
    axis: AxisReduce | MissingType = ...,
    mom_ndim: Mom_NDim = ...,
    min_periods: int | None = ...,
    adjust: bool = ...,
    zero_missing_weights: bool = ...,
    parallel: bool | None = ...,
    out: None = ...,
    dtype: None = ...,
    dim: DimsReduce | MissingType = ...,
) -> NDArray[FloatT]: ...
# out
@overload
def rolling_exp_data(
    data: Any,
    alpha: ArrayLike,
    *,
    axis: AxisReduce | MissingType = ...,
    mom_ndim: Mom_NDim = ...,
    min_periods: int | None = ...,
    adjust: bool = ...,
    zero_missing_weights: bool = ...,
    parallel: bool | None = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    dim: DimsReduce | MissingType = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def rolling_exp_data(
    data: Any,
    alpha: ArrayLike,
    *,
    axis: AxisReduce | MissingType = ...,
    mom_ndim: Mom_NDim = ...,
    min_periods: int | None = ...,
    adjust: bool = ...,
    zero_missing_weights: bool = ...,
    parallel: bool | None = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    dim: DimsReduce | MissingType = ...,
) -> NDArray[FloatT]: ...
# fallback
@overload
def rolling_exp_data(
    data: Any,
    alpha: ArrayLike,
    *,
    axis: AxisReduce | MissingType = ...,
    mom_ndim: Mom_NDim = ...,
    min_periods: int | None = ...,
    adjust: bool = ...,
    zero_missing_weights: bool = ...,
    parallel: bool | None = ...,
    out: Any = ...,
    dtype: Any = ...,
    dim: DimsReduce | MissingType = ...,
) -> NDArray[FloatT]: ...


@docfiller.decorate
def rolling_exp_data(  # pyright: ignore[reportOverlappingOverload]
    data: ArrayLike | xr.DataArray,
    alpha: ArrayLike,
    *,
    axis: AxisReduce | MissingType = MISSING,
    mom_ndim: Mom_NDim = 1,
    min_periods: int | None = None,
    adjust: bool = True,
    zero_missing_weights: bool = True,
    parallel: bool | None = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    dim: DimsReduce | MissingType = MISSING,
) -> NDArrayAny | xr.DataArray:
    """
    Moving average of central moments array.

    Parameters
    ----------
    data : array-like
    alpha : array-like
        `alpha` values.
    {axis}
    {mom_ndim}
    {min_periods}
    adjust : bool, default=True
        Same as ``adjust`` parameter of :meth:`pandas.DataFrame.ewm`
    {zero_missing_weights}
    {parallel}
    {out}
    {dtype}

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
    if isinstance(data, xr.DataArray):
        axis, dim = select_axis_dim(data, axis=axis, dim=dim, mom_ndim=mom_ndim)
        return data.copy(
            data=rolling_exp_data(
                data.to_numpy(),
                alpha=alpha,
                axis=axis,
                mom_ndim=mom_ndim,
                min_periods=min_periods,
                adjust=adjust,
                zero_missing_weights=zero_missing_weights,
                parallel=parallel,
                out=out,
                dtype=dtype,
            )
        )

    mom_ndim = validate_mom_ndim(mom_ndim)
    dtype = select_dtype(data, out=out, dtype=dtype)

    data = np.asarray(data, dtype=dtype)
    axis = normalize_axis_index(
        validate_axis(axis), data.ndim, mom_ndim, "rolling_data"
    )

    alpha = np.asarray(alpha, dtype=dtype)
    if alpha.ndim == 0:
        alpha = np.broadcast_to(alpha, data.shape[axis])
        alpha_axis = -1
    elif alpha.ndim == 1:
        alpha_axis = -1
    else:
        alpha_axis = axis

    # axes for reduction
    axes = [
        # add in data_tmp, alpha, adjust, min_periods
        tuple(range(-mom_ndim, 0)),
        (alpha_axis,),
        (),
        (),
        *axes_data_reduction(
            mom_ndim=mom_ndim,
            axis=axis,
            out_has_axis=True,
        ),
    ]

    from ._lib.factory import factory_rolling_exp_data

    min_periods = 1 if min_periods is None else max(1, min_periods)
    data_tmp = np.zeros(data.shape[-mom_ndim:], dtype=dtype)
    out = factory_rolling_exp_data(
        mom_ndim=mom_ndim,
        parallel=parallel_heuristic(parallel, data.size * mom_ndim),
    )(data_tmp, alpha, adjust, min_periods, data, out=out, axes=axes)

    return _optional_zero_missing_weight(out, mom_ndim, zero_missing_weights)


# ** Vals
@overload
def rolling_exp_vals(  # type: ignore[overload-overlap]
    x: xr.DataArray,
    *y: ArrayLike | xr.DataArray,
    alpha: ArrayLike | xr.DataArray,
    mom: Moments,
    weight: ArrayLike | xr.DataArray | None = ...,
    axis: AxisReduce | MissingType = ...,
    min_periods: int | None = ...,
    adjust: bool = ...,
    zero_missing_weights: bool = ...,
    parallel: bool | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> xr.DataArray: ...
# Array
@overload
def rolling_exp_vals(
    x: ArrayLikeArg[FloatT],
    *y: ArrayLike,
    alpha: ArrayLike,
    mom: Moments,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    min_periods: int | None = ...,
    adjust: bool = ...,
    zero_missing_weights: bool = ...,
    parallel: bool | None = ...,
    out: None = ...,
    dtype: None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# out
@overload
def rolling_exp_vals(
    x: Any,
    *y: ArrayLike,
    alpha: ArrayLike,
    mom: Moments,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    min_periods: int | None = ...,
    adjust: bool = ...,
    zero_missing_weights: bool = ...,
    parallel: bool | None = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def rolling_exp_vals(
    x: Any,
    *y: ArrayLike,
    alpha: ArrayLike,
    mom: Moments,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    min_periods: int | None = ...,
    adjust: bool = ...,
    zero_missing_weights: bool = ...,
    parallel: bool | None = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# Fallback
@overload
def rolling_exp_vals(
    x: Any,
    *y: ArrayLike,
    alpha: ArrayLike,
    mom: Moments,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    min_periods: int | None = ...,
    adjust: bool = ...,
    zero_missing_weights: bool = ...,
    parallel: bool | None = ...,
    out: Any = ...,
    dtype: Any = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...


@docfiller.decorate
def rolling_exp_vals(  # pyright: ignore[reportOverlappingOverload]
    x: ArrayLike | xr.DataArray,
    *y: ArrayLike | xr.DataArray,
    alpha: ArrayLike | xr.DataArray,
    mom: Moments,
    weight: ArrayLike | xr.DataArray | None = None,
    axis: AxisReduce | MissingType = MISSING,
    min_periods: int | None = None,
    adjust: bool = True,
    zero_missing_weights: bool = True,
    parallel: bool | None = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    # xarray specific
    dim: DimsReduce | MissingType = MISSING,
    mom_dims: MomDims | None = None,
    keep_attrs: KeepAttrs = None,
) -> NDArrayAny | xr.DataArray:
    """
    Moving average of central moments generated by values.

    Parameters
    ----------
    x : ndarray or DataArray
        Values to analyze.
    *y : array-like or DataArray
        Seconda value. Must specify if ``len(mom) == 2.`` Should either be able
        to broadcast to ``x`` or be 1d array with length ``x.shape[axis]``.
    alpha : array-like
        `alpha` values.
    {mom}
    weight : scalar or array-like or DataArray
        Weights for each point. Should either be able to broadcast to ``x`` or
        be `d array of length ``x.shape[axis]``.
    {axis}
    {min_periods}
    adjust : bool, default=True
        Same as ``adjust`` parameter of :meth:`pandas.DataFrame.ewm`
    {zero_missing_weights}
    {parallel}
    {out}
    {dtype}
    {dim}
    {mom_dims}
    {keep_attrs}

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
    mom, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
    dtype = select_dtype(x, out=out, dtype=dtype)
    weight = 1.0 if weight is None else weight

    if isinstance(x, xr.DataArray):
        input_core_dims, (x, weight, alpha, *y) = xprepare_values_for_reduction(  # type: ignore[assignment]
            x,
            weight,
            alpha,
            *y,
            axis=axis,
            dim=dim,
            narrays=mom_ndim + 2,
            dtype=dtype,
        )

        mom_dims_strict = validate_mom_dims(mom_dims=mom_dims, mom_ndim=mom_ndim)

        return xr.apply_ufunc(  # type: ignore[no-any-return]
            _rolling_exp_vals,
            x,
            weight,
            alpha,
            *y,
            input_core_dims=input_core_dims,
            output_core_dims=[[input_core_dims[0][0], *mom_dims_strict]],
            kwargs={
                "mom": mom,
                "mom_ndim": mom_ndim,
                "parallel": parallel,
                "out": out,
                "adjust": adjust,
                "min_periods": min_periods,
                "zero_missing_weights": zero_missing_weights,
            },
            keep_attrs=keep_attrs,
        )

    axis, args = prepare_values_for_reduction(
        x,
        weight,
        alpha,
        *y,
        axis=axis,
        narrays=mom_ndim + 2,
        dtype=dtype,
    )

    return _rolling_exp_vals(
        *args,
        mom=mom,
        mom_ndim=mom_ndim,
        parallel=parallel,
        out=out,
        adjust=adjust,
        min_periods=min_periods,
        zero_missing_weights=zero_missing_weights,
    )


def _rolling_exp_vals(
    x: NDArray[FloatT],
    w: NDArray[FloatT],
    alpha: NDArray[FloatT],
    *y: NDArray[FloatT],
    mom: MomentsStrict,
    mom_ndim: Mom_NDim,
    parallel: bool | None,
    out: NDArray[FloatT] | None,
    adjust: bool,
    min_periods: int | None,
    zero_missing_weights: bool,
) -> NDArray[FloatT]:
    data_tmp = np.zeros(mom_to_mom_shape(mom), dtype=x.dtype)

    from ._lib.factory import factory_rolling_exp_vals

    min_periods = 1 if min_periods is None else max(1, min_periods)
    out = factory_rolling_exp_vals(
        mom_ndim=mom_ndim, parallel=parallel_heuristic(parallel, x.size * mom_ndim)
    )(data_tmp, alpha, adjust, min_periods, x, w, *y, out=out)

    return _optional_zero_missing_weight(out, mom_ndim, zero_missing_weights)  # pyright: ignore[reportReturnType, reportArgumentType]

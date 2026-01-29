"""resample and reduce"""
# pylint: disable=duplicate-code

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr

from cmomy.core.array_utils import (
    asarray_maybe_recast,
    get_axes_from_values,
    select_dtype,
)
from cmomy.core.docstrings import docfiller
from cmomy.core.missing import MISSING
from cmomy.core.moment_params import (
    MomParamsArray,
    MomParamsXArray,
    factory_mom_params,
)
from cmomy.core.prepare import (
    PrepareDataArray,
    PrepareDataXArray,
    PrepareValsArray,
    PrepareValsXArray,
)
from cmomy.core.utils import (
    mom_to_mom_shape,
)
from cmomy.core.validate import (
    is_dataarray,
    is_dataset,
    is_ndarray,
    is_xarray,
    is_xarray_typevar,
    raise_if_wrong_value,
)
from cmomy.core.xr_utils import (
    factory_apply_ufunc_kwargs,
    raise_if_dataset,
    transpose_like,
)
from cmomy.factory import (
    factory_jackknife_data,
    factory_jackknife_vals,
    factory_resample_data,
    factory_resample_vals,
    parallel_heuristic,
)

from ._factory_sampler import (
    factory_sampler,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from cmomy.core._typing_kwargs import (
        ApplyUFuncKwargs,
        JackknifeDataKwargs,
        JackknifeValsKwargs,
    )
    from cmomy.core.moment_params import MomParamsType
    from cmomy.core.typing import (
        ArrayLikeArg,
        ArrayOrderCF,
        ArrayOrderKACF,
        AxesGUFunc,
        AxisReduceWrap,
        Casting,
        DataT,
        DimsReduce,
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
    from cmomy.core.typing_compat import Unpack

    from ._typing_kwargs import ResampleDataKwargs, ResampleValsKwargs
    from .typing import SamplerType


# * Resample data
# ** overloads
@overload
def resample_data(  # pyright: ignore[reportOverlappingOverload]
    data: DataT,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ResampleDataKwargs],
) -> DataT: ...
# array no out or dtype
@overload
def resample_data(
    data: ArrayLikeArg[FloatT],
    *,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[ResampleDataKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def resample_data(
    data: ArrayLike,
    *,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ResampleDataKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def resample_data(
    data: ArrayLike,
    *,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[ResampleDataKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def resample_data(
    data: ArrayLike,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ResampleDataKwargs],
) -> NDArrayAny: ...
# arraylike or DataT
@overload
def resample_data(
    data: ArrayLike | DataT,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ResampleDataKwargs],
) -> NDArrayAny | DataT: ...


# ** Public api
@docfiller.decorate  # type: ignore[arg-type,unused-ignore]
def resample_data(  # noqa: PLR0913
    data: ArrayLike | DataT,
    *,
    sampler: SamplerType,
    axis: AxisReduceWrap | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: MomNDim | None = None,
    mom_axes: MomAxes | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsType = None,
    rep_dim: str = "rep",
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderKACF = None,
    parallel: bool | None = None,
    axes_to_end: bool = False,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Resample and reduce data.

    Parameters
    ----------
    {data_numpy_or_dataarray_or_dataset}
    {sampler}
    {axis}
    {dim}
    {mom_ndim_data}
    {mom_axes}
    {mom_dims_data}
    {mom_params}
    {rep_dim}
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
        Resampled central moments. ``out.shape = (..., shape[axis-1], nrep, shape[axis+1], ...)``,
        where ``shape = data.shape`` and ``nrep = sampler.nrep`` .

    See Also
    --------
    random_freq
    factory_sampler
    """
    dtype = select_dtype(data, out=out, dtype=dtype)

    sampler = factory_sampler(
        sampler,
        data=data,
        axis=axis,
        dim=dim,
        mom_ndim=mom_ndim,
        mom_axes=mom_axes,
        mom_dims=mom_dims,
        mom_params=mom_params,
        rep_dim=rep_dim,
        parallel=parallel,
    )

    if is_xarray_typevar["DataT"].check(data):
        prep = PrepareDataXArray.factory(
            mom_params=mom_params,
            ndim=mom_ndim,
            axes=mom_axes,
            dims=mom_dims,
            data=data,
            default_ndim=1,
        )
        axis, dim = prep.mom_params.select_axis_dim(data, axis=axis, dim=dim)

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _resample_data,
            data,
            sampler.freq,
            input_core_dims=[prep.mom_params.core_dims(dim), [rep_dim, dim]],
            output_core_dims=[prep.mom_params.core_dims(rep_dim)],
            kwargs={
                "prep": prep.prepare_array,
                "axis": -(prep.mom_params.ndim + 1),
                "out": prep.optional_out_sample(
                    out,
                    data=data,
                    axis=axis,
                    axis_new_size=sampler.nrep,
                    axes_to_end=axes_to_end,
                    order=order,
                    dtype=dtype,
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
                output_sizes={rep_dim: sampler.nrep},
                output_dtypes=dtype if dtype is not None else np.float64,  # type: ignore[redundant-expr]
            ),
        )

        if not axes_to_end:
            xout = transpose_like(
                xout,
                template=data,
                replace={dim: rep_dim},
            )
        elif is_dataset(xout):
            xout = xout.transpose(
                ..., rep_dim, *prep.mom_params.dims, missing_dims="ignore"
            )

        if not axes_to_end and is_dataarray(data):
            dims_order = (*data.dims[:axis], rep_dim, *data.dims[axis + 1 :])  # type: ignore[union-attr,misc,operator,index,unused-ignore]
            xout = xout.transpose(*dims_order)
        return xout

    # Numpy
    prep, axis, data = PrepareDataArray.factory(
        mom_params=mom_params,
        ndim=mom_ndim,
        axes=mom_axes,
        default_ndim=1,
    ).data_for_reduction(
        data=data,
        axis=axis,
        axes_to_end=axes_to_end,
        dtype=dtype,
    )
    freq = sampler.freq
    assert is_ndarray(freq)  # noqa: S101

    return _resample_data(
        data,
        freq,
        prep=prep,
        axis=axis,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        parallel=parallel,
        fastpath=True,
    )


def _resample_data(
    data: NDArrayAny,
    freq: NDArrayAny,
    *,
    prep: PrepareDataArray,
    axis: int,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrderKACF,
    parallel: bool | None,
    fastpath: bool,
) -> NDArrayAny:
    dtype = select_dtype(data, out=out, dtype=dtype, fastpath=fastpath)

    # include inner core dimensions for freq
    axes = [
        (-2, -1),
        *prep.mom_params.axes_data_reduction(
            axis=axis,
            out_has_axis=True,
        ),
    ]

    out = prep.optional_out_sample(
        data=data,
        out=out,
        axis=axis,
        axis_new_size=freq.shape[-2],
        order=order,
        dtype=dtype,
    )

    return factory_resample_data(
        mom_ndim=prep.mom_params.ndim,
        parallel=parallel_heuristic(parallel, size=data.size),
    )(
        freq,
        data,
        out=out,
        axes=axes,
        dtype=dtype,
        casting=casting,
        order=order,
    )


# * Resample vals
# ** overloads
@overload
def resample_vals(  # pyright: ignore[reportOverlappingOverload]
    x: DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    weight: ArrayLike | xr.DataArray | DataT | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ResampleValsKwargs],
) -> DataT: ...
# array
@overload
def resample_vals(
    x: ArrayLikeArg[FloatT],
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[ResampleValsKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def resample_vals(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ResampleValsKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def resample_vals(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[ResampleValsKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def resample_vals(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ResampleValsKwargs],
) -> NDArrayAny: ...
# arraylike or DataT
@overload
def resample_vals(
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    weight: ArrayLike | xr.DataArray | DataT | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ResampleValsKwargs],
) -> NDArrayAny | DataT: ...


# ** public api
@docfiller.decorate  # type: ignore[arg-type,unused-ignore]
def resample_vals(  # noqa: PLR0913
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    sampler: SamplerType,
    mom: Moments,
    weight: ArrayLike | xr.DataArray | DataT | None = None,
    axis: AxisReduceWrap | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_dims: MomDims | None = None,
    mom_axes: MomAxes | None = None,
    mom_params: MomParamsType = None,
    rep_dim: str = "rep",
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
    Resample and reduce values.

    Parameters
    ----------
    {x_genarray}
    {y_genarray}
    {sampler}
    {mom}
    {axis}
    {dim}
    {weight_genarray}
    {mom_dims}
    {mom_axes}
    {mom_params}
    {rep_dim}
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
    out : ndarray or DataArray or Dataset
        Resampled Central moments array. ``out.shape = (...,shape[axis-1], nrep, shape[axis+1], ...)``
        where ``shape = x.shape``. and ``nrep = sampler.nrep``.  This can be overridden by setting `axes_to_end`.

    Notes
    -----
    {vals_resample_note}

    See Also
    --------
    .resample.factory_sampler
    """
    weight = 1.0 if weight is None else weight
    dtype = select_dtype(x, out=out, dtype=dtype)

    sampler = factory_sampler(
        sampler,
        data=x,
        axis=axis,
        dim=dim,
        rep_dim=rep_dim,
        parallel=parallel,
    )

    if is_xarray_typevar["DataT"].check(x):
        prep, mom = PrepareValsXArray.factory_mom(
            mom=mom, mom_params=mom_params, dims=mom_dims, recast=False
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
            axis_new_size=sampler.nrep,
            axes_to_end=axes_to_end,
            order=order,
            dtype=dtype,
            mom_axes=mom_axes,
            mom_params=prep.mom_params,
        )

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _resample_vals,
            *xargs,
            sampler.freq,
            input_core_dims=[*input_core_dims, [rep_dim, dim]],  # type: ignore[has-type]
            output_core_dims=[prep.mom_params.core_dims(rep_dim)],
            kwargs={
                "mom": mom,
                "prep": prep.prepare_array,
                "axis_neg": -1,
                "out": out,
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
                output_sizes={
                    rep_dim: sampler.nrep,
                    **dict(
                        zip(prep.mom_params.dims, mom_to_mom_shape(mom), strict=True)
                    ),
                },
                output_dtypes=dtype if dtype is not None else np.float64,  # type: ignore[redundant-expr]
            ),
        )

        if not axes_to_end:
            return transpose_like(
                xout,
                template=x,
                replace={dim: rep_dim},
                append=prep.mom_params.dims,
                mom_params_axes=mom_params_axes,
            )
        if is_dataset(x):
            return xout.transpose(
                ...,
                rep_dim,
                *prep.mom_params.dims,
                missing_dims="ignore",
            )
        return xout

    # numpy
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

    return _resample_vals(
        *args,
        sampler.freq,
        mom=mom,
        prep=prep,
        axis_neg=axis_neg,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        parallel=parallel,
        fastpath=True,
    )


def _resample_vals(
    # x, w, *y., freq
    *args: NDArrayAny,
    mom: MomentsStrict,
    prep: PrepareValsArray,
    axis_neg: int,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrderCF,
    parallel: bool | None,
    fastpath: bool,
) -> NDArrayAny:
    dtype = select_dtype(args[0], out=out, dtype=dtype, fastpath=fastpath)

    args, freq = args[:-1], args[-1]

    out, axis_sample_out = prep.out_from_values(
        out,
        val_shape=prep.get_val_shape(*args),
        mom=mom,
        axis_neg=axis_neg,
        axis_new_size=freq.shape[0],
        dtype=dtype,
        order=order,
    )
    out.fill(0.0)

    axes: AxesGUFunc = [
        # out
        (axis_sample_out, *prep.mom_params.axes),
        # freq
        (-2, -1),
        # x, weight, *y
        *get_axes_from_values(*args, axis_neg=axis_neg),
    ]

    _ = factory_resample_vals(
        mom_ndim=prep.mom_params.ndim,
        parallel=parallel_heuristic(parallel, size=args[0].size * prep.mom_params.ndim),
    )(
        out,
        freq,
        *args,
        axes=axes,
        casting=casting,
        order=order,
        signature=(dtype,) * (len(args) + 2),
    )
    return out


# * Jackknife resampling
# * Jackknife data
# ** overloads
@overload
def jackknife_data(  # pyright: ignore[reportOverlappingOverload]
    data: DataT,
    data_reduced: ArrayLike | DataT | None = ...,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[JackknifeDataKwargs],
) -> DataT: ...
# array
@overload
def jackknife_data(
    data: ArrayLikeArg[FloatT],
    data_reduced: ArrayLike | None = ...,
    *,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[JackknifeDataKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def jackknife_data(
    data: ArrayLike,
    data_reduced: ArrayLike | None = ...,
    *,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[JackknifeDataKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def jackknife_data(
    data: ArrayLike,
    data_reduced: ArrayLike | None = ...,
    *,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[JackknifeDataKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def jackknife_data(
    data: ArrayLike,
    data_reduced: ArrayLike | None = ...,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[JackknifeDataKwargs],
) -> NDArrayAny: ...
# arraylike or DataT
@overload
def jackknife_data(
    data: ArrayLike | DataT,
    data_reduced: ArrayLike | None = ...,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[JackknifeDataKwargs],
) -> NDArrayAny | DataT: ...


@docfiller.decorate  # type: ignore[arg-type,unused-ignore]
def jackknife_data(  # noqa: PLR0913
    data: ArrayLike | DataT,
    data_reduced: ArrayLike | DataT | None = None,
    *,
    axis: AxisReduceWrap | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: MomNDim | None = None,
    mom_axes: MomAxes | None = None,
    mom_axes_reduced: MomAxes | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsType = None,
    rep_dim: str | None = "rep",
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderKACF = None,
    parallel: bool | None = None,
    axes_to_end: bool = False,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Perform jackknife resample and moments data.

    This uses moments addition/subtraction to speed up jackknife resampling.

    Parameters
    ----------
    {data_numpy_or_dataarray_or_dataset}
    data_reduced : array-like or DataArray, optional
        ``data`` reduced along ``axis`` or ``dim``.  This will be calculated using
        :func:`.reduce_data` if not passed.
    {axis}
    {dim}
    {mom_ndim_data}
    {mom_axes}
    mom_axes_reduced : int or sequence of int
        Location(s) of moment dimensions in ``data_reduced``.  This option is only needed
        if ``data_reduced`` is passed in and is an array.  Defaults to ``mom_axes``, or last dimensions
        of ``data_reduced``.
    {mom_dims_data}
    {mom_params}
    {rep_dim}
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
        Jackknife resampled  along ``axis``.  That is,
        ``out[...,axis=i, ...]`` is ``reduced_data(out[...,axis=[...,i-1,i+1,...], ...])``.


    Examples
    --------
    >>> import cmomy
    >>> data = cmomy.default_rng(0).random((4, 3))
    >>> out_jackknife = jackknife_data(data, mom_ndim=1, axis=0)
    >>> out_jackknife
    array([[1.5582, 0.7822, 0.2247],
           [2.1787, 0.6322, 0.22  ],
           [1.5886, 0.5969, 0.0991],
           [1.2601, 0.4982, 0.3478]])

    Note that this is equivalent to (but typically faster than) resampling with a
    frequency table from :func:``cmomy.resample.jackknife_freq``

    >>> freq = cmomy.resample.jackknife_freq(4)
    >>> resample_data(data, sampler=dict(freq=freq), mom_ndim=1, axis=0)
    array([[1.5582, 0.7822, 0.2247],
           [2.1787, 0.6322, 0.22  ],
           [1.5886, 0.5969, 0.0991],
           [1.2601, 0.4982, 0.3478]])

    To speed up the calculation even further, pass in ``data_reduced``

    >>> data_reduced = cmomy.reduce_data(data, mom_ndim=1, axis=0)
    >>> jackknife_data(data, mom_ndim=1, axis=0, data_reduced=data_reduced)
    array([[1.5582, 0.7822, 0.2247],
           [2.1787, 0.6322, 0.22  ],
           [1.5886, 0.5969, 0.0991],
           [1.2601, 0.4982, 0.3478]])


    Also works with :class:`~xarray.DataArray` objects

    >>> xdata = xr.DataArray(data, dims=["samp", "mom"])
    >>> jackknife_data(xdata, mom_ndim=1, dim="samp", rep_dim="jackknife")
    <xarray.DataArray (jackknife: 4, mom: 3)> Size: 96B
    array([[1.5582, 0.7822, 0.2247],
           [2.1787, 0.6322, 0.22  ],
           [1.5886, 0.5969, 0.0991],
           [1.2601, 0.4982, 0.3478]])
    Dimensions without coordinates: jackknife, mom

    """
    dtype = select_dtype(data, out=out, dtype=dtype)
    mom_params = factory_mom_params(
        data,
        mom_params=mom_params,
        ndim=mom_ndim,
        axes=mom_axes,
        dims=mom_dims,
        data=data,
        default_ndim=1,
    )

    mom_axes_reduced = (
        mom_params.axes_last
        if mom_axes_reduced is None
        else MomParamsArray.factory(ndim=mom_params.ndim, axes=mom_axes_reduced).axes
    )

    if data_reduced is None:
        from cmomy.reduction import reduce_data

        data_reduced = reduce_data(
            data=data,
            mom_params=mom_params,
            dim=dim,
            axis=axis,
            parallel=parallel,
            keep_attrs=keep_attrs,
            dtype=dtype,
            casting=casting,
            order=order,
            apply_ufunc_kwargs=apply_ufunc_kwargs,
            axes_to_end=True,
        )
        mom_axes_reduced = mom_params.axes_last

    elif not is_xarray(data_reduced):
        data_reduced = asarray_maybe_recast(data_reduced, dtype=dtype, recast=False)

    if is_xarray_typevar["DataT"].check(data):
        assert isinstance(mom_params, MomParamsXArray)  # noqa: S101
        prep = PrepareDataXArray(mom_params=mom_params, recast=False)
        axis, dim = mom_params.select_axis_dim(data, axis=axis, dim=dim)
        core_dims = mom_params.core_dims(dim)

        if is_xarray(data_reduced):
            mom_axes_reduced = mom_params.axes_last

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _jackknife_data,
            data,
            data_reduced,
            input_core_dims=[core_dims, core_dims[1:]],
            output_core_dims=[core_dims],
            kwargs={
                "axis": -(mom_params.ndim + 1),
                "prep": prep.prepare_array,
                "mom_axes_reduced": mom_axes_reduced,
                "out": prep.optional_out_sample(
                    out,
                    data=data,
                    axis=axis,
                    axes_to_end=axes_to_end,
                    order=order,
                    dtype=dtype,
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
                output_dtypes=dtype if dtype is not None else np.float64,
            ),
        )

        if not axes_to_end:
            xout = transpose_like(
                xout,
                template=data,
            )
        elif is_dataset(xout):
            xout = xout.transpose(..., dim, *mom_params.dims, missing_dims="ignore")

        if rep_dim is not None:
            xout = xout.rename({dim: rep_dim})
        return xout

    # numpy
    assert isinstance(mom_params, MomParamsArray)  # noqa: S101
    assert is_ndarray(data_reduced)  # noqa: S101

    prep, axis, data = PrepareDataArray(
        mom_params=mom_params,
        recast=False,
    ).data_for_reduction(
        data=data,
        axis=axis,
        axes_to_end=axes_to_end,
        dtype=dtype,
    )

    return _jackknife_data(
        data,
        data_reduced,
        prep=prep,
        mom_axes_reduced=mom_axes_reduced,
        axis=axis,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        parallel=parallel,
        fastpath=True,
    )


def _jackknife_data(
    data: NDArrayAny,
    data_reduced: NDArrayAny,
    *,
    prep: PrepareDataArray,
    # mom_axes_reduced: MomAxes,
    mom_axes_reduced: MomAxes,
    axis: int,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrderKACF,
    parallel: bool | None,
    fastpath: bool,
) -> NDArrayAny:
    dtype = select_dtype(data, out=out, dtype=dtype, fastpath=fastpath)

    axes = [
        # data_reduce
        mom_axes_reduced,
        # data, out
        *prep.mom_params.axes_data_reduction(axis=axis, out_has_axis=True),
    ]

    out = prep.optional_out_sample(
        data=data,
        out=out,
        axis=axis,
        axis_new_size=data.shape[axis],
        order=order,
        dtype=dtype,
    )

    return factory_jackknife_data(
        mom_ndim=prep.mom_params.ndim,
        parallel=parallel_heuristic(parallel, size=data.size),
    )(
        data_reduced,
        data,
        out=out,
        axes=axes,
        dtype=dtype,
        casting=casting,
        order=order,
    )


# * Jackknife vals
# ** overloads
# xarray
@overload
def jackknife_vals(  # pyright: ignore[reportOverlappingOverload]
    x: DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    data_reduced: ArrayLike | DataT | None = ...,
    weight: ArrayLike | xr.DataArray | DataT | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[JackknifeValsKwargs],
) -> DataT: ...
# array
@overload
def jackknife_vals(
    x: ArrayLikeArg[FloatT],
    *y: ArrayLike,
    data_reduced: ArrayLike | None = ...,
    weight: ArrayLike | None = ...,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[JackknifeValsKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def jackknife_vals(
    x: ArrayLike,
    *y: ArrayLike,
    data_reduced: ArrayLike | None = ...,
    weight: ArrayLike | None = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[JackknifeValsKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def jackknife_vals(
    x: ArrayLike,
    *y: ArrayLike,
    data_reduced: ArrayLike | None = ...,
    weight: ArrayLike | None = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[JackknifeValsKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def jackknife_vals(
    x: ArrayLike,
    *y: ArrayLike,
    data_reduced: ArrayLike | None = ...,
    weight: ArrayLike | None = ...,
    out: None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[JackknifeValsKwargs],
) -> NDArrayAny: ...
# arraylike or DataT
@overload
def jackknife_vals(
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    data_reduced: ArrayLike | xr.DataArray | DataT | None = ...,
    weight: ArrayLike | None = ...,
    out: None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[JackknifeValsKwargs],
) -> NDArrayAny | DataT: ...


@docfiller.decorate
def jackknife_vals(  # noqa: PLR0913
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    data_reduced: ArrayLike | DataT | None = None,
    mom: Moments,
    axis: AxisReduceWrap | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    weight: ArrayLike | xr.DataArray | DataT | None = None,
    mom_dims: MomDims | None = None,
    mom_axes: MomAxes | None = None,
    mom_axes_reduced: MomAxes | None = None,
    mom_params: MomParamsType = None,
    rep_dim: str | None = "rep",
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderCF = None,
    parallel: bool | None = None,
    axes_to_end: bool = False,
    # xarray specific
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> xr.Dataset | xr.DataArray | NDArrayAny:
    """
    Jackknife by value.

    Parameters
    ----------
    {x_genarray}
    {y_genarray}
    data_reduced : array-like or DataArray, optional
        ``data`` reduced along ``axis`` or ``dim``.  This will be calculated using
        :func:`.reduce_vals` if not passed.  Same type restrictions as ``weight``.
    {mom}
    {axis}
    {dim}
    {weight_genarray}
    {mom_dims}
    {mom_axes}
    mom_axes_reduced : int or sequence of int
        Location(s) of moment dimensions in ``data_reduced``.  This option is only needed
        if ``data_reduced`` is passed in and is an array.  Defaults to ``mom_axes``, or last dimensions
        of ``data_reduced``.
    {mom_params}
    {rep_dim}
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
        Resampled Central moments array. ``out.shape = (...,shape[axis-1],
        shape[axis+1], ..., shape[axis], mom0, ...)`` where ``shape =
        x.shape``. That is, the resampled dimension is moved to the end, just
        before the moment dimensions.

    Notes
    -----
    {vals_resample_note}
    """
    weight = 1.0 if weight is None else weight
    dtype = select_dtype(x, out=out, dtype=dtype)
    if data_reduced is None:
        from cmomy.reduction import reduce_vals

        data_reduced = reduce_vals(
            x,
            *y,
            mom=mom,
            mom_params=mom_params,
            weight=weight,
            axis=axis,
            parallel=parallel,
            dtype=dtype,
            dim=dim,
            mom_dims=mom_dims,
            keep_attrs=keep_attrs,
            apply_ufunc_kwargs=apply_ufunc_kwargs,
            axes_to_end=True,
        )
        mom_axes_reduced = None
    elif not is_xarray(data_reduced):
        data_reduced = asarray_maybe_recast(data_reduced, dtype=dtype, recast=False)

    if is_xarray_typevar["DataT"].check(x):
        prep, mom = PrepareValsXArray.factory_mom(
            mom=mom, mom_params=mom_params, dims=mom_dims, recast=False
        )
        dim, input_core_dims, xargs = prep.values_for_reduction(
            x,
            weight,
            *y,
            axis=axis,
            dim=dim,
            dtype=dtype,
            narrays=prep.mom_params.ndim + 1,
        )

        if is_xarray(data_reduced):
            mom_axes_reduced = None

        out, mom_params_axes = prep.optional_out_from_values(
            out,
            *xargs,
            target=x,
            dim=dim,
            mom=mom,
            axis_new_size=x.sizes[dim],
            axes_to_end=axes_to_end,
            order=order,
            dtype=dtype,
            mom_axes=mom_axes,
            mom_params=prep.mom_params,
        )

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _jackknife_vals,
            *xargs,
            data_reduced,
            input_core_dims=[*input_core_dims, prep.mom_params.dims],  # type: ignore[has-type]
            output_core_dims=[prep.mom_params.core_dims(dim)],
            kwargs={
                "mom": mom,
                "prep": prep.prepare_array,
                "axis_neg": -1,
                "mom_axes_reduced": mom_axes_reduced,
                "out": out,
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
                output_sizes=dict(
                    zip(prep.mom_params.dims, mom_to_mom_shape(mom), strict=True)
                ),
                output_dtypes=dtype if dtype is not None else np.float64,  # type: ignore[redundant-expr]
            ),
        )

        if not axes_to_end:
            xout = transpose_like(
                xout,
                template=x,
                append=prep.mom_params.dims,
                mom_params_axes=mom_params_axes,
            )
        elif is_dataset(x):
            xout = xout.transpose(
                ..., dim, *prep.mom_params.dims, missing_dims="ignore"
            )

        if rep_dim is not None:
            return xout.rename({dim: rep_dim})
        return xout

    # Numpy
    assert is_ndarray(data_reduced)  # noqa: S101
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

    return _jackknife_vals(
        *args,
        data_reduced,
        mom=mom,
        prep=prep,
        mom_axes_reduced=mom_axes_reduced,
        axis_neg=axis_neg,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        parallel=parallel,
        fastpath=True,
    )


def _jackknife_vals(
    # x, weight, *y, data_reduced
    *args: NDArrayAny,
    mom: MomentsStrict,
    prep: PrepareValsArray,
    mom_axes_reduced: MomAxes | None,
    axis_neg: int,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrderCF,
    parallel: bool | None,
    fastpath: bool,
) -> NDArrayAny:
    args, data_reduced = args[:-1], args[-1]
    dtype = select_dtype(args[0], out=out, dtype=dtype, fastpath=fastpath)

    raise_if_dataset(data_reduced, "Passed Dataset for reduce_data in array context.")

    mom_axes_reduced = (
        prep.mom_params.axes_last
        if mom_axes_reduced is None
        else MomParamsArray.factory(
            ndim=prep.mom_params.ndim, axes=mom_axes_reduced
        ).axes
    )

    raise_if_wrong_value(
        tuple(data_reduced.shape[k] for k in mom_axes_reduced),
        mom_to_mom_shape(mom),
        "Wrong moment shape of data_reduced.",
    )

    if out is None and order is not None:
        out, axis_sample_out = prep.out_from_values(
            out,
            val_shape=prep.get_val_shape(*args),
            mom=mom,
            axis_neg=axis_neg,
            axis_new_size=None,
            dtype=dtype,
            order=order,
        )
    else:
        axis_sample_out = prep.get_axis_sample_out(
            axis_neg=axis_neg,
            axis=None,
            axis_new_size=args[0].shape[axis_neg],
            out_ndim=len(prep.get_val_shape(*args)) + prep.mom_params.ndim,
        )

    axes: AxesGUFunc = [
        # data_reduced
        mom_axes_reduced,
        # x, w, *y
        *get_axes_from_values(*args, axis_neg=axis_neg),
        # out
        (axis_sample_out, *prep.mom_params.axes),
    ]

    return factory_jackknife_vals(
        mom_ndim=prep.mom_params.ndim,
        parallel=parallel_heuristic(parallel, size=args[0].size * prep.mom_params.ndim),
    )(
        data_reduced,
        *args,
        out=out,
        axes=axes,
        dtype=dtype,
        casting=casting,
        order=order,
    )

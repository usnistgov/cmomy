"""resample and reduce"""

from __future__ import annotations

# if TYPE_CHECKING:
from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from cmomy.core.array_utils import (
    asarray_maybe_recast,
    axes_data_reduction,
    get_axes_from_values,
    select_dtype,
)
from cmomy.core.docstrings import docfiller
from cmomy.core.missing import MISSING
from cmomy.core.prepare import (
    prepare_data_for_reduction,
    prepare_out_from_values,
    prepare_values_for_reduction,
    xprepare_out_for_resample_data,
    xprepare_out_for_resample_vals,
    xprepare_values_for_reduction,
)
from cmomy.core.utils import (
    mom_to_mom_shape,
)
from cmomy.core.validate import (
    is_dataarray,
    is_ndarray,
    is_xarray,
    raise_if_wrong_value,
    validate_mom_and_mom_ndim,
    validate_mom_dims,
    validate_mom_dims_and_mom_ndim,
    validate_mom_ndim,
)
from cmomy.core.xr_utils import (
    factory_apply_ufunc_kwargs,
    raise_if_dataset,
    select_axis_dim,
)
from cmomy.factory import (
    factory_jackknife_data,
    factory_jackknife_vals,
    factory_resample_data,
    factory_resample_vals,
    parallel_heuristic,
)

from .sampler import factory_sampler

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from cmomy.core.typing import (
        ApplyUFuncKwargs,
        ArrayLikeArg,
        ArrayOrder,
        ArrayOrderCF,
        AxesGUFunc,
        AxisReduce,
        Casting,
        DataT,
        DimsReduce,
        DTypeLikeArg,
        FloatT,
        JackknifeDataKwargs,
        JackknifeValsKwargs,
        KeepAttrs,
        MissingType,
        Mom_NDim,
        MomDims,
        Moments,
        MomentsStrict,
        NDArrayAny,
        ResampleDataKwargs,
        ResampleValsKwargs,
        Sampler,
    )
    from cmomy.core.typing_compat import Unpack


# * Resample data
# ** overloads
@overload
def resample_data(
    data: DataT,
    *,
    sampler: Sampler,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ResampleDataKwargs],
) -> DataT: ...
# array no out or dtype
@overload
def resample_data(
    data: ArrayLikeArg[FloatT],
    *,
    sampler: Sampler,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[ResampleDataKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def resample_data(
    data: ArrayLike,
    *,
    sampler: Sampler,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ResampleDataKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def resample_data(
    data: ArrayLike,
    *,
    sampler: Sampler,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[ResampleDataKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def resample_data(
    data: ArrayLike,
    *,
    sampler: Sampler,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ResampleDataKwargs],
) -> NDArrayAny: ...


# ** Public api
@docfiller.decorate  # type: ignore[arg-type,unused-ignore]
def resample_data(
    data: ArrayLike | DataT,
    *,
    sampler: Sampler,
    mom_ndim: Mom_NDim | None = None,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    rep_dim: str = "rep",
    move_axis_to_end: bool = False,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrder = None,
    parallel: bool | None = None,
    keep_attrs: KeepAttrs = None,
    # dask specific...
    mom_dims: MomDims | None = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Resample and reduce data.

    Parameters
    ----------
    {data_numpy_or_dataarray_or_dataset}
    {sampler}
    {mom_ndim_data}
    {axis}
    {dim}
    {rep_dim}
    {move_axis_to_end}
    {out}
    {dtype}
    {casting}
    {order}
    {parallel}
    {keep_attrs}
    {mom_dims_data}
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
        mom_dims=mom_dims,
        rep_dim=rep_dim,
        parallel=parallel,
    )

    if is_xarray(data):
        mom_dims, mom_ndim = validate_mom_dims_and_mom_ndim(
            mom_dims, mom_ndim, data, mom_ndim_default=1
        )
        axis, dim = select_axis_dim(data, axis=axis, dim=dim, mom_dims=mom_dims)

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _resample_data,
            data,
            sampler.freq,
            input_core_dims=[[dim, *mom_dims], [rep_dim, dim]],  # type: ignore[misc]
            output_core_dims=[[rep_dim, *mom_dims]],  # type: ignore[misc]
            kwargs={
                "mom_ndim": mom_ndim,
                "axis": -(mom_ndim + 1),
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
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_sizes={rep_dim: sampler.nrep},
                output_dtypes=dtype or np.float64,
            ),
        )

        if not move_axis_to_end and is_dataarray(data):
            dims_order = (*data.dims[:axis], rep_dim, *data.dims[axis + 1 :])  # type: ignore[union-attr, misc,index,operator]
            xout = xout.transpose(*dims_order)
        return xout

    # Numpy
    mom_ndim = validate_mom_ndim(mom_ndim, mom_ndim_default=1)
    axis, data = prepare_data_for_reduction(
        data,
        axis=axis,
        mom_ndim=mom_ndim,
        dtype=None,
        recast=False,
        move_axis_to_end=move_axis_to_end,
    )
    freq = sampler.freq
    assert is_ndarray(freq)  # noqa: S101

    return _resample_data(
        data,
        freq,
        mom_ndim=mom_ndim,
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
    mom_ndim: Mom_NDim,
    axis: int,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrder,
    parallel: bool | None,
    fastpath: bool,
) -> NDArrayAny:
    if not fastpath:
        dtype = select_dtype(data, out=out, dtype=dtype)

    # include inner core dimensions for freq
    axes = [
        (-2, -1),
        *axes_data_reduction(mom_ndim=mom_ndim, axis=axis, out_has_axis=True),
    ]

    return factory_resample_data(
        mom_ndim=mom_ndim,
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
def resample_vals(
    x: DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    weight: ArrayLike | xr.DataArray | DataT | None = ...,
    sampler: Sampler,
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
    sampler: Sampler,
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
    sampler: Sampler,
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
    sampler: Sampler,
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
    sampler: Sampler,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ResampleValsKwargs],
) -> NDArrayAny: ...


# ** public api
@docfiller.decorate  # type: ignore[arg-type,unused-ignore]
def resample_vals(  # noqa: PLR0913
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    mom: Moments,
    sampler: Sampler,
    weight: ArrayLike | xr.DataArray | DataT | None = None,
    axis: AxisReduce | MissingType = MISSING,
    move_axis_to_end: bool = True,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderCF = None,
    parallel: bool | None = None,
    # xarray specific
    dim: DimsReduce | MissingType = MISSING,
    rep_dim: str = "rep",
    mom_dims: MomDims | None = None,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Resample and reduce values.

    Parameters
    ----------
    {x_genarray}
    {y_genarray}
    {mom}
    {sampler}
    {weight_genarray}
    {axis}
    {move_axis_to_end}
    {out}
    {dtype}
    {casting}
    {order_cf}
    {parallel}
    {dim}
    {rep_dim}
    {mom_dims}
    {keep_attrs}
    {apply_ufunc_kwargs}

    Returns
    -------
    out : ndarray or DataArray
        Resampled Central moments array. ``out.shape = (...,shape[axis-1], nrep, shape[axis+1], ...)``
        where ``shape = x.shape``. and ``nrep = sampler.nrep``.  This can be overridden by setting `move_axis_to_end`.

    Notes
    -----
    {vals_resample_note}

    See Also
    --------
    .resample.factory_sampler
    """
    mom, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
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

    if is_xarray(x):
        dim, input_core_dims, xargs = xprepare_values_for_reduction(
            x,
            weight,
            *y,
            axis=axis,
            dim=dim,
            dtype=dtype,
            recast=False,
            narrays=mom_ndim + 1,
        )

        mom_dims = validate_mom_dims(
            mom_dims=mom_dims,
            mom_ndim=mom_ndim,
        )

        def _func(*args: NDArrayAny, **kwargs: Any) -> NDArrayAny:
            x, w, *y, _freq = args
            return _resample_vals(x, w, *y, freq=_freq, **kwargs)  # type: ignore[has-type]

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _func,
            *xargs,
            sampler.freq,
            input_core_dims=[*input_core_dims, [rep_dim, dim]],  # type: ignore[has-type]
            output_core_dims=[[rep_dim, *mom_dims]],  # type: ignore[misc]
            kwargs={
                "mom": mom,
                "mom_ndim": mom_ndim,
                "axis_neg": -1,
                "out": xprepare_out_for_resample_vals(
                    target=x,
                    out=out,
                    dim=dim,
                    mom_ndim=mom_ndim,
                    move_axis_to_end=move_axis_to_end,
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
                output_sizes={
                    rep_dim: sampler.nrep,
                    **dict(zip(mom_dims, mom_to_mom_shape(mom))),
                },
                output_dtypes=dtype or np.float64,
            ),
        )

        if not move_axis_to_end and is_dataarray(x):
            dims_order = [  # type: ignore[misc]
                *(d if d != dim else rep_dim for d in x.dims),  # type: ignore[union-attr]
                *mom_dims,
            ]
            xout = xout.transpose(..., *dims_order)  # pyright: ignore[reportUnknownArgumentType]

        return xout

    # numpy
    axis_neg, args = prepare_values_for_reduction(
        x,
        weight,
        *y,
        axis=axis,
        dtype=dtype,
        recast=False,
        narrays=mom_ndim + 1,
        move_axis_to_end=move_axis_to_end,
    )

    return _resample_vals(
        *args,
        freq=sampler.freq,
        mom=mom,
        mom_ndim=mom_ndim,
        axis_neg=axis_neg,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        parallel=parallel,
        fastpath=True,
    )


def _resample_vals(
    # x, w, *y
    *args: NDArrayAny,
    freq: NDArrayAny,
    mom: MomentsStrict,
    mom_ndim: Mom_NDim,
    axis_neg: int,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrderCF,
    parallel: bool | None,
    fastpath: bool,
) -> NDArrayAny:
    if not fastpath:
        dtype = select_dtype(args[0], out=out, dtype=dtype)

    out = prepare_out_from_values(
        out,
        *args,
        mom=mom,
        axis_neg=axis_neg,
        axis_new_size=freq.shape[0],
        dtype=dtype,
        order=order,
    )

    axes: AxesGUFunc = [
        # out
        (axis_neg - mom_ndim, *range(-mom_ndim, 0)),
        # freq
        (-2, -1),
        # x, weight, *y
        *get_axes_from_values(*args, axis_neg=axis_neg),
    ]

    factory_resample_vals(
        mom_ndim=mom_ndim,
        parallel=parallel_heuristic(parallel, size=args[0].size * mom_ndim),
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
def jackknife_data(
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


@docfiller.decorate  # type: ignore[arg-type,unused-ignore]
def jackknife_data(
    data: ArrayLike | DataT,
    data_reduced: ArrayLike | DataT | None = None,
    *,
    mom_ndim: Mom_NDim | None = None,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    rep_dim: str | None = "rep",
    move_axis_to_end: bool = False,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrder = None,
    parallel: bool | None = None,
    keep_attrs: KeepAttrs = None,
    # dask specific...
    mom_dims: MomDims | None = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Perform jackknife resample and moments data.

    This uses moments addition/subtraction to speed up jackknife resampling.

    Parameters
    ----------
    {data_numpy_or_dataarray_or_dataset}
    {mom_ndim_data}
    {axis}
    {dim}
    data_reduced : array-like or DataArray, optional
        ``data`` reduced along ``axis`` or ``dim``.  This will be calculated using
        :func:`.reduce_data` if not passed.
    rep_dim : str, optional
        Optionally output ``dim`` to ``rep_dim``.
    {move_axis_to_end}
    {out}
    {dtype}
    {casting}
    {order}
    {parallel}
    {keep_attrs}
    {mom_dims_data}
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

    if data_reduced is None:
        from cmomy.reduction import reduce_data

        data_reduced = reduce_data(
            data=data,
            mom_ndim=mom_ndim,
            dim=dim,
            axis=axis,
            parallel=parallel,
            keep_attrs=keep_attrs,
            dtype=dtype,
            casting=casting,
            order=order,
            mom_dims=mom_dims,
            apply_ufunc_kwargs=apply_ufunc_kwargs,
            use_reduce=False,
        )
    elif not is_xarray(data_reduced):
        data_reduced = asarray_maybe_recast(data_reduced, dtype=dtype, recast=False)

    if is_xarray(data):
        mom_dims, mom_ndim = validate_mom_dims_and_mom_ndim(
            mom_dims, mom_ndim, data, mom_ndim_default=1
        )
        axis, dim = select_axis_dim(data, axis=axis, dim=dim, mom_dims=mom_dims)
        core_dims = [dim, *mom_dims]  # type: ignore[misc]

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _jackknife_data,
            data,
            data_reduced,
            input_core_dims=[core_dims, core_dims[1:]],
            output_core_dims=[core_dims],
            kwargs={
                "mom_ndim": mom_ndim,
                "axis": -(mom_ndim + 1),
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
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_dtypes=dtype or np.float64,
            ),
        )

        if not move_axis_to_end and is_dataarray(data):
            xout = xout.transpose(*data.dims)
        if rep_dim is not None:
            xout = xout.rename({dim: rep_dim})
        return xout

    # numpy
    mom_ndim = validate_mom_ndim(mom_ndim)
    axis, data = prepare_data_for_reduction(
        data,
        axis=axis,
        mom_ndim=mom_ndim,
        dtype=dtype,
        recast=False,
        move_axis_to_end=move_axis_to_end,
    )

    assert is_ndarray(data_reduced)  # noqa: S101

    return _jackknife_data(
        data,
        data_reduced,
        mom_ndim=mom_ndim,
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
    mom_ndim: Mom_NDim,
    axis: int,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrder,
    parallel: bool | None,
    fastpath: bool,
) -> NDArrayAny:
    if not fastpath:
        dtype = select_dtype(data, out=out, dtype=dtype)

    axes_data, axes_mom = axes_data_reduction(
        mom_ndim=mom_ndim, axis=axis, out_has_axis=True
    )
    # add axes for data_reduced
    axes = [axes_data[1:], axes_data, axes_mom]

    return factory_jackknife_data(
        mom_ndim=mom_ndim,
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
def jackknife_vals(
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


@docfiller.decorate
def jackknife_vals(  # noqa: PLR0913
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    mom: Moments,
    data_reduced: ArrayLike | DataT | None = None,
    weight: ArrayLike | xr.DataArray | DataT | None = None,
    axis: AxisReduce | MissingType = MISSING,
    move_axis_to_end: bool = True,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrder = None,
    parallel: bool | None = None,
    # xarray specific
    dim: DimsReduce | MissingType = MISSING,
    rep_dim: str | None = "rep",
    mom_dims: MomDims | None = None,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> xr.Dataset | xr.DataArray | NDArrayAny:
    """
    Jackknife by value.

    Parameters
    ----------
    {x_genarray}
    {y_genarray}
    {mom}
    data_reduced : array-like or DataArray, optional
        ``data`` reduced along ``axis`` or ``dim``.  This will be calculated using
        :func:`.reduce_vals` if not passed.  Same type restrictions as ``weight``.
    {weight_genarray}
    {axis}
    {move_axis_to_end}
    {order}
    {parallel}
    {dtype}
    {out}
    {dim}
    {rep_dim}
    {mom_dims}
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
    mom, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
    weight = 1.0 if weight is None else weight
    dtype = select_dtype(x, out=out, dtype=dtype)
    if data_reduced is None:
        from cmomy.reduction import reduce_vals

        data_reduced = reduce_vals(  # type: ignore[type-var, misc, unused-ignore]
            x,  # pyright: ignore[reportArgumentType]
            *y,
            mom=mom,
            weight=weight,
            axis=axis,
            parallel=parallel,
            dtype=dtype,
            dim=dim,
            mom_dims=mom_dims,
            keep_attrs=keep_attrs,
            apply_ufunc_kwargs=apply_ufunc_kwargs,
        )
    elif not is_xarray(data_reduced):
        data_reduced = asarray_maybe_recast(data_reduced, dtype=dtype, recast=False)

    if is_xarray(x):
        dim, input_core_dims, xargs = xprepare_values_for_reduction(
            x,
            weight,
            *y,
            axis=axis,
            dim=dim,
            dtype=dtype,
            recast=False,
            narrays=mom_ndim + 1,
        )
        mom_dims = validate_mom_dims(
            mom_dims=mom_dims, mom_ndim=mom_ndim, out=data_reduced
        )
        input_core_dims = [
            # x, weight, *y,
            *input_core_dims,  # type: ignore[has-type]
            # data_reduced
            mom_dims,
        ]

        def _func(*args: NDArrayAny, **kwargs: Any) -> NDArrayAny:
            x, weight, *y, data_reduced = args
            return _jackknife_vals(x, weight, *y, data_reduced=data_reduced, **kwargs)  # type: ignore[has-type]

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _func,
            *xargs,
            data_reduced,
            input_core_dims=input_core_dims,
            output_core_dims=[(dim, *mom_dims)],  # type: ignore[misc]
            kwargs={
                "mom": mom,
                "mom_ndim": mom_ndim,
                "axis_neg": -1,
                "out": xprepare_out_for_resample_vals(
                    target=x,
                    out=out,
                    dim=dim,
                    mom_ndim=mom_ndim,
                    move_axis_to_end=move_axis_to_end,
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
                output_sizes=dict(zip(mom_dims, mom_to_mom_shape(mom))),
                output_dtypes=dtype or np.float64,
            ),
        )

        if not move_axis_to_end and is_dataarray(x):
            xout = xout.transpose(..., *x.dims, *mom_dims)  # pyright: ignore[reportUnknownArgumentType]
        if rep_dim is not None:
            xout = xout.rename({dim: rep_dim})
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
        move_axis_to_end=move_axis_to_end,
    )

    assert is_ndarray(data_reduced)  # noqa: S101

    return _jackknife_vals(
        *args,
        data_reduced=data_reduced,  # pyright: ignore[reportUnknownArgumentType]
        mom=mom,
        mom_ndim=mom_ndim,
        axis_neg=axis_neg,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        parallel=parallel,
        fastpath=True,
    )


def _jackknife_vals(
    x: NDArrayAny,
    weight: NDArrayAny,
    *y: NDArrayAny,
    data_reduced: NDArrayAny,
    mom: MomentsStrict,
    mom_ndim: Mom_NDim,
    axis_neg: int,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrder,
    parallel: bool | None,
    fastpath: bool,
) -> NDArrayAny:
    args = [x, weight, *y]
    if not fastpath:
        dtype = select_dtype(x, out=out, dtype=dtype)

    raise_if_dataset(data_reduced, "Passed Dataset for reduce_data in array context.")
    raise_if_wrong_value(
        data_reduced.shape[-mom_ndim:],
        mom_to_mom_shape(mom),
        "Wrong moment shape of data_reduced.",
    )

    axes: AxesGUFunc = [
        # data_reduced
        tuple(range(-mom_ndim, 0)),
        # x, w, *y
        *get_axes_from_values(*args, axis_neg=axis_neg),
        # out
        (axis_neg - mom_ndim, *range(-mom_ndim, 0)),
    ]

    return factory_jackknife_vals(
        mom_ndim=mom_ndim,
        parallel=parallel_heuristic(parallel, size=args[0].size * mom_ndim),
    )(
        data_reduced,
        *args,
        out=out,
        axes=axes,
        dtype=dtype,
        casting=casting,
        order=order,
    )

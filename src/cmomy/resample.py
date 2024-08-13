"""
Routine to perform resampling (:mod:`cmomy.resample`)
=====================================================
"""

from __future__ import annotations

from itertools import starmap

# if TYPE_CHECKING:
from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ._lib.factory import (
    factory_jackknife_data,
    factory_jackknife_vals,
    factory_resample_data,
    factory_resample_vals,
)
from .core.array_utils import (
    axes_data_reduction,
    get_axes_from_values,
    normalize_axis_index,
    select_dtype,
)
from .core.docstrings import docfiller
from .core.missing import MISSING
from .core.prepare import (
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
    validate_mom_and_mom_ndim,
    validate_mom_dims,
    validate_mom_ndim,
)
from .core.xr_utils import (
    get_apply_ufunc_kwargs,
    select_axis_dim,
)
from .random import validate_rng

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from .core.typing import (
        ApplyUFuncKwargs,
        ArrayLikeArg,
        AxesGUFunc,
        AxisReduce,
        DimsReduce,
        DTypeLikeArg,
        FloatT,
        IntDTypeT,
        KeepAttrs,
        MissingCoreDimOptions,
        MissingType,
        Mom_NDim,
        MomDims,
        Moments,
        NDArrayAny,
        NDArrayInt,
    )


# * Resampling utilities ------------------------------------------------------
@docfiller.decorate
def freq_to_indices(
    freq: NDArray[IntDTypeT],
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
) -> NDArray[IntDTypeT]:
    """
    Convert a frequency array to indices array.

    This creates an "indices" array that is compatible with "freq" array.
    Note that by default, the indices for a single sample (along output[k, :])
    are randomly shuffled.  If you pass `shuffle=False`, then the output will
    be something like [[0,0,..., 1,1,..., 2,2, ...]].

    Parameters
    ----------
    {freq}
    shuffle :
        If ``True`` (default), shuffle values for each row.
    {rng}

    Returns
    -------
    ndarray :
        Indices array of shape ``(nrep, nsamp)`` where ``nsamp = freq[k,
        :].sum()`` where `k` is any row.
    """
    indices_all: list[NDArrayAny] = []

    # validate freq -> indices
    nsamps = freq.sum(-1)  # pyright: ignore[reportUnknownMemberType]
    if any(nsamps[0] != nsamps):
        msg = "Inconsistent number of samples from freq array"
        raise ValueError(msg)

    for f in freq:
        indices = np.concatenate(list(starmap(np.repeat, enumerate(f))))  # pyright: ignore[reportUnknownArgumentType]
        indices_all.append(indices)

    out = np.array(indices_all, dtype=freq.dtype)

    if shuffle:
        rng = validate_rng(rng)
        rng.shuffle(out, axis=1)

    return out


def indices_to_freq(
    indices: NDArray[IntDTypeT], ndat: int | None = None
) -> NDArray[IntDTypeT]:
    """
    Convert indices to frequency array.

    It is assumed that ``indices.shape == (nrep, nsamp)`` with ``nsamp == ndat``.
    For cases that ``nsamp != ndat``, pass in ``ndat``.
    """
    from ._lib.utils import (
        randsamp_indices_to_freq,  # pyright: ignore[reportUnknownVariableType]
    )

    ndat = indices.shape[1] if ndat is None else ndat
    freq = np.zeros((indices.shape[0], ndat), dtype=indices.dtype)

    randsamp_indices_to_freq(indices, freq)

    return freq


@docfiller.decorate
def random_indices(
    nrep: int,
    ndat: int,
    nsamp: int | None = None,
    rng: np.random.Generator | None = None,
    replace: bool = True,
) -> NDArrayAny:
    """
    Create indices for random resampling (bootstrapping).

    Parameters
    ----------
    {nrep}
    {ndat}
    {nsamp}
    {rng}
    replace :
        Whether to allow replacement.

    Returns
    -------
    indices : ndarray
        Index array of integers of shape ``(nrep, nsamp)``.
    """
    nsamp = ndat if nsamp is None else nsamp
    return validate_rng(rng).choice(ndat, size=(nrep, nsamp), replace=replace)


@docfiller.inherit(random_indices)
def random_freq(
    nrep: int,
    ndat: int,
    nsamp: int | None = None,
    rng: np.random.Generator | None = None,
    replace: bool = True,
) -> NDArrayAny:
    """
    Create frequencies for random resampling (bootstrapping).

    Returns
    -------
    freq : ndarray
        Frequency array. ``freq[rep, k]`` is the number of times to sample from the `k`th
        observation for replicate `rep`.

    See Also
    --------
    random_indices
    """
    return indices_to_freq(
        indices=random_indices(
            nrep=nrep, ndat=ndat, nsamp=nsamp, rng=rng, replace=replace
        ),
        ndat=ndat,
    )


@docfiller.decorate
def select_ndat(
    data: xr.Dataset | xr.DataArray | NDArrayAny,
    *,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: Mom_NDim | None = None,
) -> int:
    """
    Determine ndat from array.

    Parameters
    ----------
    data : ndarray or DataArray or Dataset
    {axis}
    {dim}
    mom_ndim : int, optional
        If specified, then treat ``data`` as a moments array, and wrap negative
        values for ``axis`` relative to value dimensions only.

    Returns
    -------
    int
        size of ``data`` along specified ``axis`` or ``dim``

    Examples
    --------
    >>> data = np.zeros((2, 3, 4))
    >>> select_ndat(data, axis=1)
    3
    >>> select_ndat(data, axis=-1, mom_ndim=2)
    2


    >>> xdata = xr.DataArray(data, dims=["x", "y", "mom"])
    >>> select_ndat(xdata, dim="y")
    3
    >>> select_ndat(xdata, dim="mom", mom_ndim=1)
    Traceback (most recent call last):
    ...
    ValueError: Cannot select moment dimension. axis=2, dim='mom'.
    """
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        axis, dim = select_axis_dim(data, axis=axis, dim=dim, mom_ndim=mom_ndim)
        return data.sizes[dim]

    if isinstance(axis, int):
        axis = normalize_axis_index(axis, data.ndim, mom_ndim=mom_ndim)
        return data.shape[axis]
    msg = "Must specify integer axis for array input."
    raise TypeError(msg)


# * General frequency table generation.
def _validate_resample_array(
    x: ArrayLike | xr.DataArray | xr.Dataset,
    ndat: int,
    nrep: int | None,
    is_freq: bool,
    check: bool = True,
    dtype: DTypeLike = np.int64,
) -> NDArrayAny:
    x = np.asarray(x, dtype=dtype)
    if check:
        name = "freq" if is_freq else "indices"
        if x.ndim != 2:
            msg = f"{name}.ndim={x.ndim} != 2"
            raise ValueError(msg)

        if nrep is not None and x.shape[0] != nrep:
            msg = f"{name}.shape[0]={x.shape[0]} != {nrep}"
            raise ValueError(msg)

        if is_freq:
            if x.shape[1] != ndat:
                msg = f"{name} has wrong ndat"
                raise ValueError(msg)

        else:
            # only restriction is that values in [0, ndat)
            min_, max_ = x.min(), x.max()  # pyright: ignore[reportUnknownMemberType]
            if min_ < 0 or max_ >= ndat:
                msg = f"Indices range [{min_}, {max_}) outside [0, {ndat - 1})"
                raise ValueError(msg)
    return x


def _select_nrep(
    freq: ArrayLike | xr.DataArray | xr.Dataset | None,
    nrep: int | None,
    rep_dim: str,
) -> int:
    if freq is not None:
        return (
            freq.sizes[rep_dim]
            if isinstance(freq, (xr.DataArray, xr.Dataset))
            else np.shape(freq)[0]
        )

    if nrep is not None:
        return nrep
    msg = "Must specify either freq or nrep"
    raise ValueError(msg)


@docfiller.decorate
def randsamp_freq(
    *,
    ndat: int | None = None,
    nrep: int | None = None,
    nsamp: int | None = None,
    indices: ArrayLike | None = None,
    freq: ArrayLike | None = None,
    data: xr.Dataset | xr.DataArray | NDArrayAny | None = None,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: Mom_NDim | None = None,
    check: bool = False,
    rng: np.random.Generator | None = None,
    dtype: DTypeLike = np.int64,
) -> NDArrayInt:
    """
    Convenience function to create frequency table for resampling.

    In order, the return will be one of ``freq``, frequencies from ``indices`` or
    new sample from :func:`random_freq`.

    Parameters
    ----------
    {ndat}
    {nrep}
    {nsamp}
    {freq}
    {indices}
    check : bool, default=False
        if `check` is `True`, then check `freq` and `indices` against `ndat` and `nrep`
    {rng}
    data : ndarray or DataArray or Dataset
    {axis}
    {dim}
    mom_ndim : int, optional
        If specified, then treat ``data`` as a moments array, and wrap negative
        values for ``axis`` relative to value dimensions only.
    jackknife : bool, default=False
        If ``True``, return jackknife resampling frequency array (see :func:`jackknife_freq`).
        Note that this overrides all other parameters.


    Notes
    -----
    If ``ndat`` is ``None``, attempt to set ``ndat`` using ``ndat =
    select_ndat(data, axis=axis, dim=dim, mom_ndim=mom_ndim)``. See
    :func:`select_ndat`.

    Returns
    -------
    freq : ndarray
        Frequency array.

    See Also
    --------
    random_freq
    indices_to_freq
    select_ndat

    Examples
    --------
    >>> import cmomy
    >>> rng = cmomy.random.default_rng(0)
    >>> randsamp_freq(ndat=3, nrep=5, rng=rng)
    array([[0, 2, 1],
           [3, 0, 0],
           [3, 0, 0],
           [0, 1, 2],
           [0, 2, 1]])

    Create from data and axis

    >>> data = np.zeros((2, 3, 5))
    >>> freq = randsamp_freq(data=data, axis=-1, mom_ndim=1, nrep=5, rng=rng)
    >>> freq
    array([[0, 2, 1],
           [1, 1, 1],
           [1, 0, 2],
           [0, 2, 1],
           [1, 0, 2]])


    This can also be used to convert from indices to freq array

    >>> indices = freq_to_indices(freq)
    >>> randsamp_freq(data=data, axis=-1, mom_ndim=1, indices=indices)
    array([[0, 2, 1],
           [1, 1, 1],
           [1, 0, 2],
           [0, 2, 1],
           [1, 0, 2]])

    """
    # short circuit the most likely scenario...
    if freq is not None and not check:
        return np.asarray(freq, dtype=dtype)

    if ndat is None:
        if data is None:
            msg = "Must pass either ndat or data"
            raise TypeError(msg)
        ndat = select_ndat(data=data, axis=axis, dim=dim, mom_ndim=mom_ndim)

    if freq is not None:
        freq = _validate_resample_array(
            freq,
            nrep=nrep,
            ndat=ndat,
            check=check,
            is_freq=True,
            dtype=dtype,
        )

    elif indices is not None:
        indices = _validate_resample_array(
            indices,
            nrep=nrep,
            ndat=ndat,
            check=check,
            is_freq=False,
        )

        freq = indices_to_freq(indices, ndat=ndat)

    elif nrep is not None:
        freq = random_freq(
            nrep=nrep, ndat=ndat, nsamp=nsamp, rng=validate_rng(rng), replace=True
        )

    else:
        msg = "must specify freq, indices, or nrep"
        raise ValueError(msg)

    return freq


# * Utilities
def _check_freq(freq: NDArrayAny, ndat: int) -> None:
    if freq.shape[1] != ndat:
        msg = f"{freq.shape[1]=} != {ndat=}"
        raise ValueError(msg)


# * Resample data
# ** overloads
@overload
def resample_data(
    data: xr.Dataset,
    *,
    mom_ndim: Mom_NDim,
    freq: ArrayLike | xr.DataArray | xr.Dataset | None = ...,
    nrep: int | None = ...,
    rng: np.random.Generator | None = ...,
    paired: bool = ...,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> xr.Dataset: ...
@overload
def resample_data(
    data: xr.DataArray,
    *,
    mom_ndim: Mom_NDim,
    freq: ArrayLike | xr.DataArray | None = ...,
    nrep: int | None = ...,
    rng: np.random.Generator | None = ...,
    paired: bool = ...,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> xr.DataArray: ...
# array no out or dtype
@overload
def resample_data(
    data: ArrayLikeArg[FloatT],
    *,
    mom_ndim: Mom_NDim,
    freq: ArrayLike | None = ...,
    nrep: int | None = ...,
    rng: np.random.Generator | None = ...,
    paired: bool = ...,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: None = ...,
    out: None = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArray[FloatT]: ...
# out
@overload
def resample_data(
    data: ArrayLike,
    *,
    mom_ndim: Mom_NDim,
    freq: ArrayLike | None = ...,
    nrep: int | None = ...,
    rng: np.random.Generator | None = ...,
    paired: bool = ...,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArray[FloatT],
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def resample_data(
    data: ArrayLike,
    *,
    mom_ndim: Mom_NDim,
    freq: ArrayLike | None = ...,
    nrep: int | None = ...,
    rng: np.random.Generator | None = ...,
    paired: bool = ...,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLikeArg[FloatT],
    out: None = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArray[FloatT]: ...
# fallback
@overload
def resample_data(
    data: ArrayLike,
    *,
    mom_ndim: Mom_NDim,
    freq: ArrayLike | None = ...,
    nrep: int | None = ...,
    rng: np.random.Generator | None = ...,
    paired: bool = ...,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArrayAny: ...


# TODO(wpk): Should out above be out: NDArrayAny | None = ... ?


# ** Public api
@docfiller.decorate
def resample_data(  # noqa: PLR0913
    data: ArrayLike | xr.DataArray | xr.Dataset,
    *,
    mom_ndim: Mom_NDim,
    freq: ArrayLike | xr.DataArray | xr.Dataset | None = None,
    nrep: int | None = None,
    rng: np.random.Generator | None = None,
    paired: bool = True,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    rep_dim: str = "rep",
    move_axis_to_end: bool = False,
    parallel: bool | None = None,
    dtype: DTypeLike = None,
    out: NDArrayAny | None = None,
    keep_attrs: KeepAttrs = None,
    # dask specific...
    mom_dims: MomDims | None = None,
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | xr.DataArray | xr.Dataset:
    """
    Resample data according to frequency table.

    Parameters
    ----------
    {data_numpy_or_dataarray_or_dataset}
    {mom_ndim}
    {freq}
    {nrep_optional}
    {rng}
    {paired}
    {axis}
    {dim}
    {rep_dim}
    {move_axis_to_end}
    {parallel}
    {dtype}
    {out}
    {keep_attrs}
    {mom_dims_data}
    {on_missing_core_dim}
    {apply_ufunc_kwargs}

    Returns
    -------
    out : ndarray or DataArray
        Resampled central moments. ``out.shape = (..., shape[axis-1], nrep, shape[axis+1], ...)``,
        where ``shape = data.shape`` and ``nrep = freq.shape[0]``.

    See Also
    --------
    random_freq
    randsamp_freq
    """
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        axis, dim = select_axis_dim(data, axis=axis, dim=dim, mom_ndim=mom_ndim)
        mom_dims = validate_mom_dims(mom_dims, mom_ndim, data)

        if isinstance(data, xr.Dataset) and freq is None and paired:
            freq = randsamp_freq(ndat=data.sizes[dim], nrep=nrep, rng=rng)

        apply_ufunc_kwargs = get_apply_ufunc_kwargs(
            apply_ufunc_kwargs,
            on_missing_core_dim=on_missing_core_dim,
            dask="parallelized",
            output_sizes={rep_dim: _select_nrep(freq, nrep, rep_dim)},
            # NOTE: for now just use np.float64 as the output dtype.
            # see https://github.com/pydata/xarray/issues/1699
            output_dtypes=select_dtype(data, out=out, dtype=dtype) or np.float64,
        )

        def _func(
            data: NDArrayAny, freq: NDArrayAny | None, **kwargs: Any
        ) -> NDArrayAny:
            return resample_data(data, freq=freq, **kwargs)

        xout: xr.Dataset | xr.DataArray = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _func,
            data,
            freq,
            input_core_dims=[[dim, *mom_dims], [rep_dim, dim]],
            output_core_dims=[[rep_dim, *mom_dims]],
            kwargs={
                "mom_ndim": mom_ndim,
                "parallel": parallel,
                "out": xprepare_out_for_resample_data(
                    out,
                    mom_ndim=mom_ndim,
                    axis=axis,
                    move_axis_to_end=move_axis_to_end,
                    data=data,
                ),
                "dtype": dtype,
                "axis": -1,
                "nrep": nrep,
                "rng": rng,
            },
            keep_attrs=keep_attrs,
            **apply_ufunc_kwargs,
        )

        if not move_axis_to_end and isinstance(data, xr.DataArray):
            dims_order = (*data.dims[:axis], rep_dim, *data.dims[axis + 1 :])
            xout = xout.transpose(*dims_order)
        return xout

    mom_ndim = validate_mom_ndim(mom_ndim)
    dtype = select_dtype(data, out=out, dtype=dtype)
    axis, data = prepare_data_for_reduction(
        data,
        axis=axis,
        mom_ndim=mom_ndim,
        dtype=dtype,
        move_axis_to_end=move_axis_to_end,
    )

    freq = randsamp_freq(
        data=data,
        axis=axis,
        mom_ndim=mom_ndim,
        freq=freq,  # type: ignore[arg-type]  # Should do better for this...
        rng=rng,
        nrep=nrep,
        dtype=dtype,
    )

    # include inner core dimensions for freq
    axes = [
        (-2, -1),
        *axes_data_reduction(mom_ndim=mom_ndim, axis=axis, out_has_axis=True),
    ]

    _check_freq(freq, data.shape[axis])

    return factory_resample_data(
        mom_ndim=mom_ndim,
        parallel=parallel,
        size=data.size,
    )(freq, data, out=out, axes=axes)


# * Resample vals
# ** overloads
@overload
def resample_vals(  # pyright: ignore[reportOverlappingOverload]
    x: xr.Dataset,
    *y: ArrayLike | xr.DataArray | xr.Dataset,
    mom: Moments,
    freq: ArrayLike | xr.DataArray | xr.Dataset | None = ...,
    nrep: int | None = ...,
    rng: np.random.Generator | None = ...,
    paired: bool = ...,
    weight: ArrayLike | xr.DataArray | xr.Dataset | None = ...,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> xr.Dataset: ...
@overload
def resample_vals(  # pyright: ignore[reportOverlappingOverload]
    x: xr.DataArray,
    *y: ArrayLike | xr.DataArray,
    mom: Moments,
    freq: ArrayLike | xr.DataArray | None = ...,
    nrep: int | None = ...,
    rng: np.random.Generator | None = ...,
    weight: ArrayLike | xr.DataArray | None = ...,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> xr.DataArray: ...
# array
@overload
def resample_vals(
    x: ArrayLikeArg[FloatT],
    *y: ArrayLike,
    mom: Moments,
    freq: ArrayLike | None = ...,
    nrep: int | None = ...,
    rng: np.random.Generator | None = ...,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: None = ...,
    out: None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArray[FloatT]: ...
# out
@overload
def resample_vals(
    x: ArrayLike,
    *y: ArrayLike,
    mom: Moments,
    freq: ArrayLike | None = ...,
    nrep: int | None = ...,
    rng: np.random.Generator | None = ...,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArray[FloatT],
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def resample_vals(
    x: ArrayLike,
    *y: ArrayLike,
    mom: Moments,
    freq: ArrayLike | None = ...,
    nrep: int | None = ...,
    rng: np.random.Generator | None = ...,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLikeArg[FloatT],
    out: None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArray[FloatT]: ...
# fallback
@overload
def resample_vals(
    x: ArrayLike,
    *y: ArrayLike,
    mom: Moments,
    freq: ArrayLike | None = ...,
    nrep: int | None = ...,
    rng: np.random.Generator | None = ...,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArrayAny: ...


# ** public api
@docfiller.decorate
def resample_vals(  # pyright: ignore[reportOverlappingOverload]  # noqa: PLR0914, PLR0913
    x: ArrayLike | xr.DataArray | xr.Dataset,
    *y: ArrayLike | xr.DataArray | xr.Dataset,
    mom: Moments,
    freq: ArrayLike | xr.DataArray | xr.Dataset | None = None,
    nrep: int | None = None,
    rng: np.random.Generator | None = None,
    paired: bool = True,
    weight: ArrayLike | xr.DataArray | xr.Dataset | None = None,
    axis: AxisReduce | MissingType = MISSING,
    move_axis_to_end: bool = True,
    parallel: bool | None = None,
    dtype: DTypeLike = None,
    out: NDArrayAny | None = None,
    # xarray specific
    dim: DimsReduce | MissingType = MISSING,
    rep_dim: str = "rep",
    mom_dims: MomDims | None = None,
    keep_attrs: KeepAttrs = None,
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | xr.DataArray | xr.Dataset:
    """
    Resample data according to frequency table.

    Parameters
    ----------
    x : ndarray or DataArray
        Value to analyze
    *y:  array-like or DataArray, optional
        Second value needed if len(mom)==2.
    {freq}
    {nrep_optional}
    {rng}
    {paired}
    {mom}
    {weight}
    {axis}
    {move_axis_to_end}
    {parallel}
    {dtype}
    {out}
    {dim}
    {rep_dim}
    {mom_dims}
    {keep_attrs}
    {on_missing_core_dim}
    {apply_ufunc_kwargs}

    Returns
    -------
    out : ndarray or DataArray
        Resampled Central moments array. ``out.shape = (...,shape[axis-1], nrep, shape[axis+1], ...)``
        where ``shape = x.shape``. and ``nrep = freq.shape[0]``.  This can be overridden by setting `move_axis_to_end`.

    Notes
    -----
    {vals_resample_note}

    See Also
    --------
    random_freq
    randsamp_freq
    """
    mom_validated, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
    weight = 1.0 if weight is None else weight

    if isinstance(x, (xr.DataArray, xr.Dataset)):
        dtype = select_dtype(x, out=out, dtype=dtype)
        dim, input_core_dims, xargs = xprepare_values_for_reduction(
            x,
            weight,
            *y,
            axis=axis,
            dim=dim,
            dtype=dtype,
            narrays=mom_ndim + 1,
        )

        out = xprepare_out_for_resample_vals(
            target=x,
            out=out,
            dim=dim,
            mom_ndim=mom_ndim,
            move_axis_to_end=move_axis_to_end,
        )
        mom_dims = validate_mom_dims(
            mom_dims=mom_dims,
            mom_ndim=mom_ndim,
        )

        if isinstance(x, xr.Dataset) and freq is None and paired:
            freq = randsamp_freq(ndat=x.sizes[dim], nrep=nrep, rng=rng)

        apply_ufunc_kwargs = get_apply_ufunc_kwargs(
            apply_ufunc_kwargs,
            on_missing_core_dim=on_missing_core_dim,
            dask="parallelized",
            output_sizes={
                rep_dim: _select_nrep(freq, nrep, rep_dim),
                **dict(zip(mom_dims, mom_to_mom_shape(mom))),
            },
            # NOTE: for now just use np.float64 as the output dtype.
            # see https://github.com/pydata/xarray/issues/1699
            output_dtypes=np.float64,
        )

        def _func(*args: NDArrayAny, **kwargs: Any) -> NDArrayAny:
            x, weight, *y, freq = args
            return resample_vals(x, *y, weight=weight, freq=freq, **kwargs)

        xout: xr.DataArray | xr.Dataset = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _func,
            *xargs,
            freq,
            input_core_dims=[*input_core_dims, [rep_dim, dim]],
            output_core_dims=[[rep_dim, *mom_dims]],
            kwargs={
                "mom": mom_validated,
                "parallel": parallel,
                "axis": -1,
                "out": out,
                "nrep": nrep,
                "rng": rng,
            },
            keep_attrs=keep_attrs,
            **apply_ufunc_kwargs,
        )

        if not move_axis_to_end and isinstance(x, xr.DataArray):
            dims_order = [
                *(d if d != dim else rep_dim for d in x.dims),
                *mom_dims,
            ]
            xout = xout.transpose(..., *dims_order)  # pyright: ignore[reportUnknownArgumentType]

        return xout

    dtype = select_dtype(x, out=out, dtype=dtype)
    axis_neg, args = prepare_values_for_reduction(
        x,
        weight,
        *y,
        axis=axis,
        dtype=dtype,
        narrays=mom_ndim + 1,
        move_axis_to_end=move_axis_to_end,
    )

    freq = randsamp_freq(
        data=args[0],
        axis=axis_neg,
        freq=freq,  # type: ignore[arg-type]  # because of Dataset....
        rng=rng,
        nrep=nrep,
        dtype=dtype,
    )

    out = prepare_out_from_values(
        out,
        *args,
        mom=mom_validated,
        axis_neg=axis_neg,
        axis_new_size=freq.shape[0],
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
        parallel=parallel,
        size=args[0].size,
    )(out, freq, *args, axes=axes)

    return out


# * Jackknife resampling
@docfiller.decorate
def jackknife_freq(
    ndat: int,
) -> NDArrayInt:
    r"""
    Frequency array for jackknife resampling.

    Use this frequency array to perform jackknife [1]_ resampling

    Parameters
    ----------
    {ndat}

    Returns
    -------
    freq : ndarray
        Frequency array for jackknife resampling.

    See Also
    --------
    jackknife_vals
    jackknife_data
    .reduction.reduce_vals
    .reduction.reduce_data

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Jackknife_resampling

    Examples
    --------
    >>> jackknife_freq(4)
    array([[0, 1, 1, 1],
           [1, 0, 1, 1],
           [1, 1, 0, 1],
           [1, 1, 1, 0]])

    """
    freq = np.ones((ndat, ndat), dtype=np.int64)
    np.fill_diagonal(freq, 0.0)
    return freq


# * Jackknife data
# ** overloads
@overload
def jackknife_data(
    data: xr.Dataset,
    data_reduced: xr.Dataset | None = ...,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str | None = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> xr.Dataset: ...
@overload
def jackknife_data(
    data: xr.DataArray,
    data_reduced: xr.DataArray | ArrayLike | None = ...,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str | None = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> xr.DataArray: ...
# array
@overload
def jackknife_data(
    data: ArrayLikeArg[FloatT],
    data_reduced: xr.DataArray | ArrayLike | None = ...,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str | None = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: None = ...,
    out: None = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArray[FloatT]: ...
# out
@overload
def jackknife_data(
    data: ArrayLike,
    data_reduced: xr.DataArray | ArrayLike | None = ...,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str | None = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArray[FloatT],
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def jackknife_data(
    data: ArrayLike,
    data_reduced: xr.DataArray | ArrayLike | None = ...,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str | None = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLikeArg[FloatT],
    out: None = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArray[FloatT]: ...
# fallback
@overload
def jackknife_data(
    data: ArrayLike,
    data_reduced: xr.DataArray | ArrayLike | None = ...,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str | None = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    # thing should be out: NDArrayAny | None = ...
    out: None = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArrayAny: ...


@docfiller.decorate
def jackknife_data(
    data: xr.Dataset | xr.DataArray | ArrayLike,
    data_reduced: xr.Dataset | xr.DataArray | ArrayLike | None = None,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    rep_dim: str | None = "rep",
    move_axis_to_end: bool = False,
    parallel: bool | None = None,
    dtype: DTypeLike = None,
    out: NDArrayAny | None = None,
    keep_attrs: KeepAttrs = None,
    # dask specific...
    mom_dims: MomDims | None = None,
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> xr.Dataset | xr.DataArray | NDArrayAny:
    """
    Perform jackknife resample and moments data.

    This uses moments addition/subtraction to speed up jackknife resampling.

    Parameters
    ----------
    {data_numpy_or_dataarray_or_dataset}
    {mom_ndim}
    {axis}
    {dim}
    data_reduced : array-like or DataArray, optional
        ``data`` reduced along ``axis`` or ``dim``.  This will be calculated using
        :func:`.reduce_data` if not passed.
    rep_dim : str, optional
        Optionally output ``dim`` to ``rep_dim``.
    {move_axis_to_end}
    {parallel}
    {dtype}
    {out}
    {keep_attrs}
    {mom_dims_data}
    {on_missing_core_dim}
    {apply_ufunc_kwargs}

    Returns
    -------
    out : ndarray or DataArray
        Jackknife resampled  along ``axis``.  That is,
        ``out[...,axis=i, ...]`` is ``reduced_data(out[...,axis=[...,i-1,i+1,...], ...])``.


    Examples
    --------
    >>> import cmomy
    >>> data = cmomy.random.default_rng(0).random((4, 3))
    >>> out_jackknife = jackknife_data(data, mom_ndim=1, axis=0)
    >>> out_jackknife
    array([[1.5582, 0.7822, 0.2247],
           [2.1787, 0.6322, 0.22  ],
           [1.5886, 0.5969, 0.0991],
           [1.2601, 0.4982, 0.3478]])

    Note that this is equivalent to (but typically faster than) resampling with a
    frequency table from :func:``jackknife_freq``

    >>> freq = jackknife_freq(4)
    >>> resample_data(data, freq=freq, mom_ndim=1, axis=0)
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
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        # do this here to perhaps take advantage of chunced data
        if data_reduced is None:
            from .reduction import reduce_data

            data_reduced = reduce_data(
                data=data,
                mom_ndim=mom_ndim,
                dim=dim,
                axis=axis,
                parallel=parallel,
                keep_attrs=bool(keep_attrs),
                dtype=dtype,
                mom_dims=mom_dims,
                on_missing_core_dim=on_missing_core_dim,
                apply_ufunc_kwargs=apply_ufunc_kwargs,
            )

        axis, dim = select_axis_dim(data, axis=axis, dim=dim, mom_ndim=mom_ndim)
        mom_dims = validate_mom_dims(mom_dims, mom_ndim, data)

        apply_ufunc_kwargs = get_apply_ufunc_kwargs(
            apply_ufunc_kwargs,
            on_missing_core_dim=on_missing_core_dim,
            dask="parallelized",
            # NOTE: for now just use np.float64 as the output dtype.
            # see https://github.com/pydata/xarray/issues/1699
            output_dtypes=select_dtype(data, out=out, dtype=dtype) or np.float64,
        )

        core_dims = [dim, *mom_dims]
        xout: xr.DataArray = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            jackknife_data,
            data,
            data_reduced,
            input_core_dims=[core_dims, core_dims[1:]],
            output_core_dims=[core_dims],
            kwargs={
                "mom_ndim": mom_ndim,
                "parallel": parallel,
                "out": xprepare_out_for_resample_data(
                    out,
                    mom_ndim=mom_ndim,
                    axis=axis,
                    move_axis_to_end=move_axis_to_end,
                    data=data,
                ),
                "dtype": dtype,
                "axis": -1,
            },
            keep_attrs=keep_attrs,
            **apply_ufunc_kwargs,
        )

        if not move_axis_to_end and isinstance(data, xr.DataArray):
            xout = xout.transpose(*data.dims)
        if rep_dim is not None:
            xout = xout.rename({dim: rep_dim})
        return xout

    # numpy
    mom_ndim = validate_mom_ndim(mom_ndim)
    dtype = select_dtype(data, out=out, dtype=dtype)

    if data_reduced is None:
        from .reduction import reduce_data

        data_reduced = reduce_data(
            data=data,
            mom_ndim=mom_ndim,
            dim=dim,
            axis=axis,
            parallel=parallel,
            keep_attrs=bool(keep_attrs),
            dtype=dtype,
        )
    else:
        data_reduced = np.asarray(data_reduced, dtype=dtype)

    axis, data = prepare_data_for_reduction(
        data,
        axis=axis,
        mom_ndim=mom_ndim,
        dtype=dtype,
        move_axis_to_end=move_axis_to_end,
    )
    axes_data, axes_mom = axes_data_reduction(
        mom_ndim=mom_ndim, axis=axis, out_has_axis=True
    )
    # add axes for data_reduced
    axes = [axes_data[1:], axes_data, axes_mom]

    return factory_jackknife_data(
        mom_ndim=mom_ndim,
        parallel=parallel,
        size=data.size,
    )(data_reduced, data, out=out, axes=axes)


# * Jackknife vals
# ** overloads
# xarray
@overload
def jackknife_vals(
    x: xr.Dataset,
    *y: ArrayLike | xr.Dataset,
    mom: Moments,
    data_reduced: xr.Dataset | None = ...,
    weight: ArrayLike | xr.DataArray | xr.Dataset | None = ...,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    rep_dim: str | None = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> xr.Dataset: ...
@overload
def jackknife_vals(
    x: xr.DataArray,
    *y: ArrayLike | xr.DataArray,
    mom: Moments,
    data_reduced: ArrayLike | xr.DataArray | None = ...,
    weight: ArrayLike | xr.DataArray | None = ...,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    rep_dim: str | None = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> xr.DataArray: ...
# array
@overload
def jackknife_vals(
    x: ArrayLikeArg[FloatT],
    *y: ArrayLike,
    mom: Moments,
    data_reduced: ArrayLike | None = ...,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: None = ...,
    out: None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    rep_dim: str | None = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArray[FloatT]: ...
# out
@overload
def jackknife_vals(
    x: ArrayLike,
    *y: ArrayLike,
    mom: Moments,
    data_reduced: ArrayLike | None = ...,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArray[FloatT],
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    rep_dim: str | None = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def jackknife_vals(
    x: ArrayLike,
    *y: ArrayLike,
    mom: Moments,
    data_reduced: ArrayLike | None = ...,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLikeArg[FloatT],
    out: None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    rep_dim: str | None = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArray[FloatT]: ...
# fallback
@overload
def jackknife_vals(
    x: ArrayLike,
    *y: ArrayLike,
    mom: Moments,
    data_reduced: ArrayLike | None = ...,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    rep_dim: str | None = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArrayAny: ...


@docfiller.decorate
def jackknife_vals(  # noqa: PLR0914
    x: ArrayLike | xr.DataArray | xr.Dataset,
    *y: ArrayLike | xr.DataArray | xr.Dataset,
    mom: Moments,
    data_reduced: ArrayLike | xr.DataArray | xr.Dataset | None = None,
    weight: ArrayLike | xr.DataArray | xr.Dataset | None = None,
    axis: AxisReduce | MissingType = MISSING,
    move_axis_to_end: bool = True,
    parallel: bool | None = None,
    dtype: DTypeLike = None,
    out: NDArrayAny | None = None,
    # xarray specific
    dim: DimsReduce | MissingType = MISSING,
    rep_dim: str | None = "rep",
    mom_dims: MomDims | None = None,
    keep_attrs: KeepAttrs = None,
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> xr.Dataset | xr.DataArray | NDArrayAny:
    """
    Jackknife by value.

    Parameters
    ----------
    x : ndarray or DataArray
        Value to analyze
    *y:  array-like or DataArray, optional
        Second value needed if len(mom)==2.
    {mom}
    data_reduced : array-like or DataArray, optional
        ``data`` reduced along ``axis`` or ``dim``.  This will be calculated using
        :func:`.reduce_vals` if not passed.
    {weight}
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
    {on_missing_core_dim}
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
    from .reduction import reduce_vals

    mom, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
    weight = 1.0 if weight is None else weight

    if isinstance(x, (xr.DataArray, xr.Dataset)):
        if data_reduced is None:
            data_reduced = reduce_vals(  # type: ignore[misc]
                x,  # type: ignore[arg-type]
                *y,
                mom=mom,
                weight=weight,
                axis=axis,
                parallel=parallel,
                dtype=dtype,
                dim=dim,
                mom_dims=mom_dims,
                keep_attrs=keep_attrs,
                on_missing_core_dim=on_missing_core_dim,
                apply_ufunc_kwargs=apply_ufunc_kwargs,
            )

        if isinstance(x, xr.DataArray):
            dtype = select_dtype(x, out=out, dtype=dtype)

        dim, input_core_dims, xargs = xprepare_values_for_reduction(
            x,
            weight,
            *y,
            axis=axis,
            dim=dim,
            dtype=dtype,
            narrays=mom_ndim + 1,
        )

        out = xprepare_out_for_resample_vals(
            target=x,
            out=out,
            dim=dim,
            mom_ndim=mom_ndim,
            move_axis_to_end=move_axis_to_end,
        )

        mom_dims = validate_mom_dims(
            mom_dims=mom_dims, mom_ndim=mom_ndim, out=data_reduced
        )

        input_core_dims = [
            # x, weight, *y,
            *input_core_dims,
            # data_reduced
            mom_dims,
        ]

        apply_ufunc_kwargs = get_apply_ufunc_kwargs(
            apply_ufunc_kwargs,
            on_missing_core_dim=on_missing_core_dim,
            dask="parallelized",
            output_sizes=dict(zip(mom_dims, mom_to_mom_shape(mom))),
            # NOTE: for now just use np.float64 as the output dtype.
            # see https://github.com/pydata/xarray/issues/1699
            output_dtypes=select_dtype(x, out=out, dtype=dtype) or np.float64,
        )

        def _func(*args: NDArrayAny, **kwargs: Any) -> NDArrayAny:
            x, weight, *y, data_reduced = args
            return jackknife_vals(
                x, *y, weight=weight, data_reduced=data_reduced, **kwargs
            )

        xout: xr.DataArray | xr.Dataset = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _func,
            *xargs,
            data_reduced,
            input_core_dims=input_core_dims,
            output_core_dims=[(dim, *mom_dims)],
            kwargs={
                "mom": mom,
                "parallel": parallel,
                "axis": -1,
                "out": out,
                "dtype": dtype,
            },
            keep_attrs=keep_attrs,
            **apply_ufunc_kwargs,
        )

        if not move_axis_to_end and isinstance(x, xr.DataArray):
            xout = xout.transpose(..., *x.dims, *mom_dims)  # pyright: ignore[reportUnknownArgumentType]

        if rep_dim is not None:
            xout = xout.rename({dim: rep_dim})

        return xout

    # numpy
    dtype = select_dtype(x, out=out, dtype=dtype)
    if data_reduced is None:
        data_reduced = reduce_vals(  # type: ignore[misc]
            x,  # type: ignore[arg-type]
            *y,  # type: ignore[arg-type]
            mom=mom,
            weight=weight,  # type: ignore[arg-type]
            axis=axis,
            parallel=parallel,
            dtype=dtype,
            dim=dim,
            mom_dims=mom_dims,
            keep_attrs=keep_attrs,
        )
    else:
        data_reduced = np.asarray(data_reduced, dtype=dtype)
        # TODO(wpk): probably remove this.  ufunc will catch wrong shape...
        if data_reduced.shape[-mom_ndim:] != mom_to_mom_shape(mom):
            msg = f"{data_reduced.shape[-mom_ndim:]=} inconsistent with {mom=}"
            raise ValueError(msg)

    axis_neg, args = prepare_values_for_reduction(
        x,
        weight,
        *y,
        axis=axis,
        dtype=dtype,
        narrays=mom_ndim + 1,
        move_axis_to_end=move_axis_to_end,
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
        parallel=parallel,
        size=args[0].size,
    )(data_reduced, *args, out=out, axes=axes)  # type: ignore[arg-type]

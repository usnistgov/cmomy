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

from ._utils import (
    MISSING,
    normalize_axis_index,
    parallel_heuristic,
    prepare_data_for_reduction,
    prepare_values_for_reduction,
    raise_if_wrong_shape,
    select_axis_dim,
    select_dtype,
    validate_mom_and_mom_ndim,
    validate_mom_dims,
    validate_mom_ndim,
    xprepare_data_for_reduction,
    xprepare_values_for_reduction,
)
from .docstrings import docfiller
from .random import validate_rng

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence
    from typing import Any, Literal

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from .typing import (
        ArrayLikeArg,
        ArrayOrder,
        AxisReduce,
        DimsReduce,
        DTypeLikeArg,
        FloatT,
        IntDTypeT,
        KeepAttrs,
        MissingType,
        Mom_NDim,
        MomDims,
        Moments,
        MomentsStrict,
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


def _validate_resample_array(
    x: ArrayLike,
    ndat: int,
    nrep: int | None,
    is_freq: bool,
    check: bool = True,
) -> NDArrayAny:
    x = np.asarray(x, dtype=np.int64)
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


def select_ndat(
    data: xr.DataArray | NDArrayAny,
    *,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: Mom_NDim | None = None,
) -> int:
    """
    Determine ndat from array.

    Parameters
    ----------
    data : ndarray or DataArray
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
    ValueError: Cannot select moment dimension.  axis=2, dim='mom'.
    """
    ndim = data.ndim
    if mom_ndim is not None:
        ndim -= validate_mom_ndim(mom_ndim)

    if isinstance(axis, int):
        axis = normalize_axis_index(axis, ndim)

    if isinstance(data, xr.DataArray):
        axis, dim = select_axis_dim(data.dims, axis, dim)
        if axis >= ndim:
            msg = f"Cannot select moment dimension.  {axis=}, {dim=}."
            raise ValueError(msg)

    elif not isinstance(axis, int):
        msg = "Must specify integer axis for array input."
        raise TypeError(msg)

    return data.shape[axis]


@docfiller.decorate
def randsamp_freq(
    *,
    ndat: int | None = None,
    nrep: int | None = None,
    nsamp: int | None = None,
    indices: ArrayLike | None = None,
    freq: ArrayLike | None = None,
    data: xr.DataArray | NDArrayAny | None = None,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: Mom_NDim | None = None,
    check: bool = False,
    rng: np.random.Generator | None = None,
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
    data : ndarray or DataArray
    {axis}
    {dim}
    mom_ndim : int, optional
        If specified, then treat ``data`` as a moments array, and wrap negative
        values for ``axis`` relative to value dimensions only.


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
# ** Low level resamplers
def _resample_data(
    data: NDArray[FloatT],
    freq: NDArrayInt,
    *,
    mom_ndim: Mom_NDim,
    parallel: bool | None = None,
    out: NDArrayAny | None = None,
) -> NDArray[FloatT]:
    """Resample prepared data."""
    _check_freq(freq, data.shape[-(mom_ndim + 1)])

    from ._lib.factory import factory_resample_data

    _resample = factory_resample_data(
        mom_ndim=mom_ndim, parallel=parallel_heuristic(parallel, data.size * mom_ndim)
    )
    if out is not None:
        return _resample(data, freq, out)
    return _resample(data, freq)


# ** overloads
@overload
def resample_data(  # type: ignore[overload-overlap]
    data: xr.DataArray,
    *,
    mom_ndim: Mom_NDim,
    freq: NDArrayInt,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    order: ArrayOrder = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> xr.DataArray: ...
# array no out or dtype
@overload
def resample_data(
    data: ArrayLikeArg[FloatT],
    *,
    mom_ndim: Mom_NDim,
    freq: NDArrayInt,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    order: ArrayOrder = ...,
    parallel: bool | None = ...,
    dtype: None = ...,
    out: None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# out
@overload
def resample_data(
    data: Any,
    *,
    mom_ndim: Mom_NDim,
    freq: NDArrayInt,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    order: ArrayOrder = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArray[FloatT],
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def resample_data(
    data: Any,
    *,
    mom_ndim: Mom_NDim,
    freq: NDArrayInt,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    order: ArrayOrder = ...,
    parallel: bool | None = ...,
    dtype: DTypeLikeArg[FloatT],
    out: None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# fallback
@overload
def resample_data(
    data: Any,
    *,
    mom_ndim: Mom_NDim,
    freq: NDArrayInt,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    order: ArrayOrder = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArrayAny: ...


# ** Public api
@docfiller.decorate
def resample_data(
    data: xr.DataArray | ArrayLike,
    *,
    mom_ndim: Mom_NDim,
    freq: NDArrayInt,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    rep_dim: str = "rep",
    order: ArrayOrder = None,
    parallel: bool | None = True,
    dtype: DTypeLike = None,
    out: NDArrayAny | None = None,
    keep_attrs: KeepAttrs = None,
) -> xr.DataArray | NDArrayAny:
    """
    Resample data according to frequency table.

    Parameters
    ----------
    data : array-like
        central mom array to be resampled
    {mom_ndim}
    {freq}
    {axis}
    {dim}
    {rep_dim}
    {parallel}
    {order}
    {dtype}
    {out}
    {keep_attrs}

    Returns
    -------
    out : ndarray
        Resampled central moments. ``out.shape = (..., shape[axis-1], shape[axis+1], ..., nrep, mom0, ...)``,
        where ``shape = data.shape`` and ``nrep = freq.shape[0]``.

    See Also
    --------
    random_freq
    randsamp_freq
    """
    mom_ndim = validate_mom_ndim(mom_ndim)
    dtype = select_dtype(data, out=out, dtype=dtype)

    freq = np.asarray(freq, dtype=dtype)

    if isinstance(data, xr.DataArray):
        dim, data = xprepare_data_for_reduction(
            data,
            axis=axis,
            dim=dim,
            mom_ndim=mom_ndim,
            order=order,
            dtype=dtype,
        )

        core_dims: tuple[Hashable, ...] = data.dims[-(mom_ndim + 1) :]
        return xr.apply_ufunc(  # type: ignore[no-any-return] # pyright: ignore[reportUnknownMemberType]
            _resample_data,
            data,
            freq,
            input_core_dims=[core_dims, [rep_dim, dim]],
            output_core_dims=[[rep_dim, *core_dims[1:]]],
            kwargs={
                "mom_ndim": mom_ndim,
                "parallel": parallel,
                "out": out,
                # "freq": freq
            },
            keep_attrs=keep_attrs,
        )

    # numpy array
    data = prepare_data_for_reduction(
        data, axis=axis, mom_ndim=mom_ndim, dtype=dtype, order=order
    )
    return _resample_data(
        data=data,
        freq=freq,
        mom_ndim=mom_ndim,
        parallel=parallel,
        out=out,
    )


# * Resample vals
# ** low level
def _resample_vals(
    x0: NDArray[FloatT],
    w: NDArray[FloatT],
    freq: NDArrayInt,
    *x1: NDArray[FloatT],
    mom: MomentsStrict,
    mom_ndim: Mom_NDim,
    parallel: bool | None = None,
    out: NDArray[FloatT] | None = None,
) -> NDArray[FloatT]:
    _check_freq(freq, x0.shape[-1])

    val_shape: tuple[int, ...] = np.broadcast_shapes(*(_.shape for _ in (x0, *x1, w)))[
        :-1
    ]
    mom_shape: tuple[int, ...] = tuple(m + 1 for m in mom)
    out_shape: tuple[int, ...] = (
        *val_shape,
        freq.shape[0],
        *mom_shape,
    )
    if out is None:
        out = np.zeros(out_shape, dtype=x0.dtype)
    else:
        raise_if_wrong_shape(out, out_shape)
        out.fill(0.0)

    from ._lib.factory import factory_resample_vals

    factory_resample_vals(  # type: ignore[call-arg]
        mom_ndim=mom_ndim,
        parallel=parallel_heuristic(parallel, x0.size * mom_ndim),
    )(x0, *x1, w, freq, out)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue]

    return out


# ** overloads
@overload
def resample_vals(  # type: ignore[overload-overlap]
    x: xr.DataArray,
    *y: ArrayLike | xr.DataArray,
    mom: Moments,
    freq: NDArrayInt,
    weight: ArrayLike | xr.DataArray | None = ...,
    axis: AxisReduce | MissingType = ...,
    order: ArrayOrder = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> xr.DataArray: ...
# array
@overload
def resample_vals(
    x: ArrayLikeArg[FloatT],
    *y: ArrayLike,
    mom: Moments,
    freq: NDArrayInt,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    order: ArrayOrder = ...,
    parallel: bool | None = ...,
    dtype: None = ...,
    out: None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# out
@overload
def resample_vals(
    x: Any,
    *y: ArrayLike,
    mom: Moments,
    freq: NDArrayInt,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    order: ArrayOrder = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArray[FloatT],
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def resample_vals(
    x: Any,
    *y: ArrayLike,
    mom: Moments,
    freq: NDArrayInt,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    order: ArrayOrder = ...,
    parallel: bool | None = ...,
    dtype: DTypeLikeArg[FloatT],
    out: None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# fallback
@overload
def resample_vals(
    x: Any,
    *y: ArrayLike,
    mom: Moments,
    freq: NDArrayInt,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    order: ArrayOrder = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArrayAny: ...


# ** public api
@docfiller.decorate
def resample_vals(
    x: ArrayLike | xr.DataArray,
    *y: ArrayLike | xr.DataArray,
    mom: Moments,
    freq: NDArrayInt,
    weight: ArrayLike | xr.DataArray | None = None,
    axis: AxisReduce | MissingType = MISSING,
    order: ArrayOrder = None,
    parallel: bool | None = None,
    dtype: DTypeLike = None,
    out: NDArrayAny | None = None,
    # xarray specific
    dim: DimsReduce | MissingType = MISSING,
    rep_dim: str = "rep",
    mom_dims: MomDims | None = None,
    keep_attrs: KeepAttrs = None,
) -> NDArrayAny | xr.DataArray:
    """
    Resample data according to frequency table.

    Parameters
    ----------
    x : ndarray
        Value to analyze
    *y:  array-like, optional
        Second value needed if len(mom)==2.
    {freq}
    {mom}
    {weight}
    {axis}
    {order}
    {parallel}
    {dtype}
    {out}
    {dim}
    {rep_dim}
    {mom_dims}
    {keep_attrs}

    Returns
    -------
    out : ndarray
        Resampled Central moments array. ``out.shape = (...,shape[axis-1], shape[axis+1], ..., nrep, mom0, ...)``
        where ``shape = args[0].shape``. and ``nrep = freq.shape[0]``.

    See Also
    --------
    random_freq
    randsamp_freq
    """
    mom_validated, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
    weight = 1.0 if weight is None else weight

    dtype = select_dtype(x, out=out, dtype=dtype)

    freq = np.asarray(freq, dtype=dtype)

    if isinstance(x, xr.DataArray):
        input_core_dims, (dx0, dw, *dx1) = xprepare_values_for_reduction(
            x,
            weight,
            *y,
            axis=axis,
            dim=dim,
            dtype=dtype,
            order=order,
            narrays=mom_ndim + 1,
        )

        mom_dims_strict: tuple[Hashable, ...] = validate_mom_dims(
            mom_dims=mom_dims, mom_ndim=mom_ndim
        )
        # add in freq dims:
        input_core_dims.insert(2, [rep_dim, input_core_dims[0][0]])

        return xr.apply_ufunc(  # type: ignore[no-any-return]
            _resample_vals,
            dx0,
            dw,
            freq,
            *dx1,
            input_core_dims=input_core_dims,
            output_core_dims=[[rep_dim, *mom_dims_strict]],
            kwargs={
                "mom": mom_validated,
                "mom_ndim": mom_ndim,
                "parallel": parallel,
                "out": out,
            },
            keep_attrs=keep_attrs,
        )

    (
        x0,
        w,
        *x1,
    ) = prepare_values_for_reduction(
        x, weight, *y, axis=axis, dtype=dtype, order=order, narrays=mom_ndim + 1
    )

    return _resample_vals(
        x0,
        w,
        freq,
        *x1,
        mom=mom_validated,
        mom_ndim=mom_ndim,
        parallel=parallel,
        out=out,
    )


# TODO(wpk): add coverage for these
def bootstrap_confidence_interval(  # pragma: no cover
    distribution: NDArrayAny,
    stats_val: NDArrayAny | Literal["percentile", "mean", "median"] | None = "mean",
    axis: int = 0,
    alpha: float = 0.05,
    style: Literal[None, "delta", "pm"] = None,
    **kwargs: Any,
) -> NDArrayAny:
    """
    Calculate the error bounds.

    Parameters
    ----------
    distribution : array-like
        distribution of values to consider
    stats_val : array-like, {None, 'mean','median'}, optional
        * array: perform pivotal error bounds (correct) with this as `value`.
        * percentile: percentiles, with value as median
        * mean: pivotal error bounds with mean as value
        * median: pivotal error bounds with median as value
    axis : int, default=0
        axis to analyze along
    alpha : float
        alpha value for confidence interval.
        Percent confidence = `100 * (1 - alpha)`
    style : {None, 'delta', 'pm'}
        controls style of output
    **kwargs
        extra arguments to `numpy.percentile`

    Returns
    -------
    out : array
        fist dimension will be statistics.  Other dimensions
        have shape of input less axis reduced over.
        Depending on `style` first dimension will be
        (note val is either stats_val or median):

        * None: [val, low, high]
        * delta:  [val, val-low, high - val]
        * pm : [val, (high - low) / 2]

    """
    if stats_val is None:
        p_low = 100 * (alpha / 2.0)
        p_mid = 50
        p_high = 100 - p_low
        val, low, high = np.percentile(  # pyright: ignore[reportUnknownMemberType]
            a=distribution, q=[p_mid, p_low, p_high], axis=axis, **kwargs
        )

    else:
        if isinstance(stats_val, str):
            if stats_val == "mean":
                sv = np.mean(distribution, axis=axis)
            elif stats_val == "median":
                sv = np.median(distribution, axis=axis)
            else:
                msg = "stats val should be None, mean, median, or an array"
                raise ValueError(msg)

        else:
            sv = stats_val

        q_high = 100 * (alpha / 2.0)
        q_low = 100 - q_high
        val = sv
        # fmt: off
        low = 2 * sv - np.percentile(  # pyright: ignore[reportUnknownMemberType]
            a=distribution, q=q_low, axis=axis, **kwargs
        )
        high = 2 * sv - np.percentile(  # pyright: ignore[reportUnknownMemberType]
            a=distribution, q=q_high, axis=axis, **kwargs
        )
        # fmt: on

    if style is None:
        out = np.array([val, low, high])
    elif style == "delta":
        out = np.array([val, val - low, high - val])
    elif style == "pm":
        out = np.array([val, (high - low) / 2.0])
    return out


def xbootstrap_confidence_interval(  # pragma: no cover
    x: xr.DataArray,
    stats_val: NDArrayAny | Literal["percentile", "mean", "median"] | None = "mean",
    axis: int = 0,
    dim: DimsReduce | MissingType = MISSING,
    alpha: float = 0.05,
    style: Literal[None, "delta", "pm"] = None,
    bootstrap_dim: Hashable | None = "bootstrap",
    bootstrap_coords: str | Sequence[str] | None = None,
    **kwargs: Any,
) -> xr.DataArray:
    """
    Bootstrap xarray object.

    Parameters
    ----------
    dim : str
        if passed, use reduce along this dimension
    bootstrap_dim : str, default='bootstrap'
        name of new dimension.  If `bootstrap_dim` conflicts, then
        `new_name = dim + new_name`
    bootstrap_coords : array-like or str
        coords of new dimension.
        If `None`, use default names
        If string, use this for the 'values' name
    """
    if dim is not None:
        axis = x.get_axis_num(dim)
    else:
        dim = x.dims[axis]

    template = x.isel(indexers={dim: 0})

    if bootstrap_dim is None:
        bootstrap_dim = "bootstrap"

    if bootstrap_dim in template.dims:
        bootstrap_dim = f"{dim}_{bootstrap_dim}"
    dims = (bootstrap_dim, *template.dims)

    if bootstrap_coords is None:
        bootstrap_coords = stats_val if isinstance(stats_val, str) else "stats_val"

    if isinstance(bootstrap_coords, str):
        if style is None:
            bootstrap_coords = [bootstrap_coords, "low", "high"]
        elif style == "delta":
            bootstrap_coords = [bootstrap_coords, "err_low", "err_high"]
        elif style == "pm":
            bootstrap_coords = [bootstrap_coords, "err"]
        else:
            msg = f"unknown style={style}"
            raise ValueError(msg)

    if not isinstance(stats_val, str):
        stats_val = np.array(stats_val)

    out = bootstrap_confidence_interval(
        x.to_numpy(),  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        stats_val=stats_val,
        axis=axis,
        alpha=alpha,
        style=style,
        **kwargs,
    )

    out_xr = xr.DataArray(
        out,
        dims=dims,
        coords=template.coords,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        attrs=template.attrs,
        name=template.name,
        # indexes=template.indexes,
    )

    out_xr.coords[bootstrap_dim] = bootstrap_coords  # pyright: ignore[reportUnknownMemberType]

    return out_xr

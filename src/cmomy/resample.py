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
    axes_data_reduction,
    get_axes_from_values,
    get_out_from_values,
    mom_to_mom_shape,
    normalize_axis_index,
    parallel_heuristic,
    prepare_data_for_reduction,
    prepare_values_for_reduction,
    select_axis_dim,
    select_dtype,
    validate_mom_and_mom_ndim,
    validate_mom_dims,
    validate_mom_ndim,
    xprepare_out_for_resample_data,
    xprepare_out_for_resample_vals,
    xprepare_values_for_reduction,
)
from .docstrings import docfiller
from .random import validate_rng

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from .typing import (
        ArrayLikeArg,
        AxesGUFunc,
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


@docfiller.decorate
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
    ValueError: Cannot select moment dimension. axis=2, dim='mom'.
    """
    if isinstance(data, xr.DataArray):
        axis, dim = select_axis_dim(data, axis=axis, dim=dim, mom_ndim=mom_ndim)
    elif isinstance(axis, int):
        axis = normalize_axis_index(axis, data.ndim, mom_ndim=mom_ndim)
    else:
        msg = "Must specify integer axis for array input."
        raise TypeError(msg)

    return data.shape[axis]


# * General frequency table generation.
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
        return np.asarray(freq, np.int64)

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
# ** overloads
@overload
def resample_data(
    data: xr.DataArray,
    freq: NDArrayInt,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> xr.DataArray: ...
# array no out or dtype
@overload
def resample_data(
    data: ArrayLikeArg[FloatT],
    freq: NDArrayInt,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: None = ...,
    out: None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# out
@overload
def resample_data(
    data: Any,
    freq: NDArrayInt,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArray[FloatT],
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def resample_data(
    data: Any,
    freq: NDArrayInt,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLikeArg[FloatT],
    out: None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# fallback
@overload
def resample_data(
    data: Any,
    freq: NDArrayInt,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArrayAny: ...


# TODO(wpk): Should out above be out: NDArrayAny | None = ... ?


# ** Public api
@docfiller.decorate
def resample_data(
    data: xr.DataArray | ArrayLike,
    freq: NDArrayInt,
    *,
    mom_ndim: Mom_NDim,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    rep_dim: str = "rep",
    move_axis_to_end: bool = False,
    parallel: bool | None = None,
    dtype: DTypeLike = None,
    out: NDArrayAny | None = None,
    keep_attrs: KeepAttrs = None,
) -> xr.DataArray | NDArrayAny:
    """
    Resample data according to frequency table.

    Parameters
    ----------
    data : array-like or DataArray
        Central mom array to be resampled
    {mom_ndim}
    {freq}
    {axis}
    {dim}
    {rep_dim}
    {move_axis_to_end}
    {parallel}
    {dtype}
    {out}
    {keep_attrs}

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
    if isinstance(data, xr.DataArray):
        axis, dim = select_axis_dim(data, axis=axis, dim=dim, mom_ndim=mom_ndim)

        xout: xr.DataArray = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            resample_data,
            data,
            freq,
            input_core_dims=[[dim, *data.dims[-mom_ndim:]], [rep_dim, dim]],
            output_core_dims=[[rep_dim, *data.dims[-mom_ndim:]]],
            kwargs={
                "mom_ndim": mom_ndim,
                "parallel": parallel,
                "out": xprepare_out_for_resample_data(
                    out,
                    mom_ndim=mom_ndim,
                    axis=axis,
                    move_axis_to_end=move_axis_to_end,
                ),
                "dtype": dtype,
                "axis": -1,
            },
            keep_attrs=keep_attrs,
        )

        if not move_axis_to_end:
            dims_order = (*data.dims[:axis], rep_dim, *data.dims[axis + 1 :])
            xout = xout.transpose(*dims_order)
        return xout

    mom_ndim = validate_mom_ndim(mom_ndim)
    dtype = select_dtype(data, out=out, dtype=dtype)
    freq = np.asarray(freq, dtype=dtype)

    axis, data = prepare_data_for_reduction(
        data,
        axis=axis,
        mom_ndim=mom_ndim,
        dtype=dtype,
        move_axis_to_end=move_axis_to_end,
    )

    # include inner core dimensions for freq
    axes = [
        (-2, -1),
        *axes_data_reduction(mom_ndim=mom_ndim, axis=axis, out_has_axis=True),
    ]

    _check_freq(freq, data.shape[axis])

    from ._lib.factory import factory_resample_data

    return factory_resample_data(
        mom_ndim=mom_ndim, parallel=parallel_heuristic(parallel, data.size * mom_ndim)
    )(freq, data, out=out, axes=axes)


# * Resample vals
# ** overloads
@overload
def resample_vals(  # pyright: ignore[reportOverlappingOverload]
    x: xr.DataArray,
    *y: ArrayLike | xr.DataArray,
    mom: Moments,
    freq: NDArrayInt,
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
    move_axis_to_end: bool = ...,
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
    move_axis_to_end: bool = ...,
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
    move_axis_to_end: bool = ...,
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
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    rep_dim: str = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArrayAny: ...


# ** public api
@docfiller.decorate
def resample_vals(  # pyright: ignore[reportOverlappingOverload]
    x: ArrayLike | xr.DataArray,
    *y: ArrayLike | xr.DataArray,
    mom: Moments,
    freq: NDArrayInt,
    weight: ArrayLike | xr.DataArray | None = None,
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
) -> NDArrayAny | xr.DataArray:
    """
    Resample data according to frequency table.

    Parameters
    ----------
    x : ndarray or DataArray
        Value to analyze
    *y:  array-like or DataArray, optional
        Second value needed if len(mom)==2.
    {freq}
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
    dtype = select_dtype(x, out=out, dtype=dtype)

    if isinstance(x, xr.DataArray):
        input_core_dims, xargs = xprepare_values_for_reduction(
            x,
            weight,
            *y,
            axis=axis,
            dim=dim,
            dtype=dtype,
            narrays=mom_ndim + 1,
        )

        dim = input_core_dims[0][0]
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

        xout: xr.DataArray = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _resample_vals,
            *xargs,
            input_core_dims=input_core_dims,
            output_core_dims=[[rep_dim, *mom_dims]],
            kwargs={
                "mom": mom_validated,
                "mom_ndim": mom_ndim,
                "parallel": parallel,
                "axis_neg": -1,
                "out": out,
                "freq": freq,
            },
            keep_attrs=keep_attrs,
        )

        if not move_axis_to_end:
            dims_order = [
                *(d if d != dim else rep_dim for d in x.dims),
                *mom_dims,
            ]
            xout = xout.transpose(..., *dims_order)  # pyright: ignore[reportUnknownArgumentType]

        return xout

    freq = np.asarray(freq, dtype=dtype)
    axis_neg, args = prepare_values_for_reduction(
        x,
        weight,
        *y,
        axis=axis,
        dtype=dtype,
        narrays=mom_ndim + 1,
        move_axis_to_end=move_axis_to_end,
    )

    return _resample_vals(
        *args,
        out=out,
        freq=freq,
        mom=mom_validated,
        mom_ndim=mom_ndim,
        axis_neg=axis_neg,
        parallel=parallel,
    )


# ** low level
def _resample_vals(
    x0: NDArray[FloatT],
    w: NDArray[FloatT],
    *x1: NDArray[FloatT],
    out: NDArray[FloatT] | None,
    freq: NDArrayInt,
    mom: MomentsStrict,
    mom_ndim: Mom_NDim,
    axis_neg: int,
    parallel: bool | None = None,
) -> NDArray[FloatT]:
    _check_freq(freq, x0.shape[axis_neg])

    args: tuple[NDArray[FloatT], ...] = (x0, w, *x1)  # type: ignore[arg-type]

    if out is None:
        out = get_out_from_values(
            *args,
            mom=mom,
            axis_neg=axis_neg,
            axis_new_size=freq.shape[0],
        )
    else:
        out.fill(0.0)

    axes: AxesGUFunc = [
        # out
        (axis_neg - mom_ndim, *range(-mom_ndim, 0)),
        # freq
        (-2, -1),
        # x, weight, *y
        *get_axes_from_values(*args, axis_neg=axis_neg),
    ]

    from ._lib.factory import factory_resample_vals

    factory_resample_vals(
        mom_ndim=mom_ndim,
        parallel=parallel_heuristic(parallel, x0.size * mom_ndim),
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
    reduce_vals
    reduce_data

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
) -> NDArray[FloatT]: ...
# out
@overload
def jackknife_data(
    data: Any,
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
) -> NDArray[FloatT]: ...
# dtype
@overload
def jackknife_data(
    data: Any,
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
) -> NDArray[FloatT]: ...
# fallback
@overload
def jackknife_data(
    data: Any,
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
) -> NDArrayAny: ...


@docfiller.decorate
def jackknife_data(
    data: xr.DataArray | ArrayLike,
    data_reduced: xr.DataArray | ArrayLike | None = None,
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
) -> xr.DataArray | NDArrayAny:
    """
    Perform jackknife resample and moments data.

    This uses moments addition/subtraction (see :class:`.CentralMoments.__sub__`) to speed up jackknife resampling.

    Parameters
    ----------
    data : array-like or DataArray
        Central mom array to be resampled
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
    if isinstance(data, xr.DataArray):
        axis, dim = select_axis_dim(data, axis=axis, dim=dim, mom_ndim=mom_ndim)

        core_dims = [dim, *data.dims[-mom_ndim:]]
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
                ),
                "dtype": dtype,
                "axis": -1,
            },
            keep_attrs=keep_attrs,
        )

        if not move_axis_to_end:
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

    from ._lib.factory import factory_jackknife_data

    return factory_jackknife_data(
        mom_ndim=mom_ndim, parallel=parallel_heuristic(parallel, data.size * mom_ndim)
    )(data_reduced, data, out=out, axes=axes)


# * Jackknife vals
# ** overloads
# xarray
@overload
def jackknife_vals(
    x: xr.DataArray,
    *y: xr.DataArray | ArrayLike,
    mom: Moments,
    data_reduced: xr.DataArray | ArrayLike | None = ...,
    weight: xr.DataArray | ArrayLike | None = ...,
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
) -> NDArray[FloatT]: ...
# out
@overload
def jackknife_vals(
    x: Any,
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
) -> NDArray[FloatT]: ...
# dtype
@overload
def jackknife_vals(
    x: Any,
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
) -> NDArray[FloatT]: ...
# fallback
@overload
def jackknife_vals(
    x: Any,
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
) -> NDArrayAny: ...


@docfiller.decorate
def jackknife_vals(
    x: xr.DataArray | ArrayLike,
    *y: xr.DataArray | ArrayLike,
    mom: Moments,
    data_reduced: xr.DataArray | ArrayLike | None = None,
    weight: xr.DataArray | ArrayLike | None = None,
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
) -> NDArrayAny | xr.DataArray:
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
        from .reduction import reduce_vals

        data_reduced = reduce_vals(
            x,
            *y,
            mom=mom,
            weight=weight,
            axis=axis,
            parallel=parallel,
            dtype=dtype,
            dim=dim,
            mom_dims=mom_dims,
            keep_attrs=keep_attrs,
        )
    else:
        data_reduced = (
            data_reduced.astype(dtype, copy=False)  # pyright: ignore[reportUnknownMemberType]
            if isinstance(data_reduced, (xr.DataArray, np.ndarray))
            else np.asarray(data_reduced, dtype=dtype)
        )

        if data_reduced.shape[-mom_ndim:] != mom_to_mom_shape(mom):
            msg = f"{data_reduced.shape[-mom_ndim:]=} inconsistent with {mom=}"
            raise ValueError(msg)

    if isinstance(x, xr.DataArray):
        input_core_dims, xargs = xprepare_values_for_reduction(
            x,
            weight,
            *y,
            axis=axis,
            dim=dim,
            dtype=dtype,
            narrays=mom_ndim + 1,
        )

        dim = input_core_dims[0][0]
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
            # data_reduced
            mom_dims,
            # x, weight, *y,
            *input_core_dims,
        ]

        xout: xr.DataArray = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _jackknife_vals,
            data_reduced,
            *xargs,
            input_core_dims=input_core_dims,
            output_core_dims=[(dim, *mom_dims)],
            kwargs={
                "mom_ndim": mom_ndim,
                "parallel": parallel,
                "axis_neg": -1,
                "out": out,
            },
            keep_attrs=keep_attrs,
        )

        if not move_axis_to_end:
            xout = xout.transpose(..., *x.dims, *mom_dims)  # pyright: ignore[reportUnknownArgumentType]

        if rep_dim is not None:
            xout = xout.rename({dim: rep_dim})

        return xout

    # numpy
    axis_neg, args = prepare_values_for_reduction(
        x,
        weight,
        *y,
        axis=axis,
        dtype=dtype,
        narrays=mom_ndim + 1,
        move_axis_to_end=move_axis_to_end,
    )

    return _jackknife_vals(
        data_reduced.to_numpy()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        if isinstance(data_reduced, xr.DataArray)
        else data_reduced,
        *args,
        axis_neg=axis_neg,
        mom_ndim=mom_ndim,
        parallel=parallel,
        out=out,
    )


# ** low level
def _jackknife_vals(
    data_reduced: NDArray[FloatT],
    x0: NDArray[FloatT],
    w: NDArray[FloatT],
    *x1: NDArray[FloatT],
    axis_neg: int,
    mom_ndim: Mom_NDim,
    parallel: bool | None = None,
    out: NDArray[FloatT] | None = None,
) -> NDArray[FloatT]:
    args: tuple[NDArray[FloatT], ...] = (x0, w, *x1)  # type: ignore[arg-type]

    axes: AxesGUFunc = [
        # data_reduced
        tuple(range(-mom_ndim, 0)),
        # x, w, *y
        *get_axes_from_values(*args, axis_neg=axis_neg),
        # out
        (axis_neg - mom_ndim, *range(-mom_ndim, 0)),
    ]

    from ._lib.factory import factory_jackknife_vals

    return factory_jackknife_vals(
        mom_ndim=mom_ndim, parallel=parallel_heuristic(parallel, x0.size * mom_ndim)
    )(data_reduced, *args, out=out, axes=axes)

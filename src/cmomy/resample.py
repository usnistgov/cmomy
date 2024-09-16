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

from .core.array_utils import (
    asarray_maybe_recast,
    axes_data_reduction,
    get_axes_from_values,
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
    is_dataarray,
    is_ndarray,
    is_xarray,
    raise_if_wrong_value,
    validate_mom_and_mom_ndim,
    validate_mom_dims,
    validate_mom_ndim,
)
from .core.xr_utils import (
    get_apply_ufunc_kwargs,
    raise_if_dataset,
    select_axis_dim,
)
from .factory import (
    factory_jackknife_data,
    factory_jackknife_vals,
    factory_resample_data,
    factory_resample_vals,
    parallel_heuristic,
)
from .random import validate_rng

if TYPE_CHECKING:
    from collections.abc import Hashable
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from cmomy.core.typing import MomentsStrict

    from .core.typing import (
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
        IntDTypeT,
        JackknifeDataKwargs,
        JackknifeValsKwargs,
        KeepAttrs,
        MissingCoreDimOptions,
        MissingType,
        Mom_NDim,
        MomDims,
        Moments,
        NDArrayAny,
        NDArrayInt,
        ResampleDataKwargs,
        ResampleValsKwargs,
        RngTypes,
    )
    from .core.typing_compat import Unpack


# * Resampling utilities ------------------------------------------------------
@docfiller.decorate
def random_indices(
    nrep: int,
    ndat: int,
    nsamp: int | None = None,
    rng: RngTypes | None = None,
    replace: bool = True,
) -> NDArray[IntDTypeT]:
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
    rng: RngTypes | None = None,
    replace: bool = True,
) -> NDArray[IntDTypeT]:
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


@overload
def freq_to_indices(
    freq: DataT,
    *,
    shuffle: bool = ...,
    rng: RngTypes | None = ...,
) -> DataT: ...
@overload
def freq_to_indices(
    freq: ArrayLike,
    *,
    shuffle: bool = ...,
    rng: RngTypes | None = ...,
) -> NDArray[IntDTypeT]: ...


@docfiller.decorate
def freq_to_indices(
    freq: ArrayLike | DataT,
    *,
    shuffle: bool = True,
    rng: RngTypes | None = None,
) -> NDArray[IntDTypeT] | DataT:
    """
    Convert a frequency array to indices array.

    This creates an "indices" array that is compatible with "freq" array.
    Note that by default, the indices for a single sample (along output[k, :])
    are randomly shuffled.  If you pass `shuffle=False`, then the output will
    be something like [[0,0,..., 1,1,..., 2,2, ...]].

    Parameters
    ----------
    {freq_xarray}
    shuffle :
        If ``True`` (default), shuffle values for each row.
    {rng}

    Returns
    -------
    ndarray :
        Indices array of shape ``(nrep, nsamp)`` where ``nsamp = freq[k,
        :].sum()`` where `k` is any row.
    """
    if is_xarray(freq):
        rep_dim, dim = freq.dims
        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            freq_to_indices,
            freq,
            input_core_dims=[[rep_dim, dim]],
            output_core_dims=[[rep_dim, dim]],
            # possible for data dimension to change size.
            exclude_dims={dim},
            kwargs={"shuffle": shuffle, "rng": rng},
        )
        return xout

    freq = np.asarray(freq, dtype=np.int64)
    indices_all: list[NDArrayAny] = []

    # validate freq -> indices
    nsamps = freq.sum(-1)
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


@overload
def indices_to_freq(
    indices: DataT,
    *,
    ndat: int | None = ...,
) -> DataT: ...
@overload
def indices_to_freq(
    indices: ArrayLike,
    *,
    ndat: int | None = ...,
) -> NDArray[IntDTypeT]: ...


def indices_to_freq(
    indices: ArrayLike | DataT,
    *,
    ndat: int | None = None,
) -> NDArray[IntDTypeT] | DataT:
    """
    Convert indices to frequency array.

    It is assumed that ``indices.shape == (nrep, nsamp)`` with ``nsamp ==
    ndat``. For cases that ``nsamp != ndat``, pass in ``ndat`` explicitl.
    """
    if is_xarray(indices):
        # assume dims are in order (rep, dim)
        rep_dim, dim = indices.dims
        ndat = ndat or indices.sizes[dim]
        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            indices_to_freq,
            indices,
            input_core_dims=[[rep_dim, dim]],
            output_core_dims=[[rep_dim, dim]],
            # allow for possibility that dim will change size...
            exclude_dims={dim},
            kwargs={"ndat": ndat},
        )
        return xout

    from ._lib.utils import (
        randsamp_indices_to_freq,
    )

    indices = np.asarray(indices, np.int64)
    ndat = indices.shape[1] if ndat is None else ndat
    freq = np.zeros((indices.shape[0], ndat), dtype=indices.dtype)
    randsamp_indices_to_freq(indices, freq)

    return freq


# * select ndat
@docfiller.decorate
def select_ndat(
    data: ArrayLike | xr.DataArray | xr.Dataset,
    *,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: Mom_NDim | None = None,
) -> int:
    """
    Determine ndat from array.

    Parameters
    ----------
    data : ndarray, DataArray, Dataset
    {axis}
    {dim}
    {mom_ndim_optional}

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
    from cmomy.core.array_utils import normalize_axis_index

    if is_xarray(data):
        axis, dim = select_axis_dim(data, axis=axis, dim=dim, mom_ndim=mom_ndim)
        return data.sizes[dim]

    if isinstance(axis, int):
        data = np.asarray(data)
        axis = normalize_axis_index(axis, data.ndim, mom_ndim=mom_ndim)
        return data.shape[axis]
    msg = "Must specify integer axis for array input."
    raise TypeError(msg)


# * General frequency table generation.
def _validate_resample_array(
    x: ArrayLike | DataT,
    ndat: int,
    nrep: int | None,
    is_freq: bool,
    check: bool = True,
    dtype: DTypeLike = np.int64,
) -> NDArrayAny | DataT:
    if is_xarray(x):
        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _validate_resample_array,
            x,
            kwargs={
                "ndat": ndat,
                "nrep": nrep,
                "is_freq": is_freq,
                "check": check,
                "dtype": dtype,
            },
        )
        return xout

    x = np.asarray(x, dtype=dtype)
    if check:
        name = "freq" if is_freq else "indices"
        raise_if_wrong_value(x.ndim, 2, f"`{name}` has wrong number of dimensions.")

        if nrep is not None:
            raise_if_wrong_value(x.shape[0], nrep, "Wrong nrep.")

        if is_freq:
            raise_if_wrong_value(x.shape[1], ndat, "Wrong ndat.")

        else:
            # only restriction is that values in [0, ndat)
            min_, max_ = x.min(), x.max()
            if min_ < 0 or max_ >= ndat:
                msg = f"Indices range [{min_}, {max_}) outside [0, {ndat - 1})"
                raise ValueError(msg)
    return x


def _randsamp_freq_dataarray_or_dataset(
    data: DataT,
    *,
    nrep: int,
    ndat: int,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    nsamp: int | None = None,
    rep_dim: str = "rep",
    paired: bool = True,
    mom_ndim: Mom_NDim | None = None,
    mom_dims: MomDims | None = None,
    rng: RngTypes | None = None,
) -> xr.DataArray | DataT:
    """Create a resampling DataArray or Dataset."""
    dim = select_axis_dim(data, axis=axis, dim=dim, mom_ndim=mom_ndim)[1]
    # make sure to validate rng here in case call multiple have dataset...
    rng = validate_rng(rng)

    def _get_unique_freq() -> xr.DataArray:
        return xr.DataArray(
            random_freq(nrep=nrep, ndat=ndat, nsamp=nsamp, rng=rng, replace=True),
            dims=[rep_dim, dim],
        )

    if is_dataarray(data) or paired:
        return _get_unique_freq()

    # generate non-paired dataset
    if mom_ndim:
        mom_dims = validate_mom_dims(mom_dims, mom_ndim, data)
    elif mom_dims:
        mom_dims = (mom_dims,) if isinstance(mom_dims, str) else tuple(mom_dims)  # type: ignore[arg-type]
        mom_ndim = validate_mom_ndim(len(mom_dims))
    else:
        mom_dims = ()
    dims = {dim, *mom_dims}  # type: ignore[misc]
    out: dict[Hashable, xr.DataArray] = {}
    for name, da in data.items():
        if dims.issubset(da.dims):
            out[name] = _get_unique_freq()
    if len(out) == 1:
        # return just a dataarray in this case
        return next(iter(out.values()))
    return xr.Dataset(out)  # pyright: ignore[reportReturnType]


@docfiller.decorate
def randsamp_freq(
    *,
    freq: ArrayLike | xr.DataArray | DataT | None = None,
    indices: ArrayLike | xr.DataArray | DataT | None = None,
    ndat: int | None = None,
    nrep: int | None = None,
    nsamp: int | None = None,
    data: ArrayLike | DataT | None = None,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: Mom_NDim | None = None,
    mom_dims: MomDims | None = None,
    rep_dim: str = "rep",
    paired: bool = True,
    rng: RngTypes | None = None,
    dtype: DTypeLike = np.int64,
    check: bool = False,
) -> NDArrayAny | xr.DataArray | DataT:
    """
    Convenience function to create frequency table for resampling.

    In order, the return will be one of ``freq``, frequencies from ``indices`` or
    new sample from :func:`random_freq`.

    Parameters
    ----------
    {ndat}
    {nrep}
    {nsamp}
    {freq_xarray}
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
    >>> rng = cmomy.default_rng(0)
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
        if is_xarray(freq):
            return freq if dtype is None else freq.astype(dtype, copy=False)  # pyright: ignore[reportUnknownMemberType]
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
    elif nrep is None:
        msg = "must specify freq, indices, or nrep"
        raise ValueError(msg)
    elif is_xarray(data):
        freq = _randsamp_freq_dataarray_or_dataset(  # type: ignore[type-var]
            data,
            nrep=nrep,
            axis=axis,
            dim=dim,
            nsamp=nsamp,
            ndat=ndat,
            rep_dim=rep_dim,
            paired=paired,
            mom_ndim=mom_ndim,
            mom_dims=mom_dims,
            rng=rng,
        )
    else:
        freq = random_freq(
            nrep=nrep, ndat=ndat, nsamp=nsamp, rng=validate_rng(rng), replace=True
        )

    return freq if dtype is None else freq.astype(dtype, copy=False)  # pyright: ignore[reportUnknownMemberType]


# * Utilities
def _select_nrep(
    freq: ArrayLike | xr.DataArray | xr.Dataset | None,
    nrep: int | None,
    rep_dim: str,
) -> int:
    if freq is not None:
        return freq.sizes[rep_dim] if is_xarray(freq) else np.shape(freq)[0]

    if nrep is not None:
        return nrep
    msg = "Must specify either freq or nrep"
    raise ValueError(msg)


# * Resample data
# ** overloads
@overload
def resample_data(
    data: DataT,
    *,
    freq: ArrayLike | xr.DataArray | DataT | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ResampleDataKwargs],
) -> DataT: ...
# array no out or dtype
@overload
def resample_data(
    data: ArrayLikeArg[FloatT],
    *,
    freq: ArrayLike | None = ...,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[ResampleDataKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def resample_data(
    data: ArrayLike,
    *,
    freq: ArrayLike | None = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ResampleDataKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def resample_data(
    data: ArrayLike,
    *,
    freq: ArrayLike | None = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[ResampleDataKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def resample_data(
    data: ArrayLike,
    *,
    freq: ArrayLike | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ResampleDataKwargs],
) -> NDArrayAny: ...


# ** Public api
@docfiller.decorate
def resample_data(  # noqa: PLR0913
    data: ArrayLike | DataT,
    *,
    mom_ndim: Mom_NDim = 1,
    freq: ArrayLike | xr.DataArray | DataT | None = None,
    nrep: int | None = None,
    rng: RngTypes | None = None,
    paired: bool = True,
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
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Resample data according to frequency table.

    Parameters
    ----------
    {data_numpy_or_dataarray_or_dataset}
    {mom_ndim}
    {freq_xarray}
    {nrep_optional}
    {rng}
    {paired}
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
    mom_ndim = validate_mom_ndim(mom_ndim)
    dtype = select_dtype(data, out=out, dtype=dtype)
    freq = randsamp_freq(
        data=data,
        freq=freq,
        nrep=nrep,
        rng=rng,
        dim=dim,
        axis=axis,
        rep_dim=rep_dim,
        mom_dims=mom_dims,
        mom_ndim=mom_ndim,
        paired=paired,
        dtype=dtype,
    )

    if is_xarray(data):
        axis, dim = select_axis_dim(data, axis=axis, dim=dim, mom_ndim=mom_ndim)
        mom_dims = validate_mom_dims(mom_dims, mom_ndim, data)

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _resample_data,
            data,
            freq,
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
            **get_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                on_missing_core_dim=on_missing_core_dim,
                dask="parallelized",
                output_sizes={rep_dim: _select_nrep(freq, nrep, rep_dim)},
                output_dtypes=dtype or np.float64,
            ),
        )

        if not move_axis_to_end and is_dataarray(data):
            dims_order = (*data.dims[:axis], rep_dim, *data.dims[axis + 1 :])  # type: ignore[union-attr, misc,index,operator]
            xout = xout.transpose(*dims_order)
        return xout

    # Numpy
    axis, data = prepare_data_for_reduction(
        data,
        axis=axis,
        mom_ndim=mom_ndim,
        dtype=None,
        recast=False,
        move_axis_to_end=move_axis_to_end,
    )
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
    freq: ArrayLike | xr.DataArray | DataT | None = ...,
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
    freq: ArrayLike | None = ...,
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
    freq: ArrayLike | None = ...,
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
    freq: ArrayLike | None = ...,
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
    freq: ArrayLike | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ResampleValsKwargs],
) -> NDArrayAny: ...


# ** public api
@docfiller.decorate
def resample_vals(  # noqa: PLR0913
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    mom: Moments,
    weight: ArrayLike | xr.DataArray | DataT | None = None,
    freq: ArrayLike | xr.DataArray | DataT | None = None,
    nrep: int | None = None,
    rng: RngTypes | None = None,
    paired: bool = True,
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
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Resample data according to frequency table.

    Parameters
    ----------
    {x_genarray}
    {y_genarray}
    {freq_xarray}
    {nrep_optional}
    {rng}
    {paired}
    {mom}
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
    mom, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
    weight = 1.0 if weight is None else weight
    dtype = select_dtype(x, out=out, dtype=dtype)

    freq_validated = randsamp_freq(
        data=x,
        freq=freq,
        nrep=nrep,
        rng=rng,
        dim=dim,
        axis=axis,
        rep_dim=rep_dim,
        paired=paired,
        dtype=dtype,
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
            freq_validated,
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
            **get_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                on_missing_core_dim=on_missing_core_dim,
                dask="parallelized",
                output_sizes={
                    rep_dim: _select_nrep(freq, nrep, rep_dim),
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
            xout = xout.transpose(..., *dims_order)

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
        freq=freq_validated,  # type: ignore[arg-type]
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


@docfiller.decorate
def jackknife_data(  # noqa: PLR0913
    data: ArrayLike | DataT,
    data_reduced: ArrayLike | DataT | None = None,
    *,
    mom_ndim: Mom_NDim = 1,
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
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
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
    {out}
    {dtype}
    {casting}
    {order}
    {parallel}
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
    >>> data = cmomy.default_rng(0).random((4, 3))
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
            keep_attrs=keep_attrs,
            dtype=dtype,
            casting=casting,
            order=order,
            mom_dims=mom_dims,
            on_missing_core_dim=on_missing_core_dim,
            apply_ufunc_kwargs=apply_ufunc_kwargs,
            use_reduce=False,
        )
    elif not is_xarray(data_reduced):
        data_reduced = asarray_maybe_recast(data_reduced, dtype=dtype, recast=False)

    if is_xarray(data):
        axis, dim = select_axis_dim(data, axis=axis, dim=dim, mom_ndim=mom_ndim)
        core_dims = [dim, *validate_mom_dims(mom_dims, mom_ndim, data)]

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
            **get_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                on_missing_core_dim=on_missing_core_dim,
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
    on_missing_core_dim: MissingCoreDimOptions = "copy",
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
    mom, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
    weight = 1.0 if weight is None else weight
    dtype = select_dtype(x, out=out, dtype=dtype)
    if data_reduced is None:
        from .reduction import reduce_vals

        data_reduced = reduce_vals(  # type: ignore[type-var, misc]
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
            on_missing_core_dim=on_missing_core_dim,
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
            **get_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                on_missing_core_dim=on_missing_core_dim,
                dask="parallelized",
                output_sizes=dict(zip(mom_dims, mom_to_mom_shape(mom))),
                output_dtypes=dtype or np.float64,
            ),
        )

        if not move_axis_to_end and is_dataarray(x):
            xout = xout.transpose(..., *x.dims, *mom_dims)
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

"""
Routine to perform resampling (:mod:`cmomy.resample`)
=====================================================
"""

from __future__ import annotations

from itertools import starmap
from math import prod
from typing import TYPE_CHECKING, cast

import numpy as np
import xarray as xr

from .random import validate_rng
from .utils import axis_expand_broadcast

if TYPE_CHECKING:
    from typing import Any, Hashable, Literal, Sequence

    from numpy.typing import ArrayLike, DTypeLike

    from .typing import ArrayOrder, Mom_NDim, Moments, MyNDArray, XvalStrict


##############################################################################
# resampling
###############################################################################
def freq_to_indices(
    freq: MyNDArray, shuffle: bool = True, rng: np.random.Generator | None = None
) -> MyNDArray:
    """
    Convert a frequency array to indices array.

    This creates an "indices" array that is compatible with "freq" array.
    Note that by default, the indices for a single sample (along output[k, :])
    are randomly shuffled.  If you pass `shuffle=False`, then the output will
    be something like [[0,0,..., 1,1,..., 2,2, ...]].
    """
    indices_all: list[MyNDArray] = []

    for f in freq:
        indices = np.concatenate(list(starmap(np.repeat, enumerate(f))))
        indices_all.append(indices)

    out = np.array(indices_all)

    if shuffle:
        rng = validate_rng(rng)
        rng.shuffle(out, axis=1)

    return out


def indices_to_freq(indices: MyNDArray) -> MyNDArray:
    """Convert indices to frequency array."""
    from ._lib.resample import (
        randsamp_freq_indices,  # pyright: ignore[reportUnknownVariableType]
    )

    freq = np.zeros_like(indices)
    randsamp_freq_indices(indices, freq)

    return freq


def random_indices(
    nrep: int, ndat: int, rng: np.random.Generator | None = None, replace: bool = True
) -> MyNDArray:
    """
    Create indices for random resampling (bootstrapping).

    Parameters
    ----------
    nrep, ndat : int
        Number of repetitions (`nrep`) and size of data to sample along (ndat).
    rng : Generator, optional
        Optional Generator.
    replace : bool, default=True
        Whether to allow replacement.
    """
    rng = validate_rng(rng)
    return rng.choice(ndat, size=(nrep, ndat), replace=replace)


def random_freq(
    nrep: int, ndat: int, rng: np.random.Generator | None = None, replace: bool = True
) -> MyNDArray:
    """
    Create frequencies for random resampling (bootstrapping).

    See Also
    --------
    random_indices
    """
    return indices_to_freq(
        random_indices(nrep=nrep, ndat=ndat, rng=rng, replace=replace)
    )


def _validate_resample_array(
    x: ArrayLike,
    nrep: int | None,
    ndat: int | None,
    check: bool = True,
    name: str = "array",
) -> MyNDArray:
    x = np.asarray(x, dtype=np.int64)
    if check:
        if x.ndim != 2:
            msg = f"{name}.ndim={x.ndim} != 2"
            raise ValueError(msg)

        if nrep is not None and x.shape[0] != nrep:
            msg = f"{name}.shape[0]={x.shape[0]} != {nrep}"
            raise ValueError(msg)

        if ndat is None:
            raise ValueError

        if x.shape[1] != ndat:
            msg = f"{name} has wrong ndat"
            raise ValueError(msg)
    return x


def randsamp_freq(
    nrep: int | None = None,
    ndat: int | None = None,
    indices: ArrayLike | None = None,
    freq: ArrayLike | None = None,
    check: bool = False,
    rng: np.random.Generator | None = None,
) -> MyNDArray:
    """
    Produce a random sample for bootstrapping.

    Parameters
    ----------
    ndat : int, optional
        data dimension ndat
    freq : array-like,  optional
        `shape=(nrep, ndat)`.
        If passed, use this frequency array.
        overrides ndat
    indices : array-like, , optional
        `shape=(nrep, ndat)`.
        if passed and `freq` is `None`, construct frequency
        array from this indices array

    nrep : int, optional
        if `freq` and `indices` are `None`, construct
        sample with this number of repetitions
    indices : array-like, optional
        if passed, build frequency table based on this sampling.
        shape = (nrep, ndat)
    freq : array-like, optional
        if passed, use this frequency array
    check : bool, default=False
        if `check` is `True`, then check `freq` and `indices` against `ndat` and `nrep`

    Returns
    -------
    output : ndarray
        Frequency table of shape ``(nrep, ndat)``.
    """
    if freq is not None:
        freq = _validate_resample_array(
            freq, name="freq", nrep=nrep, ndat=ndat, check=check
        )

    elif indices is not None:
        indices = _validate_resample_array(
            indices, name="indices", nrep=nrep, ndat=ndat, check=check
        )
        freq = indices_to_freq(indices)

    elif nrep is not None and ndat is not None:
        freq = random_freq(nrep=nrep, ndat=ndat, rng=validate_rng(rng), replace=True)

    else:
        msg = "must specify freq, indices, or nrep and ndat"
        raise ValueError(msg)

    return freq


def resample_data(  # noqa: PLR0914
    data: ArrayLike,
    freq: ArrayLike,
    mom: Moments,
    axis: int = 0,
    dtype: DTypeLike | None = None,
    order: ArrayOrder = None,
    parallel: bool = True,
    out: MyNDArray | None = None,
) -> MyNDArray:
    """
    Resample data according to frequency table.

    Parameters
    ----------
    data : array-like
        central mom array to be resampled
    freq : array-like
        frequency array with shape (nrep, data.shape[axis])
    mom : int or array-like
        if int or length 1, then data contains central mom.
        if length is 2, then data contains central comoments
    axis : int, default=0
        axis to reduce along
    parallel : bool
        option to run jitted pusher in parallel.
    dtype, order : object
        options to :func:`numpy.asarray`
    out : ndarray, optional
      optional output array.

    Returns
    -------
    output : array
        output shape is `(nrep,) + shape + mom`, where shape is
        the shape of data less axis, and mom is the shape of the resulting mom.
    """
    from ._lib.resample import factory_resample_data

    if isinstance(mom, int):
        mom = (mom,)

    # check inputs
    data = np.asarray(data, dtype=dtype, order=order)
    freq = np.asarray(freq, dtype=np.int64, order=order)

    if dtype is None:
        dtype = data.dtype

    nrep, ndat = freq.shape
    ndim = data.ndim - len(mom)
    if axis < 0:
        axis += ndim
    if not 0 <= axis < ndim:  # pragma: no cover
        raise ValueError

    if axis != 0:
        data = np.moveaxis(data, axis, 0)

    shape: tuple[int, ...] = data.shape[1 : -len(mom)]
    mom_shape = tuple(x + 1 for x in mom)

    target_shape = (ndat, *shape, *mom_shape)
    if data.shape != target_shape:
        msg = f"{data.shape=} != {target_shape}"
        raise ValueError(msg)

    # output
    out_shape = (nrep,) + data.shape[1:]
    if out is None:
        out = np.empty(out_shape, dtype=dtype)
    elif out.shape != out_shape:
        msg = f"{out.shape=} != {out_shape}"
        raise ValueError(msg)
    else:  # make sure out is in correct order
        out = np.asarray(out, dtype=dtype, order="C")

    meta_reshape: tuple[int, ...]
    meta_reshape = mom_shape if shape == () else (prod(shape), *mom_shape)

    data_reshape = (ndat, *meta_reshape)
    out_reshape = (nrep, *meta_reshape)

    datar = data.reshape(data_reshape)
    outr = out.reshape(out_reshape)

    resample = factory_resample_data(
        cov=len(mom) > 1, vec=len(shape) > 0, parallel=parallel
    )

    outr.fill(0.0)
    resample(datar, freq, outr)

    return outr.reshape(out.shape)


def resample_vals(  # noqa: C901,PLR0912,PLR0914,PLR0915
    x: XvalStrict,
    freq: MyNDArray,
    mom: Moments,
    axis: int = 0,
    w: MyNDArray | None = None,
    mom_ndim: Mom_NDim | None = None,
    broadcast: bool = False,
    dtype: DTypeLike | None = None,
    order: ArrayOrder = None,
    parallel: bool = True,
    out: MyNDArray | None = None,
) -> MyNDArray:
    """Resample data according to frequency table."""
    from ._lib.resample import factory_resample_vals

    if isinstance(mom, int):
        mom = (mom,)
    elif not isinstance(mom, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError

    if mom_ndim is None:
        mom_ndim = len(mom)  # type: ignore[assignment]
    if len(mom) != mom_ndim:
        raise ValueError
    mom_shape = tuple(x + 1 for x in mom)

    if mom_ndim == 1:
        y = None
    elif mom_ndim == 2:
        x, y = x
    else:
        msg = "only mom_ndim <= 2 supported"
        raise ValueError(msg)

    # check input data
    freq = np.asarray(freq, dtype=np.int64)
    nrep, ndat = freq.shape

    x = np.asarray(x, dtype=dtype, order=order)
    if dtype is None:
        dtype = x.dtype
    if w is None:
        w = np.ones_like(x)
    else:
        w = axis_expand_broadcast(
            w, shape=x.shape, axis=axis, roll=False, dtype=dtype, order=order
        )

    if y is not None:
        y = axis_expand_broadcast(
            y,
            shape=x.shape,
            axis=axis,
            roll=False,
            broadcast=broadcast,
            dtype=dtype,
            order=order,
        )

    if axis != 0:
        x = np.moveaxis(x, axis, 0)
        w = np.moveaxis(w, axis, 0)
        if y is not None:
            y = np.moveaxis(y, axis, 0)

    if len(x) != ndat:
        raise ValueError

    # output
    shape = x.shape[1:]
    out_shape = (nrep, *shape, *mom_shape)
    if out is None:
        out = np.empty(out_shape, dtype=dtype)
    elif out.shape != out_shape:
        raise ValueError
    else:
        out = np.asarray(out, dtype=dtype, order="C")

    # reshape
    data_reshape: tuple[int, ...] = (ndat,)
    out_reshape: tuple[int, ...] = (nrep,)
    if shape == ():
        pass
    else:
        meta_reshape: tuple[int, ...] = (prod(shape),)
        data_reshape = data_reshape + meta_reshape
        out_reshape = out_reshape + meta_reshape
    out_reshape = out_reshape + mom_shape

    xr = x.reshape(data_reshape)
    wr = w.reshape(data_reshape)
    outr = out.reshape(out_reshape)
    # if cov:
    #     yr = y.reshape(data_reshape)

    resample = factory_resample_vals(
        cov=y is not None, vec=len(shape) > 0, parallel=parallel
    )
    outr.fill(0.0)
    if y is not None:
        yr = y.reshape(data_reshape)
        resample(wr, xr, yr, freq, outr)
    else:
        resample(wr, xr, freq, outr)

    return outr.reshape(out.shape)


# TODO(wpk): add coverage for these
def bootstrap_confidence_interval(  # pragma: no cover
    distribution: MyNDArray,
    stats_val: MyNDArray | Literal["percentile", "mean", "median"] | None = "mean",
    axis: int = 0,
    alpha: float = 0.05,
    style: Literal[None, "delta", "pm"] = None,
    **kwargs: Any,
) -> MyNDArray:
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
    stats_val: MyNDArray | Literal["percentile", "mean", "median"] | None = "mean",
    axis: int = 0,
    dim: Hashable | None = None,
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
        axis = cast(int, x.get_axis_num(dim))  # type: ignore[redundant-cast, unused-ignore]
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

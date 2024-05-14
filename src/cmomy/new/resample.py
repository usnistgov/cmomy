"""
Routine to perform resampling (:mod:`cmomy.resample`)
=====================================================
"""

from __future__ import annotations

from itertools import starmap

# if TYPE_CHECKING:
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cmomy.new.utils import (
    parallel_heuristic,
    prepare_data_for_reduction,
    prepare_values_for_reduction,
    raise_if_wrong_shape,
    validate_mom_and_mom_ndim,
    validate_mom_ndim,
)

from .docstrings import docfiller
from .random import validate_rng

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from .typing import ArrayOrder, Mom_NDim, Moments, NDArrayAny, NDArrayInt
    from .typing import T_FloatDType as T_Float
    from .typing import T_IntDType as T_Int


##############################################################################
# resampling
###############################################################################
@docfiller.decorate
def freq_to_indices(
    freq: NDArray[T_Int], shuffle: bool = True, rng: np.random.Generator | None = None
) -> NDArray[T_Int]:
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
        indices = np.concatenate(list(starmap(np.repeat, enumerate(f))))
        indices_all.append(indices)

    out = np.array(indices_all, dtype=freq.dtype)

    if shuffle:
        rng = validate_rng(rng)
        rng.shuffle(out, axis=1)

    return out


def indices_to_freq(indices: NDArray[T_Int], ndat: int | None = None) -> NDArray[T_Int]:
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


@docfiller.decorate
def randsamp_freq(
    ndat: int,
    nrep: int | None = None,
    nsamp: int | None = None,
    indices: ArrayLike | None = None,
    freq: ArrayLike | None = None,
    check: bool = False,
    rng: np.random.Generator | None = None,
) -> NDArrayInt:
    """
    Produce a random sample for bootstrapping.

    In order, the return will be one of ``freq``, frequencies from ``indices`` or
    new sample from :func:`random_freq`.

    Parameters
    ----------
    {nrep}
    {ndat}
    {nsamp}
    {freq}
    {indices}
    check : bool, default=False
        if `check` is `True`, then check `freq` and `indices` against `ndat` and `nrep`

    Returns
    -------
    freq : ndarray
        Frequency array.

    See Also
    --------
    random_freq
    indices_to_freq
    """
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


def _check_freq(freq: NDArrayAny, ndat: int) -> None:
    if freq.shape[1] != ndat:
        msg = f"{freq.shape[1]=} != {ndat=}"
        raise ValueError(msg)


@docfiller.decorate
def resample_data(
    data: NDArray[T_Float],
    freq: NDArrayInt,
    mom_ndim: Mom_NDim,
    axis: int = -1,
    *,
    order: ArrayOrder = None,
    parallel: bool | None = True,
    out: NDArrayAny | None = None,
) -> NDArray[T_Float]:
    """
    Resample data according to frequency table.

    Parameters
    ----------
    data : array-like
        central mom array to be resampled
    {freq}
    {mom}
    {order}
    {parallel}
    {out}

    Returns
    -------
    out : ndarray
        Resampled central moments. ``out.shape = (..., shape[axis-1], shape[axis+1], ..., nrep, mom0, ...)``,
        where ``shape = data.shape`` and ``nrep = freq.shape[0]``.
    """
    mom_ndim = validate_mom_ndim(mom_ndim)
    _check_freq(freq, data.shape[axis])

    data = prepare_data_for_reduction(data, axis=axis, mom_ndim=mom_ndim, order=order)

    from ._lib.factory import factory_resample_data

    _resample = factory_resample_data(
        mom_ndim=mom_ndim, parallel=parallel_heuristic(parallel, data.size * mom_ndim)
    )

    if out is not None:
        return _resample(data, freq, out)
    return _resample(data, freq)


@docfiller.decorate
def resample_vals(
    x: NDArray[T_Float],
    *y: ArrayLike,
    mom: Moments,
    freq: NDArrayInt,
    weight: ArrayLike | None = None,
    axis: int = -1,
    order: ArrayOrder = None,
    parallel: bool | None = None,
    out: NDArray[T_Float] | None = None,
) -> NDArray[T_Float]:
    """
    Resample data according to frequency table.

    Parameters
    ----------
    x : ndarray
        Value to analyze
    *y:  array-like, optional.
        Second value needed if len(mom)==2.
    {freq}
    {mom}
    {axis}
    {weight}
    {order}
    {parallel}
    {out}

    Returns
    -------
    out : ndarray
        Resampled Central moments array. ``out.shape = (...,shape[axis-1], shape[axis+1], ..., nrep, mom0, ...)``
        where ``shape = args[0].shape``. and ``nrep = freq.shape[0]``.
    """
    mom_validated, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
    _check_freq(freq, x.shape[axis])

    weight = 1.0 if weight is None else weight
    x0, *x1, w = prepare_values_for_reduction(
        x, *y, weight, axis=axis, order=order, narrays=mom_ndim + 1
    )

    val_shape: tuple[int, ...] = np.broadcast_shapes(*(_.shape for _ in (x0, *x1, w)))[
        :-1
    ]
    mom_shape: tuple[int, ...] = tuple(m + 1 for m in mom_validated)
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

    factory_resample_vals(  # type: ignore[call-arg, type-var]
        mom_ndim=mom_ndim,
        parallel=parallel_heuristic(parallel, x0.size * mom_ndim),
    )(x0, *x1, w, freq, out)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue]

    return out

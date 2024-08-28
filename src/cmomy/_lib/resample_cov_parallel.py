"""Vectorized pushers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numba as nb

from . import _push_cov as _push
from .decorators import myguvectorize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from cmomy.core.typing import FloatT

_PARALLEL = True  # Auto generated from resample_cov.py
_vectorize = partial(myguvectorize, parallel=_PARALLEL)


@_vectorize(
    "(replicate,sample),(sample,mom0,mom1)-> (replicate,mom0,mom1)",
    [
        (nb.float32[:, :], nb.float32[:, :, :], nb.float32[:, :, :]),
        (nb.float64[:, :], nb.float64[:, :, :], nb.float64[:, :, :]),
    ],
    writable=None,
)
def resample_data_fromzero(
    freq: NDArray[FloatT],
    data: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    nrep, nsamp = freq.shape

    assert data.shape[1:] == out.shape[1:]
    assert data.shape[0] == nsamp
    assert out.shape[0] == nrep
    assert data.shape[1:] == out.shape[1:]

    out[...] = 0.0

    for irep in range(nrep):
        first_nonzero = nsamp
        for isamp in range(nsamp):
            f = freq[irep, isamp]
            if f != 0:
                first_nonzero = isamp
                out[irep, :, :] = data[isamp, :, :]
                out[irep, 0, 0] *= f
                break

        for isamp in range(first_nonzero + 1, nsamp):
            f = freq[irep, isamp]
            if f != 0:
                _push.push_data_scale(data[isamp, ...], f, out[irep, ...])


@_vectorize(
    "(replicate,mom0,mom1),(replicate,sample),(sample),(sample),(sample)",
    [
        (
            nb.float32[:, :, :],
            nb.float32[:, :],
            nb.float32[:],
            nb.float32[:],
            nb.float32[:],
        ),
        (
            nb.float64[:, :, :],
            nb.float64[:, :],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
        ),
    ],
    writable=None,
)
def resample_vals(
    out: NDArray[FloatT],
    freq: NDArray[FloatT],
    x0: NDArray[FloatT],
    w: NDArray[FloatT],
    x1: NDArray[FloatT],
) -> None:
    nrep, nsamp = freq.shape

    assert len(w) == nsamp
    assert len(x0) == nsamp
    assert len(x1) == nsamp
    assert out.shape[0] == nrep

    for irep in range(freq.shape[0]):
        for isamp in range(freq.shape[1]):
            f = freq[irep, isamp]
            if f == 0:
                continue
            _push.push_val(x0[isamp], x1[isamp], w[isamp] * f, out[irep, ...])


# * Jackknife resampling
# We take advantage of the ability to add and subtract moments here.
# Instead of out[i, ...] = reduce(data[[0, 1, ..., i-1, i+1, ...], ...]) (which is order n^2)
# we use out[i, ...] = reduce(data[:, ...]) - data[i, ...]  (which is order n or 2n)
@_vectorize(
    "(mom0,mom1),(sample,mom0,mom1) -> (sample,mom0,mom1)",
    [
        (nb.float32[:, :], nb.float32[:, :, :], nb.float32[:, :, :]),
        (nb.float64[:, :], nb.float64[:, :, :], nb.float64[:, :, :]),
    ],
    writable=None,
)
def jackknife_data_fromzero(
    data_reduced: NDArray[FloatT], data: NDArray[FloatT], out: NDArray[FloatT]
) -> None:
    assert data.shape[1:] == data_reduced.shape

    # initially fill with data_reduced
    out[:, ...] = data_reduced

    for isamp in range(data.shape[0]):
        # out[isamp, ...] = data_reduced - data[isamp, ....]
        _push.push_data_scale(data[isamp, ...], -1.0, out[isamp, ...])


@_vectorize(
    "(mom0,mom1),(sample),(sample),(sample) -> (sample,mom0,mom1)",
    [
        (
            nb.float32[:, :],
            nb.float32[:],
            nb.float32[:],
            nb.float32[:],
            nb.float32[:, :, :],
        ),
        (
            nb.float64[:, :],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:, :, :],
        ),
    ],
    writable=None,
)
def jackknife_vals_fromzero(
    data_reduced: NDArray[FloatT],
    x0: NDArray[FloatT],
    w: NDArray[FloatT],
    x1: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    nsamp = len(x0)
    assert len(x1) == nsamp
    assert len(w) == nsamp
    assert out.shape[0] == nsamp
    assert data_reduced.shape == out.shape[1:]

    # initially fill with data_reduced
    out[:, ...] = data_reduced

    for isamp in range(nsamp):
        _push.push_val(x0[isamp], x1[isamp], -w[isamp], out[isamp, ...])

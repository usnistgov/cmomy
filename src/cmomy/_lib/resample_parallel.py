"""Vectorized pushers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numba as nb

from . import _push
from .decorators import myguvectorize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from cmomy.core.typing import FloatT

_PARALLEL = True  # Auto generated from resample.py
_vectorize = partial(myguvectorize, parallel=_PARALLEL)


@_vectorize(
    "(replicate,sample), (sample,mom) -> (replicate,mom)",
    [
        (nb.float32[:, :], nb.float32[:, :], nb.float32[:, :]),
        (nb.float64[:, :], nb.float64[:, :], nb.float64[:, :]),
    ],
    writable=None,
)
def resample_data_fromzero(
    freq: NDArray[FloatT],
    data: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    nrep, nsamp = freq.shape

    assert data.shape[0] == nsamp
    assert out.shape[0] == nrep
    assert data.shape[1] == out.shape[1]

    out[...] = 0.0

    for irep in range(nrep):
        first_nonzero = nsamp
        for isamp in range(nsamp):
            f = freq[irep, isamp]
            if f != 0:
                first_nonzero = isamp
                out[irep, :] = data[isamp, :]
                out[irep, 0] *= f
                break

        for isamp in range(first_nonzero + 1, nsamp):
            f = freq[irep, isamp]
            if f != 0:
                _push.push_data_scale(data[isamp, ...], f, out[irep, ...])


@_vectorize(
    "(replicate,mom),(replicate,sample),(sample),(sample)",
    [
        (
            nb.float32[:, :],
            nb.float32[:, :],
            nb.float32[:],
            nb.float32[:],
        ),
        (
            nb.float64[:, :],
            nb.float64[:, :],
            nb.float64[:],
            nb.float64[:],
        ),
    ],
    writable=None,
)
def resample_vals(
    out: NDArray[FloatT],
    freq: NDArray[FloatT],
    x: NDArray[FloatT],
    w: NDArray[FloatT],
) -> None:
    nrep, nsamp = freq.shape

    assert len(w) == nsamp
    assert x.shape == w.shape
    assert out.shape[0] == nrep

    for irep in range(nrep):
        for isamp in range(nsamp):
            f = freq[irep, isamp]
            if f == 0:
                continue
            _push.push_val(x[isamp], w[isamp] * f, out[irep, ...])


# * Jackknife resampling
# We take advantage of the ability to add and subtract moments here.
# Instead of out[i, ...] = reduce(data[[0, 1, ..., i-1, i+1, ...], ...]) (which is order n^2)
# we use out[i, ...] = reduce(data[:, ...]) - data[i, ...]  (which is order n or 2n)
@_vectorize(
    "(mom),(sample,mom)-> (sample,mom)",
    [
        (nb.float32[:], nb.float32[:, :], nb.float32[:, :]),
        (nb.float64[:], nb.float64[:, :], nb.float64[:, :]),
    ],
    writable=None,
)
def jackknife_data_fromzero(
    data_reduced: NDArray[FloatT],
    data: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    assert data.shape[1:] == data_reduced.shape

    # initially fill with data_reduced
    out[:, ...] = data_reduced

    for isamp in range(data.shape[0]):
        # out[isamp, ...] = data_reduced - data[isamp, ....]
        _push.push_data_scale(data[isamp, ...], -1.0, out[isamp, ...])


@_vectorize(
    "(mom),(sample),(sample) ->(sample,mom)",
    [
        (nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:, :]),
        (nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:, :]),
    ],
    writable=None,
)
def jackknife_vals_fromzero(
    data_reduced: NDArray[FloatT],
    x: NDArray[FloatT],
    w: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    nsamp = len(x)
    assert len(w) == nsamp
    assert out.shape[0] == nsamp
    assert data_reduced.shape == out.shape[1:]

    # initially fill with data_reduced
    out[:, ...] = data_reduced

    for isamp in range(nsamp):
        _push.push_val(x[isamp], -w[isamp], out[isamp, ...])

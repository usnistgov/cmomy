"""Vectorized pushers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numba as nb

from . import _push
from .decorators import myguvectorize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..typing import FloatT

# NOTE: The parallel implementation is quite similar to the old
# way of doing things (with reshape and njit), but the serial is slower.
# Still, this is way simpler...


_PARALLEL = True  # Auto generated from resample.py
_vectorize = partial(myguvectorize, parallel=_PARALLEL)
# _jit = partial(myjit, parallel=_PARALLEL)


@_vectorize(
    "(sample,mom),(replicate,sample) -> (replicate,mom)",
    [
        (nb.float32[:, :], nb.float32[:, :], nb.float32[:, :]),
        (nb.float64[:, :], nb.float64[:, :], nb.float64[:, :]),
    ],
    writable=None,
)
def resample_data_fromzero(
    data: NDArray[FloatT], freq: NDArray[FloatT], out: NDArray[FloatT]
) -> None:
    nrep, nsamp = freq.shape

    assert data.shape[1:] == out.shape[1:]
    assert data.shape[0] == nsamp
    assert out.shape[0] == nrep

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
    "(sample),(sample),(replicate,sample),(replicate,mom)",
    [
        (nb.float32[:], nb.float32[:], nb.float32[:, :], nb.float32[:, :]),
        (nb.float64[:], nb.float64[:], nb.float64[:, :], nb.float64[:, :]),
    ],
)
def resample_vals(
    x: NDArray[FloatT],
    w: NDArray[FloatT],
    freq: NDArray[FloatT],
    out: NDArray[FloatT],
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


# * Other routines
# Some of these are faster (especially jitted), but not worth it.
#
# This one can accumulate.  Not sure where this would be needed?
# @_vectorize(
#     "(sample,mom),(replicate,sample),(replicate,mom)",
#     [
#         (nb.float32[:, :], nb.float32[:, :], nb.float32[:, :]),
#         (nb.float64[:, :], nb.float64[:, :], nb.float64[:, :]),
#     ],
# )
# def resample_data(data, freq, out) -> None:
#     nrep, nsamp = freq.shape

#     assert data.shape[1:] == out.shape[1:]
#     assert data.shape[0] == nsamp
#     assert out.shape[0] == nrep

#     for irep in range(nrep):
#         for isamp in range(nsamp):
#             f = freq[irep, isamp]
#             if f == 0:
#                 continue
#             _push.push_data_scale(data[isamp, ...], f, out[irep, ...])


# @_jit(
#     # out[samp, value, mom], freq[rep, samp], out[rep, value, mom]
#     [
#         (nb.float32[:, :, :], nb.float32[:, :], nb.float32[:, :, :]),
#         (nb.float64[:, :, :], nb.float64[:, :], nb.float64[:, :, :]),
#     ],
# )
# def resample_data_jit(data, freq, out):
#     nrep, nsamp = freq.shape
#     nval = data.shape[1]

#     assert data.shape[1:] == out.shape[1:]
#     assert data.shape[0] == nsamp
#     assert out.shape[0] == nrep

#     for irep in nb.prange(nrep):
#         for isamp in range(nsamp):
#             f = freq[irep, isamp]
#             if f == 0:
#                 continue
#             for k in range(nval):
#                 _push.push_data_scale(data[isamp, k, ...], f, out[irep, k, ...])


# @_jit(
#     # out[samp, value, mom], freq[rep, samp], out[rep, value, mom]
#     [
#         (nb.float32[:, :, :], nb.float32[:, :], nb.float32[:, :, :]),
#         (nb.float64[:, :, :], nb.float64[:, :], nb.float64[:, :, :]),
#     ],
# )
# def resample_data_fromzero_jit(data, freq, out):
#     nrep, nsamp = freq.shape
#     nval = data.shape[1]

#     assert data.shape[1:] == out.shape[1:]
#     assert data.shape[0] == nsamp
#     assert out.shape[0] == nrep

#     out[...] = 0.0

#     for irep in nb.prange(nrep):
#         first_nonzero = nsamp
#         for isamp in range(nsamp):
#             f = freq[irep, isamp]
#             if f != 0:
#                 first_nonzero = isamp
#                 for k in range(nval):
#                     out[irep, k, :] = data[isamp, k, :]
#                     out[irep, k, 0] *= f
#                 break

#         for isamp in range(first_nonzero + 1, nsamp):
#             f = freq[irep, isamp]
#             if f == 0:
#                 continue
#             for k in range(nval):
#                 _push.push_data_scale(data[isamp, k, ...], f, out[irep, k, ...])


# @_jit(
#     # x[samp, value], w[samp, value], freq[rep, samp], out[rep, value, mom]
#     [
#         (nb.float32[:, :], nb.float32[:, :], nb.float32[:, :], nb.float32[:, :, :]),
#         (nb.float64[:, :], nb.float64[:, :], nb.float64[:, :], nb.float64[:, :, :]),
#     ],
# )
# def resample_vals_jit(x, w, freq, out):
#     nrep, nsamp = freq.shape
#     nval = w.shape[1]

#     assert w.shape == x.shape
#     assert w.shape[0] == nsamp
#     assert out.shape[0] == nrep
#     assert out.shape[1] == nval

#     for irep in nb.prange(nrep):
#         for isamp in range(nsamp):
#             f = freq[irep, isamp]
#             if f == 0:
#                 continue
#             for k in range(nval):
#                 _push.push_val(x[isamp, k], w[isamp, k] * f, out[irep, k, ...])

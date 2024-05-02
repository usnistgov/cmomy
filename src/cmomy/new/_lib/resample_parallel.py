# mypy: disable-error-code="no-untyped-call,no-untyped-def"
"""Vectorized pushers."""

from __future__ import annotations

from functools import partial

import numba as nb

from . import _push
from .decorators import myguvectorize

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
def resample_data_fromzero(other, freq, data) -> None:
    nrep, nsamp = freq.shape

    assert other.shape[1:] == data.shape[1:]
    assert other.shape[0] == nsamp
    assert data.shape[0] == nrep

    data[...] = 0.0

    for irep in range(nrep):
        first_nonzero = nsamp
        for isamp in range(nsamp):
            f = freq[irep, isamp]
            if f != 0:
                first_nonzero = isamp
                data[irep, :] = other[isamp, :]
                data[irep, 0] *= f
                break

        for isamp in range(first_nonzero + 1, nsamp):
            f = freq[irep, isamp]
            if f != 0:
                _push.push_data_scale(other[isamp, ...], f, data[irep, ...])


@_vectorize(
    "(sample),(sample),(replicate,sample),(replicate,mom)",
    [
        (nb.float32[:], nb.float32[:], nb.float32[:, :], nb.float32[:, :]),
        (nb.float64[:], nb.float64[:], nb.float64[:, :], nb.float64[:, :]),
    ],
)
def resample_vals(w, x, freq, data) -> None:
    nrep, nsamp = freq.shape

    assert len(w) == nsamp
    assert x.shape == w.shape
    assert data.shape[0] == nrep

    for irep in range(nrep):
        for isamp in range(nsamp):
            f = freq[irep, isamp]
            if f == 0:
                continue
            _push.push_val(w[isamp] * f, x[isamp], data[irep, ...])


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
# def resample_data(other, freq, data) -> None:
#     nrep, nsamp = freq.shape

#     assert other.shape[1:] == data.shape[1:]
#     assert other.shape[0] == nsamp
#     assert data.shape[0] == nrep

#     for irep in range(nrep):
#         for isamp in range(nsamp):
#             f = freq[irep, isamp]
#             if f == 0:
#                 continue
#             _push.push_data_scale(other[isamp, ...], f, data[irep, ...])


# @_jit(
#     # data[samp, value, mom], freq[rep, samp], out[rep, value, mom]
#     [
#         (nb.float32[:, :, :], nb.float32[:, :], nb.float32[:, :, :]),
#         (nb.float64[:, :, :], nb.float64[:, :], nb.float64[:, :, :]),
#     ],
# )
# def resample_data_jit(other, freq, data):
#     nrep, nsamp = freq.shape
#     nval = other.shape[1]

#     assert other.shape[1:] == data.shape[1:]
#     assert other.shape[0] == nsamp
#     assert data.shape[0] == nrep

#     for irep in nb.prange(nrep):
#         for isamp in range(nsamp):
#             f = freq[irep, isamp]
#             if f == 0:
#                 continue
#             for k in range(nval):
#                 _push.push_data_scale(other[isamp, k, ...], f, data[irep, k, ...])


# @_jit(
#     # data[samp, value, mom], freq[rep, samp], out[rep, value, mom]
#     [
#         (nb.float32[:, :, :], nb.float32[:, :], nb.float32[:, :, :]),
#         (nb.float64[:, :, :], nb.float64[:, :], nb.float64[:, :, :]),
#     ],
# )
# def resample_data_fromzero_jit(other, freq, data):
#     nrep, nsamp = freq.shape
#     nval = other.shape[1]

#     assert other.shape[1:] == data.shape[1:]
#     assert other.shape[0] == nsamp
#     assert data.shape[0] == nrep

#     data[...] = 0.0

#     for irep in nb.prange(nrep):
#         first_nonzero = nsamp
#         for isamp in range(nsamp):
#             f = freq[irep, isamp]
#             if f != 0:
#                 first_nonzero = isamp
#                 for k in range(nval):
#                     data[irep, k, :] = other[isamp, k, :]
#                     data[irep, k, 0] *= f
#                 break

#         for isamp in range(first_nonzero + 1, nsamp):
#             f = freq[irep, isamp]
#             if f == 0:
#                 continue
#             for k in range(nval):
#                 _push.push_data_scale(other[isamp, k, ...], f, data[irep, k, ...])


# @_jit(
#     # w[samp, value], x[samp, value], freq[rep, samp], out[rep, value, mom]
#     [
#         (nb.float32[:, :], nb.float32[:, :], nb.float32[:, :], nb.float32[:, :, :]),
#         (nb.float64[:, :], nb.float64[:, :], nb.float64[:, :], nb.float64[:, :, :]),
#     ],
# )
# def resample_vals_jit(w, x, freq, data):
#     nrep, nsamp = freq.shape
#     nval = w.shape[1]

#     assert w.shape == x.shape
#     assert w.shape[0] == nsamp
#     assert data.shape[0] == nrep
#     assert data.shape[1] == nval

#     for irep in nb.prange(nrep):
#         for isamp in range(nsamp):
#             f = freq[irep, isamp]
#             if f == 0:
#                 continue
#             for k in range(nval):
#                 _push.push_val(w[isamp, k] * f, x[isamp, k], data[irep, k, ...])

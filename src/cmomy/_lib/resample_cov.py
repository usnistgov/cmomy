"""Vectorized pushers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numba as nb

from . import _push_cov as _push
from .decorators import myguvectorize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..typing import FloatT

_PARALLEL = False
_vectorize = partial(myguvectorize, parallel=_PARALLEL)
# _jit = partial(myjit, parallel=_PARALLEL)


# NOTE: The parallel implementation is quite similar to the old
# way of doing things (with reshape and njit), but the serial is slower.
# Still, this is way simpler...


@_vectorize(
    "(sample,mom0,mom1),(replicate,sample) -> (replicate,mom0,mom1)",
    [
        (nb.float32[:, :, :], nb.float32[:, :], nb.float32[:, :, :]),
        (nb.float64[:, :, :], nb.float64[:, :], nb.float64[:, :, :]),
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
                out[irep, :, :] = data[isamp, :, :]
                out[irep, 0, 0] *= f
                break

        for isamp in range(first_nonzero + 1, nsamp):
            f = freq[irep, isamp]
            if f != 0:
                _push.push_data_scale(data[isamp, ...], f, out[irep, ...])


@_vectorize(
    "(sample),(sample),(sample),(replicate,sample),(replicate,mom0,mom1)",
    [
        (
            nb.float32[:],
            nb.float32[:],
            nb.float32[:],
            nb.float32[:, :],
            nb.float32[:, :, :],
        ),
        (
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:, :],
            nb.float64[:, :, :],
        ),
    ],
)
def resample_vals(
    x0: NDArray[FloatT],
    x1: NDArray[FloatT],
    w: NDArray[FloatT],
    freq: NDArray[FloatT],
    out: NDArray[FloatT],
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


# * Other routines

# @_vectorize(
#     "(sample,mom0,mom1),(replicate,sample),(replicate,mom0,mom1)",
#     [
#         (nb.float32[:, :, :], nb.float32[:, :], nb.float32[:, :, :]),
#         (nb.float64[:, :, :], nb.float64[:, :], nb.float64[:, :, :]),
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
#     # "(sample,val,mom0, mom1),(replicate,sample),(replicate,val,mom0,mom1)",
#     [
#         (nb.float32[:, :, :, :], nb.float32[:, :], nb.float32[:, :, :, :]),
#         (nb.float64[:, :, :, :], nb.float64[:, :], nb.float64[:, :, :, :]),
#     ],
# )
# def resample_data_jit(data, freq, out) -> None:
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
#     # "(sample,val,mom0, mom1),(replicate,sample),(replicate,val,mom0,mom1)",
#     [
#         (nb.float32[:, :, :, :], nb.float32[:, :], nb.float32[:, :, :, :]),
#         (nb.float64[:, :, :, :], nb.float64[:, :], nb.float64[:, :, :, :]),
#     ],
# )
# def resample_data_fromzero_jit(data, freq, out) -> None:
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
#                     out[irep, k, :, :] = data[isamp, k, :, :]
#                     out[irep, k, 0, 0] *= f
#                 break

#         for isamp in range(first_nonzero + 1, nsamp):
#             f = freq[irep, isamp]
#             if f == 0:
#                 continue
#             for k in range(nval):
#                 _push.push_data_scale(data[isamp, k, ...], f, out[irep, k, ...])


# @_jit(
#     # w[samp, value], x0[samp, value], x1[samp, value], freq[rep, samp], out[rep, value, mom0, mom1]
#     [
#         (
#             nb.float32[:, :],
#             nb.float32[:, :],
#             nb.float32[:, :],
#             nb.float32[:, :],
#             nb.float32[:, :, :, :],
#         ),
#         (
#             nb.float64[:, :],
#             nb.float64[:, :],
#             nb.float64[:, :],
#             nb.float64[:, :],
#             nb.float64[:, :, :, :],
#         ),
#     ],
# )
# def resample_vals_jit(x0, x1, w, freq, out):
#     nrep, nsamp = freq.shape
#     nval = w.shape[1]

#     assert w.shape == x0.shape
#     assert x0.shape == x1.shape
#     assert w.shape[0] == nsamp
#     assert out.shape[0] == nrep
#     assert out.shape[1] == nval

#     for irep in nb.prange(nrep):
#         for isamp in range(nsamp):
#             f = freq[irep, isamp]
#             if f == 0:
#                 continue
#             for k in range(nval):
#                 _push.push_val(
#                     x0[isamp, k], x1[isamp, k], w[isamp, k] * f, out[irep, k, ...]
#                 )

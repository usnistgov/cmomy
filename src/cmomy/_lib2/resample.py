# mypy: disable-error-code="no-untyped-call,no-untyped-def"
"""Vectorized pushers."""

from __future__ import annotations

from functools import partial

import numba as nb

from . import pushscalar
from .utils import myguvectorize, myjit

# NOTE: The parallel implementation is quite similar to the old
# way of doing things (with reshape and njit), but the serial is slower.
# Still, this is way simpler...

_PARALLEL = False
_vectorize = partial(myguvectorize, parallel=_PARALLEL)
_jit = partial(myjit, parallel=_PARALLEL)


@_vectorize(
    "(sample,mom),(replicate,sample),(replicate,mom)",
    [
        (nb.float32[:, :], nb.float32[:, :], nb.float32[:, :]),
        (nb.float64[:, :], nb.float64[:, :], nb.float64[:, :]),
    ],
)
def resample_data(other, freq, data) -> None:
    nrep, nsamp = freq.shape

    assert other.shape[1:] == data.shape[1:]
    assert other.shape[0] == nsamp
    assert data.shape[0] == nrep

    for irep in range(nrep):
        for isamp in range(nsamp):
            f = freq[irep, isamp]
            if f == 0:
                continue
            pushscalar.push_data_scale(other[isamp, ...], f, data[irep, ...])


@_jit(
    # data[samp, value, mom], freq[rep, samp], out[rep, value, mom]
    [
        (nb.float32[:, :, :], nb.float32[:, :], nb.float32[:, :, :]),
        (nb.float64[:, :, :], nb.float64[:, :], nb.float64[:, :, :]),
    ],
)
def resample_data_jit(other, freq, data):
    nrep, nsamp = freq.shape
    nval = other.shape[1]

    assert other.shape[1:] == data.shape[1:]
    assert other.shape[0] == nsamp
    assert data.shape[0] == nrep

    for irep in nb.prange(nrep):
        for isamp in range(nsamp):
            f = freq[irep, isamp]
            if f == 0:
                continue
            for k in range(nval):
                pushscalar.push_data_scale(other[isamp, k, ...], f, data[irep, k, ...])


# Using fromzero doesn't help all that much...
# @_vectorize(
#     "(sample,mom),(replicate,sample),(replicate,mom)",
#     [
#         (nb.float32[:, :], nb.float32[:, :], nb.float32[:, :]),
#         (nb.float64[:, :], nb.float64[:, :], nb.float64[:, :]),
#     ],
# )
# def resample_data_fromzero(other, freq, data) -> None:
#     nrep, nsamp = freq.shape

#     assert other.shape[1:] == data.shape[1:]
#     assert other.shape[0] == nsamp
#     assert data.shape[0] == nrep

#     for irep in range(nrep):
#         first_nonzero = nsamp
#         for isamp in range(nsamp):
#             f = freq[irep, isamp]
#             if f != 0:
#                 first_nonzero = isamp
#                 data[irep, :] = other[isamp, :]
#                 data[irep, 0] *= f
#                 break

#         for isamp in range(first_nonzero + 1, nsamp):
#             f = freq[irep, isamp]
#             if f != 0:
#                 pushscalar.push_data_scale(other[isamp, ...], f, data[irep, ...])


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
#                 pushscalar.push_data_scale(other[isamp, k, ...], f, data[irep, k, ...])


@_vectorize(
    "(sample),(sample),(replicate,sample),(replicate,mom)",
    [
        (nb.float32[:], nb.float32[:], nb.float32[:, :], nb.float32[:, :]),
        (nb.float64[:], nb.float32[:], nb.float64[:, :], nb.float64[:, :]),
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
            pushscalar.push_val(w[isamp] * f, x[isamp], data[irep, ...])


# Old (dumb) way of doing things.  Faster is many cases
@_jit(
    # w[samp, value], x[samp, value], freq[rep, samp], out[rep, value, mom]
    [
        (nb.float32[:, :], nb.float32[:, :], nb.float32[:, :], nb.float32[:, :, :]),
        (nb.float64[:, :], nb.float64[:, :], nb.float64[:, :], nb.float64[:, :, :]),
    ],
)
def resample_vals_jit(w, x, freq, data):
    nrep, nsamp = freq.shape
    nval = w.shape[1]

    assert w.shape == x.shape
    assert w.shape[0] == nsamp
    assert data.shape[0] == nrep
    assert data.shape[1] == nval

    for irep in nb.prange(nrep):
        for isamp in range(nsamp):
            f = freq[irep, isamp]
            if f == 0:
                continue
            for k in range(nval):
                pushscalar.push_val(w[isamp, k] * f, x[isamp, k], data[irep, k, ...])


# * indexed
@_vectorize(
    "(sample,mom),(index),(group),(group),(index),(group,mom)",
    [
        (
            nb.float32[:, :],
            nb.int64[:],
            nb.int64[:],
            nb.int64[:],
            nb.float32[:],
            nb.float32[:, :],
        ),
        (
            nb.float64[:, :],
            nb.int64[:],
            nb.int64[:],
            nb.int64[:],
            nb.float64[:],
            nb.float64[:, :],
        ),
    ],
)
def reduceindexed_data(other, index, group_start, group_end, scales, data) -> None:
    ngroups = len(group_start)
    for group in range(ngroups):
        start = group_start[group]
        end = group_end[group]
        if end > start:
            # assume start from zero
            s = index[start]
            f = scales[start]
            data[group, :] = other[s, :]
            data[group, 0] *= f

            for i in range(start + 1, end):
                s = index[i]
                f = scales[i]
                pushscalar.push_data_scale(other[s, ...], f, data[group, ...])


@_vectorize(
    "(sample,mom),(index),(group),(group),(index),(group,mom)",
    [
        (
            nb.float32[:, :],
            nb.int64[:],
            nb.int64[:],
            nb.int64[:],
            nb.float32[:],
            nb.float32[:, :],
        ),
        (
            nb.float64[:, :],
            nb.int64[:],
            nb.int64[:],
            nb.int64[:],
            nb.float64[:],
            nb.float64[:, :],
        ),
    ],
)
def reduceindexed_data2(other, index, group_start, group_end, scales, data) -> None:
    ngroups = len(group_start)
    for group in range(ngroups):
        start = group_start[group]
        end = group_end[group]
        if end > start:
            for i in range(start, end):
                s = index[i]
                f = scales[i]
                pushscalar.push_data_scale(other[s, ...], f, data[group, ...])


@_jit(
    # other[sample,val,mom],index[index],start[group],end[group],scale[index],data[group,val,mom]
    [
        (
            nb.float32[:, :, :],
            nb.int64[:],
            nb.int64[:],
            nb.int64[:],
            nb.float32[:],
            nb.float32[:, :, :],
        ),
        (
            nb.float64[:, :, :],
            nb.int64[:],
            nb.int64[:],
            nb.int64[:],
            nb.float64[:],
            nb.float64[:, :, :],
        ),
    ],
)
def reduceindexed_data_jit(other, index, group_start, group_end, scales, data) -> None:
    ngroups = len(group_start)
    nval = other.shape[1]

    assert other.shape[1:] == data.shape[1:]
    assert data.shape[0] == ngroups

    for group in nb.prange(ngroups):
        start = group_start[group]
        end = group_end[group]
        if end > start:
            s = index[start]
            f = scales[start]
            for k in range(nval):
                data[group, k, :] = other[s, k, :]
                data[group, k, 0] *= f

            for i in range(start + 1, end):
                s = index[i]
                f = scales[i]
                for k in range(nval):
                    pushscalar.push_data_scale(other[s, k, ...], f, data[group, k, ...])

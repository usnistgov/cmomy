# mypy: disable-error-code="no-untyped-call,no-untyped-def"
"""Vectorized pushers."""

from __future__ import annotations

from functools import partial

import numba as nb

from . import _push_cov as _push
from .decorators import myguvectorize, myjit

_PARALLEL = True  # Auto generated from indexed_cov.py
_vectorize = partial(myguvectorize, parallel=_PARALLEL)
_jit = partial(myjit, parallel=_PARALLEL)


@_vectorize(
    "(sample,mom0,mom1),(sample),(group,mom0,mom1)",
    [
        (
            nb.float32[:, :, :],
            nb.int64[:],
            nb.float32[:, :, :],
        ),
        (
            nb.float64[:, :, :],
            nb.int64[:],
            nb.float64[:, :, :],
        ),
    ],
)
def reduce_data_grouped(other, group_idx, data) -> None:
    assert other.shape[1:] == data.shape[1:]
    assert group_idx.max() < data.shape[0]
    for s in range(other.shape[0]):
        group = group_idx[s]
        if group >= 0:
            _push.push_data(other[s, ...], data[group, ...])


@_vectorize(
    "(sample,mom0,mom1),(index),(group),(group),(index) -> (group,mom0,mom1)",
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
    writable=None,
)
def reduce_data_indexed_fromzero(
    other, index, group_start, group_end, scale, data
) -> None:
    ngroup = len(group_start)

    assert other.shape[1:] == data.shape[1:]
    assert index.shape == scale.shape
    assert len(group_end) == ngroup
    assert data.shape[0] == ngroup

    data[...] = 0.0

    for group in range(ngroup):
        start = group_start[group]
        end = group_end[group]
        if end > start:
            # assume start from zero
            s = index[start]
            f = scale[start]
            data[group, :, :] = other[s, :, :]
            data[group, 0, 0] *= f

            for i in range(start + 1, end):
                s = index[i]
                f = scale[i]
                _push.push_data_scale(other[s, ...], f, data[group, ...])


# * Other routines
# @_vectorize(
#     "(sample,mom0,mom1),(index),(group),(group),(index),(group,mom0,mom1)",
#     [
#         (
#             nb.float32[:, :, :],
#             nb.int64[:],
#             nb.int64[:],
#             nb.int64[:],
#             nb.float32[:],
#             nb.float32[:, :, :],
#         ),
#         (
#             nb.float64[:, :, :],
#             nb.int64[:],
#             nb.int64[:],
#             nb.int64[:],
#             nb.float64[:],
#             nb.float64[:, :, :],
#         ),
#     ],
# )
# def reduce_data_indexed(other, index, group_start, group_end, scale, data) -> None:
#     ngroup = len(group_start)

#     assert other.shape[1:] == data.shape[1:]
#     assert index.shape == scale.shape
#     assert len(group_end) == ngroup
#     assert data.shape[0] == ngroup

#     for group in range(ngroup):
#         start = group_start[group]
#         end = group_end[group]
#         if end > start:
#             for i in range(start, end):
#                 s = index[i]
#                 f = scale[i]
#                 _push.push_data_scale(other[s, ...], f, data[group, ...])


# @_jit(
#     # other[sample,val,mom0,mom1],index[index],start[group],end[group],scale[index],data[group,val,mom0,mom1]
#     [
#         (
#             nb.float32[:, :, :, :],
#             nb.int64[:],
#             nb.int64[:],
#             nb.int64[:],
#             nb.float32[:],
#             nb.float32[:, :, :, :],
#         ),
#         (
#             nb.float64[:, :, :, :],
#             nb.int64[:],
#             nb.int64[:],
#             nb.int64[:],
#             nb.float64[:],
#             nb.float64[:, :, :, :],
#         ),
#     ],
# )
# def reduce_data_indexed_jit(other, index, group_start, group_end, scale, data) -> None:
#     ngroup = len(group_start)
#     nval = other.shape[1]

#     assert other.shape[1:] == data.shape[1:]
#     assert index.shape == scale.shape
#     assert len(group_end) == ngroup
#     assert data.shape[0] == ngroup

#     for group in nb.prange(ngroup):
#         start = group_start[group]
#         end = group_end[group]
#         if end > start:
#             s = index[start]
#             f = scale[start]
#             for k in range(nval):
#                 data[group, k, :, :] = other[s, k, :, :]
#                 data[group, k, 0, 0] *= f

#             for i in range(start + 1, end):
#                 s = index[i]
#                 f = scale[i]
#                 for k in range(nval):
#                     _push.push_data_scale(other[s, k, ...], f, data[group, k, ...])

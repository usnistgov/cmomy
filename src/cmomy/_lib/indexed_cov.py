"""Vectorized pushers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numba as nb

from . import _push_cov as _push
from .decorators import myguvectorize, myjit

if TYPE_CHECKING:
    from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..typing import FloatT, NDArrayInt


_PARALLEL = False
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
def reduce_data_grouped(
    data: NDArray[FloatT], group_idx: NDArrayInt, out: NDArray[FloatT]
) -> None:
    assert data.shape[1:] == out.shape[1:]
    assert group_idx.max() < out.shape[0]
    for s in range(data.shape[0]):
        group = group_idx[s]
        if group >= 0:
            _push.push_data(data[s, ...], out[group, ...])


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
    data: NDArray[FloatT],
    index: NDArrayInt,
    group_start: NDArrayInt,
    group_end: NDArrayInt,
    scale: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    ngroup = len(group_start)

    assert data.shape[1:] == out.shape[1:]
    assert index.shape == scale.shape
    assert len(group_end) == ngroup
    assert out.shape[0] == ngroup

    out[...] = 0.0

    for group in range(ngroup):
        start = group_start[group]
        end = group_end[group]
        if end > start:
            # assume start from zero
            s = index[start]
            f = scale[start]
            out[group, :, :] = data[s, :, :]
            out[group, 0, 0] *= f

            for i in range(start + 1, end):
                s = index[i]
                f = scale[i]
                _push.push_data_scale(data[s, ...], f, out[group, ...])


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
# def reduce_data_indexed(data, index, group_start, group_end, scale, out) -> None:
#     ngroup = len(group_start)

#     assert data.shape[1:] == out.shape[1:]
#     assert index.shape == scale.shape
#     assert len(group_end) == ngroup
#     assert out.shape[0] == ngroup

#     for group in range(ngroup):
#         start = group_start[group]
#         end = group_end[group]
#         if end > start:
#             for i in range(start, end):
#                 s = index[i]
#                 f = scale[i]
#                 _push.push_data_scale(data[s, ...], f, out[group, ...])


# @_jit(
#     # data[sample,val,mom0,mom1],index[index],start[group],end[group],scale[index],out[group,val,mom0,mom1]
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
# def reduce_data_indexed_jit(data, index, group_start, group_end, scale, out) -> None:
#     ngroup = len(group_start)
#     nval = data.shape[1]

#     assert data.shape[1:] == out.shape[1:]
#     assert index.shape == scale.shape
#     assert len(group_end) == ngroup
#     assert out.shape[0] == ngroup

#     for group in nb.prange(ngroup):
#         start = group_start[group]
#         end = group_end[group]
#         if end > start:
#             s = index[start]
#             f = scale[start]
#             for k in range(nval):
#                 out[group, k, :, :] = data[s, k, :, :]
#                 out[group, k, 0, 0] *= f

#             for i in range(start + 1, end):
#                 s = index[i]
#                 f = scale[i]
#                 for k in range(nval):
#                     _push.push_data_scale(data[s, k, ...], f, out[group, k, ...])

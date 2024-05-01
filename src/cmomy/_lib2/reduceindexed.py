# mypy: disable-error-code="no-untyped-call,no-untyped-def"
"""Vectorized pushers."""

from __future__ import annotations

from functools import partial

import numba as nb

from . import pushscalar
from .utils import myguvectorize, myjit

_PARALLEL = False
_vectorize = partial(myguvectorize, parallel=_PARALLEL)
_jit = partial(myjit, parallel=_PARALLEL)


@_vectorize(
    "(sample,mom),(index),(group),(group),(index) -> (group,mom)",
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
    writable=None,
)
def reduceindexed_data(other, index, group_start, group_end, scales, data) -> None:
    ngroup = len(group_start)

    assert other.shape[1:] == data.shape[1:]
    assert index.shape == scales.shape
    assert len(group_end) == ngroup
    assert data.shape[0] == ngroup

    data[...] = 0.0

    for group in range(ngroup):
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
    "(sample,mom),(index),(group),(group),(index) -> (group,mom)",
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
    writable=None,
)
def reduceindexed_data2(other, index, group_start, group_end, scales, data) -> None:
    ngroup = len(group_start)

    assert other.shape[1:] == data.shape[1:]
    assert index.shape == scales.shape
    assert len(group_end) == ngroup
    assert data.shape[0] == ngroup

    data[...] = 0.0
    for group in range(ngroup):
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
    ngroup = len(group_start)
    nval = other.shape[1]

    assert other.shape[1:] == data.shape[1:]
    assert index.shape == scales.shape
    assert len(group_end) == ngroup
    assert data.shape[0] == ngroup

    for group in nb.prange(ngroup):
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

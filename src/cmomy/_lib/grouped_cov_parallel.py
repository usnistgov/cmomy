"""Vectorized pushers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numba as nb

from . import _push_cov as _push
from .decorators import myguvectorize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from cmomy.core.typing import FloatT, NDArrayInt


_PARALLEL = True  # Auto generated from grouped_cov.py
_vectorize = partial(myguvectorize, parallel=_PARALLEL)


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
    data: NDArray[FloatT],
    group_idx: NDArrayInt,
    out: NDArray[FloatT],
) -> None:
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


@_vectorize(
    "(group,mom0,mom1),(sample),(sample),(sample),(sample)",
    [
        (
            nb.float32[:, :, :],
            nb.int64[:],
            nb.float32[:],
            nb.float32[:],
            nb.float32[:],
        ),
        (
            nb.float64[:, :, :],
            nb.int64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
        ),
    ],
)
def reduce_vals_grouped(
    out: NDArray[FloatT],
    group_idx: NDArrayInt,
    x0: NDArray[FloatT],
    w: NDArray[FloatT],
    x1: NDArray[FloatT],
) -> None:
    assert group_idx.max() < out.shape[0]
    for s in range(x0.shape[0]):
        group = group_idx[s]
        if group >= 0:
            _push.push_val(x0[s], x1[s], w[s], out[group, ...])


@_vectorize(
    "(group,mom0,mom1),(index),(group),(group),(index),(sample),(sample),(sample)",
    [
        (
            nb.float32[:, :, :],
            nb.int64[:],
            nb.int64[:],
            nb.int64[:],
            nb.float32[:],
            nb.float32[:],
            nb.float32[:],
            nb.float32[:],
        ),
        (
            nb.float64[:, :, :],
            nb.int64[:],
            nb.int64[:],
            nb.int64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
        ),
    ],
)
def reduce_vals_indexed_fromzero(
    out: NDArray[FloatT],
    index: NDArrayInt,
    group_start: NDArrayInt,
    group_end: NDArrayInt,
    scale: NDArray[FloatT],
    x0: NDArray[FloatT],
    w: NDArray[FloatT],
    x1: NDArray[FloatT],
) -> None:
    ngroup = len(group_start)
    out[...] = 0.0

    for group in range(ngroup):
        start = group_start[group]
        end = group_end[group]
        if end > start:
            # assume start from zero
            s = index[start]
            f = scale[start]
            for i in range(start, end):
                s = index[i]
                f = scale[i]
                _push.push_val(x0[s], x1[s], w[s] * f, out[group, ...])

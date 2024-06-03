# mypy: disable-error-code="no-untyped-call,no-untyped-def"
"""Routines to perform resampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Hashable

    from ..typing import NDArrayAny

import numpy as np
from numba import prange

from .pushers import (
    push_datas_scale_fromzero_indexed,
    push_datas_scale_fromzero_indexed_cov,
    push_datas_scale_fromzero_indexed_cov_vec,
    push_datas_scale_fromzero_indexed_vec,
)
from .utils import myjit


@myjit()
def freq_to_index_start_end_scales(
    freq: NDArrayAny,
) -> tuple[NDArrayAny, NDArrayAny, NDArrayAny, NDArrayAny]:
    ngroup = freq.shape[0]
    ndat = freq.shape[1]

    start = np.empty(ngroup, dtype=np.int64)
    end = np.empty(ngroup, dtype=np.int64)

    size = np.count_nonzero(freq)

    index = np.empty(size, dtype=np.int64)
    scale = np.empty(size, dtype=np.int64)

    idx = 0
    for group in range(ngroup):
        start[group] = idx
        count = 0
        for k in range(ndat):
            f = freq[group, k]
            if f > 0:
                index[idx] = k
                scale[idx] = f
                idx += 1
                count += 1
        end[group] = start[group] + count

    return index, start, end, scale


@myjit()
def _index_push_datas_scale(others, index, group_start, group_end, scales, out) -> None:
    ngroups = len(group_start)
    for group in prange(ngroups):
        start = group_start[group]
        end = group_end[group]
        if end > start:
            push_datas_scale_fromzero_indexed(
                out[group, ...], others, index[start:end], scales[start:end]
            )


@myjit(parallel=True)
def _index_push_datas_scale_parallel(
    others, index, group_start, group_end, scales, out
) -> None:
    ngroups = len(group_start)
    for group in prange(ngroups):
        start = group_start[group]
        end = group_end[group]
        if end > start:
            push_datas_scale_fromzero_indexed(
                out[group, ...], others, index[start:end], scales[start:end]
            )


@myjit()
def _index_push_datas_scale_vec(
    others, index, group_start, group_end, scales, out
) -> None:
    ngroups = len(group_start)
    for group in prange(ngroups):
        start = group_start[group]
        end = group_end[group]
        if end > start:
            push_datas_scale_fromzero_indexed_vec(
                out[group, ...], others, index[start:end], scales[start:end]
            )


@myjit(parallel=True)
def _index_push_datas_scale_vec_parallel(
    others, index, group_start, group_end, scales, out
) -> None:
    ngroups = len(group_start)
    for group in prange(ngroups):
        start = group_start[group]
        end = group_end[group]
        if end > start:
            push_datas_scale_fromzero_indexed_vec(
                out[group, ...], others, index[start:end], scales[start:end]
            )


@myjit()
def _index_push_datas_scale_cov(
    others, index, group_start, group_end, scales, out
) -> None:
    ngroups = len(group_start)
    for group in prange(ngroups):
        start = group_start[group]
        end = group_end[group]
        if end > start:
            push_datas_scale_fromzero_indexed_cov(
                out[group, ...], others, index[start:end], scales[start:end]
            )


@myjit(parallel=True)
def _index_push_datas_scale_cov_parallel(
    others, index, group_start, group_end, scales, out
) -> None:
    ngroups = len(group_start)
    for group in prange(ngroups):
        start = group_start[group]
        end = group_end[group]
        if end > start:
            push_datas_scale_fromzero_indexed_cov(
                out[group, ...], others, index[start:end], scales[start:end]
            )


@myjit()
def _index_push_datas_scale_cov_vec(
    others, index, group_start, group_end, scales, out
) -> None:
    ngroups = len(group_start)
    for group in prange(ngroups):
        start = group_start[group]
        end = group_end[group]
        if end > start:
            push_datas_scale_fromzero_indexed_cov_vec(
                out[group, ...], others, index[start:end], scales[start:end]
            )


@myjit(parallel=True)
def _index_push_datas_scale_cov_vec_parallel(
    others, index, group_start, group_end, scales, out
) -> None:
    ngroups = len(group_start)
    for group in prange(ngroups):
        start = group_start[group]
        end = group_end[group]
        if end > start:
            push_datas_scale_fromzero_indexed_cov_vec(
                out[group, ...], others, index[start:end], scales[start:end]
            )


_INDEX_PUSH_DICT: dict[Hashable, Callable[..., None]] = {
    # cov, vec, parallel
    (False, False, False): _index_push_datas_scale,
    (False, False, True): _index_push_datas_scale_parallel,
    (False, True, False): _index_push_datas_scale_vec,
    (False, True, True): _index_push_datas_scale_vec_parallel,
    (True, False, False): _index_push_datas_scale_cov,
    (True, False, True): _index_push_datas_scale_cov_parallel,
    (True, True, False): _index_push_datas_scale_cov_vec,
    (True, True, True): _index_push_datas_scale_cov_vec_parallel,
}


def factory_indexed_push_data(
    cov: bool,
    vec: bool,
    parallel: bool,
) -> Callable[..., None]:
    return _INDEX_PUSH_DICT[cov, vec, parallel]

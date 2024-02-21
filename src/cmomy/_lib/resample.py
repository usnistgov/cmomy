# mypy: disable-error-code="no-untyped-call,no-untyped-def"
"""Routines to perform resampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Hashable

from numba import prange

from .pushers import (
    push_datas_scale,
    push_datas_scale_cov,
    push_datas_scale_cov_vec,
    push_datas_scale_vec,
    push_vals_scale,
    push_vals_scale_cov,
    push_vals_scale_cov_vec,
    push_vals_scale_vec,
)
from .utils import myjit

# --- * Utilities ------------------------------------------------------------------------
# put these here to avoid slow load up


@myjit()
def randsamp_freq_indices(indices, freq) -> None:
    assert freq.shape == indices.shape
    nrep, ndat = freq.shape
    for r in range(nrep):
        for d in range(ndat):
            idx = indices[r, d]
            freq[r, idx] += 1


# NOTE: this is all due to closures not being cache-able with numba
# used to use the following
#
# @lru_cache(10)
# def _factory_resample(push_datas_scale, fastmath=True, parallel=False):
#     @njit(fastmath=fastmath, parallel=parallel)
#     def resample(data, freq, out):
#         nrep = freq.shape[0]
#         for irep in prange(nrep):
#             push_datas_scale(out[irep, ...], data, freq[irep, ...])

#     return resample

# @lru_cache(10)
# def _factory_resample_vals(push_vals_scale, fastmath=True, parallel=False):
#     @njit(fastmath=fastmath, parallel=parallel)
#     def resample(W, X, freq, out):
#         nrep = freq.shape[0]
#         for irep in prange(nrep):
#             push_vals_scale(out[irep, ...], W, X, freq[irep, ...])

#     return resample


# @lru_cache(10)
# def _factory_resample_vals_cov(push_vals_scale, fastmath=True, parallel=False):
#     @njit(fastmath=fastmath, parallel=parallel)
#     def resample(W, X, Y, freq, out):
#         nrep = freq.shape[0]
#         for irep in prange(nrep):
#             push_vals_scale(out[irep, ...], W, X, Y, freq[irep, ...])

#     return resample


######################################################################
# resample data
# mom/scalar
@myjit()
def _resample_data(data, freq, out) -> None:
    nrep = freq.shape[0]
    for irep in prange(nrep):
        push_datas_scale(out[irep, ...], data, freq[irep, ...])


@myjit(parallel=True)
def _resample_data_parallel(data, freq, out) -> None:
    nrep = freq.shape[0]
    for irep in prange(nrep):
        push_datas_scale(out[irep, ...], data, freq[irep, ...])


# mom/vector
@myjit()
def _resample_data_vec(data, freq, out) -> None:
    nrep = freq.shape[0]
    for irep in prange(nrep):
        push_datas_scale_vec(out[irep, ...], data, freq[irep, ...])


@myjit(parallel=True)
def _resample_data_vec_parallel(data, freq, out) -> None:
    nrep = freq.shape[0]
    for irep in prange(nrep):
        push_datas_scale_vec(out[irep, ...], data, freq[irep, ...])


# cov/vector
@myjit()
def _resample_data_cov(data, freq, out) -> None:
    nrep = freq.shape[0]
    for irep in prange(nrep):
        push_datas_scale_cov(out[irep, ...], data, freq[irep, ...])


@myjit(parallel=True)
def _resample_data_cov_parallel(data, freq, out) -> None:
    nrep = freq.shape[0]
    for irep in prange(nrep):
        push_datas_scale_cov(out[irep, ...], data, freq[irep, ...])


# cov/vector
@myjit()
def _resample_data_cov_vec(data, freq, out) -> None:
    nrep = freq.shape[0]
    for irep in prange(nrep):
        push_datas_scale_cov_vec(out[irep, ...], data, freq[irep, ...])


@myjit(parallel=True)
def _resample_data_cov_vec_parallel(data, freq, out) -> None:
    nrep = freq.shape[0]
    for irep in prange(nrep):
        push_datas_scale_cov_vec(out[irep, ...], data, freq[irep, ...])


_RESAMPLE_DATA_DICT: dict[Hashable, Callable[..., Any]] = {
    # cov, vec, parallel
    (False, False, False): _resample_data,
    (False, False, True): _resample_data_parallel,
    (False, True, False): _resample_data_vec,
    (False, True, True): _resample_data_vec_parallel,
    (True, False, False): _resample_data_cov,
    (True, False, True): _resample_data_cov_parallel,
    (True, True, False): _resample_data_cov_vec,
    (True, True, True): _resample_data_cov_vec_parallel,
}


def factory_resample_data(cov: bool, vec: bool, parallel: bool) -> Callable[..., Any]:
    """Get resampler functions(s)."""
    return _RESAMPLE_DATA_DICT[cov, vec, parallel]


######################################################################
# resample values


# mom/scalar
@myjit()
def _resample_vals(W, X, freq, out) -> None:
    nrep = freq.shape[0]
    for irep in prange(nrep):
        push_vals_scale(out[irep, ...], W, X, freq[irep, ...])


@myjit(parallel=True)
def _resample_vals_parallel(W, X, freq, out) -> None:
    nrep = freq.shape[0]
    for irep in prange(nrep):
        push_vals_scale(out[irep, ...], W, X, freq[irep, ...])


# mom/vec
@myjit()
def _resample_vals_vec(W, X, freq, out) -> None:
    nrep = freq.shape[0]
    for irep in prange(nrep):
        push_vals_scale_vec(out[irep, ...], W, X, freq[irep, ...])


@myjit(parallel=True)
def _resample_vals_vec_parallel(W, X, freq, out) -> None:
    nrep = freq.shape[0]
    for irep in prange(nrep):
        push_vals_scale_vec(out[irep, ...], W, X, freq[irep, ...])


# cov/scalar
@myjit()
def _resample_vals_cov(W, X, Y, freq, out) -> None:
    nrep = freq.shape[0]
    for irep in prange(nrep):
        push_vals_scale_cov(out[irep, ...], W, X, Y, freq[irep, ...])


@myjit(parallel=True)
def _resample_vals_cov_parallel(W, X, Y, freq, out) -> None:
    nrep = freq.shape[0]
    for irep in prange(nrep):
        push_vals_scale_cov(out[irep, ...], W, X, Y, freq[irep, ...])


# cov/vec
@myjit()
def _resample_vals_cov_vec(W, X, Y, freq, out) -> None:
    nrep = freq.shape[0]
    for irep in prange(nrep):
        push_vals_scale_cov_vec(out[irep, ...], W, X, Y, freq[irep, ...])


@myjit(parallel=True)
def _resample_vals_cov_vec_parallel(W, X, Y, freq, out) -> None:
    nrep = freq.shape[0]
    for irep in prange(nrep):
        push_vals_scale_cov_vec(out[irep, ...], W, X, Y, freq[irep, ...])


_RESAMPLE_VALS_DICT: dict[Hashable, Callable[..., Any]] = {
    # cov, vec, parallel
    (False, False, False): _resample_vals,
    (False, False, True): _resample_vals_parallel,
    (False, True, False): _resample_vals_vec,
    (False, True, True): _resample_vals_vec_parallel,
    (True, False, False): _resample_vals_cov,
    (True, False, True): _resample_vals_cov_parallel,
    (True, True, False): _resample_vals_cov_vec,
    (True, True, True): _resample_vals_cov_vec_parallel,
}


def factory_resample_vals(cov: bool, vec: bool, parallel: bool) -> Callable[..., Any]:
    """Get resample vals functions."""
    return _RESAMPLE_VALS_DICT[cov, vec, parallel]

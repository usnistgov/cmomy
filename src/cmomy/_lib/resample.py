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
    # fromzero
    push_datas_scale_fromzero,
    push_datas_scale_fromzero_cov,
    push_datas_scale_fromzero_cov_vec,
    push_datas_scale_fromzero_vec,
    push_datas_scale_vec,
    # vals
    push_vals_scale,
    push_vals_scale_cov,
    push_vals_scale_cov_vec,
    push_vals_scale_vec,
)
from .utils import myjit

# --- * Utilities ------------------------------------------------------------------------
# put these here to avoid slow load up


@myjit()
def randsamp_indices_to_freq(indices, freqs) -> None:
    # allowed to pass in different number of samples than ndat.
    nrep, ndat = freqs.shape

    assert indices.shape[0] == nrep
    assert indices.max() < ndat

    nsamp = indices.shape[1]

    for r in range(nrep):
        for d in range(nsamp):
            idx = indices[r, d]
            freqs[r, idx] += 1


# NOTE: this is all due to closures not being cache-able with numba
# used to use the following
#
# @lru_cache(10)
# def _factory_resample(push_datas_scale, fastmath=True, parallel=False):
#     @njit(fastmath=fastmath, parallel=parallel)
#     def resample(data, freqs, out):
#         nrep = freqs.shape[0]
#         for irep in prange(nrep):
#             push_datas_scale(out[irep, ...], data, freqs[irep, ...])

#     return resample

# @lru_cache(10)
# def _factory_resample_vals(push_vals_scale, fastmath=True, parallel=False):
#     @njit(fastmath=fastmath, parallel=parallel)
#     def resample(w_s, x_s, freqs, out):
#         nrep = freqs.shape[0]
#         for irep in prange(nrep):
#             push_vals_scale(out[irep, ...], w_s, x_s, freqs[irep, ...])

#     return resample


# @lru_cache(10)
# def _factory_resample_vals_cov(push_vals_scale, fastmath=True, parallel=False):
#     @njit(fastmath=fastmath, parallel=parallel)
#     def resample(w_s, x_s, Y, freqs, out):
#         nrep = freqs.shape[0]
#         for irep in prange(nrep):
#             push_vals_scale(out[irep, ...], w_s, x_s, Y, freqs[irep, ...])

#     return resample


# * resample data
# ** mom/scalar
@myjit()
def _resample_data(data, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_datas_scale(out[irep, ...], data, freqs[irep, ...])


@myjit(parallel=True)
def _resample_data_parallel(data, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_datas_scale(out[irep, ...], data, freqs[irep, ...])


# ** mom/vector
@myjit()
def _resample_data_vec(data, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_datas_scale_vec(out[irep, ...], data, freqs[irep, ...])


@myjit(parallel=True)
def _resample_data_vec_parallel(data, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_datas_scale_vec(out[irep, ...], data, freqs[irep, ...])


# ** cov/vector
@myjit()
def _resample_data_cov(data, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_datas_scale_cov(out[irep, ...], data, freqs[irep, ...])


@myjit(parallel=True)
def _resample_data_cov_parallel(data, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_datas_scale_cov(out[irep, ...], data, freqs[irep, ...])


# ** cov/vector
@myjit()
def _resample_data_cov_vec(data, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_datas_scale_cov_vec(out[irep, ...], data, freqs[irep, ...])


@myjit(parallel=True)
def _resample_data_cov_vec_parallel(data, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_datas_scale_cov_vec(out[irep, ...], data, freqs[irep, ...])


# * From zero data
@myjit()
def _resample_data_fromzero(data, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_datas_scale_fromzero(out[irep, ...], data, freqs[irep, ...])


@myjit(parallel=True)
def _resample_data_fromzero_parallel(data, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_datas_scale_fromzero(out[irep, ...], data, freqs[irep, ...])


# ** mom/vector
@myjit()
def _resample_data_fromzero_vec(data, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_datas_scale_fromzero_vec(out[irep, ...], data, freqs[irep, ...])


@myjit(parallel=True)
def _resample_data_fromzero_vec_parallel(data, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_datas_scale_fromzero_vec(out[irep, ...], data, freqs[irep, ...])


# ** cov/vector
@myjit()
def _resample_data_fromzero_cov(data, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_datas_scale_fromzero_cov(out[irep, ...], data, freqs[irep, ...])


@myjit(parallel=True)
def _resample_data_fromzero_cov_parallel(data, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_datas_scale_fromzero_cov(out[irep, ...], data, freqs[irep, ...])


# ** cov/vector
@myjit()
def _resample_data_fromzero_cov_vec(data, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_datas_scale_fromzero_cov_vec(out[irep, ...], data, freqs[irep, ...])


@myjit(parallel=True)
def _resample_data_fromzero_cov_vec_parallel(data, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_datas_scale_fromzero_cov_vec(out[irep, ...], data, freqs[irep, ...])


_RESAMPLE_DATA_DICT: dict[Hashable, Callable[..., Any]] = {
    # cov, vec, parallel, fromzero
    (False, False, False, False): _resample_data,
    (False, False, True, False): _resample_data_parallel,
    (False, True, False, False): _resample_data_vec,
    (False, True, True, False): _resample_data_vec_parallel,
    (True, False, False, False): _resample_data_cov,
    (True, False, True, False): _resample_data_cov_parallel,
    (True, True, False, False): _resample_data_cov_vec,
    (True, True, True, False): _resample_data_cov_vec_parallel,
    (False, False, False, True): _resample_data_fromzero,
    (False, False, True, True): _resample_data_fromzero_parallel,
    (False, True, False, True): _resample_data_fromzero_vec,
    (False, True, True, True): _resample_data_fromzero_vec_parallel,
    (True, False, False, True): _resample_data_fromzero_cov,
    (True, False, True, True): _resample_data_fromzero_cov_parallel,
    (True, True, False, True): _resample_data_fromzero_cov_vec,
    (True, True, True, True): _resample_data_fromzero_cov_vec_parallel,
}


def factory_resample_data(
    cov: bool, vec: bool, parallel: bool, fromzero: bool
) -> Callable[..., Any]:
    """Get resampler functions(s)."""
    return _RESAMPLE_DATA_DICT[cov, vec, parallel, fromzero]


######################################################################
# resample values


# mom/scalar
@myjit()
def _resample_vals(w_s, x_s, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_vals_scale(out[irep, ...], w_s, x_s, freqs[irep, ...])


@myjit(parallel=True)
def _resample_vals_parallel(w_s, x_s, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_vals_scale(out[irep, ...], w_s, x_s, freqs[irep, ...])


# mom/vec
@myjit()
def _resample_vals_vec(w_s, x_s, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_vals_scale_vec(out[irep, ...], w_s, x_s, freqs[irep, ...])


@myjit(parallel=True)
def _resample_vals_vec_parallel(w_s, x_s, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_vals_scale_vec(out[irep, ...], w_s, x_s, freqs[irep, ...])


# cov/scalar
@myjit()
def _resample_vals_cov(w_s, x_s, Y, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_vals_scale_cov(out[irep, ...], w_s, x_s, Y, freqs[irep, ...])


@myjit(parallel=True)
def _resample_vals_cov_parallel(w_s, x_s, Y, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_vals_scale_cov(out[irep, ...], w_s, x_s, Y, freqs[irep, ...])


# cov/vec
@myjit()
def _resample_vals_cov_vec(w_s, x_s, Y, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_vals_scale_cov_vec(out[irep, ...], w_s, x_s, Y, freqs[irep, ...])


@myjit(parallel=True)
def _resample_vals_cov_vec_parallel(w_s, x_s, Y, freqs, out) -> None:
    nrep = freqs.shape[0]
    for irep in prange(nrep):
        push_vals_scale_cov_vec(out[irep, ...], w_s, x_s, Y, freqs[irep, ...])


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

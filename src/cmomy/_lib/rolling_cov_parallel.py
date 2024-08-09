from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numba as nb
import numpy as np

from . import _push_cov as _push
from .decorators import myguvectorize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from cmomy.core.typing import FloatT

_PARALLEL = True  # Auto generated from rolling_cov.py
_vectorize = partial(myguvectorize, parallel=_PARALLEL)


@_vectorize(
    "(mom0, mom1),(),(),(sample),(sample),(sample) -> (sample, mom0, mom1)",
    [
        (
            nb.float32[:, :],
            nb.int64,
            nb.int64,
            nb.float32[:],
            nb.float32[:],
            nb.float32[:],
            nb.float32[:, :, :],
        ),
        (
            nb.float64[:, :],
            nb.int64,
            nb.int64,
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:, :, :],
        ),
    ],
    writable=None,
)
def rolling_vals(
    data_tmp: NDArray[FloatT],
    window: int,
    min_count: int,
    x0: NDArray[FloatT],
    w: NDArray[FloatT],
    x1: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    nsamp = len(x0)
    data_tmp[...] = 0.0
    count = 0
    min_count = max(min_count, 1)

    for i in range(min(window, nsamp)):
        wi = w[i]
        if wi != 0:
            _push.push_val(x0[i], x1[i], wi, data_tmp)
            count += 1

        if count >= min_count:
            out[i, ...] = data_tmp
        else:
            out[i, ...] = np.nan

    for i in range(window, nsamp):
        i_old = i - window

        wi = w[i]
        wold = w[i_old]

        wi_valid = wi != 0.0
        wold_valid = wold != 0.0

        if wi_valid:
            count += 1
            _push.push_val(x0[i], x1[i], wi, data_tmp)

        if wold_valid:
            count -= 1
            if wold < data_tmp[0, 0]:
                _push.push_val(x0[i_old], x1[i_old], -wold, data_tmp)
            else:
                # special case.  If new weight is ==0, then we have a problem
                # assume we are subtracting the only positive element and are going back to zero
                data_tmp[...] = 0.0

        if count >= min_count:
            out[i, ...] = data_tmp
        else:
            out[i, ...] = np.nan


@_vectorize(
    "(mom0, mom1),(),(),(sample, mom0, mom1) -> (sample, mom0, mom1)",
    [
        (
            nb.float32[:, :],
            nb.int64,
            nb.int64,
            nb.float32[:, :, :],
            nb.float32[:, :, :],
        ),
        (
            nb.float64[:, :],
            nb.int64,
            nb.int64,
            nb.float64[:, :, :],
            nb.float64[:, :, :],
        ),
    ],
    writable=None,
)
def rolling_data(
    data_tmp: NDArray[FloatT],
    window: int,
    min_count: int,
    data: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    nsamp = data.shape[0]
    data_tmp[...] = 0.0
    count = 0
    min_count = max(min_count, 1)

    for i in range(min(window, nsamp)):
        wi = data[i, 0, 0]
        if wi != 0:
            _push.push_data(data[i, ...], data_tmp)
            count += 1

        if count >= min_count:
            out[i, ...] = data_tmp
        else:
            out[i, ...] = np.nan

    for i in range(window, nsamp):
        i_old = i - window

        wi = data[i, 0, 0]
        wold = data[i_old, 0, 0]

        wi_valid = wi != 0.0
        wold_valid = wold != 0.0

        if wi_valid:
            count += 1
            _push.push_data(data[i, ...], data_tmp)
        if wold_valid:
            count -= 1
            if wold < data_tmp[0, 0]:
                _push.push_data_scale(data[i_old, ...], -1.0, data_tmp)
            else:
                # special case.  If new weight is ==0, then we have a problem
                # assume we are subtracting the only positive element and are going back to zero
                data_tmp[...] = 0.0

        if count >= min_count:
            out[i, ...] = data_tmp
        else:
            out[i, ...] = np.nan


@_vectorize(
    "(mom0,mom1),(sample),(),(),(sample),(sample),(sample) -> (sample, mom0,mom1)",
    [
        (
            nb.float32[:, :],
            nb.float32[:],
            nb.boolean,
            nb.int64,
            nb.float32[:],
            nb.float32[:],
            nb.float32[:],
            nb.float32[:, :, :],
        ),
        (
            nb.float64[:, :],
            nb.float64[:],
            nb.boolean,
            nb.int64,
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.float64[:, :, :],
        ),
    ],
    writable=None,
)
def rolling_exp_vals(
    data_tmp: NDArray[FloatT],
    alpha: NDArray[FloatT],
    adjust: bool,
    min_count: int,
    x0: NDArray[FloatT],
    w: NDArray[FloatT],
    x1: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    """Exponential moving accumulation of moments array"""
    nsamp = len(x0)

    count = 0
    data_tmp[...] = 0.0
    old_weight = 0.0

    for i in range(nsamp):
        wi = w[i]
        alphai = alpha[i]
        decay = 1.0 - alphai

        # scale down
        old_weight *= decay
        data_tmp[..., 0, 0] *= decay

        if wi != 0.0:
            count += 1
            scale = wi if adjust else wi * alphai
            _push.push_val(x0[i], x1[i], scale, data_tmp)
            if not adjust:
                old_weight += alphai
                data_tmp[..., 0, 0] /= old_weight
                old_weight = 1.0

        if count >= min_count:
            out[i, ...] = data_tmp
        else:
            out[i, ...] = np.nan


@_vectorize(
    "(mom0,mom1),(sample),(),(),(sample,mom0,mom1) -> (sample, mom0,mom1)",
    [
        (
            nb.float32[:, :],
            nb.float32[:],
            nb.boolean,
            nb.int64,
            nb.float32[:, :, :],
            nb.float32[:, :, :],
        ),
        (
            nb.float64[:, :],
            nb.float64[:],
            nb.boolean,
            nb.int64,
            nb.float64[:, :, :],
            nb.float64[:, :, :],
        ),
    ],
    writable=None,
)
def rolling_exp_data(
    data_tmp: NDArray[FloatT],
    alpha: NDArray[FloatT],
    adjust: bool,
    min_count: int,
    data: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    """Exponential moving accumulation of moments array"""
    nsamp = data.shape[0]

    count = 0
    data_tmp[...] = 0.0
    old_weight = 0.0

    for i in range(nsamp):
        wi = data[i, 0, 0]
        alphai = alpha[i]
        decay = 1.0 - alphai

        # scale down
        old_weight *= decay
        data_tmp[..., 0, 0] *= decay

        if wi != 0.0:
            count += 1
            scale = 1.0 if adjust else alphai
            _push.push_data_scale(data[i, ...], scale, data_tmp)
            if not adjust:
                old_weight += alphai
                data_tmp[..., 0, 0] /= old_weight
                old_weight = 1.0

        if count >= min_count:
            out[i, ...] = data_tmp
        else:
            out[i, ...] = np.nan

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numba as nb
import numpy as np

from . import _push
from .decorators import myguvectorize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..typing import FloatT

_PARALLEL = True  # Auto generated from rolling.py
_vectorize = partial(myguvectorize, parallel=_PARALLEL)


@_vectorize(
    "(mom),(),(),(sample),(sample) -> (sample, mom)",
    [
        (
            nb.float32[:],
            nb.int64,
            nb.int64,
            nb.float32[:],
            nb.float32[:],
            nb.float32[:, :],
        ),
        (
            nb.float64[:],
            nb.int64,
            nb.int64,
            nb.float64[:],
            nb.float64[:],
            nb.float64[:, :],
        ),
    ],
    writable=None,
)
def rolling_vals(
    data_tmp: NDArray[FloatT],
    window: int,
    min_count: int,
    x: NDArray[FloatT],
    w: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    nsamp = len(x)
    data_tmp[...] = 0.0
    count = 0
    min_count = max(min_count, 1)

    for i in range(min(window, nsamp)):
        wi = w[i]
        if wi != 0:
            _push.push_val(x[i], wi, data_tmp)
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

        if wi_valid and wold_valid:
            _push.push_val(x[i], wi, data_tmp)
            _push.push_val(x[i_old], -wold, data_tmp)

        elif wi_valid:
            _push.push_val(x[i], wi, data_tmp)
            count += 1
        elif wold_valid:
            # special case.  If new weight is ==0, then we have a problem
            # assume we are subtracting the only positive element and are going back to zero
            if wold == data_tmp[0]:
                data_tmp[...] = 0.0
            else:
                _push.push_val(x[i_old], -wold, data_tmp)
            count -= 1

        if count >= min_count:
            out[i, ...] = data_tmp
        else:
            out[i, ...] = np.nan


@_vectorize(
    "(mom),(),(),(sample,mom) -> (sample, mom)",
    [
        (
            nb.float32[:],
            nb.int64,
            nb.int64,
            nb.float32[:, :],
            nb.float32[:, :],
        ),
        (
            nb.float64[:],
            nb.int64,
            nb.int64,
            nb.float64[:, :],
            nb.float64[:, :],
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
        wi = data[i, 0]
        if wi != 0:
            _push.push_data(data[i, ...], data_tmp)
            count += 1

        if count >= min_count:
            out[i, ...] = data_tmp
        else:
            out[i, ...] = np.nan

    for i in range(window, nsamp):
        i_old = i - window

        wi = data[i, 0]
        wold = data[i_old, 0]

        wi_valid = wi != 0.0
        wold_valid = wold != 0.0

        if wi_valid and wold_valid:
            _push.push_data(data[i, ...], data_tmp)
            _push.push_data_scale(data[i_old, ...], -1.0, data_tmp)

        elif wi_valid:
            _push.push_data(data[i, ...], data_tmp)
            count += 1
        elif wold_valid:
            # special case.  If new weight is ==0, then we have a problem
            # assume we are subtracting the only positive element and are going back to zero
            if wold == data_tmp[0]:
                data_tmp[...] = 0.0
            else:
                _push.push_data_scale(data[i_old, ...], -1.0, data_tmp)
            count -= 1

        if count >= min_count:
            out[i, ...] = data_tmp
        else:
            out[i, ...] = np.nan


# * Exponential moving average
@_vectorize(
    "(mom),(sample),(),(),(sample),(sample) -> (sample, mom)",
    [
        (
            nb.float32[:],
            nb.float32[:],
            nb.boolean,
            nb.int64,
            nb.float32[:],
            nb.float32[:],
            nb.float32[:, :],
        ),
        (
            nb.float64[:],
            nb.float64[:],
            nb.boolean,
            nb.int64,
            nb.float64[:],
            nb.float64[:],
            nb.float64[:, :],
        ),
    ],
    writable=None,
)
def rolling_exp_vals(
    data_tmp: NDArray[FloatT],
    alpha: NDArray[FloatT],
    adjust: bool,
    min_count: int,
    x: NDArray[FloatT],
    w: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    """Exponential moving accumulation of moments array"""
    nsamp = len(x)

    count = 0
    data_tmp[...] = 0.0
    old_weight = 0.0

    for i in range(nsamp):
        wi = w[i]
        alphai = alpha[i]
        decay = 1.0 - alphai

        # scale down
        old_weight *= decay
        data_tmp[..., 0] *= decay

        if wi != 0.0:
            count += 1
            if adjust:
                _push.push_val(x[i], wi, data_tmp)
            else:
                _push.push_val(x[i], w[i] * alphai, data_tmp)
                old_weight += alphai
                data_tmp[..., 0] /= old_weight
                old_weight = 1.0

        if count >= min_count:
            out[i, ...] = data_tmp
        else:
            out[i, ...] = np.nan


@_vectorize(
    "(mom),(sample),(),(),(sample,mom) -> (sample, mom)",
    [
        (
            nb.float32[:],
            nb.float32[:],
            nb.boolean,
            nb.int64,
            nb.float32[:, :],
            nb.float32[:, :],
        ),
        (
            nb.float64[:],
            nb.float64[:],
            nb.boolean,
            nb.int64,
            nb.float64[:, :],
            nb.float64[:, :],
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
        wi = data[i, 0]
        alphai = alpha[i]
        decay = 1.0 - alphai

        # scale down
        old_weight *= decay
        data_tmp[..., 0] *= decay

        if wi != 0.0:
            count += 1
            if adjust:
                _push.push_data(data[i, ...], data_tmp)
            else:
                _push.push_data_scale(data[i, ...], alphai, data_tmp)
                old_weight += alphai
                data_tmp[..., 0] /= old_weight
                old_weight = 1.0

        if count >= min_count:
            out[i, ...] = data_tmp
        else:
            out[i, ...] = np.nan


if not _PARALLEL:

    @_vectorize(
        "(sample),(sample),() -> (sample)",
        [
            (
                nb.float32[:],
                nb.float32[:],
                nb.boolean,
                nb.float32[:],
            ),
            (
                nb.float64[:],
                nb.float64[:],
                nb.boolean,
                nb.float64[:],
            ),
        ],
        writable=None,
    )
    def rolling_exp_var_bias_correction(
        weight: NDArray[FloatT],
        alpha: NDArray[FloatT],
        adjust: bool,
        out: NDArray[FloatT],
    ) -> None:
        """Bias from exponential moving average."""
        # make an attempt to combine weights from `data` and
        # this thing...
        sum_weight = sum_weight_2 = old_weight = 0.0

        # V1 -> sum(fweight * aweight)
        # V2 -> sum(fweight * aweight * aweight)
        #
        # Here, we treat:
        # * weight -> fweight (i.e., frequency weight)
        # * calculated weight -> aweight (non frequency weight)

        for i in range(len(alpha)):
            alphai = alpha[i]
            decay = 1.0 - alphai

            sum_weight *= decay
            sum_weight_2 *= decay * decay
            old_weight *= decay

            wi = weight[i]
            if weight[i] != 0.0:
                new_weight = 1.0 if adjust else alphai
                sum_weight += wi * new_weight
                sum_weight_2 += wi * new_weight**2
                if not adjust:
                    old_weight += new_weight
                    sum_weight /= old_weight
                    sum_weight_2 /= old_weight**2
                    old_weight = 1.0

            num = sum_weight * sum_weight
            den = num - sum_weight_2
            out[i] = num / den if den > 0.0 else np.nan

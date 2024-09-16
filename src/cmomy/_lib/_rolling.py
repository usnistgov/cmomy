"""Low level scalar pushers.  These will be wrapped by guvectorize methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb
import numpy as np

from .decorators import myjit

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from cmomy.core.typing import FloatT

from . import _push

_INLINE: bool | None = None


@myjit(
    signature=[
        (nb.int64, nb.int64, nb.float32[:, :], nb.float32[:, :]),
        (nb.int64, nb.int64, nb.float64[:, :], nb.float64[:, :]),
    ],
    inline=_INLINE,
)
def rolling_data(
    window: int, min_count: int, data: NDArray[FloatT], out: NDArray[FloatT]
) -> None:
    nsamp, mom_shape0 = data.shape
    dummy_mom = np.zeros(mom_shape0, dtype=data.dtype)
    count = 0

    for i in range(min(window, nsamp)):
        wi = data[i, 0]
        if wi != 0:
            _push.push_data(data[i, ...], dummy_mom)
            count += 1

        if count >= min_count:
            out[i, ...] = dummy_mom
        else:
            out[i, ...] = np.nan

    for i in range(window, nsamp):
        i_old = i - window

        wi = data[i, 0]
        wold = data[i_old, 0]

        wi_valid = wi != 0.0
        wold_valid = wold != 0.0

        if wi_valid:
            count += 1
            _push.push_data(data[i, ...], dummy_mom)

        if wold_valid:
            count -= 1
            if wold < dummy_mom[0]:
                _push.push_data_scale(data[i_old, ...], -1.0, dummy_mom)
            else:
                # special case.  If new weight is <=0, then we have a problem
                # assume we are subtracting the only positive element and are going back to zero
                dummy_mom[...] = 0.0

        if count >= min_count:
            out[i, ...] = dummy_mom
        else:
            out[i, ...] = np.nan


@myjit(
    signature=[
        (
            nb.int64,
            nb.int64,
            nb.float32[:],
            nb.float32[:],
            nb.float32[:, :],
        ),
        (
            nb.int64,
            nb.int64,
            nb.float64[:],
            nb.float64[:],
            nb.float64[:, :],
        ),
    ],
    inline=_INLINE,
)
def rolling_vals(
    window: int,
    min_count: int,
    x: NDArray[FloatT],
    w: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    nsamp, mom_shape0 = out.shape
    dummy_mom = np.zeros(mom_shape0, dtype=out.dtype)
    count = 0

    for i in range(min(window, nsamp)):
        wi = w[i]
        if wi != 0:
            _push.push_val(x[i], wi, dummy_mom)
            count += 1

        if count >= min_count:
            out[i, ...] = dummy_mom
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
            _push.push_val(x[i], wi, dummy_mom)

        if wold_valid:
            count -= 1
            if wold < dummy_mom[0]:
                _push.push_val(x[i_old], -wold, dummy_mom)
            else:
                # special case.  If new weight is ==0, then we have a problem
                # assume we are subtracting the only positive element and are going back to zero
                dummy_mom[...] = 0.0

        if count >= min_count:
            out[i, ...] = dummy_mom
        else:
            out[i, ...] = np.nan


@myjit(
    signature=[
        (
            nb.float32[:],
            nb.boolean,
            nb.int64,
            nb.float32[:, :],
            nb.float32[:, :],
        ),
        (
            nb.float64[:],
            nb.boolean,
            nb.int64,
            nb.float64[:, :],
            nb.float64[:, :],
        ),
    ],
    inline=_INLINE,
)
def rolling_exp_data(
    alpha: NDArray[FloatT],
    adjust: bool,
    min_count: int,
    data: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    """Exponential moving accumulation of moments array"""
    nsamp, mom_shape0 = out.shape
    dummy_mom = np.zeros(mom_shape0, dtype=out.dtype)

    count = 0
    old_weight = 0.0

    for i in range(nsamp):
        wi = data[i, 0]
        alphai = alpha[i]
        decay = 1.0 - alphai

        # scale down
        old_weight *= decay
        dummy_mom[..., 0] *= decay

        if wi != 0.0:
            count += 1

            scale = 1.0 if adjust else alphai
            _push.push_data_scale(data[i, ...], scale, dummy_mom)
            if not adjust:
                old_weight += alphai
                dummy_mom[..., 0] /= old_weight
                old_weight = 1.0

        if count >= min_count:
            out[i, ...] = dummy_mom
        else:
            out[i, ...] = np.nan


@myjit(
    signature=[
        (
            nb.float32[:],
            nb.boolean,
            nb.int64,
            nb.float32[:],
            nb.float32[:],
            nb.float32[:, :],
        ),
        (
            nb.float64[:],
            nb.boolean,
            nb.int64,
            nb.float64[:],
            nb.float64[:],
            nb.float64[:, :],
        ),
    ],
    inline=_INLINE,
)
def rolling_exp_vals(
    alpha: NDArray[FloatT],
    adjust: bool,
    min_count: int,
    x: NDArray[FloatT],
    w: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    """Exponential moving accumulation of moments array"""
    nsamp, mom_shape0 = out.shape
    dummy_mom = np.zeros(mom_shape0, dtype=out.dtype)

    count = 0
    old_weight = 0.0

    for i in range(nsamp):
        wi = w[i]
        alphai = alpha[i]
        decay = 1.0 - alphai

        # scale down
        old_weight *= decay
        dummy_mom[..., 0] *= decay

        if wi != 0.0:
            count += 1

            scale = wi if adjust else wi * alphai
            _push.push_val(x[i], scale, dummy_mom)

            if not adjust:
                old_weight += alphai
                dummy_mom[..., 0] /= old_weight
                old_weight = 1.0

        if count >= min_count:
            out[i, ...] = dummy_mom
        else:
            out[i, ...] = np.nan

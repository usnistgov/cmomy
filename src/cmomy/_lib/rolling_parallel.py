from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numba as nb
import numpy as np

from . import _rolling
from .decorators import myguvectorize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from cmomy.core.typing import FloatT

_PARALLEL = True  # Auto generated from rolling.py
_vectorize = partial(myguvectorize, parallel=_PARALLEL)


@_vectorize(
    "(),(),(sample,mom) -> (sample, mom)",
    [
        (
            nb.int64,
            nb.int64,
            nb.float32[:, :],
            nb.float32[:, :],
        ),
        (
            nb.int64,
            nb.int64,
            nb.float64[:, :],
            nb.float64[:, :],
        ),
    ],
    writable=None,
)
def rolling_data(
    window: int,
    min_count: int,
    data: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    min_count = max(min_count, 1)
    _rolling.rolling_data(window, min_count, data, out)


@_vectorize(
    "(sample, mom),(),(),(sample),(sample)",
    [
        (
            nb.float32[:, :],
            nb.int64,
            nb.int64,
            nb.float32[:],
            nb.float32[:],
        ),
        (
            nb.float64[:, :],
            nb.int64,
            nb.int64,
            nb.float64[:],
            nb.float64[:],
        ),
    ],
)
def rolling_vals(
    out: NDArray[FloatT],
    window: int,
    min_count: int,
    x: NDArray[FloatT],
    w: NDArray[FloatT],
) -> None:
    min_count = max(min_count, 1)
    _rolling.rolling_vals(window, min_count, x, w, out)


# * Exponential moving average
@_vectorize(
    "(sample),(),(),(sample,mom) -> (sample, mom)",
    [
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
    writable=None,
)
def rolling_exp_data(
    alpha: NDArray[FloatT],
    adjust: bool,
    min_count: int,
    data: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    """Exponential moving accumulation of moments array"""
    min_count = max(min_count, 1)
    _rolling.rolling_exp_data(alpha, adjust, min_count, data, out)


@_vectorize(
    "(sample,mom),(sample),(),(),(sample),(sample)",
    [
        (
            nb.float32[:, :],
            nb.float32[:],
            nb.boolean,
            nb.int64,
            nb.float32[:],
            nb.float32[:],
        ),
        (
            nb.float64[:, :],
            nb.float64[:],
            nb.boolean,
            nb.int64,
            nb.float64[:],
            nb.float64[:],
        ),
    ],
)
def rolling_exp_vals(
    out: NDArray[FloatT],
    alpha: NDArray[FloatT],
    adjust: bool,
    min_count: int,
    x: NDArray[FloatT],
    w: NDArray[FloatT],
) -> None:
    """Exponential moving accumulation of moments array"""
    min_count = max(min_count, 1)
    _rolling.rolling_exp_vals(alpha, adjust, min_count, x, w, out)


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

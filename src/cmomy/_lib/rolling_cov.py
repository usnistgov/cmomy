from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numba as nb

from . import _rolling_cov as _rolling
from .decorators import myguvectorize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from cmomy.core.typing import FloatT

_PARALLEL = False
_vectorize = partial(myguvectorize, parallel=_PARALLEL)


@_vectorize(
    "(),(),(sample,mom0,mom1) -> (sample, mom0, mom1)",
    [
        (
            nb.int64,
            nb.int64,
            nb.float32[:, :, :],
            nb.float32[:, :, :],
        ),
        (
            nb.int64,
            nb.int64,
            nb.float64[:, :, :],
            nb.float64[:, :, :],
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
    "(sample, mom0, mom1),(),(),(sample),(sample),(sample)",
    [
        (
            nb.float32[:, :, :],
            nb.int64,
            nb.int64,
            nb.float32[:],
            nb.float32[:],
            nb.float32[:],
        ),
        (
            nb.float64[:, :, :],
            nb.int64,
            nb.int64,
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
        ),
    ],
)
def rolling_vals(
    out: NDArray[FloatT],
    window: int,
    min_count: int,
    x0: NDArray[FloatT],
    w: NDArray[FloatT],
    x1: NDArray[FloatT],
) -> None:
    min_count = max(min_count, 1)
    _rolling.rolling_vals(window, min_count, x0, w, x1, out)


@_vectorize(
    "(sample),(),(),(sample,mom0,mom1) -> (sample, mom0,mom1)",
    [
        (
            nb.float32[:],
            nb.boolean,
            nb.int64,
            nb.float32[:, :, :],
            nb.float32[:, :, :],
        ),
        (
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
    "(sample,mom0,mom1),(sample),(),(),(sample),(sample),(sample)",
    [
        (
            nb.float32[:, :, :],
            nb.float32[:],
            nb.boolean,
            nb.int64,
            nb.float32[:],
            nb.float32[:],
            nb.float32[:],
        ),
        (
            nb.float64[:, :, :],
            nb.float64[:],
            nb.boolean,
            nb.int64,
            nb.float64[:],
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
    x0: NDArray[FloatT],
    w: NDArray[FloatT],
    x1: NDArray[FloatT],
) -> None:
    """Exponential moving accumulation of moments array"""
    min_count = max(min_count, 1)
    _rolling.rolling_exp_vals(alpha, adjust, min_count, x0, w, x1, out)

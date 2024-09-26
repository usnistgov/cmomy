"""Vectorized pushers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numba as nb

from . import _push_cov as _push
from .decorators import myguvectorize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from cmomy.core.typing import FloatT, NDGeneric


_PARALLEL = True  # Auto generated from push_cov.py
_decorator = partial(myguvectorize, parallel=_PARALLEL)


@_decorator(
    "(mom0, mom1),(),(),()",
    [
        (nb.float32[:, :], nb.float32, nb.float32, nb.float32),
        (nb.float64[:, :], nb.float64, nb.float64, nb.float64),
    ],
)
def push_val(
    out: NDArray[FloatT],
    x0: NDGeneric[FloatT],
    w: NDGeneric[FloatT],
    x1: NDGeneric[FloatT],
) -> None:
    _push.push_val(x0, x1, w, out)


@_decorator(
    "(mom0, mom1),(sample),(sample),(sample)",
    [
        (nb.float32[:, :], nb.float32[:], nb.float32[:], nb.float32[:]),
        (nb.float64[:, :], nb.float64[:], nb.float64[:], nb.float64[:]),
    ],
)
def reduce_vals(
    out: NDArray[FloatT],
    x0: NDArray[FloatT],
    w: NDArray[FloatT],
    x1: NDArray[FloatT],
) -> None:
    for i in range(len(x0)):
        _push.push_val(x0[i], x1[i], w[i], out)


@_decorator(
    "(mom0, mom1), (mom0, mom1)",
    [
        (nb.float32[:, :], nb.float32[:, :]),
        (nb.float64[:, :], nb.float64[:, :]),
    ],
)
def push_data(data: NDArray[FloatT], out: NDArray[FloatT]) -> None:
    _push.push_data(data, out)


@_decorator(
    "(sample, mom0, mom1), (mom0, mom1)",
    [
        (nb.float32[:, :, :], nb.float32[:, :]),
        (nb.float64[:, :, :], nb.float64[:, :]),
    ],
)
def reduce_data(data: NDArray[FloatT], out: NDArray[FloatT]) -> None:
    for i in range(data.shape[0]):
        _push.push_data(data[i, :, :], out)


@_decorator(
    "(mom0, mom1), (), (mom0, mom1)",
    [
        (nb.float32[:, :], nb.float32, nb.float32[:, :]),
        (nb.float64[:, :], nb.float64, nb.float64[:, :]),
    ],
)
def push_data_scale(data: NDArray[FloatT], scale: float, out: NDArray[FloatT]) -> None:
    _push.push_data_scale(data, scale, out)


@_decorator(
    "(sample, mom0, mom1) -> (mom0, mom1)",
    [
        (nb.float32[:, :, :], nb.float32[:, :]),
        (nb.float64[:, :, :], nb.float64[:, :]),
    ],
    writable=None,
)
def reduce_data_fromzero(data: NDArray[FloatT], out: NDArray[FloatT]) -> None:
    out[...] = 0.0
    for i in range(data.shape[0]):
        _push.push_data(data[i, :, :], out)


@_decorator(
    "(sample, mom0, mom1) -> (sample, mom0, mom1)",
    [
        (nb.float32[:, :, :], nb.float32[:, :, :]),
        (nb.float64[:, :, :], nb.float64[:, :, :]),
    ],
    writable=None,
)
def cumulative(
    data: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    out[:, ...] = data[:, ...]
    for i in range(1, data.shape[0]):
        _push.push_data(out[i - 1, ...], out[i, ...])


@_decorator(
    "(sample, mom0, mom1) -> (sample, mom0, mom1)",
    [
        (nb.float32[:, :, :], nb.float32[:, :, :]),
        (nb.float64[:, :, :], nb.float64[:, :, :]),
    ],
    writable=None,
)
def cumulative_inverse(
    data: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    out[:, ...] = data[:, ...]
    for i in range(1, data.shape[0]):
        _push.push_data_scale(data[i - 1, ...], -1.0, out[i, ...])

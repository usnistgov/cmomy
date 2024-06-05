"""Vectorized pushers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numba as nb

from . import _push_cov as _push
from .decorators import myguvectorize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..typing import FloatT, NDGeneric


_PARALLEL = False
_decorator = partial(myguvectorize, parallel=_PARALLEL)


@_decorator(
    "(),(),(),(mom0, mom1)",
    [
        (nb.float32, nb.float32, nb.float32, nb.float32[:, :]),
        (nb.float64, nb.float64, nb.float64, nb.float64[:, :]),
    ],
)
def push_val(
    x0: NDGeneric[FloatT],
    x1: NDGeneric[FloatT],
    w: NDGeneric[FloatT],
    out: NDArray[FloatT],
) -> None:
    _push.push_val(x0, x1, w, out)


@_decorator(
    "(sample),(sample),(sample),(mom0, mom1)",
    [
        (nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:, :]),
        (nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:, :]),
    ],
)
def reduce_vals(
    x0: NDArray[FloatT],
    x1: NDArray[FloatT],
    w: NDArray[FloatT],
    out: NDArray[FloatT],
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

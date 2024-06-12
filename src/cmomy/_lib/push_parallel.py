"""Vectorized pushers."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numba as nb

from . import _push
from .decorators import myguvectorize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..typing import FloatT, NDGeneric


_PARALLEL = True  # Auto generated from push.py
_decorator = partial(myguvectorize, parallel=_PARALLEL)


@_decorator(
    "(),(),(mom)",
    [
        (nb.float32, nb.float32, nb.float32[:]),
        (nb.float64, nb.float64, nb.float64[:]),
    ],
)
def push_val(x: NDGeneric[FloatT], w: NDGeneric[FloatT], out: NDArray[FloatT]) -> None:
    _push.push_val(x, w, out)


@_decorator(
    "(sample),(sample),(mom)",
    [
        (nb.float32[:], nb.float32[:], nb.float32[:]),
        (nb.float64[:], nb.float64[:], nb.float64[:]),
    ],
)
def reduce_vals(x: NDArray[FloatT], w: NDArray[FloatT], out: NDArray[FloatT]) -> None:
    for i in range(x.shape[0]):
        _push.push_val(x[i], w[i], out)


# @_decorator(
#     "(sample),(sample),(mom)",
#     [
#         (nb.float32[:], nb.float32[:], nb.float32[:]),
#         (nb.float64[:], nb.float64[:], nb.float64[:]),
#     ],
# )
# def reduce_vals_multipass(
#     x: NDArray[FloatT], w: NDArray[FloatT], out: NDArray[FloatT]
# ) -> None:
#     # first calculate average
#     xave = 0.0
#     wsum = 0.0
#     for i in range(x.shape[0]):
#         ww = w[i]
#         xave += x[i] * ww
#         wsum += ww

#     xave /= wsum
#     out[...] = 0.0
#     out[0] = wsum
#     out[1] = xave

#     # sum other moments
#     nmom = out.shape[-1]
#     if nmom > 2:
#         for i in range(x.shape[0]):
#             xx = x[i]
#             ww = w[i]
#             for m in range(2, nmom):
#                 out[m] += ww * (xx - xave) ** m

#         for m in range(2, nmom):
#             out[m] /= wsum


@_decorator(
    "(),(vars),(),(mom)",
    [
        (nb.float32, nb.float32[:], nb.float32, nb.float32[:]),
        (nb.float64, nb.float64[:], nb.float64, nb.float64[:]),
    ],
)
def push_stat(
    a: NDGeneric[FloatT],
    v: NDArray[FloatT],
    w: NDGeneric[FloatT],
    out: NDArray[FloatT],
) -> None:
    _push.push_stat(a, v, w, out)


@_decorator(
    "(sample),(sample,vars),(sample),(mom)",
    [
        (nb.float32[:], nb.float32[:, :], nb.float32[:], nb.float32[:]),
        (nb.float64[:], nb.float64[:, :], nb.float64[:], nb.float64[:]),
    ],
)
def reduce_stats(
    a: NDArray[FloatT],
    v: NDArray[FloatT],
    w: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    for i in range(a.shape[0]):
        _push.push_stat(a[i], v[i, :], w[i], out)


@_decorator(
    "(mom),(mom)",
    signature=[
        (nb.float32[:], nb.float32[:]),
        (nb.float64[:], nb.float64[:]),
    ],
)
def push_data(data: NDArray[FloatT], out: NDArray[FloatT]) -> None:
    _push.push_data(data, out)


@_decorator(
    "(sample, mom),(mom)",
    [
        (nb.float32[:, :], nb.float32[:]),
        (nb.float64[:, :], nb.float64[:]),
    ],
)
def reduce_data(data: NDArray[FloatT], out: NDArray[FloatT]) -> None:
    for i in range(data.shape[0]):
        _push.push_data(data[i, :], out)


@_decorator(
    "(sample, mom) -> (mom)",
    [
        (nb.float32[:, :], nb.float32[:]),
        (nb.float64[:, :], nb.float64[:]),
    ],
    writable=None,
)
def reduce_data_fromzero(
    data: NDArray[FloatT],
    out: NDArray[FloatT],
) -> None:
    out[...] = 0.0
    for i in range(data.shape[0]):
        _push.push_data(data[i, :], out)

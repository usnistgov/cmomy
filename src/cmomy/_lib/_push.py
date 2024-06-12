"""Low level scalar pushers.  These will be wrapped by guvectorize methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb

from .decorators import myjit
from .utils import BINOMIAL_FACTOR

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..typing import FloatT, NDGeneric


@myjit(
    signature=[
        (nb.float32, nb.float32, nb.float32[:]),
        (nb.float64, nb.float64, nb.float64[:]),
    ],
    inline=True,
)
def push_val(x: NDGeneric[FloatT], w: NDGeneric[FloatT], out: NDArray[FloatT]) -> None:
    if w == 0.0:
        return

    order = out.shape[0] - 1

    out[0] += w
    alpha = w / out[0]
    one_alpha = 1.0 - alpha

    delta = x - out[1]
    incr = delta * alpha

    out[1] += incr
    if order == 1:
        return

    for a in range(order, 2, -1):
        # c > 1
        tmp = 0.0
        delta_b = 1.0
        alpha_b = 1.0
        minus_b = 1.0
        one_alpha_b = 1.0
        for b in range(a - 1):
            c = a - b
            tmp += (
                BINOMIAL_FACTOR[a, b]
                * delta_b
                * (minus_b * alpha_b * one_alpha * out[c])
            )
            delta_b *= delta
            alpha_b *= alpha
            one_alpha_b *= one_alpha
            minus_b *= -1.0

        tmp += delta * alpha * one_alpha * delta_b * (-minus_b * alpha_b + one_alpha_b)

        out[a] = tmp

    out[2] = one_alpha * (out[2] + delta * incr)


# NOTE: Marginally faster to have separate push_stat, push_data, push_data_scale
@myjit(
    signature=[
        (nb.float32[:], nb.float32[:]),
        (nb.float64[:], nb.float64[:]),
    ],
    inline=True,
)
def push_data(data: NDArray[FloatT], out: NDArray[FloatT]) -> None:
    # w : weight
    # data[1]a : average
    # v[i] : <dx**(i+2)>

    w = data[0]
    if w == 0:
        return

    order = out.shape[0] - 1

    out[0] += w

    alpha = w / out[0]
    one_alpha = 1.0 - alpha
    delta = data[1] - out[1]
    incr = delta * alpha

    out[1] += incr

    if order == 1:
        return

    for a1 in range(order, 2, -1):
        tmp = 0.0
        delta_b = 1.0
        alpha_b = 1.0
        minus_b = 1.0
        one_alpha_b = 1.0
        for b in range(a1 - 1):
            c = a1 - b
            tmp += (
                BINOMIAL_FACTOR[a1, b]
                * delta_b
                * (
                    minus_b * alpha_b * one_alpha * out[c]
                    + one_alpha_b * alpha * data[c]
                )
            )
            delta_b *= delta
            alpha_b *= alpha
            one_alpha_b *= one_alpha
            minus_b *= -1.0

        # think I can scrap this?
        tmp += delta * alpha * one_alpha * delta_b * (-minus_b * alpha_b + one_alpha_b)
        out[a1] = tmp

    out[2] = data[2] * alpha + one_alpha * (out[2] + delta * incr)


@myjit(
    signature=[
        (nb.float32[:], nb.float32, nb.float32[:]),
        (nb.float64[:], nb.float64, nb.float64[:]),
    ],
    inline=True,
)
def push_data_scale(
    data: NDArray[FloatT], scale: NDGeneric[FloatT], out: NDArray[FloatT]
) -> None:
    # w : weight
    # data[1]a : average
    # v[i] : <dx**(i+2)>

    w = data[0] * scale
    if w == 0:
        return

    order = out.shape[0] - 1

    out[0] += w

    alpha = w / out[0]
    one_alpha = 1.0 - alpha
    delta = data[1] - out[1]
    incr = delta * alpha

    out[1] += incr

    if order == 1:
        return

    for a1 in range(order, 2, -1):
        tmp = 0.0
        delta_b = 1.0
        alpha_b = 1.0
        minus_b = 1.0
        one_alpha_b = 1.0
        for b in range(a1 - 1):
            c = a1 - b
            tmp += (
                BINOMIAL_FACTOR[a1, b]
                * delta_b
                * (
                    minus_b * alpha_b * one_alpha * out[c]
                    + one_alpha_b * alpha * data[c]
                )
            )
            delta_b *= delta
            alpha_b *= alpha
            one_alpha_b *= one_alpha
            minus_b *= -1.0

        # think I can scrap this?
        c = 0
        b = a1 - c
        tmp += delta * alpha * one_alpha * delta_b * (-minus_b * alpha_b + one_alpha_b)
        out[a1] = tmp

    out[2] = data[2] * alpha + one_alpha * (out[2] + delta * incr)


@myjit(
    signature=[
        (nb.float32, nb.float32[:], nb.float32, nb.float32[:]),
        (nb.float64, nb.float64[:], nb.float64, nb.float64[:]),
    ],
    inline=True,
)
def push_stat(
    a: NDGeneric[FloatT],
    v: NDArray[FloatT],
    w: NDGeneric[FloatT],
    out: NDArray[FloatT],
) -> None:
    # a : average
    # v[i] : <dx**(i+2)>
    # w : weight

    if w == 0:
        return

    order = out.shape[0] - 1

    out[0] += w

    alpha = w / out[0]
    one_alpha = 1.0 - alpha
    delta = a - out[1]
    incr = delta * alpha

    out[1] += incr

    if order == 1:
        return

    for a1 in range(order, 2, -1):
        tmp = 0.0
        delta_b = 1.0
        alpha_b = 1.0
        minus_b = 1.0
        one_alpha_b = 1.0
        for b in range(a1 - 1):
            c = a1 - b
            tmp += (
                BINOMIAL_FACTOR[a1, b]
                * delta_b
                * (
                    minus_b * alpha_b * one_alpha * out[c]
                    + one_alpha_b * alpha * v[c - 2]
                )
            )
            delta_b *= delta
            alpha_b *= alpha
            one_alpha_b *= one_alpha
            minus_b *= -1.0

        # think I can scrap this?
        c = 0
        b = a1 - c
        tmp += delta * alpha * one_alpha * delta_b * (-minus_b * alpha_b + one_alpha_b)
        out[a1] = tmp

    out[2] = v[0] * alpha + one_alpha * (out[2] + delta * incr)

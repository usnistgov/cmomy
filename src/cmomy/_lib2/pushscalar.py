# mypy: disable-error-code="no-untyped-call,no-untyped-def"
"""Low level scalar pushers.  These will be wrapped by guvectorize methods."""

from __future__ import annotations

import numba as nb

from .decorators import myjit
from .utils import BINOMIAL_FACTOR


@myjit(
    signature=[
        (nb.float32, nb.float32, nb.float32[:]),
        (nb.float64, nb.float64, nb.float64[:]),
    ],
    inline=True,
)
def push_val(w, x, data) -> None:
    if w == 0.0:
        return

    order = data.shape[0] - 1

    data[0] += w
    alpha = w / data[0]
    one_alpha = 1.0 - alpha

    delta = x - data[1]
    incr = delta * alpha

    data[1] += incr
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
                * (minus_b * alpha_b * one_alpha * data[c])
            )
            delta_b *= delta
            alpha_b *= alpha
            one_alpha_b *= one_alpha
            minus_b *= -1.0

        tmp += delta * alpha * one_alpha * delta_b * (-minus_b * alpha_b + one_alpha_b)

        data[a] = tmp

    data[2] = one_alpha * (data[2] + delta * incr)


# NOTE: Marginally faster to have separate push_stat, push_data, push_data_scale
@myjit(
    signature=[
        (nb.float32[:], nb.float32[:]),
        (nb.float64[:], nb.float64[:]),
    ],
    inline=True,
)
def push_data(other, data) -> None:
    # w : weight
    # other[1]a : average
    # v[i] : <dx**(i+2)>

    w = other[0]
    if w == 0:
        return

    order = data.shape[0] - 1

    data[0] += w

    alpha = w / data[0]
    one_alpha = 1.0 - alpha
    delta = other[1] - data[1]
    incr = delta * alpha

    data[1] += incr

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
                    minus_b * alpha_b * one_alpha * data[c]
                    + one_alpha_b * alpha * other[c]
                )
            )
            delta_b *= delta
            alpha_b *= alpha
            one_alpha_b *= one_alpha
            minus_b *= -1.0

        # think I can scrap this?
        tmp += delta * alpha * one_alpha * delta_b * (-minus_b * alpha_b + one_alpha_b)
        data[a1] = tmp

    data[2] = other[2] * alpha + one_alpha * (data[2] + delta * incr)


@myjit(
    signature=[
        (nb.float32[:], nb.float32, nb.float32[:]),
        (nb.float64[:], nb.float64, nb.float64[:]),
    ],
    inline=True,
)
def push_data_scale(other, scale, data) -> None:
    # w : weight
    # other[1]a : average
    # v[i] : <dx**(i+2)>

    w = other[0] * scale
    if w == 0:
        return

    order = data.shape[0] - 1

    data[0] += w

    alpha = w / data[0]
    one_alpha = 1.0 - alpha
    delta = other[1] - data[1]
    incr = delta * alpha

    data[1] += incr

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
                    minus_b * alpha_b * one_alpha * data[c]
                    + one_alpha_b * alpha * other[c]
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
        data[a1] = tmp

    data[2] = other[2] * alpha + one_alpha * (data[2] + delta * incr)


@myjit(
    signature=[
        (nb.float32, nb.float32, nb.float32[:], nb.float32[:]),
        (nb.float64, nb.float64, nb.float64[:], nb.float64[:]),
    ],
    inline=True,
)
def push_stat(w, a, v, data) -> None:
    # w : weight
    # a : average
    # v[i] : <dx**(i+2)>

    if w == 0:
        return

    order = data.shape[0] - 1

    data[0] += w

    alpha = w / data[0]
    one_alpha = 1.0 - alpha
    delta = a - data[1]
    incr = delta * alpha

    data[1] += incr

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
                    minus_b * alpha_b * one_alpha * data[c]
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
        data[a1] = tmp

    data[2] = v[0] * alpha + one_alpha * (data[2] + delta * incr)

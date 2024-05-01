# mypy: disable-error-code="no-untyped-call,no-untyped-def"
"""Low level scalar pushers.  These will be wrapped by guvectorize methods."""

from __future__ import annotations

import numba as nb

from .decorators import myjit
from .utils import BINOMIAL_FACTOR


@myjit(
    signature=[
        (nb.float32, nb.float32, nb.float32, nb.float32[:, :]),
        (nb.float64, nb.float64, nb.float64, nb.float64[:, :]),
    ],
    inline=True,
)
def push_val(w, x0, x1, data) -> None:
    if w == 0.0:
        return

    order0 = data.shape[0] - 1
    order1 = data.shape[1] - 1

    data[0, 0] += w
    alpha = w / data[0, 0]
    one_alpha = 1.0 - alpha

    delta0 = x0 - data[1, 0]
    delta1 = x1 - data[0, 1]

    incr0 = delta0 * alpha
    incr1 = delta1 * alpha

    # NOTE: decided to force order > 1
    # otherwise, this is just normal variance
    # if order0 > 0:
    #     data[1, 0] += incr0
    # if order1 > 0:
    #     data[0, 1] += incr1
    data[1, 0] += incr0
    data[0, 1] += incr1

    a0_min = max(0, 2 - order1)
    for a0 in range(order0, a0_min - 1, -1):
        a1_min = max(0, 2 - a0)
        for a1 in range(order1, a1_min - 1, -1):
            tmp = 0.0
            delta0_b0 = 1.0
            alpha_b0 = 1.0
            minus_b0 = 1.0
            one_alpha_b0 = 1.0
            for b0 in range(a0 + 1):
                c0 = a0 - b0
                f0 = BINOMIAL_FACTOR[a0, b0]

                delta1_b1 = 1.0
                alpha_bb = alpha_b0
                minus_bb = minus_b0
                one_alpha_bb = one_alpha_b0
                for b1 in range(a1 + 1):
                    c1 = a1 - b1
                    cs = c0 + c1
                    if cs == 0:
                        tmp += (
                            delta0_b0
                            * delta1_b1
                            * (minus_bb * alpha_bb * one_alpha + one_alpha_bb * alpha)
                        )
                    elif cs != 1:
                        tmp += (
                            f0
                            * BINOMIAL_FACTOR[a1, b1]
                            * delta0_b0
                            * delta1_b1
                            * (minus_bb * alpha_bb * one_alpha * data[c0, c1])
                        )
                    delta1_b1 *= delta1
                    alpha_bb *= alpha
                    one_alpha_bb *= one_alpha
                    minus_bb *= -1

                delta0_b0 *= delta0
                alpha_b0 *= alpha
                minus_b0 *= -1
                one_alpha_b0 *= one_alpha

            data[a0, a1] = tmp


@myjit(
    signature=[
        (nb.float32[:, :], nb.float32[:, :]),
        (nb.float64[:, :], nb.float64[:, :]),
    ],
    inline=True,
)
def push_data(other, data) -> None:
    w = other[0, 0]
    if w == 0.0:
        return

    order0 = data.shape[0] - 1
    order1 = data.shape[1] - 1

    data[0, 0] += w
    alpha = w / data[0, 0]
    one_alpha = 1.0 - alpha

    delta0 = other[1, 0] - data[1, 0]
    delta1 = other[0, 1] - data[0, 1]

    incr0 = delta0 * alpha
    incr1 = delta1 * alpha

    # NOTE : decided to force all orders >0
    # if order0 > 0:
    #     data[1, 0] += incr0
    # if order1 > 0:
    #     data[0, 1] += incr1
    data[1, 0] += incr0
    data[0, 1] += incr1

    a0_min = max(0, 2 - order1)
    for a0 in range(order0, a0_min - 1, -1):
        a1_min = max(0, 2 - a0)
        for a1 in range(order1, a1_min - 1, -1):
            # Alternative
            tmp = 0.0
            delta0_b0 = 1.0
            alpha_b0 = 1.0
            minus_b0 = 1.0
            one_alpha_b0 = 1.0
            for b0 in range(a0 + 1):
                c0 = a0 - b0
                f0 = BINOMIAL_FACTOR[a0, b0]

                delta1_b1 = 1.0
                alpha_bb = alpha_b0
                minus_bb = minus_b0
                one_alpha_bb = one_alpha_b0
                for b1 in range(a1 + 1):
                    c1 = a1 - b1
                    cs = c0 + c1
                    if cs == 0:
                        tmp += (
                            delta0_b0
                            * delta1_b1
                            * (minus_bb * alpha_bb * one_alpha + one_alpha_bb * alpha)
                        )
                    elif cs != 1:
                        tmp += (
                            f0
                            * BINOMIAL_FACTOR[a1, b1]
                            * delta0_b0
                            * delta1_b1
                            * (
                                minus_bb * alpha_bb * one_alpha * data[c0, c1]
                                + one_alpha_bb * alpha * other[c0, c1]
                            )
                        )
                    delta1_b1 *= delta1
                    alpha_bb *= alpha
                    one_alpha_bb *= one_alpha
                    minus_bb *= -1

                delta0_b0 *= delta0
                alpha_b0 *= alpha
                minus_b0 *= -1
                one_alpha_b0 *= one_alpha

            data[a0, a1] = tmp


@myjit(
    signature=[
        (nb.float32[:, :], nb.float32, nb.float32[:, :]),
        (nb.float64[:, :], nb.float64, nb.float64[:, :]),
    ],
    inline=True,
)
def push_data_scale(other, scale, data) -> None:
    w = other[0, 0] * scale
    if w == 0.0:
        return

    order0 = data.shape[0] - 1
    order1 = data.shape[1] - 1

    data[0, 0] += w
    alpha = w / data[0, 0]
    one_alpha = 1.0 - alpha

    delta0 = other[1, 0] - data[1, 0]
    delta1 = other[0, 1] - data[0, 1]

    incr0 = delta0 * alpha
    incr1 = delta1 * alpha

    # NOTE : decided to force all orders >0
    # if order0 > 0:
    #     data[1, 0] += incr0
    # if order1 > 0:
    #     data[0, 1] += incr1
    data[1, 0] += incr0
    data[0, 1] += incr1

    a0_min = max(0, 2 - order1)
    for a0 in range(order0, a0_min - 1, -1):
        a1_min = max(0, 2 - a0)
        for a1 in range(order1, a1_min - 1, -1):
            # Alternative
            tmp = 0.0
            delta0_b0 = 1.0
            alpha_b0 = 1.0
            minus_b0 = 1.0
            one_alpha_b0 = 1.0
            for b0 in range(a0 + 1):
                c0 = a0 - b0
                f0 = BINOMIAL_FACTOR[a0, b0]

                delta1_b1 = 1.0
                alpha_bb = alpha_b0
                minus_bb = minus_b0
                one_alpha_bb = one_alpha_b0
                for b1 in range(a1 + 1):
                    c1 = a1 - b1
                    cs = c0 + c1
                    if cs == 0:
                        tmp += (
                            delta0_b0
                            * delta1_b1
                            * (minus_bb * alpha_bb * one_alpha + one_alpha_bb * alpha)
                        )
                    elif cs != 1:
                        tmp += (
                            f0
                            * BINOMIAL_FACTOR[a1, b1]
                            * delta0_b0
                            * delta1_b1
                            * (
                                minus_bb * alpha_bb * one_alpha * data[c0, c1]
                                + one_alpha_bb * alpha * other[c0, c1]
                            )
                        )
                    delta1_b1 *= delta1
                    alpha_bb *= alpha
                    one_alpha_bb *= one_alpha
                    minus_bb *= -1

                delta0_b0 *= delta0
                alpha_b0 *= alpha
                minus_b0 *= -1
                one_alpha_b0 *= one_alpha

            data[a0, a1] = tmp

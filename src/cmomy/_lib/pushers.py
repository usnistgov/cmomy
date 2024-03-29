# mypy: disable-error-code="no-untyped-call,no-untyped-def"
"""low level routines to do pushing."""
from __future__ import annotations

from typing import Callable, NamedTuple

from .utils import BINOMIAL_FACTOR, myjit

# -- Scalars ---------------------------------------------------------------------------


@myjit(inline=True)
def _push_val(data, w, x) -> None:
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

        # c == 0
        # b = a
        tmp += delta * alpha * one_alpha * delta_b * (-minus_b * alpha_b + one_alpha_b)

        data[a] = tmp

    data[2] = one_alpha * (data[2] + delta * incr)


@myjit()
def _push_vals(data, W, X) -> None:
    ns = X.shape[0]
    for s in range(ns):
        _push_val(data, W[s], X[s])


@myjit(inline=True)
def _push_stat(data, w, a, v) -> None:
    # w : weight
    # a : average
    # v[i] : <dx**(i+2)>
    # scale : parameter to rescale the weight

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

        c = 0
        b = a1 - c
        tmp += delta * alpha * one_alpha * delta_b * (-minus_b * alpha_b + one_alpha_b)
        data[a1] = tmp

    data[2] = v[0] * alpha + one_alpha * (data[2] + delta * incr)


@myjit()
def _push_stats(data, W, A, V) -> None:
    ns = A.shape[0]
    for s in range(ns):
        _push_stat(data, W[s], A[s], V[s, ...])


@myjit(inline=True)
def _push_data(data, data_in) -> None:
    _push_stat(data, data_in[0], data_in[1], data_in[2:])


@myjit()
def _push_datas(data, data_in) -> None:
    ns = data_in.shape[0]
    for s in range(ns):
        _push_stat(data, data_in[s, 0], data_in[s, 1], data_in[s, 2:])


# Vector
@myjit()
def _push_val_vec(data, w, x) -> None:
    nv = data.shape[0]
    for k in range(nv):
        _push_val(data[k, :], w[k], x[k])


@myjit()
def _push_vals_vec(data, W, X) -> None:
    ns = X.shape[0]
    nv = data.shape[0]
    for s in range(ns):
        for k in range(nv):
            _push_val(data[k, :], W[s, k], X[s, k])


@myjit()
def _push_stat_vec(data, w, a, v) -> None:
    nv = data.shape[0]
    for k in range(nv):
        _push_stat(data[k, :], w[k], a[k], v[k, :])


@myjit()
def _push_stats_vec(data, W, A, V) -> None:
    # V[sample, moment-2, value]
    ns = A.shape[0]
    nv = data.shape[0]
    for s in range(ns):
        for k in range(nv):
            _push_stat(data[k, :], W[s, k], A[s, k], V[s, k, :])


@myjit()
def _push_data_vec(data, data_in) -> None:
    nv = data.shape[0]
    for k in range(nv):
        _push_data(data[k, :], data_in[k, :])


@myjit()
def _push_datas_vec(data, Data_in) -> None:
    ns = Data_in.shape[0]
    nv = data.shape[0]
    for s in range(ns):
        for k in range(nv):
            _push_data(data[k, :], Data_in[s, k, :])


# --- Covariance -----------------------------------------------------------------------


@myjit(inline=True)
def _push_val_cov(data, w, x0, x1) -> None:
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


@myjit()
def _push_vals_cov(data, W, X1, X2) -> None:
    ns = X1.shape[0]
    for s in range(ns):
        _push_val_cov(data, W[s], X1[s], X2[s])


@myjit(inline=True)
def _push_data_scale_cov(data, data_in, scale) -> None:
    w = data_in[0, 0] * scale
    if w == 0.0:
        return

    order0 = data.shape[0] - 1
    order1 = data.shape[1] - 1

    data[0, 0] += w
    alpha = w / data[0, 0]
    one_alpha = 1.0 - alpha

    delta0 = data_in[1, 0] - data[1, 0]
    delta1 = data_in[0, 1] - data[0, 1]

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
                                + one_alpha_bb * alpha * data_in[c0, c1]
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


@myjit()
def _push_data_cov(data, data_in) -> None:
    _push_data_scale_cov(data, data_in, 1.0)


@myjit()
def _push_datas_cov(data, datas) -> None:
    ns = datas.shape[0]
    for s in range(ns):
        _push_data_scale_cov(data, datas[s], 1.0)


# --- Vector Covariance ----------------------------------------------------------------


@myjit()
def _push_val_cov_vec(data, w, x0, x1) -> None:
    nv = data.shape[0]
    for k in range(nv):
        _push_val_cov(data[k, ...], w[k], x0[k], x1[k])


@myjit()
def _push_vals_cov_vec(data, W, X0, X1) -> None:
    nv = data.shape[0]
    ns = X0.shape[0]
    for s in range(ns):
        for k in range(nv):
            _push_val_cov(data[k, ...], W[s, k], X0[s, k], X1[s, k])


@myjit()
def _push_data_cov_vec(data, data_in) -> None:
    nv = data.shape[0]
    for k in range(nv):
        _push_data_scale_cov(data[k, ...], data_in[k, ...], 1.0)


@myjit()
def _push_datas_cov_vec(data, Datas) -> None:
    nv = data.shape[0]
    ns = Datas.shape[0]
    for s in range(ns):
        for k in range(nv):
            _push_data_scale_cov(data[k, ...], Datas[s, k, ...], 1.0)


# --- Vals Scaled Pushers -------------------------------------------------------------------


@myjit(inline=True)
def push_vals_scale(data, W, X, scale) -> None:
    ns = X.shape[0]
    for s in range(ns):
        f = scale[s]
        if f == 0:
            continue
        _push_val(data, W[s] * f, X[s])


@myjit(inline=True)
def push_vals_scale_vec(data, W, X, scale) -> None:
    ns = X.shape[0]
    nv = data.shape[0]
    for s in range(ns):
        f = scale[s]
        if f == 0:
            continue
        for k in range(nv):
            _push_val(data[k, :], W[s, k] * f, X[s, k])


@myjit(inline=True)
def push_vals_scale_cov(data, W, X1, X2, scale) -> None:
    ns = X1.shape[0]
    for s in range(ns):
        f = scale[s]
        if f == 0:
            continue
        _push_val_cov(data, W[s] * f, X1[s], X2[s])


@myjit(inline=True)
def push_vals_scale_cov_vec(data, W, X0, X1, scale) -> None:
    nv = data.shape[0]
    ns = X0.shape[0]
    for s in range(ns):
        f = scale[s]
        if f == 0:
            continue
        for k in range(nv):
            _push_val_cov(data[k, ...], W[s, k] * f, X0[s, k], X1[s, k])


# --- Datas Scaled Pushers -------------------------------------------------------------


@myjit(inline=True)
def push_datas_scale(data, data_in, scale) -> None:
    ns = data_in.shape[0]
    for s in range(ns):
        f = scale[s]
        if f == 0:
            continue
        _push_stat(data, data_in[s, 0] * f, data_in[s, 1], data_in[s, 2:])


# @myjit(inline=True)
# def _push_data_scale(data, data_in, scale):
#     _push_stat(data, data_in[0] * scale, data_in[1], data_in[2:])


@myjit(inline=True)
def push_datas_scale_vec(data, Data_in, scale) -> None:
    ns = Data_in.shape[0]
    nv = data.shape[0]
    for s in range(ns):
        f = scale[s]
        if f == 0:
            continue
        for k in range(nv):
            # _push_data_scale(data[k, :], Data_in[s, k, :], f)
            _push_stat(
                data[k, :], Data_in[s, k, 0] * f, Data_in[s, k, 1], Data_in[s, k, 2:]
            )


# Note: _push_data_scale_cov main func above
@myjit(inline=True)
def push_datas_scale_cov(data, datas, scale) -> None:
    ns = datas.shape[0]
    for s in range(ns):
        f = scale[s]
        if f == 0:
            continue
        _push_data_scale_cov(data, datas[s], f)


@myjit(inline=True)
def push_datas_scale_cov_vec(data, Datas, scale) -> None:
    nv = data.shape[0]
    ns = Datas.shape[0]
    for s in range(ns):
        f = scale[s]
        if f == 0:
            continue
        for k in range(nv):
            _push_data_scale_cov(data[k, ...], Datas[s, k, ...], f)


# @myjit()
# def _push_data_scale_vec(data, data_in, scale):
#     nv = data.shape[0]
#     if scale != 0:
#         for k in range(nv):
#             _push_data_scale(data[k, :], data_in[k, :], scale)


# @myjit()
# def _push_data_scale_cov_vec(data, data_in, scale):
#     nv = data.shape[0]
#     if scale > 0:
#         for k in range(nv):
#             _push_data_scale_cov(data[k, ...], data_in[k, ...], scale)


# --- Interaface - ---------------------------------------------------------------------
class Pusher(NamedTuple):
    """Collection of pusher functions."""

    val: Callable[..., None]
    vals: Callable[..., None]
    stat: Callable[..., None] | None
    stats: Callable[..., None] | None
    data: Callable[..., None]
    datas: Callable[..., None]


_pusher_scalar = Pusher(
    val=_push_val,
    vals=_push_vals,
    stat=_push_stat,
    stats=_push_stats,
    data=_push_data,
    datas=_push_datas,
)

_pusher_vector = Pusher(
    val=_push_val_vec,
    vals=_push_vals_vec,
    stat=_push_stat_vec,
    stats=_push_stats_vec,
    data=_push_data_vec,
    datas=_push_datas_vec,
)


_pusher_cov_scalar = Pusher(
    val=_push_val_cov,
    vals=_push_vals_cov,
    stat=None,
    stats=None,
    data=_push_data_cov,
    datas=_push_datas_cov,
)

_pusher_cov_vector = Pusher(
    val=_push_val_cov_vec,
    vals=_push_vals_cov_vec,
    stat=None,
    stats=None,
    data=_push_data_cov_vec,
    datas=_push_datas_cov_vec,
)


def factory_pushers(cov: bool = False, vec: bool = False) -> Pusher:
    """Factory method to get pusher functions."""  # D401
    if cov:
        if vec:
            return _pusher_cov_vector
        return _pusher_cov_scalar

    if vec:
        return _pusher_vector
    return _pusher_scalar

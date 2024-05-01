# mypy: disable-error-code="no-untyped-call,no-untyped-def"
"""Vectorized pushers."""

from __future__ import annotations

from functools import partial

import numba as nb

from . import pushscalar
from .decorators import myguvectorize

_PARALLEL = True  # Auto generated from pushvec.py
_decorator = partial(myguvectorize, parallel=_PARALLEL)


@_decorator(
    "(),(),(mom)",
    [
        (nb.float32, nb.float32, nb.float32[:]),
        (nb.float64, nb.float64, nb.float64[:]),
    ],
)
def push_val(w, x, data) -> None:
    pushscalar.push_val(w, x, data)


@_decorator(
    "(sample),(sample),(mom)",
    [
        (nb.float32[:], nb.float32[:], nb.float32[:]),
        (nb.float64[:], nb.float64[:], nb.float64[:]),
    ],
)
def reduce_vals(w, x, data) -> None:
    for i in range(x.shape[0]):
        pushscalar.push_val(w[i], x[i], data)


@_decorator(
    "(),(),(vars),(mom)",
    [
        (nb.float32, nb.float32, nb.float32[:], nb.float32[:]),
        (nb.float64, nb.float64, nb.float64[:], nb.float64[:]),
    ],
)
def push_stat(w, a, v, data) -> None:
    pushscalar.push_stat(w, a, v, data)


@_decorator(
    "(sample),(sample),(sample,vars),(mom)",
    [
        (nb.float32[:], nb.float32[:], nb.float32[:, :], nb.float32[:]),
        (nb.float64[:], nb.float64[:], nb.float64[:, :], nb.float64[:]),
    ],
)
def reduce_stats(w, a, v, data) -> None:
    for i in range(a.shape[0]):
        pushscalar.push_stat(w[i], a[i], v[i, :], data)


@_decorator(
    "(mom),(mom)",
    signature=[
        (nb.float32[:], nb.float32[:]),
        (nb.float64[:], nb.float64[:]),
    ],
)
def push_data(other, data) -> None:
    pushscalar.push_data(other, data)


@_decorator(
    "(sample, mom),(mom)",
    [
        (nb.float32[:, :], nb.float32[:]),
        (nb.float64[:, :], nb.float64[:]),
    ],
)
def reduce_datas(other, data) -> None:
    for i in range(other.shape[0]):
        pushscalar.push_data(other[i, :], data)


@_decorator(
    "(sample, mom) -> (mom)",
    [
        (nb.float32[:, :], nb.float32[:]),
        (nb.float64[:, :], nb.float64[:]),
    ],
    writable=None,
)
def reduce_datas_fromzero(other, data) -> None:
    data[...] = 0.0
    for i in range(other.shape[0]):
        pushscalar.push_data(other[i, :], data)

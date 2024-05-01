# mypy: disable-error-code="no-untyped-call,no-untyped-def"
"""Vectorized pushers."""

from __future__ import annotations

from functools import partial

import numba as nb

from . import pushscalar_cov
from .utils import myguvectorize

_PARALLEL = False
_decorator = partial(myguvectorize, parallel=_PARALLEL)


@_decorator(
    "(),(),(),(mom0, mom1)",
    [
        (nb.float32, nb.float32, nb.float32, nb.float32[:, :]),
        (nb.float64, nb.float64, nb.float64, nb.float64[:, :]),
    ],
)
def push_val(w, x0, x1, data) -> None:
    pushscalar_cov.push_val(w, x0, x1, data)


@_decorator(
    "(sample),(sample),(sample),(mom0, mom1)",
    [
        (nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:, :]),
        (nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:, :]),
    ],
)
def reduce_vals(w, x0, x1, data) -> None:
    for i in range(len(w)):
        pushscalar_cov.push_val(w[i], x0[i], x1[i], data)


@_decorator(
    "(mom0, mom1), (mom0, mom1)",
    [
        (nb.float32[:, :], nb.float32[:, :]),
        (nb.float64[:, :], nb.float64[:, :]),
    ],
)
def push_data(other, data) -> None:
    pushscalar_cov.push_data(other, data)


@_decorator(
    "(sample, mom0, mom1), (mom0, mom1)",
    [
        (nb.float32[:, :, :], nb.float32[:, :]),
        (nb.float64[:, :, :], nb.float64[:, :]),
    ],
)
def reduce_datas(other, data) -> None:
    for i in range(other.shape[0]):
        pushscalar_cov.push_data(other[i, :, :], data)

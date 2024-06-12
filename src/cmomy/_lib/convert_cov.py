"""Convert between central and raw moments"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numba as nb

from .decorators import myguvectorize
from .utils import BINOMIAL_FACTOR

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..typing import FloatT


_PARALLEL = False
_vectorize = partial(myguvectorize, parallel=_PARALLEL)


@_vectorize(
    "(mom0,mom1) -> (mom0,mom1)",
    [
        (nb.float32[:, :], nb.float32[:, :]),
        (nb.float64[:, :], nb.float64[:, :]),
    ],
    writable=None,
)
def central_to_raw(central: NDArray[FloatT], raw: NDArray[FloatT]) -> None:
    ave0 = central[1, 0]
    ave1 = central[0, 1]

    for n in range(central.shape[0]):  # noqa: PLR1702
        for m in range(central.shape[1]):
            nm = n + m
            if nm <= 1:
                raw[n, m] = central[n, m]
            else:
                tmp = 0.0
                ave_i = 1.0
                for i in range(n + 1):
                    ave_j = 1.0
                    for j in range(m + 1):
                        nm_ij = nm - (i + j)
                        if nm_ij == 0:
                            # both zero order
                            tmp += ave_i * ave_j
                        elif nm_ij == 1:
                            # <dx**0 * dy**1> = 0
                            pass
                        else:
                            tmp += (
                                central[n - i, m - j]
                                * ave_i
                                * ave_j
                                * BINOMIAL_FACTOR[n, i]
                                * BINOMIAL_FACTOR[m, j]
                            )
                        ave_j *= ave1
                    ave_i *= ave0
                raw[n, m] = tmp


@_vectorize(
    "(mom0,mom1) -> (mom0,mom1)",
    [
        (nb.float32[:, :], nb.float32[:, :]),
        (nb.float64[:, :], nb.float64[:, :]),
    ],
    writable=None,
)
def raw_to_central(raw: NDArray[FloatT], central: NDArray[FloatT]) -> None:
    ave0 = raw[1, 0]
    ave1 = raw[0, 1]

    for n in range(raw.shape[0]):  # noqa: PLR1702
        for m in range(raw.shape[1]):
            nm = n + m
            if nm <= 1:
                central[n, m] = raw[n, m]
            else:
                tmp = 0.0
                ave_i = 1.0
                for i in range(n + 1):
                    ave_j = 1.0
                    for j in range(m + 1):
                        nm_ij = nm - (i + j)
                        if nm_ij == 0:
                            # both zero order
                            tmp += ave_i * ave_j
                        else:
                            tmp += (
                                raw[n - i, m - j]
                                * ave_i
                                * ave_j
                                * BINOMIAL_FACTOR[n, i]
                                * BINOMIAL_FACTOR[m, j]
                            )
                        ave_j *= -ave1
                    ave_i *= -ave0
                central[n, m] = tmp

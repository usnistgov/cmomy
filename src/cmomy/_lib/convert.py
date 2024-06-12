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
    "(mom) -> (mom)",
    [
        (nb.float32[:], nb.float32[:]),
        (nb.float64[:], nb.float64[:]),
    ],
    writable=None,
)
def central_to_raw(central: NDArray[FloatT], raw: NDArray[FloatT]) -> None:
    ave = central[1]
    raw[0] = central[0]
    raw[1] = ave

    for n in range(2, central.shape[0]):
        tmp = 0.0
        ave_i = 1.0
        for i in range(n - 1):
            tmp += central[n - i] * ave_i * BINOMIAL_FACTOR[n, i]
            ave_i *= ave

        # last two
        # <dx> = 0 so skip i = n-1
        # i = n
        tmp += ave_i * ave
        raw[n] = tmp


@_vectorize(
    "(mom) -> (mom)",
    [
        (nb.float32[:], nb.float32[:]),
        (nb.float64[:], nb.float64[:]),
    ],
    writable=None,
)
def raw_to_central(raw: NDArray[FloatT], central: NDArray[FloatT]) -> None:
    ave = raw[1]
    central[0] = raw[0]
    central[1] = ave

    for n in range(2, raw.shape[0]):
        tmp = 0.0
        ave_i = 1.0
        for i in range(n - 1):
            tmp += raw[n - i] * ave_i * BINOMIAL_FACTOR[n, i]
            ave_i *= -ave

        # last two
        # right now, ave_i = (-ave)**(n-1)
        # i = n-1
        # ave * ave_i * n
        # i = n
        # 1 * (-ave) * ave_i
        tmp += ave * ave_i * (n - 1)
        central[n] = tmp

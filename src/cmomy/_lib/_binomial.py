from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..options import OPTIONS

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from cmomy.core.typing import NDArrayAny


# * Binomial factors
def _binom(n: int, k: int) -> float:
    import math

    if n > k:
        return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
    if n == k:
        return 1.0
    # n < k
    return 0.0


def factory_binomial(order: int, dtype: DTypeLike = np.float64) -> NDArrayAny:
    """Create binomial coefs at given order."""
    out = np.zeros((order + 1, order + 1), dtype=dtype)
    for n in range(order + 1):
        for k in range(order + 1):
            out[n, k] = _binom(n, k)

    return out


BINOMIAL_FACTOR = factory_binomial(OPTIONS["nmax"])

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numba import njit

from ..options import OPTIONS

if TYPE_CHECKING:
    from typing import Any, Callable

    from numpy.typing import DTypeLike

    from ..typing import F, MyNDArray


def myjit(
    signature: str | list[str] | None = None,
    *,
    parallel: bool = False,
    inline: bool | None = None,
    **kws: Any,
) -> Callable[[F], F]:
    """Perform jitting."""
    if signature is not None:  # pragma: no cover
        kws["signature_or_function"] = signature  # pragma: no cover

    if parallel:
        kws["parallel"] = parallel
    if inline is not None:
        if inline:  # pragma: no cover
            kws["inline"] = "always"
        else:  # pragma: no cover
            kws["inline"] = "never"

    return cast(  # pyright: ignore[reportReturnType]
        "Callable[[F], F]",
        njit(fastmath=OPTIONS["fastmath"], cache=OPTIONS["cache"], **kws),
    )


def _binom(n: int, k: int) -> float:
    import math

    if n > k:
        return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
    if n == k:
        return 1.0
    # n < k
    return 0.0


def factory_binomial(order: int, dtype: DTypeLike = np.float64) -> MyNDArray:
    """Create binomial coefs at given order."""
    out = np.zeros((order + 1, order + 1), dtype=dtype)
    for n in range(order + 1):
        for k in range(order + 1):
            out[n, k] = _binom(n, k)

    return out


BINOMIAL_FACTOR = factory_binomial(OPTIONS["nmax"])

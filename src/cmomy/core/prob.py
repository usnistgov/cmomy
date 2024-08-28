"""Alternatives to ndtri -> nppf and ndtr -> ncdf"""

from __future__ import annotations

from math import erf, sqrt
from typing import TYPE_CHECKING, overload

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from .typing import DTypeLikeArg, FloatingT, NDArrayAny

# so don't have to use scipy....
_SQRT2 = sqrt(2)


def _ndtr_py(x: float) -> float:
    """Cumulative distribution of standard normal variable."""
    return 0.5 * (1 + erf(x / _SQRT2))


ndtr = np.vectorize(_ndtr_py, [float])

del _ndtr_py


@overload
def ndtri(p: ArrayLike, *, dtype: DTypeLikeArg[FloatingT]) -> NDArray[FloatingT]: ...
@overload
def ndtri(p: ArrayLike, *, dtype: DTypeLike = ...) -> NDArray[np.float64]: ...


# fmt: off
def ndtri(p: ArrayLike, *, dtype: DTypeLike = np.float64) -> NDArrayAny:
    """Inverse of ndtr."""
    p = np.asarray(p, dtype=dtype)
    out = np.empty_like(p, dtype=p.dtype)
    ix1 = (p < 0.02425) & (p > 0)
    oix1 = np.sqrt(-2 * np.log(p[ix1]))
    out[ix1] = (
            (((((-7.784894002430293e-03 * oix1 - 3.223964580411365e-01) * oix1
                - 2.400758277161838e00) * oix1 - 2.549732539343734e00) * oix1
                + 4.374664141464968e00) * oix1 + 2.938163982698783e00)
            / ((((7.784695709041462e-03 * oix1 + 3.224671290700398e-01) * oix1
                + 2.445134137142996e00) * oix1 + 3.754408661907416e00) * oix1 + 1)
        )

    ix2 = (p >= 0.02425) & (p <= 0.97575)
    oix2 = (p[ix2] - 0.5)
    sq = oix2 * oix2
    out[ix2] = (
            (((((-3.969683028665376e01 * sq + 2.209460984245205e02) * sq
                - 2.759285104469687e02) * sq + 1.383577518672690e02) * sq
                - 3.066479806614716e01) * sq + 2.506628277459239e00) * oix2
        ) / (((((-5.447609879822406e01 * sq + 1.615858368580409e02) * sq
                - 1.556989798598866e02) * sq + 6.680131188771972e01) * sq
                - 1.328068155288572e01) * sq + 1)

    ix3 = (p > 0.97575) & (p < 1)
    oix3 = np.sqrt(-2 * np.log(1 - p[ix3]))
    out[ix3] = -(
            (((((-7.784894002430293e-03 * oix3 - 3.223964580411365e-01) * oix3
                - 2.400758277161838e00) * oix3 - 2.549732539343734e00) * oix3
                + 4.374664141464968e00) * oix3 + 2.938163982698783e00)
            / ((((7.784695709041462e-03 * oix3 + 3.224671290700398e-01) * oix3
                + 2.445134137142996e00) * oix3 + 3.754408661907416e00) * oix3 + 1)
        )
    out[p == 0] = -np.inf
    out[p == 1] = np.inf

    return out

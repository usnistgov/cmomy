from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb
import numpy as np

from ..options import OPTIONS
from .decorators import is_in_unsafe_thread_pool, myjit

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray

    from ..typing import IntDTypeT, NDArrayAny


# * Threading safety.  Taken from https://github.com/numbagg/numbagg/blob/main/numbagg/decorators.py
def supports_parallel() -> bool:
    """
    Checks if system supports parallel numba functions.

    If an unsafe thread pool is detected, return ``False``.

    Returns
    -------
    bool :
        ``True`` if supports parallel.  ``False`` otherwise.
    """
    return not is_in_unsafe_thread_pool()


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


# * resampling
@myjit(
    # indices[rep, nsamp], freqs[rep, ndat]
    signature=[
        # (nb.int32[:, :], nb.int32[:, :]),
        (nb.int64[:, :], nb.int64[:, :]),
    ]
)
def randsamp_indices_to_freq(
    indices: NDArray[IntDTypeT], freqs: NDArray[IntDTypeT]
) -> None:
    nrep, ndat = freqs.shape

    assert indices.shape[0] == nrep
    assert indices.max() < ndat

    nsamp = indices.shape[1]
    for r in range(nrep):
        for d in range(nsamp):
            idx = indices[r, d]
            freqs[r, idx] += 1


@myjit(
    [
        # (nb.int32[:, :],),
        (nb.int64[:, :],),
    ]
)
def freq_to_index_start_end_scales(
    freq: NDArray[IntDTypeT],
) -> tuple[
    NDArray[IntDTypeT], NDArray[IntDTypeT], NDArray[IntDTypeT], NDArray[IntDTypeT]
]:
    ngroup = freq.shape[0]
    ndat = freq.shape[1]

    start = np.empty(ngroup, dtype=freq.dtype)
    end = np.empty(ngroup, dtype=freq.dtype)

    size = np.count_nonzero(freq)

    index = np.empty(size, dtype=freq.dtype)
    scale = np.empty(size, dtype=freq.dtype)

    idx = 0
    for group in range(ngroup):
        start[group] = idx
        count = 0
        for k in range(ndat):
            f = freq[group, k]
            if f > 0:
                index[idx] = k
                scale[idx] = f
                idx += 1
                count += 1
        end[group] = start[group] + count

    return index, start, end, scale

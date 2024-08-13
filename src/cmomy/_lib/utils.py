from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb
import numpy as np

from .decorators import myjit

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from cmomy.core.typing import IntDTypeT


# * resampling
@myjit(
    signature=[
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

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numba as nb
import numpy as np

from . import _push
from .decorators import myguvectorize, myjit

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..typing import FloatT

_PARALLEL = False
_vectorize = partial(myguvectorize, parallel=_PARALLEL)
_jit = partial(myjit, parallel=_PARALLEL)


@_vectorize(
    "(sample),(sample),(mom),(),() -> (sample, mom)",
    [
        (
            nb.float32[:],
            nb.float32[:],
            nb.float32[:],
            nb.int64,
            nb.int64,
            nb.float32[:, :],
        ),
        (
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.int64,
            nb.int64,
            nb.float64[:, :],
        ),
    ],
    writable=None,
)
def move_vals(
    x: NDArray[FloatT],
    w: NDArray[FloatT],
    data_tmp: NDArray[FloatT],
    window: int,
    min_count: int,
    out: NDArray[FloatT],
) -> None:
    nsamp = len(x)
    data_tmp[...] = 0.0
    count = 0
    min_count = max(min_count, 1)

    for i in range(min(window, nsamp)):
        wi = w[i]
        xi = x[i]
        if wi != 0:
            _push.push_val(xi, wi, data_tmp)
            count += 1

        if count >= min_count:
            out[i, ...] = data_tmp
        else:
            out[i, ...] = np.nan

    for i in range(window, nsamp):
        i_old = i - window

        wi = w[i]
        wold = w[i_old]

        wi_valid = wi != 0.0
        wold_valid = wold != 0.0

        if wi_valid and wold_valid:
            _push.push_val(x[i], wi, data_tmp)
            _push.push_val(x[i_old], -wold, data_tmp)

        elif wi_valid:
            _push.push_val(x[i], wi, data_tmp)
            count += 1
        elif wold_valid:
            _push.push_val(x[i_old], -wold, data_tmp)
            count -= 1

        if count >= min_count:
            out[i, ...] = data_tmp
        else:
            out[i, ...] = np.nan


@_vectorize(
    "(sample,mom),(mom),(),() -> (sample, mom)",
    [
        (
            nb.float32[:, :],
            nb.float32[:],
            nb.int64,
            nb.int64,
            nb.float32[:, :],
        ),
        (
            nb.float64[:, :],
            nb.float64[:],
            nb.int64,
            nb.int64,
            nb.float64[:, :],
        ),
    ],
    writable=None,
)
def move_data(
    data: NDArray[FloatT],
    data_tmp: NDArray[FloatT],
    window: int,
    min_count: int,
    out: NDArray[FloatT],
) -> None:
    nsamp = data.shape[0]
    data_tmp[...] = 0.0
    count = 0
    min_count = max(min_count, 1)

    for i in range(min(window, nsamp)):
        wi = data[i, 0]
        if wi != 0:
            _push.push_data(data[i, ...], data_tmp)
            count += 1

        if count >= min_count:
            out[i, ...] = data_tmp
        else:
            out[i, ...] = np.nan

    for i in range(window, nsamp):
        i_old = i - window

        wi = data[i, 0]
        wold = data[i_old, 0]

        wi_valid = wi != 0.0
        wold_valid = wold != 0.0

        if wi_valid and wold_valid:
            _push.push_data(data[i, ...], data_tmp)
            _push.push_data_scale(data[i_old, ...], -1.0, data_tmp)

        elif wi_valid:
            _push.push_data(data[i, ...], data_tmp)
            count += 1
        elif wold_valid:
            _push.push_data_scale(data[i_old, ...], -1.0, data_tmp)
            count -= 1

        if count >= min_count:
            out[i, ...] = data_tmp
        else:
            out[i, ...] = np.nan

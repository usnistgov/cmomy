from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

from cmomy.new import reduction

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture()
def x_vals(rng: np.random.Generator) -> NDArray[np.float64]:
    return rng.random((100, 2, 4))


dtypes_mark = pytest.mark.parametrize(
    ("dtype_in", "dtype", "out_dtype", "result_dtype"),
    [
        (np.float64, None, None, np.float64),
        (np.float32, None, None, np.float32),
        (np.float32, np.float64, None, np.float64),
        (np.float32, np.float32, np.float64, np.float64),
        (np.float32, None, np.float64, np.float64),
        (None, None, None, np.float64),
        (None, np.float32, None, np.float32),
        (None, None, np.float32, np.float32),
        (None, np.float64, np.float32, np.float32),
    ],
)


@dtypes_mark
@pytest.mark.parametrize(
    ("func_reduce", "reduce_style"),
    [
        (partial(reduction.reduce_vals, mom=3, axis=0), "vals"),
        (partial(reduction.reduce_data, mom_ndim=1, axis=0), "data"),
        (
            partial(
                reduction.reduce_data_grouped,
                mom_ndim=1,
                axis=0,
                by=np.zeros(100, dtype=np.int64),
            ),
            "grouped",
        ),
    ],
)
def test_reduce_dtype(
    x_vals: NDArray[np.float64],
    dtype_in,
    dtype,
    out_dtype,
    result_dtype,
    func_reduce,
    reduce_style,
) -> None:
    if reduce_style == "vals":
        out_shape = (*x_vals.shape[1:], 4)
    elif reduce_style == "data":
        out_shape = x_vals.shape[1:]
    elif reduce_style == "grouped":
        out_shape = (*x_vals.shape[1:-1], 1, *x_vals.shape[-1:])

    out = np.zeros(out_shape, dtype=out_dtype) if out_dtype is not None else None
    vals = x_vals.tolist() if dtype_in is None else x_vals.astype(dtype_in)

    r = func_reduce(
        vals,
        dtype=dtype,
        out=out,
    )
    assert isinstance(r, np.ndarray)
    assert r.dtype.type == result_dtype

    # xarray

    rx = func_reduce(
        xr.DataArray(vals),
        dtype=dtype,
        out=out,
    )
    assert isinstance(rx, xr.DataArray)
    assert rx.dtype.type == result_dtype

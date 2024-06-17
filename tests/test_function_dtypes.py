# mypy: disable-error-code="no-untyped-def, no-untyped-call"
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

from cmomy import reduction, resample

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
        # errors
        (np.float16, None, None, "error"),
        (None, np.float16, None, "error"),
        (None, np.float32, np.float16, "error"),
    ],
)


def _do_test(func, vals, dtype, out, result_dtype) -> None:
    if result_dtype == "error":
        with pytest.raises(ValueError, match=".*not supported.*"):
            func(vals, dtype=dtype, out=out)

    else:
        r = func(vals, dtype=dtype, out=out)

        assert isinstance(r, np.ndarray)
        assert r.dtype.type == result_dtype

        rx = func(
            xr.DataArray(vals),
            dtype=dtype,
            out=out,
        )
        assert isinstance(rx, xr.DataArray)
        assert rx.dtype.type == result_dtype


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

    _do_test(func_reduce, vals, dtype, out, result_dtype)


@dtypes_mark
@pytest.mark.parametrize("style", ["grouped", "indexed"])
def test_reduce_data_grouped_indexed_dtype(
    x_vals: NDArray[np.float64],
    dtype_in,
    dtype,
    out_dtype,
    result_dtype,
    style,
    rng,
) -> None:
    ngroup = 3
    by = rng.choice(ngroup, size=x_vals.shape[0])

    out_shape = (*x_vals.shape[1:-1], ngroup, *x_vals.shape[-1:])
    out = np.zeros(out_shape, dtype=out_dtype) if out_dtype is not None else None
    vals = x_vals.tolist() if dtype_in is None else x_vals.astype(dtype_in)

    if style == "grouped":
        func = partial(reduction.reduce_data_grouped, mom_ndim=1, by=by, axis=0)
    else:
        _, index, start, end = reduction.factor_by_to_index(by)
        func = partial(
            reduction.reduce_data_indexed,
            mom_ndim=1,
            index=index,
            group_start=start,
            group_end=end,
            axis=0,
        )

    _do_test(func, vals, dtype, out, result_dtype)


@dtypes_mark
@pytest.mark.parametrize("style", ["data", "vals"])
def test_resample_data(
    x_vals: NDArray[np.float64],
    dtype_in,
    dtype,
    out_dtype,
    result_dtype,
    style,
) -> None:
    ndat = x_vals.shape[0]
    nrep = 10
    freq = resample.random_freq(nrep=nrep, ndat=ndat)

    if style == "data":
        out_shape = (*x_vals.shape[1:-1], nrep, *x_vals.shape[-1:])
        func = partial(resample.resample_data, freq=freq, mom_ndim=1, axis=0)
    else:
        out_shape = (*x_vals.shape[1:], nrep, 4)
        func = partial(resample.resample_vals, freq=freq, mom=3, axis=0)

    out = np.zeros(out_shape, dtype=out_dtype) if out_dtype is not None else None
    vals = x_vals.tolist() if dtype_in is None else x_vals.astype(dtype_in)

    _do_test(func, vals, dtype, out, result_dtype)


@pytest.mark.parametrize("to", ["central", "raw"])
@dtypes_mark
def test_convert(
    x_vals,
    dtype_in,
    dtype,
    out_dtype,
    result_dtype,
    to,
) -> None:
    from cmomy import convert

    func = partial(convert.moments_type, mom_ndim=1, to=to)

    out = np.zeros_like(x_vals, dtype=out_dtype) if out_dtype is not None else None
    vals = x_vals.tolist() if dtype_in is None else x_vals.astype(dtype_in)

    _do_test(func, vals, dtype, out, result_dtype)

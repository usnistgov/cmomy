# mypy: disable-error-code="no-untyped-def, no-untyped-call"
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

import cmomy
from cmomy import CentralMomentsArray, CentralMomentsData
from cmomy.core.array_utils import select_dtype

from ._dataarray_set_utils import (
    do_wrap_method,
    do_wrap_raw,
    do_wrap_reduce_vals,
    do_wrap_resample_vals,
)

if TYPE_CHECKING:
    from cmomy.core.typing import NDArrayAny

# NOTE: This is for testing that methods give to correct output dtype
# metho(x, out=out, dtype=dtype) -> dtype

# * Dtypes
dtype_out_marks = pytest.mark.parametrize(
    ("dtype_array", "dtype_out", "dtype", "expected"),
    [
        # 32
        (np.float32, None, None, np.float32),
        (np.float64, np.float32, None, np.float32),
        (np.float16, np.float32, np.float64, np.float32),
        # 64
        (np.float64, None, None, np.float64),
        (np.float32, np.float64, None, np.float64),
        (np.float16, np.float64, np.float32, np.float64),
        # None
        (None, None, None, np.float64),
        # strings
        ("f4", None, None, np.float32),
        ("f8", "f4", None, np.float32),
        ("f2", "f4", "f8", np.float32),
        # errors
        (np.float16, None, None, "error"),
        (np.float64, np.float16, None, "error"),
        (np.float64, None, np.float16, "error"),
    ],
)


def _do_test(func, *args, expected, **kwargs):
    if expected == "error":
        with pytest.raises(ValueError, match=".*not supported.*"):
            func(*args, **kwargs)
    else:
        out = func(*args, **kwargs)
        if not isinstance(out, np.dtype):
            out = out.dtype
        assert out.type == expected


@dtype_out_marks
@pytest.mark.parametrize(
    ("func", "kws", "out_shape"),
    [
        (cmomy.reduction.reduce_vals, {"mom": 2, "axis": 0}, (1, 2, 3)),
        (cmomy.reduction.reduce_data, {"mom_ndim": 1, "axis": 0}, (1, 2)),
        (do_wrap_method("reduce"), {"mom_ndim": 1, "axis": 0}, (1, 2)),
        (
            cmomy.reduction.reduce_data_grouped,
            {"mom_ndim": 1, "by": [0, 0, 1, 1, 1], "axis": 0},
            (2, 1, 2),
        ),
        (
            do_wrap_method("reduce"),
            {"mom_ndim": 1, "by": [0, 0, 1, 1, 1], "axis": 0},
            (2, 1, 2),
        ),
        (
            cmomy.reduction.reduce_data_indexed,
            {
                "mom_ndim": 1,
                "index": range(5),
                "group_start": [0, 2],
                "group_end": [2, 5],
                "axis": 0,
            },
            (2, 1, 2),
        ),
        (
            cmomy.resample_vals,
            {"mom": 2, "move_axis_to_end": False, "axis": 0, "nrep": 10},
            (10, 1, 2, 3),
        ),
        (
            cmomy.resample_data,
            {"mom_ndim": 1, "move_axis_to_end": False, "axis": 0, "nrep": 10},
            (10, 1, 2),
        ),
        (
            do_wrap_method("resample_and_reduce"),
            {"mom_ndim": 1, "nrep": 10, "axis": 0},
            (10, 1, 2),
        ),
        (
            cmomy.resample.jackknife_vals,
            {"mom": 2, "move_axis_to_end": False, "axis": 0},
            (5, 1, 2, 3),
        ),
        (
            cmomy.resample.jackknife_data,
            {"mom_ndim": 1, "move_axis_to_end": False, "axis": 0},
            (5, 1, 2),
        ),
        (do_wrap_method("jackknife_and_reduce"), {"mom_ndim": 1, "axis": 0}, (5, 1, 2)),
        (
            cmomy.rolling.rolling_vals,
            {"mom": 2, "move_axis_to_end": False, "window": 2, "axis": 0},
            (5, 1, 2, 3),
        ),
        (
            cmomy.rolling.rolling_data,
            {"mom_ndim": 1, "move_axis_to_end": False, "window": 2, "axis": 0},
            (5, 1, 2),
        ),
        (
            cmomy.rolling.rolling_exp_vals,
            {"mom": 2, "move_axis_to_end": False, "alpha": 0.1, "axis": 0},
            (5, 1, 2, 3),
        ),
        (
            cmomy.rolling.rolling_exp_data,
            {"mom_ndim": 1, "move_axis_to_end": False, "alpha": 0.1, "axis": 0},
            (5, 1, 2),
        ),
        (cmomy.convert.moments_type, {"mom_ndim": 1}, (5, 1, 2)),
        (do_wrap_raw, {"mom_ndim": 1}, (5, 1, 2)),
        (cmomy.utils.vals_to_data, {"mom": 2}, (5, 1, 2, 3)),
        (cmomy.convert.cumulative, {"mom_ndim": 1, "axis": 0}, (5, 1, 2)),
        (do_wrap_method("cumulative"), {"mom_ndim": 1, "axis": 0}, (5, 1, 2)),
        (do_wrap_reduce_vals, {"mom": 2, "axis": 0}, (1, 2, 3)),
        (
            do_wrap_resample_vals,
            {"mom": 2, "move_axis_to_end": False, "nrep": 10, "axis": 0},
            (10, 1, 2, 3),
        ),
        (select_dtype, {"out": None}, (5, 1, 2)),
    ],
)
def test_functions_with_out(
    dtype_array, dtype_out, dtype, expected, func, kws, out_shape, as_dataarray
) -> None:
    shape = (5, 1, 2)
    x: NDArrayAny | xr.DataArray
    if as_dataarray:
        x = xr.DataArray(np.zeros(shape, dtype=dtype_array))
    else:
        x = np.zeros(shape, dtype=dtype_array)

    if isinstance(func, str):
        cls = CentralMomentsData if as_dataarray else CentralMomentsArray
        func = getattr(cls, func)

    kwargs = {"dtype": dtype, **kws}
    if dtype_out is not None and out_shape is not None:
        kwargs["out"] = np.zeros(out_shape, dtype=dtype_out)
    _do_test(func, x, expected=expected, **kwargs)


# without an out parameter
dtype_no_out_marks = pytest.mark.parametrize(
    ("dtype_array", "dtype", "expected"),
    [
        # 32
        (np.float32, None, np.float32),
        (np.float64, np.float32, np.float32),
        # 64
        (np.float64, None, np.float64),
        (np.float32, np.float64, np.float64),
        # None
        (None, None, np.float64),
        (None, np.float32, np.float32),
        # strings
        ("f4", None, np.float32),
        ("f8", "f4", np.float32),
        # errors
        (np.float16, None, "error"),
        (None, np.float16, "error"),
    ],
)


@dtype_no_out_marks
@pytest.mark.parametrize(
    ("func", "kws"),
    [
        (cmomy.convert.moments_to_comoments, {"mom": (1, -1)}),
        (do_wrap_method("moments_to_comoments"), {"mom": (1, -1)}),
    ],
)
def test_functions_without_out(
    dtype_array, dtype, expected, func, kws, as_dataarray
) -> None:
    shape = (5, 1, 3)
    x: NDArrayAny | xr.DataArray
    if as_dataarray:
        x = xr.DataArray(np.zeros(shape, dtype=dtype_array))
    else:
        x = np.zeros(shape, dtype=dtype_array)

    kwargs = {"dtype": dtype, **kws}
    _do_test(func, x, expected=expected, **kwargs)


# * Central routines


dtype_mark = pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.float32, np.float32),
        (np.dtype(np.float32), np.float32),
        ("f4", np.float32),
        (np.dtype("f4"), np.float32),
        (np.float64, np.float64),
        (np.dtype(np.float64), np.float64),
        ("f8", np.float64),
        (np.dtype("f8"), np.float64),
        (None, np.float64),
        (np.float16, "error"),
    ],
)

dtype_base_mark = pytest.mark.parametrize("dtype_base", [np.float32, np.float64])
cls_mark = pytest.mark.parametrize("cls", [CentralMomentsArray, CentralMomentsData])
use_out_mark = pytest.mark.parametrize("use_out", [False, True])

cls_numpy_mark = pytest.mark.parametrize("cls", [CentralMomentsArray])


@dtype_base_mark
@dtype_mark
@cls_mark
def test_astype(cls, dtype_base, dtype, expected) -> None:
    obj = cls.zeros(mom=3, val_shape=(2, 3), dtype=dtype_base)
    _do_test(obj.astype, dtype, expected=expected)


@dtype_mark
@cls_mark
def test_zeros_dtype(cls, dtype, expected) -> None:
    func = partial(cls.zeros, mom=3, val_shape=(2, 3), dtype=dtype)
    _do_test(func, expected=expected)


@cls_numpy_mark
@dtype_base_mark
@dtype_mark
def test_init(cls, dtype_base, dtype, expected) -> None:
    data = np.zeros((2,), dtype=dtype_base)
    if cls == CentralMomentsData:
        data = xr.DataArray(data)  # type: ignore[assignment]

    func = partial(cls, data, mom_ndim=1, dtype=dtype)
    if dtype is None:
        expected = dtype_base
    _do_test(func, expected=expected)


@cls_mark
@dtype_base_mark
@dtype_mark
def test_new_like(cls, dtype_base, dtype, expected) -> None:
    data = np.zeros((2, 3, 4), dtype=dtype)
    data_base = np.zeros((2, 3, 4), dtype=dtype_base)

    c = cls.zeros(mom=3, val_shape=(2, 3), dtype=dtype_base)
    if cls == CentralMomentsData:
        xdata = xr.DataArray(data, dims=c.dims)

    assert c.dtype.type == dtype_base

    if expected == "error":
        with pytest.raises(ValueError, match=".*not supported.*"):
            c.new_like(data, dtype=dtype)
    elif dtype is None:
        assert c.new_like().dtype.type == dtype_base
        assert c.new_like(dtype=dtype).dtype.type == dtype_base
        assert c.new_like(obj=data).dtype.type == expected
        assert c.new_like(obj=data, dtype=dtype).dtype.type == expected
        assert c.new_like(obj=data_base).dtype.type == dtype_base
        assert c.new_like(obj=data_base, dtype=dtype).dtype.type == dtype_base
        if cls == CentralMomentsData:
            assert c.new_like(obj=xdata, dtype=dtype).dtype.type == expected

    else:
        assert c.new_like().dtype.type == dtype_base
        assert c.new_like(dtype=dtype).dtype.type == expected
        assert c.new_like(obj=data).dtype.type == expected
        assert c.new_like(obj=data, dtype=dtype).dtype.type == expected
        assert c.new_like(obj=data_base).dtype.type == dtype_base
        assert c.new_like(obj=data_base, dtype=dtype).dtype.type == expected
        if cls == CentralMomentsData:
            assert c.new_like(obj=xdata).dtype.type == expected
            assert c.new_like(obj=xdata, dtype=dtype).dtype.type == expected

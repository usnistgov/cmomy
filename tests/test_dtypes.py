# mypy: disable-error-code="no-untyped-def, no-untyped-call"
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

import cmomy
from cmomy import CentralMoments, xCentralMoments
from cmomy._utils import select_dtype

if TYPE_CHECKING:
    from cmomy.typing import NDArrayAny

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
    ("func", "style", "kws", "out_shape"),
    [
        (cmomy.reduction.reduce_vals, "reduce", {"mom": 2}, (1, 2, 3)),
        (cmomy.reduction.reduce_data, "reduce", {"mom_ndim": 1}, (1, 2)),
        (
            cmomy.reduction.reduce_data_grouped,
            "reduce",
            {"mom_ndim": 1, "by": [0, 0, 1, 1, 1]},
            (2, 1, 2),
        ),
        (
            cmomy.reduction.reduce_data_indexed,
            "reduce",
            {
                "mom_ndim": 1,
                "index": range(5),
                "group_start": [0, 2],
                "group_end": [2, 5],
            },
            (2, 1, 2),
        ),
        (
            cmomy.resample_vals,
            "resample",
            {"mom": 2, "move_axis_to_end": False},
            (10, 1, 2, 3),
        ),
        (
            cmomy.resample_data,
            "resample",
            {"mom_ndim": 1, "move_axis_to_end": False},
            (10, 1, 2),
        ),
        (
            cmomy.resample.jackknife_vals,
            "jackknife",
            {"mom": 2, "move_axis_to_end": False},
            (5, 1, 2, 3),
        ),
        (
            cmomy.resample.jackknife_data,
            "jackknife",
            {"mom_ndim": 1, "move_axis_to_end": False},
            (5, 1, 2),
        ),
        (
            cmomy.rolling.rolling_vals,
            "rolling",
            {"mom": 2, "move_axis_to_end": False, "window": 2},
            (5, 1, 2, 3),
        ),
        (
            cmomy.rolling.rolling_data,
            "rolling",
            {"mom_ndim": 1, "move_axis_to_end": False, "window": 2},
            (5, 1, 2),
        ),
        (
            cmomy.rolling.rolling_exp_vals,
            "rolling",
            {"mom": 2, "move_axis_to_end": False, "alpha": 0.1},
            (5, 1, 2, 3),
        ),
        (
            cmomy.rolling.rolling_exp_data,
            "rolling",
            {"mom_ndim": 1, "move_axis_to_end": False, "alpha": 0.1},
            (5, 1, 2),
        ),
        (cmomy.convert.moments_type, "convert", {"mom_ndim": 1}, (5, 1, 2)),
        (cmomy.convert.vals_to_data, "convert", {"mom": 2}, (5, 1, 2, 3)),
        (cmomy.convert.cumulative, "reduce", {"mom_ndim": 1}, (5, 1, 2)),
        (
            "from_resample_vals",
            "resample",
            {"mom": 2, "move_axis_to_end": False},
            (10, 1, 2, 3),
        ),
        ("from_vals", "reduce", {"mom": 2}, (1, 2, 3)),
        (select_dtype, "convert", {"out": None}, (5, 1, 2)),
    ],
)
def test_functions_with_out(
    dtype_array, dtype_out, dtype, expected, func, style, kws, out_shape, as_dataarray
) -> None:
    shape = (5, 1, 2)
    x: NDArrayAny | xr.DataArray
    if as_dataarray:
        x = xr.DataArray(np.zeros(shape, dtype=dtype_array))
    else:
        x = np.zeros(shape, dtype=dtype_array)

    if isinstance(func, str):
        cls = xCentralMoments if as_dataarray else CentralMoments
        func = getattr(cls, func)

    kwargs = {"dtype": dtype, **kws}
    if style != "convert":
        kwargs["axis"] = 0
    if style == "resample":
        kwargs["freq"] = cmomy.random_freq(ndat=5, nrep=10)
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
    ("func", "style", "kws"),
    [
        (cmomy.convert.moments_to_comoments, "convert", {"mom": (1, -1)}),
    ],
)
def test_functions_without_out(
    dtype_array, dtype, expected, func, style, kws, as_dataarray
) -> None:
    shape = (5, 1, 3)
    x: NDArrayAny | xr.DataArray
    if as_dataarray:
        x = xr.DataArray(np.zeros(shape, dtype=dtype_array))
    else:
        x = np.zeros(shape, dtype=dtype_array)

    kwargs = {"dtype": dtype, **kws}
    if style != "convert":
        kwargs["axis"] = 0
    if style == "resample":
        kwargs["freq"] = cmomy.random_freq(ndat=5, nrep=10)

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
cls_mark = pytest.mark.parametrize("cls", [CentralMoments, xCentralMoments])
use_out_mark = pytest.mark.parametrize("use_out", [False, True])


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


@cls_mark
@dtype_base_mark
@dtype_mark
def test_init(cls, dtype_base, dtype, expected) -> None:
    data = np.zeros((2,), dtype=dtype_base)
    if cls == xCentralMoments:
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
    if cls == xCentralMoments:
        xdata = xr.DataArray(data)

    c = cls.zeros(mom=3, val_shape=(2, 3), dtype=dtype_base)

    assert c.dtype.type == dtype_base

    if expected == "error":
        with pytest.raises(ValueError, match=".*not supported.*"):
            c.new_like(data, dtype=dtype)
    elif dtype is None:
        assert c.new_like().dtype.type == dtype_base
        assert c.new_like(dtype=dtype).dtype.type == dtype_base
        assert c.new_like(data=data).dtype.type == expected
        assert c.new_like(data=data, dtype=dtype).dtype.type == expected
        assert c.new_like(data=data_base).dtype.type == dtype_base
        assert c.new_like(data=data_base, dtype=dtype).dtype.type == dtype_base
        if cls == xCentralMoments:
            assert c.new_like(data=xdata, dtype=dtype).dtype.type == expected

    else:
        assert c.new_like().dtype.type == dtype_base
        assert c.new_like(dtype=dtype).dtype.type == expected
        assert c.new_like(data=data).dtype.type == expected
        assert c.new_like(data=data, dtype=dtype).dtype.type == expected
        assert c.new_like(data=data_base).dtype.type == dtype_base
        assert c.new_like(data=data_base, dtype=dtype).dtype.type == expected
        if cls == xCentralMoments:
            assert c.new_like(data=xdata).dtype.type == expected
            assert c.new_like(data=xdata, dtype=dtype).dtype.type == expected

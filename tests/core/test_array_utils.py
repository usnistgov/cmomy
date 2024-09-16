# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cmomy.core import array_utils


# * catch all args only test
def _do_test(func, *args, expected=None, match=None, **kwargs):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected, match=match):
            func(*args, **kwargs)
    else:
        assert func(*args, **kwargs) == expected


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        ("c", "C"),
        ("F", "F"),
        (None, None),
        ("k", None),
        ("anything", None),
    ],
)
def test_arrayorder_to_arrayorder_cf(arg, expected) -> None:
    _do_test(array_utils.arrayorder_to_arrayorder_cf, arg, expected=expected)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((0, 3), 0),
        ((1, 3), 1),
        ((2, 3), 2),
        ((3, 3), ValueError),
        ((-1, 3), 2),
        ((-2, 3), 1),
        ((-3, 3), 0),
        ((-4, 3), ValueError),
        ((0, 3, 1), 0),
        ((1, 3, 1), 1),
        ((2, 3, 1), ValueError),
        ((-1, 3, 1), 1),
        ((-2, 3, 1), 0),
        ((-3, 3, 1), ValueError),
        ((0, 3, 2), 0),
        ((1, 3, 2), ValueError),
        ((-1, 3, 2), 0),
        ((-2, 3, 2), ValueError),
    ],
)
def test_normalize_axis_index(args, expected):
    _do_test(array_utils.normalize_axis_index, *args, expected=expected)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (((0, 1, 2), 3), (0, 1, 2)),
        ((3, 3), ValueError),
        (((-1, -2, -3), 3), (2, 1, 0)),
        ((-4, 3), ValueError),
        (((0, 1), 3, 1), (0, 1)),
        ((2, 3, 1), ValueError),
        (((-1, -2), 3, 1), (1, 0)),
        ((-3, 3, 1), ValueError),
        ((0, 3, 2), (0,)),
        ((1, 3, 2), ValueError),
        ((-1, 3, 2), (0,)),
    ],
)
def test_normalize_axis_tuple(args, expected):
    _do_test(array_utils.normalize_axis_tuple, *args, expected=expected)


@pytest.mark.parametrize(
    ("args", "expected"), [((0, 4), -4), ((-1, 4), -1), ((2, 4), -2)]
)
def test_positive_to_negative_index(args, expected) -> None:
    _do_test(array_utils.positive_to_negative_index, *args, expected=expected)


@pytest.mark.parametrize(
    ("args", "kwargs", "expected"),
    [
        (
            (),
            {"mom_ndim": 1, "axis": -2},
            [(-2, -1), (-1,)],
        ),
        (
            (),
            {"mom_ndim": 2, "axis": -3},
            [(-3, -2, -1), (-2, -1)],
        ),
        (
            ((), -2),
            {"mom_ndim": 1, "axis": -3},
            [(-3, -1), (), (-2,), (-1,)],
        ),
        (
            (),
            {"mom_ndim": 1, "axis": -2, "out_has_axis": True},
            [(-2, -1), (-2, -1)],
        ),
        (
            (),
            {"mom_ndim": 2, "axis": -3, "out_has_axis": True},
            [(-3, -2, -1), (-3, -2, -1)],
        ),
        (
            ((), -2),
            {"mom_ndim": 1, "axis": -3, "out_has_axis": True},
            [(-3, -1), (), (-2,), (-3, -1)],
        ),
    ],
)
def test_axes_data_reduction(args, kwargs, expected) -> None:
    _do_test(array_utils.axes_data_reduction, *args, expected=expected, **kwargs)


def _e(dtype):
    return np.empty(2, dtype=dtype)


def _x(dtype):
    return xr.DataArray(_e(dtype))


def _s(dtype):
    return xr.Dataset({"x": _x(dtype)})


@pytest.mark.parametrize(
    ("x", "out", "dtype", "expected"),
    [
        # array
        (_e(np.float64), None, None, np.float64),
        (_e(np.float64), _e(np.float32), None, np.float32),
        (_e(np.float64), None, np.float32, np.float32),
        (_e(np.float16), _e(np.float32), np.float64, np.float32),
        (_e(np.float16), None, None, ValueError),
        # dataarray
        (_x(np.float64), None, None, np.float64),
        (_x(np.float64), _x(np.float32), None, np.float32),
        (_x(np.float64), None, np.float32, np.float32),
        (_x(np.float16), _x(np.float32), np.float64, np.float32),
        (_x(np.float16), None, None, ValueError),
        # dataset
        (_s(np.float64), None, None, None),
        (_s(np.float64), _s(np.float32), None, None),
        (_s(np.float64), None, np.float32, np.float32),
        (_s(np.float16), _s(np.float32), np.float64, np.float64),
        (_s(np.float16), None, None, None),
    ],
)
def test_select_dtype(x, out, dtype, expected) -> None:
    def func(*args, **kwargs):
        out = array_utils.select_dtype(*args, **kwargs)
        return out if out is None else out.type

    _do_test(func, x, expected=expected, out=out, dtype=dtype)


@pytest.mark.parametrize(
    ("shape", "axis", "out"),
    [
        ((10, 3), 0, (1, 10, 3)),
        ((10, 3), 1, (10, 1, 3)),
        ((2, 3, 4), 0, (1, 2, 3, 4)),
        ((2, 3, 4), 1, (2, 1, 3, 4)),
        ((2, 3, 4), 2, (2, 3, 1, 4)),
    ],
)
def test_optional_keepdims(shape, axis, out) -> None:
    x = np.empty(shape)

    for keepdims in [True, False]:
        assert array_utils.optional_keepdims(x, axis=axis, keepdims=keepdims).shape == (
            out if keepdims else shape
        )


@pytest.mark.parametrize("shape", [(10,), (2, 2)])
@pytest.mark.parametrize("dtype", [np.float32, None])
@pytest.mark.parametrize("broadcast", [True, False])
def test_dummy_array(shape, dtype, broadcast) -> None:
    out = array_utils.dummy_array(shape, dtype, broadcast)

    assert out.shape == shape
    assert out.dtype.type is (dtype or np.float64)
    assert out.flags["OWNDATA"] is not broadcast

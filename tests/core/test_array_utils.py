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
        ((-1j, 3), 2),
        ((-2, 3), 1),
        ((-3, 3), 0),
        ((-4, 3), ValueError),
        ((0, 3, 1), 0),
        ((1, 3, 1), 1),
        ((2, 3, 1), 2),
        ((2j, 3, 1), ValueError),
        ((-1, 3, 1), 2),
        ((-2, 3, 1), 1),
        ((-3, 3, 1), 0),
        ((-4, 3, 1), ValueError),
        ((-1j, 3, 1), 1),
        ((-2j, 3, 1), 0),
        ((-3j, 3, 1), ValueError),
        ((0, 3, 2), 0),
        ((-1j, 3, 2), 0),
        ((-2j, 3, 2), ValueError),
        ((1, 3, 2), 1),
        ((1j, 3, 2), ValueError),
        ((-1, 3, 2), 2),
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
        (((0, -2), 3, 1), (0, 1)),
        (((0, -1j), 3, 1), (0, 1)),
        (((-1, -2), 3, 1), (2, 1)),
        (((-1j, -2j), 3, 1), (1, 0)),
        ((-3, 3, 1), (0,)),
        ((-3j, 3, 1), ValueError),
        ((0, 3, 2), (0,)),
        ((-1j, 3, 2), (0,)),
        ((1, 3, 2), (1,)),
        ((1j, 3, 2), ValueError),
        (((0, 0), 2), ValueError),
    ],
)
def test_normalize_axis_tuple(args, expected):
    _do_test(array_utils.normalize_axis_tuple, *args, expected=expected)


@pytest.mark.parametrize(
    ("args", "expected"), [((0, 4), -4), ((-1, 4), -1), ((2, 4), -2)]
)
def test_positive_to_negative_index(args, expected) -> None:
    _do_test(array_utils.positive_to_negative_index, *args, expected=expected)


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

    for keepdims in (True, False):
        assert array_utils.optional_keepdims(x, axis=axis, keepdims=keepdims).shape == (
            out if keepdims else shape
        )


@pytest.mark.parametrize(
    ("data", "src", "dest", "expected"),
    [
        (5, (0, 1), (-1, -2), [2, 3, 4, 1, 0]),
        (5, (-2, 0), (0, -2), [3, 1, 2, 0, 4]),
        ((2, 3, 4, 5, 6), (0, 1), (-1, -2), [4, 5, 6, 3, 2]),
        ((2, 3, 4, 5, 6), (-2, 0), (0, -2), [5, 3, 4, 2, 6]),
    ],
)
def test_reorder(data, src, dest, expected) -> None:
    assert array_utils.reorder(data, src, dest) == expected

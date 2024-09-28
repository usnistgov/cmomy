# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cmomy.core import validate
from cmomy.core.missing import MISSING


# * catch all args only test
def _do_test(func, *args, expected=None, match=None, **kwargs):
    if isinstance(expected, type):
        with pytest.raises(expected, match=match):
            func(*args, **kwargs)
    else:
        assert func(*args, **kwargs) == expected


def test_raise_if_wrong_value() -> None:
    x = np.ones((2, 3, 4))
    validate.raise_if_wrong_value(x.shape, (2, 3, 4))
    with pytest.raises(ValueError):
        validate.raise_if_wrong_value(x.shape, (1, 2, 3, 4))


@pytest.mark.parametrize(
    "x",
    [
        [1, 2],
        np.zeros(3),
        xr.DataArray([0]),
        xr.DataArray([0]).to_dataset(name="data"),
    ],
)
def test_typeguards(x) -> None:
    assert validate.is_ndarray(x) is isinstance(x, np.ndarray)
    assert validate.is_dataarray(x) is isinstance(x, xr.DataArray)
    assert validate.is_dataset(x) is isinstance(x, xr.Dataset)
    assert validate.is_xarray(x) is isinstance(x, (xr.Dataset, xr.DataArray))


# * validate not none
@pytest.mark.parametrize(
    ("arg", "expected", "match"),
    [
        (None, TypeError, ".*is not supported"),
        ("a", "a", None),
        (1, 1, None),
    ],
)
def test_validate_not_none(arg, expected, match) -> None:
    _do_test(validate.validate_not_none, arg, expected=expected, match=match)


# * Moment validation
@pytest.mark.parametrize(
    ("arg", "expected", "match"),
    [
        (0, ValueError, ".* must be either 1 or 2"),
        (3, ValueError, ".* must be either 1 or 2"),
        (1, 1, None),
        (2, 2, None),
    ],
)
def test_validate_mom_ndim(arg, expected, match) -> None:
    _do_test(validate.validate_mom_ndim, arg, expected=expected, match=match)


@pytest.mark.parametrize(
    ("arg", "expected", "match"),
    [
        (3, (3,), None),
        ((3,), (3,), None),
        ([3], (3,), None),
        ([3, 3], (3, 3), None),
        (0, ValueError, r".* must be an integer, .*"),
        ((0,), ValueError, r".* must be an integer, .*"),
        ((3, 0), ValueError, r".* must be an integer, .*"),
        ((0, 3), ValueError, r".* must be an integer, .*"),
        ([3, 3, 3], ValueError, r".* must be an integer, .*"),
    ],
)
def test_is_mom_tuple(arg, expected, match) -> None:
    _do_test(validate.validate_mom, arg, expected=expected, match=match)


@pytest.mark.parametrize(
    ("kws", "expected"),
    [
        ({"mom": 3, "mom_ndim": 1}, ((3,), 1)),
        ({"mom": (3,), "mom_ndim": 1}, ((3,), 1)),
        ({"mom_ndim": 1, "shape": (1, 2, 3)}, ((2,), 1)),
        ({"mom_ndim": 2, "shape": (1, 2, 3)}, ((1, 2), 2)),
        ({"mom_ndim": 2, "shape": (2, 3)}, ((1, 2), 2)),
        ({"mom": (2, 2)}, ((2, 2), 2)),
        ({"mom_ndim": 1, "shape": (2, 1)}, ValueError),
        ({"mom_ndim": 2, "shape": (2, 1, 1)}, ValueError),
        ({"mom_ndim": 2, "shape": (3,)}, ValueError),
        ({"mom": 0, "mom_ndim": 1}, ValueError),
        ({"mom": 3, "mom_ndim": 2}, ValueError),
        ({"mom": (3, 0), "mom_ndim": 2}, ValueError),
        ({"mom": (3, 3), "mom_ndim": 1}, ValueError),
        ({"mom": None, "mom_ndim": None}, ValueError),
        ({"mom": None, "mom_ndim": 1}, ValueError),
        ({"mom": None, "mom_ndim": 2, "shape": (2,)}, ValueError),
        ({"mom": None, "mom_ndim": 3, "shape": (2, 3, 4)}, ValueError),
        ({"mom": (2, 2, 2), "mom_ndim": None}, ValueError),
        ({"mom": (2, 2), "mom_ndim": 1}, ValueError),
    ],
)
def test_validate_mom_and_mom_ndim(kws, expected) -> None:
    _do_test(validate.validate_mom_and_mom_ndim, expected=expected, **kws)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((None, 1), ("mom_0",)),
        ((None, 2), ("mom_0", "mom_1")),
        (("a", 1), ("a",)),
        ((("a", "b"), 2), ("a", "b")),
        ((["a"], 1), ("a",)),
        ((["a", "b"], 2), ("a", "b")),
        (({"a"}, 1), TypeError),
        ((["a", "b"], 1), ValueError),
        (("a", 2), ValueError),
        ((("a,"), 2), ValueError),
    ],
)
def test_validate_mom_dims(args, expected):
    _do_test(validate.validate_mom_dims, *args, expected=expected)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((None, 1), (("mom_0",), 1)),
        ((None, 2), (("mom_0", "mom_1"), 2)),
        (("a", None), (("a",), 1)),
        ((("a", "b"), None), (("a", "b"), 2)),
        ((["a"], None), (("a",), 1)),
        ((["a", "b"], None), (("a", "b"), 2)),
        ((["a", "b"], 1), ValueError),
        (("a", 2), ValueError),
        ((("a,"), 2), ValueError),
        ((None, None), (None, None)),
        ((None, None, None, 1), (("mom_0",), 1)),
    ],
)
def test_validate_optional_mom_dims_mom_ndim(args, expected):
    _do_test(validate.validate_optional_mom_dims_and_mom_ndim, *args, expected=expected)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((None, 1), (("mom_0",), 1)),
        ((None, 2), (("mom_0", "mom_1"), 2)),
        (("a", None), (("a",), 1)),
        ((("a", "b"), None), (("a", "b"), 2)),
        ((["a"], None), (("a",), 1)),
        ((["a", "b"], None), (("a", "b"), 2)),
        ((["a", "b"], 1), ValueError),
        (("a", 2), ValueError),
        ((("a,"), 2), ValueError),
        ((None, None), ValueError),
        ((None, None, None, 1), (("mom_0",), 1)),
    ],
)
def test_validate_mom_dims_mom_ndim(args, expected):
    _do_test(validate.validate_mom_dims_and_mom_ndim, *args, expected=expected)


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        (1, 1),
        ((1, 2), (1, 2)),
        (None, None),
        (MISSING, TypeError),
    ],
)
def test_validate_axis_mult(arg, expected) -> None:
    _do_test(validate.validate_axis_mult, arg, expected=expected)

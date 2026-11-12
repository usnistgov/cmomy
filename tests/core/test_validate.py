from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cmomy
from cmomy._lib._binomial import MOM_MAX
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
    ("arg", "expected"),
    [
        (1, (1,)),
        ((1,), (1,)),
        ((1, 2), (1, 2)),
        ([1, 2], (1, 2)),
        (0, ValueError),
        ([0, 1], ValueError),
        ((_ for _ in (1, 2)), (1, 2)),
    ],
)
def test_validate_mom(arg, expected) -> None:
    _do_test(validate.validate_mom, arg, expected=expected)


def test_validate_mom_max_order() -> None:
    assert validate.validate_mom(MOM_MAX) == (MOM_MAX,)
    assert validate.validate_mom((MOM_MAX, MOM_MAX)) == (MOM_MAX, MOM_MAX)

    match = r".*is greater than the maximum order.*"
    for mom in (MOM_MAX + 1, (MOM_MAX + 1,), (1, MOM_MAX + 1), (MOM_MAX + 1, 1)):
        with pytest.raises(ValueError, match=match):
            validate.validate_mom(mom)


def test_reduce_vals_above_max_order() -> None:
    # orders above MOM_MAX index BINOMIAL_FACTOR out of bounds and return
    # garbage, so they must be rejected instead.
    x = np.arange(10, dtype=np.float64)
    assert cmomy.reduce_vals(x, mom=MOM_MAX, axis=0).shape == (MOM_MAX + 1,)
    with pytest.raises(ValueError, match=r".*is greater than the maximum order.*"):
        cmomy.reduce_vals(x, mom=MOM_MAX + 1, axis=0)


@pytest.mark.parametrize(
    ("arg", "expected"),
    [(0, (0,)), ((1, 2), (1, 2)), ((1, 2, 3), ValueError), ([0], (0,))],
)
def test_validate_mom_axes(arg, expected) -> None:
    _do_test(validate.validate_mom_axes, arg, expected=expected)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            (1,),
            (1, (-1,)),
        ),
        (
            (2,),
            (2, (-2, -1)),
        ),
        (
            (None, 0),
            (1, (0,)),
        ),
        (
            (None, [1, 2]),
            (2, (1, 2)),
        ),
        (
            (1, [1, 2]),
            ValueError,
        ),
        (
            (2, [1, 2]),
            (2, (1, 2)),
        ),
        (
            (None, None, 1),
            (1, (-1,)),
        ),
        (
            (2, 1),
            ValueError,
        ),
    ],
)
def test_validate_mom_ndim_and_mom_axes(args, expected) -> None:
    _do_test(validate.validate_mom_ndim_and_mom_axes, *args, expected=expected)


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
        ((None, None), ValueError),
        ((None, None, None, 1), (("mom_0",), 1)),
    ],
)
def test_validate_mom_dims_mom_ndim(args, expected):
    _do_test(validate.validate_mom_dims_and_mom_ndim, *args, expected=expected)


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        (1, (1,)),
        ((1, 2), (1, 2)),
        (None, TypeError),
        (MISSING, TypeError),
    ],
)
def test_validate_axis_mult(arg, expected) -> None:
    _do_test(validate.validate_axis_mult, arg, expected=expected)

# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cmomy.core.moment_params import (
    MomParamsArray,
    MomParamsXArray,
    MomParamsXArrayOptional,
    default_mom_params_xarray,
    factory_mom_params,
)
from cmomy.core.validate import is_xarray


# * catch all args only test
def _do_test(func, *args, expected=None, match=None, **kwargs):
    if isinstance(expected, type):
        with pytest.raises(expected, match=match):
            func(*args, **kwargs)
    else:
        assert func(*args, **kwargs) == expected


@pytest.mark.parametrize(
    ("kws", "expected"),
    [
        ({"ndim": 1}, (1, (-1,))),
        ({"ndim": 2}, (2, (-2, -1))),
        ({"axes": -2}, (1, (-2,))),
        ({"axes": [1, 2]}, (2, (1, 2))),
        ({}, ValueError),
        ({"default_ndim": 1}, (1, (-1,))),
        ({"ndim": 2, "axes": -1}, ValueError),
    ],
)
def test_MomParamsArray(kws, expected) -> None:
    def _func(*args, **kwargs):
        m = MomParamsArray.factory(*args, **kwargs)
        return m.ndim, m.axes

    _do_test(_func, expected=expected, **kws)


@pytest.mark.parametrize(
    ("kws", "expected"),
    [
        ({"ndim": 1}, (1, ("mom_0",))),
        ({"ndim": 2}, (2, ("mom_0", "mom_1"))),
        ({"dims": "a"}, (1, ("a",))),
        ({"dims": ("a", "b")}, (2, ("a", "b"))),
        ({}, ValueError),
        ({"default_ndim": 1}, (1, ("mom_0",))),
        ({"ndim": 2, "dims": "a"}, ValueError),
    ],
)
def test_MomParamsXArray(kws, expected) -> None:
    def _func(*args, **kwargs):
        m = MomParamsXArray.factory(*args, **kwargs)
        return (
            m.ndim,
            m.dims,
        )

    _do_test(_func, expected=expected, **kws)


@pytest.mark.parametrize(
    "data",
    [xr.DataArray(np.zeros((1, 2, 3)), dims=["a", "b", "c"])],
)
@pytest.mark.parametrize(
    ("kws", "expected"),
    [
        ({"ndim": 1}, (1, ("c",))),
        ({"ndim": 2}, (2, ("b", "c"))),
        ({"dims": "a"}, (1, ("a",))),
        ({"dims": ("mom_0", "mom_1")}, (2, ("mom_0", "mom_1"))),
        ({}, ValueError),
        ({"default_ndim": 1}, (1, ("c",))),
        ({"ndim": 2, "dims": "a"}, ValueError),
        ({"axes": 0}, (1, ("a",))),
        ({"axes": (0, 1)}, (2, ("a", "b"))),
    ],
)
def test_MomParamsXArray_data(data, kws, expected) -> None:
    def _func(*args, **kwargs):
        m = MomParamsXArray.factory(*args, **kwargs)
        return (
            m.ndim,
            m.dims,
        )

    _do_test(_func, expected=expected, data=data, **kws)


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
def test_MomParamsXArray_other(args, expected) -> None:
    def _func(*args, **kwargs):
        m = MomParamsXArray.factory(*args, **kwargs)
        return m.dims, m.ndim

    kws = dict(zip(["dims", "ndim", "axes", "default_ndim"], args))
    _do_test(_func, expected=expected, **kws)


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
def test_MomParamsXArrayOptional(args, expected) -> None:
    def _func(*args, **kwargs):
        m = MomParamsXArrayOptional.factory(*args, **kwargs)
        return m.dims, m.ndim

    kws = dict(zip(["dims", "ndim", "axes", "default_ndim"], args))
    _do_test(_func, expected=expected, **kws)


# * select axis/dim
def _wrap_select_method(method):
    def func(*args, **kws):
        if "mom_dims" in kws:
            kws = kws.copy()
            mom_dims = kws.pop("mom_dims")
            mom_params = MomParamsXArray.factory(dims=mom_dims)
        else:
            mom_params = default_mom_params_xarray

        return getattr(mom_params, method)(*args, **kws)

    return func


@pytest.mark.parametrize(
    "data", [xr.DataArray(np.zeros((1, 1, 1)), dims=("a", "b", "mom"))]
)
@pytest.mark.parametrize(
    ("kws", "expected"),
    [
        ({}, ValueError),
        ({"default_axis": 0, "default_dim": "hello"}, ValueError),
        ({"axis": 0, "dim": "a"}, ValueError),
        ({"default_axis": 0}, (0, "a")),
        ({"default_axis": -1}, (2, "mom")),
        ({"default_dim": "a"}, (0, "a")),
        ({"default_dim": "mom"}, (2, "mom")),
        ({"axis": -1}, (2, "mom")),
        ({"axis": -1j, "mom_dims": ("mom",)}, (1, "b")),
        (
            {
                "axis": -1j,
                "mom_dims": (
                    "b",
                    "mom",
                ),
            },
            (0, "a"),
        ),
        (
            {
                "axis": -1,
                "mom_dims": (
                    "a",
                    "b",
                    "mom",
                ),
            },
            ValueError,
        ),
        ({"axis": 2, "mom_dims": ("mom",)}, ValueError),
        ({"dim": "a", "mom_dims": ("a",)}, ValueError),
        ({"axis": 2, "mom_dims": ("mom",), "allow_select_mom_axes": True}, (2, "mom")),
        ({"dim": "mom", "mom_dims": ("a",)}, (2, "mom")),
        ({"dim": "hello"}, ValueError),
    ],
)
def test_select_axis_dim(data, kws, expected) -> None:
    _do_test(_wrap_select_method("select_axis_dim"), data, expected=expected, **kws)


@pytest.mark.parametrize(
    "data",
    [
        xr.Dataset(
            {
                "data0": xr.DataArray(np.zeros((1, 1, 1)), dims=("a", "b", "mom")),
                "data1": xr.DataArray(np.zeros((1, 1)), dims=("a", "mom")),
            }
        )
    ],
)
@pytest.mark.parametrize(
    ("kws", "expected"),
    [
        # errors
        ({}, ValueError),
        ({"axis": 0}, ValueError),
        ({"dim": "a"}, (0, "a")),
        ({"dim": "mom", "mom_dims": ("mom",)}, ValueError),
        (
            {"dim": "mom", "mom_dims": ("mom",), "allow_select_mom_axes": True},
            (0, "mom"),
        ),
    ],
)
def test_select_axis_dim_dataset(data, kws, expected) -> None:
    _do_test(_wrap_select_method("select_axis_dim"), data, expected=expected, **kws)


@pytest.mark.parametrize(
    "data", [xr.DataArray(np.zeros((1, 1, 1)), dims=("a", "b", "mom"))]
)
@pytest.mark.parametrize(
    ("kws", "expected"),
    [
        # errors
        ({}, ValueError),
        ({"default_axis": 0, "default_dim": "hello"}, ValueError),
        ({"axis": 0, "dim": "a"}, ValueError),
        ({"axis": 2, "mom_dims": ("mom",)}, ValueError),
        ({"dim": "mom", "mom_dims": ("mom",)}, ValueError),
        ({"axis": (0, 2), "mom_dims": ("mom",)}, ValueError),
        ({"dim": ("a", "mom"), "mom_dims": ("mom",)}, ValueError),
        ({"dim": "a", "mom_dims": ("a",)}, ValueError),
        (
            {"dim": "a", "mom_dims": ("a",), "allow_select_mom_axes": True},
            ((0,), ("a",)),
        ),
        # other
        ({"axis": 0}, ((0,), ("a",))),
        ({"axis": 1}, ((1,), ("b",))),
        ({"axis": -1}, ((2,), ("mom",))),
        ({"axis": -1j, "mom_dims": ("mom",)}, ((1,), ("b",))),
        ({"dim": "a"}, ((0,), ("a",))),
        ({"dim": "b"}, ((1,), ("b",))),
        ({"dim": "mom"}, ((2,), ("mom",))),
        ({"axis": (0, 1)}, ((0, 1), ("a", "b"))),
        ({"axis": (1, 0)}, ((1, 0), ("b", "a"))),
        ({"axis": None}, ((0, 1, 2), ("a", "b", "mom"))),
        ({"axis": None, "mom_dims": ("mom",)}, ((0, 1), ("a", "b"))),
        ({"dim": ("a", "b")}, ((0, 1), ("a", "b"))),
        ({"dim": ("b", "a")}, ((1, 0), ("b", "a"))),
        ({"dim": ("a", "mom")}, ((0, 2), ("a", "mom"))),
        ({"dim": None}, ((0, 1, 2), ("a", "b", "mom"))),
        ({"dim": None, "mom_dims": ("mom",)}, ((0, 1), ("a", "b"))),
        ({"default_axis": (0, 1)}, ((0, 1), ("a", "b"))),
        ({"default_dim": None, "mom_dims": ("mom",)}, ((0, 1), ("a", "b"))),
        # using "other"
        ({"dim": "mom", "mom_dims": ("a", "b")}, ((2,), ("mom",))),
    ],
)
def test_select_axis_dim_mult(data, kws, expected) -> None:
    _do_test(
        _wrap_select_method("select_axis_dim_mult"), data, expected=expected, **kws
    )


@pytest.mark.parametrize(
    "data",
    [
        xr.Dataset(
            {
                "data0": xr.DataArray(np.zeros((1, 1, 1)), dims=("a", "b", "mom")),
                "data1": xr.DataArray(np.zeros((1, 1)), dims=("a", "mom")),
            }
        )
    ],
)
@pytest.mark.parametrize(
    ("kws", "expected"),
    [
        # errors
        ({}, ValueError),
        ({"axis": 0}, ValueError),
        ({"dim": None}, ((), ("a", "b", "mom"))),
        ({"dim": None, "mom_dims": ("mom",)}, ((), ("a", "b"))),
        ({"dim": "mom", "mom_dims": ("mom",)}, ValueError),
        (
            {"dim": "mom", "mom_dims": ("mom",), "allow_select_mom_axes": True},
            ((), ("mom",)),
        ),
        # This is exactly what select does.  It calculates mom_dims from dimensions of "first"
        # array.
        ({"dim": None, "mom_dims": ("b", "mom")}, ((), ("a",))),
        ({"dim": "a"}, ((), ("a",))),
        ({"dim": "b"}, ((), ("b",))),
        ({"dim": ("a", "b")}, ((), ("a", "b"))),
    ],
)
def test_select_axis_dim_mult_dataset(data, kws, expected) -> None:
    _do_test(
        _wrap_select_method("select_axis_dim_mult"), data, expected=expected, **kws
    )


@pytest.mark.parametrize(
    ("args", "mom_params_kws", "kwargs", "expected"),
    [
        (
            (),
            {"ndim": 1},
            {"axis": -2},
            [(-2, -1), (-1,)],
        ),
        (
            (),
            {"ndim": 2},
            {"axis": -3},
            [(-3, -2, -1), (-2, -1)],
        ),
        (
            ((), -2),
            {"ndim": 1},
            {"axis": -3},
            [(-3, -1), (), (-2,), (-1,)],
        ),
        (
            (),
            {"ndim": 1},
            {"axis": -2, "out_has_axis": True},
            [(-2, -1), (-2, -1)],
        ),
        (
            (),
            {"ndim": 2},
            {"axis": -3, "out_has_axis": True},
            [(-3, -2, -1), (-3, -2, -1)],
        ),
        (
            ((), -2),
            {"ndim": 1},
            {"axis": -3, "out_has_axis": True},
            [(-3, -1), (), (-2,), (-3, -1)],
        ),
        (
            (),
            {"axes": (1, 2)},
            {"axis": -1, "out_has_axis": True},
            [(-1, 1, 2), (-1, 1, 2)],
        ),
    ],
)
def test_axes_data_reduction(args, mom_params_kws, kwargs, expected) -> None:
    mom_params = MomParamsArray.factory(mom_params_kws)
    _do_test(mom_params.axes_data_reduction, *args, expected=expected, **kwargs)


@pytest.mark.parametrize(
    ("data", "mom_params_kwargs"),
    [
        (
            np.zeros((1, 2, 3, 4)),
            {"ndim": 2},
        ),
        (
            np.zeros((1, 3, 4, 2)),
            {"axes": (1, 2)},
        ),
        (
            xr.DataArray(np.zeros((3, 4, 1, 2)), dims=["mom0", "mom1", "a", "b"]),
            {"dims": ("mom0", "mom1")},
        ),
    ],
)
@pytest.mark.parametrize(
    ("mom", "mom_shape", "val_shape"),
    [
        ((2, 3), (3, 4), (1, 2)),
    ],
)
def test_getters(data, mom_params_kwargs, mom, mom_shape, val_shape) -> None:
    mom_params = factory_mom_params(data, **mom_params_kwargs)

    assert mom_params.get_mom(data) == mom
    assert mom_params.get_mom_shape(data) == mom_shape
    assert mom_params.get_val_shape(data) == val_shape

    other = np.zeros(10)
    if is_xarray(data):
        other = xr.DataArray(other, dims="mom0")  # type: ignore[assignment]
        with pytest.raises(ValueError):
            mom_params.get_mom_shape(other)

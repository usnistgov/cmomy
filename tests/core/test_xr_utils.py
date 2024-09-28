# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cmomy.core import xr_utils
from cmomy.core.validate import is_dataarray


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
        (
            {"apply_ufunc_kwargs": None},
            {
                "dask": "parallel",
                "dask_gufunc_kwargs": {},
                "on_missing_core_dim": "copy",
            },
        ),
        (
            {"apply_ufunc_kwargs": {"on_missing_core_dim": "raise"}},
            {
                "on_missing_core_dim": "raise",
                "dask": "parallel",
                "dask_gufunc_kwargs": {},
            },
        ),
        (
            {
                "apply_ufunc_kwargs": {"on_missing_core_dim": "raise"},
                "on_missing_core_dim": "copy",
            },
            {
                "on_missing_core_dim": "raise",
                "dask": "parallel",
                "dask_gufunc_kwargs": {},
            },
        ),
        (
            {"output_sizes": {"rec": 2}},
            {
                "dask": "parallel",
                "dask_gufunc_kwargs": {"output_sizes": {"rec": 2}},
                "on_missing_core_dim": "copy",
            },
        ),
        (
            {"output_sizes": {"rec": 2}, "dask_gufunc_kwargs": {"hello": "there"}},
            {
                "dask": "parallel",
                "dask_gufunc_kwargs": {"hello": "there", "output_sizes": {"rec": 2}},
                "on_missing_core_dim": "copy",
            },
        ),
        (
            {"output_dtypes": float},
            {
                "dask": "parallel",
                "dask_gufunc_kwargs": {},
                "output_dtypes": float,
                "on_missing_core_dim": "copy",
            },
        ),
    ],
)
def test_factory_apply_ufunc_kwargs(kws, expected) -> None:
    _do_test(xr_utils.factory_apply_ufunc_kwargs, expected=expected, **kws)


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
        ({"axis": -1, "mom_dims": ("mom",)}, (1, "b")),
        (
            {
                "axis": -1,
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
        ({"dim": "mom", "mom_dims": ("a",)}, (2, "mom")),
        ({"dim": "hello"}, ValueError),
    ],
)
def test_select_axis_dim(data, kws, expected) -> None:
    _do_test(xr_utils.select_axis_dim, data, expected=expected, **kws)


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
    ],
)
def test_select_axis_dim_dataset(data, kws, expected) -> None:
    _do_test(xr_utils.select_axis_dim, data, expected=expected, **kws)


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
        # other
        ({"axis": 0}, ((0,), ("a",))),
        ({"axis": 1}, ((1,), ("b",))),
        ({"axis": -1}, ((2,), ("mom",))),
        ({"axis": -1, "mom_dims": ("mom",)}, ((1,), ("b",))),
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
    _do_test(xr_utils.select_axis_dim_mult, data, expected=expected, **kws)


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
        # This is exactly what select does.  It calculates mom_dims from dimensions of "first"
        # array.
        ({"dim": None, "mom_dims": ("b", "mom")}, ((), ("a",))),
        ({"dim": "a"}, ((), ("a",))),
        ({"dim": "b"}, ((), ("b",))),
        ({"dim": ("a", "b")}, ((), ("a", "b"))),
    ],
)
def test_select_axis_dim_mult_dataset(data, kws, expected) -> None:
    _do_test(xr_utils.select_axis_dim_mult, data, expected=expected, **kws)


@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize(
    "indexer",
    [
        {"a": 0},
        {"a": slice(1, None)},
        {"a": 0, "b": slice(1, None), "c": slice(2, None)},
    ],
)
def test_replace_coords_from_isel(indexer, drop):
    x_with_coords = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=["a", "b", "c"],
        coords={"a": ("a", [1, 2]), "b": ("b", list("abc")), "c": ("c", [4, 5, 6, 7])},
    )

    x_without = xr.DataArray(np.ones((2, 3, 4)), dims=list("abc"))

    t = xr_utils.replace_coords_from_isel(
        x_with_coords, x_without.isel(indexer), indexer, drop=drop
    )

    xr.testing.assert_identical(t, x_with_coords.isel(indexer, drop=drop) + 1)


@pytest.mark.parametrize("x", [xr.DataArray(np.zeros((2, 3, 4)), dims=["a", "b", "c"])])
@pytest.mark.parametrize(
    ("kws", "expected"),
    [
        ({"mom_dims": None}, ("a", "b", "c")),
        ({"mom_dims": "a"}, ("b", "c", "a")),
        ({"mom_dims": "b"}, ("a", "c", "b")),
        ({"mom_dims": ("b", "a")}, ("c", "b", "a")),
        ({"mom_dims": "a", "mom_ndim": 2}, ValueError),
    ],
)
def test_move_mom_dims_to_end(x, kws, expected) -> None:
    if isinstance(expected, type):
        with pytest.raises(expected):
            xr_utils.move_mom_dims_to_end(x, **kws)
    else:
        assert xr_utils.move_mom_dims_to_end(x, **kws).dims == expected


def test_raise_if_dataset() -> None:
    x = xr.Dataset()

    with pytest.raises(TypeError, match="Dataset not allowed."):
        xr_utils.raise_if_dataset(x)

    msg = "hello there"
    with pytest.raises(TypeError, match=msg):
        xr_utils.raise_if_dataset(x, msg=msg)


@pytest.mark.parametrize(
    ("data", "mom_dims", "expected"),
    [
        (xr.DataArray(np.zeros((1, 2, 3)), dims=["a", "b", "c"]), ("c",), (3,)),
        (xr.DataArray(np.zeros((1, 2, 3)), dims=["a", "b", "mom"]), ("b", "a"), (2, 1)),
        (
            xr.Dataset(
                {"data0": xr.DataArray(np.zeros((1, 2, 3)), dims=["a", "b", "c"])}
            ),
            ("c", "b"),
            (3, 2),
        ),
    ],
)
def test_get_mom_shape(data, mom_dims, expected) -> None:
    assert xr_utils.get_mom_shape(data, mom_dims) == expected


@pytest.mark.parametrize(
    "data",
    [
        xr.Dataset(
            {
                "a": xr.DataArray(np.zeros((1, 2, 3))),
                "b": xr.DataArray(np.zeros((1, 2, 3))),
                "c": xr.DataArray(np.zeros((1, 2, 3))),
            }
        ),
        xr.DataArray(np.zeros((1, 2, 3))),
    ],
)
@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.float32, np.float32),
        (
            {"a": np.float32},
            {"a": np.float32, "b": np.dtype("f8"), "c": np.dtype("f8")},
        ),
        (
            {"a": np.float32, "b": np.int64},
            {"a": np.float32, "b": np.int64, "c": np.dtype("f8")},
        ),
    ],
)
def test_astype_dtype_dict(data, dtype, expected) -> None:
    if is_dataarray(data) and isinstance(dtype, dict):
        with pytest.raises(ValueError):
            xr_utils.astype_dtype_dict(data, dtype)

    else:
        assert xr_utils.astype_dtype_dict(data, dtype) == expected


@pytest.mark.parametrize("data", [np.zeros((3, 3))])
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"mom_dims": None, "mom_ndim": None, "mom_ndim_default": None},
            {},
        ),
        (
            {
                "mom_dims": None,
                "mom_ndim": None,
                "mom_ndim_default": None,
                "include_mom_ndim": True,
            },
            ValueError,
        ),
        (
            {
                "mom_dims": None,
                "mom_ndim": None,
                "mom_ndim_default": 1,
                "include_mom_ndim": True,
            },
            {"mom_ndim": 1},
        ),
    ],
)
def test_get_mom_dims_kws_array(data, kwargs, expected) -> None:
    _do_test(xr_utils.get_mom_dims_kws, data, **kwargs, expected=expected)


@pytest.mark.parametrize(
    "data", [xr.DataArray(np.zeros((2, 3, 3)), dims=["a", "b", "c"])]
)
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"mom_dims": None, "mom_ndim": None, "mom_ndim_default": None},
            ValueError,
        ),
        (
            {"mom_dims": None, "mom_ndim": None, "mom_ndim_default": 1},
            {"mom_dims": ("c",)},
        ),
        (
            {
                "mom_dims": None,
                "mom_ndim": None,
                "mom_ndim_default": 1,
                "include_mom_ndim": True,
            },
            {"mom_dims": ("c",), "mom_ndim": 1},
        ),
        (
            {"mom_dims": "a", "mom_ndim": None, "mom_ndim_default": None},
            {"mom_dims": ("a",)},
        ),
        (
            {"mom_dims": "a", "mom_ndim": 2, "mom_ndim_default": None},
            ValueError,
        ),
        (
            {
                "mom_dims": "a",
                "mom_ndim": None,
                "mom_ndim_default": 1,
                "include_mom_ndim": True,
            },
            {"mom_dims": ("a",), "mom_ndim": 1},
        ),
        (
            {"mom_dims": None, "mom_ndim": 2, "mom_ndim_default": None},
            {"mom_dims": ("b", "c")},
        ),
    ],
)
def test_get_mom_dims_kws_dataarray(data, kwargs, expected) -> None:
    kwargs = kwargs.copy()
    kwargs["out"] = data
    _do_test(xr_utils.get_mom_dims_kws, data, **kwargs, expected=expected)

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

    with pytest.raises(TypeError, match=r"Dataset not allowed."):
        xr_utils.raise_if_dataset(x)

    msg = r"hello there"
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


@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize(
    "indexer",
    [
        {"a": 0},
        {"a": 0, "b": [1, 2]},
        {"a": slice(1, None)},
        {"a": 0, "b": slice(1, None), "c": slice(2, None)},
    ],
)
@pytest.mark.parametrize(
    "x_with_coords",
    [
        xr.DataArray(
            np.zeros((2, 3, 4)),
            dims=["a", "b", "c"],
            coords={
                "a": ("a", [1, 2]),
                "b": ("b", list("abc")),
                "c": ("c", [4, 5, 6, 7]),
            },
        ),
        xr.Dataset(
            {
                "x": (("a", "b", "c"), np.zeros((2, 3, 4))),
                "y": (("a", "b"), np.zeros((2, 3))),
                "z": (("b", "c"), np.zeros((3, 4))),
                "w": (("c",), np.zeros(4)),
            },
            coords={
                "a": ("a", [1, 2]),
                "b": ("b", list("abc")),
                "c": ("c", [4, 5, 6, 7]),
            },
        ),
    ],
)
def test_replace_coords_from_isel(indexer, drop, x_with_coords):
    x_without = (x_with_coords + 1).drop_vars(("a", "b", "c"))

    t = xr_utils.replace_coords_from_isel(
        x_with_coords, x_without.isel(indexer), indexer, drop=drop
    )

    xr.testing.assert_identical(t, x_with_coords.isel(indexer, drop=drop) + 1)


def test_replace_coords_form_isel_error() -> None:
    da = xr.DataArray([1, 2, 3], dims="a")
    ds = da.to_dataset(name="hello")
    with pytest.raises(TypeError, match=r"template and selected.*"):
        _ = xr_utils.replace_coords_from_isel(  # type: ignore[type-var]
            da,
            ds,
            {"a": [0]},
        )

# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cmomy.core import xr_utils


# * catch all args only test
def _do_test(func, *args, expected=None, match=None, **kwargs):
    if isinstance(expected, type):
        with pytest.raises(expected, match=match):
            func(*args, **kwargs)
    else:
        assert func(*args, **kwargs) == expected


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
        ({"axis": -1, "mom_ndim": 1}, (1, "b")),
        ({"axis": -1, "mom_ndim": 2}, (0, "a")),
        ({"axis": -1, "mom_ndim": 3}, ValueError),
        ({"axis": 2, "mom_ndim": 1}, ValueError),
        ({"dim": "hello"}, ValueError),
        ({"axis": "a"}, ValueError),
    ],
)
def test_select_axis_dim(data, kws, expected) -> None:
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
        ({"axis": 2, "mom_ndim": 1}, ValueError),
        ({"dim": "mom", "mom_ndim": 1}, ValueError),
        ({"axis": (0, 2), "mom_ndim": 1}, ValueError),
        ({"dim": ("a", "mom"), "mom_ndim": 1}, ValueError),
        # other
        ({"axis": 0}, ((0,), ("a",))),
        ({"axis": 1}, ((1,), ("b",))),
        ({"axis": -1}, ((2,), ("mom",))),
        ({"axis": -1, "mom_ndim": 1}, ((1,), ("b",))),
        ({"dim": "a"}, ((0,), ("a",))),
        ({"dim": "b"}, ((1,), ("b",))),
        ({"dim": "mom"}, ((2,), ("mom",))),
        ({"axis": (0, 1)}, ((0, 1), ("a", "b"))),
        ({"axis": (1, 0)}, ((1, 0), ("b", "a"))),
        ({"axis": None}, ((0, 1, 2), ("a", "b", "mom"))),
        ({"axis": None, "mom_ndim": 1}, ((0, 1), ("a", "b"))),
        ({"dim": ("a", "b")}, ((0, 1), ("a", "b"))),
        ({"dim": ("b", "a")}, ((1, 0), ("b", "a"))),
        ({"dim": ("a", "mom")}, ((0, 2), ("a", "mom"))),
        ({"dim": None}, ((0, 1, 2), ("a", "b", "mom"))),
        ({"dim": None, "mom_ndim": 1}, ((0, 1), ("a", "b"))),
        ({"default_axis": (0, 1)}, ((0, 1), ("a", "b"))),
        ({"default_dim": None, "mom_ndim": 1}, ((0, 1), ("a", "b"))),
    ],
)
def test_select_axis_dim_mult(data, kws, expected) -> None:
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

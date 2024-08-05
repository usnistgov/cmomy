# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cmomy import utils


def _do_test(func, *args, expected=None, match=None, **kwargs):
    if isinstance(expected, type):
        with pytest.raises(expected, match=match):
            func(*args, **kwargs)
    else:
        assert func(*args, **kwargs) == expected


@pytest.mark.parametrize(
    "x", [np.zeros((1, 2, 3, 4)), xr.DataArray(np.zeros((1, 2, 3, 4)))]
)
@pytest.mark.parametrize(
    "func", [lambda *args, **kwargs: utils.moveaxis(*args, **kwargs).shape]
)
@pytest.mark.parametrize(
    ("kws", "expected"),
    [
        ({"axis": 0, "dest": -1, "mom_ndim": None}, (2, 3, 4, 1)),
        ({"axis": 0, "dest": -1, "mom_ndim": 1}, (2, 3, 1, 4)),
        ({"axis": 0, "dest": -1, "mom_ndim": 2}, (2, 1, 3, 4)),
        ({"axis": (1, 0), "dest": (-2, -1), "mom_ndim": 1}, (3, 2, 1, 4)),
        ({"axis": (1, 0), "dest": (-2,), "mom_ndim": 1}, ValueError),
    ],
)
def test_moveaxis(x, kws, expected, func):
    _do_test(func, x, **kws, expected=expected)


@pytest.mark.parametrize("x", [xr.DataArray(np.zeros((1, 2, 3, 4)))])
@pytest.mark.parametrize(
    ("kws", "expected"),
    [
        ({"dim": "dim_0", "dest": -1, "mom_ndim": None}, (2, 3, 4, 1)),
        ({"dim": "dim_0", "dest": -1, "mom_ndim": 1}, (2, 3, 1, 4)),
        ({"dim": "dim_0", "dest": -1, "mom_ndim": 2}, (2, 1, 3, 4)),
        ({"dim": ("dim_1", "dim_0"), "dest": (-2, -1), "mom_ndim": 1}, (3, 2, 1, 4)),
        ({"dim": ("dim_1", "dim_0"), "dest": (-2,), "mom_ndim": 1}, ValueError),
        ({"dim": "dim_0", "dest_dim": "dim_3", "mom_ndim": None}, (2, 3, 4, 1)),
        ({"dim": "dim_0", "dest_dim": "dim_2", "mom_ndim": 1}, (2, 3, 1, 4)),
        ({"dim": "dim_0", "dest_dim": "dim_1", "mom_ndim": 2}, (2, 1, 3, 4)),
        (
            {"dim": ("dim_1", "dim_0"), "dest_dim": ("dim_1", "dim_2"), "mom_ndim": 1},
            (3, 2, 1, 4),
        ),
        ({"dim": ("dim_1", "dim_0"), "dest_dim": "dim_2", "mom_ndim": 1}, ValueError),
    ],
)
@pytest.mark.parametrize(
    "func", [lambda *args, **kwargs: utils.moveaxis(*args, **kwargs).shape]
)
def test_moveaxis_dataarray(x, kws, expected, func):
    _do_test(func, x, **kws, expected=expected)


@pytest.mark.parametrize(
    ("shape", "kwargs", "expected", "match"),
    [
        (
            (2,),
            {"name": "weight", "mom_ndim": 2},
            ValueError,
            ".*must be.*",
        ),
        (
            (3, 3),
            {"name": "yave", "mom_ndim": 1},
            ValueError,
            ".*requires mom_ndim.*",
        ),
        (
            (3, 3),
            {"name": "yvar", "mom_ndim": 1},
            ValueError,
            ".*requires mom_ndim.*",
        ),
        ((2,), {"name": "thing", "mom_ndim": 1}, ValueError, ".*Unknown option.*"),
    ],
)
def test_select_moment_errors(shape, kwargs, expected, match) -> None:
    data = np.empty(shape)
    _do_test(utils.select_moment, data, **kwargs, expected=expected, match=match)


def test_select_moment_errors_coords() -> None:
    data = xr.DataArray(np.empty((3, 3)))
    with pytest.raises(ValueError, match=".*must equal.*"):
        utils.select_moment(data, "ave", mom_ndim=2, coords_combined="hello")


def _do_test_select_moment_mom_ndim(
    data,
    mom_ndim,
    index,
    dim_combined,
    coords_combined,
    **kwargs,
):
    out = utils.select_moment(
        data,
        **kwargs,
        mom_ndim=mom_ndim,
        dim_combined=dim_combined,
        coords_combined=coords_combined,
    )

    np.testing.assert_allclose(out, np.asarray(data)[index])

    if (
        isinstance(out, xr.DataArray)
        and kwargs["name"] in {"ave", "var"}
        and (mom_ndim != 1 or not kwargs.get("squeeze", False))
    ):
        dim_combined = dim_combined or "variable"
        if coords_combined is None:
            coords_combined = list(data.dims[-mom_ndim:])
        elif isinstance(coords_combined, str):
            coords_combined = [coords_combined]
        else:
            coords_combined = list(coords_combined)

        assert out.dims[-1] == dim_combined
        assert out.coords[dim_combined].to_numpy().tolist() == coords_combined


@pytest.mark.parametrize(
    "shape",
    [
        (3,),
        (2, 3),
        (1, 2, 3),
    ],
)
@pytest.mark.parametrize(
    ("kwargs", "index"),
    [
        ({"name": "weight"}, (..., 0)),
        ({"name": "ave"}, (..., [1])),
        ({"name": "var"}, (..., [2])),
        ({"name": "ave", "squeeze": True}, (..., 1)),
        ({"name": "var", "squeeze": True}, (..., 2)),
        ({"name": "xave"}, (..., 1)),
        ({"name": "xvar"}, (..., 2)),
        ({"name": "cov"}, (..., 2)),
    ],
)
@pytest.mark.parametrize("dim_combined", ["variable", "hello"])
@pytest.mark.parametrize("coords_combined", [None, "thing"])
def test_select_moment_mom_ndim_1(
    rng, shape, kwargs, index, as_dataarray, dim_combined, coords_combined
) -> None:
    data = rng.random(shape)
    if as_dataarray:
        data = xr.DataArray(data)
    _do_test_select_moment_mom_ndim(
        data,
        1,
        index,
        dim_combined,
        coords_combined,
        **kwargs,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (3, 3),
        (2, 3, 3),
        (1, 2, 3, 3),
    ],
)
@pytest.mark.parametrize(
    ("kwargs", "index"),
    [
        ({"name": "weight"}, (..., 0, 0)),
        ({"name": "ave"}, (..., [1, 0], [0, 1])),
        ({"name": "var"}, (..., [2, 0], [0, 2])),
        ({"name": "xave"}, (..., 1, 0)),
        ({"name": "xvar"}, (..., 2, 0)),
        ({"name": "yave"}, (..., 0, 1)),
        ({"name": "yvar"}, (..., 0, 2)),
        ({"name": "cov"}, (..., 1, 1)),
    ],
)
@pytest.mark.parametrize("dim_combined", ["hello"])
@pytest.mark.parametrize("coords_combined", [None, ("thing", "other")])
def test_select_moment_mom_ndim_2(
    rng,
    shape,
    kwargs,
    index,
    as_dataarray,
    dim_combined,
    coords_combined,
) -> None:
    data = rng.random(shape)
    if as_dataarray:
        data = xr.DataArray(data)
    _do_test_select_moment_mom_ndim(
        data, 2, index, dim_combined, coords_combined, **kwargs
    )

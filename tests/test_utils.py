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

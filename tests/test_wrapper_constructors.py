# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
"""
Basic tests of constructors...
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import cmomy
from cmomy.core.validate import is_xarray


def _select_mod(data):
    return xr if is_xarray(data) else np


data_mark = pytest.mark.parametrize(
    "data",
    [
        [0.0, 0.0, 0.0],
        np.zeros((10, 3)),
        xr.DataArray(np.zeros((10, 3))),
        xr.DataArray(np.zeros((10, 3))).to_dataset(name="data0"),
    ],
)


@data_mark
@pytest.mark.parametrize("copy", [True, False])
def test_wrap(data, copy) -> None:
    c = cmomy.wrap(data, copy=copy)

    if is_xarray(data):
        assert isinstance(c, cmomy.CentralMomentsData)
    else:
        assert isinstance(c, cmomy.CentralMomentsArray)

    tester = _select_mod(data).testing.assert_allclose
    c2 = c.assign_moment(weight=10.0, copy=False)
    if copy:
        with pytest.raises(AssertionError):
            tester(c2.obj, data)
    elif not isinstance(data, list):
        tester(c2.obj, data)


@data_mark
def test_zeros_like(data) -> None:
    c = cmomy.wrap(data)
    z = cmomy.zeros_like(c)
    assert type(c) is type(z)

    xp = _select_mod(data)

    xp.testing.assert_allclose(
        z.obj,
        xp.zeros_like(data),
    )

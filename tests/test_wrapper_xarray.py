# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

import pytest
import xarray as xr

import cmomy

mark_simple_dataset = pytest.mark.parametrize(
    "c",
    [
        cmomy.CentralMomentsData(
            xr.Dataset(
                {
                    "a": xr.DataArray(cmomy.random.default_rng().random((2, 3, 4))),
                }
            )
        )
    ],
)


@mark_simple_dataset
@pytest.mark.parametrize("method", ["dtype", "shape", "val_shape", "ndim", "val_ndim"])
def test_dataset_not_implemented(c, method) -> None:
    with pytest.raises(NotImplementedError):
        getattr(c, method)


@mark_simple_dataset
def test_dataset_not_implemented_2(c) -> None:
    with pytest.raises(NotImplementedError):
        c.moveaxis(0, -1)

    with pytest.raises(NotImplementedError):
        c.std()

    with pytest.raises(ZeroDivisionError):
        _ = c - c


@mark_simple_dataset
def test_pipe(c) -> None:
    c2 = c.pipe(lambda x: x + 1, _reorder=False)
    xr.testing.assert_allclose(c2.obj, c.obj + 1)


@mark_simple_dataset
def test_getitem(c) -> None:
    with pytest.raises(KeyError):
        c["_hello"]

    for k, v in c.obj.items():
        d = c[k]
        xr.testing.assert_equal(d.obj, v)

        with pytest.raises(ValueError):
            _ = d[..., 0]


@mark_simple_dataset
def test__dtype(c) -> None:
    assert c["a"]._dtype.name == "float64"
    assert c._dtype is None

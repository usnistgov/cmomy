# mypy: disable-error-code="no-untyped-def, no-untyped-call"
"""Some simple tests for factory methods of xCentral"""
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import xarray as xr

import cmomy

if TYPE_CHECKING:
    from typing import Callable

    from cmomy.typing import F


def my_fixture(**kws) -> Callable[[F], F]:
    return cast("Callable[[F], F]", pytest.fixture(scope="module", **kws))  # pyright: ignore[reportReturnType]


@my_fixture(params=[3, (3, 3)])
def mom(request):
    return request.param


@my_fixture()
def mom_tuple(mom):
    if isinstance(mom, int):
        return (mom,)
    return mom


@my_fixture(params=[(10,), (10, 2), (10, 2, 2)])
def shape(request):
    return request.param


@my_fixture()
def axis(shape, rng):
    return rng.integers(0, len(shape))


@my_fixture()
def xy(shape, mom_tuple, rng):
    x = rng.random(shape)

    if len(mom_tuple) == 2:
        y = rng.random(shape)
        return (x, y)
    return x


@my_fixture()
def dc(xy, mom, axis):
    return cmomy.CentralMoments.from_vals(xy, mom=mom, w=None, axis=axis)


@my_fixture()
def dcx(dc):
    return dc.to_xcentralmoments()


def test_CS(dc, dcx) -> None:
    np.testing.assert_allclose(dc, dcx)


def test_from_data(dc, dcx) -> None:
    mom_ndim = dc.mom_ndim

    t = cmomy.CentralMoments.from_data(dc.data, mom_ndim=mom_ndim)

    dims = [f"hello_{i}" for i in range(len(dc.data.shape))]
    o1 = cmomy.xCentralMoments.from_data(dc.data, dims=dims, mom_ndim=mom_ndim)

    np.testing.assert_allclose(t, o1)

    # create from xarray?
    o2 = cmomy.xCentralMoments.from_data(
        dcx.to_dataarray().rename(dict(zip(dcx.dims, dims))), mom_ndim=mom_ndim
    )
    xr.testing.assert_allclose(o1.to_dataarray(), o2.to_dataarray())


def test_from_datas(dc, dcx) -> None:
    mom_ndim = dc.mom_ndim

    for axis in range(dc.val_ndim):
        t = cmomy.CentralMoments.from_datas(dc.data, axis=axis, mom_ndim=mom_ndim)

        dims = dcx.dims[:axis] + dcx.dims[axis + 1 :]

        o1 = cmomy.xCentralMoments.from_datas(
            dc.data, axis=axis, mom_ndim=mom_ndim, dims=dims
        )

        np.testing.assert_allclose(t, o1)

        dim = dcx.dims[axis]
        o2 = cmomy.xCentralMoments.from_datas(
            dcx.to_dataarray(), dim=dim, mom_ndim=mom_ndim
        )

        xr.testing.assert_allclose(o1.to_dataarray(), o2.to_dataarray())


def test_from_raw(dc, dcx) -> None:
    mom_ndim = dc.mom_ndim

    t = cmomy.CentralMoments.from_raw(dc.to_raw(), mom_ndim=mom_ndim)

    o1 = cmomy.xCentralMoments.from_raw(dc.to_raw(), mom_ndim=mom_ndim)

    np.testing.assert_allclose(t, o1)

    o2 = cmomy.xCentralMoments.from_raw(
        dcx.to_raw(),
        mom_ndim=mom_ndim,
        convert_kws={"axis": -1 if mom_ndim == 1 else (-2, -1)},
    )
    xr.testing.assert_allclose(o1.to_dataarray(), o2.to_dataarray())

    if mom_ndim == 2:
        with pytest.raises(ValueError):
            o2 = cmomy.xCentralMoments.from_raw(dcx.to_raw(), mom_ndim=mom_ndim + 1)  # pyright: ignore[reportArgumentType]


def test_from_raws(dc, dcx) -> None:
    mom_ndim = dc.mom_ndim

    for axis in range(dc.val_ndim):
        # first test from raws
        raws = dc.to_raw()
        t = cmomy.CentralMoments.from_raws(raws, axis=axis, mom_ndim=mom_ndim)
        r = dc.reduce(axis=axis)

        np.testing.assert_allclose(t.to_numpy(), r.to_numpy())

        # test xCentral
        o1 = cmomy.xCentralMoments.from_raws(raws, axis=axis, mom_ndim=mom_ndim)

        np.testing.assert_allclose(t, o1)

        dim = dcx.dims[axis]
        cmomy.xCentralMoments.from_raws(dcx.to_raw(), dim=dim, mom_ndim=mom_ndim)


def test_from_vals(xy, shape, mom) -> None:
    dims = tuple(f"hello_{i}" for i in range(len(shape)))
    xy_xr: xr.DataArray | tuple[xr.DataArray, xr.DataArray]
    if isinstance(xy, tuple):
        xy_xr = tuple(xr.DataArray(xx, dims=dims) for xx in xy)  # type: ignore[assignment]
    else:
        xy_xr = xr.DataArray(xy, dims=dims)

    for axis in range(len(shape)):
        t = cmomy.xCentralMoments.from_vals(xy, axis=axis, mom=mom)

        # dims of output
        o1 = cmomy.xCentralMoments.from_vals(
            xy, axis=axis, mom=mom, dims=dims[:axis] + dims[axis + 1 :]
        )
        np.testing.assert_allclose(t, o1)

        o2 = cmomy.xCentralMoments.from_vals(
            xy_xr, dim=dims[axis], mom=mom, dtype=np.float64
        )

        xr.testing.assert_allclose(o1.to_dataarray(), o2.to_dataarray())


def test_from_resample_vals(xy, shape, mom) -> None:
    dims = tuple(f"hello_{i}" for i in range(len(shape)))
    xy_xr: xr.DataArray | tuple[xr.DataArray, xr.DataArray]
    if isinstance(xy, tuple):
        xy_xr = tuple(xr.DataArray(xx, dims=dims) for xx in xy)  # type: ignore[assignment]
    else:
        xy_xr = xr.DataArray(xy, dims=dims)

    for axis in range(len(shape)):
        t, freq = cmomy.xCentralMoments.from_resample_vals(
            xy, nrep=10, full_output=True, axis=axis, mom=mom
        )  # type : ignore

        # dims of output
        o1 = cmomy.xCentralMoments.from_resample_vals(
            xy, axis=axis, mom=mom, freq=freq, dims=dims[:axis] + dims[axis + 1 :]
        )
        np.testing.assert_allclose(t, o1)

        o2 = cmomy.xCentralMoments.from_resample_vals(
            xy_xr,
            dim=dims[axis],
            mom=mom,
            freq=freq,  # w=xr.DataArray(1.0)  # NOTE: had this for coverage, but ignoring anyway...
        )

        xr.testing.assert_allclose(o1.to_dataarray(), o2.to_dataarray())


def test_from_resample_vals2() -> None:
    coords = {"a": [1], "b": [2, 3]}
    attrs = {"hello": "there"}
    x = xr.DataArray(np.zeros((10, 1, 2)), dims=("rec", "a", "b"))
    xc = x.assign_coords(coords).rename("hello").assign_attrs(attrs)

    t = cmomy.xCentralMoments.from_resample_vals(xc, dim="rec", mom=2, nrep=10)

    xr.testing.assert_equal(t.coords.to_dataset(), xc.isel(rec=0).coords.to_dataset())
    assert t.name == "hello"
    assert t.attrs == attrs

    t = cmomy.xCentralMoments.from_resample_vals(
        x, dim="rec", mom=2, nrep=10, coords=coords, name="hello", attrs=attrs
    )
    xr.testing.assert_equal(t.coords.to_dataset(), xc.isel(rec=0).coords.to_dataset())
    assert t.name == "hello"
    assert t.attrs == attrs


def test_from_vals2() -> None:
    coords = {"a": [1], "b": [2, 3]}
    x = xr.DataArray(np.zeros((10, 1, 2)), dims=("rec", "a", "b"))
    xc = x.assign_coords(coords).rename("hello")

    t = cmomy.xCentralMoments.from_vals(xc, dim="rec", mom=2)

    xr.testing.assert_equal(t.coords.to_dataset(), xc.isel(rec=0).coords.to_dataset())
    assert t.name == "hello"

    t = cmomy.xCentralMoments.from_vals(
        x, dim="rec", mom=2, coords=coords, name="hello"
    )
    xr.testing.assert_equal(t.coords.to_dataset(), xc.isel(rec=0).coords.to_dataset())
    assert t.name == "hello"


def test_resample_and_reduce(dc, dcx) -> None:
    for axis in range(dc.val_ndim):
        t, freq = dc.resample_and_reduce(nrep=10, full_output=True, axis=axis)

        o = dcx.resample_and_reduce(freq=freq, dim=dcx.dims[axis])

        np.testing.assert_allclose(t.data, o.data)

        assert o.val_dims == ("rep",) + dcx.val_dims[:axis] + dcx.val_dims[axis + 1 :]

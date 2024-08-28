# mypy: disable-error-code="no-untyped-def, no-untyped-call"
"""Some simple tests for factory methods of xCentral"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import xarray as xr

from cmomy import CentralMoments, resample, xCentralMoments

if TYPE_CHECKING:
    from typing import Callable

    from cmomy.core.typing import F


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
    return (x,)


@my_fixture()
def dc(xy, mom, axis):
    return CentralMoments.from_vals(*xy, mom=mom, weight=None, axis=axis)


@my_fixture()
def dcx(dc):
    return dc.to_x()


def test_CS(dc, dcx) -> None:
    np.testing.assert_allclose(dc, dcx)


def test_init(dc, dcx) -> None:
    mom_ndim = dc.mom_ndim

    t = CentralMoments(dc.to_numpy(), mom_ndim=mom_ndim)

    dims = [f"hello_{i}" for i in range(len(dc.obj.shape))]
    o1 = CentralMoments(dc.obj, mom_ndim=mom_ndim).to_x(dims=dims)

    np.testing.assert_allclose(t, o1)

    # create from xarray?
    o2 = xCentralMoments(dcx.obj.rename(dict(zip(dcx.dims, dims))), mom_ndim=mom_ndim)
    xr.testing.assert_allclose(o1.obj, o2.obj)


def test_init_reduce(dc, dcx) -> None:
    mom_ndim = dc.mom_ndim

    for axis in range(dc.obj.ndim - dc.mom_ndim):
        t = CentralMoments(dc.obj, mom_ndim=mom_ndim).reduce(axis=axis)

        dims = dcx.dims[:axis] + dcx.dims[axis + 1 :]

        o1 = (
            CentralMoments(
                dc.obj,
                mom_ndim=mom_ndim,
            )
            .reduce(axis=axis)
            .to_x(dims=dims)
        )

        np.testing.assert_allclose(t, o1)

        dim = dcx.dims[axis]
        o2 = xCentralMoments(dcx.obj, mom_ndim=mom_ndim).reduce(dim=dim)

        xr.testing.assert_allclose(o1.obj, o2.obj)


def test_from_raw(dc, dcx) -> None:
    mom_ndim = dc.mom_ndim

    t = CentralMoments.from_raw(dc.to_raw(), mom_ndim=mom_ndim)

    o1 = CentralMoments.from_raw(dc.to_raw(), mom_ndim=mom_ndim).to_x()

    np.testing.assert_allclose(t, o1)

    o2 = xCentralMoments.from_raw(
        dcx.to_raw(),
        mom_ndim=mom_ndim,
    )
    xr.testing.assert_allclose(o1.obj, o2.obj)

    if mom_ndim == 2:
        with pytest.raises(ValueError):
            o2 = CentralMoments.from_raw(dcx.to_raw(), mom_ndim=mom_ndim + 1)  # type: ignore[assignment] # pyright: ignore[reportArgumentType]


def test_from_raws(dc, dcx) -> None:
    mom_ndim = dc.mom_ndim

    for axis in range(dc.obj.ndim - dc.mom_ndim):
        # first test from raws
        raws = dc.to_raw()
        t = CentralMoments.from_raw(raws, mom_ndim=mom_ndim).reduce(axis=axis)
        r = dc.reduce(axis=axis)

        np.testing.assert_allclose(t.to_numpy(), r.to_numpy(), atol=1e-14)

        # test xCentral
        o1 = CentralMoments.from_raw(raws, mom_ndim=mom_ndim).reduce(axis=axis).to_x()

        np.testing.assert_allclose(t, o1)

        dim = dcx.dims[axis]
        o2 = xCentralMoments.from_raw(dcx.to_raw(), mom_ndim=mom_ndim).reduce(dim=dim)

        np.testing.assert_allclose(t, o2)


def test_from_vals(xy, shape, mom) -> None:
    dims = tuple(f"hello_{i}" for i in range(len(shape)))
    xy_xr = tuple(xr.DataArray(xx, dims=dims) for xx in xy)

    for axis in range(len(shape)):
        t = CentralMoments.from_vals(*xy, axis=axis, mom=mom).to_x()

        # dims of output
        o1 = CentralMoments.from_vals(
            *xy,
            axis=axis,
            mom=mom,
        ).to_x(dims=dims[:axis] + dims[axis + 1 :])
        np.testing.assert_allclose(t, o1)

        o2 = xCentralMoments.from_vals(
            *xy_xr,
            dim=dims[axis],
            mom=mom,
        )

        xr.testing.assert_allclose(o1.obj, o2.obj)


def test_from_resample_vals(xy, shape, mom) -> None:
    dims = tuple(f"hello_{i}" for i in range(len(shape)))
    xy_xr = tuple(xr.DataArray(xx, dims=dims) for xx in xy)

    for axis in range(len(shape)):
        freq = resample.random_freq(nrep=10, ndat=xy[0].shape[axis])

        t = CentralMoments.from_resample_vals(
            *xy, freq=freq, axis=axis, mom=mom
        ).to_x()  # type : ignore

        # dims of output
        o1 = CentralMoments.from_resample_vals(
            *xy,
            axis=axis,
            mom=mom,
            freq=freq,
        ).to_x(dims=(*dims[:axis], *dims[axis + 1 :], "rep"))

        np.testing.assert_allclose(t, o1)

        o2 = xCentralMoments.from_resample_vals(
            *xy_xr,
            dim=dims[axis],
            mom=mom,
            freq=freq,  # w=xr.DataArray(1.0)  # NOTE: had this for coverage, but ignoring anyway...
        )

        xr.testing.assert_allclose(o1.obj, o2.obj)


keep_attrs_mark = pytest.mark.parametrize("keep_attrs", [False, True])


@keep_attrs_mark
def test_from_resample_vals2(keep_attrs) -> None:
    coords = {"a": [1], "b": [2, 3]}
    attrs = {"hello": "there"}
    x = xr.DataArray(np.zeros((10, 1, 2)), dims=("rec", "a", "b"))
    xc = x.assign_coords(coords).rename("hello").assign_attrs(attrs)

    freq = resample.random_freq(nrep=10, ndat=xc.sizes["rec"])

    t = xCentralMoments.from_resample_vals(
        xc, dim="rec", mom=2, freq=freq, keep_attrs=keep_attrs
    )
    xr.testing.assert_equal(t.coords.to_dataset(), xc.isel(rec=0).coords.to_dataset())
    assert t.name == "hello"
    assert t.attrs == (attrs if keep_attrs else {})

    t = (
        xCentralMoments.from_resample_vals(
            x,
            dim="rec",
            mom=2,
            freq=freq,
        )
        .assign_coords(coords)
        .assign_attrs(attrs)
        .rename("hello")
    )
    xr.testing.assert_equal(t.coords.to_dataset(), xc.isel(rec=0).coords.to_dataset())
    assert t.name == "hello"
    assert t.attrs == attrs


@keep_attrs_mark
def test_from_vals2(keep_attrs) -> None:
    coords = {"a": [1], "b": [2, 3]}
    attrs = {"hello": "there"}
    x = xr.DataArray(np.zeros((10, 1, 2)), dims=("rec", "a", "b"))
    xc = x.assign_coords(coords).rename("hello").assign_attrs(attrs)

    t = xCentralMoments.from_vals(xc, dim="rec", mom=2, keep_attrs=keep_attrs)

    xr.testing.assert_equal(t.coords.to_dataset(), xc.isel(rec=0).coords.to_dataset())
    assert t.name == "hello"
    assert t.attrs == (attrs if keep_attrs else {})

    t = (
        xCentralMoments.from_vals(
            x,
            dim="rec",
            mom=2,
        )
        .assign_coords(coords)
        .assign_attrs(attrs)
        .rename("hello")
    )
    xr.testing.assert_equal(t.coords.to_dataset(), xc.isel(rec=0).coords.to_dataset())
    assert t.name == "hello"
    assert t.attrs == attrs


@pytest.mark.parametrize("parallel", [None])
def test_resample_and_reduce(dc, dcx, parallel) -> None:
    val_ndim = dc.obj.ndim - dc.mom_ndim
    for axis in range(val_ndim):
        freq = resample.random_freq(nrep=10, ndat=dc.val_shape[axis])
        t = dc.resample_and_reduce(freq=freq, axis=axis, parallel=parallel)
        o = dcx.resample_and_reduce(freq=freq, dim=dcx.dims[axis], parallel=parallel)

        np.testing.assert_allclose(t, o)

        dims = dcx.dims[:val_ndim]
        assert o.dims[:val_ndim] == (*dims[:axis], "rep", *dims[axis + 1 :])

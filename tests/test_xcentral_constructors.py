# mypy: disable-error-code="no-untyped-def, no-untyped-call"
"""
Some simple tests for factory methods of xCentral

Think most of this is covered by tests_xarray_support.....
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import xarray as xr

from cmomy import CentralMomentsArray, CentralMomentsData, resample

if TYPE_CHECKING:
    from typing import Callable

    from cmomy.core.typing import FuncT


def my_fixture(**kws) -> Callable[[FuncT], FuncT]:
    return cast("Callable[[FuncT], FuncT]", pytest.fixture(scope="module", **kws))


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
    return CentralMomentsArray.from_vals(*xy, mom=mom, weight=None, axis=axis)


@my_fixture()
def dcx(dc):
    return dc.to_x()


def test_CS(dc, dcx) -> None:
    np.testing.assert_allclose(dc, dcx)


def test_init(dc, dcx) -> None:
    mom_ndim = dc.mom_ndim

    t = CentralMomentsArray(dc.to_numpy(), mom_ndim=mom_ndim)

    dims = [f"hello_{i}" for i in range(len(dc.obj.shape))]
    o1 = CentralMomentsArray(dc.obj, mom_ndim=mom_ndim).to_x(dims=dims)

    np.testing.assert_allclose(t, o1)

    # create from xarray?
    o2 = CentralMomentsData(
        dcx.obj.rename(dict(zip(dcx.dims, dims))), mom_ndim=mom_ndim
    )
    xr.testing.assert_allclose(o1.obj, o2.obj)


def test_init_reduce(dc, dcx) -> None:
    for axis in range(dc.val_ndim):
        t = dc.reduce(axis=axis)
        dims = dcx.dims[:axis] + dcx.dims[axis + 1 :]

        o1 = t.to_x(dims=dims)
        np.testing.assert_allclose(t, o1)

        dim = dcx.dims[axis]
        o2 = dcx.reduce(dim=dim)
        xr.testing.assert_allclose(o1.obj, o2.obj)


def test_from_raw(dc, dcx) -> None:
    mom_ndim = dc.mom_ndim

    o1 = CentralMomentsArray.from_raw(dc.to_raw(), mom_ndim=mom_ndim).to_x()
    o2 = CentralMomentsData.from_raw(
        dcx.to_raw(),
        mom_ndim=mom_ndim,
    )
    xr.testing.assert_allclose(o1.obj, o2.obj)


def test_from_vals(xy, shape, mom) -> None:
    dims = tuple(f"hello_{i}" for i in range(len(shape)))
    xy_xr = tuple(xr.DataArray(xx, dims=dims) for xx in xy)

    for axis in range(len(shape)):
        t = CentralMomentsArray.from_vals(*xy, axis=axis, mom=mom).to_x()

        # dims of output
        o1 = CentralMomentsArray.from_vals(
            *xy,
            axis=axis,
            mom=mom,
        ).to_x(dims=dims[:axis] + dims[axis + 1 :])
        np.testing.assert_allclose(t, o1)

        o2 = CentralMomentsData.from_vals(
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

        t = CentralMomentsArray.from_resample_vals(
            *xy, freq=freq, axis=axis, mom=mom
        ).to_x()  # type : ignore

        # dims of output
        o1 = CentralMomentsArray.from_resample_vals(
            *xy,
            axis=axis,
            mom=mom,
            freq=freq,
        ).to_x(dims=(*dims[:axis], *dims[axis + 1 :], "rep"))

        np.testing.assert_allclose(t, o1)

        o2 = CentralMomentsData.from_resample_vals(
            *xy_xr,
            dim=dims[axis],
            mom=mom,
            freq=freq,  # w=xr.DataArray(1.0)  # NOTE: had this for coverage, but ignoring anyway...
        )

        xr.testing.assert_allclose(o1.obj, o2.obj)


keep_attrs_mark = pytest.mark.parametrize("keep_attrs", [False, True])


@pytest.mark.parametrize(
    "func",
    [
        partial(
            CentralMomentsData.from_resample_vals, dim="rec", mom=2, nrep=10, rng=12
        ),
        partial(CentralMomentsData.from_vals, dim="rec", mom=2),
    ],
)
@keep_attrs_mark
def test_from_vals_resample_vals2(keep_attrs, func) -> None:
    coords = {"a": [1], "b": [2, 3]}
    attrs = {"hello": "there"}
    x = xr.DataArray(np.zeros((10, 1, 2)), dims=("rec", "a", "b"))
    xc = x.assign_coords(coords).rename("hello").assign_attrs(attrs)

    t = func(xc, keep_attrs=keep_attrs)
    xr.testing.assert_equal(t.coords.to_dataset(), xc.isel(rec=0).coords.to_dataset())
    if keep_attrs:
        # edge case with from_resample_vals where `freq` is a datarray.  Name will be dropped if not keep_attrs.
        assert t.name == "hello"
    assert t.attrs == (attrs if keep_attrs else {})

    t = func(x).assign_coords(coords).assign_attrs(attrs).rename("hello")
    xr.testing.assert_equal(t.coords.to_dataset(), xc.isel(rec=0).coords.to_dataset())
    if keep_attrs:
        # see above
        assert t.name == "hello"
    assert t.attrs == attrs

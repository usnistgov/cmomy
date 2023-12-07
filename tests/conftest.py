# mypy: disable-error-code="no-untyped-def, no-untyped-call"
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import xarray as xr
from module_utilities import cached

import cmomy.central as central
import cmomy.resample as resample
import cmomy.xcentral as xcentral
from cmomy._testing import get_cmom, get_comom


class Data:
    """wrapper around stuff for generic testing."""

    # _count = 0

    def __init__(self, shape, axis, style, mom, nsplit=3):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.axis = axis
        self.style = style
        self.mom = mom
        self.nsplit = nsplit
        self._cache: dict[str, Any] = {}

    @cached.prop
    def cov(self):
        if isinstance(self.mom, int):
            return False
        if isinstance(self.mom, tuple) and len(self.mom) == 2:
            return True

    @property
    def mom_ndim(self):
        if self.cov:
            return 2
        else:
            return 1

    @property
    def broadcast(self):
        return self.style == "broadcast"

    @property
    def cls(self):
        return central.CentralMoments

    @cached.prop
    def val_shape(self):
        val_shape = list(self.shape)
        val_shape.pop(self.axis)
        return tuple(val_shape)

    def _get_data(self, style=None):
        if style is None or style == "total":
            return np.random.rand(*self.shape)
        elif style == "broadcast":
            return np.random.rand(self.shape[self.axis])
        else:
            raise ValueError("bad style")

    @cached.prop
    def xdata(self):
        return self._get_data()

    @cached.prop
    def ydata(self):
        return self._get_data(style=self.style)

    @cached.prop
    def w(self):
        if self.style is None:
            return None
        #            return np.array(1.0)
        else:
            return self._get_data(style=self.style)
        return self._get_weight()

    @cached.prop
    def x(self):
        if self.cov:
            return (self.xdata, self.ydata)
        else:
            return self.xdata

    @cached.prop
    def split_data(self):
        v = self.xdata.shape[self.axis] // self.nsplit
        splits = [v * i for i in range(1, self.nsplit)]
        X = np.split(self.xdata, splits, axis=self.axis)

        if self.style == "total":
            W = np.split(self.w, splits, axis=self.axis)
        elif self.style == "broadcast":
            W = np.split(self.w, splits)
        else:
            W = [self.w for xx in X]

        if self.cov:
            if self.style == "broadcast":
                Y = np.split(self.ydata, splits)
            else:
                Y = np.split(self.ydata, splits, axis=self.axis)

            # pack X, Y
            X = list(zip(X, Y))  # type: ignore

        return W, X

    @property
    def W(self):
        return self.split_data[0]

    @property
    def X(self):
        return self.split_data[1]

    @cached.prop
    def data_fix(self) -> Any:
        if self.cov:
            return get_comom(
                w=self.w,
                x=self.x[0],
                y=self.x[1],
                moments=self.mom,
                axis=self.axis,
                broadcast=self.broadcast,
            )
        else:
            return get_cmom(
                w=self.w, x=self.x, moments=self.mom, axis=self.axis, last=True
            )

    @cached.prop
    def data_test(self):
        return central.central_moments(
            x=self.x,
            mom=self.mom,
            w=self.w,
            axis=self.axis,
            last=True,
            broadcast=self.broadcast,
        )

    @cached.prop
    def s(self):
        s = self.cls.zeros(val_shape=self.val_shape, mom=self.mom)
        s.push_vals(x=self.x, w=self.w, axis=self.axis, broadcast=self.broadcast)
        return s

    @cached.prop
    def S(self):
        return [
            self.cls.from_vals(
                x=xx, w=ww, axis=self.axis, mom=self.mom, broadcast=self.broadcast
            )
            for ww, xx in zip(self.W, self.X)
        ]

    @property
    def values(self):
        return self.data_test

    def unpack(self, *args):
        out = tuple(getattr(self, x) for x in args)
        if len(out) == 1:
            out = out[0]
        return out

    def test_values(self, x):
        np.testing.assert_allclose(self.values, x)

    @property
    def raw(self):
        if self.style == "total":
            if not self.cov:
                raw = np.array(
                    [
                        np.average(self.x**i, weights=self.w, axis=self.axis)
                        for i in range(self.mom + 1)
                    ]
                )
                raw[0, ...] = self.w.sum(self.axis)

                raw = np.moveaxis(raw, 0, -1)

            else:
                raw = np.zeros_like(self.data_test)
                for i in range(self.mom[0] + 1):
                    for j in range(self.mom[1] + 1):
                        raw[..., i, j] = np.average(
                            self.x[0] ** i * self.x[1] ** j,
                            weights=self.w,
                            axis=self.axis,
                        )

                raw[..., 0, 0] = self.w.sum(self.axis)

        else:
            raw = None
        return raw

    @cached.prop
    def indices(self):
        ndat = self.xdata.shape[self.axis]
        nrep = 10
        return np.random.choice(ndat, (nrep, ndat), replace=True)

    @cached.prop
    def freq(self):
        return resample.randsamp_freq(indices=self.indices)

    @cached.prop
    def xdata_resamp(self):
        xdata = self.xdata

        if self.axis != 0:
            xdata = np.moveaxis(xdata, self.axis, 0)

        return np.take(xdata, self.indices, axis=0)

    @cached.prop
    def ydata_resamp(self):
        ydata = self.ydata

        if self.style == "broadcast":
            return np.take(ydata, self.indices, axis=0)
        else:
            if self.axis != 0:
                ydata = np.moveaxis(ydata, self.axis, 0)
            return np.take(ydata, self.indices, axis=0)

    @property
    def x_resamp(self):
        if self.cov:
            return (self.xdata_resamp, self.ydata_resamp)
        else:
            return self.xdata_resamp

    @cached.prop
    def w_resamp(self) -> Any:
        w = self.w

        if self.style is None:
            return w
        elif self.style == "broadcast":
            return np.take(w, self.indices, axis=0)
        else:
            if self.axis != 0:
                w = np.moveaxis(w, self.axis, 0)
            return np.take(w, self.indices, axis=0)

    @cached.prop
    def data_test_resamp(self) -> Any:
        return central.central_moments(
            x=self.x_resamp,
            mom=self.mom,
            w=self.w_resamp,
            axis=1,
            broadcast=self.broadcast,
        )

    # xcentral specific stuff
    @property
    def cls_xr(self):
        return xcentral.xCentralMoments

    @cached.prop
    def s_xr(self):
        return self.cls_xr.from_vals(
            x=self.x, w=self.w, axis=self.axis, mom=self.mom, broadcast=self.broadcast
        )

    @cached.prop
    def xdata_xr(self):
        dims = [f"dim_{i}" for i in range(len(self.shape) - 1)]
        dims.insert(self.axis, "rec")
        return xr.DataArray(self.xdata, dims=dims)

    @cached.prop
    def ydata_xr(self):
        if self.style is None or self.style == "total":
            dims = self.xdata_xr.dims
        else:
            dims = "rec"

        return xr.DataArray(self.ydata, dims=dims)

    @cached.prop
    def w_xr(self):
        if self.style is None:
            return None
        elif self.style == "broadcast":
            dims = "rec"
        else:
            dims = self.xdata_xr.dims

        return xr.DataArray(self.w, dims=dims)

    @property
    def x_xr(self):
        if self.cov:
            return (self.xdata_xr, self.ydata_xr)
        else:
            return self.xdata_xr

    @cached.prop
    def data_test_xr(self):
        return xcentral.xcentral_moments(
            x=self.x_xr, mom=self.mom, dim="rec", w=self.w_xr, broadcast=self.broadcast
        )

    @cached.prop
    def W_xr(self) -> Any:
        if isinstance(self.w_xr, xr.DataArray):
            dims = self.w_xr.dims
            return [xr.DataArray(_, dims=dims) for _ in self.W]
        else:
            return self.W

    @cached.prop
    def X_xr(self):
        xdims = self.xdata_xr.dims

        if self.cov:
            ydims = self.ydata_xr.dims

            return [
                (xr.DataArray(x, dims=xdims), xr.DataArray(y, dims=ydims))
                for x, y in self.X
            ]
        else:
            return [xr.DataArray(x, dims=xdims) for x in self.X]

    @cached.prop
    def S_xr(self):
        return [
            self.cls_xr.from_vals(
                x=x, w=w, axis=self.axis, mom=self.mom, broadcast=self.broadcast
            )
            for w, x, in zip(self.W, self.X)
        ]


# Fixutre
# def get_params():
#     for shape, axis in [(20, 0), ((20, 2, 3), 0), ((2, 20, 3), 1), ((2, 3, 20), 2)]:
#         for style in [None, "total", "broadcast"]:
#             for mom in [4, (3, 3)]:
#                 yield Data(shape, axis, style, mom)


# @pytest.fixture(params=get_params(), scope="module")
# def other(request):
#     return request.param
def get_params():
    for shape, axis in [(20, 0), ((20, 2, 3), 0), ((2, 20, 3), 1), ((2, 3, 20), 2)]:
        for style in [None, "total", "broadcast"]:
            for mom in [4, (3, 3)]:
                yield shape, axis, style, mom


@pytest.fixture(params=get_params(), scope="module")  # type: ignore
def other(request):
    return Data(*request.param)


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

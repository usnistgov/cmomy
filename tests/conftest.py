# mypy: disable-error-code="no-untyped-def, no-untyped-call"
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import xarray as xr
from module_utilities import cached

import cmomy
from cmomy import central, resample, xcentral

from ._simple_cmom import get_cmom, get_comom

if TYPE_CHECKING:
    from cmomy.typing import Moments, MyNDArray


default_rng = cmomy.random.default_rng(0)


@pytest.fixture(scope="session")
def rng():
    return default_rng


class Data:  # noqa: PLR0904
    """wrapper around stuff for generic testing."""

    # _count = 0

    def __init__(
        self,
        shape: int | tuple[int, ...],
        axis: int,
        style: str | None,
        mom: Moments,
        nsplit: int = 3,
    ) -> None:
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.axis = axis
        self.style = style
        self.mom = mom
        self.nsplit = nsplit
        self._cache: dict[str, Any] = {}

    @cached.prop
    def cov(self) -> bool | None:
        if isinstance(self.mom, int):
            return False
        if isinstance(self.mom, tuple) and len(self.mom) == 2:
            return True
        return None

    @property
    def mom_ndim(self) -> int:
        if self.cov:
            return 2
        return 1

    @property
    def broadcast(self) -> bool:
        return self.style == "broadcast"

    @property
    def cls(self) -> type[central.CentralMoments]:
        return central.CentralMoments

    @cached.prop
    def val_shape(self) -> tuple[int, ...]:
        val_shape = list(self.shape)
        val_shape.pop(self.axis)
        return tuple(val_shape)

    def _get_data(self, style: str | None = None) -> MyNDArray:
        if style is None or style == "total":
            return default_rng.random(self.shape)  # pyright: ignore[reportReturnType]
        if style == "broadcast":
            return default_rng.random(self.shape[self.axis])
        msg = "bad style"
        raise ValueError(msg)

    @cached.prop
    def xdata(self) -> MyNDArray:
        return self._get_data()

    @cached.prop
    def ydata(self) -> MyNDArray:
        return self._get_data(style=self.style)

    @cached.prop
    def w(self) -> MyNDArray | None:
        if self.style is None:
            return None
        return self._get_data(style=self.style)

    @cached.prop
    def x(self) -> MyNDArray | tuple[MyNDArray, MyNDArray]:
        if self.cov:
            return (self.xdata, self.ydata)
        return self.xdata

    @cached.prop
    def split_data(self) -> tuple[list[MyNDArray] | list[None], list[MyNDArray]]:
        v = self.xdata.shape[self.axis] // self.nsplit
        splits = [v * i for i in range(1, self.nsplit)]
        X = np.split(self.xdata, splits, axis=self.axis)

        W: list[MyNDArray] | list[None]
        if self.style == "total":
            W = np.split(self.w, splits, axis=self.axis)  # type: ignore[arg-type]
        elif self.style == "broadcast":
            W = np.split(self.w, splits)  # type: ignore[arg-type]
        else:
            W = cast("list[None]", [self.w for _ in X])

        if self.cov:
            if self.style == "broadcast":
                Y = np.split(self.ydata, splits)
            else:
                Y = np.split(self.ydata, splits, axis=self.axis)

            # pack X, Y
            X = list(zip(X, Y))  # type: ignore[arg-type]

        return W, X  # pyright: ignore[reportReturnType]

    @property
    def W(self) -> list[MyNDArray] | list[None]:
        return self.split_data[0]

    @property
    def X(self) -> list[MyNDArray]:
        return self.split_data[1]

    @cached.prop
    def data_fix(self) -> MyNDArray:
        if self.cov:
            return get_comom(  # type: ignore[no-any-return]
                w=self.w,
                x=self.x[0],
                y=self.x[1],
                moments=self.mom,
                axis=self.axis,
                broadcast=self.broadcast,
            )
        return get_cmom(w=self.w, x=self.x, moments=self.mom, axis=self.axis, last=True)  # type: ignore[no-any-return]

    @cached.prop
    def data_test(self) -> MyNDArray:
        return central.central_moments(
            x=self.x,
            mom=self.mom,
            w=self.w,
            axis=self.axis,
            last=True,
            broadcast=self.broadcast,
        )

    @cached.prop
    def s(self) -> central.CentralMoments:
        s = self.cls.zeros(val_shape=self.val_shape, mom=self.mom)
        s.push_vals(x=self.x, w=self.w, axis=self.axis, broadcast=self.broadcast)
        return s

    @cached.prop
    def S(self) -> list[central.CentralMoments]:
        return [
            self.cls.from_vals(
                x=xx, w=ww, axis=self.axis, mom=self.mom, broadcast=self.broadcast
            )
            for ww, xx in zip(self.W, self.X)
        ]

    # @property
    # def values(self) -> MyNDArray:
    #     return self.data_test
    def to_values(self) -> MyNDArray:
        return self.data_test

    def unpack(self, *args) -> Any:
        out = tuple(getattr(self, x) for x in args)
        if len(out) == 1:
            out = out[0]
        return out

    def test_values(self, x, **kws) -> None:
        np.testing.assert_allclose(self.to_values(), x, **kws)

    @property
    def raw(self) -> MyNDArray | None:
        if self.style == "total":
            if not self.cov:
                raw = np.array(
                    [
                        np.average(self.x**i, weights=self.w, axis=self.axis)  # type: ignore[operator]
                        for i in range(self.mom + 1)  # type: ignore[operator]
                    ]
                )
                raw[0, ...] = self.w.sum(self.axis)  # type: ignore[union-attr]

                raw = np.moveaxis(raw, 0, -1)

            else:
                raw = np.zeros_like(self.data_test)
                for i in range(self.mom[0] + 1):  # type: ignore[index]
                    for j in range(self.mom[1] + 1):  # type: ignore[index, misc]
                        raw[..., i, j] = np.average(
                            self.x[0] ** i * self.x[1] ** j,
                            weights=self.w,
                            axis=self.axis,
                        )

                raw[..., 0, 0] = self.w.sum(self.axis)  # type: ignore[union-attr]

        else:
            raw = None
        return raw

    @cached.prop
    def indices(self) -> MyNDArray:
        ndat = self.xdata.shape[self.axis]
        nrep = 10
        return default_rng.choice(ndat, (nrep, ndat), replace=True)

    @cached.prop
    def freq(self) -> MyNDArray:
        return resample.randsamp_freq(indices=self.indices)

    @cached.prop
    def xdata_resamp(self) -> MyNDArray:
        xdata = self.xdata

        if self.axis != 0:
            xdata = np.moveaxis(xdata, self.axis, 0)

        return np.take(xdata, self.indices, axis=0)

    @cached.prop
    def ydata_resamp(self) -> MyNDArray:
        ydata = self.ydata

        if self.style == "broadcast":
            return np.take(ydata, self.indices, axis=0)

        if self.axis != 0:
            ydata = np.moveaxis(ydata, self.axis, 0)
        return np.take(ydata, self.indices, axis=0)

    @property
    def x_resamp(self) -> MyNDArray | tuple[MyNDArray, MyNDArray]:
        if self.cov:
            return (self.xdata_resamp, self.ydata_resamp)
        return self.xdata_resamp

    @cached.prop
    def w_resamp(self) -> MyNDArray | None:
        w = self.w

        if self.style is None:
            return w
        if self.style == "broadcast":
            return np.take(w, self.indices, axis=0)  # type: ignore[arg-type]
        if self.axis != 0:
            w = np.moveaxis(w, self.axis, 0)  # type: ignore[arg-type]
        return np.take(w, self.indices, axis=0)  # type: ignore[arg-type]

    @cached.prop
    def data_test_resamp(self) -> MyNDArray:
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

        dims = "rec" if self.style == "broadcast" else self.xdata_xr.dims

        return xr.DataArray(self.w, dims=dims)

    @property
    def x_xr(self):
        if self.cov:
            return (self.xdata_xr, self.ydata_xr)
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
        return [xr.DataArray(x, dims=xdims) for x in self.X]

    @cached.prop
    def S_xr(self):
        return [
            self.cls_xr.from_vals(
                x=x, w=w, axis=self.axis, mom=self.mom, broadcast=self.broadcast
            )
            for w, x in zip(self.W, self.X)
        ]


# Fixture
# def get_params():
#     for shape, axis in [(20, 0), ((20, 2, 3), 0), ((2, 20, 3), 1), ((2, 3, 20), 2)]:
#         for style in [None, "total", "broadcast"]:
#             for mom in [4, (3, 3)]:
#                 yield Data(shape, axis, style, mom)


# @pytest.fixture(params=get_params(), scope="module")
# def other(request):
#     return request.param
def get_params():
    for shape, axis in [(10, 0), ((10, 2, 3), 0), ((2, 10, 3), 1), ((2, 3, 10), 2)]:
        for style in [None, "total", "broadcast"]:
            for mom in [3, (3, 3)]:
                yield shape, axis, style, mom


@pytest.fixture(params=get_params(), scope="module")  # type: ignore[call-overload]
def other(request):
    return Data(*request.param)

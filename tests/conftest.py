# mypy: disable-error-code="no-untyped-def, no-untyped-call"
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import xarray as xr
from module_utilities import cached

import cmomy.random
import cmomy.reduction
import cmomy.resample
from cmomy import CentralMoments, xCentralMoments

from ._simple_cmom import get_cmom, get_comom

if TYPE_CHECKING:
    from cmomy.typing import Moments, NDArrayAny


default_rng = cmomy.random.default_rng(0)


@pytest.fixture(scope="session")
def rng():
    return default_rng


class Data:
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
    def cls(self) -> type[CentralMoments]:
        return CentralMoments

    @cached.prop
    def val_shape(self) -> tuple[int, ...]:
        val_shape = list(self.shape)
        val_shape.pop(self.axis)
        return tuple(val_shape)

    def _get_data(self, style: str | None = None) -> NDArrayAny:
        if style is None or style == "total":
            return default_rng.random(self.shape)  # pyright: ignore[reportReturnType]
        if style == "broadcast":
            return default_rng.random(self.shape[self.axis])
        msg = "bad style"
        raise ValueError(msg)

    @cached.prop
    def xdata(self) -> NDArrayAny:
        return self._get_data()

    @cached.prop
    def ydata(self) -> NDArrayAny:
        return self._get_data(style=self.style)

    @cached.prop
    def w(self) -> NDArrayAny | None:
        if self.style is None:
            return None
        return self._get_data(style=self.style)

    @cached.prop
    def xy_tuple(self) -> tuple[NDArrayAny] | tuple[NDArrayAny, NDArrayAny]:
        if self.cov:
            return (self.xdata, self.ydata)
        return (self.xdata,)

    @cached.prop
    def split_data(
        self,
    ) -> tuple[list[NDArrayAny] | list[None], list[tuple[NDArrayAny, ...]]]:
        v = self.xdata.shape[self.axis] // self.nsplit
        splits = [v * i for i in range(1, self.nsplit)]
        X = np.split(self.xdata, splits, axis=self.axis)

        W: list[NDArrayAny] | list[None]
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

        else:
            X = [(x,) for x in X]  # type: ignore[misc]

        return W, X  # type: ignore[return-value] # pyright: ignore[reportReturnType]

    @property
    def W(self) -> list[NDArrayAny] | list[None]:
        return self.split_data[0]

    @property
    def X(self) -> list[tuple[NDArrayAny, ...]]:
        return self.split_data[1]

    @cached.prop
    def data_fix(self) -> NDArrayAny:
        if self.cov:
            return get_comom(  # type: ignore[no-any-return]
                w=self.w,
                x=self.xy_tuple[0],
                y=self.xy_tuple[1],  # type: ignore[misc]
                moments=self.mom,
                axis=self.axis,
                broadcast=self.broadcast,
            )
        return get_cmom(  # type: ignore[no-any-return]
            w=self.w, x=self.xy_tuple[0], moments=self.mom, axis=self.axis, last=True
        )

    @cached.prop
    def data_test(self) -> NDArrayAny:
        return cmomy.reduction.reduce_vals(  # type: ignore[return-value]
            *self.xy_tuple,
            mom=self.mom,
            weight=self.w,
            axis=self.axis,
        )

    @cached.prop
    def s(self) -> CentralMoments:
        s = self.cls.zeros(val_shape=self.val_shape, mom=self.mom)
        s.push_vals(*self.xy_tuple, weight=self.w, axis=self.axis)
        return s

    @cached.prop
    def S(self) -> CentralMoments:
        return [  # type: ignore[return-value]
            self.cls.from_vals(
                *xx,
                weight=ww,
                axis=self.axis,
                mom=self.mom,
            )
            for ww, xx in zip(self.W, self.X)
        ]

    # @property
    # def values(self) -> NDArrayAny:
    #     return self.data_test
    def to_values(self) -> NDArrayAny:
        return self.data_test

    def unpack(self, *args) -> Any:
        out = tuple(getattr(self, x) for x in args)
        if len(out) == 1:
            out = out[0]
        return out

    def test_values(self, x, **kws) -> None:
        np.testing.assert_allclose(self.to_values(), x, **kws)

    @property
    def raw(self) -> NDArrayAny | None:
        if self.style == "total":
            if not self.cov:
                raw = np.array(
                    [
                        np.average(self.xdata**i, weights=self.w, axis=self.axis)
                        for i in range(self.mom + 1)  # type: ignore[operator]
                    ]
                )
                raw[0, ...] = self.w.sum(self.axis)  # type: ignore[union-attr]

                raw = np.moveaxis(raw, 0, -1)

            else:
                raw = np.zeros_like(self.data_test)
                for i in range(self.mom[0] + 1):  # type: ignore[index]
                    for j in range(self.mom[1] + 1):  # type: ignore[index,misc]
                        raw[..., i, j] = np.average(
                            self.xdata**i * self.ydata**j,
                            weights=self.w,
                            axis=self.axis,
                        )

                raw[..., 0, 0] = self.w.sum(self.axis)  # type: ignore[union-attr]

        else:
            raw = None
        return raw

    @property
    def nrep(self) -> int:
        return 10

    @property
    def ndat(self) -> int:
        return self.xdata.shape[self.axis]

    @cached.prop
    def indices(self) -> NDArrayAny:
        return default_rng.choice(self.ndat, (self.nrep, self.ndat), replace=True)

    @cached.prop
    def freq(self) -> NDArrayAny:
        return cmomy.resample.randsamp_freq(indices=self.indices, ndat=self.ndat)

    @cached.prop
    def xdata_resamp(self) -> NDArrayAny:
        xdata = self.xdata

        if self.axis != 0:
            xdata = np.moveaxis(xdata, self.axis, 0)

        return np.take(xdata, self.indices, axis=0)

    @cached.prop
    def ydata_resamp(self) -> NDArrayAny:
        ydata = self.ydata

        if self.style == "broadcast":
            return np.take(ydata, self.indices, axis=0)

        if self.axis != 0:
            ydata = np.moveaxis(ydata, self.axis, 0)
        return np.take(ydata, self.indices, axis=0)

    @property
    def xy_tuple_resamp(self) -> tuple[NDArrayAny] | tuple[NDArrayAny, NDArrayAny]:
        if self.cov:
            return (self.xdata_resamp, self.ydata_resamp)
        return (self.xdata_resamp,)

    @cached.prop
    def w_resamp(self) -> NDArrayAny | None:
        w = self.w

        if self.style is None:
            return w
        if self.style == "broadcast":
            return np.take(w, self.indices, axis=0)  # type: ignore[arg-type]
        if self.axis != 0:
            w = np.moveaxis(w, self.axis, 0)  # type: ignore[arg-type]
        return np.take(w, self.indices, axis=0)  # type: ignore[arg-type]

    @cached.prop
    def data_test_resamp(self) -> NDArrayAny:
        return np.moveaxis(
            cmomy.reduction.reduce_vals(  # type: ignore[arg-type]
                *self.xy_tuple_resamp,
                mom=self.mom,
                weight=self.w_resamp,
                axis=1,
            ),
            0,
            -(self.mom_ndim + 1),
        )

    # xcentral specific stuff
    @property
    def cls_xr(self):
        return xCentralMoments

    @cached.prop
    def s_xr(self) -> xCentralMoments:
        return self.s.to_xcentralmoments()

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
    def xy_tuple_xr(self) -> tuple[xr.DataArray, ...]:
        if self.cov:
            return (self.xdata_xr, self.ydata_xr)
        return (self.xdata_xr,)

    @cached.prop
    def data_test_xr(self):
        assert isinstance(self.xy_tuple_xr[0], xr.DataArray)
        return cmomy.reduction.reduce_vals(
            *self.xy_tuple_xr,
            mom=self.mom,
            dim="rec",
            weight=self.w_xr,
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
        return [(xr.DataArray(x[0], dims=xdims),) for x in self.X]

    @cached.prop
    def S_xr(self):
        return [
            self.cls_xr.from_vals(
                *xy,
                weight=w,
                axis=self.axis,
                mom=self.mom,
            )
            for w, xy in zip(self.W_xr, self.X_xr)
        ]


def get_params():
    for shape, axis in [(10, 0), ((10, 2, 3), 0), ((2, 10, 3), 1), ((2, 3, 10), 2)]:
        for style in [None, "total", "broadcast"]:
            for mom in [3, (3, 3)]:
                yield shape, axis, style, mom


@pytest.fixture(params=get_params(), scope="module")  # type: ignore[call-overload]
def other(request):
    return Data(*request.param)

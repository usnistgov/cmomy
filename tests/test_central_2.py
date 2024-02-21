# mypy: disable-error-code="no-untyped-def, no-untyped-call"
from __future__ import annotations

import numpy as np
import pytest

from cmomy import central_moments

from ._simple_cmom import get_cmom, get_comom


class DataContainer:
    def __init__(self, x, y, w) -> None:
        self.x = x
        self.y = y
        self.w = w

    def xy(self, cov):
        if cov:
            return (self.x, self.y)
        return self.x

    def result_expected(self, axis, mom, broadcast=None, cov=None):
        if cov is None:
            cov = isinstance(mom, tuple) and len(mom) == 2

        if not cov:
            return get_cmom(x=self.x, w=self.w, moments=mom, axis=axis, last=True)
        return get_comom(
            x=self.x,
            y=self.y,
            w=self.w,
            moments=mom,
            axis=axis,
            broadcast=broadcast,  # pyright: ignore[reportArgumentType]
        )

    def result_central_moments(self, axis, mom, broadcast=None, cov=None, **kws):
        if cov is None:
            cov = isinstance(mom, tuple) and len(mom) == 2
        return central_moments(
            x=self.xy(cov),
            w=self.w,
            mom=mom,
            axis=axis,
            broadcast=broadcast,  # pyright: ignore[reportArgumentType]
            **kws,  # pyright: ignore[reportArgumentType]
        )

    @classmethod
    def from_params(cls, shape, axis, style, rng):
        if isinstance(shape, int):
            shape = (shape,)

        x = rng.random(shape)

        if style is None:
            y = rng.random(shape)
            w = None
        if style == "total":
            y = rng.random(shape)
            w = rng.random(shape)
        elif style == "broadcast":
            y = rng.random(shape[axis])
            w = rng.random(shape[axis])
        return cls(x=x, y=y, w=w)


class ExpectedResults:
    def __init__(self, data, shape, axis, mom, style) -> None:
        self.data = data
        self.shape = shape
        self.axis = axis
        self.mom = mom
        self.style = style

    @property
    def broadcast(self):
        return self.style == "broadcast"

    @property
    def shape_tuple(self):
        if isinstance(self.shape, int):
            return (self.shape,)
        return self.shape

    @property
    def mom_tuple(self):
        if isinstance(self.mom, int):
            return (self.mom,)
        return self.mom

    @property
    def cov(self):
        return self.data.y is not None

    def _result_kws(self, **kws):
        return dict(axis=self.axis, mom=self.mom, broadcast=self.broadcast, **kws)

    def result_expected(self):
        return self.data.result_expected(**self._result_kws())

    def result_central_moments(self, **kws):
        return self.data.result_central_moments(**self._result_kws(**kws))

    @property
    def x(self):
        return self.data.xy

    @property
    def w(self):
        return self.data.w


@pytest.fixture(
    scope="module",
    params=[
        (10, 0),
        ((10,), 0),
        ((1, 2, 3), 0),
        ((5, 6, 7), 0),
        ((5, 6, 7), 1),
        ((5, 6, 7), 2),
        ((5, 6, 7), -1),
        ((5, 6, 7), -2),
    ],
)
def shape_axis(request):
    return request.param


@pytest.fixture(scope="module")
def shape(shape_axis):
    return shape_axis[0]


@pytest.fixture(scope="module")
def shape_tuple(shape):
    if isinstance(shape, int):
        return (shape,)
    return shape


@pytest.fixture(scope="module")
def axis(shape_axis):
    return shape_axis[1]


@pytest.fixture(scope="module", params=[None, "total", "broadcast"])
def style(request):
    return request.param


@pytest.fixture(scope="module")
def data(shape, axis, style, rng):
    return DataContainer.from_params(shape=shape, axis=axis, style=style, rng=rng)


@pytest.fixture(scope="module", params=[3, (3, 3)])
def mom(request):
    return request.param


@pytest.fixture(scope="module")
def result_container(data, shape, axis, mom, style):
    return ExpectedResults(data=data, shape=shape, axis=axis, mom=mom, style=style)


def test_result(result_container) -> None:
    r = result_container
    a = r.result_expected()
    b = r.result_central_moments()
    np.testing.assert_allclose(a, b)


# @pytest.fixture(scope="module")
# def xdata(shape_tuple):
#     return default_rng.random(shape_tuple)


# @pytest.fixture(scope="module")
# def ydata(shape_tuple, style, axis):
#     shape = shape_tuple
#     if style is None or style == "total":
#         return default_rng.random(shape)
#     elif style == "broadcast":
#         return default_rng.random(shape[axis])
#     else:
#         raise ValueError


# @pytest.fixture(scope="module")
# def wdata(shape_tuple, style, axis):
#     shape = shape_tuple
#     if style is None:
#         return None
#     elif style == "total":
#         return default_rng.random(shape)
#     elif style == "broadcast":
#         return default_rng.random(shape[axis])


# @pytest.fixture(scope="module")
# def mom_tuple(mom):
#     if isinstance(mom, int):
#         return (mom,)
#     else:
#         return mom


# @pytest.fixture(scope="module")
# def cov(mom_tuple):
#     return len(mom_tuple) == 2


# @pytest.fixture(scope="module")
# def xydata(xdata, ydata, cov):
#     if cov:
#         return (xdata, ydata)
#     else:
#         return xdata


# @pytest.fixture(scope="module")
# def val_shape(shape_tuple, axis):
#     shape = shape_tuple
#     axis = normalize_axis_index(axis, len(shape))
#     return shape[:axis] + shape[axis + 1 :]


# @pytest.fixture(scope="module")
# def broadcast(style):
#     return style == "broadcast"


# @pytest.fixture(scope="module")
# def expected(xdata, ydata, wdata, mom_tuple, broadcast, axis):

#     if len(mom_tuple) == 1:
#         return get_cmom(w=wdata, x=xdata, moments=mom_tuple[0], axis=axis, last=True)

#     else:
#         return get_comom(
#             w=wdata, x=xdata, y=ydata, axis=axis, moments=mom_tuple, broadcast=broadcast
#         )

# @pytest.fixture
# def c_obj(xydata, wdata, mom, broadcast, axis):
#     return CentralMoments.from_vals(
#         x=xydata, w=wdata, axis=axis, mom=mom, broadcast=broadcast
#     )

# def test_simple(expected):
#     assert isinstance(expected, np.ndarray)


# def test_central_moments(xydata, wdata, mom, broadcast, axis, expected):
#     out = central_moments(
#         x=xydata, mom=mom, w=wdata, axis=axis, last=True, broadcast=broadcast
#     )
#     np.testing.assert_allclose(out, expected)
#     # out = central_moments(

#     # test using data
#     out = np.zeros_like(expected)
#     _ = central_moments(
#         x=xydata, mom=mom, w=wdata, axis=axis, last=True, broadcast=broadcast, out=out
#     )
#     np.testing.assert_allclose(out, expected)


# def test_mom_ndim():
#     with pytest.raises(ValueError):
#         CentralMoments(np.zeros((4, 4)), mom_ndim=0)

#     with pytest.raises(ValueError):
#         CentralMoments(np.zeros((4, 4)), mom_ndim=3)


# def test_c_obj(c_obj, expected):
#     np.testing.assert_allclose(c_obj.data, expected, rtol=1e-10, atol=1e-10)


# def test_properties(c_obj, shape_tuple, val_shape, mom_tuple):
#     assert val_shape == c_obj.val_shape

#     assert mom_tuple == c_obj.mom
#     assert len(mom_tuple) == c_obj.mom_ndim


# def test_push_vals(xydata, wdata, mom, mom_tuple, broadcast, axis, c_obj, val_shape):

#     assert val_shape == c_obj.val_shape

#     # create new
#     new = c_obj.zeros_like()
#     new.push_vals(x=xydata, w=wdata, axis=axis, broadcast=broadcast)
#     np.testing.assert_allclose(new.data, c_obj.data)

#     # create new
#     new = CentralMoments.zeros(mom=mom, val_shape=val_shape)
#     new.push_vals(x=xydata, w=wdata, axis=axis, broadcast=broadcast)
#     np.testing.assert_allclose(new.data, c_obj.data)


# def test_push(xydata, wdata, mom, mom_tuple, broadcast, style, axis, c_obj, val_shape):

#     if len(mom_tuple) == 1:
#         x = xydata
#         y = None
#     else:
#         x, y = xydata

#     x = np.moveaxis(x, axis, 0)

#     # w
#     if style is None:
#         w_ = (None for _ in x)

#     elif style == "total":
#         w_ = (_ for _ in np.moveaxis(wdata, axis, 0))

#     else:
#         w_ = (_ for _ in wdata)

#     if y is None:
#         xy_ = (_ for _ in x)

#     else:
#         if style is None or style == "total":
#             y = np.moveaxis(y, axis, 0)

#         xy_ = zip(x, y)

#     new = CentralMoments.zeros(val_shape=val_shape, mom=mom)

#     for xy, w in zip(xy_, w_):
#         new.push_val(x=xy, w=w, broadcast=broadcast)

#     np.testing.assert_allclose(c_obj.data, new.data)


# def split_data(xdata, ydata, wdata, axis, style, nsplit):

#     v = xdata.shape[axis] // nsplit
#     splits = [v * i for i in range(1, nsplit)]
#     X = np.split(xdata, splits, axis=axis)

#     if style == "total":
#         W = np.split(wdata, splits, axis=axis)
#         Y = np.split(ydata, splits, axis=axis)
#     elif style == "broadcast":
#         W = np.split(wdata, splits)
#         Y = np.split(ydata, splits)
#     elif style is None:
#         W = [wdata for _ in X]
#         Y = np.split(ydata, splits, axis=axis)

#     # Stopping here for now.  Will continue down  the road
#     pass

#     # # test from vals
#     # c = CentralMoments.from_vals(x=xydata, w=wdata, axis=axis, mom=mom, broadcast=broadcast)
#     # np.testing.assert_allclose(c.data, expected, rtol=1e-10, atol=1e-10)

#     # # test push


# # class TestCentral:
# #     @pytest.fixture(autouse=True)
# #     def setup(self, shape, axis):
# #         self.x = default_rng.random(shape)

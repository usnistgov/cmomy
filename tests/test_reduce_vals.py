from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import xarray as xr

# from cmomy import central_from_values
from cmomy.reduction import reduce_vals

from ._simple_cmom import get_cmom, get_comom

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from cmomy.typing import ArrayOrder, Mom_NDim, NDArrayAny


@pytest.fixture(scope="module", params=[(), (2,), (2, 3)])
def val_shape(request: pytest.FixtureRequest) -> tuple[int, ...]:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(scope="module", params=[100])
def nsamp(request: pytest.FixtureRequest) -> int:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def axes(val_shape: tuple[int, ...]) -> list[int]:
    return list(range(len(val_shape) + 1))


@pytest.fixture(scope="module")
def val_shapes(
    val_shape: tuple[int, ...], axes: list[int], nsamp: int
) -> list[tuple[int, ...]]:
    out = []
    for axis in axes:
        shape = list(val_shape)
        shape.insert(axis, nsamp)
        out.append(tuple(shape))
    return out


@pytest.fixture(scope="module")
def x_values(
    val_shapes: list[tuple[int, ...]], rng: np.random.Generator
) -> list[NDArrayAny]:
    return [rng.random(val_shape) for val_shape in val_shapes]


def _v_to_value(
    rng: np.random.Generator, style: str, nsamp: int, val_shape: tuple[int, ...]
) -> None | NDArrayAny:
    if style is None:
        return None
    if style == "nsamp":
        return rng.random(nsamp)
    return rng.random(val_shape)


@pytest.fixture(scope="module", params=[None, "nsamp", "total"])
def w_values(
    request: pytest.FixtureRequest,
    val_shapes: list[tuple[int, ...]],
    nsamp: int,
    rng: np.random.Generator,
) -> list[NDArrayAny | None]:
    return [
        _v_to_value(rng, request.param, nsamp=nsamp, val_shape=val_shape)
        for val_shape in val_shapes
    ]


@pytest.fixture(scope="module", params=["nsamp", "total"])
def y_values(
    request: pytest.FixtureRequest,
    val_shapes: list[tuple[int, ...]],
    nsamp: int,
    rng: np.random.Generator,
) -> list[NDArrayAny | None]:
    return [
        _v_to_value(rng, request.param, nsamp=nsamp, val_shape=val_shape)
        for val_shape in val_shapes
    ]


@pytest.fixture(scope="module")
def dims_values(
    axes: list[int], val_shapes: list[tuple[int, ...]]
) -> list[tuple[str, ...]]:
    out = []
    for axis, shape in zip(axes, val_shapes):
        dims = [f"dim_{i}" for i in range(len(shape))]
        dims[axis] = "rec"
        out.append(tuple(dims))
    return out


@pytest.fixture(scope="module")
def dx_values(
    x_values: list[NDArrayAny], dims_values: list[tuple[str, ...]]
) -> list[xr.DataArray]:
    return [xr.DataArray(_x, dims=_dims) for _x, _dims in zip(x_values, dims_values)]


def _v_to_dv(values: list[Any], dims_values: list[tuple[str, ...]]) -> list[Any]:
    out: list[Any] = []
    for _v, _dims in zip(values, dims_values):
        if _v is None:
            out.append(_v)
        elif _v.ndim == 1:
            out.append(xr.DataArray(_v, dims="rec"))
        else:
            out.append(xr.DataArray(_v, dims=_dims))
    return out


@pytest.fixture(scope="module")
def dw_values(w_values: list[Any], dims_values: list[tuple[str, ...]]) -> list[Any]:
    return _v_to_dv(w_values, dims_values)


@pytest.fixture(scope="module")
def dy_values(y_values: list[Any], dims_values: list[tuple[str, ...]]) -> list[Any]:
    return _v_to_dv(y_values, dims_values)


# * Moments
mom_central = pytest.mark.parametrize("mom", [2, 5])


@mom_central
@pytest.mark.parametrize("out_style", [None, "set"])
def test_central_moments(
    mom: Mom_NDim,
    axes: list[int],
    x_values: list[NDArrayAny],
    w_values: list[Any],
    dx_values: list[xr.DataArray],
    dw_values: list[Any],
    out_style: Any,
) -> None:
    for (
        axis,
        x,
        w,
        dx,
        dw,
    ) in zip(axes, x_values, w_values, dx_values, dw_values):
        expected = get_cmom(w, x, mom, axis=axis)  # type: ignore[no-untyped-call]

        for xx, ww in [(x, w), (dx, dw)]:
            if out_style is None:
                out = reduce_vals(xx, mom=mom, weight=ww, axis=axis)
            elif out_style == "set":
                out = np.zeros_like(expected)
                _ = reduce_vals(xx, mom=mom, weight=ww, axis=axis, out=out)
            np.testing.assert_allclose(out, expected)


@mom_central
@pytest.mark.parametrize("dtype", [None, np.float64])
@pytest.mark.parametrize("order", [None, "C"])
def test_central_moments_dtype_order(
    mom: Mom_NDim,
    axes: list[int],
    x_values: list[NDArrayAny],
    w_values: list[Any],
    dtype: DTypeLike,
    order: ArrayOrder,
) -> None:
    for axis, x, w in zip(axes, x_values, w_values):
        expected = get_cmom(  # type: ignore[no-untyped-call]
            w if w is None else w.astype(dtype), x.astype(dtype), mom, axis=axis
        )
        out = reduce_vals(x.astype(dtype), mom=mom, weight=w, axis=axis, order=order)

        assert out.dtype == expected.dtype
        np.testing.assert_allclose(out, expected)


def test_central_moments_raises() -> None:
    x = np.ones((10, 20))
    out = np.ones((2, 3))
    with pytest.raises(ValueError):
        reduce_vals(x, mom=5, axis=0, out=out)


# ** Comoments
comom_central = pytest.mark.parametrize("mom", [(2, 2), (4, 4)])


@comom_central
@pytest.mark.parametrize("out_style", [None, "set"])
def test_central_comoments(
    mom: Mom_NDim,
    axes: list[int],
    x_values: list[NDArrayAny],
    y_values: list[Any],
    w_values: list[Any],
    dx_values: list[xr.DataArray],
    dy_values: list[Any],
    dw_values: list[Any],
    out_style: Any,
) -> None:
    for axis, x, y, w, dx, dy, dw in zip(
        axes, x_values, y_values, w_values, dx_values, dy_values, dw_values
    ):
        expected = get_comom(w, x, y=y, moments=mom, axis=axis)  # type: ignore[no-untyped-call]

        for xx, yy, ww in [(x, y, w), (dx, dy, dw)]:
            if out_style is None:
                out = reduce_vals(xx, yy, mom=mom, weight=ww, axis=axis)
            elif out_style == "set":
                out = np.zeros_like(expected)
                _ = reduce_vals(x, yy, mom=mom, weight=w, axis=axis, out=out)
            np.testing.assert_allclose(out, expected)

        # if out_style is None:
        #     out = central_from_values.central_moments(x=(x, y), mom=mom, w=w, axis=axis, broadcast=True)
        # elif out_style == "set":
        #     out = np.zeros_like(expected)
        #     _ = central_from_values.central_moments(x=(x, y), mom=mom, w=w, axis=axis, out=out, broadcast=True)
        # assert out.dtype == expected.dtype
        # np.testing.assert_allclose(out, expected)


# @comom_central
# @pytest.mark.parametrize("dtype", [None, np.float64])
# @pytest.mark.parametrize("order", [None, "C"])
# def test_central_comoments_dtype_order(
#         mom: Mom_NDim,
#         axes: list[int],
#         x_values: list[NDArray],
#         y_values: list[Any],
#         w_values: list[Any],
#         dtype: DTypeLike, order: ArrayOrder) -> None:

#     for axis, x, y, w in zip(axes, x_values, y_values, w_values):
#         expected = get_comom(w=w if w is None else w.astype(dtype), x=x.astype(dtype), y=y.astype(dtype), moments=mom, axis=axis)
#         out = central_from_values.central_moments(x=(x, y), mom=mom, w=w, axis=axis, dtype=dtype, order=order, broadcast=True)

#         assert out.dtype == expected.dtype
#         np.testing.assert_allclose(out, expected)


# def test_central_comoments_raises():

#     x = np.ones(10)

#     with pytest.raises(ValueError):
#         central_from_values.central_moments(x=(x, x), mom=(3,3,3), axis=0)

#     with pytest.raises(TypeError):
#         central_from_values.central_moments(x=x, mom=(3,3), axis=0)

#     with pytest.raises(ValueError):
#         central_from_values.central_moments(x=(x, x, x), mom=(3,3), axis=0)

#     y = np.ones((10, 10))

#     with pytest.raises(ValueError):
#         central_from_values.central_moments(x=(x, y), mom=(3,3), axis=0)

#     out = np.ones((2, 3))
#     with pytest.raises(ValueError):
#         central_from_values.central_moments(x=(x, x), mom=(3, 3), axis=0, out=out)


#     # no last
#     xx = np.ones((10, 2, 3))
#     out = central_from_values.central_moments(x=(xx, xx), mom=(3, 3), axis=0, last=False)
#     assert out.shape == (4, 4, 2, 3)

#     out = central_from_values.central_moments(x=(xx, xx), mom=(3, 3), axis=0, last=True)
#     assert out.shape == (2, 3, 4, 4)


# # * DataArray

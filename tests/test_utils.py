# mypy: disable-error-code="no-untyped-def, no-untyped-call, call-overload"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

import cmomy
from cmomy import utils
from cmomy.core.utils import mom_to_mom_ndim
from cmomy.core.validate import is_dataarray, is_ndarray

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from cmomy.core.typing import Mom_NDim, NDArrayAny


def _do_test(func, *args, expected=None, match=None, **kwargs):
    if isinstance(expected, type):
        with pytest.raises(expected, match=match):
            func(*args, **kwargs)
    else:
        assert func(*args, **kwargs) == expected


@pytest.mark.parametrize(
    "x", [np.zeros((1, 2, 3, 4)), xr.DataArray(np.zeros((1, 2, 3, 4)))]
)
@pytest.mark.parametrize(
    "func", [lambda *args, **kwargs: utils.moveaxis(*args, **kwargs).shape]
)
@pytest.mark.parametrize(
    ("kws", "expected"),
    [
        ({"axis": 0, "dest": -1, "mom_ndim": None}, (2, 3, 4, 1)),
        ({"axis": 0, "dest": -1, "mom_ndim": 1}, (2, 3, 1, 4)),
        ({"axis": 0, "dest": -1, "mom_ndim": 2}, (2, 1, 3, 4)),
        ({"axis": (1, 0), "dest": (-2, -1), "mom_ndim": 1}, (3, 2, 1, 4)),
        ({"axis": (1, 0), "dest": (-2,), "mom_ndim": 1}, ValueError),
    ],
)
def test_moveaxis(x, kws, expected, func):
    _do_test(func, x, **kws, expected=expected)


@pytest.mark.parametrize("x", [xr.DataArray(np.zeros((1, 2, 3, 4)))])
@pytest.mark.parametrize(
    ("kws", "expected"),
    [
        ({"dim": "dim_0", "dest": -1, "mom_ndim": None}, (2, 3, 4, 1)),
        ({"dim": "dim_0", "dest": -1, "mom_ndim": 1}, (2, 3, 1, 4)),
        ({"dim": "dim_0", "dest": -1, "mom_ndim": 2}, (2, 1, 3, 4)),
        ({"dim": ("dim_1", "dim_0"), "dest": (-2, -1), "mom_ndim": 1}, (3, 2, 1, 4)),
        ({"dim": ("dim_1", "dim_0"), "dest": (-2,), "mom_ndim": 1}, ValueError),
        ({"dim": "dim_0", "dest_dim": "dim_3", "mom_ndim": None}, (2, 3, 4, 1)),
        ({"dim": "dim_0", "dest_dim": "dim_2", "mom_ndim": 1}, (2, 3, 1, 4)),
        ({"dim": "dim_0", "dest_dim": "dim_1", "mom_ndim": 2}, (2, 1, 3, 4)),
        (
            {"dim": ("dim_1", "dim_0"), "dest_dim": ("dim_1", "dim_2"), "mom_ndim": 1},
            (3, 2, 1, 4),
        ),
        ({"dim": ("dim_1", "dim_0"), "dest_dim": "dim_2", "mom_ndim": 1}, ValueError),
    ],
)
@pytest.mark.parametrize(
    "func", [lambda *args, **kwargs: utils.moveaxis(*args, **kwargs).shape]
)
def test_moveaxis_dataarray(x, kws, expected, func):
    _do_test(func, x, **kws, expected=expected)


@pytest.mark.parametrize(
    ("shape", "kwargs", "expected", "match"),
    [
        (
            (2,),
            {"name": "weight", "mom_ndim": 2},
            ValueError,
            ".*must be.*",
        ),
        (
            (3, 3),
            {"name": "yave", "mom_ndim": 1},
            ValueError,
            ".*Unknown option.*",
        ),
        (
            (3, 3),
            {"name": "yvar", "mom_ndim": 1},
            ValueError,
            ".*Unknown option.*",
        ),
        ((2,), {"name": "thing", "mom_ndim": 1}, ValueError, ".*Unknown option.*"),
    ],
)
def test_select_moment_errors(shape, kwargs, expected, match) -> None:
    data = np.empty(shape)
    _do_test(utils.select_moment, data, **kwargs, expected=expected, match=match)


def test_select_moment_errors_coords() -> None:
    data = xr.DataArray(np.empty((3, 3)))
    with pytest.raises(ValueError, match=".*must equal.*"):
        utils.select_moment(data, "ave", mom_ndim=2, coords_combined="hello")


def _do_test_select_moment_mom_ndim(
    data,
    mom_ndim,
    index,
    dim_combined,
    coords_combined,
    **kwargs,
):
    out = utils.select_moment(
        data,
        **kwargs,
        mom_ndim=mom_ndim,
        dim_combined=dim_combined,
        coords_combined=coords_combined,
    )

    np.testing.assert_allclose(out, np.asarray(data)[index])

    if (
        is_dataarray(out)
        and kwargs["name"] in {"ave", "var"}
        and (mom_ndim != 1 or not kwargs.get("squeeze", True))
    ):
        dim_combined = dim_combined or "variable"
        if coords_combined is None:
            coords_combined = list(data.dims[-mom_ndim:])
        elif isinstance(coords_combined, str):
            coords_combined = [coords_combined]
        else:
            coords_combined = list(coords_combined)

        assert out.dims[-1] == dim_combined
        assert out.coords[dim_combined].to_numpy().tolist() == coords_combined


@pytest.mark.parametrize(
    "shape",
    [
        (3,),
        (2, 3),
        (1, 2, 3),
    ],
)
@pytest.mark.parametrize(
    ("kwargs", "index"),
    [
        ({"name": "weight"}, (..., 0)),
        ({"name": "ave"}, (..., 1)),
        ({"name": "var"}, (..., 2)),
        ({"name": "ave", "squeeze": False}, (..., [1])),
        ({"name": "var", "squeeze": False}, (..., [2])),
        ({"name": "xave"}, (..., 1)),
        ({"name": "xvar"}, (..., 2)),
        ({"name": "cov"}, (..., 2)),
    ],
)
@pytest.mark.parametrize("dim_combined", ["variable", "hello"])
@pytest.mark.parametrize("coords_combined", [None, "thing"])
def test_select_moment_mom_ndim_1(
    rng, shape, kwargs, index, as_dataarray, dim_combined, coords_combined
) -> None:
    data = rng.random(shape)
    if as_dataarray:
        data = xr.DataArray(data)
    _do_test_select_moment_mom_ndim(
        data,
        1,
        index,
        dim_combined,
        coords_combined,
        **kwargs,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (3, 3),
        (2, 3, 3),
        (1, 2, 3, 3),
    ],
)
@pytest.mark.parametrize(
    ("kwargs", "index"),
    [
        ({"name": "weight"}, (..., 0, 0)),
        ({"name": "ave"}, (..., [1, 0], [0, 1])),
        ({"name": "var"}, (..., [2, 0], [0, 2])),
        ({"name": "xave"}, (..., 1, 0)),
        ({"name": "xvar"}, (..., 2, 0)),
        ({"name": "yave"}, (..., 0, 1)),
        ({"name": "yvar"}, (..., 0, 2)),
        ({"name": "cov"}, (..., 1, 1)),
        ({"name": "xmom_0"}, (..., 0, slice(None))),
        ({"name": "xmom_1"}, (..., 1, slice(None))),
        ({"name": "ymom_0"}, (..., slice(None), 0)),
        ({"name": "ymom_1"}, (..., slice(None), 1)),
    ],
)
@pytest.mark.parametrize("dim_combined", ["hello"])
@pytest.mark.parametrize("coords_combined", [None, ("thing", "other")])
def test_select_moment_mom_ndim_2(
    rng,
    shape,
    kwargs,
    index,
    as_dataarray,
    dim_combined,
    coords_combined,
) -> None:
    data = rng.random(shape)
    if as_dataarray:
        data = xr.DataArray(data)
    _do_test_select_moment_mom_ndim(
        data, 2, index, dim_combined, coords_combined, **kwargs
    )


def _do_test_assign_moment_mom_ndim(
    data,
    mom_ndim,
    index,
    copy,
    scalar,
    name,
    use_xr_value=True,
    **kwargs,
):
    value: float | NDArrayAny | xr.DataArray
    if scalar:
        value = -10
    elif is_dataarray(data):
        template = utils.select_moment(
            data, name, mom_ndim=mom_ndim, squeeze=kwargs.get("squeeze", True)
        )
        value = xr.full_like(template, -10)
        if "variable" in value.dims:
            kwargs["dim_combined"] = "variable"

        if not use_xr_value:
            value = np.asarray(value)

    else:
        shape = data[index].shape
        if len(shape) > 2:
            shape = shape[-2:]
        value = np.full(shape, fill_value=-10)

    check = data.copy()
    if is_dataarray(check):
        check.data[index] = value
    else:
        check[index] = value

    out = utils.assign_moment(
        data,
        {name: value},
        **kwargs,
        copy=copy,
        mom_ndim=mom_ndim,
    )

    if not copy:
        assert np.shares_memory(out, data)

    np.testing.assert_allclose(out, check)

    # CentralMoments
    c0 = cmomy.wrap(data, mom_ndim=mom_ndim)
    c1 = c0.assign_moment({name: value}, **kwargs, copy=copy)

    np.testing.assert_allclose(out, c1)

    if not copy:
        assert np.shares_memory(c0, c1)


@pytest.mark.parametrize(
    "shape",
    [
        (3,),
        (2, 3),
        (1, 2, 3),
    ],
)
@pytest.mark.parametrize(
    ("kwargs", "index"),
    [
        ({"name": "weight"}, (..., 0)),
        ({"name": "ave"}, (..., 1)),
        ({"name": "var"}, (..., 2)),
        ({"name": "ave", "squeeze": False}, (..., [1])),
        ({"name": "var", "squeeze": False}, (..., [2])),
        ({"name": "xave"}, (..., 1)),
        ({"name": "xvar"}, (..., 2)),
        ({"name": "cov"}, (..., 2)),
        ({"name": "all"}, (...,)),
    ],
)
@pytest.mark.parametrize("scalar", [True, False])
@pytest.mark.parametrize("copy", [True, False])
def test_assign_moment_mom_ndim_1(
    rng,
    shape,
    kwargs,
    index,
    as_dataarray,
    scalar,
    copy,
) -> None:
    data = rng.random(shape)
    if as_dataarray:
        data = xr.DataArray(data)

    _do_test_assign_moment_mom_ndim(data, 1, index, scalar=scalar, copy=copy, **kwargs)

    if as_dataarray:
        _do_test_assign_moment_mom_ndim(
            data, 1, index, scalar=scalar, copy=copy, use_xr_value=False, **kwargs
        )


@pytest.mark.parametrize(
    "shape",
    [
        (3, 3),
        (2, 3, 3),
        (1, 2, 3, 3),
    ],
)
@pytest.mark.parametrize(
    ("kwargs", "index"),
    [
        ({"name": "weight"}, (..., 0, 0)),
        ({"name": "ave"}, (..., [1, 0], [0, 1])),
        ({"name": "var"}, (..., [2, 0], [0, 2])),
        ({"name": "xave"}, (..., 1, 0)),
        ({"name": "xvar"}, (..., 2, 0)),
        ({"name": "yave"}, (..., 0, 1)),
        ({"name": "yvar"}, (..., 0, 2)),
        ({"name": "cov"}, (..., 1, 1)),
        ({"name": "xmom_0"}, (..., 0, slice(None))),
        ({"name": "xmom_1"}, (..., 1, slice(None))),
        ({"name": "ymom_0"}, (..., slice(None), 0)),
        ({"name": "ymom_1"}, (..., slice(None), 1)),
    ],
)
@pytest.mark.parametrize("scalar", [True, False])
@pytest.mark.parametrize("copy", [True, False])
def test_assign_moment_mom_ndim_2(
    rng,
    shape,
    kwargs,
    index,
    as_dataarray,
    scalar,
    copy,
) -> None:
    data = rng.random(shape)
    if as_dataarray:
        data = xr.DataArray(data)

    _do_test_assign_moment_mom_ndim(data, 2, index, scalar=scalar, copy=copy, **kwargs)


_rng = np.random.default_rng(0)


@pytest.mark.parametrize(
    ("data", "mom_ndim"),
    [
        (_rng.random(3), 1),
        (_rng.random((3, 3)), 1),
        (_rng.random((3, 3, 3)), 1),
        (_rng.random((3, 3)), 2),
        (_rng.random((3, 3, 3)), 2),
        (_rng.random((3, 3, 3, 3)), 2),
    ],
)
@pytest.mark.parametrize(
    "wrapper",
    [lambda x: x, xr.DataArray, lambda x: xr.Dataset({"x": xr.DataArray(x)})],
)
@pytest.mark.parametrize(
    "moment",
    [
        {"weight": 0, "ave": 1},
        {"weight": 0, "var": 1},
    ],
)
def test_assign_moment_multiple(data, mom_ndim, wrapper, moment) -> None:
    kwargs = {"mom_ndim": mom_ndim}
    data = wrapper(data)

    expected = data
    for k, v in moment.items():
        expected = cmomy.assign_moment(expected, {k: v}, **kwargs)

    out = cmomy.assign_moment(data, moment, **kwargs)

    if is_ndarray(data):
        np.testing.assert_allclose(expected, out)
    else:
        xr.testing.assert_allclose(expected, out)


# * Vals -> Data
@pytest.mark.parametrize(
    ("xshape", "yshape", "wshape", "mom", "out_shape"),
    [
        ((2,), None, None, (2,), (2, 3)),
        (None, None, (2,), (2,), (2, 3)),
        ((2, 3), None, (4, 1, 1), (2,), (4, 2, 3, 3)),
        ((2,), (2,), None, (2, 2), (2, 3, 3)),
        ((1, 2), (3, 1, 1), None, (2, 2), (3, 1, 2, 3, 3)),
        ((2,), (2,), (1, 2), (2, 2), (1, 2, 3, 3)),
    ],
)
def test_vals_to_data(xshape, yshape, wshape, mom, out_shape) -> None:
    w: ArrayLike = 0.1 if wshape is None else np.full(wshape, 0.1)
    x: ArrayLike = 0.2 if xshape is None else np.full(xshape, 0.2)
    y: ArrayLike = 0.3 if yshape is None else np.full(yshape, 0.3)

    xy: tuple[ArrayLike, ...] = (x,) if len(mom) == 1 else (x, y)
    out = utils.vals_to_data(*xy, weight=w, mom=mom)
    assert out.shape == out_shape

    mom_ndim = mom_to_mom_ndim(mom)

    np.testing.assert_allclose(
        utils.select_moment(out, "weight", mom_ndim=mom_ndim), 0.1
    )
    np.testing.assert_allclose(utils.select_moment(out, "xave", mom_ndim=mom_ndim), 0.2)
    if mom_ndim == 2:
        np.testing.assert_allclose(
            utils.select_moment(out, "yave", mom_ndim=mom_ndim), 0.3
        )

    np.testing.assert_allclose(
        out.sum(),
        (
            np.prod(out.shape[: -len(mom)])
            * (0.1 + 0.2 + (0.0 if len(mom) == 1 else 0.3))
        ),
    )


def test_vals_to_data_errors() -> None:
    x = np.zeros(3)

    with pytest.raises(ValueError):
        utils.vals_to_data(x, mom=(2, 2))

    with pytest.raises(ValueError):
        utils.vals_to_data(x, x, mom=2)


def test_vals_to_data_xarray() -> None:
    x = xr.DataArray(np.full((1, 2), 0.2), dims=["a", "b"])
    out = utils.vals_to_data(x, weight=0.1, mom=2, dtype=np.float32)
    assert out.dtype.type == np.float32
    assert out.dims == ("a", "b", "mom_0")
    assert out.shape == (1, 2, 3)

    y = xr.DataArray(np.full((3, 4), 0.3), dims=["c", "d"])

    out0 = utils.vals_to_data(x, y, weight=0.1, mom=(2, 2), mom_dims=("x", "y"))
    assert out0.dims == ("a", "b", "c", "d", "x", "y")
    assert out0.shape == (1, 2, 3, 4, 3, 3)

    out = xr.DataArray(
        np.zeros((1, 2, 3, 4, 3, 3), dtype=np.float32),
        dims=["a", "b", "c", "d", "x", "y"],
    )

    w = xr.full_like(x, 0.1)
    check = utils.vals_to_data(x, y, weight=w, out=out, mom=(2, 2))
    assert check.data is out.data

    xr.testing.assert_allclose(out0, out.transpose(*out0.dims))
    mom_ndim: Mom_NDim = 2
    np.testing.assert_allclose(
        utils.select_moment(out, "weight", mom_ndim=mom_ndim), 0.1
    )
    np.testing.assert_allclose(utils.select_moment(out, "xave", mom_ndim=mom_ndim), 0.2)
    np.testing.assert_allclose(utils.select_moment(out, "yave", mom_ndim=mom_ndim), 0.3)

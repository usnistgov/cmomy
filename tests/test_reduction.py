# mypy: disable-error-code="no-untyped-def, no-untyped-call, call-overload, assignment"
# pyright: reportCallIssue=false, reportArgumentType=false

from __future__ import annotations

from contextlib import nullcontext
from functools import partial

import numpy as np
import pytest
import xarray as xr

import cmomy


# * reduce_vals -----------------------------------------------------------------
def test_reduce_vals_broadcast(rng: np.random.Generator) -> None:
    func = partial(cmomy.reduce_vals, axis=-1, mom=(2, 2))

    x = rng.random((2, 10))
    y = rng.random((2, 10))

    a = func(x, y)
    b = func(x[None, ...], y)
    c = func(x, y[None, ...])

    assert a.shape == (2, 3, 3)
    assert b.shape == (1, 2, 3, 3)
    assert c.shape == (1, 2, 3, 3)

    np.testing.assert_allclose(a[None, ...], b)
    np.testing.assert_allclose(b, c)


@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((10, 2, 3), 0),
        ((2, 10, 3), 1),
        ((2, 3, 10), 2),
    ],
)
def test_reduce_vals_axis(rng, shape, axis) -> None:
    x = rng.random(shape)
    func = partial(cmomy.reduce_vals, mom=3)

    a = func(x, axis=axis)
    b = func(np.moveaxis(x, axis, -1), axis=-1)

    np.testing.assert_allclose(a, b)


# * reduce_data ---------------------------------------------------------------
@pytest.mark.parametrize(
    ("shape", "axis", "mom_ndim"),
    [
        ((2, 3, 4, 3), 0, 1),
        ((2, 3, 4, 3), 1, 1),
        ((2, 3, 4, 3), 1, 2),
        ((2, 3, 4, 3), 2, 1),
    ],
)
def test_reduce_data_keepdims(shape, axis, mom_ndim, rng, as_dataarray: bool) -> None:
    x = rng.random(shape)
    if as_dataarray:
        x = xr.DataArray(x)

    kws = {"mom_ndim": mom_ndim, "axis": axis}

    check = cmomy.reduce_data(x, **kws, keepdims=False)

    new_shape = list(shape)
    new_shape[axis] = 1
    new_shape = tuple(new_shape)  # type: ignore[assignment]

    out = cmomy.reduce_data(x, **kws, keepdims=True)
    assert out.shape == new_shape

    np.testing.assert_allclose(np.squeeze(out, axis), check)

    cls = cmomy.CentralMomentsData if as_dataarray else cmomy.CentralMomentsArray
    c = cls(x, mom_ndim=mom_ndim).reduce(
        axis=axis,
        keepdims=True,
    )
    assert c.shape == new_shape  # type: ignore[comparison-overlap]
    np.testing.assert_allclose(c, out)


@pytest.mark.parametrize(
    "shape",
    [(2, 3, 4, 5)],
)
@pytest.mark.parametrize(
    ("kwargs", "out_shape"),
    [
        (
            {"axis": 0, "mom_ndim": 1},
            (3, 4, 5),
        ),
        (
            {"axis": 0, "mom_ndim": 1, "keepdims": True},
            (1, 3, 4, 5),
        ),
        (
            {"axis": 0, "mom_ndim": 1, "keepdims": True, "axes_to_end": True},
            (3, 4, 1, 5),
        ),
        (
            {"axis": -1, "mom_axes": 1},
            (2, 3, 4),
        ),
        (
            {"axis": -1, "mom_axes": 0, "axes_to_end": True},
            (3, 4, 2),
        ),
        (
            {"axis": -1, "mom_axes": 1, "keepdims": True},
            (2, 3, 4, 1),
        ),
        (
            {"axis": -1, "mom_axes": 1, "keepdims": True, "axes_to_end": True},
            (2, 4, 1, 3),
        ),
        (
            {"axis": (0, -1), "mom_axes": (1, 2)},
            (3, 4),
        ),
        (
            {"axis": -1, "mom_axes": (0, 1), "axes_to_end": True},
            (4, 2, 3),
        ),
        (
            {"axis": (0, -1), "mom_axes": (1, 2), "keepdims": True},
            (1, 3, 4, 1),
        ),
        (
            {
                "axis": (0, -1),
                "mom_axes": (1, 2),
                "keepdims": True,
                "axes_to_end": True,
            },
            (1, 1, 3, 4),
        ),
    ],
)
def test_reduce_data_out(
    rng,
    shape,
    kwargs,
    out_shape,
) -> None:
    data = rng.random(shape)
    out = np.empty(out_shape, dtype=data.dtype)

    checks = [cmomy.reduce_data(data, **kwargs, out=o) for o in [out, None]]

    assert np.shares_memory(out, checks[0])
    np.testing.assert_allclose(*checks)


@pytest.mark.parametrize("mom_ndim", [1, 2])
def test_reduce_data_axis_none(rng, mom_ndim) -> None:
    data = rng.random((10, 2, 3, 4))

    data_collapse = data.reshape(-1, *data.shape[-mom_ndim:])

    expected = cmomy.reduce_data(data_collapse, axis=0, mom_ndim=mom_ndim)

    check = cmomy.reduce_data(data, axis=None, mom_ndim=mom_ndim)

    np.testing.assert_allclose(check, expected)


def test_reduce_data_dim_none(rng) -> None:
    ds = xr.Dataset(
        {
            "data0": xr.DataArray(
                rng.random((2, 3, 4, 5)), dims=["a", "b", "c", "mom"]
            ),
            "data1": xr.DataArray(rng.random((2, 3, 5)), dims=["a", "b", "mom"]),
            "data2": xr.DataArray(rng.random((4, 5)), dims=["c", "mom"]),
            "data3": xr.DataArray(rng.random((4,)), dims=["c"]),
        }
    )

    with pytest.raises(ValueError):
        cmomy.reduce_data(ds["data0"], mom_ndim=1, dim="d")

    with pytest.raises(ValueError, match="Dimensions .*"):
        cmomy.reduce_data(ds, mom_ndim=1, dim="d")

    # not using map with dim=None picks dim from first array
    out = cmomy.reduce_data(ds, dim=None, mom_dims="mom", use_map=False)
    for k in ["data0"]:
        xr.testing.assert_equal(
            out[k], cmomy.reduce_data(ds[k], dim=None, mom_dims="mom")
        )
    for k in ["data1", "data2", "data3"]:
        xr.testing.assert_equal(out[k], ds[k])

    out = cmomy.reduce_data(ds, dim=None, mom_dims="mom", use_map=True)
    for k in ["data0", "data1", "data2"]:
        xr.testing.assert_equal(
            out[k], cmomy.reduce_data(ds[k], dim=None, mom_dims="mom")
        )
    for k in ["data3"]:
        xr.testing.assert_equal(out[k], ds[k])

    out2 = cmomy.reduce_data(ds, dim=("a", "b", "c"), mom_dims="mom", use_map=True)
    xr.testing.assert_equal(out, out2)

    # specify a and b
    out = cmomy.reduce_data(ds, dim=("a", "b"), mom_dims="mom", use_map=True)
    for k in ["data0", "data1"]:
        xr.testing.assert_equal(
            out[k], cmomy.reduce_data(ds[k], dim=("a", "b"), mom_dims="mom")
        )
    for k in ["data2", "data3"]:
        xr.testing.assert_equal(out[k], ds[k])

    for use_map in [True, False]:
        xr.testing.assert_allclose(
            cmomy.reduce_data(ds, dim=(), mom_dims="mom", use_map=use_map), ds
        )


# * utils ---------------------------------------------------------------------
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"ndat": 5, "block": -1}, nullcontext([0] * 5)),
        ({"ndat": 5, "block": 5}, nullcontext([0] * 5)),
        ({"ndat": 5, "block": 6}, pytest.raises(ValueError)),
        ({"ndat": 4, "block": 1}, nullcontext([0, 1, 2, 3])),
        ({"ndat": 4, "block": 2}, nullcontext([0, 0, 1, 1])),
        ({"ndat": 4, "block": 3}, nullcontext([0, 0, 0, -1])),
        ({"ndat": 5, "block": 2, "mode": "drop_last"}, nullcontext([0, 0, 1, 1, -1])),
        ({"ndat": 5, "block": 2, "mode": "expand_last"}, nullcontext([0, 0, 1, 1, 1])),
        ({"ndat": 5, "block": 2, "mode": "drop_first"}, nullcontext([-1, 0, 0, 1, 1])),
        ({"ndat": 5, "block": 2, "mode": "expand_first"}, nullcontext([0, 0, 0, 1, 1])),
        (
            {"ndat": 5, "block": 2, "mode": "hello"},
            pytest.raises(ValueError, match="Unknown .*"),
        ),
    ],
)
def test_block_by(kwargs, expected) -> None:
    with expected as e:
        by = cmomy.reduction.block_by(**kwargs)
        np.testing.assert_allclose(by, e)


# * grouped -------------------------------------------------------------------
@pytest.mark.parametrize(
    ("shape", "mom_ndim"),
    [
        ((16, 3), 1),
        ((16, 3, 3), 2),
    ],
)
@pytest.mark.parametrize("by", [[0] * 4 + [1] * 4 + [2] * 4 + [3] * 4])
def get_reduce_data_grouped_indexed(rng, shape, mom_ndim, by):
    data = rng.random(shape)
    expected = cmomy.reduce_data(data.reshape(4, 4, *shape[1:]), axis=1, mom_ndim=2)
    check = cmomy.reduce_data_grouped(data, by=by, axis=0, mom_ndim=mom_ndim)
    np.testing.assert_allclose(check, expected)

    _group, index, start, end = cmomy.reduction.factor_by_to_index(by)

    check = cmomy.reduction.reduce_data_indexed(
        data, index=index, group_start=start, group_end=end, axis=0, mom_ndim=mom_ndim
    )
    np.testing.assert_allclose(check, expected)


def test_indexed_bad_scale(rng: np.random.Generator) -> None:
    data = rng.random((10, 2, 3))
    by = [0] * 5 + [1] * 5
    _group, index, start, end = cmomy.reduction.factor_by_to_index(by)
    # bad scale
    with pytest.raises(ValueError, match=".*`scale` and `index`.*"):
        _ = cmomy.reduction.reduce_data_indexed(
            data,
            mom_ndim=1,
            index=index,
            group_start=start,
            group_end=end,
            scale=[1] * 11,
            axis=0,
        )


@pytest.mark.parametrize(
    "by",
    [
        [0] * 10 + [1] * 10,
        [0] * 9,
    ],
)
def test_grouped_bad_by(by: list[int]) -> None:
    data = np.zeros((10, 2, 4))
    with pytest.raises(ValueError, match=".*Wrong length of `by`.*"):
        cmomy.reduce_data_grouped(data, mom_ndim=1, by=by, axis=0)


# * utils ---------------------------------------------------------------------
def test__validate_index() -> None:
    index = [0, 1, 2, 3]
    group_start = [0, 2]
    group_end = [2, 4]

    index_, start_, end_ = cmomy.reduction._validate_index(
        4, index, group_start, group_end
    )

    np.testing.assert_allclose(index, index_)
    np.testing.assert_allclose(group_start, start_)
    np.testing.assert_allclose(group_end, end_)

    # index outside bounds
    with pytest.raises(ValueError, match=".*min.*< 0.*"):
        _ = cmomy.reduction._validate_index(4, [-1, 0, 1, 2], group_start, group_end)

    # index outside max
    with pytest.raises(ValueError, match=".*max.*>.*"):
        _ = cmomy.reduction._validate_index(4, [0, 1, 2, 3, 4], group_start, group_end)

    # mismatch group start/end
    with pytest.raises(ValueError, match=r".*`group_start` and `group_end`.*"):
        _ = cmomy.reduction._validate_index(4, index, [0, 1], [1, 2, 3])

    # end < start
    with pytest.raises(ValueError, match=".*end < start.*"):
        _ = cmomy.reduction._validate_index(4, index, [0, 2], [2, 1])
    # zero length index
    index = []
    group_start = [0]
    group_end = [0]

    index_, start_, end_ = cmomy.reduction._validate_index(
        4, index, group_start, group_end
    )

    assert len(index_) == 0
    np.testing.assert_allclose(group_start, start_)
    np.testing.assert_allclose(group_end, end_)

    # bad end
    with pytest.raises(ValueError, match=".*With zero length.*"):
        _ = cmomy.reduction._validate_index(4, index, group_start, [10])

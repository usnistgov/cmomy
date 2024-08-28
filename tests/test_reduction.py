# mypy: disable-error-code="no-untyped-def, no-untyped-call, call-overload, assignment"
# pyright: reportCallIssue=false, reportArgumentType=false

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import xarray as xr

import cmomy


@pytest.mark.parametrize(
    "by",
    [
        [0] * 10 + [1] * 10,
        [0] * 9,
    ],
)
def test_grouped_bad_by(by: list[int]) -> None:
    data = np.zeros((10, 2, 4))
    with pytest.raises(ValueError, match=".*data.shape.*"):
        cmomy.reduce_data_grouped(data, mom_ndim=1, by=by, axis=0)


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
    with pytest.raises(ValueError, match=r".*len.*start.*len.*end.*"):
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


def test_indexed(rng: np.random.Generator) -> None:
    data = rng.random((10, 2, 3))

    by = [0] * 5 + [1] * 5

    a = cmomy.reduce_data_grouped(data, mom_ndim=1, by=by, axis=0)

    _groups, index, start, end = cmomy.reduction.factor_by_to_index(by)

    b = cmomy.reduction.reduce_data_indexed(
        data,
        mom_ndim=1,
        index=index,
        group_start=start,
        group_end=end,
        scale=[1] * 10,
        axis=0,
    )

    np.testing.assert_allclose(a, b)

    # bad scale

    with pytest.raises(ValueError, match=".*len.*scale.*"):
        _ = cmomy.reduction.reduce_data_indexed(
            data,
            mom_ndim=1,
            index=index,
            group_start=start,
            group_end=end,
            scale=[1] * 11,
            axis=0,
        )


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
    ("shape", "axis", "mom"),
    [
        ((2, 3, 4), 0, 3),
        ((2, 3, 4), 1, 3),
        ((2, 3, 4), 2, 3),
    ],
)
def test_reduce_vals_keepdims(
    shape: tuple[int, ...],
    axis: int,
    mom: int,
    rng: np.random.Generator,
    as_dataarray: bool,
) -> None:
    x = rng.random(shape)
    if as_dataarray:
        x = xr.DataArray(x)  # type: ignore[assignment]

    kws = {"mom": mom, "axis": axis}

    check = cmomy.reduce_vals(x, **kws, keepdims=False)  # type: ignore[call-overload]

    new_shape = [*shape, *cmomy.utils.mom_to_mom_shape(mom)]
    new_shape[axis] = 1
    new_shape = tuple(new_shape)  # type: ignore[assignment]

    out = cmomy.reduce_vals(x, **kws, keepdims=True)  # type: ignore[call-overload]
    assert out.shape == new_shape

    np.testing.assert_allclose(np.squeeze(out, axis), check)

    cls = cmomy.xCentralMoments if as_dataarray else cmomy.CentralMoments
    c = cls.from_vals(x, **kws, keepdims=True)  # type: ignore[attr-defined]
    assert c.shape == new_shape

    np.testing.assert_allclose(c, out)


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

    cls = cmomy.xCentralMoments if as_dataarray else cmomy.CentralMoments
    c = cls(x, mom_ndim=mom_ndim).reduce(
        axis=axis, keepdims=True, **({"use_reduce": True} if as_dataarray else {})
    )
    assert c.shape == new_shape
    np.testing.assert_allclose(c, out)


@pytest.mark.parametrize(
    ("shape", "kws"),
    [
        ((10, 3, 4), {"axis": 0, "mom_ndim": 1}),
        ((10, 3, 4), {"axis": (0, 1), "mom_ndim": 1}),
        ((10, 3, 4), {"axis": 0, "mom_ndim": 2}),
    ],
)
def test_reduce_data_use_reduce(rng, shape, kws) -> None:
    data = xr.DataArray(rng.random(shape))
    a = cmomy.reduce_data(data, **kws, use_reduce=True)
    b = cmomy.reduce_data(data, **kws, use_reduce=False)
    xr.testing.assert_allclose(a, b)

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest
import xarray as xr

import cmomy


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
            pytest.raises(ValueError, match=r"Unknown .*"),
        ),
    ],
)
def test_block_by(kwargs, expected) -> None:
    with expected as e:
        by = cmomy.grouped.block_by(**kwargs)
        np.testing.assert_allclose(by, e)


def test__validate_index() -> None:
    index = [0, 1, 2, 3]
    group_start = [0, 2]
    group_end = [2, 4]

    from cmomy.grouped._reduction import _validate_index

    index_, start_, end_ = _validate_index(4, index, group_start, group_end)

    np.testing.assert_allclose(index, index_)
    np.testing.assert_allclose(group_start, start_)
    np.testing.assert_allclose(group_end, end_)

    # index outside bounds
    with pytest.raises(ValueError, match=r".*min.*< 0.*"):
        _ = _validate_index(4, [-1, 0, 1, 2], group_start, group_end)

    # index outside max
    with pytest.raises(ValueError, match=r".*max.*>.*"):
        _ = _validate_index(4, [0, 1, 2, 3, 4], group_start, group_end)

    # mismatch group start/end
    with pytest.raises(ValueError, match=r".*`group_start` and `group_end`.*"):
        _ = _validate_index(4, index, [0, 1], [1, 2, 3])

    # end < start
    with pytest.raises(ValueError, match=r".*end < start.*"):
        _ = _validate_index(4, index, [0, 2], [2, 1])
    # zero length index
    index = []
    group_start = [0]
    group_end = [0]

    index_, start_, end_ = _validate_index(4, index, group_start, group_end)

    assert len(index_) == 0
    np.testing.assert_allclose(group_start, start_)
    np.testing.assert_allclose(group_end, end_)

    # bad end
    with pytest.raises(ValueError, match=r".*With zero length.*"):
        _ = _validate_index(4, index, group_start, [10])


# * grouped -------------------------------------------------------------------
@pytest.mark.parametrize(
    ("shape", "mom_ndim"),
    [
        ((16, 3), 1),
        ((16, 3, 3), 2),
    ],
)
@pytest.mark.parametrize("by", [[0] * 4 + [1] * 4 + [2] * 4 + [3] * 4])
def test_reduce_data_grouped_indexed(rng, shape, mom_ndim, by):
    data = rng.random(shape)
    expected = cmomy.reduce_data(
        data.reshape(4, 4, *shape[1:]), axis=1, mom_ndim=mom_ndim
    )
    check = cmomy.reduce_data_grouped(data, by=by, axis=0, mom_ndim=mom_ndim)
    np.testing.assert_allclose(check, expected)

    index, start, end, _group = cmomy.grouped.factor_by_to_index(by)

    check = cmomy.grouped.reduce_data_indexed(
        data, index=index, group_start=start, group_end=end, axis=0, mom_ndim=mom_ndim
    )
    np.testing.assert_allclose(check, expected)


@pytest.mark.parametrize(
    ("shapex", "shapey", "shapew", "mom"),
    [
        (
            160,
            None,
            None,
            3,
        ),
        (
            160,
            None,
            160,
            3,
        ),
        (
            (160, 3),
            None,
            (160, 3),
            3,
        ),
        (
            160,
            160,
            None,
            (3, 3),
        ),
        (
            (160, 3),
            (160, 3),
            (160, 3),
            (3, 3),
        ),
    ],
)
@pytest.mark.parametrize("by", [[0] * 40 + [1] * 40 + [2] * 40 + [3] * 40])
def test_reduce_vals_grouped(rng, shapex, shapey, shapew, mom, by) -> None:
    x = rng.random(shapex)
    y = (rng.random(shapey),) if shapey else ()
    weight = rng.random(shapew) if shapew else None
    expected = cmomy.reduce_vals(
        x.reshape(4, 40, *x.shape[1:]),
        *(yy.reshape(4, 40, *yy.shape[1:]) for yy in y),
        weight=weight.reshape(4, 40, *weight.shape[1:])
        if weight is not None
        else weight,
        axis=1,
        mom=mom,
    )

    check = cmomy.reduce_vals_grouped(
        x, *y, weight=weight, by=by, mom=mom, axis=0, axes_to_end=False
    )
    np.testing.assert_allclose(expected, check)

    index, start, end, _group = cmomy.grouped.factor_by_to_index(by)

    check = cmomy.grouped.reduce_vals_indexed(
        x,
        *y,
        weight=weight,
        mom=mom,
        axis=0,
        axes_to_end=False,
        index=index,
        group_start=start,
        group_end=end,
    )
    np.testing.assert_allclose(expected, check)


def test_indexed_bad_scale(rng: np.random.Generator) -> None:
    data = rng.random((10, 2, 3))
    by = [0] * 5 + [1] * 5
    index, start, end, _group = cmomy.grouped.factor_by_to_index(by)
    # bad scale
    with pytest.raises(ValueError, match=r".*`scale` and `index`.*"):
        _ = cmomy.grouped.reduce_data_indexed(
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
    with pytest.raises(ValueError, match=r".*Wrong length of `by`.*"):
        cmomy.reduce_data_grouped(data, mom_ndim=1, by=by, axis=0)


@pytest.mark.parametrize(
    ("selected", "template"),
    [
        (
            xr.DataArray([0, 1], dims="a"),
            xr.DataArray(range(6), dims="a", coords={"a": list("abcdef")}),
        ),
    ],
)
@pytest.mark.parametrize(
    ("index", "start", "end", "groups"),
    [
        (np.arange(6), np.array([0, 3]), np.array([3, 6]), np.array(["one", "two"])),
    ],
)
@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("first", ("a", "d")),
        ("last", ("c", "f")),
        ("group", ("one", "two")),
        (None, [0, 1]),
    ],
)
def test__apply_coords_policy_indexed(
    selected, template, index, start, end, groups, policy, expected
) -> None:
    from cmomy.grouped._reduction import _apply_coords_policy_indexed

    out = _apply_coords_policy_indexed(
        selected=selected,
        template=template,
        dim="a",
        coords_policy=policy,
        index=index,
        group_start=start,
        group_end=end,
        groups=groups,
    )

    assert np.all(out.coords["a"].values == np.array(expected))


@pytest.mark.parametrize(
    ("selected", "template"),
    [
        (
            xr.DataArray([0, 1], dims="a"),
            xr.DataArray(range(6), dims="a", coords={"a": list("abcdef")}),
        ),
    ],
)
@pytest.mark.parametrize(
    ("by", "groups"),
    [
        (np.array([0, 0, 0, 1, 1, 1]), np.array(["one", "two"])),
    ],
)
@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("first", ("a", "d")),
        ("last", ("c", "f")),
        ("group", ("one", "two")),
        (None, [0, 1]),
    ],
)
def test__apply_coords_policy_grouped(
    selected, template, by, groups, policy, expected
) -> None:
    from cmomy.grouped._reduction import _apply_coords_policy_grouped

    out = _apply_coords_policy_grouped(
        selected=selected,
        template=template,
        dim="a",
        coords_policy=policy,
        by=by,
        groups=groups,
    )

    assert np.all(out.coords["a"].values == np.array(expected))


@pytest.mark.parametrize("data", [xr.DataArray([1, 2, 3], dims="a")])
@pytest.mark.parametrize(
    "group_dim",
    [None, "b"],
)
def test__optional_group_dim(data, group_dim) -> None:
    from cmomy.grouped._reduction import _optional_group_dim

    out = _optional_group_dim(data, "a", group_dim)

    if group_dim is None:
        assert out.dims == data.dims
    else:
        assert out.dims == (group_dim,)

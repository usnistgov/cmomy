from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cmomy.new import utils

# def test_shape_insert() -> None:
#     assert utils.shape_insert_axis(shape=(1, 2, 3), axis=0, new_size=10) == (
#         10,
#         1,
#         2,
#         3,
#     )

#     assert utils.shape_insert_axis(shape=(1, 2, 3), axis=-1, new_size=10) == (
#         1,
#         2,
#         3,
#         10,
#     )

#     with pytest.raises(ValueError):
#         utils.shape_insert_axis(shape=(1, 2, 3), axis=None, new_size=10)


# def test_axis_expand_broadcast() -> None:
#     with pytest.raises(TypeError):
#         utils.axis_expand_broadcast([1, 2, 3], shape=(3, 10), axis=0, verify=False)

#     x = np.arange(3)

#     with pytest.raises(ValueError):
#         utils.axis_expand_broadcast(x, shape=(3, 2), expand=True, axis=None)

#     with pytest.raises(ValueError):
#         utils.axis_expand_broadcast(x, shape=(4, 2), expand=True, axis=0)

#     expected = np.tile(x, (2, 1)).T
#     np.testing.assert_allclose(
#         utils.axis_expand_broadcast(x, shape=(3, 2), expand=True, axis=0), expected
#     )

#     np.testing.assert_allclose(
#         utils.axis_expand_broadcast(x, shape=(2, 3), axis=1, roll=False), expected.T
#     )
#     np.testing.assert_allclose(
#         utils.axis_expand_broadcast(x, shape=(2, 3), axis=1, roll=True), expected
#     )


# * Moment validation


@pytest.mark.parametrize(("mom_ndim", "expected"), [(0, -1), (1, 1), (2, 2), (3, -1)])
def test_validate_mom_ndim(mom_ndim: int, expected: int) -> None:
    if expected < 0:
        with pytest.raises(ValueError, match=r".* must be either 1 or 2"):
            utils.validate_mom_ndim(mom_ndim)
    else:
        assert utils.validate_mom_ndim(mom_ndim) == expected


@pytest.mark.parametrize(
    ("mom", "expected"),
    [
        (0, "error"),
        ((0,), "error"),
        ((3, 0), "error"),
        ((0, 3), "error"),
        (3, (3,)),
        ((3,), (3,)),
        ([3], (3,)),
        ([3, 3], (3, 3)),
        ([3, 3, 3], "error"),
    ],
)
def test_is_mom_tuple(mom: tuple[int, ...], expected: tuple[int, ...] | str) -> None:
    if expected == "error":
        with pytest.raises(ValueError, match=r".* must be an integer, .*"):
            utils.validate_mom(mom)
    else:
        assert utils.validate_mom(mom) == expected


@pytest.mark.parametrize(
    ("mom", "mom_ndim", "shape", "expected_mom", "expected_mom_ndim"),
    [
        (3, 1, None, (3,), 1),
        ((3,), 1, None, (3,), 1),
        (None, 1, (1, 2, 3), (2,), 1),
        (0, 1, None, "error", "error"),
        (None, 1, (2, 1), "error", "error"),
        (3, 2, None, "error", "error"),
        ((3, 0), 2, None, "error", "error"),
        ((3, 3), 1, None, "error", "error"),
        (None, 2, (1, 2, 3), (1, 2), 2),
        (None, 2, (2, 3), (1, 2), 2),
        (None, 2, (2, 1, 1), "error", "error"),
        (None, 2, (3,), "error", "error"),
    ],
)
def test_validate_mom_and_mom_ndim(
    mom, mom_ndim, shape, expected_mom, expected_mom_ndim
) -> None:
    if expected_mom == "error":
        with pytest.raises(ValueError):
            utils.validate_mom_and_mom_ndim(mom=mom, mom_ndim=mom_ndim, shape=shape)

    else:
        assert utils.validate_mom_and_mom_ndim(
            mom=mom, mom_ndim=mom_ndim, shape=shape
        ) == (expected_mom, expected_mom_ndim)


def test_validate_mom_and_mom_ndim_2() -> None:
    with pytest.raises(ValueError):
        utils.validate_mom_and_mom_ndim(mom=None, mom_ndim=None)
    with pytest.raises(ValueError):
        utils.validate_mom_and_mom_ndim(mom=None, mom_ndim=1)

    with pytest.raises(ValueError):
        utils.validate_mom_and_mom_ndim(mom=None, mom_ndim=2, shape=(2,))

    with pytest.raises(ValueError):
        utils.validate_mom_and_mom_ndim(mom=None, mom_ndim=3, shape=(2, 3, 4))  # type: ignore[arg-type]

    assert utils.validate_mom_and_mom_ndim(mom=(2, 2), mom_ndim=None) == ((2, 2), 2)

    with pytest.raises(ValueError):
        utils.validate_mom_and_mom_ndim(mom=(2, 2, 2), mom_ndim=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        utils.validate_mom_and_mom_ndim(mom=(2, 2), mom_ndim=1)


def test_mom_to_mom_ndim() -> None:
    assert utils.mom_to_mom_ndim(2) == 1
    assert utils.mom_to_mom_ndim((2, 2)) == 2

    with pytest.raises(ValueError):
        utils.mom_to_mom_ndim((2, 2, 2))  # type: ignore[arg-type]

    # this should be fine
    utils.mom_to_mom_ndim([2, 2])  # type: ignore[arg-type]


def test_select_mom_ndim() -> None:
    assert utils.select_mom_ndim(mom=2, mom_ndim=None) == 1
    assert utils.select_mom_ndim(mom=(2, 2), mom_ndim=None) == 2

    with pytest.raises(ValueError):
        utils.select_mom_ndim(mom=(2, 2), mom_ndim=1)

    with pytest.raises(TypeError):
        utils.select_mom_ndim(mom=None, mom_ndim=None)

    assert utils.select_mom_ndim(mom=None, mom_ndim=1) == 1
    with pytest.raises(ValueError):
        utils.select_mom_ndim(mom=None, mom_ndim=3)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("parallel", "size", "cutoff", "expected"),
    [
        (None, 100, 100, False),
        (None, 101, 100, True),
        (True, 100, 100, True),
        (True, 101, 100, True),
        (False, 100, 100, False),
        (False, 101, 100, False),
    ],
)
def test_parallel_heuristic(parallel, size, cutoff, expected) -> None:
    assert utils.parallel_heuristic(parallel, size, cutoff) == expected


# * prepare values/data
dtype_mark = pytest.mark.parametrize("dtype", [np.float32, np.float64, None])
order_mark = pytest.mark.parametrize("order", ["C", None])
prepare_values_mark = pytest.mark.parametrize(
    "axis, xshape, xshape2, yshape, yshape2, wshape, wshape2",
    [
        (0, (10, 2, 3), (2, 3, 10), (), (10,), (), (10,)),
        (0, (10, 2, 3), (2, 3, 10), (10, 2, 1), (2, 1, 10), (1, 10), (1, 10)),
        (1, (2, 10, 3), (2, 3, 10), (3, 10), (3, 10), (10,), (10,)),
        # Note that "wrong" shapes are passed through.
        (1, (2, 10, 3), "error", (10, 3), (10, 3), (10,), (10,)),
        (1, (2, 10, 3), (2, 3, 10), (1, 10, 3), (1, 3, 10), (10,), (10,)),
        # bad shape on w
        (1, (2, 10, 3), "error", (10,), (10,), (11,), (10,)),
        (1, (2, 10, 3), "error", (11,), (10,), (10,), (10,)),
        (2, (2, 3, 10), (2, 3, 10), (3, 10), (3, 10), (10,), (10,)),
        (2, (2, 3, 10), (2, 3, 10), (1, 3, 10), (1, 3, 10), (10,), (10,)),
        (-1, (2, 3, 10), (2, 3, 10), (3, 10), (3, 10), (10,), (10,)),
        (-1, (2, 3, 10), (2, 3, 10), (1, 3, 10), (1, 3, 10), (10,), (10,)),
        # no axis
        (None, (2, 3, 10), "error", (1, 3, 10), (1, 3, 10), (10,), (10,)),
    ],
)


@prepare_values_mark
@dtype_mark
@order_mark
def test_prepare_values_for_reduction(
    dtype, order, axis, xshape, xshape2, yshape, yshape2, wshape, wshape2
) -> None:
    xv, yv, wv = 1, 2, 3

    target = np.full(xshape, dtype=dtype or np.float64, fill_value=xv)
    y = yv if yshape == () else np.full(yshape, fill_value=yv, dtype=int)
    w = wv if wshape == () else np.full(wshape, fill_value=wv, dtype=int)
    with pytest.raises(ValueError, match=r"Number of arrays .*"):
        utils.prepare_values_for_reduction(
            target, y, w, narrays=2, axis=axis, order=order
        )

    # with pytest.raises(ValueError, match=r"Must specify axis.*"):
    #     utils.prepare_values_for_reduction(target, y, w, narrays=3, axis=None, order=order)

    if xshape2 == "error":
        with pytest.raises(ValueError):
            utils.prepare_values_for_reduction(
                target, y, w, narrays=3, axis=axis, order=order
            )

    else:
        x, y, w = utils.prepare_values_for_reduction(
            target, y, w, narrays=3, axis=axis, order=order
        )

        for xx, vv, ss in zip([x, y, w], [xv, yv, wv], [xshape2, yshape2, wshape2]):
            assert xx.shape == ss
            assert xx.dtype == np.dtype(dtype or np.float64)
            np.testing.assert_allclose(xx, vv)

            if order == "C":
                assert xx.flags["C_CONTIGUOUS"]


@dtype_mark
@order_mark
@pytest.mark.parametrize(
    ("axis", "mom_ndim", "shape", "shape2"),
    [
        (0, 1, (10, 2, 3, 4), (2, 3, 10, 4)),
        (1, 1, (2, 10, 3, 4), (2, 3, 10, 4)),
        (2, 1, (2, 3, 10, 4), (2, 3, 10, 4)),
        (-1, 1, (2, 3, 10, 4), (2, 3, 10, 4)),
        (-2, 1, (2, 10, 3, 4), (2, 3, 10, 4)),
        (0, 2, (10, 2, 3, 4), (2, 10, 3, 4)),
        (1, 2, (2, 10, 3, 4), (2, 10, 3, 4)),
        (-1, 2, (2, 10, 3, 4), (2, 10, 3, 4)),
        (-2, 2, (10, 2, 3, 4), (2, 10, 3, 4)),
        (None, 1, (10, 2, 3, 4), "error"),
    ],
)
def test_prepare_data_for_reduction(
    dtype, order, axis, mom_ndim, shape, shape2
) -> None:
    data = np.ones(shape, dtype=dtype)

    if shape2 == "error":
        with pytest.raises(ValueError):
            out = utils.prepare_data_for_reduction(
                data, axis=axis, mom_ndim=mom_ndim, order=order
            )

    else:
        out = utils.prepare_data_for_reduction(
            data, axis=axis, mom_ndim=mom_ndim, order=order
        )

        assert out.shape == shape2

        assert out.dtype == np.dtype(dtype or np.float64)
        if order == "C":
            assert out.flags["C_CONTIGUOUS"]


@dtype_mark
@order_mark
def test_prepare_values_for_push_val(dtype, order) -> None:
    x = np.ones((2, 3, 4), dtype=dtype, order="F")
    w = 1.0

    for arg in utils.prepare_values_for_push_val(x, w, order=order):
        if order == "C":
            assert arg.flags["C_CONTIGUOUS"]
        assert arg.dtype == np.dtype(dtype or np.float64)


# * xarray stuff
# @dtype_mark
# @order_mark
def test_xprepare_values_for_reduction_0():
    target = xr.DataArray(np.ones((2, 3, 4)))
    other = np.full((3, 4), fill_value=2)

    # wrong number of arrays
    with pytest.raises(ValueError):
        utils.xprepare_values_for_reduction(
            target, other, narrays=3, axis=None, dim="rec"
        )

    # no axis or dim
    with pytest.raises(ValueError):
        utils.xprepare_values_for_reduction(
            target, other, narrays=2, axis=None, dim=None
        )

    with pytest.raises(TypeError):
        utils.xprepare_values_for_reduction(other, other, narrays=2, axis=0, dim=None)


@pytest.mark.parametrize(
    ("dim", "xshape", "xshape2", "yshape", "yshape2"),
    [
        ("dim_0", (2, 3, 4), (3, 4, 2), (2,), (2,)),
        ("dim_1", (2, 3, 4), (2, 4, 3), (4, 3), (4, 3)),
        ("dim_2", (2, 3, 4), (2, 3, 4), (4,), (4,)),
        ("dim_0", (2, 3, 4), (3, 4, 2), (2, 3, 4), (3, 4, 2)),
    ],
)
@dtype_mark
@order_mark
def test_xprepare_values_for_reduction_1(
    dtype, order, dim, xshape, xshape2, yshape, yshape2
) -> None:
    target = xr.DataArray(np.ones(xshape, dtype=dtype))
    other = np.ones(yshape, dtype=np.float32)

    core_dims, (x, y) = utils.xprepare_values_for_reduction(
        target, other, narrays=2, axis=None, dim=dim, order=order
    )

    assert core_dims == [[dim]] * 2

    assert x.shape == xshape2
    assert y.shape == yshape2
    assert x.dtype == np.dtype(dtype or np.float64)
    assert y.dtype == np.dtype(dtype or np.float64)

    if order == "C":
        assert x.data.flags["C_CONTIGUOUS"]
        assert y.flags["C_CONTIGUOUS"]

    if xshape == yshape:
        # also do xr test
        other = xr.DataArray(other)
        core_dims, (x, y) = utils.xprepare_values_for_reduction(
            target, other, narrays=2, axis=None, dim=dim, order=order
        )

        assert core_dims == [[dim]] * 2

        assert x.shape == xshape2
        assert y.shape == other.shape
        assert x.dtype == np.dtype(dtype or np.float64)
        assert y.dtype == np.dtype(dtype or np.float64)

        if order == "C":
            assert x.data.flags["C_CONTIGUOUS"]
            assert y.data.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize(
    ("dim_or_axis", "mom_ndim", "shape", "shape2"),
    [
        ("dim_0", 1, (10, 2, 3, 4), (2, 3, 10, 4)),
        ("dim_1", 1, (2, 10, 3, 4), (2, 3, 10, 4)),
        ("dim_2", 1, (2, 3, 10, 4), (2, 3, 10, 4)),
        (-1, 1, (2, 3, 10, 4), (2, 3, 10, 4)),
        ("dim_0", 2, (10, 2, 3, 4), (2, 10, 3, 4)),
        (-1, 2, (2, 10, 3, 4), (2, 10, 3, 4)),
    ],
)
@dtype_mark
@order_mark
def test_xprepare_data_for_reduction_0(
    dtype, order, dim_or_axis, mom_ndim, shape, shape2
):
    data = xr.DataArray(np.ones(shape, dtype=np.float32))

    if isinstance(dim_or_axis, str):
        dim, axis = dim_or_axis, None
    else:
        dim, axis = None, dim_or_axis

    dim, out = utils.xprepare_data_for_reduction(
        data, axis=axis, dim=dim, mom_ndim=mom_ndim, order=order, dtype=dtype
    )
    assert out.shape == shape2
    assert out.dtype == np.dtype(dtype or np.float32)


@pytest.mark.parametrize(
    ("mom_ndim", "mom_dims", "expected"),
    [
        (1, None, ("mom_0",)),
        (2, None, ("mom_0", "mom_1")),
        (1, "a", ("a",)),
        (2, ("a", "b"), ("a", "b")),
        (1, ["a"], ("a",)),
        (2, ["a", "b"], ("a", "b")),
        (1, {"a"}, TypeError),
        (1, ["a", "b"], ValueError),
        (2, "a", ValueError),
        (2, ("a,"), ValueError),
    ],
)
def test_validate_mom_dims(mom_ndim, mom_dims, expected):
    if isinstance(expected, tuple):
        assert utils.validate_mom_dims(mom_dims, mom_ndim) == expected
    else:
        with pytest.raises(expected):
            utils.validate_mom_dims(mom_dims, mom_ndim)


def test_select_axis_dim() -> None:
    dims = ("a", "b", "mom")

    with pytest.raises(ValueError):
        utils.select_axis_dim(dims)

    with pytest.raises(ValueError):
        utils.select_axis_dim(dims, default_axis=0, default_dim="hello")

    with pytest.raises(ValueError):
        utils.select_axis_dim(dims, axis=0, dim="a")

    assert utils.select_axis_dim(dims, default_axis=0) == (0, "a")
    assert utils.select_axis_dim(dims, default_axis=-1) == (-1, "mom")

    assert utils.select_axis_dim(dims, default_dim="a") == (0, "a")
    assert utils.select_axis_dim(dims, default_dim="mom") == (2, "mom")

    with pytest.raises(ValueError):
        utils.select_axis_dim(dims, dim="hello")

    with pytest.raises(ValueError):
        utils.select_axis_dim(dims, axis="a")  # type: ignore[arg-type]


def test_move_mom_dims_to_end() -> None:
    x = xr.DataArray(np.zeros((2, 3, 4)), dims=["a", "b", "c"])

    assert utils.move_mom_dims_to_end(x, mom_dims=None) is x
    assert utils.move_mom_dims_to_end(x, mom_dims="a").dims == ("b", "c", "a")
    assert utils.move_mom_dims_to_end(x, mom_dims="b").dims == ("a", "c", "b")
    assert utils.move_mom_dims_to_end(x, mom_dims=("b", "a")).dims == ("c", "b", "a")

    with pytest.raises(ValueError):
        utils.move_mom_dims_to_end(x, mom_dims="a", mom_ndim=2)


@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize(
    "indexer",
    [
        {"a": 0},
        {"a": slice(1, None)},
        {"a": 0, "b": slice(1, None), "c": slice(2, None)},
    ],
)
def test_replace_coords_from_isel(indexer, drop):
    x_with_coords = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=["a", "b", "c"],
        coords={"a": ("a", [1, 2]), "b": ("b", list("abc")), "c": ("c", [4, 5, 6, 7])},
    )

    x_without = xr.DataArray(np.ones((2, 3, 4)), dims=list("abc"))

    t = utils.replace_coords_from_isel(
        x_with_coords, x_without.isel(indexer), indexer, drop=drop
    )

    xr.testing.assert_identical(t, x_with_coords.isel(indexer, drop=drop) + 1)


def test_raise_if_wrong_shape() -> None:
    x = np.ones((2, 3, 4))

    utils.raise_if_wrong_shape(x, (2, 3, 4))

    with pytest.raises(ValueError):
        utils.raise_if_wrong_shape(x, (1, 2, 3, 4))

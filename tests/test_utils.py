# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cmomy import _utils as utils


# * Order validation
@pytest.mark.parametrize(
    ("order", "expected"),
    [
        ("c", "C"),
        ("F", "F"),
        (None, None),
        ("k", None),
        ("anything", None),
    ],
)
def test_arrayorder_to_arrayorder_cf(order, expected) -> None:
    assert utils.arrayorder_to_arrayorder_cf(order) == expected


# * validate not none
@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (None, "error"),
        ("a", "a"),
        (1, 1),
    ],
)
def test_validate_not_none(x, expected) -> None:
    if expected == "error":
        with pytest.raises(TypeError, match=".*is not supported.*"):
            utils.validate_not_none(x)

    else:
        assert utils.validate_not_none(x) == expected


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
        utils.validate_mom_and_mom_ndim(mom=None, mom_ndim=3, shape=(2, 3, 4))

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
        utils.select_mom_ndim(mom=None, mom_ndim=3)


@pytest.mark.parametrize(
    ("mom", "mom_shape"),
    [
        (1, (2,)),
        ((1,), (2,)),
        ((1, 2), (2, 3)),
    ],
)
def test_mom_to_mom_shape(mom, mom_shape) -> None:
    assert utils.mom_to_mom_shape(mom) == mom_shape
    assert utils.mom_shape_to_mom(mom_shape) == utils.validate_mom(mom)


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
            target,
            y,  # type: ignore[arg-type]
            w,  # type: ignore[arg-type]
            narrays=2,
            axis=axis,
            order=order,
            dtype=dtype,
        )

    if xshape2 == "error":
        error = TypeError if axis is None else ValueError

        with pytest.raises(error):
            utils.prepare_values_for_reduction(
                target,
                y,  # type: ignore[arg-type]
                w,  # type: ignore[arg-type]
                narrays=3,
                axis=axis,
                dtype=dtype,
                order=order,
            )

    else:
        _axis, (x, y, w) = utils.prepare_values_for_reduction(
            target,
            y,  # type: ignore[arg-type]
            w,  # type: ignore[arg-type]
            narrays=3,
            axis=axis,
            order=order,
            dtype=dtype,
        )

        for xx, vv, ss in zip([x, y, w], [xv, yv, wv], [xshape2, yshape2, wshape2]):
            assert xx.shape == ss
            assert xx.dtype == np.dtype(dtype or np.float64)
            np.testing.assert_allclose(xx, vv)

            if order == "C":
                assert xx.flags["C_CONTIGUOUS"]


@dtype_mark
@order_mark
def test_prepare_values_for_push_val(dtype, order) -> None:
    x = np.ones((2, 3, 4), dtype=dtype, order="F")
    w = 1.0

    for arg in utils.prepare_values_for_push_val(x, w, order=order, dtype=dtype):
        if order == "C":
            assert arg.flags["C_CONTIGUOUS"]
        assert arg.dtype == np.dtype(dtype or np.float64)


# * xarray stuff
# @dtype_mark
# @order_mark
def test_xprepare_values_for_reduction_0():
    dtype = None
    target = xr.DataArray(np.ones((2, 3, 4)))
    other = np.full((3, 4), fill_value=2)

    # wrong number of arrays
    with pytest.raises(ValueError, match=".*Number of arrays.*"):
        utils.xprepare_values_for_reduction(
            target,
            other,
            narrays=3,
            axis=None,
            dim="rec",
            dtype=dtype,
        )

    # no axis or dim
    with pytest.raises(ValueError):
        utils.xprepare_values_for_reduction(
            target,
            other,
            narrays=2,
            axis=utils.MISSING,
            dim=utils.MISSING,
            dtype=dtype,
        )

    with pytest.raises(TypeError):
        utils.xprepare_values_for_reduction(
            other,  # type: ignore[arg-type]
            other,
            narrays=2,
            axis=0,
            dim=None,
            dtype=dtype,
        )


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
        target,
        other,
        narrays=2,
        axis=utils.MISSING,
        dim=dim,
        order=order,
        dtype=dtype,
    )

    assert core_dims == [[dim]] * 2

    assert x.shape == xshape2
    assert y.shape == yshape2
    assert x.dtype == np.dtype(dtype or np.float64)
    assert y.dtype == np.dtype(dtype or np.float64)

    if order == "C":
        assert x.data.flags["C_CONTIGUOUS"]  # type: ignore[union-attr]
        assert y.flags["C_CONTIGUOUS"]

    if xshape == yshape:
        # also do xr test
        other = xr.DataArray(other)  # type: ignore[assignment]
        core_dims, (x, y) = utils.xprepare_values_for_reduction(
            target,
            other,
            narrays=2,
            axis=utils.MISSING,
            dim=dim,
            order=order,
            dtype=dtype,
        )

        assert core_dims == [[dim]] * 2

        assert x.shape == xshape2
        assert y.shape == other.shape
        assert x.dtype == np.dtype(dtype or np.float64)
        assert y.dtype == np.dtype(dtype or np.float64)

        if order == "C":
            assert x.data.flags["C_CONTIGUOUS"]  # type: ignore[union-attr]
            assert y.data.flags["C_CONTIGUOUS"]  # type: ignore[union-attr]


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


@pytest.mark.parametrize(
    ("shape", "axis", "out"),
    [
        ((10, 3), 0, (1, 10, 3)),
        ((10, 3), 1, (10, 1, 3)),
        ((2, 3, 4), 0, (1, 2, 3, 4)),
        ((2, 3, 4), 1, (2, 1, 3, 4)),
        ((2, 3, 4), 2, (2, 3, 1, 4)),
    ],
)
def test_optional_keepdims(shape, axis, out) -> None:
    x = np.empty(shape)

    for keepdims in [True, False]:
        assert utils.optional_keepdims(x, axis=axis, keepdims=keepdims).shape == (
            out if keepdims else shape
        )


def test_select_axis_dim() -> None:
    data = xr.DataArray(np.zeros((1, 1, 1)), dims=("a", "b", "mom"))

    with pytest.raises(ValueError):
        utils.select_axis_dim(data)

    with pytest.raises(ValueError):
        utils.select_axis_dim(data, default_axis=0, default_dim="hello")

    with pytest.raises(ValueError):
        utils.select_axis_dim(data, axis=0, dim="a")

    assert utils.select_axis_dim(data, default_axis=0) == (0, "a")
    assert utils.select_axis_dim(data, default_axis=-1) == (2, "mom")

    assert utils.select_axis_dim(data, default_dim="a") == (0, "a")
    assert utils.select_axis_dim(data, default_dim="mom") == (2, "mom")

    assert utils.select_axis_dim(data, axis=-1) == (2, "mom")
    assert utils.select_axis_dim(data, axis=-1, mom_ndim=1) == (1, "b")
    assert utils.select_axis_dim(data, axis=-1, mom_ndim=2) == (0, "a")

    with pytest.raises(ValueError):
        utils.select_axis_dim(data, axis=-1, mom_ndim=3)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        utils.select_axis_dim(data, axis=2, mom_ndim=1)

    with pytest.raises(ValueError):
        utils.select_axis_dim(data, dim="hello")

    with pytest.raises(ValueError):
        utils.select_axis_dim(data, axis="a")  # type: ignore[arg-type]


def test_validate_axis_mult() -> None:
    assert utils.validate_axis_mult(1) == 1
    assert utils.validate_axis_mult((1, 2)) == (1, 2)
    assert utils.validate_axis_mult(None) is None

    with pytest.raises(TypeError):
        utils.validate_axis_mult(utils.MISSING)


def test_select_axis_dim_mult() -> None:
    data = xr.DataArray((np.zeros((1, 1, 1))), dims=("a", "b", "mom"))

    with pytest.raises(ValueError):
        utils.select_axis_dim_mult(data)

    with pytest.raises(ValueError):
        utils.select_axis_dim_mult(data, default_axis=0, default_dim="hello")

    with pytest.raises(ValueError):
        utils.select_axis_dim_mult(data, axis=0, dim="a")

    assert utils.select_axis_dim_mult(data, axis=0) == ((0,), ("a",))
    assert utils.select_axis_dim_mult(data, axis=1) == ((1,), ("b",))
    assert utils.select_axis_dim_mult(data, axis=-1) == ((2,), ("mom",))
    assert utils.select_axis_dim_mult(data, axis=-1, mom_ndim=1) == ((1,), ("b",))

    with pytest.raises(ValueError):
        utils.select_axis_dim_mult(data, axis=2, mom_ndim=1)

    assert utils.select_axis_dim_mult(data, dim="a") == ((0,), ("a",))
    assert utils.select_axis_dim_mult(data, dim="b") == ((1,), ("b",))
    assert utils.select_axis_dim_mult(data, dim="mom") == ((2,), ("mom",))
    with pytest.raises(ValueError):
        utils.select_axis_dim_mult(data, dim="mom", mom_ndim=1)

    assert utils.select_axis_dim_mult(data, axis=(0, 1)) == ((0, 1), ("a", "b"))
    assert utils.select_axis_dim_mult(data, axis=(1, 0)) == ((1, 0), ("b", "a"))
    assert utils.select_axis_dim_mult(data, axis=(0, 2)) == ((0, 2), ("a", "mom"))
    assert utils.select_axis_dim_mult(data, axis=None) == ((0, 1, 2), ("a", "b", "mom"))
    assert utils.select_axis_dim_mult(data, axis=None, mom_ndim=1) == (
        (0, 1),
        ("a", "b"),
    )
    with pytest.raises(ValueError):
        utils.select_axis_dim_mult(data, axis=(0, 2), mom_ndim=1)

    assert utils.select_axis_dim_mult(data, dim=("a", "b")) == ((0, 1), ("a", "b"))
    assert utils.select_axis_dim_mult(data, dim=("b", "a")) == ((1, 0), ("b", "a"))
    assert utils.select_axis_dim_mult(data, dim=("a", "mom")) == ((0, 2), ("a", "mom"))
    assert utils.select_axis_dim_mult(data, dim=None) == ((0, 1, 2), ("a", "b", "mom"))
    assert utils.select_axis_dim_mult(data, dim=None, mom_ndim=1) == (
        (0, 1),
        ("a", "b"),
    )
    with pytest.raises(ValueError):
        utils.select_axis_dim_mult(data, dim=("a", "mom"), mom_ndim=1)

    # infer defaults
    assert utils.select_axis_dim_mult(data, default_axis=(0, 1)) == ((0, 1), ("a", "b"))
    assert utils.select_axis_dim_mult(data, default_dim=None, mom_ndim=1) == (
        (0, 1),
        ("a", "b"),
    )


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

# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cmomy import _utils as utils
from cmomy import _validate as validate
from cmomy._missing import MISSING


# * catch all args only test
def _do_test(func, *args, expected=None, match=None, **kwargs):
    if isinstance(expected, type):
        with pytest.raises(expected, match=match):
            func(*args, **kwargs)
    else:
        assert func(*args, **kwargs) == expected


# * Order validation
@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        ("c", "C"),
        ("F", "F"),
        (None, None),
        ("k", None),
        ("anything", None),
    ],
)
def test_arrayorder_to_arrayorder_cf(arg, expected) -> None:
    _do_test(utils.arrayorder_to_arrayorder_cf, arg, expected=expected)


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        (2, 1),
        ((2, 2), 2),
        ((2, 2, 2), ValueError),
        ([2, 2], 2),
    ],
)
def test_mom_to_mom_ndim(arg, expected) -> None:
    _do_test(utils.mom_to_mom_ndim, arg, expected=expected)


@pytest.mark.parametrize(
    ("kws", "expected"),
    [
        ({"mom": 2, "mom_ndim": None}, 1),
        ({"mom": (2, 2), "mom_ndim": None}, 2),
        ({"mom": (2, 2), "mom_ndim": 1}, ValueError),
        ({"mom": None, "mom_ndim": None}, TypeError),
        ({"mom": None, "mom_ndim": 1}, 1),
        ({"mom": None, "mom_ndim": 3}, ValueError),
    ],
)
def test_select_mom_ndim(kws, expected) -> None:
    _do_test(utils.select_mom_ndim, expected=expected, **kws)


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
    assert utils.mom_shape_to_mom(mom_shape) == validate.validate_mom(mom)


# * prepare values/data
dtype_mark = pytest.mark.parametrize("dtype", [np.float32, np.float64, None])
order_mark = pytest.mark.parametrize("order", ["C", None])
prepare_values_mark = pytest.mark.parametrize(
    "axis, xshape, xshape2, yshape, yshape2, wshape, wshape2",
    [
        (0, (10, 2, 3), (2, 3, 10), (), (10,), (), (10,)),
        (0, (10, 2, 3), (2, 3, 10), (10, 2, 1), (2, 1, 10), (10, 1, 1), (1, 1, 10)),
        (1, (2, 10, 3), (2, 3, 10), (10, 3), (3, 10), (10,), (10,)),
        # Note that "wrong" shapes are passed through.
        (1, (2, 10, 3), "error", (3, 10), (3, 10), (10,), (10,)),
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
# @order_mark
def test_prepare_values_for_reduction(
    dtype, axis, xshape, xshape2, yshape, yshape2, wshape, wshape2
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
            )

    else:
        _axis, (x, y, w) = utils.prepare_values_for_reduction(
            target,
            y,  # type: ignore[arg-type]
            w,  # type: ignore[arg-type]
            narrays=3,
            axis=axis,
            dtype=dtype,
        )

        for xx, vv, ss in zip([x, y, w], [xv, yv, wv], [xshape2, yshape2, wshape2]):
            assert xx.shape == ss
            assert xx.dtype == np.dtype(dtype or np.float64)
            np.testing.assert_allclose(xx, vv)


# * xarray stuff
@pytest.mark.parametrize(
    ("target", "other"),
    [
        (xr.DataArray(np.ones((2, 3, 4))), np.full((3, 4), fill_value=2)),
    ],
)
@pytest.mark.parametrize(
    ("kws", "raises", "match"),
    [
        (
            {"narrays": 3, "axis": None, "dim": "rec", "dtype": np.float32},
            ValueError,
            ".*Number of arrays.*",
        ),
        (
            {
                "narrays": 2,
                "axis": MISSING,
                "dim": MISSING,
                "dtype": np.float32,
            },
            ValueError,
            None,
        ),
        ({"narrays": 2, "axis": 0, "dim": None, "dtype": np.float32}, TypeError, None),
    ],
)
def test_xprepare_values_for_reduction_0(target, other, kws, raises, match):
    with pytest.raises(raises, match=match):
        utils.xprepare_values_for_reduction(target, other, **kws)


@pytest.mark.parametrize(
    ("dim", "xshape", "xshape2", "yshape", "yshape2"),
    [
        ("dim_0", (2, 3, 4), (2, 3, 4), (2,), (2,)),
        ("dim_1", (2, 3, 4), (2, 3, 4), (3, 4), (4, 3)),
        ("dim_2", (2, 3, 4), (2, 3, 4), (4,), (4,)),
        ("dim_0", (2, 3, 4), (2, 3, 4), (2, 3, 4), (3, 4, 2)),
    ],
)
@dtype_mark
def test_xprepare_values_for_reduction_1(
    dtype, dim, xshape, xshape2, yshape, yshape2
) -> None:
    target = xr.DataArray(np.ones(xshape, dtype=dtype))
    other = np.ones(yshape, dtype=np.float32)

    core_dims, (x, y) = utils.xprepare_values_for_reduction(
        target,
        other,
        narrays=2,
        axis=MISSING,
        dim=dim,
        dtype=dtype,
    )

    assert core_dims == [[dim]] * 2

    assert x.shape == xshape2
    assert y.shape == yshape2
    assert x.dtype == np.dtype(dtype or np.float64)
    assert y.dtype == np.dtype(dtype or np.float64)

    if xshape == yshape:
        # also do xr test
        other = xr.DataArray(other)  # type: ignore[assignment]
        core_dims, (x, y) = utils.xprepare_values_for_reduction(
            target,
            other,
            narrays=2,
            axis=MISSING,
            dim=dim,
            dtype=dtype,
        )

        assert core_dims == [[dim]] * 2

        assert x.shape == xshape2
        assert y.shape == other.shape
        assert x.dtype == np.dtype(dtype or np.float64)
        assert y.dtype == np.dtype(dtype or np.float64)


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


@pytest.mark.parametrize(
    "data", [xr.DataArray(np.zeros((1, 1, 1)), dims=("a", "b", "mom"))]
)
@pytest.mark.parametrize(
    ("kws", "expected"),
    [
        ({}, ValueError),
        ({"default_axis": 0, "default_dim": "hello"}, ValueError),
        ({"axis": 0, "dim": "a"}, ValueError),
        ({"default_axis": 0}, (0, "a")),
        ({"default_axis": -1}, (2, "mom")),
        ({"default_dim": "a"}, (0, "a")),
        ({"default_dim": "mom"}, (2, "mom")),
        ({"axis": -1}, (2, "mom")),
        ({"axis": -1, "mom_ndim": 1}, (1, "b")),
        ({"axis": -1, "mom_ndim": 2}, (0, "a")),
        ({"axis": -1, "mom_ndim": 3}, ValueError),
        ({"axis": 2, "mom_ndim": 1}, ValueError),
        ({"dim": "hello"}, ValueError),
        ({"axis": "a"}, ValueError),
    ],
)
def test_select_axis_dim(data, kws, expected) -> None:
    _do_test(utils.select_axis_dim, data, expected=expected, **kws)


@pytest.mark.parametrize(
    "data", [xr.DataArray(np.zeros((1, 1, 1)), dims=("a", "b", "mom"))]
)
@pytest.mark.parametrize(
    ("kws", "expected"),
    [
        # errors
        ({}, ValueError),
        ({"default_axis": 0, "default_dim": "hello"}, ValueError),
        ({"axis": 0, "dim": "a"}, ValueError),
        ({"axis": 2, "mom_ndim": 1}, ValueError),
        ({"dim": "mom", "mom_ndim": 1}, ValueError),
        ({"axis": (0, 2), "mom_ndim": 1}, ValueError),
        ({"dim": ("a", "mom"), "mom_ndim": 1}, ValueError),
        # other
        ({"axis": 0}, ((0,), ("a",))),
        ({"axis": 1}, ((1,), ("b",))),
        ({"axis": -1}, ((2,), ("mom",))),
        ({"axis": -1, "mom_ndim": 1}, ((1,), ("b",))),
        ({"dim": "a"}, ((0,), ("a",))),
        ({"dim": "b"}, ((1,), ("b",))),
        ({"dim": "mom"}, ((2,), ("mom",))),
        ({"axis": (0, 1)}, ((0, 1), ("a", "b"))),
        ({"axis": (1, 0)}, ((1, 0), ("b", "a"))),
        ({"axis": None}, ((0, 1, 2), ("a", "b", "mom"))),
        ({"axis": None, "mom_ndim": 1}, ((0, 1), ("a", "b"))),
        ({"dim": ("a", "b")}, ((0, 1), ("a", "b"))),
        ({"dim": ("b", "a")}, ((1, 0), ("b", "a"))),
        ({"dim": ("a", "mom")}, ((0, 2), ("a", "mom"))),
        ({"dim": None}, ((0, 1, 2), ("a", "b", "mom"))),
        ({"dim": None, "mom_ndim": 1}, ((0, 1), ("a", "b"))),
        ({"default_axis": (0, 1)}, ((0, 1), ("a", "b"))),
        ({"default_dim": None, "mom_ndim": 1}, ((0, 1), ("a", "b"))),
    ],
)
def test_select_axis_dim_mult(data, kws, expected) -> None:
    _do_test(utils.select_axis_dim_mult, data, expected=expected, **kws)


@pytest.mark.parametrize("x", [xr.DataArray(np.zeros((2, 3, 4)), dims=["a", "b", "c"])])
@pytest.mark.parametrize(
    ("kws", "expected"),
    [
        ({"mom_dims": None}, ("a", "b", "c")),
        ({"mom_dims": "a"}, ("b", "c", "a")),
        ({"mom_dims": "b"}, ("a", "c", "b")),
        ({"mom_dims": ("b", "a")}, ("c", "b", "a")),
        ({"mom_dims": "a", "mom_ndim": 2}, ValueError),
    ],
)
def test_move_mom_dims_to_end(x, kws, expected) -> None:
    if isinstance(expected, type):
        with pytest.raises(expected):
            utils.move_mom_dims_to_end(x, **kws)
    else:
        assert utils.move_mom_dims_to_end(x, **kws).dims == expected


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

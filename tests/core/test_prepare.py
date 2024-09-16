# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cmomy.core import prepare
from cmomy.core.missing import MISSING

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
        prepare.prepare_values_for_reduction(
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
            prepare.prepare_values_for_reduction(
                target,
                y,  # type: ignore[arg-type]
                w,  # type: ignore[arg-type]
                narrays=3,
                axis=axis,
                dtype=dtype,
            )

    else:
        _axis, (x, y, w) = prepare.prepare_values_for_reduction(
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
        prepare.xprepare_values_for_reduction(target, other, **kws)


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

    dim_out, core_dims, (x, y) = prepare.xprepare_values_for_reduction(
        target,
        other,
        narrays=2,
        axis=MISSING,
        dim=dim,
        dtype=dtype,
    )

    assert dim_out == dim
    assert core_dims == [[dim]] * 2

    assert x.shape == xshape2
    assert y.shape == yshape2
    assert x.dtype == np.dtype(dtype or target.dtype)
    assert y.dtype == np.dtype(dtype or other.dtype)

    if xshape == yshape:
        # also do xr test
        other = xr.DataArray(other)  # type: ignore[assignment]
        dim_out, core_dims, (x, y) = prepare.xprepare_values_for_reduction(
            target,
            other,
            narrays=2,
            axis=MISSING,
            dim=dim,
            dtype=dtype,
        )

        assert dim_out == dim
        assert core_dims == [[dim]] * 2

        assert x.shape == xshape2
        assert y.shape == other.shape
        assert x.dtype == np.dtype(dtype or target.dtype)
        assert y.dtype == np.dtype(dtype or other.dtype)


@pytest.mark.parametrize(
    ("kws", "expected"),
    [
        (
            {
                "out": None,
                "mom_ndim": 1,
                "axis": 0,
                "move_axis_to_end": False,
            },
            None,
        ),
        (
            {
                "out": np.zeros((2, 3, 4)),
                "mom_ndim": 1,
                "axis": 0,
                "move_axis_to_end": False,
            },
            np.zeros((3, 2, 4)),
        ),
        (
            {
                "out": np.zeros((2, 3, 4)),
                "mom_ndim": 1,
                "axis": 0,
                "move_axis_to_end": True,
            },
            np.zeros((2, 3, 4)),
        ),
        (
            {
                "out": np.zeros((2, 3, 4)),
                "mom_ndim": 1,
                "axis": 0,
                "move_axis_to_end": False,
                "data": np.zeros((2, 3, 4)),
            },
            np.zeros((3, 2, 4)),
        ),
        # Silently ignore passing out value for dataset output...
        (
            {
                "out": np.zeros((2, 3, 4)),
                "mom_ndim": 1,
                "axis": 0,
                "move_axis_to_end": False,
                "data": xr.Dataset({"data0": xr.DataArray(np.zeros((2, 3, 4)))}),
            },
            None,
        ),
    ],
)
def test_xprepare_out_for_resample_data(kws, expected) -> None:
    func = prepare.xprepare_out_for_resample_data
    if expected is None:
        assert func(**kws) is None
    elif isinstance(expected, type):
        with pytest.raises(expected):
            func(**kws)
    else:
        np.testing.assert_allclose(func(**kws), expected)  # type: ignore[arg-type]

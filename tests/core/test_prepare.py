# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cmomy import moveaxis
from cmomy.core import prepare
from cmomy.core.missing import MISSING

# * prepare values/data
dtype_mark = pytest.mark.parametrize("dtype", [np.float32, np.float64, None])
order_mark = pytest.mark.parametrize("order", ["C", None])


@pytest.mark.parametrize(
    (
        "axes_to_end",
        "axis",
        "xshape",
        "xshape2",
        "yshape",
        "yshape2",
        "wshape",
        "wshape2",
    ),
    [
        # axes_to_end
        (True, 0, (10, 2, 3), (2, 3, 10), (), (10,), (), (10,)),
        (
            True,
            0,
            (10, 2, 3),
            (2, 3, 10),
            (10, 2, 1),
            (2, 1, 10),
            (10, 1, 1),
            (1, 1, 10),
        ),
        (True, 1, (2, 10, 3), (2, 3, 10), (10, 3), (3, 10), (10,), (10,)),
        # Note that "wrong" shapes are passed through.
        (True, 1, (2, 10, 3), "error", (3, 10), (3, 10), (10,), (10,)),
        (True, 1, (2, 10, 3), (2, 3, 10), (1, 10, 3), (1, 3, 10), (10,), (10,)),
        # bad shape on w
        (True, 1, (2, 10, 3), "error", (10,), (10,), (11,), (10,)),
        (True, 1, (2, 10, 3), "error", (11,), (10,), (10,), (10,)),
        (True, 2, (2, 3, 10), (2, 3, 10), (3, 10), (3, 10), (10,), (10,)),
        (True, 2, (2, 3, 10), (2, 3, 10), (1, 3, 10), (1, 3, 10), (10,), (10,)),
        (True, -1, (2, 3, 10), (2, 3, 10), (3, 10), (3, 10), (10,), (10,)),
        (True, -1, (2, 3, 10), (2, 3, 10), (1, 3, 10), (1, 3, 10), (10,), (10,)),
        # no axis
        (True, None, (2, 3, 10), "error", (1, 3, 10), (1, 3, 10), (10,), (10,)),
        #
        # no axes_to_end
        #
        (False, 0, (10, 2, 3), (10, 2, 3), (), (10,), (), (10,)),
        (
            False,
            0,
            (10, 2, 3),
            (10, 2, 3),
            (10, 2, 1),
            (10, 2, 1),
            (10, 1, 1),
            (10, 1, 1),
        ),
        (False, 1, (2, 10, 3), (2, 10, 3), (10, 3), (10, 3), (10,), (10,)),
        # Note that "wrong" shapes are passed through.
        (False, 1, (2, 10, 3), "error", (3, 10), (3, 10), (10,), (10,)),
        (False, 1, (2, 10, 3), (2, 10, 3), (1, 10, 3), (1, 10, 3), (10,), (10,)),
        # bad shape on w
        (False, 1, (2, 10, 3), "error", (10,), (10,), (11,), (10,)),
        (False, 1, (2, 10, 3), "error", (11,), (10,), (10,), (10,)),
        (False, 2, (2, 3, 10), (2, 3, 10), (3, 10), (3, 10), (10,), (10,)),
        (False, 2, (2, 3, 10), (2, 3, 10), (1, 3, 10), (1, 3, 10), (10,), (10,)),
        (False, -1, (2, 3, 10), (2, 3, 10), (3, 10), (3, 10), (10,), (10,)),
        (False, -1, (2, 3, 10), (2, 3, 10), (1, 3, 10), (1, 3, 10), (10,), (10,)),
        # no axis
        (False, None, (2, 3, 10), "error", (1, 3, 10), (1, 3, 10), (10,), (10,)),
    ],
)
@dtype_mark
# @order_mark
def test_prepare_values_for_reduction(
    axes_to_end, dtype, axis, xshape, xshape2, yshape, yshape2, wshape, wshape2
) -> None:
    xv, yv, wv = 1, 2, 3

    prep = prepare.PrepareValsArray.factory(ndim=2, recast=True)

    target = np.full(xshape, dtype=dtype or np.float64, fill_value=xv)
    y = yv if yshape == () else np.full(yshape, fill_value=yv, dtype=int)
    w = wv if wshape == () else np.full(wshape, fill_value=wv, dtype=int)
    with pytest.raises(ValueError, match=r"Number of arrays .*"):
        prep.values_for_reduction(
            target,
            y,
            w,
            narrays=2,
            axis=axis,
            dtype=dtype,
            axes_to_end=axes_to_end,
        )

    if xshape2 == "error":
        error = TypeError if axis is None else ValueError

        with pytest.raises(error):
            prep.values_for_reduction(
                target,
                y,
                w,
                narrays=3,
                axis=axis,
                dtype=dtype,
                axes_to_end=axes_to_end,
            )

    else:
        prep, _axis, (x, y, w) = prep.values_for_reduction(
            target,
            y,
            w,
            narrays=3,
            axis=axis,
            dtype=dtype,
            axes_to_end=axes_to_end,
        )

        for xx, vv, ss in zip([x, y, w], [xv, yv, wv], [xshape2, yshape2, wshape2]):
            assert xx.shape == ss
            assert xx.dtype == np.dtype(dtype or np.float64)
            np.testing.assert_allclose(xx, vv)


@pytest.mark.parametrize(
    ("mom_axes", "args", "axis_sample_out"),
    [
        # axis_neg, axis, axis_new_size, out_ndim
        ((-2, -1), (-10, None, None, 5), -12),
        ((-2, -1), (-1, 2, None, 5), -3),
        ((1, 2), (-2, 1, 10, 5), 3),
        ((1, 2), (-2, None, 10, 5), 3),
        ((1, 3), (-2, 1, 10, 5), 2),
        ((1, 3), (-2, 1, None, 5), -4),
    ],
)
def test_get_axis_sample_out(mom_axes, args, axis_sample_out) -> None:
    prep = prepare.PrepareValsArray.factory(axes=mom_axes)
    assert prep.get_axis_sample_out(*args) == axis_sample_out


@dtype_mark
@order_mark
@pytest.mark.parametrize(
    (
        "mom",
        "shapes",
        "axis_neg",
        "axis_new_size",
        "mom_axes",
        "out_shape",
        "axis_sample",
    ),
    [
        ((3,), [(10, 2, 3), (10,)], -3, None, None, (2, 3, 4), None),
        ((3,), [(10, 2, 3), (10,)], -3, 5, None, (5, 2, 3, 4), 0),
        ((3,), [(10, 2, 3), (10,)], -3, None, (1,), (2, 4, 3), None),
        ((3,), [(10, 2, 3), (10,)], -3, 5, (1,), (5, 4, 2, 3), 0),
        ((3,), [(2, 10, 3), (10,)], -2, None, None, (2, 3, 4), None),
        ((3,), [(2, 10, 3), (10,)], -2, 5, None, (2, 5, 3, 4), 1),
        ((3,), [(2, 10, 3), (10,)], -2, None, (-2,), (2, 4, 3), None),
        ((3,), [(2, 10, 3), (10,)], -2, 5, (-2,), (2, 5, 4, 3), 1),
        ((3,), [(2, 10, 3), (10,)], -2, None, (1,), (2, 4, 3), None),
        ((3,), [(2, 10, 3), (10,)], -2, 5, (1,), (2, 4, 5, 3), 2),
        ((3,), [(2, 10, 3), (10,)], -2, None, (0,), (4, 2, 3), None),
        ((3,), [(2, 10, 3), (10,)], -2, 5, (0,), (4, 2, 5, 3), 2),
        # more
        ((3, 4), [(10, 2, 3), (10,), (10,)], -3, None, None, (2, 3, 4, 5), None),
        ((3, 4), [(10, 2, 3), (10,), (10,)], -3, 1, None, (1, 2, 3, 4, 5), None),
        ((3, 4), [(10, 2, 3), (10,), (10,)], -3, None, (1, 2), (2, 4, 5, 3), None),
        ((3, 4), [(10, 2, 3), (10,), (10,)], -3, 1, (1, 2), (1, 4, 5, 2, 3), None),
        ((3, 4), [(2, 10, 3), (10,), (10,)], -2, None, None, (2, 3, 4, 5), None),
        ((3, 4), [(2, 10, 3), (10,), (10,)], -2, 1, None, (2, 1, 3, 4, 5), None),
        ((3, 4), [(2, 10, 3), (10,), (10,)], -2, None, (1, 2), (2, 4, 5, 3), None),
        ((3, 4), [(2, 10, 3), (10,), (10,)], -2, 1, (1, 2), (2, 4, 5, 1, 3), 3),
        ((3, 4), [(2, 10, 3), (10,), (10,)], -2, None, (1, 3), (2, 4, 3, 5), None),
        ((3, 4), [(2, 10, 3), (10,), (10,)], -2, 1, (1, 3), (2, 4, 1, 5, 3), 2),
        (
            (3, 4),
            [(2, 10, 3), (1, 1, 10, 3), (10,)],
            -2,
            None,
            None,
            (1, 2, 3, 4, 5),
            None,
        ),
    ],
)
def test_prepare_out_from_values(
    dtype, order, mom, shapes, axis_neg, axis_new_size, mom_axes, out_shape, axis_sample
) -> None:
    ndim = len(mom)
    prep = prepare.PrepareValsArray.factory(ndim=ndim, axes=mom_axes, recast=False)

    out, axis_sample_out = prep.out_from_values(
        None,
        *(np.zeros(shape) for shape in shapes),
        mom=mom,
        axis_neg=axis_neg,
        axis_new_size=axis_new_size,
        dtype=dtype,
        order=order,
    )

    assert out.shape == out_shape

    if order == "C":
        check = out
    elif axis_new_size is None:
        check = moveaxis(out, mom_params=prep.mom_params, axes_to_end=True)
    else:
        if axis_sample is None:
            axis_sample = prep.mom_params.normalize_axis_index(
                axis_neg - ndim, out.ndim
            )
        axis_sample_out = prep.mom_params.normalize_axis_index(
            axis_sample_out, out.ndim
        )
        assert axis_sample_out == axis_sample

        check = moveaxis(
            out, axis_sample, -(ndim + 1), mom_params=prep.mom_params, axes_to_end=True
        )

    assert check.flags["C_CONTIGUOUS"]


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
    prep = prepare.PrepareValsXArray.factory(ndim=2)
    with pytest.raises(raises, match=match):
        prep.values_for_reduction(target, other, **kws)


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
    prep = prepare.PrepareValsXArray.factory(ndim=2, recast=True)

    dim_out, core_dims, (x, y) = prep.values_for_reduction(
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
        other = xr.DataArray(other)  # pylint: disable=redefined-variable-type
        dim_out, core_dims, (x, y) = prep.values_for_reduction(
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
    "data",
    [xr.DataArray(np.zeros((2, 3, 4, 5)))],
)
@pytest.mark.parametrize(
    ("kws", "mom_params_kws", "expected"),
    [
        (
            {
                "out": None,
                "axis": 0,
                "axes_to_end": False,
            },
            {
                "ndim": 1,
            },
            None,
        ),
        (
            {
                "out": None,
                "axis": 0,
                "axes_to_end": False,
                "order": "c",
            },
            {
                "ndim": 1,
            },
            (3, 4, 2, 5),
        ),
        (
            {
                "out": None,
                "axis": 0,
                "axes_to_end": True,
                "order": "c",
            },
            {
                "ndim": 1,
            },
            None,
        ),
        (
            {
                "out": None,
                "axis": 0,
                "axes_to_end": False,
                "order": "c",
                "axis_new_size": 10,
            },
            {
                "ndim": 1,
            },
            (3, 4, 10, 5),
        ),
        (
            {
                "out": np.zeros((2, 3, 4, 5)),
                "axis": 0,
                "axes_to_end": False,
            },
            {
                "ndim": 1,
            },
            (3, 4, 2, 5),
        ),
        (
            {
                "out": np.zeros((2, 3, 4, 5)),
                "axis": 1,
                "axes_to_end": False,
            },
            {
                "axes": (0, 2),
            },
            (5, 3, 2, 4),
        ),
        (
            {
                "out": np.zeros((2, 3, 4)),
                "axis": 0,
                "axes_to_end": True,
            },
            {
                "ndim": 1,
            },
            (2, 3, 4),
        ),
        # Silently ignore passing out value for dataset output...
        (
            {
                "out": np.zeros((2, 3, 4)),
                "axis": 0,
                "axes_to_end": False,
                "data": xr.Dataset({"data0": xr.DataArray(np.zeros((2, 3, 4)))}),
            },
            {
                "ndim": 1,
            },
            None,
        ),
    ],
)
def test_xprepare_out_for_resample_data(data, kws, mom_params_kws, expected) -> None:
    from cmomy.core.moment_params import factory_mom_params

    kws = kws.copy()
    kws.setdefault("data", data)
    kws.setdefault("dtype", np.float64)
    kws.setdefault("order", None)
    mom_params = factory_mom_params(kws["data"], mom_params=mom_params_kws, data=data)

    func = prepare.PrepareDataXArray(mom_params=mom_params).optional_out_sample

    if isinstance(expected, type):
        with pytest.raises(expected):
            func(**kws)
    else:
        out = func(**kws)
        if out is None:
            assert expected is None
        else:
            assert out.shape == expected


def test__prepare_secondary_value_for_reduction() -> None:
    ds = xr.Dataset({"a": xr.DataArray([1, 2, 3], dims="x")})

    with pytest.raises(TypeError, match=r"Passed Dataset.*"):
        _ = prepare._prepare_array_secondary_value_for_reduction(
            ds,
            axis=1,
            axes_to_end=False,
            recast=False,
            nsamp=10,
            dtype=np.float64,
        )

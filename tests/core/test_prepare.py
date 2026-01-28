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


@pytest.mark.parametrize("data", [np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)])
@pytest.mark.parametrize(
    ("mom_axes", "axis"),
    [
        (1, (-4, -2)),
        (2, (0, 1)),
        (-1, (-2,)),
        ((1, 2), (0, -1)),
        ((0, -1), (1, 2)),
        ((-2, -1), (-4, -3)),
    ],
)
@pytest.mark.parametrize("axes_to_end", [True, False])
def test_prepare_data_data_for_reduction_multiple(
    mom_axes, axis, axes_to_end, data
) -> None:
    prep = prepare.PrepareDataArray.factory(axes=mom_axes)

    prep_check, axis_check, out = prep.data_for_reduction_multiple(
        data, axis=axis, axes_to_end=axes_to_end, dtype=None
    )

    if axes_to_end:
        assert prep_check.mom_params.axes == prep.mom_params.axes_last
        assert axis_check == tuple(
            range(
                data.ndim - len(axis) - prep.mom_params.ndim,
                data.ndim - prep.mom_params.ndim,
            )
        )
        np.testing.assert_allclose(
            moveaxis(data, axis, mom_params=prep.mom_params, axes_to_end=True), out
        )
    else:
        assert prep_check.mom_params.axes == prep.mom_params.axes
        assert axis_check == prep.mom_params.normalize_axis_tuple(axis, data.ndim)
        np.testing.assert_allclose(data, out)


@pytest.mark.parametrize(
    "data",
    [np.empty((2, 3, 4, 5))],
)
@pytest.mark.parametrize(
    ("out", "kws", "expected"),
    [
        (np.zeros((2, 3)), {}, (2, 3)),
        (None, {"order": None}, None),
        (None, {"order": "c", "axis_new_size": 10, "axis": 1}, (2, 10, 4, 5)),
    ],
)
def test_prepare_data_optional_out_sample(data, out, kws, expected) -> None:
    kwargs = kws.copy()
    for k, v in {
        "axis": 0,
        "axis_new_size": None,
        "order": None,
        "dtype": np.float64,
    }.items():
        kwargs.setdefault(k, v)

    out = prepare.PrepareDataArray.optional_out_sample(out, data=data, **kwargs)

    if out is None:
        assert expected is None
    else:
        assert out.shape == expected


@pytest.mark.parametrize(
    "data",
    [np.empty((2, 3, 4, 5))],
)
@pytest.mark.parametrize(
    ("mom_axes", "out", "kws", "expected"),
    [
        (-1, np.zeros((2, 3)), {}, (2, 3)),
        (-1, None, {"axis": 1}, (2, 3, 4, 5)),
        (-1, None, {"order": "c", "axis_new_size": 10, "axis": 1}, (2, 10, 4, 5)),
        (-1, None, {"order": None, "axis_new_size": 10, "axis": 1}, (2, 10, 4, 5)),
        ((0, 1), None, {"order": "c", "axis_new_size": 10, "axis": 3}, (2, 3, 4, 10)),
        ((0, 1), None, {"order": None, "axis_new_size": 10, "axis": 3}, (2, 3, 4, 10)),
    ],
)
def test_prepare_data_out_sample(data, mom_axes, out, kws, expected) -> None:
    kwargs = kws.copy()
    for k, v in {
        "axis": 0,
        "axis_new_size": None,
        "order": None,
        "dtype": np.float64,
    }.items():
        kwargs.setdefault(k, v)

    prep = prepare.PrepareDataArray.factory(axes=mom_axes)

    out = prep.out_sample(out, data=data, **kwargs)

    if out is None:
        assert expected is None
    else:
        assert out.shape == expected

        if kwargs["order"] is None:
            assert moveaxis(
                out, kwargs["axis"], mom_params=prep.mom_params, axes_to_end=True
            ).flags.c_contiguous
        elif kwargs["order"] == "c":
            assert out.flags.c_contiguous


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
        # axes_to_end ---------------------------------------------------------
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
        # no axes_to_end ------------------------------------------------------
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
def test_prepare_values_values_for_reduction(
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

        for xx, vv, ss in zip(
            [x, y, w], [xv, yv, wv], [xshape2, yshape2, wshape2], strict=True
        ):
            assert xx.shape == ss
            assert xx.dtype == np.dtype(dtype or np.float64)
            np.testing.assert_allclose(xx, vv)


@pytest.mark.parametrize(
    ("shapes", "expected"),
    [
        ([(10, 2, 3)], (10, 2, 3)),
        ([(10, 2, 3), (10,)], (10, 2, 3)),
        ([(10, 2, 3), (4, 5, 10, 1, 1), (10,)], (4, 5, 10, 2, 3)),
    ],
)
def test_prepare_values_get_axis_sample_out(shapes, expected) -> None:
    assert (
        prepare.PrepareValsArray.get_val_shape(*(np.empty(shape) for shape in shapes))
        == expected
    )


@pytest.mark.parametrize(
    ("mom_axes", "args", "axis_sample_out"),
    [
        # axis_neg, axis, axis_new_size, out_ndim
        ((-2, -1), (-10, None, MISSING, 5), -12),
        ((-2, -1), (-1, 2, MISSING, 5), -3),
        ((1, 2), (-2, 1, 10, 5), 3),
        ((1, 2), (-2, None, 10, 5), 3),
        ((1, 3), (-2, 1, 10, 5), 2),
        ((1, 3), (-2, 1, MISSING, 5), -4),
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
        "val_shape",
        "axis_neg",
        "axis_new_size",
        "mom_axes",
        "out_shape",
        "axis_sample",
    ),
    [
        ((3,), (10, 2, 3), -3, MISSING, None, (2, 3, 4), None),
        ((3,), (10, 2, 3), -3, 5, None, (5, 2, 3, 4), 0),
        ((3,), (10, 2, 3), -3, MISSING, (1,), (2, 4, 3), None),
        ((3,), (10, 2, 3), -3, 5, (1,), (5, 4, 2, 3), 0),
        ((3,), (2, 10, 3), -2, MISSING, None, (2, 3, 4), None),
        ((3,), (2, 10, 3), -2, 5, None, (2, 5, 3, 4), 1),
        ((3,), (2, 10, 3), -2, MISSING, (-2,), (2, 4, 3), None),
        ((3,), (2, 10, 3), -2, 5, (-2,), (2, 5, 4, 3), 1),
        ((3,), (2, 10, 3), -2, MISSING, (1,), (2, 4, 3), None),
        ((3,), (2, 10, 3), -2, 5, (1,), (2, 4, 5, 3), 2),
        ((3,), (2, 10, 3), -2, MISSING, (0,), (4, 2, 3), None),
        ((3,), (2, 10, 3), -2, 5, (0,), (4, 2, 5, 3), 2),
        # more
        ((3, 4), (10, 2, 3), -3, MISSING, None, (2, 3, 4, 5), None),
        ((3, 4), (10, 2, 3), -3, 1, None, (1, 2, 3, 4, 5), None),
        ((3, 4), (10, 2, 3), -3, MISSING, (1, 2), (2, 4, 5, 3), None),
        ((3, 4), (10, 2, 3), -3, 1, (1, 2), (1, 4, 5, 2, 3), None),
        ((3, 4), (2, 10, 3), -2, MISSING, None, (2, 3, 4, 5), None),
        ((3, 4), (2, 10, 3), -2, 1, None, (2, 1, 3, 4, 5), None),
        ((3, 4), (2, 10, 3), -2, MISSING, (1, 2), (2, 4, 5, 3), None),
        ((3, 4), (2, 10, 3), -2, 1, (1, 2), (2, 4, 5, 1, 3), 3),
        ((3, 4), (2, 10, 3), -2, MISSING, (1, 3), (2, 4, 3, 5), None),
        ((3, 4), (2, 10, 3), -2, 1, (1, 3), (2, 4, 1, 5, 3), 2),
        (
            (3, 4),
            (1, 2, 10, 3),
            -2,
            MISSING,
            None,
            (1, 2, 3, 4, 5),
            None,
        ),
    ],
)
def test_prepare_values_out_from_values(
    dtype,
    order,
    mom,
    val_shape,
    axis_neg,
    axis_new_size,
    mom_axes,
    out_shape,
    axis_sample,
) -> None:
    ndim = len(mom)
    prep = prepare.PrepareValsArray.factory(ndim=ndim, axes=mom_axes, recast=False)

    out, axis_sample_out = prep.out_from_values(
        None,
        val_shape=val_shape,
        mom=mom,
        axis_neg=axis_neg,
        axis_new_size=axis_new_size,
        dtype=dtype,
        order=order,
    )

    assert out.shape == out_shape

    if order == "C":
        check = out
    elif axis_new_size is MISSING:
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

    assert check.flags.c_contiguous


# * xarray stuff
# ** Data
data_xarray_mark = pytest.mark.parametrize(
    "data", [xr.DataArray(np.zeros((2, 3, 4, 5)))]
)


@data_xarray_mark
@pytest.mark.parametrize(
    ("mom_axes", "out", "axis", "kws", "expected"),
    [
        (-1, None, (0,), {"axes_to_end": True}, None),
        (-1, np.zeros((2, 3)), (0,), {"axes_to_end": True}, (2, 3)),
        (-1, None, (0,), {"order": None}, None),
        (-1, None, (0,), {"order": "c"}, (3, 4, 5)),
        (1, None, (0,), {"order": "c"}, (4, 5, 3)),
        (-1, None, (0,), {"order": "c", "keepdims": True}, (3, 4, 1, 5)),
        (1, None, (0,), {"order": "c", "keepdims": True}, (4, 5, 1, 3)),
    ],
)
def test_prepare_data_xarray_optional_out_reduce(
    data, mom_axes, out, axis, kws, expected
) -> None:
    prep = prepare.PrepareDataXArray.factory(data=data, axes=mom_axes)
    dim = tuple(data.dims[k] for k in axis)

    kwargs = kws.copy()
    for k, v in {
        "keepdims": False,
        "axes_to_end": False,
        "order": None,
        "dtype": np.float64,
    }.items():
        kwargs.setdefault(k, v)

    out = prep.optional_out_reduce(out, target=data, dim=dim, **kwargs)

    if out is None:
        assert expected is None
    else:
        assert out.shape == expected

        if kwargs["order"] == "c" and kwargs["keepdims"]:
            assert moveaxis(
                out,
                (
                    *range(-len(axis) - prep.mom_params.ndim, -prep.mom_params.ndim),
                    *prep.mom_params.axes_last,
                ),
                (*axis, *prep.mom_params.get_axes(data)),
            ).flags.c_contiguous


@data_xarray_mark
@pytest.mark.parametrize(
    ("mom_axes", "out", "kws", "expected"),
    [
        (-1, None, {"axes_to_end": True}, None),
        (-1, np.zeros((2, 3)), {"axes_to_end": True}, (2, 3)),
        (-1, None, {"order": "c"}, (2, 3, 4, 5)),
        (0, None, {"order": "c"}, (3, 4, 5, 2)),
    ],
)
def test_prepare_data_xarray_optional_out_transform(
    data, mom_axes, out, kws, expected
) -> None:
    prep = prepare.PrepareDataXArray.factory(data=data, axes=mom_axes)

    kwargs = kws.copy()
    for k, v in {"axes_to_end": False, "order": None, "dtype": np.float64}.items():
        kwargs.setdefault(k, v)

    out = prep.optional_out_transform(out, target=data, **kwargs)

    if out is None:
        assert expected is None
    else:
        assert out.shape == expected

        if kwargs["order"] == "c":
            assert moveaxis(
                out,
                prep.mom_params.axes_last,
                prep.mom_params.get_axes(data),
            ).flags.c_contiguous


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
def test_prepare_data_xarray_optional_out_sample(
    data, kws, mom_params_kws, expected
) -> None:
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


# ** values
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
def test_prepare_values_xarray_values_for_reduction_0(
    target, other, kws, raises, match
):
    prep = prepare.PrepareValsXArray.factory(ndim=2)
    with pytest.raises(raises, match=match):
        prep.values_for_reduction(target, other, **kws)


@pytest.mark.parametrize(
    ("dim", "xshape", "xshape2", "yshape", "yshape2"),
    [
        ("dim_0", (2, 3, 4), (3, 4, 2), (2,), (2,)),
        ("dim_1", (2, 3, 4), (2, 4, 3), (3, 4), (4, 3)),
        ("dim_2", (2, 3, 4), (2, 3, 4), (4,), (4,)),
        ("dim_0", (2, 3, 4), (3, 4, 2), (2, 3, 4), (3, 4, 2)),
    ],
)
@dtype_mark
def test_prepare_values_xarray_values_for_reduction_1(
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
        assert y.shape == yshape2
        assert x.dtype == np.dtype(dtype or target.dtype)
        assert y.dtype == np.dtype(dtype or other.dtype)


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


@pytest.mark.parametrize(
    "target", [xr.DataArray(np.empty((10, 3, 4)), dims=list("abc"))]
)
@pytest.mark.parametrize(
    ("out", "mom", "args", "kws", "expected"),
    [
        (
            np.zeros((2, 3)),  # white like.  This should pass through
            (4,),
            (
                xr.DataArray(np.empty((3, 4, 10)), dims=["b", "c", "a"]),
                np.empty(10),
            ),
            {"axes_to_end": True},
            (2, 3),
        ),
        (
            None,
            (4,),
            (
                xr.DataArray(np.empty((3, 4, 10)), dims=["b", "c", "a"]),
                np.empty(10),
            ),
            {"axes_to_end": False, "order": "c"},
            (3, 4, 5),
        ),
        (
            np.zeros((3, 4, 6)),  # white lie.  This should pass through
            (4,),
            (
                xr.DataArray(np.empty((3, 4, 10)), dims=["b", "c", "a"]),
                np.empty(10),
            ),
            {"axes_to_end": False, "order": "c"},
            (3, 4, 6),
        ),
        (
            None,
            (4,),
            (
                xr.DataArray(np.empty((3, 4, 10)), dims=["b", "c", "a"]),
                np.empty(10),
            ),
            {"axes_to_end": False, "order": "c", "axis_new_size": None},
            (3, 4, 10, 5),
        ),
        (
            np.zeros((10, 3, 4, 6)),  # white lie.  This should pass through
            (4,),
            (
                xr.DataArray(np.empty((3, 4, 10)), dims=["b", "c", "a"]),
                np.empty(10),
            ),
            {"axes_to_end": False, "order": "c", "axis_new_size": None},
            (3, 4, 10, 6),
        ),
        (
            None,
            (4,),
            (
                xr.DataArray(np.empty((3, 4, 10)), dims=["b", "c", "a"]).to_dataset(
                    name="hello"
                ),
                np.empty(10),
            ),
            {"axes_to_end": False, "order": "c"},
            TypeError,
        ),
        # mom_axes
        (
            None,
            (4,),
            (
                xr.DataArray(np.empty((3, 4, 10)), dims=["b", "c", "a"]),
                np.empty(10),
            ),
            {"axes_to_end": False, "order": "c", "mom_axes": (0,)},
            (3, 4, 5),
        ),
        # mom_axes
        (
            None,
            (4,),
            (
                xr.DataArray(np.empty((3, 4, 10)), dims=["b", "c", "a"]),
                np.empty(10),
            ),
            {"axes_to_end": False, "order": "c", "mom_axes": (1,)},
            (3, 4, 5),
        ),
        (
            None,
            (4,),
            (
                xr.DataArray(np.empty((3, 4, 10)), dims=["b", "c", "a"]),
                np.empty(10),
            ),
            {"axes_to_end": False, "order": "c", "mom_axes": (0,), "axis_new_size": 10},
            (3, 4, 10, 5),
        ),
        # mom_axes
        (
            None,
            (4,),
            (
                xr.DataArray(np.empty((3, 4, 10)), dims=["b", "c", "a"]),
                np.empty(10),
            ),
            {"axes_to_end": False, "order": "c", "mom_axes": (1,), "axis_new_size": 10},
            (3, 4, 10, 5),
        ),
    ],
)
def test_prepare_values_xarray_optional_out_from_values(
    target,
    out,
    mom,
    args,
    kws,
    expected,
) -> None:
    prep = prepare.PrepareValsXArray.factory(ndim=len(mom))

    kwargs = kws.copy()
    for k, v in {
        "dim": "a",
        "axis_new_size": MISSING,
        "axes_to_end": False,
        "order": None,
        "dtype": np.float64,
        "mom_axes": None,
        "mom_params": None,
    }.items():
        kwargs.setdefault(k, v)

    if isinstance(expected, type):
        with pytest.raises(expected):
            _ = prep.optional_out_from_values(
                out, *args, target=target, mom=mom, **kwargs
            )
        return

    out, _ = prep.optional_out_from_values(out, *args, target=target, mom=mom, **kwargs)

    if out is None:
        assert expected is None
    else:
        assert out.shape == expected

        if kwargs["order"] == "c":
            if kwargs["axis_new_size"] is not MISSING:
                check = np.moveaxis(
                    out, -(len(mom) + 1), target.get_axis_num(kwargs["dim"])
                )
            else:
                check = out

            mom_axes_last = range(-len(mom), 0)
            mom_axes = kwargs.get("mom_axes", None)
            if mom_axes is None:
                mom_axes = mom_axes_last
            check = np.moveaxis(check, mom_axes_last, mom_axes)

            assert check.flags.c_contiguous

# mypy: disable-error-code="no-untyped-def, no-untyped-call, call-overload"
from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import xarray as xr

import cmomy
from cmomy import convert
from cmomy.core.utils import mom_to_mom_shape


# * Convert
def get_raw_moments(*xy, weight, mom, axis):
    if len(xy) == 1:
        x = xy[0]
        if isinstance(mom, tuple):
            mom = mom[0]
        return np.stack(
            [np.average(x**i, weights=weight, axis=axis) for i in range(mom + 1)],
            axis=-1,
        )

    assert len(xy) == len(mom) == 2
    x, y = xy
    shape = (*x.shape[:axis], *x.shape[axis + 1 :], *mom_to_mom_shape(mom))
    out = np.zeros(shape)
    for i in range(mom[0] + 1):
        for j in range(mom[1] + 1):
            out[..., i, j] = np.average(x**i * y**j, weights=weight, axis=axis)
    return out


@pytest.fixture
def data_and_kwargs(rng, request):
    shapes, kwargs = request.param
    if isinstance(shapes, list):
        data = [rng.random(s) for s in shapes]
    else:
        data = rng.random(shapes)
    return data, kwargs


def unpack_data_to_xy_weight(data, mom):
    if len(data) == len(mom):
        return data, None
    if len(data) == len(mom) + 1:
        return data[:-1], data[-1]
    msg = f"bad unpack: {len(data)=}, {len(mom)=}"
    raise ValueError(msg)


@pytest.mark.parametrize(
    "data_and_kwargs",
    [
        ([20], {"mom": (4,), "axis": 0}),
        ([20, 20], {"mom": (4,), "axis": 0}),
        ([(20, 3), (20, 3)], {"mom": (4,), "axis": 0}),
        ([(20, 3), (20, 3), (20, 3)], {"mom": (4, 4), "axis": 0}),
    ],
    indirect=True,
)
def test_raw(data_and_kwargs) -> None:
    data, kwargs = data_and_kwargs
    xy, weight = unpack_data_to_xy_weight(data, kwargs["mom"])

    raw = get_raw_moments(*xy, weight=weight, **kwargs)
    c = cmomy.wrap_reduce_vals(*xy, weight=weight, **kwargs)
    np.testing.assert_allclose(c.to_raw(weight=1), raw)

    # from raw
    cr = cmomy.wrap_raw(raw, mom_ndim=len(kwargs["mom"]))  # pyright: ignore[reportArgumentType, reportCallIssue]
    np.testing.assert_allclose(cr, c.assign_moment(weight=1.0))


# * Moments to comoments
@pytest.mark.parametrize("mom", [(1,), (1, 2, 3), 1])
def test__validate_mom_moments_to_comoments_mom(mom):
    with pytest.raises(ValueError, match="Must supply length 2.*"):
        convert._validate_mom_moments_to_comoments(mom=mom, mom_orig=4)


@pytest.mark.parametrize(
    ("mom_orig", "mom", "expected"),
    [
        (3, (1, 2), (1, 2)),
        (3, (1, -1), (1, 2)),
        (3, (-1, 1), (2, 1)),
        (3, (0, 3), "error"),
        (3, (3, 0), "error"),
        (3, (1, 1), (1, 1)),
        (3, (1, 3), "error"),
        (3, (3, 1), "error"),
    ],
)
def test__validate_mom_moments_to_comoments(mom_orig, mom, expected) -> None:
    if expected == "error":
        with pytest.raises(ValueError, match=".* inconsistent with original moments.*"):
            convert._validate_mom_moments_to_comoments(mom=mom, mom_orig=mom_orig)

    else:
        assert (
            convert._validate_mom_moments_to_comoments(mom=mom, mom_orig=mom_orig)
            == expected
        )


@pytest.mark.parametrize("shape", [(10,), (10, 3)])
@pytest.mark.parametrize("dtype", [None, np.float32])
def test_moments_to_comoments(rng, shape, dtype) -> None:
    import cmomy

    x = rng.random(shape)

    data1 = cmomy.reduce_vals(x, axis=0, mom=3)
    data2 = cmomy.reduce_vals(x, x, axis=0, mom=(1, 2))

    out = convert.moments_to_comoments(data1, mom=(1, -1), dtype=dtype)

    if dtype is not None:
        assert out.dtype.type == dtype
    np.testing.assert_allclose(data2, out)

    # CentralMoments
    c = cmomy.CentralMoments(data1, mom_ndim=1)
    c2 = cmomy.CentralMoments(data2, mom_ndim=2)
    np.testing.assert_allclose(c2, c.moments_to_comoments(mom=(1, -1)))

    cx = c.to_x()
    c2x = c2.to_x(mom_dims=("a", "b"))

    xr.testing.assert_allclose(
        cx.moments_to_comoments(mom=(1, -1), mom_dims2=("a", "b")).obj, c2x.obj
    )

    # same mom name as original
    xr.testing.assert_allclose(
        cx.moments_to_comoments(mom=(1, -1)).obj,
        c2x.obj.rename({"a": "mom_0", "b": "mom_1"}),
    )

    # raise error for mom_ndim=2
    for _c in [c2, c2x]:
        with pytest.raises(ValueError, match="Only implemented for.*"):
            c2.moments_to_comoments(mom=(1, -1))

    # keep attrs
    cx = c.to_x(attrs={"hello": "there"})

    xr.testing.assert_allclose(
        cx.moments_to_comoments(mom=(1, -1), mom_dims2=("a", "b")).obj, c2x.obj
    )

    c2x = c2.to_x(mom_dims=("a", "b"), attrs={"hello": "there"})
    xr.testing.assert_allclose(
        cx.moments_to_comoments(mom=(1, -1), mom_dims2=("a", "b"), keep_attrs=True).obj,
        c2x.obj,
    )


# * Cumulative
@pytest.mark.parametrize(
    ("shape", "axis", "mom_ndim"),
    [
        ((10, 3), 0, 1),
        ((10, 2, 3), 0, 1),
        ((2, 10, 3), 1, 1),
        ((10, 2, 2), 0, 2),
        ((2, 5, 2, 2), 1, 2),
    ],
)
def test_cumulative(rng, shape, axis, mom_ndim) -> None:
    import cmomy

    data = rng.random(shape)
    cout = cmomy.convert.cumulative(data, axis=axis, mom_ndim=mom_ndim)

    cr = cmomy.CentralMoments(data, mom_ndim=mom_ndim).reduce(axis=axis)

    # check last
    np.testing.assert_allclose(np.take(cout, -1, axis=axis), cr)

    # dumb way
    cn = cr.zeros_like()
    for i in range(data.shape[axis]):
        cn.push_data(np.take(data, i, axis=axis))
        np.testing.assert_allclose(cn, np.take(cout, i, axis=axis))

    check = cmomy.convert.cumulative(cout, axis=axis, mom_ndim=mom_ndim, inverse=True)
    np.testing.assert_allclose(check, data)


@pytest.mark.parametrize("parallel", [True, False])
def test_cumulative_options(rng, parallel) -> None:
    import cmomy

    func = partial(cmomy.convert.cumulative, mom_ndim=1, axis=0, parallel=parallel)
    ifunc = partial(
        cmomy.convert.cumulative, mom_ndim=1, axis=0, inverse=True, parallel=parallel
    )
    data = rng.random((10, 3))
    xdata = xr.DataArray(data, dims=["a", "b"], attrs={"hello": "there"})

    out = np.zeros_like(data)

    np.testing.assert_allclose(ifunc(func(data)), data)
    xr.testing.assert_allclose(ifunc(func(xdata)), xdata)

    for d in [data, xdata]:
        assert func(d.astype(np.float32)).dtype.type == np.float32
        assert func(d, dtype=np.float32).dtype.type == np.float32
        assert (
            func(d.astype(np.float32), out=out, dtype=np.float16).dtype.type
            == np.float64
        )

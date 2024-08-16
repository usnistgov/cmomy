# mypy: disable-error-code="no-untyped-def, no-untyped-call, call-overload"
from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import xarray as xr

from cmomy import convert


# * Convert
def test_to_raw_moments(other) -> None:
    raw = other.raw
    if raw is not None:
        # straight convert

        r = convert.moments_type(other.to_values(), mom_ndim=other.mom_ndim, to="raw")
        np.testing.assert_allclose(r, raw)

        out = np.zeros_like(raw)
        _ = convert.moments_type(
            other.to_values(), mom_ndim=other.mom_ndim, to="raw", out=out
        )
        np.testing.assert_allclose(out, raw)

        np.testing.assert_allclose(raw, other.s.to_raw())

        # test with weights
        for w in [10.0, 1.0]:
            expected = raw.copy()
            if other.mom_ndim == 1:
                expected[..., 0] = w
            else:
                expected[..., 0, 0] = w

            np.testing.assert_allclose(expected, other.s.to_raw(weight=w))

            if w == 1.0:
                np.testing.assert_allclose(expected, other.s.rmom())


def test_raises_convert_moments() -> None:
    x = np.zeros(3)

    for to in ["raw", "central"]:
        with pytest.raises(ValueError):
            convert.moments_type(x, mom_ndim=2, to=to)  # type: ignore[call-overload]


def test_to_central_moments(other) -> None:
    raw = other.s.to_raw()

    cen = convert.moments_type(raw, to="central", mom_ndim=other.mom_ndim)
    np.testing.assert_allclose(cen, other.to_values())

    # also test from raw method
    t = other.cls.from_raw(raw, mom_ndim=other.mom_ndim)
    np.testing.assert_allclose(t.to_values(), other.to_values(), rtol=1e-6, atol=1e-14)


def test_from_raw(other) -> None:
    raws = np.array([s.to_raw() for s in other.S])
    t = other.cls.from_raw(
        raws,
        mom_ndim=other.mom_ndim,
    ).reduce(axis=0)
    np.testing.assert_allclose(t.to_values(), other.to_values())


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

    # xCentralMoments
    cx = c.to_x()
    c2x = c2.to_x(mom_dims=("a", "b"))

    xr.testing.assert_allclose(
        cx.moments_to_comoments(mom=(1, -1), mom_dims2=("a", "b")).values, c2x.values
    )

    # same mom name as original
    xr.testing.assert_allclose(
        cx.moments_to_comoments(mom=(1, -1)).values,
        c2x.values.rename({"a": "mom_0", "b": "mom_1"}),  # noqa: PD011
    )

    # raise error for mom_ndim=2
    for _c in [c2, c2x]:
        with pytest.raises(ValueError, match="Only implemented for.*"):
            c2.moments_to_comoments(mom=(1, -1))

    # keep attrs
    cx = c.to_x(attrs={"hello": "there"})

    xr.testing.assert_allclose(
        cx.moments_to_comoments(mom=(1, -1), mom_dims2=("a", "b")).values, c2x.values
    )

    c2x = c2.to_x(mom_dims=("a", "b"), attrs={"hello": "there"})
    xr.testing.assert_allclose(
        cx.moments_to_comoments(
            mom=(1, -1), mom_dims2=("a", "b"), keep_attrs=True
        ).values,
        c2x.values,
    )


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

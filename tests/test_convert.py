# mypy: disable-error-code="no-untyped-def, no-untyped-call"
import numpy as np
import pytest
import xarray as xr

# import cmomy.central as central
from cmomy import convert


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
        cx.moments_to_comoments(mom=(1, -1), mom_dims=("a", "b")).values, c2x.values
    )

    # raise error for mom_ndim=2
    for _c in [c2, c2x]:
        with pytest.raises(ValueError, match="Only implemented for.*"):
            c2.moments_to_comoments(mom=(1, -1))

    # keep attrs
    cx = c.to_x(attrs={"hello": "there"})

    xr.testing.assert_allclose(
        cx.moments_to_comoments(mom=(1, -1), mom_dims=("a", "b")).values, c2x.values
    )

    c2x = c2.to_x(mom_dims=("a", "b"), attrs={"hello": "there"})
    xr.testing.assert_allclose(
        cx.moments_to_comoments(
            mom=(1, -1), mom_dims=("a", "b"), keep_attrs=True
        ).values,
        c2x.values,
    )


@pytest.mark.parametrize("shape", [(10,), (10, 3)])
@pytest.mark.parametrize("mom", [(3,), (3, 3)])
def test_assign_weight(rng, shape, mom) -> None:
    import cmomy

    x = rng.random(shape)
    xy = (x,) if len(mom) == 1 else (x, x)

    c1 = cmomy.CentralMoments.from_vals(*xy, mom=mom, axis=0)
    c2 = cmomy.CentralMoments.from_vals(*xy, mom=mom, axis=0, weight=2)

    cc = c1.assign_weight(x.shape[0] * 2, copy=True)
    np.testing.assert_allclose(cc.values, c2.values)
    assert not np.shares_memory(cc.data, c1.data)

    ca = c1.copy()
    cc = ca.assign_weight(x.shape[0] * 2, copy=False)
    np.testing.assert_allclose(cc.values, c2.values)
    np.testing.assert_allclose(cc.values, ca.values)
    assert np.shares_memory(cc.data, ca.data)

    # xcentral
    cx1 = c1.to_x()
    cx2 = c2.to_x()

    ccx = cx1.assign_weight(x.shape[0] * 2, copy=True)
    xr.testing.assert_allclose(cx2.values, ccx.values)
    assert not np.shares_memory(ccx.data, cx1.data)

    ccx = cx1.assign_weight(x.shape[0] * 2, copy=False)
    xr.testing.assert_allclose(cx2.values, ccx.values)
    xr.testing.assert_allclose(cx1.values, ccx.values)
    assert np.shares_memory(ccx.data, cx1.data)

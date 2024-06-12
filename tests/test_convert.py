# mypy: disable-error-code="no-untyped-def, no-untyped-call"
import numpy as np
import pytest

# import cmomy.central as central
from cmomy import convert


def test_to_raw_moments(other) -> None:
    raw = other.raw
    if raw is not None:
        # straight convert

        r = convert(other.to_values(), mom_ndim=other.mom_ndim, to="raw")
        np.testing.assert_allclose(r, raw)

        out = np.zeros_like(raw)
        _ = convert(other.to_values(), mom_ndim=other.mom_ndim, to="raw", out=out)
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
            convert(x, mom_ndim=2, to=to)  # type: ignore[call-overload]


def test_to_central_moments(other) -> None:
    raw = other.s.to_raw()

    cen = convert(raw, to="central", mom_ndim=other.mom_ndim)
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

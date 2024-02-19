# mypy: disable-error-code="no-untyped-def, no-untyped-call"
import numpy as np
import pytest

# import cmomy.central as central
from cmomy import convert


def test_to_raw_moments(other) -> None:
    raw = other.raw
    if raw is not None:
        # straight convert

        if not other.cov:
            # non -1 axis

            r = convert.to_raw_moments(np.moveaxis(other.to_values(), -1, 0), axis=0)
            np.testing.assert_allclose(raw, r)

            r = convert.to_raw_moments(other.to_values(), axis=None)

        else:
            out = np.zeros_like(raw)
            r = convert.to_raw_comoments(other.to_values(), axis=None, out=out)

        np.testing.assert_allclose(raw, r)
        np.testing.assert_allclose(raw, other.s.to_raw())


def test_raises_convert_moments() -> None:
    x = np.zeros((2, 3, 4))

    with pytest.raises(ValueError):
        convert._convert_moments(
            x, axis=(0,), target_axis=(0, 1), func=lambda *args: args
        )

    with pytest.raises(ValueError):
        convert._convert_moments(
            x,
            axis=(-1,),
            target_axis=(-1,),
            func=lambda *args: args,
            out=np.zeros((2, 3, 3)),
        )


def test_to_central_moments(other) -> None:
    raw = other.s.to_raw()
    if not other.cov:
        cen = convert.to_central_moments(raw, axis=None)
    else:
        cen = convert.to_central_comoments(raw, axis=None)
    np.testing.assert_allclose(cen, other.to_values())

    # also test from raw method
    t = other.cls.from_raw(raw, mom=other.mom, convert_kws={"order": "C"})
    np.testing.assert_allclose(t.to_values(), other.to_values(), rtol=1e-6, atol=1e-14)


def test_from_raws(other) -> None:
    raws = np.array([s.to_raw() for s in other.S])
    t = other.cls.from_raws(
        raws, mom_ndim=other.mom_ndim, axis=0, convert_kws={"order": "C"}
    )
    np.testing.assert_allclose(t.to_values(), other.to_values())

# mypy: disable-error-code="no-untyped-def, no-untyped-call"
"""Simple test for numerical stability"""

import numpy as np
import pytest

import cmomy
from cmomy._lib._binomial import factory_binomial
from cmomy.reduction import reduce_vals


def algo(vals, mom, norm=True):
    order = mom
    M = np.zeros((order + 1,), dtype=float)

    bfac = factory_binomial(mom + 1)

    for x in vals:
        delta = x - M[1]

        M[0] += 1
        M[1] += delta / M[0]

        for p in range(order, 2, -1):
            Mp = (
                M[p] + ((M[0] - 1) / (-M[0]) ** p + ((M[0] - 1) / M[0]) ** p) * delta**p
            )

            for k in range(1, p - 2 + 1):
                Mp += bfac[p, k] * M[p - k] * (-delta / M[0]) ** k

            M[p] = Mp

        M[2] += (M[0] - 1) / M[0] * delta**2

    if norm:
        M[2:] /= M[0]

    return M


def algo2(vals, mom):
    out = np.empty((mom + 1,), dtype=float)
    out[0] = len(vals)
    out[1] = vals.mean(axis=0)

    out[2:] = ((vals - out[1])[:, None] ** np.arange(2, mom + 1)).mean(axis=0)
    return out


def test_stability() -> None:
    rng = np.random.default_rng(seed=0)
    x = rng.random(10000)

    mom = 5
    moments = reduce_vals(x, mom=mom, axis=0)
    c = cmomy.wrap_reduce_vals(x, mom=mom, axis=0)
    test = algo(x, mom=mom)
    test2 = algo2(x, mom=mom)

    # all should be good
    np.testing.assert_allclose(c.to_numpy(), moments)
    np.testing.assert_allclose(moments, test)
    np.testing.assert_allclose(moments, test2)

    # scaling
    # y -> x * a + b
    # <y> = <x * a + b>  = <x> * a + b
    # <(y - <y>)**n> = <(x*a + b - <x * a + b>)**n> = a ** n * <dx>**n

    a = 1
    b = 1e6

    # true value just by shifting first moment
    moments_test = moments.copy()
    moments_test[1] = moments_test[1] * a + b
    moments_test[2:] *= a ** np.arange(2, mom + 1)

    # calculated
    moments_shift = reduce_vals(x * a + b, mom=5, axis=0)
    c_shift = cmomy.wrap_reduce_vals(x * a + b, mom=5, axis=0)
    test_shift = algo(x * a + b, mom=5)
    test2_shift = algo2(x * a + b, mom=5)

    np.testing.assert_allclose(moments_test, c_shift.to_numpy())
    np.testing.assert_allclose(moments_test, test_shift)

    # with pytest.raises(AssertionError):
    np.testing.assert_allclose(moments_test, moments_shift)

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(moments_test, test2_shift)

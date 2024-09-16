# mypy: disable-error-code="no-untyped-def, no-untyped-call, call-overload"
"""
Hammer out that reduce_vals gives same results as our "dumb" equations...
"""

from __future__ import annotations

import numpy as np
import pytest

import cmomy

from ._simple_cmom import get_cmom, get_comom

rng = np.random.default_rng(14)


@pytest.mark.parametrize(
    ("x", "weight", "axis"),
    [
        (rng.random(20), None, 0),
        (rng.random(20), rng.random(20), 0),
        (rng.random((20, 2)), None, 0),
        (rng.random((20, 2)), rng.random((20, 1)), 0),
        (rng.random((2, 20)), np.asarray(2.0), 1),
        (rng.random((2, 20)), rng.random(20), 1),
        (rng.random((1, 2, 20)), None, -1),
        (rng.random((1, 2, 20)), rng.random(20), 2),
        (rng.random((1, 2, 20)), rng.random((2, 20)), 2),
        (rng.random((1, 2, 20)), rng.random((1, 2, 20)), 2),
    ],
)
@pytest.mark.parametrize("mom", [5])
def test_reduce_1(x, weight, mom, axis) -> None:
    expected = get_cmom(weight, x, mom, axis=axis)

    check = cmomy.reduce_vals(x, weight=weight, mom=mom, axis=axis)
    np.testing.assert_allclose(check, expected, atol=1e-16)

    data = cmomy.utils.vals_to_data(x=x, weight=weight, mom=mom)
    check = cmomy.reduce_data(data, mom_ndim=1, axis=axis)
    np.testing.assert_allclose(check, expected, atol=1e-16)


@pytest.mark.parametrize(
    ("x", "y", "weight", "axis"),
    [
        (rng.random(20), rng.random(20), None, 0),
        (rng.random(20), rng.random(20), rng.random(20), 0),
        (rng.random((20, 2)), rng.random((20, 1)), None, 0),
        (rng.random((20, 2)), rng.random((20, 1)), rng.random((20, 1)), 0),
        (rng.random((2, 20)), rng.random(20), np.asarray(2.0), 1),
        (rng.random((2, 20)), rng.random((2, 20)), rng.random(20), 1),
        (rng.random((1, 2, 20)), rng.random(20), None, -1),
        (rng.random((1, 2, 20)), rng.random((2, 20)), rng.random(20), 2),
        (rng.random((1, 2, 20)), rng.random((1, 2, 20)), rng.random((2, 20)), 2),
        (rng.random((1, 2, 20)), rng.random(20), rng.random((1, 2, 20)), 2),
    ],
)
@pytest.mark.parametrize("mom", [(5, 5)])
def test_reduce_2(x, y, weight, mom, axis) -> None:
    expected = get_comom(weight, x, y, mom, axis=axis)

    check = cmomy.reduce_vals(x, y, weight=weight, mom=mom, axis=axis)
    np.testing.assert_allclose(check, expected, atol=1e-16)

    data = cmomy.utils.vals_to_data(x, y, weight=weight, mom=mom)
    check = cmomy.reduce_data(data, mom_ndim=2, axis=axis)
    np.testing.assert_allclose(check, expected, atol=1e-16)

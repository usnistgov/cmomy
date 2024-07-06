import numpy as np
import pytest

try:
    from scipy.special import ndtr, ndtri

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


pytestmark = [
    pytest.mark.scipy,
    pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed"),
]


def test_ndtr(rng) -> None:
    from cmomy import _prob

    x = rng.random(100)

    a = ndtr(x)
    b = _prob.ndtr(x)
    np.testing.assert_allclose(a, b)

    aa = ndtri(a)
    bb = _prob.ndtri(b)

    np.testing.assert_allclose(aa, bb)

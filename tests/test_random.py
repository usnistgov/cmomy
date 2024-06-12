import numpy as np
import pytest

from cmomy import random


def test_set_rng() -> None:
    rng = np.random.default_rng()

    random.set_internal_rng(rng)

    assert random.default_rng() is rng

    assert random.get_internal_rng() is rng

    random._DATA = {}

    with pytest.raises(ValueError):
        random.get_internal_rng()


def test_default_rng() -> None:
    rng = random.default_rng()

    assert rng is random.get_internal_rng()

    rng2 = np.random.default_rng()

    assert random.default_rng(rng2) is rng2
    assert random.default_rng() is rng


def test_validate_rng() -> None:
    rs = np.random.RandomState()
    with pytest.warns(UserWarning):
        out = random.validate_rng(rs)  # type: ignore[arg-type]

    assert out is rs  # type: ignore[comparison-overlap]

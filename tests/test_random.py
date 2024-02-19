import numpy as np
import pytest

import cmomy


def test_set_rng() -> None:
    rng = np.random.default_rng()

    cmomy.random.set_internal_rng(rng)

    assert cmomy.random.default_rng() is rng

    assert cmomy.random.get_internal_rng() is rng

    cmomy.random._DATA = {}

    with pytest.raises(ValueError):
        cmomy.random.get_internal_rng()


def test_default_rng() -> None:
    rng = cmomy.random.default_rng()

    assert rng is cmomy.random.get_internal_rng()

    rng2 = np.random.default_rng()

    assert cmomy.random.default_rng(rng2) is rng2
    assert cmomy.random.default_rng() is rng


def test_validate_rng() -> None:
    rs = np.random.RandomState()
    with pytest.warns(UserWarning):
        out = cmomy.random.validate_rng(rs)  # type: ignore[arg-type]

    assert out is rs  # type: ignore[comparison-overlap]

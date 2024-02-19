import numpy as np
import pytest

from cmomy import utils


def test_shape_insert() -> None:
    assert utils.shape_insert_axis((1, 2, 3), 0, 10) == (10, 1, 2, 3)

    assert utils.shape_insert_axis((1, 2, 3), -1, 10) == (1, 2, 3, 10)

    with pytest.raises(ValueError):
        utils.shape_insert_axis((1, 2, 3), None, 10)


def test_axis_expand_broadcast() -> None:
    with pytest.raises(TypeError):
        utils.axis_expand_broadcast([1, 2, 3], shape=(3, 10), axis=0, verify=False)

    x = np.arange(3)

    with pytest.raises(ValueError):
        utils.axis_expand_broadcast(x, shape=(3, 2), expand=True, axis=None)

    with pytest.raises(ValueError):
        utils.axis_expand_broadcast(x, shape=(4, 2), expand=True, axis=0)

    expected = np.tile(x, (2, 1)).T
    np.testing.assert_allclose(
        utils.axis_expand_broadcast(x, shape=(3, 2), expand=True, axis=0), expected
    )

    np.testing.assert_allclose(
        utils.axis_expand_broadcast(x, shape=(2, 3), axis=1, roll=False), expected.T
    )
    np.testing.assert_allclose(
        utils.axis_expand_broadcast(x, shape=(2, 3), axis=1, roll=True), expected
    )

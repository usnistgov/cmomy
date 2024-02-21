import numpy as np
import pytest

from cmomy import utils


def test_shape_insert() -> None:
    assert utils.shape_insert_axis(shape=(1, 2, 3), axis=0, new_size=10) == (
        10,
        1,
        2,
        3,
    )

    assert utils.shape_insert_axis(shape=(1, 2, 3), axis=-1, new_size=10) == (
        1,
        2,
        3,
        10,
    )

    with pytest.raises(ValueError):
        utils.shape_insert_axis(shape=(1, 2, 3), axis=None, new_size=10)


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


def test_validate_mom_and_mom_ndim() -> None:
    with pytest.raises(ValueError):
        utils.validate_mom_and_mom_ndim(mom=None, mom_ndim=None)
    with pytest.raises(ValueError):
        utils.validate_mom_and_mom_ndim(mom=None, mom_ndim=1)

    with pytest.raises(ValueError):
        utils.validate_mom_and_mom_ndim(mom=None, mom_ndim=2, shape=(2,))

    with pytest.raises(ValueError):
        utils.validate_mom_and_mom_ndim(mom=None, mom_ndim=3, shape=(2, 3, 4))  # type: ignore[arg-type]

    assert utils.validate_mom_and_mom_ndim(mom=(2, 2), mom_ndim=None) == ((2, 2), 2)

    with pytest.raises(ValueError):
        utils.validate_mom_and_mom_ndim(mom=(2, 2, 2), mom_ndim=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        utils.validate_mom_and_mom_ndim(mom=(2, 2), mom_ndim=1)


def test_mom_to_mom_ndim() -> None:
    assert utils.mom_to_mom_ndim(2) == 1
    assert utils.mom_to_mom_ndim((2, 2)) == 2

    with pytest.raises(ValueError):
        utils.mom_to_mom_ndim((2, 2, 2))  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        utils.mom_to_mom_ndim([2, 2])  # type: ignore[arg-type]


def test_select_mom_ndim() -> None:
    assert utils.select_mom_ndim(mom=2, mom_ndim=None) == 1
    assert utils.select_mom_ndim(mom=(2, 2), mom_ndim=None) == 2

    with pytest.raises(ValueError):
        utils.select_mom_ndim(mom=(2, 2), mom_ndim=1)

    with pytest.raises(TypeError):
        utils.select_mom_ndim(mom=None, mom_ndim=None)

    assert utils.select_mom_ndim(mom=None, mom_ndim=1) == 1
    with pytest.raises(ValueError):
        utils.select_mom_ndim(mom=None, mom_ndim=3)  # type: ignore[arg-type]

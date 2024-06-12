from __future__ import annotations

import numpy as np
import pytest

import cmomy


@pytest.mark.parametrize(
    "by",
    [
        [0] * 10 + [1] * 10,
        [0] * 9,
    ],
)
def test_grouped_bad_by(by: list[int]) -> None:
    data = np.zeros((10, 2, 4))
    with pytest.raises(ValueError, match=".*data.shape.*"):
        cmomy.reduce_data_grouped(data, mom_ndim=1, by=by, axis=0)


def test__validate_index() -> None:
    index = [0, 1, 2, 3]
    group_start = [0, 2]
    group_end = [2, 4]

    index_, start_, end_ = cmomy.reduction._validate_index(
        4, index, group_start, group_end
    )

    np.testing.assert_allclose(index, index_)
    np.testing.assert_allclose(group_start, start_)
    np.testing.assert_allclose(group_end, end_)

    # index outside bounds
    with pytest.raises(ValueError, match=".*min.*< 0.*"):
        _ = cmomy.reduction._validate_index(4, [-1, 0, 1, 2], group_start, group_end)

    # index outside max
    with pytest.raises(ValueError, match=".*max.*>.*"):
        _ = cmomy.reduction._validate_index(4, [0, 1, 2, 3, 4], group_start, group_end)

    # mismatch group start/end
    with pytest.raises(ValueError, match=r".*len.*start.*len.*end.*"):
        _ = cmomy.reduction._validate_index(4, index, [0, 1], [1, 2, 3])

    # end < start
    with pytest.raises(ValueError, match=".*end < start.*"):
        _ = cmomy.reduction._validate_index(4, index, [0, 2], [2, 1])
    # zero length index
    index = []
    group_start = [0]
    group_end = [0]

    index_, start_, end_ = cmomy.reduction._validate_index(
        4, index, group_start, group_end
    )

    assert len(index_) == 0
    np.testing.assert_allclose(group_start, start_)
    np.testing.assert_allclose(group_end, end_)

    # bad end
    with pytest.raises(ValueError, match=".*With zero length.*"):
        _ = cmomy.reduction._validate_index(4, index, group_start, [10])


def test_indexed(rng: np.random.Generator) -> None:
    data = rng.random((10, 2, 3))

    by = [0] * 5 + [1] * 5

    a = cmomy.reduce_data_grouped(data, mom_ndim=1, by=by, axis=0)

    _groups, index, start, end = cmomy.reduction.factor_by_to_index(by)

    b = cmomy.reduction.reduce_data_indexed(
        data,
        mom_ndim=1,
        index=index,
        group_start=start,
        group_end=end,
        scale=[1] * 10,
        axis=0,
    )

    np.testing.assert_allclose(a, b)

    # bad scale

    with pytest.raises(ValueError, match=".*len.*scale.*"):
        _ = cmomy.reduction.reduce_data_indexed(
            data,
            mom_ndim=1,
            index=index,
            group_start=start,
            group_end=end,
            scale=[1] * 11,
            axis=0,
        )

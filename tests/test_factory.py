from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cmomy._lib import (
    convert,
    convert_cov,
    factory,
    indexed,
    indexed_cov,
    indexed_cov_parallel,
    indexed_parallel,
    moving,
    moving_cov,
    moving_cov_parallel,
    moving_parallel,
    push,
    push_cov,
    push_cov_parallel,
    push_parallel,
    resample,
    resample_cov,
    resample_cov_parallel,
    resample_parallel,
)

if TYPE_CHECKING:
    from typing import Any, Callable

    from cmomy.typing import Mom_NDim

    Func = Callable[..., Any]


@pytest.mark.parametrize(
    ("mom_ndim", "parallel", "expected"),
    [
        (
            1,
            False,
            (
                push.push_val,
                push.reduce_vals,
                push.push_data,
                push.reduce_data,
            ),
        ),
        (
            1,
            True,
            (
                push_parallel.push_val,
                push_parallel.reduce_vals,
                push_parallel.push_data,
                push_parallel.reduce_data,
            ),
        ),
        (
            2,
            False,
            (
                push_cov.push_val,
                push_cov.reduce_vals,
                push_cov.push_data,
                push_cov.reduce_data,
            ),
        ),
        (
            2,
            True,
            (
                push_cov_parallel.push_val,
                push_cov_parallel.reduce_vals,
                push_cov_parallel.push_data,
                push_cov_parallel.reduce_data,
            ),
        ),
    ],
)
def test_factory_pusher(
    mom_ndim: Mom_NDim, parallel: bool, expected: tuple[Func | None, ...]
) -> None:
    assert factory.factory_pusher(mom_ndim, parallel) == expected


@pytest.mark.parametrize(
    ("mom_ndim", "parallel", "expected"),
    [
        (1, False, resample.resample_vals),
        (1, True, resample_parallel.resample_vals),
        (2, False, resample_cov.resample_vals),
        (2, True, resample_cov_parallel.resample_vals),
    ],
)
def test_factory_resample_vals(
    mom_ndim: Mom_NDim,
    parallel: bool,
    expected: factory.ResampleVals | factory.ResampleValsCov,
) -> None:
    assert factory.factory_resample_vals(mom_ndim, parallel) == expected


@pytest.mark.parametrize(
    ("mom_ndim", "parallel", "expected"),
    [
        (1, False, resample.resample_data_fromzero),
        (1, True, resample_parallel.resample_data_fromzero),
        (2, False, resample_cov.resample_data_fromzero),
        (2, True, resample_cov_parallel.resample_data_fromzero),
    ],
)
def test_factory_resample_data(
    mom_ndim: Mom_NDim, parallel: bool, expected: factory.ResampleData
) -> None:
    assert factory.factory_resample_data(mom_ndim, parallel) == expected


@pytest.mark.parametrize(
    ("mom_ndim", "parallel", "expected"),
    [
        (1, False, resample.jackknife_vals_fromzero),
        (1, True, resample_parallel.jackknife_vals_fromzero),
        (2, False, resample_cov.jackknife_vals_fromzero),
        (2, True, resample_cov_parallel.jackknife_vals_fromzero),
    ],
)
def test_factory_jackknife_vals(
    mom_ndim: Mom_NDim,
    parallel: bool,
    expected: factory.JackknifeVals | factory.JackknifeValsCov,
) -> None:
    assert factory.factory_jackknife_vals(mom_ndim, parallel) == expected


@pytest.mark.parametrize(
    ("mom_ndim", "parallel", "expected"),
    [
        (1, False, resample.jackknife_data_fromzero),
        (1, True, resample_parallel.jackknife_data_fromzero),
        (2, False, resample_cov.jackknife_data_fromzero),
        (2, True, resample_cov_parallel.jackknife_data_fromzero),
    ],
)
def test_factory_jackknife_data(
    mom_ndim: Mom_NDim, parallel: bool, expected: factory.JackknifeData
) -> None:
    assert factory.factory_jackknife_data(mom_ndim, parallel) == expected


@pytest.mark.parametrize(
    ("mom_ndim", "parallel", "expected"),
    [
        (1, False, push.reduce_vals),
        (1, True, push_parallel.reduce_vals),
        (2, False, push_cov.reduce_vals),
        (2, True, push_cov_parallel.reduce_vals),
    ],
)
def test_reduce_vals(
    mom_ndim: Mom_NDim,
    parallel: bool,
    expected: factory.ReduceVals | factory.ReduceValsCov,
) -> None:
    assert factory.factory_reduce_vals(mom_ndim, parallel) == expected


@pytest.mark.parametrize(
    ("mom_ndim", "parallel", "expected"),
    [
        (1, False, push.reduce_data_fromzero),
        (1, True, push_parallel.reduce_data_fromzero),
        (2, False, push_cov.reduce_data_fromzero),
        (2, True, push_cov_parallel.reduce_data_fromzero),
    ],
)
def test_reduce_data(
    mom_ndim: Mom_NDim, parallel: bool, expected: factory.ReduceData
) -> None:
    assert factory.factory_reduce_data(mom_ndim, parallel) == expected


@pytest.mark.parametrize(
    ("mom_ndim", "parallel", "expected"),
    [
        (1, False, indexed.reduce_data_grouped),
        (1, True, indexed_parallel.reduce_data_grouped),
        (2, False, indexed_cov.reduce_data_grouped),
        (2, True, indexed_cov_parallel.reduce_data_grouped),
    ],
)
def test_reduce_data_grouped(
    mom_ndim: Mom_NDim, parallel: bool, expected: factory.ReduceDataGrouped
) -> None:
    assert factory.factory_reduce_data_grouped(mom_ndim, parallel) == expected


@pytest.mark.parametrize(
    ("mom_ndim", "parallel", "expected"),
    [
        (1, False, indexed.reduce_data_indexed_fromzero),
        (1, True, indexed_parallel.reduce_data_indexed_fromzero),
        (2, False, indexed_cov.reduce_data_indexed_fromzero),
        (2, True, indexed_cov_parallel.reduce_data_indexed_fromzero),
    ],
)
def test_reduce_data_indexed(
    mom_ndim: Mom_NDim, parallel: bool, expected: factory.ReduceDataIndexed
) -> None:
    assert factory.factory_reduce_data_indexed(mom_ndim, parallel) == expected


@pytest.mark.parametrize(
    ("mom_ndim", "to", "expected"),
    [
        (1, "central", convert.raw_to_central),
        (1, "raw", convert.central_to_raw),
        (2, "central", convert_cov.raw_to_central),
        (2, "raw", convert_cov.central_to_raw),
    ],
)
def test_reduce_convert(mom_ndim: Mom_NDim, to: str, expected: factory.Convert) -> None:
    assert factory.factory_convert(mom_ndim, to) == expected


@pytest.mark.parametrize(
    ("mom_ndim", "parallel", "inverse", "expected"),
    [
        (1, False, False, push.cumulative),
        (1, True, False, push_parallel.cumulative),
        (2, False, False, push_cov.cumulative),
        (2, True, False, push_cov_parallel.cumulative),
        # inverse
        (1, False, True, push.cumulative_inverse),
        (1, True, True, push_parallel.cumulative_inverse),
        (2, False, True, push_cov.cumulative_inverse),
        (2, True, True, push_cov_parallel.cumulative_inverse),
    ],
)
def test_cumulative_data(
    mom_ndim: Mom_NDim, parallel: bool, inverse: bool, expected: factory.Convert
) -> None:
    assert factory.factory_cumulative(mom_ndim, parallel, inverse) == expected


@pytest.mark.parametrize(
    ("mom_ndim", "parallel", "expected"),
    [
        (1, False, moving.move_vals),
        (1, True, moving_parallel.move_vals),
        (2, False, moving_cov.move_vals),
        (2, True, moving_cov_parallel.move_vals),
    ],
)
def test_move_vals(
    mom_ndim: Mom_NDim,
    parallel: bool,
    expected: factory.MoveVals | factory.MoveValsCov,
) -> None:
    assert factory.factory_move_vals(mom_ndim, parallel) == expected


@pytest.mark.parametrize(
    ("mom_ndim", "parallel", "expected"),
    [
        (1, False, moving.move_data),
        (1, True, moving_parallel.move_data),
        (2, False, moving_cov.move_data),
        (2, True, moving_cov_parallel.move_data),
    ],
)
def test_move_data(
    mom_ndim: Mom_NDim,
    parallel: bool,
    expected: factory.MoveData,
) -> None:
    assert factory.factory_move_data(mom_ndim, parallel) == expected


@pytest.mark.parametrize(
    ("mom_ndim", "parallel", "expected"),
    [
        (1, False, moving.move_exp_vals),
        (1, True, moving_parallel.move_exp_vals),
        (2, False, moving_cov.move_exp_vals),
        (2, True, moving_cov_parallel.move_exp_vals),
    ],
)
def test_move_exp_vals(
    mom_ndim: Mom_NDim,
    parallel: bool,
    expected: factory.MoveExpVals | factory.MoveExpValsCov,
) -> None:
    assert factory.factory_move_exp_vals(mom_ndim, parallel) == expected


@pytest.mark.parametrize(
    ("mom_ndim", "parallel", "expected"),
    [
        (1, False, moving.move_exp_data),
        (1, True, moving_parallel.move_exp_data),
        (2, False, moving_cov.move_exp_data),
        (2, True, moving_cov_parallel.move_exp_data),
    ],
)
def test_move_exp_data(
    mom_ndim: Mom_NDim,
    parallel: bool,
    expected: factory.MoveExpData,
) -> None:
    assert factory.factory_move_exp_data(mom_ndim, parallel) == expected

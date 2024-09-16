from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cmomy._lib import (
    convert,
    convert_cov,
    indexed,
    indexed_cov,
    indexed_cov_parallel,
    indexed_parallel,
    push,
    push_cov,
    push_cov_parallel,
    push_parallel,
    resample,
    resample_cov,
    resample_cov_parallel,
    resample_parallel,
    rolling,
    rolling_cov,
    rolling_cov_parallel,
    rolling_parallel,
)
from cmomy.factory import (
    factory_convert,
    factory_cumulative,
    factory_jackknife_data,
    factory_jackknife_vals,
    factory_pusher,
    factory_reduce_data,
    factory_reduce_data_grouped,
    factory_reduce_data_indexed,
    factory_reduce_vals,
    factory_resample_data,
    factory_resample_vals,
    factory_rolling_data,
    factory_rolling_exp_data,
    factory_rolling_exp_vals,
    factory_rolling_vals,
    parallel_heuristic,
)

if TYPE_CHECKING:
    from typing import Any, Callable

    from cmomy import factory
    from cmomy.core.typing import Mom_NDim

    Func = Callable[..., Any]


# * parallel heuristic
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        # typical
        ((None, 100, 100), False),
        ((None, 101, 100), True),
        ((True, 100, 100), True),
        ((True, 101, 100), True),
        ((False, 100, 100), False),
        ((False, 101, 100), False),
        # no size
        ((None, None), False),
        ((None, None, 100), False),
        # no cutoff
        ((None, 1000), False),
        ((None, 10001), True),
    ],
)
def test_parallel_heuristic(args: tuple[Any, ...], expected: bool) -> None:
    assert parallel_heuristic(*args) == expected


@pytest.mark.parametrize(
    ("factory", "mom_ndim", "parallel", "expected"),
    [
        # pushers
        (
            factory_pusher,
            1,
            False,
            (
                push.push_val,
                push.reduce_vals,
                push.push_data,
                push.push_data_scale,
                push.reduce_data,
            ),
        ),
        (
            factory_pusher,
            1,
            True,
            (
                push_parallel.push_val,
                push_parallel.reduce_vals,
                push_parallel.push_data,
                push_parallel.push_data_scale,
                push_parallel.reduce_data,
            ),
        ),
        (
            factory_pusher,
            2,
            False,
            (
                push_cov.push_val,
                push_cov.reduce_vals,
                push_cov.push_data,
                push_cov.push_data_scale,
                push_cov.reduce_data,
            ),
        ),
        (
            factory_pusher,
            2,
            True,
            (
                push_cov_parallel.push_val,
                push_cov_parallel.reduce_vals,
                push_cov_parallel.push_data,
                push_cov_parallel.push_data_scale,
                push_cov_parallel.reduce_data,
            ),
        ),
        # resample_vals
        (factory_resample_vals, 1, False, resample.resample_vals),
        (factory_resample_vals, 1, True, resample_parallel.resample_vals),
        (factory_resample_vals, 2, False, resample_cov.resample_vals),
        (factory_resample_vals, 2, True, resample_cov_parallel.resample_vals),
        # resample_data
        (factory_resample_data, 1, False, resample.resample_data_fromzero),
        (
            factory_resample_data,
            1,
            True,
            resample_parallel.resample_data_fromzero,
        ),
        (factory_resample_data, 2, False, resample_cov.resample_data_fromzero),
        (
            factory_resample_data,
            2,
            True,
            resample_cov_parallel.resample_data_fromzero,
        ),
        # jacknife_vals
        (factory_jackknife_vals, 1, False, resample.jackknife_vals_fromzero),
        (
            factory_jackknife_vals,
            1,
            True,
            resample_parallel.jackknife_vals_fromzero,
        ),
        (
            factory_jackknife_vals,
            2,
            False,
            resample_cov.jackknife_vals_fromzero,
        ),
        (
            factory_jackknife_vals,
            2,
            True,
            resample_cov_parallel.jackknife_vals_fromzero,
        ),
        # jackknife_data
        (factory_jackknife_data, 1, False, resample.jackknife_data_fromzero),
        (
            factory_jackknife_data,
            1,
            True,
            resample_parallel.jackknife_data_fromzero,
        ),
        (
            factory_jackknife_data,
            2,
            False,
            resample_cov.jackknife_data_fromzero,
        ),
        (
            factory_jackknife_data,
            2,
            True,
            resample_cov_parallel.jackknife_data_fromzero,
        ),
        # reduce_vals
        (factory_reduce_vals, 1, False, push.reduce_vals),
        (factory_reduce_vals, 1, True, push_parallel.reduce_vals),
        (factory_reduce_vals, 2, False, push_cov.reduce_vals),
        (factory_reduce_vals, 2, True, push_cov_parallel.reduce_vals),
        # reduce_data
        (factory_reduce_data, 1, False, push.reduce_data_fromzero),
        (factory_reduce_data, 1, True, push_parallel.reduce_data_fromzero),
        (factory_reduce_data, 2, False, push_cov.reduce_data_fromzero),
        (factory_reduce_data, 2, True, push_cov_parallel.reduce_data_fromzero),
        # reduce_data_grouped
        (factory_reduce_data_grouped, 1, False, indexed.reduce_data_grouped),
        (
            factory_reduce_data_grouped,
            1,
            True,
            indexed_parallel.reduce_data_grouped,
        ),
        (
            factory_reduce_data_grouped,
            2,
            False,
            indexed_cov.reduce_data_grouped,
        ),
        (
            factory_reduce_data_grouped,
            2,
            True,
            indexed_cov_parallel.reduce_data_grouped,
        ),
        # reduce_data_indexed
        (
            factory_reduce_data_indexed,
            1,
            False,
            indexed.reduce_data_indexed_fromzero,
        ),
        (
            factory_reduce_data_indexed,
            1,
            True,
            indexed_parallel.reduce_data_indexed_fromzero,
        ),
        (
            factory_reduce_data_indexed,
            2,
            False,
            indexed_cov.reduce_data_indexed_fromzero,
        ),
        (
            factory_reduce_data_indexed,
            2,
            True,
            indexed_cov_parallel.reduce_data_indexed_fromzero,
        ),
        # rolling_vals
        (factory_rolling_vals, 1, False, rolling.rolling_vals),
        (factory_rolling_vals, 1, True, rolling_parallel.rolling_vals),
        (factory_rolling_vals, 2, False, rolling_cov.rolling_vals),
        (factory_rolling_vals, 2, True, rolling_cov_parallel.rolling_vals),
        # rolling_data
        (factory_rolling_data, 1, False, rolling.rolling_data),
        (factory_rolling_data, 1, True, rolling_parallel.rolling_data),
        (factory_rolling_data, 2, False, rolling_cov.rolling_data),
        (factory_rolling_data, 2, True, rolling_cov_parallel.rolling_data),
        # rolling_exp_vals
        (factory_rolling_exp_vals, 1, False, rolling.rolling_exp_vals),
        (factory_rolling_exp_vals, 1, True, rolling_parallel.rolling_exp_vals),
        (factory_rolling_exp_vals, 2, False, rolling_cov.rolling_exp_vals),
        (
            factory_rolling_exp_vals,
            2,
            True,
            rolling_cov_parallel.rolling_exp_vals,
        ),
        # rolling_exp_data
        (factory_rolling_exp_data, 1, False, rolling.rolling_exp_data),
        (factory_rolling_exp_data, 1, True, rolling_parallel.rolling_exp_data),
        (factory_rolling_exp_data, 2, False, rolling_cov.rolling_exp_data),
        (
            factory_rolling_exp_data,
            2,
            True,
            rolling_cov_parallel.rolling_exp_data,
        ),
    ],
)
def test_factory_general(
    factory: Any, mom_ndim: Mom_NDim, parallel: bool, expected: Any
) -> None:
    assert factory(mom_ndim, parallel) == expected


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
    assert factory_convert(mom_ndim, to) == expected


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
    assert factory_cumulative(mom_ndim, parallel, inverse) == expected

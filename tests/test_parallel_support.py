# mypy: disable-error-code="no-untyped-def, no-untyped-call, call-overload, assignment, arg-type"
# pyright: reportCallIssue=false, reportArgumentType=false
"""
Test that parallel=True and parallel=False give same answers...
"""

from functools import partial

import numpy as np
import pytest

import cmomy

from ._dataarray_set_utils import (
    do_reduce_data_grouped,
    do_reduce_data_indexed,
)

func_params_data_rolling = [
    (partial(cmomy.rolling.rolling_data, window=2), None),
    (partial(cmomy.rolling.rolling_exp_data, alpha=0.2), None),
]

func_params_data = [
    (cmomy.reduce_data, None),
    (do_reduce_data_grouped, None),
    (do_reduce_data_indexed, None),
    (partial(cmomy.resample_data, sampler={"nrep": 10, "rng": 0}), None),
    (cmomy.resample.jackknife_data, None),
    (cmomy.convert.cumulative, None),
    *func_params_data_rolling,
]


func_params_vals_rolling = [
    (partial(cmomy.rolling.rolling_vals, window=2), None),
    (partial(cmomy.rolling.rolling_exp_vals, alpha=0.2), None),
]


func_params_vals = [
    (cmomy.reduce_vals, None),
    (partial(cmomy.resample_vals, sampler={"nrep": 20, "rng": 0}), None),
    (cmomy.resample.jackknife_vals, None),
    *func_params_vals_rolling,
]


# fixture
@pytest.fixture
def data_and_kwargs(rng, request):
    shapes, kwargs = request.param
    if isinstance(shapes, list):
        data = [rng.random(s) for s in shapes]
    else:
        data = rng.random(shapes)
    return data, kwargs


# * Data
@pytest.mark.parametrize(
    ("func", "kwargs_callback"),
    func_params_data,
)
@pytest.mark.parametrize(
    "data_and_kwargs",
    [
        # shape, kwargs
        ((10, 3), {"axis": 0, "mom_ndim": 1}),
        ((10, 2, 3), {"axis": 0, "mom_ndim": 1}),
        ((10, 3, 3), {"axis": 0, "mom_ndim": 2}),
        ((10, 2, 3, 3), {"axis": 0, "mom_ndim": 2}),
    ],
    indirect=True,
)
def test_parallel_data(func, kwargs_callback, data_and_kwargs):
    data, kwargs = data_and_kwargs
    kws = kwargs_callback(kwargs.copy()) if kwargs_callback else kwargs
    np.testing.assert_allclose(*(func(data, **kws, parallel=p) for p in [False, True]))


@pytest.mark.parametrize(
    ("nrep", "ndat", "nsamp"),
    [
        (20, 10, None),
        (20, 10, 5),
        (100, 50, None),
    ],
)
def test_parallel_freq_to_indices(nrep, ndat, nsamp) -> None:
    indices = cmomy.resample.random_indices(nrep=nrep, ndat=ndat, nsamp=nsamp)
    freqs = [
        cmomy.resample.indices_to_freq(indices, ndat=ndat, parallel=p)
        for p in [True, False]
    ]
    np.testing.assert_equal(*freqs)

    idxs = [cmomy.resample.freq_to_indices(freqs[0], parallel=p) for p in [True, False]]
    np.testing.assert_equal(*idxs)

    # make sure round trip is ok
    np.testing.assert_allclose(np.sort(indices, axis=-1), idxs[0])


# had some weird stuff with rolling in parallel.  hammer it...
@pytest.mark.slow
@pytest.mark.parametrize(("func", "kwargs_callback"), func_params_data_rolling)
@pytest.mark.parametrize(
    "data_and_kwargs",
    [
        ((100, 2, 3), {"axis": 0, "mom_ndim": 1}),
        ((100, 2, 3, 3), {"axis": 0, "mom_ndim": 2}),
    ],
    indirect=True,
)
@pytest.mark.parametrize("repeat", range(10))
def test_parallel_data_rolling(func, kwargs_callback, data_and_kwargs, repeat):  # noqa: ARG001
    data, kwargs = data_and_kwargs
    kws = kwargs_callback(kwargs.copy()) if kwargs_callback else kwargs
    np.testing.assert_allclose(*(func(data, **kws, parallel=p) for p in [False, True]))


# * vals
@pytest.mark.parametrize(
    ("func", "kwargs_callback"),
    func_params_vals,
)
@pytest.mark.parametrize(
    "data_and_kwargs",
    [
        ([10], {"axis": 0, "mom": 2}),
        ([(10, 2)], {"axis": 0, "mom": 2}),
        ([(10, 3), (10,)], {"axis": 0, "mom": (2, 2)}),
        ([(10, 2, 3), (10, 2, 1)], {"axis": 0, "mom": (2, 2)}),
    ],
    indirect=True,
)
def test_parallel_vals(func, kwargs_callback, data_and_kwargs):
    xy, kwargs = data_and_kwargs
    kws = kwargs_callback(kwargs.copy()) if kwargs_callback else kwargs

    np.testing.assert_allclose(*(func(*xy, **kws, parallel=p) for p in [False, True]))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("func", "kwargs_callback"),
    func_params_vals,
)
@pytest.mark.parametrize(
    "data_and_kwargs",
    [
        # shapes, kwargs
        ([(100, 2)], {"axis": 0, "mom": 2}),
        ([(100, 2), (100, 2)], {"axis": 0, "mom": (2, 2)}),
    ],
    indirect=True,
)
@pytest.mark.parametrize("repeat", range(10))
def test_parallel_vals_rolling(func, kwargs_callback, data_and_kwargs, repeat):  # noqa: ARG001
    xy, kwargs = data_and_kwargs
    kws = kwargs_callback(kwargs.copy()) if kwargs_callback else kwargs

    np.testing.assert_allclose(*(func(*xy, **kws, parallel=p) for p in [False, True]))


# * Pushers
@pytest.mark.parametrize("method", ["push_data", "push_datas"])
@pytest.mark.parametrize(
    "data_and_kwargs",
    [
        ((20, 3), {"mom_ndim": 1}),
        ((20, 10, 3), {"mom_ndim": 1}),
        ((20, 3, 3), {"mom_ndim": 2}),
        ((20, 10, 3, 3), {"mom_ndim": 2}),
    ],
    indirect=True,
)
def test_parallel_push_data(method, data_and_kwargs):
    data, kwargs = data_and_kwargs
    mom_ndim = kwargs["mom_ndim"]

    shape = data.shape[1:]
    a, b = (
        cmomy.CentralMomentsArray(np.zeros(shape), mom_ndim=mom_ndim) for _ in range(2)
    )

    if method == "push_data":
        for d in data:
            a.push_data(d, parallel=False)
            b.push_data(d, parallel=True)

    else:
        a.push_datas(data, axis=0, parallel=False)
        b.push_datas(data, axis=0, parallel=True)

    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(a, cmomy.reduce_data(data, axis=0, mom_ndim=mom_ndim))


@pytest.mark.parametrize("method", ["push_data", "push_datas"])
@pytest.mark.parametrize(
    "data_and_kwargs",
    [
        ([20], {"mom": (2,)}),
        ([(20, 2)], {"mom": (2,)}),
        ([20, 20], {"mom": (2, 2)}),
        ([(20, 2), (20, 2)], {"mom": (2, 2)}),
    ],
    indirect=True,
)
def test_parallel_push_val(method, data_and_kwargs):
    data, kwargs = data_and_kwargs
    mom = kwargs["mom"]

    val_shape = data[0].shape[1:]
    a, b = (
        cmomy.CentralMomentsArray.zeros(mom=mom, val_shape=val_shape) for _ in range(2)
    )

    if method == "push_data":
        for d in zip(*data):
            a.push_val(*d, parallel=False)
            b.push_val(*d, parallel=True)

    else:
        a.push_vals(*data, axis=0, parallel=False)
        b.push_vals(*data, axis=0, parallel=True)

    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(a, cmomy.reduce_vals(*data, axis=0, mom=mom))

# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import xarray as xr

from cmomy import CentralMoments, resample
from cmomy.reduction import (
    resample_data_indexed,
)

# import cmomy
# import cmomy.indexed
# from cmomy import resample
# from cmomy.resample import (  # , xbootstrap_confidence_interval
#     bootstrap_confidence_interval,
#     freq_to_indices,
# )


@pytest.mark.parametrize("ndat", [50])
def test_freq_indices(ndat, rng) -> None:
    indices = rng.choice(ndat, (20, ndat), replace=True)

    freq0 = resample.indices_to_freq(indices)
    freq1 = resample.randsamp_freq(indices=indices, ndat=ndat)

    np.testing.assert_allclose(freq0, freq1)

    # round trip should be identical as well
    indices1 = resample.freq_to_indices(freq0)
    resample.indices_to_freq(indices1)

    np.testing.assert_allclose(freq0, freq1)

    freq0 = resample.randsamp_freq(nrep=10, ndat=ndat, rng=np.random.default_rng(123))

    freq1 = resample.randsamp_freq(nrep=10, ndat=ndat, rng=np.random.default_rng(456))
    assert not np.all(freq0 == freq1)

    freq1 = resample.randsamp_freq(nrep=10, ndat=ndat, rng=np.random.default_rng(123))
    np.testing.assert_allclose(freq0, freq1)

    # test bad freq
    freq = np.array([[5, 0], [0, 4]])
    with pytest.raises(ValueError, match="Inconsistent number of samples .*"):
        resample.freq_to_indices(freq)


parallel_parametrize = pytest.mark.parametrize(
    "parallel", [True, False]
)  # True, False])
fromzero_parametrize = pytest.mark.parametrize("fromzero", [False, True])


def test_central_randsamp_freq():
    c = CentralMoments.zeros(mom=4, val_shape=(10, 4))

    freq0 = resample.randsamp_freq(nrep=10, ndat=10, rng=np.random.default_rng(0))
    freq1 = c.randsamp_freq(nrep=10, axis=0, rng=np.random.default_rng(0))

    np.testing.assert_allclose(freq0, freq1)

    freq2 = resample.randsamp_freq(
        data=c.data, axis=0, rng=np.random.default_rng(0), nrep=10
    )
    np.testing.assert_allclose(freq0, freq2)

    # error if no ndat or data
    with pytest.raises(TypeError, match="Must pass .*"):
        resample.randsamp_freq()


def test_select_ndat() -> None:
    data = np.zeros((2, 3, 4, 5))

    assert resample.select_ndat(data, axis=0) == 2
    assert resample.select_ndat(data, axis=-1) == 5

    assert resample.select_ndat(data, axis=-1, mom_ndim=1) == 4
    assert resample.select_ndat(data, axis=-1, mom_ndim=2) == 3

    with pytest.raises(TypeError, match="Must specify .*"):
        resample.select_ndat(data)

    with pytest.raises(ValueError):
        resample.select_ndat(data, axis=2, mom_ndim=2)

    xdata = xr.DataArray(data)

    assert resample.select_ndat(xdata, dim="dim_0") == 2
    assert resample.select_ndat(xdata, axis=1) == 3

    with pytest.raises(ValueError):
        resample.select_ndat(xdata, dim="dim_2", mom_ndim=2)

    with pytest.raises(ValueError):
        resample.select_ndat(xdata)


@parallel_parametrize
@pytest.mark.parametrize("mom", [2, (2, 2)])
def test_resample_vec(parallel, mom, rng):
    x = rng.random((100, 10))
    xx = x[..., None]

    xy: tuple[Any, ...]
    xxyy: tuple[Any, ...]

    if isinstance(mom, tuple):
        xy = (x, x)
        xxyy = (xx, xx)
    else:
        xy = (x,)
        xxyy = (xx,)

    c1 = CentralMoments.from_vals(*xy, axis=0, mom=mom)
    c2 = CentralMoments.from_vals(*xxyy, axis=0, mom=mom)

    np.testing.assert_allclose(c1.data, c2.data[:, 0, ...])

    freq = c1.randsamp_freq(nrep=10, axis=0)

    cc1 = c1.resample_and_reduce(
        freq=freq,
        parallel=parallel,
        axis=0,
    )
    cc2 = c2.resample_and_reduce(
        freq=freq,
        parallel=parallel,
        axis=0,
    )

    np.testing.assert_allclose(cc1.data, cc2.data[0, ...])

    # using indexed
    out1 = resample_data_indexed(
        c1.data, freq=freq, mom_ndim=c1.mom_ndim, parallel=parallel, axis=0
    )
    np.testing.assert_allclose(cc1.data, out1)

    out2 = resample_data_indexed(
        c2.data, freq=freq, mom_ndim=c2.mom_ndim, parallel=parallel, axis=0
    )
    np.testing.assert_allclose(cc2.data, out2)


def test_resample_indices(rng) -> None:
    indices = resample.random_indices(nrep=5, ndat=10, rng=rng)

    freq = resample.indices_to_freq(indices)

    for shuffle in [True, False]:
        idx = resample.freq_to_indices(freq, shuffle=shuffle)
        np.testing.assert_allclose(freq, resample.indices_to_freq(idx))


def test_validate_resample_array() -> None:
    np.zeros((2, 3, 4))

    with pytest.raises(ValueError):
        resample._validate_resample_array(
            np.zeros((2, 3, 4)), nrep=2, ndat=3, is_freq=True
        )

    with pytest.raises(ValueError):
        resample._validate_resample_array(
            np.zeros((2, 3)), nrep=3, ndat=3, is_freq=True
        )

    with pytest.raises(ValueError):
        resample._validate_resample_array(
            np.zeros((2, 3)), nrep=2, ndat=5, is_freq=True
        )

    # indices
    _ = resample._validate_resample_array(
        np.zeros((2, 3)), nrep=2, ndat=5, is_freq=False
    )

    with pytest.raises(ValueError):
        resample._validate_resample_array(
            np.zeros((2, 3)) + 10, nrep=2, ndat=5, is_freq=False
        )


def test_randsamp_freq() -> None:
    f0 = resample.randsamp_freq(nrep=10, ndat=5, rng=np.random.default_rng(0))

    f1 = resample.random_freq(nrep=10, ndat=5, rng=np.random.default_rng(0))

    np.testing.assert_allclose(f0, f1)

    with pytest.raises(ValueError):
        resample.randsamp_freq(ndat=10)


def test_randsamp_freq_2() -> None:
    nrep = 200
    ndat = 100

    freq = resample.randsamp_freq(nrep=nrep, ndat=ndat)

    np.testing.assert_allclose(
        freq, resample.randsamp_freq(ndat=ndat, nrep=nrep, freq=freq, check=True)
    )

    # bad ndata
    with pytest.raises(ValueError, match=".*has wrong ndat.*"):
        _ = resample.randsamp_freq(ndat=10, nrep=nrep, freq=freq, check=True)


def test_resample_resample_data(rng) -> None:
    x = rng.random((100, 10, 3))

    c = CentralMoments.from_vals(x, mom=3, axis=0)

    freq = c.randsamp_freq(nrep=5, axis=0, rng=rng)

    with pytest.raises(ValueError):
        out = resample.resample_data(
            c.data, freq=freq, mom_ndim=1, out=np.zeros((10, 3, 4)), axis=0
        )

    c2 = c.resample_and_reduce(freq=freq, axis=0)

    v = c.moveaxis(0, 1).data

    out = resample.resample_data(
        v, freq=freq, mom_ndim=1, axis=-1, out=np.zeros((3, 5, 4)), dtype=np.float64
    )

    np.testing.assert_allclose(c2.data, out)

    out = resample_data_indexed(
        v, freq=freq, mom_ndim=1, axis=-1, out=np.zeros((3, 5, 4)), dtype=np.float64
    )
    np.testing.assert_allclose(c2.data, out)

    with pytest.raises(ValueError):
        resample.resample_data(c.data, freq=freq, mom_ndim=1, axis=1)


def test_resample_resample_vals(rng) -> None:
    x = rng.random((10, 3))

    freq = resample.random_freq(nrep=5, ndat=10, rng=rng)

    c = CentralMoments.from_resample_vals(x, freq=freq, mom=3, axis=0)

    out = resample.resample_vals(
        x,
        freq=freq,
        mom=3,
        axis=0,
    )
    np.testing.assert_allclose(c.data, out)

    with pytest.raises(TypeError):
        resample.resample_vals(x, freq=freq, mom=[3])  # type: ignore[call-overload]

    with pytest.raises(ValueError):
        resample.resample_vals(x, freq=freq, mom=(3, 3, 3), axis=0)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        resample.resample_vals(x, freq=freq[:, :-1], mom=3, axis=0)

    out = np.zeros((3, 5, 4))
    _ = resample.resample_vals(x, freq=freq, mom=3, out=out, axis=0)
    np.testing.assert_allclose(c.data, out)

    out = np.zeros((4, 4, 4))
    with pytest.raises(ValueError):
        resample.resample_vals(x, freq=freq, mom=3, out=out, axis=0)

    c2 = CentralMoments.from_resample_vals(x, x, freq=freq, mom=(3, 3), axis=0)

    np.testing.assert_allclose(c2.data[..., :, 0], c.data)


@pytest.mark.slow()
@parallel_parametrize
def test_resample_vals(other, parallel) -> None:
    # test basic resampling
    if other.style == "total":
        datar = resample.resample_vals(
            *other.xy_tuple,
            mom=other.mom,
            freq=other.freq,
            axis=other.axis,
            weight=other.w,
            parallel=parallel,
        )

        np.testing.assert_allclose(datar, other.data_test_resamp)


@pytest.mark.slow()
@parallel_parametrize
def test_stats_resample_vals(other, parallel) -> None:
    if other.style == "total":
        t = other.cls.from_resample_vals(
            *other.xy_tuple,
            weight=other.w,
            mom=other.mom,
            freq=other.freq,
            axis=other.axis,
            parallel=parallel,
            order="C",
        )
        np.testing.assert_allclose(t.data, other.data_test_resamp)


@pytest.mark.slow()
@parallel_parametrize
def test_resample_data(other, parallel, rng) -> None:
    nrep = 10

    if len(other.val_shape) > 0:
        for axis in range(other.s.val_ndim):
            data = other.data_test

            ndat = data.shape[axis]

            idx = rng.choice(ndat, (nrep, ndat), replace=True)
            freq = resample.randsamp_freq(indices=idx, ndat=ndat)

            if axis != 0:
                data = np.rollaxis(data, axis, 0)
            data = np.take(data, idx, axis=0)
            data_ref = (
                other.cls(data, mom_ndim=other.mom_ndim).reduce(axis=1).moveaxis(0, -1)
            )

            t = other.s.resample_and_reduce(
                freq=freq,
                axis=axis,
                parallel=parallel,
            )
            np.testing.assert_allclose(data_ref, t.data)

            # indexed
            out = resample_data_indexed(
                other.s.data,
                freq=freq,
                mom_ndim=other.s.mom_ndim,
                axis=axis,
                parallel=parallel,
            )
            np.testing.assert_allclose(data_ref, out)


@pytest.mark.slow()
@parallel_parametrize
def test_resample_against_vals(other, parallel, rng) -> None:
    nrep = 10

    if len(other.val_shape) > 0:
        s = other.s

        for axis in range(s.val_ndim):
            ndat = s.val_shape[axis]
            idx = rng.choice(ndat, (nrep, ndat), replace=True)
            freq = resample.indices_to_freq(idx)

            t0 = s.resample_and_reduce(
                freq=freq,
                axis=axis,
                parallel=parallel,
            )

            t1 = s.resample(idx, axis=axis, last=True).reduce(axis=-1)

            np.testing.assert_allclose(t0.to_values(), t1.to_values())


def test_resample_zero_weight(rng) -> None:
    freq_zero = np.zeros((10, 10), dtype=int)
    c = CentralMoments.zeros(mom=(2, 2), val_shape=(10, 2))
    c._data[...] = 10.0

    c2 = c.resample_and_reduce(freq=freq_zero, axis=0)
    np.testing.assert_allclose(c2.data, 0.0)

    x = rng.random((10, 10, 2))

    c = CentralMoments.from_vals(x, x, mom=(2, 2), axis=0)

    c2 = c.resample_and_reduce(freq=freq_zero, axis=0)

    np.testing.assert_allclose(c2.data, 0.0)

    # indexed:
    out = resample_data_indexed(c.data, freq=freq_zero, mom_ndim=2, axis=0)
    np.testing.assert_allclose(c2.data, out)


@pytest.mark.slow()
def test_bootstrap_stats(other) -> None:
    x = other.xdata
    axis = other.axis
    alpha = 0.05

    # test styles
    test = resample.bootstrap_confidence_interval(
        x, stats_val=None, axis=axis, alpha=alpha
    )

    p_low = 100 * (alpha / 2.0)
    p_mid = 50
    p_high = 100 - p_low

    expected = np.percentile(x, [p_mid, p_low, p_high], axis=axis)
    np.testing.assert_allclose(test, expected)

    # 'mean'
    test = resample.bootstrap_confidence_interval(
        x, stats_val="mean", axis=axis, alpha=alpha
    )

    q_high = 100 * (alpha / 2.0)
    q_low = 100 - q_high
    stats_val = x.mean(axis=axis)
    val = stats_val
    low = 2 * stats_val - np.percentile(a=x, q=q_low, axis=axis)
    high = 2 * stats_val - np.percentile(a=x, q=q_high, axis=axis)

    expected = np.array([val, low, high])
    np.testing.assert_allclose(test, expected)


# # * Arbitrary number of samples in resample.


@parallel_parametrize
def test_resample_nsamp(other, parallel) -> None:
    nrep = 10

    if len(other.val_shape) > 0:
        s = other.s

        for axis in range(s.val_ndim):
            ndat = s.val_shape[axis]

            for nsamp in [ndat + 1, ndat - 1]:
                indices = resample.random_indices(nrep=nrep, ndat=ndat, nsamp=nsamp)
                freq = resample.indices_to_freq(indices, ndat=ndat)

                t0 = s.resample_and_reduce(
                    freq=freq,
                    axis=axis,
                    parallel=parallel,
                )

                t1 = s.resample(indices, axis=axis, last=True).reduce(axis=-1)
                np.testing.assert_allclose(t0.to_values(), t1.to_values())

                # test indexed resample

                out = resample_data_indexed(
                    s.data, freq=freq, mom_ndim=s.mom_ndim, axis=axis, parallel=parallel
                )
                np.testing.assert_allclose(t0.to_values(), out)

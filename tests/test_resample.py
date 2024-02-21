# mypy: disable-error-code="no-untyped-def, no-untyped-call"
import numpy as np
import pytest

import cmomy
from cmomy import resample
from cmomy.resample import (  # , xbootstrap_confidence_interval
    bootstrap_confidence_interval,
)


@pytest.mark.parametrize("ndat", [50])
def test_freq_indices(ndat, rng) -> None:
    indices = rng.choice(10, (20, 10), replace=True)

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


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("mom", [2, (2, 2)])
def test_resample_vec(parallel, mom):
    rng = cmomy.random.default_rng()

    x = rng.random((100, 10))
    xx = x[..., None]

    if isinstance(mom, tuple):
        c1 = cmomy.CentralMoments.from_vals((x, x), axis=0, mom=mom)
        c2 = cmomy.CentralMoments.from_vals((xx, xx), axis=0, mom=mom)
    else:
        c1 = cmomy.CentralMoments.from_vals(x, axis=0, mom=mom)
        c2 = cmomy.CentralMoments.from_vals(xx, axis=0, mom=mom)

    np.testing.assert_allclose(c1.data, c2.data[:, 0, ...])

    freq = cmomy.resample.randsamp_freq(nrep=10, ndat=10)

    cc1 = c1.resample_and_reduce(
        freq=freq, parallel=parallel, resample_kws={"order": "C"}
    )
    cc2 = c2.resample_and_reduce(
        freq=freq, parallel=parallel, resample_kws={"order": "C"}
    )

    np.testing.assert_allclose(cc1.data, cc2.data[:, 0, ...])


def test_resample_indices(rng) -> None:
    indices = cmomy.resample.random_indices(nrep=5, ndat=10, rng=rng)

    freq = cmomy.resample.indices_to_freq(indices)

    for shuffle in [True, False]:
        idx = cmomy.resample.freq_to_indices(freq, shuffle=shuffle)
        np.testing.assert_allclose(freq, cmomy.resample.indices_to_freq(idx))


def test_validate_resample_array() -> None:
    np.zeros((2, 3, 4))

    with pytest.raises(ValueError):
        cmomy.resample._validate_resample_array(np.zeros((2, 3, 4)), nrep=2, ndat=3)

    with pytest.raises(ValueError):
        cmomy.resample._validate_resample_array(np.zeros((2, 3)), nrep=3, ndat=3)

    with pytest.raises(ValueError):
        cmomy.resample._validate_resample_array(np.zeros((2, 3)), nrep=2, ndat=None)

    with pytest.raises(ValueError):
        cmomy.resample._validate_resample_array(np.zeros((2, 3)), nrep=2, ndat=5)


def test_randsamp_freq() -> None:
    f0 = cmomy.resample.randsamp_freq(nrep=10, ndat=5, rng=np.random.default_rng(0))

    f1 = cmomy.resample.random_freq(nrep=10, ndat=5, rng=np.random.default_rng(0))

    np.testing.assert_allclose(f0, f1)

    with pytest.raises(ValueError):
        cmomy.resample.randsamp_freq()


def test_resample_resample_data(rng) -> None:
    x = rng.random((100, 10, 3))

    c = cmomy.CentralMoments.from_vals(x, mom=3)

    freq = cmomy.resample.random_freq(nrep=5, ndat=10, rng=rng)

    with pytest.raises(ValueError):
        out = cmomy.resample.resample_data(
            c.data, freq=freq, mom=3, out=np.zeros((10, 3, 4))
        )

    c2 = c.resample_and_reduce(freq=freq)

    v = np.moveaxis(c.data, 0, 1)

    out = cmomy.resample.resample_data(
        v, freq=freq, mom=3, axis=-1, out=np.zeros((5, 3, 4)), dtype=np.float64
    )

    np.testing.assert_allclose(c2.data, out)

    with pytest.raises(ValueError):
        cmomy.resample.resample_data(c.data, freq=freq, mom=4, axis=0)


def test_resample_resample_vals(rng) -> None:
    x = rng.random((10, 3))

    freq = cmomy.resample.random_freq(nrep=5, ndat=10, rng=rng)

    c = cmomy.CentralMoments.from_resample_vals(x, freq=freq, mom=3)

    out = cmomy.resample.resample_vals(
        x, freq=freq, mom=3, mom_ndim=1, dtype=np.float64
    )
    np.testing.assert_allclose(c.data, out)

    with pytest.raises(TypeError):
        cmomy.resample.resample_vals(x, freq=freq, mom=[3])  # type: ignore[arg-type]  # this is on purpose for testing

    with pytest.raises(ValueError):
        cmomy.resample.resample_vals(x, freq=freq, mom=3, mom_ndim=2)

    with pytest.raises(ValueError):
        cmomy.resample.resample_vals(x, freq=freq, mom=(3, 3, 3), mom_ndim=3)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        cmomy.resample.resample_vals(x, freq=freq[:, :-1], mom=3, mom_ndim=1)

    out = np.zeros((5, 3, 4))
    _ = cmomy.resample.resample_vals(x, freq=freq, mom=3, out=out)
    np.testing.assert_allclose(c.data, out)

    out = np.zeros((4, 4, 4))
    with pytest.raises(ValueError):
        cmomy.resample.resample_vals(x, freq=freq, mom=3, out=out)

    c2 = cmomy.CentralMoments.from_resample_vals((x, x), freq=freq, mom=(3, 3))

    np.testing.assert_allclose(c2.data[..., :, 0], c.data)


@pytest.mark.slow()
@pytest.mark.parametrize("parallel", [True, False])
def test_resample_vals(other, parallel) -> None:
    # test basic resampling
    if other.style == "total":
        datar = resample.resample_vals(
            x=other.x,
            mom=other.mom,
            freq=other.freq,
            axis=other.axis,
            w=other.w,
            mom_ndim=other.s._mom_ndim,
            broadcast=other.broadcast,
            parallel=parallel,
        )

        np.testing.assert_allclose(datar, other.data_test_resamp)


@pytest.mark.slow()
@pytest.mark.parametrize("parallel", [True, False])
def test_stats_resample_vals(other, parallel) -> None:
    if other.style == "total":
        t = other.cls.from_resample_vals(
            x=other.x,
            w=other.w,
            mom=other.mom,
            freq=other.freq,
            axis=other.axis,
            broadcast=other.broadcast,
            parallel=parallel,
            resample_kws={"order": "C"},
        )
        np.testing.assert_allclose(t.data, other.data_test_resamp)

        # test based on indices
        t = other.cls.from_resample_vals(
            x=other.x,
            w=other.w,
            mom=other.mom,
            indices=other.indices,
            axis=other.axis,
            broadcast=other.broadcast,
            parallel=parallel,
        )
        np.testing.assert_allclose(t.data, other.data_test_resamp)


@pytest.mark.slow()
@pytest.mark.parametrize("parallel", [True, False])
def test_resample_data(other, parallel, rng) -> None:
    nrep = 10

    if len(other.val_shape) > 0:
        for axis in range(other.s.val_ndim):
            data = other.data_test

            ndat = data.shape[axis]

            idx = rng.choice(ndat, (nrep, ndat), replace=True)
            freq = resample.randsamp_freq(indices=idx)

            if axis != 0:
                data = np.rollaxis(data, axis, 0)
            data = np.take(data, idx, axis=0)
            data_ref = other.cls.from_datas(data, mom_ndim=other.mom_ndim, axis=1)

            t = other.s.resample_and_reduce(freq=freq, axis=axis, parallel=parallel)
            np.testing.assert_allclose(data_ref, t.data)


@pytest.mark.slow()
@pytest.mark.parametrize("parallel", [True, False])
def test_resample_against_vals(other, parallel, rng) -> None:
    nrep = 10

    if len(other.val_shape) > 0:
        s = other.s

        for axis in range(s.val_ndim):
            ndat = s.val_shape[axis]
            idx = rng.choice(ndat, (nrep, ndat), replace=True)

            t0 = s.resample_and_reduce(indices=idx, axis=axis, parallel=parallel)

            t1 = s.resample(idx, axis=axis).reduce(1)

            np.testing.assert_allclose(t0.to_values(), t1.to_values())


def test_resample_zero_weight() -> None:
    import cmomy

    freq = np.zeros((10, 10), dtype=int)
    c = cmomy.CentralMoments.zeros(mom=(2, 2), val_shape=(10, 2))

    c2 = c.resample_and_reduce(nrep=10, axis=0)

    np.testing.assert_allclose(c2.data, 0.0)

    rng = cmomy.random.default_rng()

    x = rng.random((10, 10, 2))

    c = cmomy.CentralMoments.from_vals((x, x), mom=(2, 2))

    c2 = c.resample_and_reduce(freq=freq)

    np.testing.assert_allclose(c2.data, 0.0)


@pytest.mark.slow()
def test_bootstrap_stats(other) -> None:
    x = other.xdata
    axis = other.axis
    alpha = 0.05

    # test styles
    test = bootstrap_confidence_interval(x, stats_val=None, axis=axis, alpha=alpha)

    p_low = 100 * (alpha / 2.0)
    p_mid = 50
    p_high = 100 - p_low

    expected = np.percentile(x, [p_mid, p_low, p_high], axis=axis)
    np.testing.assert_allclose(test, expected)

    # 'mean'
    test = bootstrap_confidence_interval(x, stats_val="mean", axis=axis, alpha=alpha)

    q_high = 100 * (alpha / 2.0)
    q_low = 100 - q_high
    stats_val = x.mean(axis=axis)
    val = stats_val
    low = 2 * stats_val - np.percentile(a=x, q=q_low, axis=axis)
    high = 2 * stats_val - np.percentile(a=x, q=q_high, axis=axis)

    expected = np.array([val, low, high])
    np.testing.assert_allclose(test, expected)

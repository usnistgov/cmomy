# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import xarray as xr

import cmomy
from cmomy import CentralMoments, resample
from cmomy.reduction import (
    resample_data_indexed,
)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from cmomy.typing import Mom_NDim


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
    x = rng.random((50, 10))
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

    np.testing.assert_allclose(cc1.data, cc2.data[:, 0, ...])

    # using indexed
    out1 = resample_data_indexed(
        c1.data, freq=freq, mom_ndim=c1.mom_ndim, parallel=parallel, axis=0
    )
    np.testing.assert_allclose(
        cc1.data,
        out1,
    )

    out2 = resample_data_indexed(
        c2.data, freq=freq, mom_ndim=c2.mom_ndim, parallel=parallel, axis=0
    )
    np.testing.assert_allclose(
        cc2.data,
        out2,
    )


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

    np.testing.assert_allclose(c2.moveaxis(0, -1).data, out)

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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
@parallel_parametrize
def test_resample_data(other, parallel, rng) -> None:
    nrep = 10

    if len(other.val_shape) > 0:
        for axis in range(other.s.val_ndim):
            data = other.data_test

            ndat = data.shape[axis]

            idx = rng.choice(ndat, (nrep, ndat), replace=True)
            freq = resample.randsamp_freq(indices=idx, ndat=ndat)

            data = np.take(data, idx, axis=axis)
            data_ref = other.cls(data, mom_ndim=other.mom_ndim).reduce(axis=axis + 1)

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
            np.testing.assert_allclose(
                data_ref,
                out,
            )


@pytest.mark.slow
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

            t1 = s.resample(idx, axis=axis, last=False).reduce(axis=axis + 1)

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
    np.testing.assert_allclose(
        c2.data,
        out,
    )


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

                t1 = s.resample(indices, axis=axis, last=False).reduce(axis=axis + 1)
                np.testing.assert_allclose(t0.to_values(), t1.to_values())

                # test indexed resample

                out = resample_data_indexed(
                    s.data, freq=freq, mom_ndim=s.mom_ndim, axis=axis, parallel=parallel
                )
                np.testing.assert_allclose(
                    t0.to_values(),
                    out,
                )


@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((10, 2, 4, 4), 0),
        ((2, 10, 4, 4), 1),
    ],
)
@pytest.mark.parametrize("mom_ndim", [1, 2])
def test_jackknife_data(rng, mom_ndim, shape, axis) -> None:
    data = rng.random(shape)
    freq = resample.jackknife_freq(data.shape[axis])

    out0 = resample.resample_data(data, mom_ndim=mom_ndim, freq=freq, axis=axis)
    out1 = resample.jackknife_data(data, mom_ndim=mom_ndim, axis=axis)
    np.testing.assert_allclose(
        out0,
        out1,
    )

    # using calculated data_reduced
    data_reduced = cmomy.reduce_data(data, mom_ndim=mom_ndim, axis=axis)
    out1 = resample.jackknife_data(
        data,
        mom_ndim=mom_ndim,
        axis=axis,
        data_reduced=data_reduced,
    )
    np.testing.assert_allclose(
        out0,
        out1,
    )

    # make sure we're actually using data_reduced
    out1 = resample.jackknife_data(
        data, mom_ndim=mom_ndim, axis=axis, data_reduced=np.zeros_like(data_reduced)
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(out0, out1)

    # dataarray
    xdata = xr.DataArray(data, attrs={"hello": "there"})
    dim = xdata.dims[axis]

    xout0 = resample.resample_data(xdata, mom_ndim=mom_ndim, freq=freq, dim=dim)
    np.testing.assert_allclose(out0, xout0)

    xout1 = resample.jackknife_data(xdata, mom_ndim=mom_ndim, dim=dim)
    xr.testing.assert_allclose(xout0, xout1)

    # using data_reduced
    xout1 = resample.jackknife_data(
        xdata, mom_ndim=mom_ndim, axis=axis, data_reduced=data_reduced
    )
    xr.testing.assert_allclose(xout0, xout1)

    xdata_reduced = cmomy.reduce_data(xdata, mom_ndim=mom_ndim, dim=dim)
    xout1 = resample.jackknife_data(
        xdata, mom_ndim=mom_ndim, dim=dim, data_reduced=xdata_reduced
    )
    xr.testing.assert_allclose(xout0, xout1)

    # keep_attrs
    xout0 = resample.resample_data(
        xdata, mom_ndim=mom_ndim, freq=freq, dim=dim, keep_attrs=True
    )
    np.testing.assert_allclose(out0, xout0)

    xout1 = resample.jackknife_data(xdata, mom_ndim=mom_ndim, dim=dim, keep_attrs=True)
    xr.testing.assert_allclose(xout0, xout1)


@pytest.mark.parametrize(("shape", "axis"), [((10, 4, 4), 0)])
@pytest.mark.parametrize("mom_ndim", [1, 2])
@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_jackknife_data_extras(mom_ndim, shape, axis, parallel, dtype):
    rng = np.random.default_rng(0)
    data = rng.random(shape)

    # using out parameter
    out0 = resample.resample_data(
        data,
        mom_ndim=mom_ndim,
        freq=resample.jackknife_freq(data.shape[axis]),
        axis=axis,
        dtype=dtype,
    )

    assert out0.dtype.type == dtype

    data_reduced = cmomy.reduce_data(data, axis=axis, mom_ndim=mom_ndim, dtype=dtype)

    out_ = np.zeros_like(out0, dtype=dtype)
    out1 = resample.jackknife_data(
        data,
        mom_ndim=mom_ndim,
        axis=axis,
        parallel=parallel,
        out=out_,
        data_reduced=data_reduced.tolist(),
    )

    assert out_ is out1
    np.testing.assert_allclose(out1, out0, rtol=1e-5)
    assert out1.dtype.type == dtype

    xdata = xr.DataArray(data)
    xout1 = resample.jackknife_data(
        xdata, mom_ndim=mom_ndim, axis=axis, rep_dim=None, dtype=dtype
    )

    np.testing.assert_allclose(xout1, out0, rtol=1e-5)
    assert xout1.dtype.type == dtype
    assert xout1.dims == xdata.dims


@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((50,), 0),
        ((50, 2), 0),
        ((2, 50), 1),
    ],
)
@pytest.mark.parametrize("mom", [(3,), (3, 3)])
@pytest.mark.parametrize("use_weight", [True, False])
def test_jackknife_vals(rng, shape, axis, mom, use_weight) -> None:
    x = rng.random(shape)

    xy = (x,) if len(mom) == 1 else (x, x)
    freq = resample.jackknife_freq(x.shape[axis])

    weight = rng.random(shape) if use_weight else None

    out0 = resample.resample_vals(*xy, freq=freq, mom=mom, weight=weight, axis=axis)
    out1 = resample.jackknife_vals(*xy, mom=mom, weight=weight, axis=axis)
    np.testing.assert_allclose(out0, out1)

    data_reduced = cmomy.reduce_vals(*xy, mom=mom, weight=weight, axis=axis)
    out1 = resample.jackknife_vals(
        *xy, mom=mom, weight=weight, axis=axis, data_reduced=data_reduced
    )
    np.testing.assert_allclose(out0, out1)

    for dxy in [
        tuple(xr.DataArray(_) for _ in xy),
        (xr.DataArray(xy[0]), *(_ for _ in xy[1:])),
    ]:
        xout1 = resample.jackknife_vals(*dxy, mom=mom, weight=weight, axis=axis)
        np.testing.assert_allclose(out0, xout1)


@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((50,), 0),
    ],
)
@pytest.mark.parametrize("mom", [(3,)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_jackknife_vals_extras(shape, axis, mom: Mom_NDim, dtype: DTypeLike) -> None:
    rng = np.random.default_rng(0)
    x = rng.random(shape)
    xy = (x,)

    freq = resample.jackknife_freq(x.shape[axis])

    out0 = resample.resample_vals(*xy, freq=freq, mom=mom, axis=axis, dtype=dtype)
    assert out0.dtype.type == dtype

    data_reduced = cmomy.reduce_vals(*xy, mom=mom, axis=axis, dtype=dtype)
    out1 = resample.jackknife_vals(
        *xy,
        mom=mom,
        axis=axis,
        data_reduced=data_reduced.tolist(),
        dtype=dtype,
    )
    np.testing.assert_allclose(out0, out1, rtol=1e-5)
    assert out0.dtype.type == dtype

    # using out
    out_ = np.zeros_like(out0, dtype=dtype)
    out1 = resample.jackknife_vals(*xy, mom=mom, axis=axis, out=out_)
    np.testing.assert_allclose(out0, out1, rtol=1e-5)
    assert out1 is out_

    # wrong data_reduced shape...
    with pytest.raises(ValueError, match=".* inconsistent with.*"):
        out1 = resample.jackknife_vals(
            *xy,
            mom=mom,
            axis=axis,
            data_reduced=np.zeros((50, 2)),
        )

    dx = xr.DataArray(x)
    xout1 = resample.jackknife_vals(dx, mom=mom, dim="dim_0", rep_dim=None, dtype=dtype)
    np.testing.assert_allclose(out1, xout1)
    assert xout1.dtype.type == dtype
    assert xout1.dims == ("dim_0", "mom_0")

    # inherit mom_dims from data_reduced

    xdata_reduced = xr.DataArray(data_reduced, dims=["hello_moment"])

    xout1 = resample.jackknife_vals(
        dx,
        mom=mom,
        dim="dim_0",
        mom_dims="passed_moment",
        data_reduced=xdata_reduced,
        dtype=dtype,
    )
    np.testing.assert_allclose(xout1, out1)
    assert xout1.dims == ("rep", "hello_moment")

    xout1 = resample.jackknife_vals(
        dx,
        mom=mom,
        dim="dim_0",
        mom_dims="passed_moment",
        data_reduced=data_reduced,
        dtype=dtype,
    )
    np.testing.assert_allclose(xout1, out1)
    assert xout1.dims == ("rep", "passed_moment")

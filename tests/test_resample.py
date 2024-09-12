# mypy: disable-error-code="no-untyped-def, no-untyped-call, assignment, arg-type, call-overload"
# pyright: reportCallIssue=false, reportArgumentType=false
"""
Test basics of resampling.

Just testing cmomy.resample module for numpy arrays

Test xarray in test_xarray_support
Test parallel in test_parallel_support
Test dtypes in test_dtype_support

Test wrapped object elsewhere...
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest
import xarray as xr

import cmomy
from cmomy import CentralMoments, resample
from cmomy.core.validate import is_dataset, is_xarray


# * Main tests
@pytest.fixture
def data_and_kwargs(rng, request):
    shapes, kwargs = request.param
    if isinstance(shapes, list):
        data = [rng.random(s) for s in shapes]
    else:
        data = rng.random(shapes)
    return data, kwargs


def unpack_data_to_xy_weight(data, mom):
    if len(data) == len(mom):
        return data, None
    if len(data) == len(mom) + 1:
        return data[:-1], data[-1]
    msg = f"bad unpack: {len(data)=}, {len(mom)=}"
    raise ValueError(msg)


data_params = [
    ((20, 4), {"mom_ndim": 1, "axis": 0}),
    ((2, 20, 4), {"mom_ndim": 1, "axis": 0}),
    ((20, 3, 3), {"mom_ndim": 2, "axis": 0}),
    ((20, 2, 3, 3), {"mom_ndim": 2, "axis": 0}),
]

vals_params = [
    ([20], {"mom": (3,), "axis": 0}),
    ([20, 20], {"mom": (3,), "axis": 0}),
    ([(20, 2)], {"mom": (3,), "axis": 0}),
    ([20, 20], {"mom": (3, 3), "axis": 0}),
    ([(20, 2), (20, 2), (20, 2)], {"mom": (3, 3), "axis": 0}),
]


# ** resample
@pytest.mark.parametrize("data_and_kwargs", data_params, indirect=True)
@pytest.mark.parametrize("nrep", [10])
def test_resample_data_0(data_and_kwargs, nrep):
    data, kwargs = data_and_kwargs
    axis, mom_ndim = (kwargs[k] for k in ("axis", "mom_ndim"))
    ndat = cmomy.resample.select_ndat(data, axis=axis)
    indices = cmomy.random_indices(nrep, ndat, rng=123)

    # expected from reduction
    expected = cmomy.reduce_data(
        data.take(indices, axis=axis), mom_ndim=mom_ndim, axis=axis + 1
    )

    # using freq
    freq = cmomy.resample.indices_to_freq(indices)
    np.testing.assert_allclose(
        cmomy.resample_data(data, freq=freq, axis=axis, mom_ndim=mom_ndim),
        expected,
    )

    # using same rng
    np.testing.assert_allclose(
        cmomy.resample_data(data, rng=123, nrep=nrep, axis=axis, mom_ndim=mom_ndim),
        expected,
    )

    # against using reduce_data_indexed....
    np.testing.assert_allclose(
        cmomy.reduction.resample_data_indexed(
            data, freq=freq, axis=axis, mom_ndim=mom_ndim
        ),
        expected,
    )


@pytest.mark.parametrize("data_and_kwargs", vals_params, indirect=True)
@pytest.mark.parametrize("nrep", [10])
def test_resample_vals_0(data_and_kwargs, nrep):
    data, kwargs = data_and_kwargs
    axis, mom = (kwargs[k] for k in ("axis", "mom"))
    ndat = cmomy.resample.select_ndat(data[0], axis=axis)
    indices = cmomy.random_indices(nrep, ndat, rng=123)

    # expected
    data_take = [d.take(indices, axis=axis) for d in data]
    xy_take, weight_take = unpack_data_to_xy_weight(data_take, mom)
    expected = cmomy.reduce_vals(*xy_take, weight=weight_take, mom=mom, axis=axis + 1)

    xy, weight = unpack_data_to_xy_weight(data, mom)
    freq = cmomy.resample.indices_to_freq(indices)
    np.testing.assert_allclose(
        cmomy.resample_vals(
            *xy, freq=freq, weight=weight, **kwargs, move_axis_to_end=False
        ),
        expected,
    )

    np.testing.assert_allclose(
        cmomy.resample_vals(
            *xy, weight=weight, nrep=nrep, rng=123, **kwargs, move_axis_to_end=False
        ),
        expected,
    )


# ** Jackknife
@pytest.mark.parametrize("data_and_kwargs", data_params, indirect=True)
@pytest.mark.parametrize("pass_reduced", [True, False])
def test_jackknife_data_0(data_and_kwargs, pass_reduced, as_dataarray):
    data, kwargs = data_and_kwargs

    axis, mom_ndim = (kwargs[k] for k in ("axis", "mom_ndim"))
    ndat = cmomy.resample.select_ndat(data, axis=axis)

    freq = cmomy.resample.jackknife_freq(ndat)
    indices = cmomy.resample.freq_to_indices(freq, shuffle=False)

    # expected from reduction
    expected = cmomy.reduce_data(
        data.take(indices, axis=axis), mom_ndim=mom_ndim, axis=axis + 1
    )

    if as_dataarray:
        data = xr.DataArray(data)

    # using freq
    np.testing.assert_allclose(
        cmomy.resample_data(data, freq=freq, axis=axis, mom_ndim=mom_ndim),
        expected,
    )

    # using jackknife
    kws = {"data_reduced": cmomy.reduce_data(data, **kwargs)} if pass_reduced else {}
    np.testing.assert_allclose(
        cmomy.resample.jackknife_data(data, axis=axis, mom_ndim=mom_ndim, **kws),
        expected,
    )

    if as_dataarray and pass_reduced:
        # also pass in array value for data_reduced
        kws["data_reduced"] = kws["data_reduced"].to_numpy()
        np.testing.assert_allclose(
            cmomy.resample.jackknife_data(data, axis=axis, mom_ndim=mom_ndim, **kws),
            expected,
        )


@pytest.mark.parametrize("data_and_kwargs", vals_params, indirect=True)
@pytest.mark.parametrize("pass_reduced", [True, False])
def test_jackknife_vals_0(data_and_kwargs, pass_reduced, as_dataarray):
    data, kwargs = data_and_kwargs

    axis, mom = (kwargs[k] for k in ("axis", "mom"))
    ndat = cmomy.resample.select_ndat(data[0], axis=axis)

    freq = cmomy.resample.jackknife_freq(ndat)
    indices = cmomy.resample.freq_to_indices(freq, shuffle=False)

    # expected
    data_take = [d.take(indices, axis=axis) for d in data]
    xy_take, weight_take = unpack_data_to_xy_weight(data_take, mom)
    expected = cmomy.reduce_vals(*xy_take, weight=weight_take, mom=mom, axis=axis + 1)

    if as_dataarray:
        data = [xr.DataArray(d) for d in data]

    xy, weight = unpack_data_to_xy_weight(data, mom)
    # using jackknife freq
    np.testing.assert_allclose(
        cmomy.resample_vals(
            *xy, freq=freq, weight=weight, **kwargs, move_axis_to_end=False
        ),
        expected,
    )

    # using actual jackknife
    kws = (
        {"data_reduced": cmomy.reduce_vals(*xy, weight=weight, **kwargs)}
        if pass_reduced
        else {}
    )
    np.testing.assert_allclose(
        cmomy.resample.jackknife_vals(
            *xy, weight=weight, **kwargs, **kws, move_axis_to_end=False
        ),
        expected,
    )
    if as_dataarray and pass_reduced:
        kws["data_reduced"] = kws["data_reduced"].to_numpy()
        np.testing.assert_allclose(
            cmomy.resample.jackknife_vals(
                *xy, weight=weight, **kwargs, **kws, move_axis_to_end=False
            ),
            expected,
        )


def test_jackknife_data_rep_dim() -> None:
    data = xr.DataArray(np.zeros((10, 3)))
    assert (
        cmomy.resample.jackknife_data(data, mom_ndim=1, axis=0, rep_dim=None).dims
        == data.dims
    )
    assert cmomy.resample.jackknife_vals(
        data, mom=(3,), axis=0, rep_dim=None, mom_dims="mom", move_axis_to_end=False
    ).dims == (*data.dims, "mom")


# * Testing utilities
@pytest.mark.parametrize("ndat", [50])
@pytest.mark.parametrize("nrep", [20])
@pytest.mark.parametrize("nsamp", [None, 20])
def test_freq_indices_roundtrip(ndat, nrep, nsamp) -> None:
    indices = cmomy.random_indices(nrep, ndat, nsamp, rng=123)
    freq = cmomy.resample.indices_to_freq(indices, ndat=ndat)

    # test that just creating gives same answer
    np.testing.assert_equal(
        cmomy.random_freq(nrep, ndat, nsamp, rng=123),
        freq,
    )

    # test round trip
    np.testing.assert_allclose(
        cmomy.resample.freq_to_indices(freq, shuffle=False),
        np.sort(indices, axis=-1),
    )


def test_freq_to_indices_error() -> None:
    freq = np.array([[5, 0], [0, 4]])
    with pytest.raises(ValueError, match="Inconsistent number of samples .*"):
        resample.freq_to_indices(freq)


@pytest.mark.parametrize("style", ["array-like", "array", "dataarray", "dataset"])
@pytest.mark.parametrize(
    ("nrep", "ndat", "nsamp"),
    [
        (20, 10, None),
        (20, 10, 5),
    ],
)
def test_freq_to_indices_types(rng, nrep, ndat, nsamp, style) -> None:
    indices = resample.random_indices(nrep=nrep, ndat=ndat, nsamp=nsamp, rng=rng)

    if style == "array-like":
        idx = indices.tolist()
    elif style == "array":
        idx = indices
    elif style == "dataarray":
        idx = xr.DataArray(indices)
    elif style == "dataset":
        idx = xr.Dataset(
            {
                "x0": xr.DataArray(indices),
                "x1": xr.DataArray(
                    resample.random_indices(nrep=nrep, ndat=ndat, nsamp=nsamp, rng=rng)
                ),
            }
        )

    assert_allclose = (
        xr.testing.assert_allclose if is_xarray(idx) else np.testing.assert_allclose
    )

    freq0 = resample.indices_to_freq(idx, ndat=ndat)
    freq1 = resample.randsamp_freq(indices=idx, ndat=ndat)

    assert_allclose(freq0, freq1)

    # round trip
    idx1 = resample.freq_to_indices(freq0, shuffle=False)
    assert (
        type(freq0)
        is type(freq1)
        is type(idx1)
        is (np.ndarray if style == "array-like" else type(idx))
    )
    if is_dataset(idx):
        assert_allclose(idx1, idx.map(np.sort, axis=-1))
    else:
        np.testing.assert_allclose(idx1, np.sort(idx, axis=-1))
    assert_allclose(freq0, resample.indices_to_freq(idx1, ndat=ndat))


def test_central_randsamp_freq():
    c = CentralMoments.zeros(mom=4, val_shape=(10, 4))
    freq0 = resample.randsamp_freq(nrep=10, ndat=10, rng=np.random.default_rng(0))
    freq2 = resample.randsamp_freq(
        data=c.obj, axis=0, rng=np.random.default_rng(0), nrep=10
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


@pytest.mark.parametrize(
    ("freq", "nrep", "expected"),
    [
        (None, None, ValueError),
        (None, 10, 10),
        (np.zeros((10, 2)), 5, 10),
        (xr.DataArray(np.zeros((10, 2)), dims=["rep", "dim"]), 5, 10),
        (
            xr.Dataset({"a": xr.DataArray(np.zeros((10, 2)), dims=["rep", "dim"])}),
            5,
            10,
        ),
    ],
)
def test__select_nrep(freq, nrep, expected) -> None:
    func = resample._select_nrep
    kwargs = {"freq": freq, "nrep": nrep, "rep_dim": "rep"}
    if isinstance(expected, type):
        with pytest.raises(expected):
            func(**kwargs)
    else:
        assert func(**kwargs) == expected


@pytest.mark.parametrize(
    ("data", "kwargs", "expected"),
    [
        (
            np.zeros((2, 3, 4)),
            {"nrep": 2, "ndat": 3, "is_freq": True},
            pytest.raises(ValueError),
        ),
        (
            np.zeros((2, 3)),
            {"nrep": 3, "ndat": 3, "is_freq": True},
            pytest.raises(ValueError),
        ),
        (
            np.zeros((2, 3)),
            {"nrep": None, "ndat": 3, "is_freq": True},
            nullcontext(None),
        ),
        (
            np.zeros((2, 3)),
            {"nrep": 2, "ndat": 5, "is_freq": True},
            pytest.raises(ValueError),
        ),
        (np.zeros((2, 3)), {"nrep": 2, "ndat": 5, "is_freq": False}, nullcontext(None)),
        (
            np.zeros((2, 3)) + 10,
            {"nrep": 2, "ndat": 5, "is_freq": False},
            pytest.raises(ValueError),
        ),
    ],
)
def test_validate_resample_array(data, kwargs, expected) -> None:
    with expected:
        resample._validate_resample_array(data, **kwargs)


def test_randsamp_freq() -> None:
    ndat = 5
    nrep = 10

    f0 = resample.randsamp_freq(nrep=nrep, ndat=ndat, rng=np.random.default_rng(0))
    f1 = resample.random_freq(nrep=nrep, ndat=ndat, rng=np.random.default_rng(0))
    np.testing.assert_allclose(f0, f1)

    # test short circuit
    f2 = resample.randsamp_freq(freq=f1, check=False)
    assert f1 is f2

    f2 = resample.randsamp_freq(nrep=nrep, ndat=ndat, freq=f1, check=True)
    assert f1 is f2

    with pytest.raises(ValueError):
        resample.randsamp_freq(ndat=10)

    with pytest.raises(ValueError, match=".*Wrong ndat.*"):
        _ = resample.randsamp_freq(ndat=10, nrep=nrep, freq=f1, check=True)


@pytest.mark.parametrize(
    "expected", [resample.random_freq(nrep=10, ndat=5, rng=np.random.default_rng(0))]
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"data": np.zeros((5, 2)), "axis": 0},
        {"data": xr.DataArray(np.zeros((5, 2))), "dim": "dim_0"},
        {"data": xr.Dataset({"x0": xr.DataArray(np.zeros((5, 2)))}), "dim": "dim_0"},
        # test that mom_ndim -> mom_dims removed x1
        {
            "data": xr.Dataset(
                {
                    "x0": xr.DataArray(
                        np.zeros((5, 2, 3, 3)), dims=["a", "b", "mom0", "mom1"]
                    ),
                    "x1": xr.DataArray(np.zeros(5), dims=["a"]),
                }
            ),
            "dim": "a",
            "mom_ndim": 2,
            "paired": False,
        },
    ],
)
def test_randsamp_freq_from_data(expected, kwargs) -> None:
    out = resample.randsamp_freq(**kwargs, nrep=10, rng=np.random.default_rng(0))
    np.testing.assert_allclose(out, expected)

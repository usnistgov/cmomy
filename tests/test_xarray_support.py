# mypy: disable-error-code="no-untyped-def, no-untyped-call, call-overload, assignment, arg-type"
# pyright: reportCallIssue=false, reportArgumentType=false
"""
Test that basic operations on DataArrays give same results and on np.ndarrays
and that operations on datasets give same results as on dataarrays.

Also test that chunking works.
"""

from __future__ import annotations

import inspect
from functools import partial

import numpy as np
import pytest
import xarray as xr

import cmomy
from cmomy.core.validate import is_dataarray, is_dataset

from ._dataarray_set_utils import (
    do_bootstrap_data,
    do_moveaxis,
    do_reduce_data_grouped,
    do_reduce_data_indexed,
    do_wrap,
    do_wrap_method,
    do_wrap_raw,
    do_wrap_reduce_vals,
    do_wrap_resample_vals,
    moments_to_comoments_kwargs,
    remove_dim_from_kwargs,
)


# * Utils
def create_data(
    rng,
    shapes_and_dims,
    dim,
    **kwargs,  # noqa: ARG001
) -> xr.Dataset | xr.DataArray:
    if isinstance(shapes_and_dims, tuple):
        shape, dims = shapes_and_dims
        ds = xr.DataArray(rng.random(shape), dims=dims)
    else:
        ds = xr.Dataset(
            {
                name: xr.DataArray(rng.random(shape), dims=dims)
                for name, (shape, dims) in zip(
                    [f"data{k}" for k in range(len(shapes_and_dims))], shapes_and_dims
                )
            }
        )

    if dim:
        # coordinates along sampled dimension
        ds = ds.assign_coords({dim: (dim, range(ds.sizes[dim]))})
    return ds


def fix_kws_for_array(kwargs, data):
    kwargs = kwargs.copy()

    # transform dim -> axis
    dim = kwargs.pop("dim", None)
    if dim is not None:
        kwargs["axis"] = data.dims.index(dim)
    return kwargs


def get_reshaped_val(val, reshape):
    if val is None:
        return val
    val = val.to_numpy()
    if reshape:
        val = val.reshape(reshape)
    return val


@pytest.fixture
def as_dataarray(request):
    # Use this to flag data_and_kwargs and fixture_vals fixture.
    return request.param


# * functions
# ** Data
func_params_data_common = [
    (partial(cmomy.reduce_data, use_reduce=True), None),
    (partial(cmomy.reduce_data, use_reduce=False), None),
    (do_reduce_data_grouped, None),
    (do_reduce_data_indexed, None),  # default coords_policy="first"
    (partial(cmomy.resample_data, nrep=20, rng=0, paired=True), None),
    (cmomy.resample.jackknife_data, None),
    (partial(do_bootstrap_data, nrep=20, method="percentile"), None),
    (partial(do_bootstrap_data, nrep=20, method="bca"), None),
    (cmomy.convert.moments_type, remove_dim_from_kwargs),
    (cmomy.convert.cumulative, None),
    (cmomy.convert.moments_to_comoments, moments_to_comoments_kwargs),
    (partial(cmomy.utils.select_moment, name="weight"), remove_dim_from_kwargs),
    (partial(cmomy.utils.select_moment, name="ave"), remove_dim_from_kwargs),
    (
        partial(cmomy.utils.assign_moment, weight=1),
        remove_dim_from_kwargs,
    ),
    (
        partial(cmomy.utils.assign_moment, ave=1),
        remove_dim_from_kwargs,
    ),
    (partial(cmomy.rolling.rolling_data, window=2), None),
    (partial(cmomy.rolling.rolling_exp_data, alpha=0.2), None),
    (do_wrap, remove_dim_from_kwargs),
    (do_wrap_raw, remove_dim_from_kwargs),
    (do_wrap_method("reduce"), None),
    (partial(do_wrap_method("resample_and_reduce"), nrep=20, rng=0), None),
    (do_wrap_method("jackknife_and_reduce"), None),
    (do_wrap_method("cumulative"), None),
    (do_wrap_method("to_raw"), remove_dim_from_kwargs),
    (do_wrap_method("moments_to_comoments"), moments_to_comoments_kwargs),
]

func_params_data_dataarray = [
    (do_moveaxis, None),
    # TODO(wpk): Seems that coords_policy="group" would be best default ...
    (partial(do_wrap_method("reduce"), by=[0] * 5 + [1] * 5), None),
]

func_params_data_dataset = [
    (partial(do_reduce_data_indexed, coords_policy="groups"), None),
    (partial(do_reduce_data_indexed, coords_policy=None), None),
    (
        partial(do_wrap_method("reduce"), by=[0] * 5 + [1] * 5, coords_policy="group"),
        None,
    ),
]

# ** Vals
func_params_vals_common = [
    (cmomy.reduce_vals, None),
    (partial(cmomy.resample_vals, nrep=20, rng=0), None),
    (cmomy.resample.jackknife_vals, None),
    (cmomy.utils.vals_to_data, remove_dim_from_kwargs),
    (partial(cmomy.rolling.rolling_vals, window=2), None),
    (partial(cmomy.rolling.rolling_exp_vals, alpha=0.2), None),
    # wrap
    (do_wrap_reduce_vals, None),
    (do_wrap_resample_vals, None),
]


# * data parameters
# ** data
data_params_simple = [
    # make sure these have reduction variable and mom in all variables
    (
        [((10, 2, 3), ("a", "b", "mom")), ((10, 3), ("a", "mom"))],
        {"mom_ndim": 1, "dim": "a"},
    ),
    (
        [((2, 10, 3), ("a", "b", "mom")), ((10, 3), ("b", "mom"))],
        {"mom_ndim": 1, "dim": "b"},
    ),
    (
        [
            ((2, 10, 3, 3), ("a", "b", "mom0", "mom1")),
            ((10, 3, 3), ("b", "mom0", "mom1")),
        ],
        {"mom_ndim": 2, "dim": "b"},
    ),
]


data_params = [
    *data_params_simple,
    (
        [((10, 2, 3), ("a", "b", "mom")), ((2, 3), ("b", "mom"))],
        {"mom_ndim": 1, "dim": "a"},
    ),
    (
        [
            ((10, 2, 3, 3), ("a", "b", "mom0", "mom1")),
            ((2, 3, 3), ("b", "mom0", "mom1")),
        ],
        {"mom_ndim": 2, "dim": "a"},
    ),
    # different moment names
    (
        [((10, 2, 3), ("a", "b", "mom")), ((2, 3), ("b", "mom_other"))],
        {"mom_ndim": 1, "dim": "a", "mom_dims": "mom"},
    ),
]


@pytest.fixture
def data_and_kwargs(rng, as_dataarray, request):
    shapes_and_dims, kwargs = request.param
    if as_dataarray:
        shapes_and_dims = shapes_and_dims[0]
    return create_data(rng, shapes_and_dims, **kwargs), kwargs


# ** vals
vals_params_push = [
    {
        "kwargs": {"mom": (3,), "dim": "b"},
        "x": [((2, 10), ("a", "b")), (10, "b")],
        "y": None,
        "weight": None,
    },
    {
        "kwargs": {"mom": (3,), "dim": "b"},
        "x": [((2, 10), ("a", "b")), (10, "b")],
        "y": None,
        "weight": (10, "b"),
    },
    {
        "kwargs": {"mom": (3, 3), "dim": "b"},
        "x": [((2, 10), ("a", "b")), (10, "b")],
        "y": (10, "b"),
        "weight": None,
    },
    {
        "kwargs": {"mom": (3, 3), "dim": "b"},
        "x": [((2, 10), ("a", "b")), (10, "b")],
        "y": (10, "b"),
        "weight": (10, "b"),
    },
]

vals_params_dataarray = [
    *vals_params_push,
    {
        "kwargs": {"mom": (3,), "dim": "a"},
        "x": [((10, 2), ("a", "b")), (2, "b")],
        "y": None,
        "weight": None,
    },
    {
        "kwargs": {"mom": (3,), "dim": "a"},
        "x": [((10, 2), ("a", "b")), (2, "b")],
        "y": None,
        "weight": (10, "a"),
        "weight_reshape": (10, 1),
    },
]


vals_params = [
    *vals_params_dataarray,
    {
        "kwargs": {"mom": (3,), "dim": "a"},
        "x": [((10, 2), ("a", "b")), (2, "b")],
        "y": None,
        "weight": [(10, "a"), (10, "a")],
    },
    {
        "kwargs": {"mom": (3,), "dim": "b"},
        "x": [((2, 10), ("a", "b")), (10, "b")],
        "y": None,
        "weight": [(10, "b"), (10, "b")],
    },
    # mom_ndim -> 2
    {
        "kwargs": {"mom": (3, 3), "dim": "b"},
        "x": [((2, 10), ("a", "b")), (10, "b")],
        "y": (10, "b"),
        "weight": [(10, "b"), (10, "b")],
    },
    {
        "kwargs": {"mom": (3, 3), "dim": "b"},
        "x": [((2, 10), ("a", "b")), (10, "b")],
        "y": [(10, "b"), ((2, 10), ("a", "b"))],
        "weight": None,
    },
    {
        "kwargs": {"mom": (3, 3), "dim": "b"},
        "x": [((2, 10), ("a", "b")), (10, "b")],
        "y": [(10, "b"), ((2, 10), ("a", "b"))],
        "weight": (10, "b"),
    },
    {
        "kwargs": {"mom": (3, 3), "dim": "b"},
        "x": [((2, 10), ("a", "b")), (10, "b")],
        "y": [(10, "b"), ((2, 10), ("a", "b"))],
        "weight": [(10, "b"), (10, "b")],
    },
]


@pytest.fixture
def fixture_vals(rng, as_dataarray, request):
    """Ugly way to do things, but works."""
    kwargs, x, y, weight = (request.param[k] for k in ("kwargs", "x", "y", "weight"))
    if as_dataarray:
        x, y, weight = (a[0] if isinstance(a, list) else a for a in (x, y, weight))

    out = {
        "kwargs": kwargs,
        "x": create_data(rng, x, **kwargs),
        "y": y if y is None else create_data(rng, y, **kwargs),
        "weight": weight if weight is None else create_data(rng, weight, **kwargs),
    }

    for k, v in request.param.items():
        if "_reshape" in k:
            out[k] = v  # noqa: PERF403
    return out


# * Data tests
@pytest.mark.parametrize(
    ("func", "kwargs_callback"),
    [
        *func_params_data_common,
        *func_params_data_dataarray,
    ],
)
@pytest.mark.parametrize("data_and_kwargs", data_params, indirect=True)
@pytest.mark.parametrize(
    "as_dataarray",
    [True],
    indirect=True,
)
def test_func_data_dataarray(
    data_and_kwargs,
    func,
    kwargs_callback,
) -> None:
    data, kwargs = data_and_kwargs
    if kwargs_callback:
        kwargs = kwargs_callback(kwargs)

    assert is_dataarray(data)

    check = func(data, **kwargs)
    kws_array = fix_kws_for_array(kwargs, data)
    np.testing.assert_allclose(check, func(data.values, **kws_array))


@pytest.mark.parametrize(
    ("func", "kwargs_callback"),
    [
        *func_params_data_common,
        *func_params_data_dataset,
    ],
)
@pytest.mark.parametrize("data_and_kwargs", data_params, indirect=True)
@pytest.mark.parametrize("as_dataarray", [False], indirect=True)
def test_func_data_dataset(data_and_kwargs, func, kwargs_callback) -> None:
    data, kwargs = data_and_kwargs
    kws = kwargs_callback(kwargs.copy()) if kwargs_callback else kwargs

    # coordinates along sampled dimension
    out = func(data, **kws)

    if "dim" in kws and "move_axis_to_end" in inspect.signature(func).parameters:
        kws = {"move_axis_to_end": True, **kws}

    for k in data:
        da = data[k]
        if ("dim" not in kws or kws["dim"] in da.dims) and (
            "mom_dims" not in kws or kws["mom_dims"] in da.dims
        ):
            da = func(da, **kws)
    xr.testing.assert_allclose(out[k], da)


@pytest.mark.parametrize(
    "data_and_kwargs",
    [
        *data_params_simple,
    ],
    indirect=True,
)
@pytest.mark.parametrize("as_dataarray", [True, False])
def test_func_dataarray_and_dataset_push_data(data_and_kwargs, as_dataarray) -> None:
    data, kwargs = data_and_kwargs
    assert is_dataarray(data) is as_dataarray

    mom_ndim = kwargs["mom_ndim"]
    dim = kwargs["dim"]
    expected = cmomy.wrap(data, mom_ndim=mom_ndim).reduce(dim=dim)
    a, b = (cmomy.zeros_like(expected) for _ in range(2))

    for _, d in data.groupby(dim):
        a.push_data(d.squeeze(dim))
    b.push_datas(data, dim=dim)

    xr.testing.assert_allclose(a.obj, expected.obj)
    xr.testing.assert_allclose(b.obj, expected.obj)

    # test push_data with scale....
    c = cmomy.wrap(data)
    a = c.isel({dim: slice(None, -1)}).reduce(dim=dim)
    b = c.reduce(dim=dim) - c.isel({dim: -1}, drop=True)
    xr.testing.assert_allclose(a.obj, b.obj)


# * Vals to array
@pytest.mark.parametrize(
    ("func", "kwargs_callback"),
    func_params_vals_common,
)
@pytest.mark.parametrize("as_dataarray", [True], indirect=True)
@pytest.mark.parametrize("fixture_vals", vals_params_dataarray, indirect=True)
def test_func_vals_dataarray(fixture_vals, func, kwargs_callback):
    kwargs, x, y, weight = (fixture_vals[k] for k in ("kwargs", "x", "y", "weight"))
    assert is_dataarray(x)

    if kwargs_callback:
        kwargs = kwargs_callback(kwargs.copy())

    args = (x,) if y is None else (x, y)

    check = func(*args, weight=weight, **kwargs)
    kws_array = fix_kws_for_array(kwargs, x)

    xx = get_reshaped_val(x, None)
    yy = get_reshaped_val(y, fixture_vals.get("y_reshape"))
    ww = get_reshaped_val(weight, fixture_vals.get("weight_reshape"))

    args = (xx,) if yy is None else (xx, yy)
    np.testing.assert_allclose(check, func(*args, weight=ww, **kws_array))


# * Vals dataset
@pytest.mark.parametrize(
    ("func", "kwargs_callback"),
    func_params_vals_common,
)
@pytest.mark.parametrize("as_dataarray", [False], indirect=True)
@pytest.mark.parametrize("fixture_vals", vals_params, indirect=True)
def test_func_vals_dataset(fixture_vals, func, kwargs_callback):
    kwargs, x, y, weight = (fixture_vals[k] for k in ("kwargs", "x", "y", "weight"))

    if kwargs_callback:
        kwargs = kwargs_callback(kwargs.copy())

    out = func(*((x,) if y is None else (x, y)), weight=weight, **kwargs)

    for name in x:
        da = x[name]
        if "dim" not in kwargs or kwargs["dim"] in da.dims:
            if y is not None:
                dy = y if is_dataarray(y) else y[name]
                xy = (da, dy)
            else:
                xy = (da,)

            if weight is not None:
                w = weight if is_dataarray(weight) else weight[name]
            else:
                w = weight

            da = func(*xy, **kwargs, weight=w)

        xr.testing.assert_allclose(out[name], da)


@pytest.mark.parametrize("as_dataarray", [True, False], indirect=True)
@pytest.mark.parametrize("fixture_vals", vals_params_push, indirect=True)
def test_func_dataarray_and_dataset_push_vals(fixture_vals, as_dataarray) -> None:
    kwargs, x, y, weight = (fixture_vals[k] for k in ("kwargs", "x", "y", "weight"))
    assert is_dataarray(x) is as_dataarray

    xy = (x,) if y is None else (x, y)
    expected = cmomy.wrap_reduce_vals(*xy, weight=weight, **kwargs)

    kws = kwargs.copy()

    mom = kws.pop("mom")
    dim = kws["dim"]
    if len(mom) == 1 and weight is None:
        a = cmomy.zeros_like(expected)
        for _, d in x.groupby(dim):
            a.push_val(d.squeeze(dim))
        xr.testing.assert_allclose(a.obj, expected.obj)

    b = cmomy.zeros_like(expected)
    b.push_vals(*xy, weight=weight, **kws)
    xr.testing.assert_allclose(b.obj, expected.obj)


# * Resample [need spectial treatment]
@pytest.mark.parametrize(
    "data",
    [
        xr.Dataset(
            {
                "data0": xr.DataArray(
                    np.zeros((2, 3, 4, 3, 3)), dims=["a", "b", "c", "mom0", "mom1"]
                ),
                "data1": xr.DataArray(
                    np.zeros((2, 5, 3, 3)), dims=["a", "d", "momA", "momB"]
                ),
                "data2": xr.DataArray(
                    np.zeros((2, 3, 3, 3)), dims=["a", "b", "mom0", "mom1"]
                ),
            }
        )
    ],
)
@pytest.mark.parametrize(
    ("dim", "paired", "kws", "names"),
    [
        ("a", True, {}, []),
        ("b", True, {}, []),
        ("c", True, {}, []),
        ("d", True, {}, []),
        ("a", False, {}, ["data0", "data1", "data2"]),
        ("b", False, {}, ["data0", "data2"]),
        # result in dataset with one variable -> DataArray...
        ("c", False, {}, []),
        ("d", False, {}, []),
        # passing mom_dims
        ("a", False, {"mom_dims": ("mom0", "mom1")}, ["data0", "data2"]),
        ("a", False, {"mom_dims": ("momA", "momB")}, []),
    ],
)
def test_randsamp_freq_dataset(
    data,
    dim,
    paired,
    kws,
    names,
    get_zero_rng,
) -> None:
    nrep = 10
    rep_dim = "rep"
    kwargs = {"nrep": nrep, "rep_dim": rep_dim, "paired": paired, **kws}

    # paired
    out = cmomy.resample.randsamp_freq(
        data=data,
        dim=dim,
        **kwargs,
        rng=0,
    )

    if names:
        rng = get_zero_rng()
        expected = xr.Dataset(
            {
                k: xr.DataArray(
                    cmomy.resample.random_freq(
                        ndat=data.sizes[dim], nrep=nrep, rng=rng
                    ),
                    dims=[rep_dim, dim],
                )
                for k in names
            }
        )
    else:
        # Single dataarray
        expected = xr.DataArray(
            cmomy.resample.random_freq(
                ndat=data.sizes[dim],
                nrep=nrep,
                rng=0,
            ),
            dims=[rep_dim, dim],
        )

    xr.testing.assert_allclose(out, expected)

    # and should get back self from this
    assert cmomy.resample.randsamp_freq(freq=out, check=False, dtype=None) is out


@pytest.mark.parametrize("data_and_kwargs", data_params, indirect=True)
@pytest.mark.parametrize("as_dataarray", [False], indirect=True)
@pytest.mark.parametrize("nrep", [20])
@pytest.mark.parametrize("paired", [False])
def test_resample_data_dataset(data_and_kwargs, nrep, paired) -> None:
    ds, kwargs = data_and_kwargs
    assert not is_dataarray(ds)

    dfreq = cmomy.randsamp_freq(data=ds, **kwargs, nrep=nrep, rng=0, paired=paired)

    out = cmomy.resample_data(ds, **kwargs, freq=dfreq)
    dim = kwargs["dim"]
    for name in ds:
        da = ds[name]
        if dim in da.dims and (
            "mom_dims" not in kwargs or kwargs["mom_dims"] in da.dims
        ):
            da = cmomy.resample_data(
                da,
                **kwargs,
                freq=dfreq if is_dataarray(dfreq) else dfreq[name],
                move_axis_to_end=True,
            )

        xr.testing.assert_allclose(out[name], da)

    # indirect
    xr.testing.assert_allclose(
        out,
        cmomy.resample_data(
            ds,
            **kwargs,
            nrep=nrep,
            rng=0,
            paired=paired,
        ),
    )


@pytest.mark.parametrize("nrep", [20])
@pytest.mark.parametrize("paired", [False])
@pytest.mark.parametrize("as_dataarray", [False], indirect=True)
@pytest.mark.parametrize("fixture_vals", vals_params, indirect=True)
def test_resample_vals_dataset(fixture_vals, paired, nrep) -> None:
    kwargs, x, y, weight = (fixture_vals[k] for k in ("kwargs", "x", "y", "weight"))

    dim = kwargs["dim"]
    dfreq = cmomy.randsamp_freq(data=x, dim=dim, nrep=nrep, rng=0, paired=paired)

    xy = (x,) if y is None else (x, y)
    out = cmomy.resample_vals(*xy, weight=weight, **kwargs, freq=dfreq)

    for name in x:
        da = x[name]
        if kwargs["dim"] in da.dims:
            if y is not None:
                dy = y if is_dataarray(y) else y[name]
                _xy = (da, dy)
            else:
                _xy = (da,)

            if weight is not None:
                w = weight if is_dataarray(weight) else weight[name]
            else:
                w = weight

            da = cmomy.resample_vals(
                *_xy,
                weight=w,
                **kwargs,
                freq=dfreq if is_dataarray(dfreq) else dfreq[name],
            )

        xr.testing.assert_allclose(out[name], da)

    # testing indirect
    xr.testing.assert_allclose(
        out,
        cmomy.resample_vals(
            *xy,
            weight=weight,
            **kwargs,
            nrep=nrep,
            rng=0,
            paired=paired,
        ),
    )


# * Chunking
try:
    import dask  # noqa: F401  # pyright: ignore[reportUnusedImport]

    HAS_DASK = True
except ImportError:
    HAS_DASK = False

mark_dask_only = pytest.mark.skipif(not HAS_DASK, reason="dask not installed")


def _is_chunked(ds):
    if is_dataset(ds):
        return ds.chunks != {}

    return ds.chunks is not None


@pytest.mark.slow
@mark_dask_only
@pytest.mark.parametrize("data_and_kwargs", data_params, indirect=True)
@pytest.mark.parametrize("as_dataarray", [False], indirect=True)
@pytest.mark.parametrize(
    ("func", "kwargs_callback"),
    func_params_data_common,
)
def test_func_data_chunking(data_and_kwargs, func, kwargs_callback) -> None:
    ds, kwargs = data_and_kwargs
    ds_chunked = ds.chunk({kwargs["dim"]: -1})

    kws = kwargs_callback(kwargs.copy()) if kwargs_callback else kwargs

    # coordinates along sampled dimension
    out = func(ds, **kws)
    out_chunked = func(ds_chunked, **kws)

    xr.testing.assert_allclose(out, out_chunked)

    assert _is_chunked(ds_chunked)
    assert not _is_chunked(out)
    assert _is_chunked(out_chunked)

    for k, da in ds.items():
        if ("dim" not in kws or kws["dim"] in da.dims) and (
            "mom_dims" not in kws or kws["mom_dims"] in da.dims
        ):
            a = func(da, **kws)
            b = func(ds_chunked[k], **kws)

            xr.testing.assert_allclose(a, b)
            assert not _is_chunked(a)
            assert _is_chunked(b)


@pytest.mark.slow
@mark_dask_only
@pytest.mark.parametrize(
    ("func", "kwargs_callback"),
    func_params_vals_common,
)
@pytest.mark.parametrize("as_dataarray", [False], indirect=True)
@pytest.mark.parametrize("fixture_vals", vals_params, indirect=True)
def test_func_vals_chunking(fixture_vals, func, kwargs_callback):
    kwargs, x, y, weight = (fixture_vals[k] for k in ("kwargs", "x", "y", "weight"))
    x_chunked = x.chunk({kwargs["dim"]: -1})

    kws = kwargs_callback(kwargs.copy()) if kwargs_callback else kwargs
    out = func(*((x,) if y is None else (x, y)), weight=weight, **kws)
    out_chunked = func(
        *((x_chunked,) if y is None else (x_chunked, y)), weight=weight, **kws
    )

    xr.testing.assert_allclose(out, out_chunked)
    assert not _is_chunked(out)
    assert _is_chunked(out_chunked)

    for k, da in x.items():
        if "dim" not in kws or kws["dim"] in da.dims:
            if y is not None:
                dy = y if is_dataarray(y) else y[k]
                xy = (da, dy)
                xy_chunked = (x_chunked[k], dy)
            else:
                xy = (da,)
                xy_chunked = (x_chunked[k],)

            if weight is not None:
                w = weight if is_dataarray(weight) else weight[k]
            else:
                w = weight

            a = func(*xy, **kws, weight=w)
            b = func(*xy_chunked, **kws, weight=w)

            xr.testing.assert_allclose(a, b)
            assert not _is_chunked(a)
            assert _is_chunked(b)


@pytest.mark.slow
@mark_dask_only
@pytest.mark.parametrize(
    ("func", "kwargs_callback"),
    [
        (partial(cmomy.reduce_data, use_reduce=False), None),
        (do_reduce_data_grouped, None),
        (partial(do_reduce_data_indexed, coords_policy=None), None),
        (partial(cmomy.resample_data, rng=0, nrep=20, paired=True), None),
        (partial(cmomy.resample_data, rng=0, nrep=20, paired=False), None),
        (cmomy.resample.jackknife_data, None),
        (cmomy.convert.moments_type, remove_dim_from_kwargs),
        (cmomy.convert.cumulative, None),
        (partial(cmomy.rolling.rolling_data, window=2), None),
        (partial(cmomy.rolling.rolling_exp_data, alpha=0.2), None),
    ],
)
@pytest.mark.parametrize("as_dataarray", [False], indirect=True)
@pytest.mark.parametrize(
    "data_and_kwargs",
    [
        (((10, 3), None), {"mom_ndim": 1, "dim": "dim_0"}),
        (((10, 3, 3), None), {"mom_ndim": 2, "dim": "dim_0"}),
    ],
    indirect=True,
)
def test_func_data_chunking_out_parameter(
    data_and_kwargs,
    func,
    kwargs_callback,
) -> None:
    data, kwargs = data_and_kwargs
    data_chunked = data.chunk({kwargs["dim"]: -1})

    kws = kwargs if kwargs_callback is None else kwargs_callback(kwargs.copy())
    res = func(data, **kws)

    out = np.zeros_like(res)
    res_chunk = func(data_chunked, **kws, out=out)

    xr.testing.assert_allclose(res, res_chunk)
    assert _is_chunked(res_chunk)
    assert np.shares_memory(res_chunk.compute(), out)


@pytest.mark.slow
@mark_dask_only
@pytest.mark.parametrize(
    ("func", "kwargs_callback"),
    [
        (cmomy.reduce_vals, None),
        (partial(cmomy.resample_vals, rng=0, nrep=20, paired=True), None),
        (partial(cmomy.resample_vals, rng=0, nrep=20, paired=False), None),
        (cmomy.resample.jackknife_vals, None),
        # (cmomy.utils.vals_to_data, remove_dim_from_kwargs),  # noqa: ERA001
        (partial(cmomy.rolling.rolling_vals, window=2), None),
        (partial(cmomy.rolling.rolling_exp_vals, alpha=0.2), None),
    ],
)
@pytest.mark.parametrize(
    ("dim", "mom", "shape"),
    [
        ("dim_0", 3, (10,)),
    ],
)
def test_func_vals_chunking_out_parameter(rng, func, kwargs_callback, dim, mom, shape):
    data = xr.DataArray(rng.random(shape))
    data_chunked = data.chunk({dim: -1})

    kws = {"dim": dim, "mom": mom, "weight": None}
    kws = kws if kwargs_callback is None else kwargs_callback(kws)

    res = func(data, **kws)

    out = np.zeros_like(res)
    res_chunk = func(data_chunked, **kws, out=out)

    xr.testing.assert_allclose(res, res_chunk)
    assert _is_chunked(res_chunk)
    assert np.shares_memory(res_chunk.compute(), out)

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


# * data tests
mark_data_kwargs = pytest.mark.parametrize(
    ("kwargs", "shapes_and_dims"),
    [
        (
            {"mom_ndim": 1, "dim": "a"},
            [((10, 2, 3), ("a", "b", "mom")), ((2, 3), ("b", "mom"))],
        ),
        (
            {"mom_ndim": 1, "dim": "b"},
            [((2, 10, 3), ("a", "b", "mom")), ((10, 3), ("b", "mom"))],
        ),
        (
            {"mom_ndim": 2, "dim": "a"},
            [
                ((10, 2, 3, 3), ("a", "b", "mom0", "mom1")),
                ((2, 3, 3), ("b", "mom0", "mom1")),
            ],
        ),
        (
            {"mom_ndim": 2, "dim": "b"},
            [
                ((2, 10, 3, 3), ("a", "b", "mom0", "mom1")),
                ((10, 3, 3), ("b", "mom0", "mom1")),
            ],
        ),
        # different moment names
        (
            {"mom_ndim": 1, "dim": "a", "mom_dims": "mom"},
            [((10, 2, 3), ("a", "b", "mom")), ((2, 3), ("b", "mom_other"))],
        ),
    ],
)


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


@mark_data_kwargs
@pytest.mark.parametrize(
    ("func", "kwargs_callback"),
    [
        *func_params_data_common,
        *func_params_data_dataarray,
    ],
)
def test_func_data_dataarray(
    rng,
    kwargs,
    shapes_and_dims,
    func,
    kwargs_callback,
) -> None:
    data = create_data(rng, shapes_and_dims[0], **kwargs)
    if kwargs_callback:
        kwargs = kwargs_callback(kwargs)

    check = func(data, **kwargs)
    kws_array = fix_kws_for_array(kwargs, data)
    np.testing.assert_allclose(check, func(data.values, **kws_array))


@mark_data_kwargs
@pytest.mark.parametrize(
    ("func", "kwargs_callback"),
    [
        *func_params_data_common,
        *func_params_data_dataset,
    ],
)
def test_func_data_dataset(rng, kwargs, shapes_and_dims, func, kwargs_callback) -> None:
    data = create_data(rng, shapes_and_dims, **kwargs)
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


# * Vals to array
vals_params = [
    {
        "kwargs": {"mom": (3,), "dim": "a"},
        "x": [((10, 2), ("a", "b")), (2, "b")],
        "y": None,
        "weight": None,
        "do_array": True,
    },
    {
        "kwargs": {"mom": (3,), "dim": "a"},
        "x": [((10, 2), ("a", "b")), (2, "b")],
        "y": None,
        "weight": (10, "a"),
        "do_array": True,
        "weight_reshape": (10, 1),
    },
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
        "weight": None,
        "do_array": True,
    },
    {
        "kwargs": {"mom": (3,), "dim": "b"},
        "x": [((2, 10), ("a", "b")), (10, "b")],
        "y": None,
        "weight": (10, "b"),
        "do_array": True,
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
        "weight": None,
        "do_array": True,
    },
    {
        "kwargs": {"mom": (3, 3), "dim": "b"},
        "x": [((2, 10), ("a", "b")), (10, "b")],
        "y": (10, "b"),
        "weight": (10, "b"),
        "do_array": True,
    },
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


@pytest.fixture(params=vals_params)
def fixture_vals_dataset(request, rng):
    """Ugly way to do things, but works."""
    kwargs, x, y, weight = (request.param[k] for k in ("kwargs", "x", "y", "weight"))
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


@pytest.fixture(params=filter(lambda x: x.get("do_array", False), vals_params))  # type: ignore[attr-defined]
def fixture_vals_dataarray(request, rng):
    """Ugly way to do things, but works."""
    kwargs, x, y, weight = (request.param[k] for k in ("kwargs", "x", "y", "weight"))
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


@pytest.mark.parametrize(
    ("func", "kwargs_callback"),
    func_params_vals_common,
)
def test_func_vals_dataarray(fixture_vals_dataarray, func, kwargs_callback):
    kwargs, x, y, weight = (
        fixture_vals_dataarray[k] for k in ("kwargs", "x", "y", "weight")
    )

    if kwargs_callback:
        kwargs = kwargs_callback(kwargs.copy())

    args = (x,) if y is None else (x, y)

    check = func(*args, weight=weight, **kwargs)
    kws_array = fix_kws_for_array(kwargs, x)

    xx = get_reshaped_val(x, None)
    yy = get_reshaped_val(y, fixture_vals_dataarray.get("y_reshape"))
    ww = get_reshaped_val(weight, fixture_vals_dataarray.get("weight_reshape"))

    args = (xx,) if yy is None else (xx, yy)
    np.testing.assert_allclose(check, func(*args, weight=ww, **kws_array))


# * Vals dataset
@pytest.mark.parametrize(
    ("func", "kwargs_callback"),
    func_params_vals_common,
)
def test_func_vals_dataset(fixture_vals_dataset, func, kwargs_callback):
    kwargs, x, y, weight = (
        fixture_vals_dataset[k] for k in ("kwargs", "x", "y", "weight")
    )

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


@mark_data_kwargs
@pytest.mark.parametrize("nrep", [20])
@pytest.mark.parametrize("paired", [False])
def test_resample_data_dataset(rng, kwargs, shapes_and_dims, nrep, paired) -> None:
    ds = create_data(rng, shapes_and_dims, **kwargs)

    dfreq = cmomy.randsamp_freq(data=ds, **kwargs, nrep=nrep, rng=0, paired=paired)

    out = cmomy.resample_data(ds, **kwargs, freq=dfreq)  # type: ignore[type-var]
    dim = kwargs["dim"]
    for name in ds:
        da = ds[name]
        if dim in da.dims and (
            "mom_dims" not in kwargs or kwargs["mom_dims"] in da.dims
        ):
            da = cmomy.resample_data(
                da,
                **kwargs,
                freq=dfreq if is_dataarray(dfreq) else dfreq[name],  # type: ignore[index]
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
def test_resample_vals_dataset(fixture_vals_dataset, paired, nrep) -> None:
    kwargs, x, y, weight = (
        fixture_vals_dataset[k] for k in ("kwargs", "x", "y", "weight")
    )

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
@mark_data_kwargs
@pytest.mark.parametrize(
    ("func", "kwargs_callback"),
    func_params_data_common,
)
def test_func_data_chunking(
    rng, kwargs, shapes_and_dims, func, kwargs_callback
) -> None:
    ds = create_data(rng, shapes_and_dims, **kwargs)
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
def test_func_vals_chunking(fixture_vals_dataset, func, kwargs_callback):
    kwargs, x, y, weight = (
        fixture_vals_dataset[k] for k in ("kwargs", "x", "y", "weight")
    )
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
@pytest.mark.parametrize(
    ("dim", "mom_ndim", "shape"),
    [
        ("dim_0", 1, (10, 3)),
        ("dim_0", 2, (10, 3, 3)),
    ],
)
def test_func_data_chunking_out_parameter(
    rng, func, kwargs_callback, dim, mom_ndim, shape
) -> None:
    data = xr.DataArray(rng.random(shape))
    data_chunked = data.chunk({dim: -1})

    kws = {"dim": dim, "mom_ndim": mom_ndim}
    kws = kws if kwargs_callback is None else kwargs_callback(kws)

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

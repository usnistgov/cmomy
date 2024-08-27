# mypy: disable-error-code="no-untyped-def, no-untyped-call, call-overload, assignment, arg-type"
# pyright: reportCallIssue=false, reportArgumentType=false

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pytest
import xarray as xr

import cmomy


# * fixtures
def create_data_dataset(
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


def _get_values_params():
    return [
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
        },
        {
            "kwargs": {"mom": (3,), "dim": "b"},
            "x": [((2, 10), ("a", "b")), (10, "b")],
            "y": None,
            "weight": (10, "b"),
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
        },
        {
            "kwargs": {"mom": (3, 3), "dim": "b"},
            "x": [((2, 10), ("a", "b")), (10, "b")],
            "y": (10, "b"),
            "weight": (10, "b"),
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


@pytest.fixture(params=_get_values_params())
def fixture_vals_dataset(request, rng) -> Any:
    """Ugly way to do things, but works."""
    kwargs, x, y, weight = (request.param[k] for k in ("kwargs", "x", "y", "weight"))
    return {
        "kwargs": kwargs,
        "x": create_data_dataset(rng, x, **kwargs),
        "y": y if y is None else create_data_dataset(rng, y, **kwargs),
        "weight": weight
        if weight is None
        else create_data_dataset(rng, weight, **kwargs),
    }


# * General
def _remove_dim_from_kwargs(kwargs):
    kwargs.pop("dim")
    return kwargs


def _moments_to_comoments_kwargs(kwargs):
    for k in ("dim", "mom_ndim"):
        kwargs.pop(k)
    kwargs["mom"] = (1, -1)
    return kwargs


def _get_by(n):
    n0 = n // 2
    return [0] * n0 + [1] * (n - n0)


def _reduce_data_grouped(ds, dim, **kwargs):
    by = _get_by(ds.sizes[dim])
    return cmomy.reduction.reduce_data_grouped(ds, dim=dim, **kwargs, by=by)


def _reduce_data_indexed(ds, dim, **kwargs):
    by = _get_by(ds.sizes[dim])
    kwargs["groups"], kwargs["index"], kwargs["group_start"], kwargs["group_end"] = (
        cmomy.reduction.factor_by_to_index(by)
    )
    coords_policy = kwargs.pop("coords_policy", "first")
    if isinstance(ds, xr.DataArray) and coords_policy in {"first", "last"}:
        coords_policy = None

    return cmomy.reduction.reduce_data_indexed(
        ds,
        dim=dim,
        **kwargs,
        coords_policy=coords_policy,
    )


def _resample_data(ds, dim, nrep, paired, **kwargs):
    return cmomy.resample_data(
        ds,
        dim=dim,
        nrep=nrep,
        rng=np.random.default_rng(0),
        paired=paired,
        **kwargs,
    )


def _resample_vals(x, *y, weight, nrep, paired, **kwargs):
    return cmomy.resample_vals(
        x,
        *y,
        weight=weight,
        nrep=nrep,
        paired=paired,
        rng=np.random.default_rng(0),
        **kwargs,
    )


@mark_data_kwargs
@pytest.mark.parametrize(
    ("func", "kwargs_callback"),
    [
        (_reduce_data_grouped, None),
        (partial(_reduce_data_indexed, coords_policy="first"), None),
        (partial(_reduce_data_indexed, coords_policy="groups"), None),
        (partial(_reduce_data_indexed, coords_policy=None), None),
        (partial(_resample_data, nrep=20, paired=True), None),
        (cmomy.resample.jackknife_data, None),
        (cmomy.convert.moments_type, _remove_dim_from_kwargs),
        (cmomy.convert.cumulative, None),
        (cmomy.convert.moments_to_comoments, _moments_to_comoments_kwargs),
        (partial(cmomy.utils.select_moment, name="weight"), _remove_dim_from_kwargs),
        (partial(cmomy.utils.select_moment, name="ave"), _remove_dim_from_kwargs),
        (
            partial(cmomy.utils.assign_moment, weight=1),
            _remove_dim_from_kwargs,
        ),
        (
            partial(cmomy.utils.assign_moment, ave=1),
            _remove_dim_from_kwargs,
        ),
        (partial(cmomy.rolling.rolling_data, window=2), None),
        (partial(cmomy.rolling.rolling_exp_data, alpha=0.2), None),
    ],
)
def test_func_data_dataset(rng, kwargs, shapes_and_dims, func, kwargs_callback) -> None:
    ds = create_data_dataset(rng, shapes_and_dims, **kwargs)
    kws = kwargs_callback(kwargs.copy()) if kwargs_callback else kwargs

    # coordinates along sampled dimension
    out = func(ds, **kws)

    if "dim" in kws:
        kws = {"move_axis_to_end": True, **kws}

    for k in ds:
        da = ds[k]

        if ("dim" not in kws or kws["dim"] in da.dims) and (
            "mom_dims" not in kws or kws["mom_dims"] in da.dims
        ):
            da = func(da, **kws)
    xr.testing.assert_allclose(out[k], da)


@pytest.mark.parametrize(
    ("func", "kwargs_callback"),
    [
        (cmomy.reduce_vals, None),
        (partial(_resample_vals, nrep=20, paired=True), None),
        (partial(cmomy.rolling.rolling_vals, window=2), None),
        (partial(cmomy.rolling.rolling_exp_vals, alpha=0.2), None),
        (cmomy.resample.jackknife_vals, None),
        (cmomy.utils.vals_to_data, _remove_dim_from_kwargs),
    ],
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
                dy = y if isinstance(y, xr.DataArray) else y[name]
                xy = (da, dy)
            else:
                xy = (da,)

            if weight is not None:
                w = weight if isinstance(weight, xr.DataArray) else weight[name]
            else:
                w = weight

            da = func(*xy, **kwargs, weight=w)

        xr.testing.assert_allclose(out[name], da)


# * Special cases...
# ** reduce data
def _do_reduce_data_dataset(ds, dim, **kwargs) -> None:
    out = cmomy.reduce_data(ds, dim=dim, **kwargs)

    for name in ds:
        da = ds[name]
        if (dim is None or dim in da.dims) and (
            kwargs.get("use_reduce", True)
            or ("mom_dims" not in kwargs or kwargs["mom_dims"] in da.dims)
        ):
            da = cmomy.reduce_data(da, dim=dim, **kwargs)

        xr.testing.assert_allclose(out[name], da)


@mark_data_kwargs
@pytest.mark.parametrize("use_reduce", [True, False])
def test_reduce_data_dataset(rng, kwargs, shapes_and_dims, use_reduce) -> None:
    return _do_reduce_data_dataset(
        create_data_dataset(rng, shapes_and_dims, **kwargs),
        **kwargs,
        use_reduce=use_reduce,
    )


@pytest.mark.parametrize(
    ("kwargs", "shapes_and_dims"),
    [
        # NOTE: make data0, data1 same dimensions, otherwise use_reduce leads to different answers....
        (
            {"mom_ndim": 1, "dim": None},
            [((10, 2, 3), ("a", "b", "mom")), ((10, 2, 3), ("a", "b", "mom"))],
        ),
        (
            {"mom_ndim": 2, "dim": None},
            [
                ((2, 10, 3, 3), ("a", "b", "mom0", "mom1")),
                ((2, 10, 3, 3), ("a", "b", "mom0", "mom1")),
            ],
        ),
    ],
)
@pytest.mark.parametrize("use_reduce", [True, False])
def test_reduce_data_dataset_none(rng, kwargs, shapes_and_dims, use_reduce) -> None:
    return _do_reduce_data_dataset(
        create_data_dataset(rng, shapes_and_dims, **kwargs),
        **kwargs,
        use_reduce=use_reduce,
    )


# * Resample
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
        rng=get_zero_rng(),
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
                ndat=data.sizes[dim], nrep=nrep, rng=get_zero_rng()
            ),
            dims=[rep_dim, dim],
        )

    xr.testing.assert_allclose(out, expected)

    # and should get back self from this
    assert cmomy.resample.randsamp_freq(freq=out, check=False, dtype=None) is out


@mark_data_kwargs
@pytest.mark.parametrize("nrep", [20])
@pytest.mark.parametrize("paired", [False])
def test_resample_data_dataset(
    get_zero_rng, rng, kwargs, shapes_and_dims, nrep, paired
) -> None:
    ds = create_data_dataset(rng, shapes_and_dims, **kwargs)

    dfreq = cmomy.randsamp_freq(
        data=ds, **kwargs, nrep=nrep, rng=get_zero_rng(), paired=paired
    )

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
                freq=dfreq if isinstance(dfreq, xr.DataArray) else dfreq[name],  # type: ignore[index]
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
            rng=get_zero_rng(),
            paired=paired,
        ),
    )


@pytest.mark.parametrize("nrep", [20])
@pytest.mark.parametrize("paired", [False])
def test_resample_vals_dataset(
    fixture_vals_dataset, get_zero_rng, paired, nrep
) -> None:
    kwargs, x, y, weight = (
        fixture_vals_dataset[k] for k in ("kwargs", "x", "y", "weight")
    )

    dim = kwargs["dim"]
    _rng = get_zero_rng()

    dfreq = cmomy.randsamp_freq(data=x, dim=dim, nrep=nrep, rng=_rng, paired=paired)

    xy = (x,) if y is None else (x, y)
    out = cmomy.resample_vals(*xy, weight=weight, **kwargs, freq=dfreq)

    for name in x:
        da = x[name]
        if kwargs["dim"] in da.dims:
            if y is not None:
                dy = y if isinstance(y, xr.DataArray) else y[name]
                _xy = (da, dy)
            else:
                _xy = (da,)

            if weight is not None:
                w = weight if isinstance(weight, xr.DataArray) else weight[name]
            else:
                w = weight

            da = cmomy.resample_vals(
                *_xy,
                weight=w,
                **kwargs,
                freq=dfreq if isinstance(dfreq, xr.DataArray) else dfreq[name],
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
            rng=get_zero_rng(),
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
    if isinstance(ds, xr.Dataset):
        return ds.chunks != {}

    return ds.chunks is not None


@pytest.mark.slow
@mark_dask_only
@mark_data_kwargs
@pytest.mark.parametrize(
    ("func", "kwargs_callback"),
    [
        (partial(cmomy.reduce_data, use_reduce=False), None),
        (_reduce_data_grouped, None),
        (partial(_reduce_data_indexed, coords_policy=None), None),
        (partial(_resample_data, nrep=20, paired=True), None),
        (partial(_resample_data, nrep=20, paired=False), None),
        (cmomy.resample.jackknife_data, None),
        (cmomy.convert.moments_type, _remove_dim_from_kwargs),
        (cmomy.convert.cumulative, None),
        (cmomy.convert.moments_to_comoments, _moments_to_comoments_kwargs),
        (partial(cmomy.utils.select_moment, name="weight"), _remove_dim_from_kwargs),
        (partial(cmomy.utils.select_moment, name="ave"), _remove_dim_from_kwargs),
        (
            partial(cmomy.utils.assign_moment, weight=1),
            _remove_dim_from_kwargs,
        ),
        (
            partial(cmomy.utils.assign_moment, ave=1),
            _remove_dim_from_kwargs,
        ),
        (partial(cmomy.rolling.rolling_data, window=2), None),
        (partial(cmomy.rolling.rolling_exp_data, alpha=0.2), None),
    ],
)
def test_func_data_chunking(
    rng, kwargs, shapes_and_dims, func, kwargs_callback
) -> None:
    ds = create_data_dataset(rng, shapes_and_dims, **kwargs)
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
    [
        (cmomy.reduce_vals, None),
        (partial(_resample_vals, nrep=20, paired=True), None),
        (partial(_resample_vals, nrep=20, paired=False), None),
        (cmomy.resample.jackknife_vals, None),
        (cmomy.utils.vals_to_data, _remove_dim_from_kwargs),
        (partial(cmomy.rolling.rolling_vals, window=2), None),
        (partial(cmomy.rolling.rolling_exp_vals, alpha=0.2), None),
    ],
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
                dy = y if isinstance(y, xr.DataArray) else y[k]
                xy = (da, dy)
                xy_chunked = (x_chunked[k], dy)
            else:
                xy = (da,)
                xy_chunked = (x_chunked[k],)

            if weight is not None:
                w = weight if isinstance(weight, xr.DataArray) else weight[k]
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
        (_reduce_data_grouped, None),
        (partial(_reduce_data_indexed, coords_policy=None), None),
        (partial(_resample_data, nrep=20, paired=True), None),
        (partial(_resample_data, nrep=20, paired=False), None),
        (cmomy.resample.jackknife_data, None),
        (cmomy.convert.moments_type, _remove_dim_from_kwargs),
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
        (partial(_resample_vals, nrep=20, paired=True), None),
        (partial(_resample_vals, nrep=20, paired=False), None),
        (cmomy.resample.jackknife_vals, None),
        # (cmomy.utils.vals_to_data, _remove_dim_from_kwargs),  # noqa: ERA001
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

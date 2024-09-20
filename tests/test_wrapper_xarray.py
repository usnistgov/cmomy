# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import xarray as xr

import cmomy

try:
    import dask  # noqa: F401  # pyright: ignore[reportUnusedImport, reportMissingImports]

    HAS_DASK = True
except ImportError:
    HAS_DASK = False

mark_dask_only = pytest.mark.skipif(not HAS_DASK, reason="dask not installed")


@pytest.fixture
def c_dataset() -> cmomy.CentralMomentsData[xr.Dataset]:
    return cmomy.CentralMomentsData(
        xr.Dataset(
            {
                "a": xr.DataArray(
                    cmomy.default_rng().random((2, 3, 4)),
                    attrs={"hello": "there"},
                    coords={"dim_0": range(2)},
                ),
            }
        )
    )


def get_first(c) -> cmomy.CentralMomentsData[xr.DataArray]:
    return next(iter(c.values()))


def test_init_error() -> None:
    with pytest.raises(TypeError):
        cmomy.CentralMomentsData(np.zeros(3))  # type: ignore[type-var]


@pytest.mark.parametrize("method", ["dtype", "shape", "val_shape", "ndim", "val_ndim"])
def test_dataset_not_implemented(c_dataset, method) -> None:
    with pytest.raises(NotImplementedError):
        getattr(c_dataset, method)


def test_dataset_not_implemented_2(c_dataset) -> None:
    with pytest.raises(NotImplementedError):
        c_dataset.moveaxis(0, -1)

    with pytest.raises(NotImplementedError):
        c_dataset.std()

    with pytest.raises(ZeroDivisionError):
        _ = c_dataset - c_dataset


def test_pipe(c_dataset) -> None:
    c2 = c_dataset.pipe(lambda x: x + 1, _reorder=False)
    xr.testing.assert_allclose(c2.obj, c_dataset.obj + 1)


def test_as_dict(c_dataset) -> None:
    # add a new variable missing mom_dims...
    c = c_dataset.assign(other=xr.DataArray(np.zeros(3), dims=["b"]))
    d = c.as_dict()
    assert list(d.keys()) == list(c.keys())
    for k, x in zip(c.keys(), c.values()):
        xr.testing.assert_allclose(x.obj, c[k].obj)
    for k, x in c.items():
        xr.testing.assert_allclose(x.obj, c[k].obj)  # noqa: PLR1733

    c = get_first(c_dataset)
    with pytest.raises(NotImplementedError):
        c.as_dict()  # pyright: ignore[reportAttributeAccessIssue]


def test_getitem(c_dataset) -> None:
    with pytest.raises(KeyError):
        c_dataset["_hello"]

    for k, v in c_dataset.obj.items():
        d = c_dataset[k]
        xr.testing.assert_equal(d.obj, v)

        with pytest.raises(ValueError):
            _ = d[..., 0]


def test_iter(c_dataset) -> None:
    c = c_dataset
    assert list(c) == list(c.keys())

    ca = get_first(c_dataset)
    for c, expected in zip(ca, ca.obj):
        xr.testing.assert_allclose(c.obj, expected)

    c = cmomy.wrap(xr.DataArray(np.zeros(3)))
    with pytest.raises(ValueError, match=".*Can only iterate.*"):
        list(c)


def test__dtype(c_dataset) -> None:
    d = get_first(c_dataset)._dtype
    assert d is not None
    assert d.name == "float64"
    assert c_dataset._dtype is None


def test_new_like_errors() -> None:
    c = cmomy.wrap(np.zeros((2, 3))).to_x()

    # missing mom_dims
    with pytest.raises(ValueError, match=".*Cannot create.*"):
        c.new_like(xr.DataArray(np.zeros((2, 3))))

    # with strict verify
    with pytest.raises(ValueError, match=".*Wrong `obj.sizes`"):
        c.new_like(xr.DataArray(np.zeros((3,)), dims=c.dims[-1]), verify=True)  # type: ignore[arg-type]

    new = c.new_like(copy=True)
    assert not np.shares_memory(new.obj, c.obj)

    new = c.new_like(fastpath=True)
    xr.testing.assert_allclose(c.obj, new.obj)

    d = np.zeros((2, 3))
    # respect copy with fastpath=False
    new = c.new_like(d, fastpath=False, copy=True)
    assert not np.shares_memory(new, d)

    # ignore copy with fastpath=True
    new = c.new_like(d, fastpath=True, copy=True)
    assert np.shares_memory(new, d)

    new = c.new_like(dtype=np.float32)
    assert new.dtype.type == np.float32


@pytest.mark.parametrize(
    ("func", "method", "shape", "dims", "dim"),
    [
        (
            partial(cmomy.reduce_data, mom_ndim=1),
            "push_datas",
            (20, 2, 3),
            ("rec", "a", "mom"),
            "rec",
        ),
        (
            partial(cmomy.reduce_vals, mom=2),
            "push_vals",
            (20, 2),
            ("rec", "a"),
            "rec",
        ),
    ],
)
@pytest.mark.parametrize("as_dataset", [True, False])
def test_push_datas_and_vals(rng, as_dataset, func, method, shape, dims, dim) -> None:
    data = xr.DataArray(rng.random(shape), dims=dims)
    if as_dataset:
        data = data.to_dataset(name="data")  # type: ignore[assignment]

    expected = func(data, dim=dim)
    base = cmomy.wrap(xr.zeros_like(expected))

    xr.testing.assert_allclose(
        getattr(base.zeros_like(), method)(data, dim=dim).obj,
        expected,
    )

    da = data["data"] if as_dataset else data
    xr.testing.assert_allclose(
        getattr(base.zeros_like(), method)(da, dim=dim).obj,
        expected,
    )
    xr.testing.assert_allclose(
        getattr(base.zeros_like(), method)(
            da.to_numpy(), axis=da.get_axis_num(dim)
        ).obj,
        expected,
    )


@pytest.mark.parametrize("policy", ["group", "first", "last", None])
def test_reduce_data_grouped(rng, policy) -> None:
    by = [0] * 5 + [1] * 5

    data = xr.DataArray(
        rng.random((10, 3)),
        dims=["rec", "mom"],
        coords={"by": ("rec", by)},
    )

    dim = "rec"
    kws = {"dim": dim, "coords_policy": policy}

    if policy in {"first", "last"}:
        _, index, start, end = cmomy.reduction.factor_by_to_index(by)
        expected = cmomy.reduction.reduce_data_indexed(
            data, **kws, index=index, group_start=start, group_end=end
        )
    else:
        expected = cmomy.reduce_data_grouped(data, dim=dim, by=by)

    a = cmomy.wrap(data).reduce(by=by, **kws)
    b = cmomy.wrap(data).reduce(by="by", **kws)

    xr.testing.assert_allclose(a.obj, expected)
    xr.testing.assert_allclose(b.obj, expected)


def test_to_dataarray_to_dataset(c_dataset) -> None:
    assert c_dataset is c_dataset.to_dataset()
    xr.testing.assert_equal(
        c_dataset.to_dataarray().obj,
        c_dataset.obj.to_dataarray(),
    )

    c_dataarray = get_first(c_dataset)
    assert c_dataarray.to_dataarray() is c_dataarray
    xr.testing.assert_equal(c_dataarray.to_dataset().obj, c_dataarray.obj.to_dataset())


def test_dims(c_dataset) -> None:
    assert c_dataset.dims == tuple(c_dataset.dims)

    c = c_dataset.to_dataarray()
    assert c.dims == c.obj.dims


def test_name(c_dataset) -> None:
    c = c_dataset
    with pytest.raises(NotImplementedError):
        _ = c.name

    assert c.to_dataarray().name is None

    assert c.to_dataarray(name="hello").name == "hello"


@mark_dask_only
def test_chunk(c_dataset) -> None:
    c = c_dataset
    dim = c.dims[0]
    assert c.chunk({dim: -1}).obj.chunks == c.obj.chunk({dim: -1}).chunks


@pytest.mark.parametrize(
    "attr",
    ["sizes", "attrs"],
)
def test_attributes(c_dataset, attr) -> None:
    assert getattr(c_dataset, attr) == getattr(c_dataset, attr)


def test_coords(c_dataset) -> None:
    xr.testing.assert_allclose(
        c_dataset.coords.to_dataset(), c_dataset.obj.coords.to_dataset()
    )


@mark_dask_only
def test_as_numpy(c_dataset) -> None:
    assert c_dataset.chunk({c_dataset.dims[0]: -1}).as_numpy().obj.chunks == {}


@mark_dask_only
def test_compute(c_dataset) -> None:
    assert c_dataset.chunk({c_dataset.dims[0]: -1}).compute().obj.chunks == {}


def test_assign_attrs(c_dataset) -> None:
    c = c_dataset
    assert (
        c.assign_attrs(thing="something").attrs
        == c.obj.assign_attrs(thing="something").attrs
    )


def test_assign(c_dataset) -> None:
    other = xr.DataArray(np.zeros(3), dims=["_b"])

    c = c_dataset.assign(other=other)
    xr.testing.assert_allclose(c.obj["other"], other)

    with pytest.raises(NotImplementedError):
        get_first(c_dataset).assign(other=other)


@pytest.mark.parametrize(
    ("method", "args", "kwargs"),
    [
        ("assign_coords", (), {"other": ("dim_0", range(2))}),
        ("rename", (), {"dim_0": "dim"}),
        ("swap_dims", (), {"dim_0": "dim"}),
        ("transpose", ("dim_1", ...), {}),
    ],
)
def test_xarray_methods(c_dataset, method, args, kwargs, as_dataarray) -> None:
    c = get_first(c_dataset) if as_dataarray else c_dataset

    xr.testing.assert_allclose(
        getattr(c, method)(*args, **kwargs).obj,
        getattr(c.obj, method)(*args, **kwargs),
    )


@pytest.fixture
def c_xarray_method(
    c_dataset: cmomy.CentralMomentsData[xr.Dataset],
) -> cmomy.CentralMomentsData[xr.Dataset]:
    return c_dataset.assign_coords(dim_0=range(2), other=("dim_0", [0, 0]))


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("set_index", {"z": ["dim_0", "other"]}),
        ("drop_vars", {"names": "dim_0"}),
    ],
)
def test_xarray_methods_2(c_xarray_method, method, kwargs, as_dataarray) -> None:
    c = get_first(c_xarray_method) if as_dataarray else c_xarray_method
    xr.testing.assert_allclose(
        getattr(c, method)(**kwargs).obj,
        getattr(c.obj, method)(**kwargs),
    )


def test_reset_index(c_xarray_method, as_dataarray) -> None:
    c = c_xarray_method.set_index(z=["dim_0", "other"])
    if as_dataarray:
        c = get_first(c)
    xr.testing.assert_allclose(
        c.reset_index("z").obj,
        c.obj.reset_index("z"),
    )


def test_to_array(c_dataset) -> None:
    with pytest.raises(NotImplementedError):
        _ = c_dataset.to_array()

    c = get_first(c_dataset)
    a = c.to_array(copy=True)
    assert not np.shares_memory(a, c)

    a = c.to_array()
    assert np.shares_memory(a, c)

    assert isinstance(a, cmomy.CentralMomentsArray)
    assert isinstance(c.to_c(), cmomy.CentralMomentsArray)

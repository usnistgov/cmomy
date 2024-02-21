# mypy: disable-error-code="no-untyped-def, no-untyped-call"

import numpy as np
import pytest
import xarray as xr

import cmomy
from cmomy import xcentral


def xtest(a, b) -> None:
    xr.testing.assert_allclose(a, b.transpose(*a.dims))


def test_fix_test(other) -> None:
    np.testing.assert_allclose(other.data_test_xr, other.data_fix)


def test_s(other) -> None:
    xtest(other.data_test_xr, other.s_xr.to_dataarray())


def scramble_xr(x, rng=None):
    if rng is None:
        rng = cmomy.random.default_rng()

    if isinstance(x, tuple):
        return tuple(scramble_xr(_) for _ in x)

    if isinstance(x, xr.DataArray):
        order = list(x.dims)
        rng.shuffle(order)  # pyright: ignore[reportArgumentType]
        return x.transpose(*order)
    return x


def test_select_axis_dim() -> None:
    from cmomy.xcentral import _select_axis_dim

    dims = ("a", "b", "mom")

    with pytest.raises(ValueError):
        _select_axis_dim(dims)

    with pytest.raises(ValueError):
        _select_axis_dim(dims, default_axis=0, default_dim="hello")

    with pytest.raises(ValueError):
        _select_axis_dim(dims, axis=0, dim="a")

    assert _select_axis_dim(dims, default_axis=0) == (0, "a")
    assert _select_axis_dim(dims, default_axis=-1) == (-1, "mom")

    assert _select_axis_dim(dims, default_dim="a") == (0, "a")
    assert _select_axis_dim(dims, default_dim="mom") == (2, "mom")

    with pytest.raises(ValueError):
        _select_axis_dim(dims, dim="hello")

    with pytest.raises(ValueError):
        _select_axis_dim(dims, axis="a")  # type: ignore[arg-type]


def test_move_mom_dims_to_end() -> None:
    from cmomy.xcentral import _move_mom_dims_to_end

    x = xr.DataArray(np.zeros((2, 3, 4)), dims=["a", "b", "c"])

    assert _move_mom_dims_to_end(x, mom_dims=None) is x
    assert _move_mom_dims_to_end(x, mom_dims="a").dims == ("b", "c", "a")
    assert _move_mom_dims_to_end(x, mom_dims="b").dims == ("a", "c", "b")
    assert _move_mom_dims_to_end(x, mom_dims=("b", "a")).dims == ("c", "b", "a")

    with pytest.raises(ValueError):
        _move_mom_dims_to_end(x, mom_dims="a", mom_ndim=2)


def test__xcentral_moments() -> None:
    from cmomy.xcentral import _xcentral_moments

    x = xr.DataArray(np.zeros((2, 3)), dims=("a", "b"))

    with pytest.raises(TypeError):
        _xcentral_moments([1, 2, 3], mom=3)  # type: ignore[arg-type]

    assert _xcentral_moments(x, mom=3, mom_dims="mom", dim="a").dims == ("b", "mom")

    with pytest.raises(ValueError):
        _xcentral_moments(x, mom=3, mom_dims=("c", "d"))


def test__xcentral_comoments() -> None:
    from cmomy.xcentral import _xcentral_comoments

    x = xr.DataArray(np.zeros((2, 3)), dims=("a", "b"))
    xy = (x, x)

    with pytest.raises(ValueError):
        _xcentral_comoments(xy, mom=(2,))

    with pytest.raises(TypeError):
        _xcentral_comoments([x, x], mom=2)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        _xcentral_comoments((x, x, x), mom=2)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        _xcentral_comoments((x.to_numpy(), x), mom=2)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        _xcentral_comoments((x, x.to_numpy()), mom=2)  # type: ignore[arg-type]

    y = xr.zeros_like(x).rename({"a": "_a", "b": "_b"})
    with pytest.raises(ValueError):
        _xcentral_comoments((x, y), mom=2)

    y = xr.DataArray(np.zeros((3, 4)), dims=("a", "b"))
    with pytest.raises(ValueError):
        _xcentral_comoments((x, y), mom=2)

    with pytest.raises(ValueError):
        _xcentral_comoments(xy, mom=2, mom_dims=("mom0",))  # type: ignore[arg-type]


def test_raises_create() -> None:
    x = xr.DataArray(np.zeros((2, 3, 4)), dims=("a", "b", "mom"))

    with pytest.raises(TypeError):
        cmomy.xCentralMoments(np.zeros((2, 3)))  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        cmomy.xCentralMoments(x, mom_ndim=3)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        cmomy.xCentralMoments(xr.DataArray([1], dims="a"), mom_ndim=2)

    with pytest.raises(ValueError):
        cmomy.xCentralMoments(xr.DataArray(np.zeros((2, 1))), mom_ndim=1)


def test_new_like() -> None:
    c = cmomy.xCentralMoments.zeros(mom=2, val_shape=(2, 3), dims=("a", "b", "mom"))

    n = c.new_like(data=c.to_dataarray())

    n.data[...] = 1

    np.testing.assert_allclose(c.data, n.data)

    c.zero()

    data = np.ones_like(c.data)

    n = c.new_like(data)

    n.data[...] = 1

    np.testing.assert_allclose(n.data, data)


def test_new_like2() -> None:
    c = cmomy.xCentralMoments.zeros(val_shape=(2, 3), mom=2, dims=("a", "b", "mom"))

    x0 = c.new_like()

    assert x0.shape == (2, 3, 3)
    assert x0.dims == ("a", "b", "mom")

    x1 = c.new_like(strict=True)
    assert x1.shape == (2, 3, 3)
    assert x1.dims == ("a", "b", "mom")

    x2 = cmomy.xCentralMoments.from_data(
        np.zeros((2, 3, 4)), dims=("a", "b", "mom"), mom_ndim=1
    )

    assert x2.shape == (2, 3, 4)
    assert x2.dims == ("a", "b", "mom")

    with pytest.raises(ValueError):
        c.new_like(np.zeros((2, 3, 4)), strict=True)


def test__single_index_selector() -> None:
    c = cmomy.xCentralMoments.from_data(np.arange(3), mom_ndim=1)

    xr.testing.assert_allclose(c._single_index_dataarray(0), xr.DataArray(0))
    xr.testing.assert_allclose(c._single_index_dataarray(1), xr.DataArray(1))
    xr.testing.assert_allclose(c._single_index_dataarray(2), xr.DataArray(2))

    c = cmomy.xCentralMoments.from_data(np.arange(3 * 3).reshape(3, 3), mom_ndim=2)

    xr.testing.assert_allclose(
        c._single_index_dataarray(0),
        xr.DataArray([0, 0], dims="variable", coords={"variable": ["mom_0", "mom_1"]}),
    )

    xr.testing.assert_allclose(
        c._single_index_dataarray(1, dim_combined="v", coords_combined=["a", "b"]),
        xr.DataArray([3, 1], dims="v", coords={"v": ["a", "b"]}),
    )

    xr.testing.assert_allclose(
        c._single_index_dataarray(2),
        xr.DataArray([6, 2], dims="variable", coords={"variable": ["mom_0", "mom_1"]}),
    )


def test_transpose() -> None:
    c = cmomy.xCentralMoments.zeros(
        val_shape=(2, 3), mom=(4, 6), dims=("a", "b", "mom_0", "mom_1")
    )

    assert c.transpose("b", "a").dims == ("b", "a", "mom_0", "mom_1")
    assert c.transpose("mom_1", "mom_0", "b", "a").dims == ("b", "a", "mom_0", "mom_1")


def test__xverify_value(rng) -> None:
    c = cmomy.xCentralMoments.zeros(val_shape=(2,), mom=3, dims=("a", "mom"))
    datas = xr.DataArray(rng.random((10, 2, 4)), dims=("rec", "a", "mom"))
    values, target_output = c._xverify_value(x=datas, target="datas", dim="rec", axis=0)

    np.testing.assert_allclose(values, datas)
    np.testing.assert_allclose(values, target_output)

    with pytest.raises(ValueError):
        c._xverify_value(x=datas, target="hello", dim="rec")

    x = xr.DataArray(rng.random((10, 2)), dims=("rec", "a"))

    a = cmomy.xCentralMoments.from_vals(x, dim="rec", val_shape=2, mom=2)  # type: ignore[arg-type]
    b = cmomy.xCentralMoments.from_vals(x, axis=0, mom=2)

    xr.testing.assert_allclose(a.to_dataarray(), b.to_dataarray())

    c = a.zeros_like()

    c.push_vals(x, w=np.ones(10), dim="rec")
    xr.testing.assert_allclose(a.to_dataarray(), c.to_dataarray())

    c.zero()
    c.push_vals(x, w=np.ones((10, 2)), dim="rec")
    xr.testing.assert_allclose(a.to_dataarray(), c.to_dataarray())

    with pytest.raises(ValueError):
        c.push_vals(x, w=np.ones((10, 3)), dim="rec")


def test_create(other) -> None:
    t = xcentral.xCentralMoments.zeros(mom=other.mom, val_shape=other.val_shape)

    # from array
    t.push_vals(other.x, w=other.w, axis=other.axis, broadcast=other.broadcast)
    xtest(other.data_test_xr, t.to_dataarray())

    # from xarray
    t.zero()
    t.push_vals(
        x=scramble_xr(other.x_xr),
        w=scramble_xr(other.w_xr),
        dim="rec",
        broadcast=other.broadcast,
    )
    xtest(other.data_test_xr, t.to_dataarray())


def test_zeros() -> None:
    template = xr.DataArray(np.zeros((2, 3, 4)), dims=("a", "b", "mom"))

    new = cmomy.xCentralMoments.zeros(mom_ndim=1, template=template)

    assert new.shape == (2, 3, 4)
    assert new.dims == ("a", "b", "mom")


def test_from_vals(other) -> None:
    t = xcentral.xCentralMoments.from_vals(
        x=other.x, w=other.w, mom=other.mom, axis=other.axis, broadcast=other.broadcast
    )
    xtest(other.data_test_xr, t.to_dataarray())

    t = xcentral.xCentralMoments.from_vals(
        x=scramble_xr(other.x_xr),
        w=scramble_xr(other.w_xr),  # pyright: ignore[reportArgumentType]
        dim="rec",
        mom=other.mom,
        broadcast=other.broadcast,
    )
    xtest(other.data_test_xr, t.to_dataarray())


def test_mean_var_simple(rng) -> None:
    x = xr.DataArray(rng.random((100, 4)), dims=("rec", "a"))

    c = cmomy.xCentralMoments.from_vals(x, mom=3, dim="rec")

    xr.testing.assert_allclose(c.mean(), x.mean(dim="rec"))
    xr.testing.assert_allclose(c.var(), x.var(dim="rec"))


def test_assign_xr_attr() -> None:
    c = cmomy.xCentralMoments.zeros(val_shape=(2, 3), mom=3, dims=("a", "b", "mom"))

    assert c.rename("hello").name == "hello"

    assert c.assign_attrs({"thing": "other"}).attrs == {"thing": "other"}


def test_from_vals2() -> None:
    x = xr.DataArray(np.zeros((10, 2)), dims=("rec", "a"), attrs={"hello": "there"})

    t = cmomy.xCentralMoments.from_vals(x, dim="rec", mom=2)

    assert t.attrs == {"hello": "there"}

    t = cmomy.xCentralMoments.from_vals(x, dim="rec", mom=2, attrs={"thing": "other"})

    assert t.attrs == {"hello": "there", "thing": "other"}


def test_push_val(other) -> None:
    if other.axis == 0 and other.style == "total" and other.s._mom_ndim == 1:
        t = other.s_xr.zeros_like()
        for ww, xx in zip(other.w, other.x):
            t.push_val(x=xx, w=ww, broadcast=other.broadcast)
        xtest(other.data_test_xr, t.to_dataarray())

        t.zero()
        for ww, xx in zip(other.w_xr, other.x_xr):
            t.push_val(x=scramble_xr(xx), w=scramble_xr(ww), broadcast=other.broadcast)
        xtest(other.data_test_xr, t.to_dataarray())


def test_push_vals_mult(other) -> None:
    t = other.s_xr.zeros_like()
    for ww, xx in zip(other.W, other.X):
        t.push_vals(x=xx, w=ww, axis=other.axis, broadcast=other.broadcast)
    xtest(other.data_test_xr, t.to_dataarray())

    t.zero()
    for ww, xx in zip(other.W_xr, other.X_xr):
        t.push_vals(
            x=scramble_xr(xx), w=scramble_xr(ww), dim="rec", broadcast=other.broadcast
        )
    xtest(other.data_test_xr, t.to_dataarray())


def test_combine(other) -> None:
    t = other.s_xr.zeros_like()
    for s in other.S_xr:
        t.push_data(scramble_xr(s.to_dataarray()))
    xtest(other.data_test_xr, t.to_dataarray())


def test_from_data() -> None:
    data = np.zeros((2, 3, 4))

    c = cmomy.xCentralMoments.from_data(
        data, mom=3, dims=("a", "b", "mom"), val_shape=(2, 3)
    )

    assert c.dims == ("a", "b", "mom")
    assert c.mom == (3,)

    with pytest.raises(ValueError):
        cmomy.xCentralMoments.from_data(data, mom=2)

    with pytest.raises(ValueError):
        cmomy.xCentralMoments.from_data(data, val_shape=(1, 2), mom=3)

    data = xr.DataArray(np.zeros((2, 3, 4)), dims=("a", "b", "mom"))  # type: ignore[assignment]

    c = cmomy.xCentralMoments.from_data(data, mom=3, val_shape=(2, 3), copy_kws={})

    assert c.dims == ("a", "b", "mom")
    assert c.mom == (3,)

    with pytest.raises(ValueError):
        cmomy.xCentralMoments.from_data(data, mom=2)

    with pytest.raises(ValueError):
        cmomy.xCentralMoments.from_data(data, val_shape=(1, 2), mom=3)


def test_from_datas(other) -> None:
    # set the rng for reproduciblility right now:
    cmomy.random.default_rng(0)

    datas = xr.concat([s.to_dataarray() for s in other.S_xr], dim="rec")
    datas = scramble_xr(datas).transpose(*(..., *other.s_xr.mom_dims))  # pyright: ignore[reportAttributeAccessIssue]

    for verify in [True, False]:
        t = other.cls_xr.from_datas(
            datas,
            mom=other.mom,
            dim="rec",
            verify=verify,
        )
        xtest(other.data_test_xr, t.to_dataarray())

    with pytest.raises(ValueError):
        t = other.cls_xr.from_datas(
            datas,
            mom=other.mom,
            dim="rec",
            val_shape=(2, 3, 4, 5, 6, 7),
        )


def test_push_datas(other) -> None:
    datas = xr.concat([s.to_dataarray() for s in other.S_xr], dim="rec")

    datas = scramble_xr(datas).transpose(*(..., *other.s_xr.mom_dims))  # pyright: ignore[reportAttributeAccessIssue]

    t = other.s_xr.zeros_like()
    t.push_datas(datas, dim="rec")
    xtest(other.data_test_xr, t.to_dataarray())


# def test_push_stat(other):
#     if other.s._mom_ndim == 1:

#         t = other.s_xr.zeros_like()
#         for s in other.S_xr:
#             t.push_stat(s.mean(), v=s.to_dataarray()[..., 2:], w=s.weight())
#         xtest(other.data_test_xr, t.to_dataarray())


# def test_from_stat(other):
#     if other.s._mom_ndim == 1:
#         t = other.cls.from_stat(
#             a=other.s.mean(),
#             v=other.s.to_dataarray()[..., 2:],
#             w=other.s.weight(),
#             mom=other.mom,
#         )
#         other.test_values(t.to_dataarray())


# def test_from_stats(other):
#     if other.s._mom_ndim == 1:
#         t = other.s.zeros_like()
#         t.push_stats(
#             a=np.array([s.mean() for s in other.S]),
#             v=np.array([s.to_dataarray()[..., 2:] for s in other.S]),
#             w=np.array([s.weight() for s in other.S]),
#             axis=0,
#         )
#         other.test_values(t.to_dataarray())


def test_add(other) -> None:
    t = other.s_xr.zeros_like()
    for s in other.S_xr:
        t = t + s
    xtest(other.data_test_xr, t.to_dataarray())


def test_sum(other) -> None:
    t = sum(other.S_xr, other.s_xr.zeros_like())
    xtest(other.data_test_xr, t.to_dataarray())  # pyright: ignore[reportAttributeAccessIssue]


def test_iadd(other) -> None:
    t = other.s_xr.zeros_like()
    for s in other.S_xr:
        t += s
    xtest(other.data_test_xr, t.to_dataarray())


def test_sub(other) -> None:
    t = other.s_xr - sum(other.S_xr[1:], other.s_xr.zeros_like())
    xtest(t.to_dataarray(), other.S_xr[0].to_dataarray())


def test_isub(other) -> None:
    t = other.s_xr.copy()
    for s in other.S_xr[1:]:
        t -= s
    xtest(t.to_dataarray(), other.S_xr[0].to_dataarray())


def test_mult(other) -> None:
    s = other.s_xr

    xtest((s * 2).to_dataarray(), (s + s).to_dataarray())

    t = s.copy()
    t *= 2
    xtest(t.to_dataarray(), (s + s).to_dataarray())


def test_resample_and_reduce(other, rng) -> None:
    ndim = len(other.val_shape)

    if ndim > 0:
        for axis in range(ndim):
            ndat = other.val_shape[axis]
            nrep = 10

            idx = rng.choice(ndat, (nrep, ndat), replace=True)

            t0 = other.s.resample_and_reduce(indices=idx, axis=axis)

            dim = f"dim_{axis}"
            t1 = other.s_xr.resample_and_reduce(indices=idx, dim=dim, rep_dim="hello")

            with pytest.raises(ValueError):
                other.s_xr.resample_and_reduce(indices=idx, dim="hello", rep_dim="rep")
            with pytest.raises(ValueError):
                other.s_xr.resample_and_reduce(indices=idx, dim="mom_0")

            np.testing.assert_allclose(t0.data, t1.data)

            # check dims
            dims = list(other.s_xr.to_dataarray().dims)
            dims.pop(axis)
            dims = ("hello", *dims)  # type: ignore[assignment]
            assert t1.to_dataarray().dims == dims

            # resample
            tr = other.s.resample(idx, axis=axis)

            # note: tx may be in different order than tr
            tx = other.s_xr.isel(**{dim: xr.DataArray(idx, dims=["hello", dim])})

            np.testing.assert_allclose(tr.data, tx.transpose("hello", dim, ...).data)

            # # check dims
            # assert tx.dims == ('hello', ) + other.s_xr.to_dataarray().dims

            # reduce
            xtest(t1.to_dataarray(), tx.reduce(dim=dim).to_dataarray())

            # block:
            xtest(
                t1.to_dataarray(),
                tx.block(dim=dim, block_size=None).to_dataarray().isel({dim: 0}),
            )

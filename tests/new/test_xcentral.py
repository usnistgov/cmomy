# mypy: disable-error-code="no-untyped-def, no-untyped-call"
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cmomy.new import resample
from cmomy.new.central_dataarray import xCentralMoments
from cmomy.new.central_numpy import CentralMoments

# import cmomy
# from cmomy import xcentral
from cmomy.random import default_rng


def xtest(a, b) -> None:
    xr.testing.assert_allclose(a, b.transpose(*a.dims))


def test_fix_test(other) -> None:
    np.testing.assert_allclose(other.data_test_xr, other.data_fix)


def test_s(other) -> None:
    xtest(other.data_test_xr, other.s_xr.to_dataarray())


def scramble_xr(*args: xr.DataArray, rng=None) -> tuple[xr.DataArray, ...]:
    if rng is None:
        rng = default_rng()

    out = []
    for x in args:
        if isinstance(x, xr.DataArray):
            order = list(x.dims)
            rng.shuffle(order)  # pyright: ignore[reportArgumentType]
            out.append(x.transpose(*order))
        else:
            out.append(x)
    return tuple(out)


def test_new_like() -> None:
    c = xCentralMoments.zeros(mom=2, val_shape=(2, 3), dims=("a", "b", "mom"))

    n = c.new_like(data=c.to_dataarray())

    n.data[...] = 1

    np.testing.assert_allclose(c.data, n.data)

    c.zero()

    data = np.ones_like(c.data)

    n = c.new_like(data)

    n.data[...] = 1

    np.testing.assert_allclose(n.data, data)


def test_new_like2() -> None:
    c = xCentralMoments.zeros(val_shape=(2, 3), mom=2, dims=("a", "b", "mom"))

    x0 = c.new_like()

    assert x0.shape == (2, 3, 3)
    assert x0.dims == ("a", "b", "mom")

    x1 = c.new_like(verify=True)
    assert x1.shape == (2, 3, 3)
    assert x1.dims == ("a", "b", "mom")

    x2 = CentralMoments.from_data(np.zeros((2, 3, 4)), mom_ndim=1).to_xcentralmoments(
        dims=("a", "b", "mom")
    )

    assert x2.shape == (2, 3, 4)
    assert x2.dims == ("a", "b", "mom")

    with pytest.raises(ValueError):
        c.new_like(np.zeros((2, 3, 4)), verify=True)


def test__single_index_selector() -> None:
    c = CentralMoments.from_data(
        np.arange(3), mom_ndim=1, dtype=np.float64
    ).to_xcentralmoments()

    xr.testing.assert_allclose(c._single_index_dataarray(0), xr.DataArray(0))
    xr.testing.assert_allclose(c._single_index_dataarray(1), xr.DataArray(1))
    xr.testing.assert_allclose(c._single_index_dataarray(2), xr.DataArray(2))

    c = CentralMoments.from_data(
        np.arange(3 * 3).reshape(3, 3), mom_ndim=2, dtype=np.float64
    ).to_xcentralmoments()

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
    c = xCentralMoments.zeros(
        val_shape=(2, 3), mom=(4, 6), dims=("a", "b", "mom_0", "mom_1")
    )

    assert c.transpose("b", "a").dims == ("b", "a", "mom_0", "mom_1")
    assert c.transpose("mom_1", "mom_0", "b", "a").dims == ("b", "a", "mom_0", "mom_1")


def test_create(other) -> None:
    t = xCentralMoments.zeros(mom=other.mom, val_shape=other.val_shape)

    # from array
    t.push_vals(*other.xy_tuple, weight=other.w, axis=other.axis)
    xtest(other.data_test_xr, t.to_dataarray())

    # from xarray
    t.zero()

    t.push_vals(
        *scramble_xr(*other.xy_tuple_xr),
        weight=scramble_xr(other.w_xr)[0],
        dim="rec",
    )
    xtest(other.data_test_xr, t.to_dataarray())


def test_zeros() -> None:
    template = xr.DataArray(np.zeros((2, 3, 4)), dims=("a", "b", "mom"))

    # should we be able to create solely on template?
    new = xCentralMoments.zeros(mom=3, val_shape=(2, 3), template=template)

    assert new.shape == (2, 3, 4)
    assert new.dims == ("a", "b", "mom")


def test_from_vals(other) -> None:
    t = CentralMoments.from_vals(
        *other.xy_tuple,
        weight=other.w,
        mom=other.mom,
        axis=other.axis,
    ).to_xcentralmoments()
    xtest(other.data_test_xr, t.to_dataarray())

    t = xCentralMoments.from_vals(
        *scramble_xr(*other.xy_tuple_xr),
        weight=scramble_xr(other.w_xr)[0],  # pyright: ignore[reportArgumentType]
        dim="rec",
        mom=other.mom,
    )
    xtest(other.data_test_xr, t.to_dataarray())


def test_mean_var_simple(rng) -> None:
    x = xr.DataArray(rng.random((100, 4)), dims=("rec", "a"))

    c = xCentralMoments.from_vals(x, mom=3, dim="rec")

    xr.testing.assert_allclose(c.mean(), x.mean(dim="rec"))
    xr.testing.assert_allclose(c.var(), x.var(dim="rec"))


def test_assign_xr_attr() -> None:
    c = xCentralMoments.zeros(val_shape=(2, 3), mom=3, dims=("a", "b", "mom"))

    assert c.rename("hello").name == "hello"

    assert c.assign_attrs({"thing": "other"}).attrs == {"thing": "other"}


@pytest.mark.parametrize("keep_attrs", [False, True])
def test_from_vals2(keep_attrs) -> None:
    x = xr.DataArray(np.zeros((10, 2)), dims=("rec", "a"), attrs={"hello": "there"})

    t = xCentralMoments.from_vals(x, dim="rec", mom=2, keep_attrs=keep_attrs)
    assert t.attrs == ({"hello": "there"} if keep_attrs else {})

    t = xCentralMoments.from_vals(
        x, dim="rec", mom=2, keep_attrs=keep_attrs
    ).assign_attrs({"thing": "other"})

    assert t.attrs == (
        {"hello": "there", "thing": "other"} if keep_attrs else {"thing": "other"}
    )


def test_push_val(other) -> None:
    if other.axis == 0 and other.style == "total" and other.s.mom_ndim == 1:
        t = other.s_xr.zeros_like()
        for ww, xx in zip(other.w, other.xdata):
            t.push_val(xx, weight=ww)
        xtest(other.data_test_xr, t.to_dataarray())

        t.zero()
        for ww, xx in zip(other.w_xr, other.xdata_xr):
            t.push_val(scramble_xr(xx)[0], weight=scramble_xr(ww)[0])
        xtest(other.data_test_xr, t.to_dataarray())


def test_push_vals_mult(other) -> None:
    t = other.s_xr.zeros_like()
    for ww, xx in zip(other.W, other.X):
        t.push_vals(*xx, weight=ww, axis=other.axis)
    xtest(other.data_test_xr, t.to_dataarray())

    t.zero()
    for ww, xx in zip(other.W_xr, other.X_xr):
        t.push_vals(*scramble_xr(*xx), weight=scramble_xr(ww)[0], dim="rec")
    xtest(other.data_test_xr, t.to_dataarray())


def test_combine(other) -> None:
    t = other.s_xr.zeros_like()
    for s in other.S_xr:
        t.push_data(scramble_xr(s.to_dataarray())[0])
    xtest(other.data_test_xr, t.to_dataarray())


def test_from_data() -> None:
    data = np.zeros((2, 3, 4))

    c = CentralMoments.from_data(
        data,
        mom_ndim=1,
    ).to_xcentralmoments(dims=("a", "b", "mom"))

    assert c.dims == ("a", "b", "mom")
    assert c.mom == (3,)

    with pytest.raises(TypeError):
        xCentralMoments.from_data(data, mom_ndim=1)

    data = xr.DataArray(np.zeros((2, 3, 4)), dims=("a", "b", "mom"))  # type: ignore[assignment]

    c = xCentralMoments.from_data(data, mom_ndim=1)

    assert c.dims == ("a", "b", "mom")
    assert c.mom == (3,)

    # with pytest.raises(ValueError):
    #     xCentralMoments.from_data(data, mom_ndim=3)


def test_from_data_reduce(other) -> None:
    # set the rng for reproduciblility right now:
    import cmomy.new.random

    cmomy.new.random.default_rng(0)

    datas = xr.concat([s.to_dataarray() for s in other.S_xr], dim="rec")
    datas = scramble_xr(datas)[0].transpose(*(..., *other.s_xr.mom_dims))  # pyright: ignore[reportAttributeAccessIssue]

    t = other.cls_xr.from_data(
        datas,
        mom_ndim=other.mom_ndim,
    ).reduce(dim="rec")
    xtest(other.data_test_xr, t.to_dataarray())


def test_push_datas(other) -> None:
    datas = xr.concat([s.to_dataarray() for s in other.S_xr], dim="rec")

    datas = scramble_xr(datas)[0].transpose(*(..., *other.s_xr.mom_dims))  # pyright: ignore[reportAttributeAccessIssue]

    t = other.s_xr.zeros_like()
    t.push_datas(datas, dim="rec")
    xtest(other.data_test_xr, t.to_dataarray())


def test_add(other) -> None:
    t = other.s_xr.zeros_like()
    for s in other.S_xr:
        t += s
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
            freq = resample.indices_to_freq(idx)

            t0 = other.s.resample_and_reduce(freq=freq, axis=axis)

            dim = f"dim_{axis}"
            t1 = other.s_xr.resample_and_reduce(freq=freq, dim=dim, rep_dim="hello")

            with pytest.raises(ValueError):
                other.s_xr.resample_and_reduce(freq=freq, dim="hello", rep_dim="rep")
            with pytest.raises(ValueError):
                other.s_xr.resample_and_reduce(freq=freq, dim="mom_0")

            np.testing.assert_allclose(t0.data, t1.data)

            # check dims
            dims = list(other.s_xr.to_dataarray().dims)
            dims.pop(axis)
            dims = (*dims[: -other.mom_ndim], "hello", *dims[-other.mom_ndim :])  # type: ignore[assignment]
            assert t1.to_dataarray().dims == dims

            # resample
            tr = other.s.resample(idx, axis=axis)

            # note: tx may be in different order than tr
            tx = other.s_xr.isel(**{dim: xr.DataArray(idx, dims=["hello", dim])})

            np.testing.assert_allclose(
                tr.data, tx.transpose(..., "hello", dim, *tx.mom_dims).data
            )

            # # check dims
            # assert tx.dims == ('hello', ) + other.s_xr.to_dataarray().dims

            # reduce
            xtest(t1.to_dataarray(), tx.reduce(dim=dim).to_dataarray())

            # block:
            xtest(
                t1.to_dataarray(),
                tx.block(dim=dim, block_size=None).to_dataarray().isel({dim: 0}),
            )


@pytest.mark.parametrize(
    "selector",
    [
        {"dim_0": [0, 1]},
        {"dim_0": 0},
        {"dim_0": [0]},
        {"mom_0": 0},
    ],
)
def test_isel(selector) -> None:
    cx = CentralMoments.from_data(
        np.arange(4 * 4 * 5 * 5).reshape(4, 4, 5, 5).astype(None), mom_ndim=2
    ).to_x()

    if any(key in cx.mom_dims for key in selector):
        # check that this raises an error
        with pytest.raises(ValueError, match=".*Moments changed.*"):
            cx.isel(selector)

    else:
        xr.testing.assert_equal(
            cx.isel(selector).to_values(),
            cx.to_values().isel(selector),
        )

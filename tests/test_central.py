# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

from cmomy import CentralMoments

if TYPE_CHECKING:
    from cmomy.typing import Mom_NDim


# Tests
def test_fix_test(other) -> None:
    np.testing.assert_allclose(other.data_fix, other.data_test)


@pytest.mark.parametrize(
    ("dtype", "ok"),
    [
        (np.float32, True),
        (np.float64, True),
        (np.int32, False),
        (np.int64, False),
        (np.float16, False),
    ],
)
def test_bad_dtype(dtype, ok) -> None:
    data = np.zeros(4, dtype=dtype)
    if ok:
        CentralMoments(data, mom_ndim=1)
    else:
        with pytest.raises(ValueError, match=".* not supported.* float32 or float64.*"):
            CentralMoments(data, mom_ndim=1)


def test_values() -> None:
    c = CentralMoments.zeros(mom=3)

    assert c.values is c.to_values()  # noqa: PD011


def test_getitem() -> None:
    c = CentralMoments.zeros(val_shape=(2, 3, 4), mom=3)

    assert c[..., 0, :].shape == (2, 3, 4)
    assert c[0, ...].shape == (3, 4, 4)
    assert c[:, 1:, 2:, :].shape == (2, 2, 2, 4)

    with pytest.raises(ValueError, match=".*has wrong mom_shape.*"):
        _ = c[..., 1:]


def test_new_like() -> None:
    c = CentralMoments.zeros(val_shape=(2, 3), mom=2)

    x0 = c.new_like()

    assert x0.shape == (2, 3, 3)

    x1 = c.new_like(verify=True)
    assert x1.shape == (2, 3, 3)

    with pytest.raises(ValueError):
        c.new_like(np.zeros((2, 3, 4)), verify=True)

    # veirfy correct mom_shape, incorrect leading shape
    with pytest.raises(ValueError):
        c.new_like(np.zeros((3, 3, 3)), verify=True)

    # this should work fine without verify
    assert c.new_like(np.zeros((3, 3)), verify=False).shape == (3, 3)

    # no verify by wrong mom shape
    with pytest.raises(ValueError):
        c.new_like(np.zeros((3, 4)), verify=False)


def test_zeros() -> None:
    c = CentralMoments.zeros(val_shape=2, mom=2)
    assert c.shape == (2, 3)
    assert c.mom == (2,)

    c = CentralMoments.zeros(mom=2)
    assert c.shape == (3,)
    assert c.mom == (2,)

    c = CentralMoments.zeros(mom=(2, 2))
    assert c.shape == (3, 3)
    assert c.mom == (2, 2)


# exceptions
def test_raises_centralmoments_init() -> None:
    with pytest.raises(ValueError):
        CentralMoments(np.zeros((2, 3, 4)), mom_ndim=3)  # type: ignore[call-overload]

    with pytest.raises(ValueError):
        CentralMoments([1, 2, 3], mom_ndim=1)

    with pytest.raises(ValueError):
        CentralMoments(np.zeros((1, 1)), mom_ndim=1)

    # fastpath with wrong type
    with pytest.raises(TypeError, match="Must pass ndarray.*"):
        CentralMoments([1, 2, 3], mom_ndim=1, fastpath=True)


def test_raises_zeros() -> None:
    assert CentralMoments.zeros(
        val_shape=2,
        mom=2,
        order="C",
    ).shape == (2, 3)


def test_raises_init() -> None:
    data = np.zeros((1, 2, 3))

    c = CentralMoments(data, mom_ndim=1, copy=False)
    assert np.shares_memory(data, c.data)

    c = CentralMoments(data, mom_ndim=1, copy=True)
    assert not np.shares_memory(data, c.data)


def test_cmom(other) -> None:
    expected = other.data_fix.copy()

    if other.mom_ndim == 1:
        expected[..., 0] = 1.0
        expected[..., 1] = 0.0
    else:
        expected[..., 0, 0] = 1.0
        expected[..., 1, 0] = 0.0
        expected[..., 0, 1] = 0.0

    np.testing.assert_allclose(other.s.cmom(), expected)


def test_to_dataarray() -> None:
    c = CentralMoments.zeros(mom=4, val_shape=(2, 3))

    assert c.to_dataarray(dims=("a", "b")).dims == ("a", "b", "mom_0")
    assert c.to_dataarray(dims=("a", "b"), mom_dims="mom").dims == ("a", "b", "mom")

    with pytest.raises(ValueError):
        c.to_dataarray(dims=("a", "b"), mom_dims=("mom0", "mom1"))

    assert c.to_dataarray(dims=("a", "b", "c")).dims == ("a", "b", "c")

    with pytest.raises(ValueError):
        c.to_dataarray(dims=("a",))

    with pytest.raises(ValueError):
        c.to_dataarray(dims=("a", "b", "c", "d"))

    out = c.to_dataarray(copy=True)

    out[...] = 1

    np.testing.assert_allclose(out, c.data + 1)

    out = c.to_dataarray(copy=False)
    out[...] = 1

    np.testing.assert_allclose(out, c.data)


@pytest.mark.parametrize(
    ("mom_ndim", "dims", "mom_dims", "dims_all"),
    [
        (1, None, None, ("dim_0", "dim_1", "mom_0")),
        (1, ("a", "b"), None, ("a", "b", "mom_0")),
        (1, ("a", "b", "c"), None, ("a", "b", "c")),
        (1, ("a", "b"), "c", ("a", "b", "c")),
        (1, "a", "b", "error"),
        (2, None, None, ("dim_0", "dim_1", "mom_0", "mom_1")),
        (2, ("a", "b"), None, ("a", "b", "mom_0", "mom_1")),
        (2, ("a", "b", "c", "d"), None, ("a", "b", "c", "d")),
        (2, ("a", "b"), ("c", "d"), ("a", "b", "c", "d")),
        (2, ("a",), ("c", "d"), "error"),
    ],
)
def test_to_dataarray2(mom_ndim, dims, mom_dims, dims_all) -> None:
    c = CentralMoments.zeros(mom=(3,) * mom_ndim, val_shape=(2, 3))

    if dims_all == "error":
        with pytest.raises(ValueError):
            c.to_dataarray(dims=dims, mom_dims=mom_dims)

    else:
        expected = xr.DataArray(
            c.data,
            dims=dims_all,
        )

        xr.testing.assert_allclose(
            c.to_dataarray(dims=dims, mom_dims=mom_dims), expected
        )

        xr.testing.assert_allclose(c.to_dataarray(template=expected), expected)


def test_raises_mom_ndim() -> None:
    with pytest.raises(ValueError):
        CentralMoments(np.zeros((4, 4)), mom_ndim=0)  # type: ignore[call-overload]

    with pytest.raises(ValueError):
        CentralMoments(np.zeros((4, 4)), mom_ndim=3)  # type: ignore[call-overload]


def test_raises_data_ndim() -> None:
    with pytest.raises(ValueError):
        CentralMoments(np.zeros(4), mom_ndim=2)


def test_raises_push_cov_bad_x() -> None:
    c = CentralMoments.zeros(mom=(2, 2))
    x = 1.0

    with pytest.raises(ValueError):
        c.push_val(x)

    c.push_val(x, x)

    # vals
    x = [1.0, 2.0]  # type: ignore[assignment]
    with pytest.raises(ValueError):
        c.push_vals(x, axis=0)
    c.push_vals(x, x, axis=0)


def test_pipe(rng) -> None:
    x = rng.random(10)

    c = CentralMoments.from_vals(x, mom=3, axis=0)

    c1 = c.pipe(lambda x: x + 1, _reorder=False)
    c2 = c.pipe("__add__", 1, _reorder=False)
    # c3 = c.pipe("__add__", 1, _reorder=False, _check_mom=False)

    for cc in [c1, c2]:
        np.testing.assert_allclose(cc.data, c.data + 1)

    np.testing.assert_allclose(cc.data, c.pipe(lambda x: x + 1))


def test_raise_if_scalar() -> None:
    c = CentralMoments.zeros(mom=2)

    match = r"Not implemented for scalar"
    with pytest.raises(ValueError, match=match):
        c.block(None, axis=0)

    indices = np.zeros((10, 10), dtype=np.int64)
    with pytest.raises(ValueError, match=match):
        c.resample(indices=indices, axis=0)

    freq = np.zeros((100, 100), dtype=np.int64)
    with pytest.raises(ValueError, match=match):
        c.resample_and_reduce(freq=freq, axis=0)

    with pytest.raises(ValueError, match=match):
        c.reduce(axis=0)

    with pytest.raises(ValueError, match=match):
        c.reshape((2, 2))

    with pytest.raises(ValueError, match=match):
        c.moveaxis(0, 1)


def test_raises_operations(rng) -> None:
    v = rng.random(10)

    c0 = CentralMoments.from_vals(v, mom=2, axis=0)

    data = CentralMoments.from_vals(v, mom=2, axis=0).data

    with pytest.raises(TypeError):
        _ = c0 + data

    c1 = CentralMoments.from_vals(v, mom=3, axis=0)
    with pytest.raises(ValueError):
        _ = c0 + c1

    c3 = c0.zeros_like()

    with pytest.raises(ValueError):
        _ = c3 - c0

    with pytest.raises(ValueError):
        c3 -= c0


def test_wrap_axis() -> None:
    c = CentralMoments.zeros(mom=3, val_shape=(2, 3, 4))

    for axis, expected in [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, "error"),
        (-1, 2),
        (-2, 1),
        (-3, 0),
        (-4, "error"),
    ]:
        if expected == "error":
            with pytest.raises(Exception, match=".*out of bounds.*"):
                c._wrap_axis(axis)

        else:
            assert c._wrap_axis(axis) == expected

    assert c._wrap_axis(-2, ndim=2) == 0


def test_ndim() -> None:
    data = np.empty((1, 2, 3))
    s = CentralMoments(data, mom_ndim=1)
    assert data.ndim == s.ndim


def test_other_data(other) -> None:
    np.testing.assert_allclose(other.data_fix, other.data_test)


def test_s(other) -> None:
    other.test_values(other.s.to_numpy())


def test_push(other) -> None:
    s, xy_tuple, w, axis, _broadcast = other.unpack(
        "s",
        "xy_tuple",
        "w",
        "axis",
        "broadcast",
    )
    t = s.zeros_like()
    t.push_vals(*xy_tuple, weight=w, axis=axis)
    other.test_values(t.to_numpy())


def test_create(other) -> None:
    t = other.cls.zeros(mom=other.mom, val_shape=other.val_shape)
    t.push_vals(*other.xy_tuple, weight=other.w, axis=other.axis)
    other.test_values(t.to_numpy())


def test_from_vals(other) -> None:
    t = other.cls.from_vals(
        *other.xy_tuple,
        weight=other.w,
        axis=other.axis,
        mom=other.mom,
    )
    other.test_values(t.to_values())

    # test var:
    if other.s.mom_ndim == 1 and other.style is None:
        np.testing.assert_allclose(t.var(), np.var(other.xdata, axis=other.axis))
        np.testing.assert_allclose(t.std(), np.std(other.xdata, axis=other.axis))


def test_push_val(other) -> None:
    if other.axis == 0 and other.style == "total":
        t = other.s.zeros_like()
        if other.s.mom_ndim == 1:
            for ww, xx in zip(other.w, other.xdata):
                t.push_val(xx, weight=ww)
            other.test_values(t.to_values())


def test_push_vals_mult(other) -> None:
    t = other.s.zeros_like()
    for ww, xx, _ in zip(other.W, other.X, other.S):
        t.push_vals(*xx, weight=ww, axis=other.axis)
    other.test_values(t.to_values())


@pytest.mark.parametrize("order", ["C", None])
def test_combine(other, order) -> None:
    t = other.s.zeros_like()
    for s in other.S:
        t.push_data(s.to_values(), order=order)
    other.test_values(t.to_values())


def test_init_reduce(other) -> None:
    datas = np.array([s.to_numpy() for s in other.S])
    t = other.cls(datas, mom_ndim=other.mom_ndim).reduce(axis=0)
    other.test_values(t.to_values())


def test_push_datas(other) -> None:
    datas = np.array([s.to_numpy() for s in other.S])
    t = other.s.zeros_like()
    t.push_datas(datas, axis=0)
    other.test_values(t.to_values())


# def test_push_stat(other) -> None:
#     if other.s.mom_ndim == 1:
#         t = other.s.zeros_like()
#         for s in other.S:
#             t.push_stat(s.mean(), v=s.to_numpy()[..., 2:], weight=s.weight())
#         other.test_values(t.to_values())


# TODO(wpk): add this?
# def test_push_stats(other) -> None:
#     if other.s.mom_ndim == 1:
#         datas = np.array([s.to_numpy() for s in other.S])
#         t = other.s.zeros_like()
#         t.push_stats()
#         for s in other.S:
#             t.push_stat(s.mean(), v=s.to_numpy()[..., 2:], w=s.weight())
#         other.test_values(t.to_values())


@pytest.mark.parametrize(("val_shape", "mom"), [(None, 2), ((2, 2), (2, 2))])
def test_push_zero_weight(val_shape, mom, rng) -> None:
    c = CentralMoments.zeros(mom=mom, val_shape=val_shape)

    v = rng.random(val_shape)

    v = (v, v) if isinstance(mom, tuple) else (v,)

    c.push_val(*v, weight=0.0)
    np.testing.assert_allclose(c, 0.0)


@pytest.mark.parametrize(("val_shape", "mom"), [(None, 2), ((2, 2), (2, 2))])
def test_push_order_1(val_shape, mom, rng) -> None:
    cov = isinstance(mom, tuple)
    mom_shape = mom if cov else ()
    shape = (100, *mom_shape)

    x = rng.random(shape)

    mom1 = (1, 1) if cov else 1
    xx = (x, x) if cov else (x,)

    c1 = CentralMoments.zeros(mom=mom1, val_shape=val_shape)  # type: ignore[call-overload]
    c2 = CentralMoments.from_vals(*xx, axis=0, mom=mom)

    # if not cov:
    #     a = x.mean(axis=0)
    #     v = x.var(axis=0)
    #     c1.push_stat(a, v, weight=100.0)
    #     np.testing.assert_allclose(c1.weight(), c2.weight())
    #     np.testing.assert_allclose(c1.mean(), c2.mean())

    #     c1.zero()

    c1.push_vals(*xx, axis=0)
    np.testing.assert_allclose(c1.weight(), c2.weight())
    np.testing.assert_allclose(c1.mean(), c2.mean())


# TODO(wpk): Add this back?
# def test_from_stat(other) -> None:
#     if other.s.mom_ndim == 1:
#         t = other.cls.from_stat(
#             a=other.s.mean(),
#             v=other.s.to_numpy()[..., 2:],
#             w=other.s.weight(),
#             mom=other.mom,
#         )
#         other.test_values(t.to_values())

#         t = other.cls.from_stat(
#             a=other.s.mean(),
#             v=other.s.to_numpy()[..., 2:],
#             w=other.s.weight(),
#             mom=other.mom,
#             dtype=np.float32,
#             val_shape=t.val_shape,
#         )

#         other.test_values(t.to_values(), rtol=1e-4)


# def test_from_stats(other) -> None:
#     if other.s.mom_ndim == 1:
#         a = np.array([s.mean() for s in other.S])
#         v = np.array([s.to_numpy()[..., 2:] for s in other.S])
#         w = np.array([s.weight() for s in other.S])

#         t = other.s.zeros_like()
#         t.push_stats(
#             a=a,
#             v=v,
#             w=w,
#             axis=0,
#         )
#         other.test_values(t.to_values())

#         for val_shape in [None, other.s.val_shape]:
#             t = other.cls.from_stats(
#                 a=a, v=v, w=w, mom=other.s.mom, val_shape=val_shape
#             )
#             other.test_values(t.to_values())

#     else:
#         t = other.s.zeros_like()
#         with pytest.raises(NotImplementedError):
#             t.push_stats(1.0)


def test_add(other) -> None:
    t = other.s.zeros_like()
    for s in other.S:
        t += s
    other.test_values(t.to_values())


def test_sum(other) -> None:
    t = sum(other.S, other.s.zeros_like())
    other.test_values(t.to_values())  # pyright: ignore[reportAttributeAccessIssue]


def test_iadd(other) -> None:
    t = other.s.zeros_like()
    for s in other.S:
        t += s
    other.test_values(t.to_values())


def test_sub(other) -> None:
    t = other.s - sum(other.S[1:], other.s.zeros_like())
    np.testing.assert_allclose(t.to_values(), other.S[0].to_values())


def test_isub(other) -> None:
    t = other.s.copy()
    for s in other.S[1:]:
        t -= s
    np.testing.assert_allclose(t.to_values(), other.S[0].to_values())


def test_mult(other) -> None:
    s = other.s

    np.testing.assert_allclose((s * 2).to_values(), (s + s).to_values())

    t = s.copy()
    t *= 2
    np.testing.assert_allclose(t.to_values(), (s + s).to_values())


def test_reduce(other) -> None:
    ndim = len(other.val_shape)
    if ndim > 0:
        for axis in range(ndim):
            t = other.s.reduce(axis=axis)

            f = other.cls(other.data_test, mom_ndim=other.mom_ndim).reduce(axis=axis)
            np.testing.assert_allclose(t.data, f.data)


def test_reshape(other) -> None:
    ndim = len(other.val_shape)
    if ndim > 0:
        for axis in range(ndim):
            new_shape = list(other.s.val_shape)
            new_shape = tuple(new_shape[:axis] + [1, -1] + new_shape[axis + 1 :])  # type: ignore[assignment]

            t = other.s.reshape(new_shape)

            new_shape2 = new_shape + other.s.mom_shape

            f = other.data_test.reshape(new_shape2)
            np.testing.assert_allclose(t.data, f)


def test_moveaxis(other) -> None:
    ndim = len(other.val_shape)
    if ndim > 1:
        for axis in range(1, ndim):
            # move axis to 0

            t = other.s.moveaxis(axis, 0)

            f = np.moveaxis(other.data_test, axis, 0)

            np.testing.assert_allclose(t.data, f)


@pytest.mark.parametrize("mom_ndim", [1, 2])
def test_block(rng, mom_ndim: Mom_NDim) -> None:
    x = rng.random((10, 10, 10))

    mom: tuple[int] | tuple[int, int]
    if mom_ndim == 1:
        mom = (3,)
        c = CentralMoments.from_vals(x, axis=0, mom=mom)
    else:
        mom = (3, 3)
        y = rng.random((10, 10, 10))
        c = CentralMoments.from_vals(x, y, axis=0, mom=mom)

    c1 = CentralMoments(c.data[::2, ...], mom_ndim=mom_ndim)
    c2 = CentralMoments(c.data[1::2, ...], mom_ndim=mom_ndim)

    # make axis last dimension before moments
    c3 = (c1 + c2).moveaxis(0, -1)

    np.testing.assert_allclose(
        c3.data,
        c.block(2, axis=0).data,
    )

    # using grouped
    group_idx = np.arange(5).repeat(2)
    np.testing.assert_allclose(
        c.reduce(by=group_idx, axis=0).to_numpy(),
        c.block(2, axis=0).to_numpy(),
    )

    c1 = CentralMoments(c.data[:, ::2, ...], mom_ndim=mom_ndim)
    c2 = CentralMoments(c.data[:, 1::2, ...], mom_ndim=mom_ndim)

    # move to last dimension
    c3 = (c1 + c2).moveaxis(1, -1)

    np.testing.assert_allclose(c3.data, c.block(2, axis=1))

    np.testing.assert_allclose(
        c.block(None, axis=0).moveaxis(-1, 0).data[0, ...], c.reduce(axis=0)
    )
    np.testing.assert_allclose(
        c.reduce(by=group_idx, axis=1).to_numpy(), c.block(2, axis=1).to_numpy()
    )


def test_block_odd_size(rng) -> None:
    x = rng.random(10)
    data = np.zeros((10, 3))
    data[:, 0] = 1
    data[:, 1] = x
    data[:, 2] = 0

    c0 = CentralMoments(data, mom_ndim=1).block(3, axis=0)

    c1 = CentralMoments.from_vals(x[:9].reshape(3, -1), mom=2, axis=1)

    np.testing.assert_allclose(c0, c1)

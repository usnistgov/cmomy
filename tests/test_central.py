# mypy: disable-error-code="no-untyped-def, no-untyped-call"
import numpy as np
import pytest

import cmomy
from cmomy import central


# Tests
def test_fix_test(other) -> None:
    np.testing.assert_allclose(other.data_fix, other.data_test)


# test central_moments with out parameter
def test_central_moments_out(other) -> None:
    out = np.zeros_like(other.data_test)

    _ = central.central_moments(
        x=other.x,
        mom=other.mom,
        w=other.w,
        axis=other.axis,
        last=True,
        broadcast=other.broadcast,
        out=out,
    )

    np.testing.assert_allclose(out, other.data_test)

    out.fill(0.0)
    central.central_moments(
        x=other.x,
        mom=other.mom,
        w=other.w,
        axis=other.axis,
        last=False,
        broadcast=other.broadcast,
        out=out,
        dtype=np.float64,
    )

    np.testing.assert_allclose(out, other.data_test)

    out.fill(0.0)

    out2 = out[0, ...]

    with pytest.raises(ValueError):
        _ = central.central_moments(
            x=other.x,
            mom=other.mom,
            w=other.w,
            axis=other.axis,
            last=True,
            broadcast=other.broadcast,
            out=out2,
            dtype=np.float64,
        )

    if other.cov:
        for val, mom, error in [
            (other.x, (3, 3, 3), ValueError),
            ([other.x[0], other.x[1]], (3, 3), TypeError),
            ((other.x[0], other.x[1], other.x[0]), (3, 3), ValueError),
        ]:
            with pytest.raises(error):
                _ = central.central_moments(
                    x=val,
                    mom=mom,  # type: ignore[arg-type]
                    w=other.w,
                    axis=other.axis,
                    last=True,
                    broadcast=other.broadcast,
                    out=out2,
                    dtype=np.float64,
                )

        if len(other.shape) > 2 and other.style == "total":
            out.fill(0.0)
            with pytest.raises(ValueError):
                _ = central.central_moments(
                    x=(other.x[0], other.x[1][0, ...]),
                    mom=other.mom,
                    w=other.w,
                    axis=other.axis,
                    last=True,
                    broadcast=other.broadcast,
                    out=out,
                    dtype=np.float64,
                )


def test_new_like() -> None:
    c = cmomy.CentralMoments.zeros(val_shape=(2, 3), mom=2)

    x0 = c.new_like()

    assert x0.shape == (2, 3, 3)

    x1 = c.new_like(strict=True)
    assert x1.shape == (2, 3, 3)

    x2 = c.new_like(np.zeros((2, 3, 4)), mom=3)
    assert x2.shape == (2, 3, 4)
    assert x2.mom == (3,)

    with pytest.raises(ValueError):
        c.new_like(np.zeros((2, 3, 4)), strict=True)


def test_zeros() -> None:
    with pytest.raises(ValueError):
        cmomy.CentralMoments.zeros(mom=(3,), mom_ndim=2)

    c = cmomy.CentralMoments.zeros(shape=(2, 3, 4), mom_ndim=1)
    assert c.shape == (2, 3, 4)
    assert c.mom == (3,)

    c = cmomy.CentralMoments.zeros(shape=(2, 3, 4), mom_ndim=2)
    assert c.shape == (2, 3, 4)
    assert c.mom == (2, 3)

    with pytest.raises(ValueError):
        c = cmomy.CentralMoments.zeros(val_shape=(2,), mom_ndim=1)

    c = cmomy.CentralMoments.zeros(val_shape=2, mom=2)
    assert c.shape == (2, 3)
    assert c.mom == (2,)

    c = cmomy.CentralMoments.zeros(mom=2)
    assert c.shape == (3,)
    assert c.mom == (2,)

    c = cmomy.CentralMoments.zeros(mom=(2, 2))
    assert c.shape == (3, 3)
    assert c.mom == (2, 2)


# exceptions
def test_raises_centralmoments_init() -> None:
    with pytest.raises(ValueError):
        cmomy.CentralMoments(np.zeros((2, 3, 4)), mom_ndim=3)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        cmomy.CentralMoments([1, 2, 3], mom_ndim=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        cmomy.CentralMoments(np.zeros((1, 1)), mom_ndim=1)


def test_raises_zeros() -> None:
    with pytest.raises(ValueError):
        cmomy.CentralMoments.zeros()

    with pytest.raises(ValueError):
        cmomy.CentralMoments.zeros(shape=(1, 2))

    assert cmomy.CentralMoments.zeros(
        val_shape=2, mom=2, zeros_kws={"order": "C"}
    ).shape == (2, 3)


def test_raises_from_data() -> None:
    data = np.zeros((1, 2, 3))

    with pytest.raises(ValueError):
        cmomy.CentralMoments.from_data(data, val_shape=(2, 2), mom=3)


def test_raises_from_datas() -> None:
    datas = np.zeros((10, 2, 3))

    c = cmomy.CentralMoments.from_datas(datas, mom_ndim=1, dtype=np.float32)
    assert c.shape == (2, 3)
    assert c.dtype == np.dtype(np.float32)

    with pytest.raises(ValueError):
        cmomy.CentralMoments.from_datas(datas, mom=3, val_shape=(4, 5))


def test_raises_from_vals(rng) -> None:
    x = rng.random(10)

    a = cmomy.CentralMoments.from_vals(x, dtype=np.float32, mom=2)
    b = cmomy.CentralMoments.from_vals(x, mom=2)

    assert a.dtype == np.dtype(np.float32)
    assert b.dtype == np.dtype(np.float64)

    np.testing.assert_allclose(a.data, b.data, rtol=1e-5)

    with pytest.raises(ValueError):
        cmomy.CentralMoments.from_vals(x, val_shape=(2, 3), mom=2)


def test_to_xarray() -> None:
    c = cmomy.CentralMoments.zeros(mom=4, val_shape=(2, 3))

    assert c.to_xarray(dims=("a", "b")).dims == ("a", "b", "mom_0")
    assert c.to_xarray(dims=("a", "b"), mom_dims="mom").dims == ("a", "b", "mom")

    with pytest.raises(ValueError):
        c.to_xarray(dims=("a", "b"), mom_dims=("mom0", "mom1"))

    assert c.to_xarray(dims=("a", "b", "c")).dims == ("a", "b", "c")

    with pytest.raises(ValueError):
        c.to_xarray(dims=("a",))

    with pytest.raises(ValueError):
        c.to_xarray(dims=("a", "b", "c", "d"))

    out = c.to_xarray(copy=True)

    out[...] = 1

    np.testing.assert_allclose(out, c.data + 1)

    out = c.to_xarray(copy=False)
    out[...] = 1

    np.testing.assert_allclose(out, c.data)


def test_get_target_shape() -> None:
    c = cmomy.CentralMoments.zeros(mom=3, val_shape=(1, 2))

    x = np.zeros((10, 1, 2))

    assert c._get_target_shape(x, style="val") == (1, 2)

    with pytest.raises(ValueError):
        c._get_target_shape(x, style="vals")

    assert c._get_target_shape(x, style="vals", axis=0) == (10, 1, 2)
    assert c._get_target_shape(x, style="data") == c.shape

    with pytest.raises(ValueError):
        c._get_target_shape(x, style="datas")

    assert c._get_target_shape(x, style="datas", axis=0) == (10, 1, 2, 4)

    with pytest.raises(ValueError):
        c._get_target_shape(x, "vars")

    with pytest.raises(ValueError):
        c._get_target_shape(x, "test")  # type: ignore[arg-type]


def test_verify_value() -> None:
    c = cmomy.CentralMoments.zeros(mom=3, val_shape=(1, 2))

    x = np.zeros((10, 1, 2, 3))

    with pytest.raises(TypeError):
        c._verify_value(x=x, target=[1, 2, 3], shape_flat=c.val_shape_flat)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        c.push_val(x)


def test_raises_mom_ndim() -> None:
    with pytest.raises(ValueError):
        central.CentralMoments(np.zeros((4, 4)), mom_ndim=0)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        central.CentralMoments(np.zeros((4, 4)), mom_ndim=3)  # type: ignore[arg-type]


def test_raises_data_ndim() -> None:
    with pytest.raises(ValueError):
        central.CentralMoments(np.zeros(4), mom_ndim=2)


def test_raises_push_cov_bad_x() -> None:
    c = central.CentralMoments.zeros(mom=(2, 2))
    x = 1.0

    with pytest.raises(TypeError):
        c.push_val(x=x)

    c.push_val(x=(x, x))

    with pytest.raises(ValueError):
        c.push_val(x=(x, x, x))

    # vals
    x = [1.0, 2.0]  # type: ignore[assignment]
    with pytest.raises(TypeError):
        c.push_vals(x=x)

    c.push_vals(x=(x, x))

    with pytest.raises(ValueError):
        c.push_vals(x=(x, x, x))


def test_pipe(rng) -> None:
    x = rng.random(10)

    c = central.CentralMoments.from_vals(x, mom=3)

    c1 = c.pipe(lambda x: x + 1, _order=False)
    c2 = c.pipe("__add__", 1, _order=False)
    c3 = c.pipe("__add__", 1, _order=False, _check_mom=False)

    for cc in [c1, c2, c3]:
        np.testing.assert_allclose(cc.data, c.data + 1)

    with pytest.raises(AttributeError):
        c3 = c.pipe(lambda x: x + 1, _order=True)


def test_raise_if_scalar() -> None:
    c = central.CentralMoments.zeros(mom=2)

    with pytest.raises(ValueError):
        c.block()

    with pytest.raises(ValueError):
        c.resample(cmomy.resample.random_indices(10, 1))

    with pytest.raises(ValueError):
        c.resample_and_reduce(nrep=10)

    with pytest.raises(ValueError):
        c.reduce()

    with pytest.raises(ValueError):
        c.reshape((2, 2))

    with pytest.raises(ValueError):
        c.moveaxis(0, 1)


def test_raises_operations(rng) -> None:
    v = rng.random(10)

    c0 = central.CentralMoments.from_vals(v, mom=2)

    x1 = cmomy.xCentralMoments.from_vals(v, mom=2)

    with pytest.raises(TypeError):
        c0 + x1  # type: ignore[operator]

    for _mom in [3, (2, 2)]:
        c1 = central.CentralMoments.from_vals(v, mom=3)
        with pytest.raises(ValueError):
            _ = c0 + c1

    c3 = c0.zeros_like()

    with pytest.raises(ValueError):
        _ = c3 - c0

    with pytest.raises(ValueError):
        c3 -= c0


def test_wrap_axis() -> None:
    c = cmomy.CentralMoments.zeros(mom=3, val_shape=(2, 3, 4))

    assert c._wrap_axis(-1) == 2
    assert c._wrap_axis(-2, ndim=2) == 0


def test_ndim() -> None:
    data = np.empty((1, 2, 3))

    s = central.CentralMoments(data, 1)

    assert data.ndim == s.ndim


def test_s(other) -> None:
    other.test_values(other.s.to_numpy())


def test_push(other) -> None:
    s, x, w, axis, broadcast = other.unpack(
        "s",
        "x",
        "w",
        "axis",
        "broadcast",
    )
    t = s.zeros_like()
    t.push_vals(x, w=w, axis=axis, broadcast=broadcast)
    other.test_values(t.to_numpy())


def test_create(other) -> None:
    t = other.cls.zeros(mom=other.mom, val_shape=other.val_shape)
    t.push_vals(other.x, w=other.w, axis=other.axis, broadcast=other.broadcast)
    other.test_values(t.to_numpy())


def test_from_vals(other) -> None:
    t = other.cls.from_vals(
        x=other.x, w=other.w, axis=other.axis, mom=other.mom, broadcast=other.broadcast
    )
    other.test_values(t.to_values())

    # test var:
    if other.s.mom_ndim == 1 and other.style is None:
        np.testing.assert_allclose(t.var(), np.var(other.x, axis=other.axis))
        np.testing.assert_allclose(t.std(), np.std(other.x, axis=other.axis))


def test_push_val(other) -> None:
    if other.axis == 0 and other.style == "total":
        t = other.s.zeros_like()
        if other.s.mom_ndim == 1:
            for ww, xx in zip(other.w, other.x):
                t.push_val(x=xx, w=ww, broadcast=other.broadcast)
            other.test_values(t.to_values())


def test_push_vals_mult(other) -> None:
    t = other.s.zeros_like()
    for ww, xx, _ in zip(other.W, other.X, other.S):
        t.push_vals(x=xx, w=ww, axis=other.axis, broadcast=other.broadcast)
    other.test_values(t.to_values())


def test_combine(other) -> None:
    t = other.s.zeros_like()
    for s in other.S:
        t.push_data(s.to_values())
    other.test_values(t.to_values())


def test_from_datas(other) -> None:
    datas = np.array([s.to_numpy() for s in other.S])
    for verify in [True, False]:
        t = other.cls.from_datas(datas, mom=other.mom, verify=verify)
        other.test_values(t.to_values())


def test_push_datas(other) -> None:
    datas = np.array([s.to_numpy() for s in other.S])
    t = other.s.zeros_like()
    t.push_datas(datas)
    other.test_values(t.to_values())


def test_push_stat(other) -> None:
    if other.s.mom_ndim == 1:
        t = other.s.zeros_like()
        for s in other.S:
            t.push_stat(s.mean(), v=s.to_numpy()[..., 2:], w=s.weight())
        other.test_values(t.to_values())


@pytest.mark.parametrize(("val_shape", "mom"), [(None, 2), ((2, 2), (2, 2))])
def test_push_zero_weight(val_shape, mom) -> None:
    rng = cmomy.random.default_rng()

    c = central.CentralMoments.zeros(mom=mom, val_shape=val_shape)

    v = rng.random(val_shape)

    if not isinstance(mom, tuple):
        c.push_stat(v, v, w=0.0)

        np.testing.assert_allclose(c, 0.0)

    else:
        v = (v, v)

    c.push_val(v, w=0.0)

    np.testing.assert_allclose(c, 0.0)


@pytest.mark.parametrize(("val_shape", "mom"), [(None, 2), ((2, 2), (2, 2))])
def test_push_order_1(val_shape, mom) -> None:
    rng = cmomy.random.default_rng()

    cov = isinstance(mom, tuple)
    mom_shape = mom if cov else ()
    shape = (100, *mom_shape)

    x = rng.random(shape)

    mom1 = (1, 1) if cov else 1
    xx = (x, x) if cov else x

    c1 = central.CentralMoments.zeros(mom=mom1, val_shape=val_shape)  # type: ignore[arg-type]
    c2 = central.CentralMoments.from_vals(x=xx, axis=0, mom=mom)  # type: ignore[arg-type]

    if not cov:
        a = x.mean(axis=0)
        v = x.var(axis=0)

        c1.push_stat(a, v, w=100.0)

        np.testing.assert_allclose(c1.weight(), c2.weight())
        np.testing.assert_allclose(c1.mean(), c2.mean())

        c1.zero()

    c1.push_vals(xx, axis=0)  # type: ignore[arg-type]
    np.testing.assert_allclose(c1.weight(), c2.weight())
    np.testing.assert_allclose(c1.mean(), c2.mean())


def test_from_stat(other) -> None:
    if other.s.mom_ndim == 1:
        t = other.cls.from_stat(
            a=other.s.mean(),
            v=other.s.to_numpy()[..., 2:],
            w=other.s.weight(),
            mom=other.mom,
        )
        other.test_values(t.to_values())

        t = other.cls.from_stat(
            a=other.s.mean(),
            v=other.s.to_numpy()[..., 2:],
            w=other.s.weight(),
            mom=other.mom,
            dtype=np.float32,
            val_shape=t.val_shape,
        )

        other.test_values(t.to_values(), rtol=1e-4)


def test_from_stats(other) -> None:
    if other.s.mom_ndim == 1:
        a = np.array([s.mean() for s in other.S])
        v = np.array([s.to_numpy()[..., 2:] for s in other.S])
        w = np.array([s.weight() for s in other.S])

        t = other.s.zeros_like()
        t.push_stats(
            a=a,
            v=v,
            w=w,
            axis=0,
        )
        other.test_values(t.to_values())

        for val_shape in [None, other.s.val_shape]:
            t = other.cls.from_stats(
                a=a, v=v, w=w, mom=other.s.mom, val_shape=val_shape
            )
            other.test_values(t.to_values())

    else:
        t = other.s.zeros_like()
        with pytest.raises(NotImplementedError):
            t.push_stats(1.0)


def test_add(other) -> None:
    t = other.s.zeros_like()
    for s in other.S:
        t = t + s
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
            t = other.s.reduce(axis)

            f = other.cls.from_datas(
                other.data_test, axis=axis, mom_ndim=other.mom_ndim
            )
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


def test_block(rng) -> None:
    x = rng.random((100, 10, 10))

    c = cmomy.CentralMoments.from_vals(x, axis=0, mom=3)

    c1 = cmomy.CentralMoments.from_data(c.data[::2, ...], mom_ndim=1)
    c2 = cmomy.CentralMoments.from_data(c.data[1::2, ...], mom_ndim=1)

    c3 = c1 + c2

    np.testing.assert_allclose(c3.data, c.block(2, axis=0).data)

    c1 = cmomy.CentralMoments.from_data(c.data[:, ::2, ...], mom_ndim=1)
    c2 = cmomy.CentralMoments.from_data(c.data[:, 1::2, ...], mom_ndim=1)

    c3 = (c1 + c2).moveaxis(1, 0)

    np.testing.assert_allclose(c3.data, c.block(2, axis=1).data)

    np.testing.assert_allclose(c.block().data[0, ...], c.reduce())

# mypy: disable-error-code="no-untyped-def, no-untyped-call"
import numpy as np
import pytest

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


# exceptions
def test_mom_ndim() -> None:
    with pytest.raises(ValueError):
        central.CentralMoments(np.zeros((4, 4)), mom_ndim=0)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        central.CentralMoments(np.zeros((4, 4)), mom_ndim=3)  # type: ignore[arg-type]


def test_data_ndim() -> None:
    with pytest.raises(ValueError):
        central.CentralMoments(np.zeros(4), mom_ndim=2)


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
    t = other.cls.from_datas(datas, mom=other.mom)
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


def test_from_stat(other) -> None:
    if other.s.mom_ndim == 1:
        t = other.cls.from_stat(
            a=other.s.mean(),
            v=other.s.to_numpy()[..., 2:],
            w=other.s.weight(),
            mom=other.mom,
        )
        other.test_values(t.to_values())


def test_from_stats(other) -> None:
    if other.s.mom_ndim == 1:
        t = other.s.zeros_like()
        t.push_stats(
            a=np.array([s.mean() for s in other.S]),
            v=np.array([s.to_numpy()[..., 2:] for s in other.S]),
            w=np.array([s.weight() for s in other.S]),
            axis=0,
        )
        other.test_values(t.to_values())


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

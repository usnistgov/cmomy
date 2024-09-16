# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

import functools
import operator
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import xarray as xr

import cmomy
from cmomy import CentralMomentsArray
from cmomy.core.typing import SelectMoment

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


@pytest.fixture(
    params=[
        # shape, mom_ndim
        (3, 1),
        ((10, 2, 3), 1),
        ((10, 3, 3), 2),
    ]
)
def wrapped(rng, request) -> CentralMomentsArray[np.float64]:
    shape, mom_ndim = request.param
    data = rng.random(shape)
    return CentralMomentsArray(data, mom_ndim=mom_ndim)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"obj": np.zeros((2, 3, 4)), "mom_ndim": 3}, ValueError, None),
        (
            {"obj": np.zeros((2, 1)), "mom_ndim": 1},
            ValueError,
            "Moments must be positive",
        ),
        ({"obj": [1, 2, 3], "mom_ndim": 2}, ValueError, ".*Possibly more mom_ndim.*"),
        ({"obj": [1, 2, 3], "mom_ndim": 1, "fastpath": True}, TypeError, "Must pass.*"),
        ({"obj": np.zeros(3, dtype=np.float16), "mom_ndim": 1}, ValueError, None),
    ],
)
def test_init_raises(kwargs, error, match) -> None:
    with pytest.raises(error, match=match):
        CentralMomentsArray(**kwargs)


def test_check_dtype(wrapped) -> None:
    assert wrapped.obj.dtype.type == np.float64


def test_mom(wrapped) -> None:
    assert wrapped.mom_shape == wrapped.obj.shape[-wrapped.mom_ndim :]


def test_properties(wrapped) -> None:
    for attr in ["shape", "dtype", "ndim"]:
        assert getattr(wrapped, attr) == getattr(wrapped.obj, attr)

    assert wrapped.val_shape == wrapped.shape[: -wrapped.mom_ndim]
    assert wrapped.val_ndim == wrapped.ndim - wrapped.mom_ndim


def test_getitem(wrapped) -> None:
    assert np.shares_memory(wrapped[...], wrapped)

    if wrapped.ndim == 1:
        with pytest.raises(ValueError):
            wrapped[0, ...]

    else:
        _ = wrapped[0, ...]

        with pytest.raises(ValueError):
            wrapped[..., 0]


def test_new_like(wrapped) -> None:
    np.testing.assert_allclose(wrapped.new_like().obj, np.zeros_like(wrapped.obj))

    out = np.ones_like(wrapped.obj)
    assert wrapped.new_like(out).obj is out

    # new with different shape but same moments

    out = np.ones(wrapped.obj.shape[-wrapped.mom_ndim :])
    wrapped.new_like(out, verify=False)

    # but with verify get an error
    if out.shape != wrapped.obj.shape:
        with pytest.raises(ValueError):
            wrapped.new_like(out, verify=True)

    # wrong shape
    out = np.ones((5,))
    with pytest.raises(ValueError):
        wrapped.new_like(out, verify=False)

    # using fastpath, should still catch this
    with pytest.raises(ValueError):
        wrapped.new_like(out, fastpath=True)


def test_astype(wrapped) -> None:
    assert wrapped.astype(None).obj is wrapped.obj
    assert wrapped.astype(None, copy=True).obj is not wrapped.obj
    assert wrapped.astype(np.float32).obj.dtype.type == np.float32
    assert wrapped.astype(None, order="F").obj.flags["F_CONTIGUOUS"]


def test_zeros_like(wrapped) -> None:
    new = wrapped.zeros_like()
    assert new.obj.shape == wrapped.obj.shape
    assert new.mom_ndim == wrapped.mom_ndim
    np.testing.assert_allclose(new.obj, 0)


def test_copy(wrapped) -> None:
    new = wrapped.copy()
    assert new.obj.shape == wrapped.obj.shape
    assert new.mom_ndim == wrapped.mom_ndim
    np.testing.assert_allclose(new.obj, wrapped.obj)
    assert not np.shares_memory(new.obj, wrapped.obj)


@pytest.mark.parametrize(
    ("kwargs", "mom_ndim", "shape"),
    [
        ({"mom": 3}, 1, (4,)),
        ({"mom": (3, 3)}, 2, (4, 4)),
        ({"val_shape": 3, "mom": 3}, 1, (3, 4)),
        ({"val_shape": (2, 3), "mom": 3}, 1, (2, 3, 4)),
        ({"val_shape": 3, "mom": (3, 3)}, 2, (3, 4, 4)),
        ({"val_shape": (2, 3), "mom": (3, 3)}, 2, (2, 3, 4, 4)),
    ],
)
def test_zeros(kwargs, mom_ndim, shape) -> None:
    c = CentralMomentsArray.zeros(**kwargs)
    assert c.mom_ndim == mom_ndim
    assert c.obj.shape == shape


def test__raises() -> None:
    new = CentralMomentsArray.zeros(mom=(1, 1))
    with pytest.raises(ValueError):
        new._raise_if_not_mom_ndim_1()

    with pytest.raises(NotImplementedError):
        new._raise_not_implemented()

    with pytest.raises(ValueError):
        new._raise_if_wrong_mom_shape((2,))


def test_to_dataarray() -> None:
    c = CentralMomentsArray.zeros(mom=4, val_shape=(2, 3))

    assert isinstance(c.to_dataarray(), cmomy.CentralMomentsData)

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
    assert not np.shares_memory(out, c.obj)

    out = c.to_dataarray(copy=False)
    assert np.shares_memory(out, c.obj)

    template = xr.DataArray(
        np.zeros((2, 3, 5)),
        dims=["x", "y", "z"],
        attrs={"hello": "there"},
        coords={"x": ["a", "b"]},
    )
    out = c.to_dataarray(template=template)
    xr.testing.assert_equal(out.obj, template)


def test__raise_if_scalar() -> None:
    c = cmomy.wrap(np.zeros(3))
    with pytest.raises(ValueError):
        c._raise_if_scalar()


def test_resample(wrapped) -> None:
    indices = cmomy.random_indices(10, wrapped.shape[0])
    if wrapped.ndim <= wrapped.mom_ndim:
        with pytest.raises(ValueError):
            wrapped.resample(indices, axis=0)

    else:
        freq = cmomy.resample.indices_to_freq(indices)
        expected = wrapped.resample_and_reduce(axis=0, freq=freq)
        np.testing.assert_allclose(
            wrapped.resample(indices, axis=0).reduce(axis=1),
            expected,
        )
        np.testing.assert_allclose(
            wrapped.resample(indices, axis=0, last=True).reduce(axis=-1),
            expected.moveaxis(0, -1),
        )


@pytest.mark.parametrize(
    ("val_shape", "mom"),
    [
        ((10,), (3,)),
        ((10, 3), (3,)),
        ((10,), (3, 3)),
        ((10, 3), (3, 3)),
    ],
)
@pytest.mark.parametrize("use_weight", [False, True])
def test_vals(rng, val_shape, mom, use_weight):
    xy = tuple(rng.random(val_shape) for _ in mom)

    weight = rng.random(val_shape) if use_weight else None

    expected = cmomy.reduce_vals(*xy, weight=weight, axis=0, mom=mom)

    c = CentralMomentsArray.zeros(mom=mom, val_shape=val_shape[1:])

    c.push_vals(*xy, weight=weight, axis=0)
    np.testing.assert_allclose(expected, c)

    c.zero()

    if weight is not None:
        for *_xy, w in zip(*xy, weight):
            c.push_val(*_xy, weight=w)
    else:
        for args in zip(*xy):
            c.push_val(*args)
    np.testing.assert_allclose(expected, c)

    # from_vals
    c = CentralMomentsArray.from_vals(*xy, weight=weight, axis=0, mom=mom)
    np.testing.assert_allclose(c, expected)

    # resample...
    c = CentralMomentsArray.from_resample_vals(
        *xy, weight=weight, axis=0, mom=mom, nrep=20, rng=np.random.default_rng(0)
    )
    expected = cmomy.resample_vals(
        *xy, weight=weight, axis=0, mom=mom, nrep=20, rng=np.random.default_rng(0)
    )
    np.testing.assert_allclose(c, expected)


@pytest.mark.parametrize("c", [CentralMomentsArray.zeros(mom=(3, 3))])
@pytest.mark.parametrize(
    "args",
    [
        (1,),
        (1, 1, 1),
    ],
)
def test_push_val_wrong_numpy(c, args) -> None:
    with pytest.raises(ValueError):
        c.push_val(*args)


def test_push_data_simple(wrapped: CentralMomentsArray[Any]) -> None:
    new = wrapped.zeros_like()
    new.push_data(wrapped.obj)
    np.testing.assert_allclose(new, wrapped)


@pytest.mark.parametrize(
    ("shape", "mom_ndim"),
    [
        ((10, 4), 1),
        ((10, 3, 4), 1),
        ((10, 4, 4), 2),
        ((10, 3, 4, 4), 2),
    ],
)
def test_push_data(rng, shape, mom_ndim) -> None:
    data = rng.random(shape)

    check = cmomy.reduce_data(data, axis=0, mom_ndim=mom_ndim)

    c = CentralMomentsArray(np.zeros(shape[1:]), mom_ndim=mom_ndim)
    c.push_datas(data, axis=0)
    np.testing.assert_allclose(c, check)

    c.zero()
    for d in data:
        c.push_data(d)

    np.testing.assert_allclose(c, check)


@pytest.mark.parametrize(
    ("shape", "mom_ndim"),
    [
        ((10, 4), 1),
        ((10, 3, 4), 1),
        ((10, 4, 4), 2),
        ((10, 3, 4, 4), 2),
    ],
)
def test_cumulative(rng, shape, mom_ndim) -> None:
    data = rng.random(shape)
    check = cmomy.convert.cumulative(data, axis=0, mom_ndim=mom_ndim)
    c = CentralMomentsArray(data, mom_ndim=mom_ndim)
    np.testing.assert_allclose(c.cumulative(axis=0), check)


@pytest.mark.parametrize("shape", [(4,), (10, 4)])
def test_moments_to_comoments(rng, shape) -> None:
    data = rng.random(shape)

    check = cmomy.convert.moments_to_comoments(data, mom=(2, -1))

    c = CentralMomentsArray(data, mom_ndim=1)

    np.testing.assert_allclose(check, c.moments_to_comoments(mom=(2, -1)))


@pytest.mark.parametrize(
    ("attr", "name"),
    [
        ("weight", "weight"),
        ("mean", "ave"),
        ("var", "var"),
        (
            "std",
            lambda data, mom_ndim: np.sqrt(
                cmomy.select_moment(data, "var", mom_ndim=mom_ndim)
            ),
        ),
        ("cov", "cov"),
        (
            "cmom",
            lambda data, mom_ndim: cmomy.assign_moment(
                data, {"weight": 1, "ave": 0}, mom_ndim=mom_ndim, copy=True
            ),
        ),
        (
            "rmom",
            lambda data, mom_ndim: cmomy.assign_moment(
                cmomy.convert.moments_type(data, mom_ndim=mom_ndim, to="raw"),
                weight=1.0,
                mom_ndim=mom_ndim,
                copy=False,
            ),
        ),
    ],
)
def test_select_moment(
    wrapped: CentralMomentsArray[Any], attr, name: str | Callable[..., Any]
) -> None:
    val = getattr(wrapped, attr)()
    data, mom_ndim = wrapped.obj, wrapped.mom_ndim
    if isinstance(name, str):
        check = cmomy.select_moment(data, cast(SelectMoment, name), mom_ndim=mom_ndim)
    elif callable(name):
        check = name(data, mom_ndim)
    np.testing.assert_allclose(val, check)


@pytest.mark.parametrize(
    ("shape", "mom_ndim"),
    [
        ((10, 4), 1),
        ((10, 3, 4), 1),
        ((10, 4, 4), 2),
        ((10, 3, 4, 4), 2),
    ],
)
def test_opertors(rng, shape, mom_ndim) -> None:
    data = rng.random(shape)
    c = CentralMomentsArray(data, mom_ndim=mom_ndim)

    # test addition
    check = c.reduce(axis=0)

    out = functools.reduce(operator.add, c)
    np.testing.assert_allclose(out, check)

    # test inplace add
    c1 = c[0].zeros_like()
    for cc in c:
        c1 += cc
    np.testing.assert_allclose(c1, check)

    # test subtraction
    check = cmomy.reduce_data(data[:-1], axis=0, mom_ndim=mom_ndim)
    c1 = c.reduce(axis=0) - c[-1]
    np.testing.assert_allclose(check, c1)

    # inplace subtraction
    c1 = c.reduce(axis=0)
    c1 -= c[-1]
    np.testing.assert_allclose(check, c1)

    # test multiple
    check = cmomy.assign_moment(
        data,
        {"weight": cmomy.select_moment(data, "weight", mom_ndim=mom_ndim) * 2},
        mom_ndim=mom_ndim,
    )

    c2 = c * 2
    np.testing.assert_allclose(check, c2)

    c2 = c.copy()
    c2 *= 2
    np.testing.assert_allclose(check, c2)


def test_operator_raises() -> None:
    c0 = CentralMomentsArray.zeros(mom=3)
    c1 = CentralMomentsArray.zeros(mom=4)

    with pytest.raises(TypeError):
        _ = c0 + 1  # type: ignore[operator]

    with pytest.raises(ValueError):
        _ = c0 + c1


@pytest.mark.parametrize(
    ("shape", "mom_ndim", "axis", "dest"),
    [
        ((2, 3, 4), 1, 1, 0),
        ((1, 2, 3, 4), 1, (0, -1), (1, 0)),
        ((1, 2, 3, 4), 2, 0, -1),
        ((1, 2, 3, 4, 5), 2, (1, 2), (0, -1)),
    ],
)
def test_moveaxis(rng, shape, mom_ndim, axis, dest) -> None:
    data = rng.random(shape)
    check = cmomy.moveaxis(data, axis, dest, mom_ndim=mom_ndim)
    c = CentralMomentsArray(data, mom_ndim=mom_ndim)

    np.testing.assert_equal(check, c.moveaxis(axis, dest))


@pytest.mark.parametrize(
    "name",
    ["weight", "ave", "xvar"],
)
def test_assign_moment(rng, wrapped, name) -> None:
    v = rng.random()
    m = {name: v}

    c = wrapped.assign_moment(m)

    data, mom_ndim = wrapped.obj, wrapped.mom_ndim
    check = cmomy.assign_moment(data, m, mom_ndim=mom_ndim)

    np.testing.assert_allclose(check, c)


def _reduce_block(data, axis, block, **kwargs):
    groups = cmomy.reduction.block_by(data.shape[axis], block)
    return cmomy.reduce_data_grouped(data, by=groups, axis=axis, **kwargs)


@pytest.mark.parametrize(
    ("attr", "func", "fkws"),
    [
        (
            "resample_and_reduce",
            cmomy.resample_data,
            lambda: {"nrep": 10, "rng": np.random.default_rng(0)},
        ),
        ("jackknife_and_reduce", cmomy.resample.jackknife_data, lambda: {}),  # noqa: PIE807
        ("reduce", cmomy.reduce_data, lambda: {}),  # noqa: PIE807
        ("reduce", _reduce_block, lambda: {"block": 2}),
    ],
)
def test_reduce(wrapped, attr, func, fkws) -> None:
    data, mom_ndim = wrapped.obj, wrapped.mom_ndim
    meth = getattr(wrapped, attr)
    if data.ndim - mom_ndim > 0:
        r = meth(axis=0, **fkws())
        check = func(data, axis=0, mom_ndim=mom_ndim, **fkws())
        np.testing.assert_allclose(r, check)

    else:
        with pytest.raises(ValueError, match="No dimension to reduce.*"):
            meth(axis=0, **fkws())


def test_from_raw(wrapped) -> None:
    data, mom_ndim = wrapped.obj, wrapped.mom_ndim

    raw = wrapped.to_raw()
    new = CentralMomentsArray.from_raw(raw, mom_ndim=wrapped.mom_ndim)

    np.testing.assert_allclose(
        raw, cmomy.convert.moments_type(data, mom_ndim=mom_ndim, to="raw")
    )
    np.testing.assert_allclose(
        new, cmomy.convert.moments_type(raw, mom_ndim=mom_ndim, to="central")
    )


def test_jackknife_and_reduce(rng) -> None:
    data = rng.random((10, 2, 3))

    c = cmomy.CentralMomentsArray(data)

    a = c.jackknife_and_reduce(axis=0)
    b = c.jackknife_and_reduce(axis=0, data_reduced=c.reduce(axis=0))

    np.testing.assert_allclose(a, b)

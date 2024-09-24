# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

import cmomy

if TYPE_CHECKING:
    from cmomy.core.typing import Mom_NDim


shapes_mark = pytest.mark.parametrize(
    ("xshape", "yshape", "wshape", "axis", "mom"),
    [
        ((10, 2, 3), None, None, 0, (3,)),
        ((10, 2, 3), None, (10, 2, 3), 0, (3,)),
        ((2, 10, 3), None, (10,), 1, (3,)),
        ((2, 3, 10), None, (3, 10), 2, (3,)),
        ((10, 2, 3), (10,), None, 0, (3, 3)),
        ((10, 2, 3), (10, 1, 1), (10, 2, 3), 0, (3, 3)),
        ((2, 10, 3), (10, 3), (10,), 1, (3, 3)),
        ((2, 3, 10), (10,), (3, 10), 2, (3, 3)),
    ],
)


def get_params(
    rng: np.random.Generator, xshape, yshape, wshape, axis, mom_ndim, as_dataarray
):
    x = rng.random(xshape)
    w = None if wshape is None else rng.random(wshape)
    y = None if yshape is None else rng.random(yshape)

    if as_dataarray:
        dims = ["a", "b", "c"]
        x = xr.DataArray(x, dims=dims)
        w, y = (
            None
            if a is None
            else xr.DataArray(a, dims=(dims[axis] if a.ndim == 1 else dims[-a.ndim :]))
            for a in (w, y)
        )
    xy = (x,) if mom_ndim == 1 else (x, rng.random(yshape))

    return xy, w


@shapes_mark
@pytest.mark.parametrize(
    ("func", "kwargs", "style"),
    [
        (cmomy.resample.resample_vals, {}, "resample"),
        (cmomy.wrap_resample_vals, {}, "resample"),
        (cmomy.resample.jackknife_vals, {}, None),
        (
            cmomy.rolling.rolling_vals,
            {"window": 3, "center": False},
            None,
        ),
        (cmomy.rolling.rolling_exp_vals, {"alpha": 0.2}, None),
    ],
)
def test_vals_move_axis_to_end(
    rng,
    xshape,
    yshape,
    wshape,
    axis,
    mom,
    as_dataarray,
    func,
    kwargs,
    style,
) -> None:
    mom_ndim: Mom_NDim = len(mom)  # type: ignore[assignment]

    xy, w = get_params(rng, xshape, yshape, wshape, axis, mom_ndim, as_dataarray)

    kws = {"weight": w, "mom": mom, "axis": axis, **kwargs}
    if style == "resample":
        kws["sampler"] = cmomy.resample.factory_sampler(ndat=xshape[axis], nrep=20)

    outs = [func(*xy, **kws, move_axis_to_end=m) for m in (True, False)]

    np.testing.assert_allclose(
        outs[0],
        np.moveaxis(np.asarray(outs[1]), axis, -(mom_ndim + 1)),
    )

    # using out parameter
    _outs = [np.zeros_like(o) for o in outs]
    outs2 = [
        func(*xy, **kws, out=o, move_axis_to_end=m)
        for m, o in zip((True, False), _outs)
    ]

    for a, b, c in zip(outs2, _outs, outs):
        np.shares_memory(a, b)
        np.testing.assert_allclose(a, c)


from cmomy.core.array_utils import normalize_axis_index


@pytest.mark.parametrize(
    ("shape", "axis", "mom_ndim"),
    [
        ((10, 1, 2, 3), 0, 1),
        ((1, 10, 2, 3), 1, 1),
        ((1, 2, 10, 3), -1, 1),
        ((10, 1, 2, 3), 0, 2),
        ((1, 10, 2, 3), -1, 2),
    ],
)
@pytest.mark.parametrize(
    ("func", "kwargs", "style"),
    [
        (cmomy.resample.resample_data, {}, "resample"),
        (cmomy.resample.jackknife_data, {}, "jack"),
        (cmomy.reduction.reduce_data_grouped, {}, "group"),
        (cmomy.reduction.reduce_data_indexed, {}, "index"),
        (cmomy.rolling.rolling_data, {"window": 4, "center": False}, "roll"),
        (cmomy.rolling.rolling_exp_data, {"alpha": 0.2}, "roll"),
        (cmomy.convert.cumulative, {}, "convert"),
    ],
)
def test_data_move_axis_to_end(
    rng,
    shape,
    axis,
    mom_ndim,
    as_dataarray,
    func,
    kwargs,
    style,
) -> None:
    data = rng.random(shape)
    if as_dataarray:
        data = xr.DataArray(data)

    kws = dict(axis=axis, mom_ndim=mom_ndim, **kwargs)
    ndat = cmomy.resample.select_ndat(data, axis=axis, mom_ndim=mom_ndim)
    if style == "resample":
        kws["sampler"] = cmomy.resample.factory_sampler(ndat=ndat, nrep=20)
    elif style == "group":
        kws["by"] = rng.choice(4, size=ndat)
    elif style == "index":
        groups = rng.choice(4, size=ndat)
        _, kws["index"], kws["group_start"], kws["group_end"] = (
            cmomy.reduction.factor_by_to_index(groups)
        )

    outs = [func(data, **kws, move_axis_to_end=m) for m in (True, False)]
    np.testing.assert_allclose(
        outs[0],
        np.moveaxis(
            np.asarray(outs[1]),
            normalize_axis_index(axis, data.ndim, mom_ndim),
            -(mom_ndim + 1),
        ),
    )

    # using out parameter
    _outs = [np.zeros_like(o) for o in outs]
    outs2 = [
        func(data, **kws, out=o, move_axis_to_end=m)
        for m, o in zip((True, False), _outs)
    ]

    for a, b, c in zip(outs2, _outs, outs):
        np.shares_memory(a, b)
        np.testing.assert_allclose(a, c)

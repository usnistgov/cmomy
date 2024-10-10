# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

import cmomy
from cmomy.core.moment_params import factory_mom_params

from ._dataarray_set_utils import remove_axis_from_kwargs

if TYPE_CHECKING:
    from cmomy.core.typing import MomNDim


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
def test_vals_move_axes_to_end(
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
    mom_ndim: MomNDim = len(mom)  # type: ignore[assignment]

    xy, w = get_params(rng, xshape, yshape, wshape, axis, mom_ndim, as_dataarray)

    kws = {"weight": w, "mom": mom, "axis": axis, **kwargs}
    if style == "resample":
        kws["sampler"] = cmomy.resample.factory_sampler(ndat=xshape[axis], nrep=20)

    outs = [func(*xy, **kws, move_axes_to_end=m) for m in (True, False)]

    np.testing.assert_allclose(
        outs[0],
        np.moveaxis(np.asarray(outs[1]), axis, -(mom_ndim + 1)),
    )

    # using out parameter
    _outs = [np.zeros_like(o) for o in outs]
    outs2 = [
        func(*xy, **kws, out=o, move_axes_to_end=m)
        for m, o in zip((True, False), _outs)
    ]

    for a, b, c in zip(outs2, _outs, outs):
        np.shares_memory(a, b)
        np.testing.assert_allclose(a, c)


@pytest.mark.parametrize(
    ("shape", "axis", "mom_ndim", "mom_axes"),
    [
        ((10, 1, 2, 3), 0, 1, None),
        ((1, 10, 2, 3), 1, 1, None),
        ((1, 2, 10, 3), -1j, 1, None),
        ((10, 1, 2, 3), 0, 2, None),
        ((1, 10, 2, 3), -1j, 2, None),
        # mom_axes
        ((10, 3, 1, 2), 0, None, 1),
        ((3, 10, 1, 1), 1, None, 0),
        ((1, 3, 10, 2), -2, None, 1),
        ((10, 2, 3, 1), 0, None, (1, 2)),
        ((2, 3, 10, 1), -2, None, (0, 1)),
    ],
)
@pytest.mark.parametrize(
    ("func", "kwargs", "kwargs_callback"),
    [
        (cmomy.resample.resample_data, {"sampler": {"nrep": 20, "rng": 0}}, None),
        (cmomy.resample.jackknife_data, {}, None),
        (cmomy.reduction.reduce_data_grouped, {"by": [0] * 5 + [1] * 5}, None),
        (
            cmomy.reduction.reduce_data_indexed,
            {"index": range(10), "group_start": [0, 5], "group_end": [5, 10]},
            None,
        ),
        (cmomy.rolling.rolling_data, {"window": 4, "center": False}, None),
        (cmomy.rolling.rolling_exp_data, {"alpha": 0.2}, None),
        (cmomy.convert.cumulative, {}, None),
        (cmomy.convert.moments_type, {}, remove_axis_from_kwargs),
    ],
)
def test_data_move_axes_to_end(
    rng,
    shape,
    axis,
    mom_ndim,
    mom_axes,
    as_dataarray,
    func,
    kwargs,
    kwargs_callback,
) -> None:
    data = rng.random(shape)
    if as_dataarray:
        data = xr.DataArray(data)

    kws = dict(axis=axis, mom_ndim=mom_ndim, **kwargs, mom_axes=mom_axes)
    kws = kwargs_callback(kws) if kwargs_callback else kws

    outs = [func(data, **kws, move_axes_to_end=m) for m in (True, False)]

    # axes movers
    mom_params = factory_mom_params(None, ndim=mom_ndim, axes=mom_axes)
    axes0 = mom_params.axes
    axes1 = mom_params.axes_last
    if "axis" in kws:
        axes0 = (axis, *axes0)
        axes1 = (-1j, *axes1)

    np.testing.assert_allclose(
        outs[0],
        cmomy.moveaxis(
            outs[1], axes0, axes1, mom_params=mom_params, allow_select_mom_axes=True
        ),
    )

    # check that moving axis to end on data gives same result
    kws2 = kws.copy()
    if "axis" in kws2:
        kws2["axis"] = -1j
    kws2["mom_axes"] = mom_params.axes_last
    kws2["mom_ndim"] = mom_params.ndim
    check = func(
        cmomy.moveaxis(
            data, axes0, axes1, mom_params=mom_params, allow_select_mom_axes=True
        ),
        **kws2,
    )
    np.testing.assert_allclose(outs[0], check)

    # using out parameter
    _outs = [np.zeros_like(o) for o in outs]
    outs2 = [
        func(data, **kws, out=o, move_axes_to_end=m)
        for m, o in zip((True, False), _outs)
    ]

    for a, b, c in zip(outs2, _outs, outs):
        np.shares_memory(a, b)
        np.testing.assert_allclose(a, c)

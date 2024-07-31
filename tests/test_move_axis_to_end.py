from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

import cmomy

if TYPE_CHECKING:
    from cmomy.typing import Mom_NDim


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
as_dataarray = pytest.mark.parametrize("as_dataarray", [False, True])


def get_params(rng, xshape, yshape, wshape, axis, mom_ndim, as_dataarray):
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
@as_dataarray
@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (cmomy.resample.resample_vals, {}),
        (cmomy.resample.jackknife_vals, {}),
        (cmomy.rolling.rolling_vals, {"window": 3, "center": False}),
        (cmomy.rolling.rolling_exp_vals, {"alpha": 0.2}),
    ],
)
def test_resample_vals_move_axis_to_end(
    rng, xshape, yshape, wshape, axis, mom, as_dataarray, func, kwargs
) -> None:
    mom_ndim: Mom_NDim = len(mom)

    xy, w = get_params(rng, xshape, yshape, wshape, axis, mom_ndim, as_dataarray)

    kws = {"weight": w, "mom": mom, "axis": axis, **kwargs}
    if func == cmomy.resample.resample_vals:
        kws["freq"] = cmomy.randsamp_freq(ndat=xshape[axis], nrep=20)

    outs = [func(*xy, **kws, move_axis_to_end=m) for m in (True, False)]

    np.testing.assert_allclose(
        outs[0],
        np.moveaxis(np.asarray(outs[1]), axis, -(mom_ndim + 1)),
    )

    # using out parameter
    _outs = [np.zeros_like(o) for o in outs]
    outs2 = (
        func(*xy, **kws, out=o, move_axis_to_end=m)
        for m, o in zip((True, False), _outs)
    )

    for a, b in zip(outs2, _outs):
        np.shares_memory(a, b)
        np.testing.assert_allclose(a, b)

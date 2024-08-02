# mypy: disable-error-code="no-untyped-def, no-untyped-call, arg-type, index, assignment"
# pyright:  reportArgumentType=false, reportAssignmentType=false
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import cmomy
from cmomy import rolling

if TYPE_CHECKING:
    from typing import TypedDict

    from cmomy.core.typing import Mom_NDim, MomentsStrict

    class RollingDict(TypedDict):
        window: int
        min_periods: int | None
        center: bool

    class RollingExpDict(TypedDict):
        alpha: float
        adjust: bool
        min_periods: int | None


@pytest.mark.parametrize("mom_ndim", [1, 2])
def test__optional_zero_missing_weights(mom_ndim) -> None:
    data = np.ones((2, 3, 4))

    data[(..., *(0,) * mom_ndim)] = np.nan

    # no change
    out = rolling._optional_zero_missing_weight(data.copy(), mom_ndim, False)
    np.testing.assert_equal(out, data)

    # change
    check = data.copy()
    check[(..., *(0,) * mom_ndim)] = 0.0
    out = rolling._optional_zero_missing_weight(data.copy(), mom_ndim, True)
    np.testing.assert_allclose(out, check)


# * construct
@pytest.mark.parametrize(
    ("shape", "axis", "window"),
    [
        ((10,), 0, 3),
        ((5, 6), 0, 3),
        ((5, 6), (0, 1), 3),
        ((5, 6), (0, 1), (2, 3)),
        ((5, 6, 7, 8), (0, 1, 2), (2, 3, 4)),
        ((5, 6, 7, 8), (0, 2), (2, 4)),
    ],
)
@pytest.mark.parametrize("center", [True, False])
def test_construct_rolling_window_array(shape, axis, window, center, as_dataarray):
    data = np.arange(np.prod(shape)).reshape(shape).astype(np.float64)

    xdata = xr.DataArray(data)
    _axis = (axis,) if isinstance(axis, int) else axis
    _window = (window,) * len(_axis) if isinstance(window, int) else window
    r = xdata.rolling(
        {xdata.dims[a]: win for a, win in zip(_axis, _window)}, center=center
    )
    c = r.construct({xdata.dims[a]: f"_rolling_{a}" for a in _axis})

    if as_dataarray:
        data = xdata

    out = rolling.construct_rolling_window_array(
        data, axis=axis, window=window, center=center
    )

    np.testing.assert_allclose(c, out)

    if isinstance(window, tuple) and len(window) == 3:
        with pytest.raises(ValueError):
            rolling.construct_rolling_window_array(
                data,
                axis=axis,
                window=window[:-1],
                center=center,
            )


def test_construct_rolling_window_array_mom_ndim(as_dataarray) -> None:
    shape = (2, 3, 4, 5)
    data = np.arange(np.prod(shape)).reshape(shape).astype(np.float64)
    if as_dataarray:
        data = xr.DataArray(data)

    func = rolling.construct_rolling_window_array

    axis = (0, 1)
    out = func(data, axis=axis, window=3)
    out1 = func(data, axis=axis, mom_ndim=1, window=3)
    out2 = func(data, axis=axis, mom_ndim=2, window=3)

    np.testing.assert_allclose(
        cmomy.moveaxis(out, (-2, -1), (-3, -2)),
        out1,
    )
    np.testing.assert_allclose(
        cmomy.moveaxis(out, (-2, -1), (-4, -3)),
        out2,
    )

    with pytest.raises(ValueError):
        func(data, axis=2, mom_ndim=2, window=3)


# * Move data
@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((10,), 0),
        ((10, 2, 3), 0),
        ((2, 10, 3), 1),
    ],
)
@pytest.mark.parametrize("window", [3, 8])
@pytest.mark.parametrize("min_periods", [None, 2])
@pytest.mark.parametrize("center", [False, True])
def test_rolling_data(
    rng, shape, axis, window, min_periods, center, as_dataarray
) -> None:
    """
    Simple test like rolling_vals...

    Probably need a more robust test in the future...
    """
    x = rng.random(shape)
    dx = xr.DataArray(x)
    r = dx.rolling({dx.dims[axis]: window}, min_periods=min_periods, center=center)

    kws: RollingDict = {"window": window, "min_periods": min_periods, "center": center}

    # data
    if as_dataarray:
        x = dx

    data = cmomy.convert.vals_to_data(
        x,
        mom=3,
    )

    out = rolling.rolling_data(
        data,
        axis=axis,
        mom_ndim=1,
        **kws,
    )

    np.testing.assert_allclose(out[..., 1], r.mean())
    np.testing.assert_allclose(out[..., 2], r.var(ddof=0))

    # vals
    out2 = rolling.rolling_vals(x, axis=axis, mom=3, move_axis_to_end=False, **kws)

    np.testing.assert_allclose(
        out,
        out2,
        atol=1e-14,
    )


@pytest.mark.parametrize("window", [3, 8])
@pytest.mark.parametrize("min_periods", [None, 2])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("missing", [True, False])
@pytest.mark.parametrize("mom_ndim", [1, 2])
def test_rolling_data_vals_missing(  # noqa: PLR0914
    rng: np.random.Generator,
    window: int,
    min_periods: int | None,
    center: bool,
    missing: bool,
    mom_ndim: Mom_NDim,
) -> None:
    shape = (100, 3)
    mom: tuple[int] | tuple[int, int] = (3,) * mom_ndim  # type: ignore[assignment]

    kws: RollingDict = {"window": window, "min_periods": min_periods, "center": center}

    x = rng.random(shape)
    y = rng.random(shape)
    w = np.ones_like(x)

    if missing:
        idx = rng.choice(range(20, x.shape[0]), size=10, replace=False).tolist()
        # include the first and missing chunk
        idx = [0, *range(5, 5 + window + 1), *idx]

        x[idx, ...] = np.nan
        y[idx, ...] = np.nan
        w[idx, ...] = 0.0

    xy = (x,) if mom_ndim == 1 else (x, y)
    widx = (..., 0) if mom_ndim == 1 else (..., 0, 0)
    xidx = (..., 1) if mom_ndim == 1 else (..., 1, 0)
    x2idx = (..., 2) if mom_ndim == 1 else (..., 2, 0)

    data = cmomy.convert.vals_to_data(
        *xy,
        weight=w,
        mom=mom,
    )

    rolling.rolling_data(
        data,
        window=window,
        min_periods=min_periods,
        center=center,
        axis=0,
        mom_ndim=mom_ndim,
    )
    outd = rolling.rolling_data(data, **kws, axis=0, mom_ndim=mom_ndim)
    out = np.moveaxis(
        rolling.rolling_vals(*xy, **kws, axis=0, mom=mom, weight=w), -(mom_ndim + 1), 0
    )

    np.testing.assert_allclose(out, outd, atol=1e-14)

    # just to make sure, we also use construct
    data_rolling = rolling.construct_rolling_window_array(
        data, axis=0, fill_value=0.0, mom_ndim=mom_ndim, **kws
    )

    outc = cmomy.reduce_data(data_rolling, mom_ndim=mom_ndim, axis=-1)
    count = (data_rolling[widx] != 0.0).sum(axis=-1)

    outc = np.where(
        count[(..., *(None,) * mom_ndim)]
        >= (window if min_periods is None else min_periods),
        outc,
        np.nan,
    )
    w2 = outc[widx]
    w2[np.isnan(w2)] = 0.0
    np.testing.assert_allclose(outc, out, atol=1e-14)

    # compare to pands
    rx = pd.DataFrame(x).rolling(**kws)
    rw = pd.DataFrame(w).replace(0.0, np.nan).rolling(**kws)

    np.testing.assert_allclose(out[widx], rw.sum().fillna(0.0))
    np.testing.assert_allclose(out[xidx], rx.mean())
    np.testing.assert_allclose(out[x2idx], rx.var(ddof=0), atol=1e-14)

    if mom_ndim == 2:
        dfy = pd.DataFrame(y)
        ry = dfy.rolling(**kws)
        np.testing.assert_allclose(out[..., 0, 1], ry.mean())
        np.testing.assert_allclose(out[..., 0, 2], ry.var(ddof=0), atol=1e-14)
        np.testing.assert_allclose(out[..., 1, 1], rx.cov(dfy, ddof=0), atol=1e-14)


@pytest.mark.parametrize("mom_ndim", [1, 2])
@pytest.mark.parametrize("window", [3, 8])
@pytest.mark.parametrize("min_periods", [None, 2])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("missing", [True, False])
def test_rolling_weights(  # noqa: PLR0914
    rng, mom_ndim, window, min_periods, center, missing
) -> None:
    # test unequal weights...
    data = rng.random((100, 3, 3))

    kws: RollingDict = {"window": window, "min_periods": min_periods, "center": center}

    if missing:
        idx = rng.choice(range(20, data.shape[0]), size=10, replace=False)
        # include the first and missing chunk
        idx = [0, *range(5, 5 + window + 1), *list(idx)]

        data[idx, ...] = np.nan
        data[(idx, ..., *(0,) * mom_ndim)] = 0.0

    out = rolling.rolling_data(data, **kws, axis=0, mom_ndim=mom_ndim)
    data_rolling = rolling.construct_rolling_window_array(
        data, axis=0, fill_value=0.0, mom_ndim=mom_ndim, **kws
    )

    outc = cmomy.reduce_data(data_rolling, mom_ndim=mom_ndim, axis=-1)
    count = (data_rolling[(..., *((0,) * mom_ndim))] != 0.0).sum(axis=-1)

    outc = np.where(
        count[(..., *((None,) * mom_ndim))]
        >= (window if min_periods is None else min_periods),
        outc,
        np.nan,
    )
    w2 = outc[(..., *(0,) * mom_ndim)]
    w2[np.isnan(w2)] = 0.0

    np.testing.assert_allclose(out, outc)

    # testing val
    w = data[(..., *(0,) * mom_ndim)]
    x = data[(..., *(1,) * mom_ndim)]
    y = data[(..., *(2,) * mom_ndim)]

    mom = (3,) * mom_ndim

    xy = (x,) if mom_ndim == 1 else (x, y)

    out = rolling.rolling_vals(*xy, weight=w, **kws, axis=0, mom=mom)

    wr = data_rolling[(..., *(0,) * mom_ndim)]
    xr = data_rolling[(..., *(1,) * mom_ndim)]
    yr = data_rolling[(..., *(2,) * mom_ndim)]

    xyr = (xr,) if mom_ndim == 1 else (xr, yr)

    outc = cmomy.reduce_vals(*xyr, weight=wr, mom=mom, axis=-1)
    count = (wr != 0.0).sum(axis=-1)

    outc = np.where(
        count[(..., *((None,) * mom_ndim))]
        >= (window if min_periods is None else min_periods),
        outc,
        np.nan,
    )
    w2 = outc[(..., *(0,) * mom_ndim)]
    w2[np.isnan(w2)] = 0.0

    outc = np.moveaxis(outc, 0, -(mom_ndim + 1))
    np.testing.assert_allclose(out, outc, atol=1e-14)


@pytest.mark.parametrize(
    ("shape", "axis", "mom_ndim"),
    [
        ((10, 3), 0, 1),
        ((10, 3, 3), 0, 1),
        ((3, 10, 3), 1, 1),
        ((10, 3, 3), 0, 2),
        ((2, 10, 3, 3), 1, 2),
    ],
)
@pytest.mark.parametrize("window", [3, 8])
@pytest.mark.parametrize("min_periods", [None, 2])
@pytest.mark.parametrize("center", [False, True])
def test_rolling_data_from_constructed_windows(
    rng, shape, axis, mom_ndim, window, min_periods, center
) -> None:
    data = rng.random(shape)

    out = rolling.rolling_data(
        data,
        axis=axis,
        mom_ndim=mom_ndim,
        window=window,
        min_periods=min_periods,
        center=center,
    )

    data_rolling = rolling.construct_rolling_window_array(
        data, axis=axis, window=window, center=center, fill_value=0.0, mom_ndim=mom_ndim
    )
    out2 = cmomy.reduce_data(data_rolling, axis=-1, mom_ndim=mom_ndim)
    # clean up counts...
    count = (data_rolling[(..., *((0,) * mom_ndim))] != 0.0).sum(axis=-1)

    out2 = np.where(
        count[(..., *((None,) * mom_ndim))]
        >= (window if min_periods is None else min_periods),
        out2,
        np.nan,
    )

    # nan weights -> 0
    w = out2[(..., *(0,) * mom_ndim)]
    w[np.isnan(w)] = 0.0
    out2 = cmomy.convert.assign_weight(out2, w, mom_ndim=mom_ndim, copy=False)

    np.testing.assert_allclose(out, out2)


# * rolling exp
@pytest.mark.parametrize("alpha", [0.2, 0.4])
@pytest.mark.parametrize("adjust", [True, False])
@pytest.mark.parametrize("missing", [True, False])
@pytest.mark.parametrize("min_periods", [None, 3])
@pytest.mark.parametrize("mom_ndim", [1, 2])
def test_rolling_exp_data_vals_missing(  # noqa: PLR0914
    rng: np.random.Generator,
    alpha: float,
    adjust: bool,
    missing: bool,
    min_periods: int | None,
    mom_ndim: Mom_NDim,
) -> None:
    shape = (100, 3)
    mom: tuple[int] | tuple[int, int] = (3,) * mom_ndim  # pyright: ignore[reportAssignmentType]

    kws: RollingExpDict = {"alpha": alpha, "adjust": adjust, "min_periods": min_periods}

    x = rng.random(shape)
    y = rng.random(shape)
    w = np.ones_like(x)

    if missing:
        idx = rng.choice(range(20, x.shape[0]), size=10, replace=False).tolist()
        # include the first and missing chunk
        idx = [0, *range(5, 10), *idx]

        x[idx, ...] = np.nan
        y[idx, ...] = np.nan
        w[idx, ...] = 0.0

    xy = (x,) if mom_ndim == 1 else (x, y)
    widx = (..., 0) if mom_ndim == 1 else (..., 0, 0)
    xidx = (..., 1) if mom_ndim == 1 else (..., 1, 0)
    x2idx = (..., 2) if mom_ndim == 1 else (..., 2, 0)

    data = cmomy.convert.vals_to_data(
        *xy,
        weight=w,
        mom=mom,
    )

    outd = rolling.rolling_exp_data(data, **kws, axis=0, mom_ndim=mom_ndim)
    out = rolling.rolling_exp_vals(*xy, weight=w, **kws, axis=0, mom=mom)

    out = np.moveaxis(out, -(mom_ndim + 1), 0)
    np.testing.assert_allclose(out, outd, atol=1e-14)

    rx = pd.DataFrame(x).ewm(**kws)
    rw = pd.DataFrame(w).replace(0.0, np.nan).ewm(**kws)

    if adjust:
        np.testing.assert_allclose(out[widx], rw.sum().fillna(0.0))
    np.testing.assert_allclose(out[xidx], rx.mean())
    np.testing.assert_allclose(out[x2idx], rx.var(bias=True))

    if mom_ndim == 2:
        dfy = pd.DataFrame(y)
        ry = dfy.ewm(**kws)
        np.testing.assert_allclose(out[..., 0, 1], ry.mean())
        np.testing.assert_allclose(out[..., 0, 2], ry.var(bias=True))
        np.testing.assert_allclose(out[..., 1, 1], rx.cov(dfy, bias=True))


@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((10,), 0),
        ((10, 2, 3), 0),
        ((2, 10, 3), 1),
    ],
)
@pytest.mark.parametrize("alpha", [0.2, 0.4])
@pytest.mark.parametrize("adjust", [True])
def test_rolling_exp_simple(rng, shape, axis, alpha, adjust) -> None:
    x = rng.random(shape)
    dx = xr.DataArray(x)
    rolling_kws = {"window": {dx.dims[axis]: alpha}, "window_type": "alpha"}

    out = rolling.rolling_exp_vals(
        x,
        alpha=alpha,
        mom=1,
        axis=axis,
        adjust=adjust,
    )

    # move to original position
    out = np.moveaxis(out, -2, axis)

    data = cmomy.convert.vals_to_data(
        x,
        mom=1,
    )
    outd = rolling.rolling_exp_data(
        data,
        alpha=alpha,
        mom_ndim=1,
        axis=axis,
        adjust=adjust,
    )

    np.testing.assert_allclose(out, outd)

    # if have numbagg, do this.
    try:
        np.testing.assert_allclose(out[..., 1], dx.rolling_exp(**rolling_kws).mean())
        x_count = (~np.isnan(x)).astype(x.dtype)
        np.testing.assert_allclose(
            out[..., 0], xr.DataArray(x_count).rolling_exp(**rolling_kws).sum()
        )
    except ImportError:
        pass

    # using xarray
    xout = rolling.rolling_exp_vals(
        dx,
        alpha=alpha,
        mom=1,
        dim=dx.dims[axis],
        adjust=adjust,
    ).transpose(*dx.dims, ...)

    assert isinstance(xout, xr.DataArray)
    np.testing.assert_allclose(out, xout)

    xout = rolling.rolling_exp_data(
        xr.DataArray(data),
        alpha=alpha,
        mom_ndim=1,
        axis=axis,
        adjust=adjust,
    )

    assert isinstance(xout, xr.DataArray)
    np.testing.assert_allclose(out, xout)


@pytest.mark.parametrize("adjust", [True, False])
@pytest.mark.parametrize("alpha", [0.2, 0.8, None])
@pytest.mark.parametrize("mom_ndim", [1, 2])
def test_rolling_exp_weight(
    rng: np.random.Generator,
    adjust: bool,
    alpha: float | None,
    mom_ndim: Mom_NDim,
) -> None:
    axis = 0
    shape = (50, 2, 3)
    mom: MomentsStrict = (3,) * mom_ndim

    weight = rng.random(shape)
    xy = tuple(rng.random(shape) for _ in range(mom_ndim))

    data = cmomy.convert.vals_to_data(
        *xy,
        weight=weight,
        mom=mom,
    )

    alphas = (
        rng.random(shape[axis])
        if alpha is None
        else np.full(shape[axis], alpha, dtype=data.dtype)
    )

    freq = np.zeros((shape[axis],) * 2)
    _w = np.array([], dtype=np.float64)
    old_weight = 0.0
    for k in range(shape[axis]):
        _w *= 1 - alphas[k]
        _w = np.append(_w, 1.0 if adjust else alphas[k])
        if not adjust:
            old_weight = old_weight * (1 - alphas[k]) + alphas[k]
            _w /= old_weight
            old_weight = 1.0

        freq[k, : k + 1] = _w

    a = rolling.rolling_exp_vals(
        *xy, weight=weight, alpha=alphas, mom=mom, axis=axis, adjust=adjust
    )
    b = cmomy.resample_vals(*xy, mom=mom, freq=freq, weight=weight, axis=axis)  # pyright: ignore[reportCallIssue]
    np.testing.assert_allclose(a, b)

    c = rolling.rolling_exp_data(
        data, alpha=alphas, mom_ndim=mom_ndim, axis=axis, adjust=adjust
    )
    np.testing.assert_allclose(
        a,
        np.moveaxis(c, axis, -(mom_ndim + 1)),
    )


@pytest.mark.parametrize("adjust", [True, False])
@pytest.mark.parametrize("mom_ndim", [1, 2])
def test_rolling_exp_multiple_alpha(
    rng: np.random.Generator,
    adjust: bool,
    mom_ndim: Mom_NDim,
) -> None:
    axis = 0
    shape = (50, 2)
    mom: MomentsStrict = (3,) * mom_ndim

    alphas = rng.random(shape)
    weight = rng.random(shape)
    xy = tuple(rng.random(shape) for _ in range(mom_ndim))

    data = cmomy.convert.vals_to_data(*xy, weight=weight, mom=mom)

    a = rolling.rolling_exp_vals(
        *xy, weight=weight, alpha=alphas, mom=mom, axis=axis, adjust=adjust
    )
    c = rolling.rolling_exp_data(
        data, alpha=alphas, mom_ndim=mom_ndim, axis=axis, adjust=adjust
    )
    np.testing.assert_allclose(
        a,
        np.moveaxis(c, axis, -(mom_ndim + 1)),
    )

    for k in range(shape[-1]):
        b = rolling.rolling_exp_vals(
            *(_[..., k] for _ in xy),
            weight=weight[..., k],
            alpha=alphas[..., k],
            mom=mom,
            axis=axis,
            adjust=adjust,
        )
        np.testing.assert_allclose(b, a[k, ...])

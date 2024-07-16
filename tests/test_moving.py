# mypy: disable-error-code="no-untyped-def, no-untyped-call"
from functools import partial

import numpy as np
import pytest
import xarray as xr

import cmomy
from cmomy import moving


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
def test_construct_rolling_window_array(shape, axis, window, center):
    data = np.arange(np.prod(shape)).reshape(shape).astype(np.float64)

    xdata = xr.DataArray(data)

    _axis = (axis,) if isinstance(axis, int) else axis
    _window = (window,) * len(_axis) if isinstance(window, int) else window
    r = xdata.rolling(
        {xdata.dims[a]: win for a, win in zip(_axis, _window)}, center=center
    )
    c = r.construct({xdata.dims[a]: f"_rolling_{a}" for a in _axis})

    out = moving.construct_rolling_window_array(
        data, axis=axis, window=window, center=center
    )

    np.testing.assert_allclose(c, out)


def test_construct_rolling_window_array_mom_ndim() -> None:
    shape = (2, 3, 4, 5)
    data = np.arange(np.prod(shape)).reshape(shape).astype(np.float64)

    func = partial(moving.construct_rolling_window_array, window=3)

    axis = (0, 1)
    out = func(data, axis=axis)
    out1 = func(data, axis=axis, mom_ndim=1)
    out2 = func(data, axis=axis, mom_ndim=2)

    np.testing.assert_allclose(
        np.moveaxis(out, (-2, -1), (-3, -2)),
        out1,
    )
    np.testing.assert_allclose(
        np.moveaxis(out, (-2, -1), (-4, -3)),
        out2,
    )

    with pytest.raises(ValueError):
        func(data, axis=2, mom_ndim=2)


@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((10,), 0),
        ((10, 2, 3), 0),
        ((2, 10, 3), 1),
    ],
)
@pytest.mark.parametrize("window", [3, 8])
@pytest.mark.parametrize("min_count", [None, 2])
@pytest.mark.parametrize("center", [False, True])
def test_move_data(rng, shape, axis, window, min_count, center) -> None:
    """
    Simple test like move_vals...

    Probably need a more robust test in the future...
    """
    x = rng.random(shape)
    dx = xr.DataArray(x)
    r = dx.rolling({dx.dims[axis]: window}, min_periods=min_count, center=center)

    data = np.zeros((*shape, 3))
    data[..., 0] = 1
    data[..., 1] = x
    out = moving.move_data(
        data, axis=axis, mom_ndim=1, window=window, min_count=min_count, center=center
    )

    np.testing.assert_allclose(out[..., 1], r.mean())

    np.testing.assert_allclose(out[..., 2], r.var(ddof=0))


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
@pytest.mark.parametrize("min_count", [None, 2])
@pytest.mark.parametrize("center", [False, True])
def test_move_data_from_constructed_windows(
    rng, shape, axis, mom_ndim, window, min_count, center
) -> None:
    rng = np.random.default_rng(0)
    data = rng.random(shape)

    out = moving.move_data(
        data,
        axis=axis,
        mom_ndim=mom_ndim,
        window=window,
        min_count=min_count,
        center=center,
    )

    data_rolling = moving.construct_rolling_window_array(
        data, axis=axis, window=window, center=center, fill_value=0.0, mom_ndim=mom_ndim
    )
    out2 = cmomy.reduce_data(data_rolling, axis=-1, mom_ndim=mom_ndim)
    # clean up counts...
    count = (data_rolling[..., *((0,) * mom_ndim)] != 0.0).sum(axis=-1)

    out2 = np.where(
        count[..., *((None,) * mom_ndim)]
        >= (window if min_count is None else min_count),
        out2,
        np.nan,
    )

    # nan weights -> 0
    w = out2[..., *(0,) * mom_ndim]
    w[np.isnan(w)] = 0.0
    out2 = cmomy.convert.assign_weight(out2, w, mom_ndim=mom_ndim, copy=False)

    np.testing.assert_allclose(out, out2)

# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr

try:
    from scipy import stats as st  # pyright: ignore[reportMissingImports]
    from scipy.special import ndtr, ndtri  # pyright: ignore[reportMissingImports]

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

pytestmark = [
    pytest.mark.scipy,
    pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed"),
]

import cmomy

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from cmomy.core.typing import Moments


def test_ndtr(rng: np.random.Generator) -> None:
    from cmomy.core import prob

    x = rng.random(100)

    a = ndtr(x)
    b = prob.ndtr(x)
    np.testing.assert_allclose(a, b)

    aa = ndtri(a)
    bb = prob.ndtri(b)

    np.testing.assert_allclose(aa, bb)


@pytest.fixture(
    params=[
        ((10,), 0),
        ((2, 10), 1),
        ((10, 2), 0),
        ((2, 3, 10), 2),
        ((2, 10, 3), 1),
    ]
)
def shape_axis(request: pytest.FixtureRequest) -> tuple[tuple[int, ...], int]:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def shape(shape_axis) -> tuple[int, ...]:  # noqa: FURB118
    return shape_axis[0]  # type: ignore[no-any-return]


@pytest.fixture
def axis(shape_axis) -> int:  # noqa: FURB118
    return shape_axis[1]  # type: ignore[no-any-return]


@pytest.fixture
def data(rng: np.random.Generator, shape: tuple[int, ...]) -> NDArray[np.float64]:
    return rng.random(shape)


@pytest.fixture
def xdata(data) -> xr.DataArray:
    return xr.DataArray(data)


@pytest.fixture(params=[0.05])
def alpha(request: pytest.FixtureRequest) -> float:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def nrep() -> int:
    return 20


@pytest.fixture
def mom() -> Moments:
    return (2,)


@pytest.fixture
def sampler(data, axis, nrep) -> cmomy.resample.IndexSampler:
    return cmomy.resample.factory_sampler(
        data=data,
        axis=axis,
        nrep=nrep,
        rng=0,
    )


@pytest.fixture
def theta_hat(
    data: NDArray[np.float64], mom: Moments, axis: int
) -> NDArray[np.float64]:
    return cmomy.reduce_vals(data, mom=mom, axis=axis)


@pytest.fixture
def theta_boot(
    data: NDArray[np.float64], mom: Moments, sampler, axis
) -> NDArray[np.float64]:
    return np.moveaxis(
        cmomy.resample_vals(data, mom=mom, axis=axis, sampler=sampler), -2, axis
    )


@pytest.fixture
def theta_jack(
    data: NDArray[np.float64],
    mom: Moments,
    theta_hat: NDArray[np.float64],
    axis,
) -> NDArray[np.float64]:
    return np.moveaxis(
        cmomy.resample.jackknife_vals(data, mom=mom, data_reduced=theta_hat, axis=axis),
        -2,
        axis,
    )


@pytest.fixture
def xtheta_hat(xdata: xr.DataArray, mom, axis) -> xr.DataArray:
    return cmomy.reduce_vals(xdata, mom=mom, axis=axis)


@pytest.fixture
def xtheta_boot(
    xdata: xr.DataArray, mom: Moments, sampler: cmomy.IndexSampler, axis
) -> xr.DataArray:
    out = cmomy.resample_vals(xdata, mom=mom, axis=axis, sampler=sampler)
    dims_out = list(out.dims)
    dims_out[axis] = out.dims[-2]
    dims_out[-2] = out.dims[axis]
    return out.transpose(*dims_out)


@pytest.fixture
def xtheta_jack(
    xdata: xr.DataArray,
    mom: Moments,
    theta_hat: NDArray[np.float64],
    axis,
) -> xr.DataArray:
    out = cmomy.resample.jackknife_vals(
        xdata, mom=mom, data_reduced=theta_hat, axis=axis
    )

    dims_out = list(out.dims)
    dims_out[axis] = out.dims[-2]
    dims_out[-2] = out.dims[axis]
    return out.transpose(*dims_out)


@pytest.fixture(params=["percentile", "basic", "bca"])
def method(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def scipy_boot_mean(data, nrep, alpha, method, axis) -> st._resampling.BootstrapResult:
    return st.bootstrap(
        [data],
        np.mean,
        n_resamples=nrep,
        axis=axis,
        method=method,
        confidence_level=(1.0 - alpha),
        random_state=np.random.default_rng(0),
    )


@pytest.fixture
def scipy_boot_var(data, nrep, alpha, method, axis) -> st._resampling.BootstrapResult:
    return st.bootstrap(
        [data],
        np.var,
        n_resamples=nrep,
        axis=axis,
        method=method,
        confidence_level=(1.0 - alpha),
        random_state=np.random.default_rng(0),
    )


@pytest.fixture
def scipy_boot_mean_log(
    data, nrep, alpha, method, axis
) -> st._resampling.BootstrapResult:
    return st.bootstrap(
        [data],
        lambda x, axis=None: np.log(np.mean(x, axis=axis)),
        n_resamples=nrep,
        axis=axis,
        method=method,
        confidence_level=(1.0 - alpha),
        random_state=np.random.default_rng(0),
    )


def test_mean(
    theta_hat,
    theta_boot,
    theta_jack,
    method,
    alpha,
    scipy_boot_mean,
    axis,
) -> None:
    from cmomy.confidence_interval import bootstrap_confidence_interval

    result = bootstrap_confidence_interval(
        theta_hat=theta_hat[..., 1],
        theta_boot=theta_boot[..., 1],
        theta_jack=theta_jack[..., 1],
        alpha=alpha,
        axis=axis,
        method=method,
    )

    np.testing.assert_allclose(scipy_boot_mean.confidence_interval.low, result[0, ...])
    np.testing.assert_allclose(scipy_boot_mean.confidence_interval.high, result[1, ...])


def test_mean_log(
    theta_hat,
    theta_boot,
    theta_jack,
    method,
    alpha,
    scipy_boot_mean_log,
    axis,
) -> None:
    from cmomy.confidence_interval import bootstrap_confidence_interval

    result = bootstrap_confidence_interval(
        theta_hat=np.log(theta_hat[..., 1]),
        theta_boot=np.log(theta_boot[..., 1]),
        theta_jack=np.log(theta_jack[..., 1]),
        alpha=alpha,
        axis=axis,
        method=method,
    )

    np.testing.assert_allclose(
        scipy_boot_mean_log.confidence_interval.low, result[0, ...]
    )
    np.testing.assert_allclose(
        scipy_boot_mean_log.confidence_interval.high, result[1, ...]
    )


def test_xmean_3(
    xtheta_hat,
    xtheta_boot,
    xtheta_jack,
    method,
    alpha,
    scipy_boot_mean,
) -> None:
    from cmomy.confidence_interval import bootstrap_confidence_interval

    result = bootstrap_confidence_interval(
        theta_hat=xtheta_hat[..., 1],
        theta_boot=xtheta_boot[..., 1],
        theta_jack=xtheta_jack[..., 1],
        dim="rep",
        alpha=alpha,
        method=method,
    )

    np.testing.assert_allclose(scipy_boot_mean.confidence_interval.low, result[0, ...])
    np.testing.assert_allclose(scipy_boot_mean.confidence_interval.high, result[1, ...])


@pytest.mark.parametrize("method", ["basic", "bca"])
def test_xmean_2a(
    theta_hat,
    xtheta_boot,
    xtheta_jack,
    method,
    alpha,
) -> None:
    from cmomy.confidence_interval import bootstrap_confidence_interval

    with pytest.raises(TypeError, match=".*`theta_hat`.*"):
        _ = bootstrap_confidence_interval(
            theta_hat=theta_hat[..., 1],
            theta_boot=xtheta_boot[..., 1],
            theta_jack=xtheta_jack[..., 1],
            dim="rep",
            alpha=alpha,
            method=method,
        )


@pytest.mark.parametrize("method", ["bca"])
def test_xmean_2b(
    xtheta_hat,
    xtheta_boot,
    theta_jack,
    method,
    alpha,
) -> None:
    from cmomy.confidence_interval import bootstrap_confidence_interval

    with pytest.raises(TypeError, match=".*`theta_jack`.*"):
        _ = bootstrap_confidence_interval(
            theta_hat=xtheta_hat[..., 1],
            theta_boot=xtheta_boot[..., 1],
            theta_jack=theta_jack[..., 1],
            dim="rep",
            alpha=alpha,
            method=method,
        )


def test_var(
    theta_hat,
    theta_boot,
    theta_jack,
    method,
    alpha,
    scipy_boot_var,
    axis,
) -> None:
    from cmomy.confidence_interval import bootstrap_confidence_interval

    result = bootstrap_confidence_interval(
        theta_hat=theta_hat[..., 2],
        theta_boot=theta_boot[..., 2],
        theta_jack=theta_jack[..., 2],
        alpha=(alpha / 2, 1 - alpha / 2),
        axis=axis,
        method=method,
    )
    np.testing.assert_allclose(scipy_boot_var.confidence_interval.low, result[0, ...])
    np.testing.assert_allclose(scipy_boot_var.confidence_interval.high, result[1, ...])

    with pytest.raises(TypeError):
        result = bootstrap_confidence_interval(
            theta_hat=theta_hat[..., 2],
            theta_boot=theta_boot[..., 2],
            theta_jack=theta_jack[..., 2],
            alpha=(alpha / 2, 1 - alpha / 2),
            method=method,
        )

    if method in {"bca", "basic"}:
        with pytest.raises(ValueError):
            bootstrap_confidence_interval(
                theta_hat=None,
                theta_boot=theta_boot[..., 2],
                theta_jack=theta_jack[..., 2],
                alpha=(alpha / 2, 1 - alpha / 2),
                axis=axis,
                method=method,
            )

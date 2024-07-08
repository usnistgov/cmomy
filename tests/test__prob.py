# mypy: disable-error-code="no-untyped-def, no-untyped-call"
# pyright: reportCallIssue=false, reportArgumentType=false

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

try:
    from scipy import stats as st
    from scipy.special import ndtr, ndtri

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

    from cmomy.typing import Moments


def test_ndtr(rng: np.random.Generator) -> None:
    from cmomy import _prob

    x = rng.random(100)

    a = ndtr(x)
    b = _prob.ndtr(x)
    np.testing.assert_allclose(a, b)

    aa = ndtri(a)
    bb = _prob.ndtri(b)

    np.testing.assert_allclose(aa, bb)


@pytest.fixture(params=[(10,), (2, 10), (2, 3, 10)])
def shape(request: pytest.FixtureRequest) -> tuple[int, ...]:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def data(rng: np.random.Generator, shape: tuple[int, ...]) -> NDArray[np.float64]:
    return rng.random(shape)


@pytest.fixture(params=[0.05])
def alpha(request: pytest.FixtureRequest) -> float:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def nrep() -> int:
    return 20


@pytest.fixture
def mom() -> Moments:
    return (3,)


@pytest.fixture
def theta_hat(data: NDArray[np.float64], mom: Moments) -> NDArray[np.float64]:
    return cmomy.reduce_vals(data, mom=mom, axis=-1)


@pytest.fixture
def theta_boot(data: NDArray[np.float64], mom: Moments, nrep) -> NDArray[np.float64]:
    freq = cmomy.randsamp_freq(
        data=data, axis=-1, nrep=nrep, rng=np.random.default_rng(0)
    )
    return cmomy.resample_vals(data, mom=mom, axis=-1, freq=freq)


@pytest.fixture
def theta_jack(
    data: NDArray[np.float64], mom: Moments, theta_hat: NDArray[np.float64]
) -> NDArray[np.float64]:
    return cmomy.resample.jackknife_vals(data, mom=mom, data_reduced=theta_hat, axis=-1)


@pytest.fixture(params=["percentile", "basic", "bca"])
def method(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def scipy_boot_mean(data, nrep, alpha, method) -> st._resampling.BootstrapResult:
    return st.bootstrap(
        [data],
        np.mean,
        n_resamples=nrep,
        axis=-1,
        vectorized=True,
        method=method,
        confidence_level=(1.0 - alpha),
        random_state=np.random.default_rng(0),
    )


def test_mean(
    theta_hat, theta_boot, theta_jack, method, alpha, scipy_boot_mean
) -> None:
    from cmomy._bca import bootstrap_confidence_interval

    result = bootstrap_confidence_interval(
        theta_hat=theta_hat[..., 1],
        theta_boot=theta_boot[..., 1],
        theta_jack=theta_jack[..., 1],
        alpha=alpha,
        axis=-1,
        method=method,
    )
    np.testing.assert_allclose(scipy_boot_mean.confidence_interval.low, result[0, ...])
    np.testing.assert_allclose(scipy_boot_mean.confidence_interval.high, result[1, ...])

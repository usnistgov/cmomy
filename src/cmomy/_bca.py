from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from ._compat import copy_if_needed
from ._prob import ndtr, ndtri
from .docstrings import docfiller

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray

    from .typing import FloatDTypes, FloatT, NDArrayAny


from collections.abc import Iterable


class InstabilityWarning(UserWarning):
    """Issued when results may be unstable."""

    def __init__(self, msg: str | None = None) -> None:
        if msg is None:
            msg = "Data instability encountered; results may not be reliable."
        self.args = (msg,)


def _compute_a(
    theta_jack: NDArray[FloatT],
    axis: int = -1,
) -> FloatT | NDArray[FloatT]:
    delta = theta_jack.mean(axis=axis, keepdims=True) - theta_jack
    num = (delta**3).sum(axis=axis)
    den = (delta**2).sum(axis=axis)
    return num / (6.0 * den ** (1.5))  # type: ignore[no-any-return]


def _percentile_of_score(
    a: NDArrayAny, score: float | NDArrayAny, axis: int = -1
) -> NDArray[np.float64]:
    n = a.shape[axis]
    return ((a < score).sum(axis=axis) + (a <= score).sum(axis=axis)) / (2 * n)  # type: ignore[no-any-return]


def _compute_z0(
    theta_hat: float | NDArrayAny,
    theta_boot: NDArray[FloatT],
    axis: int = -1,
) -> NDArray[FloatT]:
    percentile = _percentile_of_score(theta_boot, theta_hat, axis=axis)
    return ndtri(percentile, dtype=theta_boot.dtype)


# modified from scipy.stats._resampling._percentile_along_axis
def _quantile_along_axis(
    theta_boot: NDArray[FloatT], alpha: NDArray[FloatT], axis: int = -1
) -> FloatT | NDArray[FloatT]:
    if axis != -1:
        theta_boot = np.moveaxis(theta_boot, axis, -1)

    shape: tuple[int, ...] = theta_boot.shape[:-1]
    broadcast_shape: tuple[int, ...] = (
        (*shape, alpha.shape[-1]) if theta_boot.ndim == alpha.ndim else shape
    )
    alpha = np.broadcast_to(alpha, broadcast_shape)
    quantiles = np.zeros_like(alpha, dtype=np.float64)

    indices: tuple[int, ...]
    for indices in np.ndindex(shape):
        idx: tuple[int | slice, ...] = (*indices, slice(None))

        alpha_i = alpha[idx]
        if np.any(np.isnan(alpha_i)):
            # e.g. when bootstrap distribution has only one unique element
            msg = (
                "The BCa confidence interval cannot be calculated."
                " This problem is known to occur when the distribution"
                " is degenerate or the statistic is np.min."
            )
            warnings.warn(InstabilityWarning(msg), stacklevel=2)
            quantiles[idx] = np.nan
        else:
            quantiles[idx] = np.quantile(theta_boot[idx], alpha_i)

    return quantiles[()]  # type: ignore[return-value]


@docfiller.decorate
def bootstrap_confidence_interval(
    theta_boot: NDArrayAny,
    theta_hat: float | FloatDTypes | NDArrayAny | None = None,
    theta_jack: NDArrayAny | None = None,
    alpha: float | FloatDTypes | Iterable[float | FloatDTypes] = 0.05,
    axis: int = -1,
    method: Literal["percentile", "basic", "bca"] = "bca",
) -> NDArrayAny:
    r"""
    Create the bootstrap confidence interval using BCa analysis.

    Parameters
    ----------
    theta_hat : ndarray
        Results of :math:`\theta` from original data set.
    theta_boot : ndarray
        Bootstrapped resampling of original data set.
    theta_jack : ndarray
        Jackknife resampled data.
    alphas : float or iterable of floats
        The quantiles to use for confidence interval. If ``alpha`` is a float,
        then Use (alpha/2, 1-alpha/2) for confidence intervals (e.g., pass
        `alpha=0.05` for the two-sided 95% confidence interval). If ``alpha``
        is an iterable, use these values.
    {axis}
    method : {{'percentile', 'basic', 'bca'}}, default: ``'BCa'``
            Whether to return the 'percentile' bootstrap confidence interval
            (``'percentile'``), the 'basic' (AKA 'reverse') bootstrap confidence
            interval (``'basic'``), or the bias-corrected and accelerated bootstrap
            confidence interval (``'BCa'``).


    Returns
    -------
    confindence_interval : ndarray
        Array of confidence intervals. ``confidence_interval[i, ...]``
        corresponds ``alphas[i]``. That is, ``confidence_interval.shape =
        (nalpha, shape[0], ..., shape[ndim])`` Where `shape =
        theta_hat.shape``.


    See Also
    --------
    scipy.stats.bootstrap

    """
    dtype = theta_boot.dtype

    if isinstance(alpha, Iterable):
        alphas = np.array(list(alpha), dtype=dtype)
    else:
        alphas = np.asarray([alpha * 0.5, 1.0 - alpha * 0.5])

    method_ = method.lower()
    if method_ in {"basic", "bca"}:
        if theta_hat is None:
            msg = f"Must specify theta_hat for {method=}"
            raise ValueError(msg)
        theta_hat_ = np.array(
            theta_hat, ndmin=1, dtype=dtype, copy=copy_if_needed(None)
        )

        # expand theta hat?
        if theta_hat_.ndim < theta_boot.ndim:
            theta_hat_ = np.expand_dims(theta_hat, axis=axis)

    if method_ == "bca":
        theta_jack = np.asarray(theta_jack, dtype=dtype)

        a = _compute_a(theta_jack, axis)
        z0 = _compute_z0(theta_hat_, theta_boot, axis)

        z0_expanded = np.expand_dims(z0, -1)
        a_expanded = np.expand_dims(a, -1)
        zs = z0_expanded + ndtri(alphas).reshape((1,) * z0.ndim + alphas.shape)
        avals = ndtr(z0_expanded + zs / (1 - a_expanded * zs))
        ci = np.moveaxis(_quantile_along_axis(theta_boot, avals, axis), -1, 0)

    else:
        ci = np.quantile(theta_boot, alphas, axis=axis)
        if method_ == "basic":
            ci = 2 * np.squeeze(theta_hat_, axis=axis) - ci[-1::-1, ...]

    return ci

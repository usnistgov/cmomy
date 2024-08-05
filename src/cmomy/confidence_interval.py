"""Routines to calculate confidence intervals from resampled data."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast, overload

import numpy as np
import xarray as xr

from .core.compat import copy_if_needed
from .core.docstrings import docfiller
from .core.missing import MISSING
from .core.prob import ndtr, ndtri
from .core.validate import validate_axis
from .core.xr_utils import select_axis_dim

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import DTypeLike, NDArray

    from .core.typing import (
        AxisReduce,
        DimsReduce,
        FloatDTypes,
        FloatT,
        KeepAttrs,
        MissingType,
        NDArrayAny,
    )


from collections.abc import Iterable


class InstabilityWarning(UserWarning):
    """Issued when results may be unstable."""

    def __init__(self, msg: str | None = None) -> None:  # pragma: no cover
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
) -> NDArray[FloatT]:
    if axis != -1:
        theta_boot = np.moveaxis(theta_boot, axis, -1)

    shape: tuple[int, ...] = theta_boot.shape[:-1]
    broadcast_shape: tuple[int, ...] = (
        (*shape, alpha.shape[-1]) if theta_boot.ndim == alpha.ndim else shape
    )
    alpha = np.broadcast_to(alpha, broadcast_shape)
    quantiles = np.zeros_like(alpha, dtype=theta_boot.dtype)

    indices: tuple[int, ...]
    for indices in np.ndindex(shape):
        idx: tuple[int | slice, ...] = (*indices, slice(None))

        alpha_i = alpha[idx]
        if np.any(np.isnan(alpha_i)):  # pragma: no cover
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

    return quantiles


@overload
def bootstrap_confidence_interval(
    theta_boot: xr.DataArray,
    theta_hat: float | FloatDTypes | NDArrayAny | xr.DataArray | None = ...,
    theta_jack: NDArrayAny | xr.DataArray | None = ...,
    *,
    alpha: float | FloatDTypes | Iterable[float | FloatDTypes] = ...,
    axis: AxisReduce | MissingType = ...,
    method: Literal["percentile", "basic", "bca"] = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    ci_dim: str = ...,
    keep_attrs: KeepAttrs = ...,
) -> xr.DataArray: ...


@overload
def bootstrap_confidence_interval(
    theta_boot: NDArray[FloatT],
    theta_hat: float | FloatDTypes | NDArrayAny | xr.DataArray | None = ...,
    theta_jack: NDArrayAny | xr.DataArray | None = ...,
    *,
    alpha: float | FloatDTypes | Iterable[float | FloatDTypes] = ...,
    axis: AxisReduce | MissingType = ...,
    method: Literal["percentile", "basic", "bca"] = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    ci_dim: str = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...


@docfiller.decorate
def bootstrap_confidence_interval(
    theta_boot: NDArray[FloatT] | xr.DataArray,
    theta_hat: float | FloatDTypes | NDArrayAny | xr.DataArray | None = None,
    theta_jack: NDArrayAny | xr.DataArray | None = None,
    *,
    alpha: float | FloatDTypes | Iterable[float | FloatDTypes] = 0.05,
    axis: AxisReduce | MissingType = MISSING,
    method: Literal["percentile", "basic", "bca"] = "bca",
    # xarray specific
    dim: DimsReduce | MissingType = MISSING,
    ci_dim: str = "alpha",
    keep_attrs: KeepAttrs = None,
) -> NDArray[FloatT] | xr.DataArray:
    r"""
    Create the bootstrap confidence interval.

    The general idea is to analyze some function :math:`\theta` which is a function of the central moments.

    Parameters
    ----------
    theta_boot : ndarray
        Bootstrapped resampled values of :math:`\theta`
    theta_hat : ndarray
        Results of :math:`\theta` from original data set. Needed for ``method``
        values `'basic'`` and ``'bca'``. Note that this array should have shape
        as ``theta_boot`` with ``axis`` either removed, or of size ``1``.
    theta_jack : ndarray
        Jackknife resampled data.  Needed for ``method`` ``'basic'``.
        Note that this array should have the same shape as ``theta_boot`` except along ``axis``.
    alphas : float or iterable of floats
        The quantiles to use for confidence interval. If ``alpha`` is a float,
        then Use (alpha/2, 1-alpha/2) for confidence intervals (e.g., pass
        `alpha=0.05` for the two-sided 95% confidence interval). If ``alpha``
        is an iterable, use these values.
    {axis}
    method : {{'percentile', 'basic', 'bca'}}, default=``'bca'``
            Whether to return the 'percentile' bootstrap confidence interval
            (``'percentile'``), the 'basic' (AKA 'reverse') bootstrap confidence
            interval (``'basic'``), or the bias-corrected and accelerated bootstrap
            confidence interval (``'BCa'``).
    {dim}
    ci_dim : str, default="alpha"
        Name of confidence level dimension of :class:`~xarray.DataArray` output.
    {keep_attrs}

    Returns
    -------
    confindence_interval : ndarray
        Array of confidence intervals. ``confidence_interval[i, ...]``
        corresponds ``alphas[i]``. That is, ``confidence_interval.shape =
        (nalpha, shape[0], ..., shape[axis-1], shape[axis+1], ...)`` Where ``shape =
        theta_boot.shape``.


    See Also
    --------
    .reduce_data : Create ``theta_hat`` from moments data
    .resample_data : Create ``theta_boot`` from moments data
    .jackknife_data : Create ``theta_jack`` from moments data
    .reduce_vals : Create ``theta_hat`` from values
    .resample_vals : Create ``theta_boot`` from values
    .jackknife_vals : Create ``theta_jack`` from values

    scipy.stats.bootstrap : Scipy analog function


    Examples
    --------
    Calculate the bootstrap statistics of the log of the mean.

    >>> import cmomy
    >>> x = cmomy.random.default_rng(0).random((20))
    >>> freq = cmomy.randsamp_freq(nrep=50, ndat=20, rng=np.random.default_rng(0))
    >>> theta_boot = np.log(cmomy.resample_vals(x, mom=1, axis=0, freq=freq)[..., 1])
    >>> bootstrap_confidence_interval(
    ...     theta_boot=theta_boot, axis=0, method="percentile"
    ... )
    array([-1.0016, -0.4722])

    To use the `basic` analysis, must also pass in ``theta_hat``

    >>> theta_hat = np.log(cmomy.reduce_vals(x, mom=1, axis=0)[..., 1])
    >>> bootstrap_confidence_interval(
    ...     theta_boot=theta_boot, theta_hat=theta_hat, axis=0, method="basic"
    ... )
    array([-0.8653, -0.3359])


    To use `bca`, also need jackknife resampled data.

    >>> theta_jack = np.log(cmomy.resample.jackknife_vals(x, mom=1, axis=0)[..., 1])
    >>> bootstrap_confidence_interval(
    ...     theta_boot=theta_boot,
    ...     theta_hat=theta_hat,
    ...     theta_jack=theta_jack,
    ...     axis=0,
    ...     method="bca",
    ... )
    array([-0.986 , -0.4517])


    These results are the same as using :func:`scipy.stats.bootstrap`, but should be faster.

    >>> from scipy.stats import bootstrap
    >>> out = bootstrap(
    ...     [x],
    ...     lambda x, axis=None: np.log(np.mean(x, axis=axis)),
    ...     n_resamples=50,
    ...     axis=0,
    ...     random_state=np.random.default_rng(0),
    ...     method="bca",
    ... )
    >>> np.array((out.confidence_interval.low, out.confidence_interval.high))
    array([-0.986 , -0.4517])

    Moreover, you can use pre-averaged data.
    """
    dtype: DTypeLike = theta_boot.dtype  # pyright: ignore[reportUnknownMemberType]
    if isinstance(alpha, Iterable):
        alphas = np.array(list(alpha), dtype=dtype)
    else:
        alphas = np.asarray([alpha * 0.5, 1.0 - alpha * 0.5])

    if isinstance(theta_boot, xr.DataArray):
        axis, dim = select_axis_dim(theta_boot, axis=axis, dim=dim)

        if isinstance(theta_jack, xr.DataArray):
            theta_jack = theta_jack.rename({dim: "_rep_jack"})
        elif theta_jack is not None:
            theta_jack = np.moveaxis(np.asarray(theta_jack, dtype=dtype), axis, -1)

        return cast(
            xr.DataArray,
            xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
                lambda *args, **kwargs: np.moveaxis(  # pyright: ignore[reportUnknownLambdaType]
                    bootstrap_confidence_interval(*args, **kwargs),  # pyright: ignore[reportUnknownArgumentType]
                    0,
                    -1,  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
                ),
                theta_boot,
                theta_hat,
                theta_jack,
                input_core_dims=[[dim], [], ["_rep_jack"]],
                output_core_dims=[[ci_dim]],
                kwargs={
                    "axis": -1,
                    "method": method,
                },
                keep_attrs=keep_attrs,
            )
            .transpose(ci_dim, ...)
            .assign_coords({ci_dim: alphas}),
        )

    axis = validate_axis(axis)
    method_ = method.lower()
    if method_ in {"basic", "bca"}:
        if theta_hat is None:
            msg = f"Must specify theta_hat for {method=}"
            raise ValueError(msg)
        theta_hat_: NDArray[FloatT] = np.array(
            theta_hat, ndmin=1, dtype=dtype, copy=copy_if_needed(None)
        )

        # expand theta hat?
        if theta_hat_.ndim < theta_boot.ndim:
            theta_hat_ = np.expand_dims(theta_hat, axis=axis)

    ci: NDArray[FloatT]
    if method_ == "bca":
        theta_jack = np.asarray(theta_jack, dtype=dtype)

        a = _compute_a(theta_jack, axis)
        z0 = _compute_z0(theta_hat_, theta_boot, axis)  # pyright: ignore[reportPossiblyUnboundVariable]

        z0_expanded = np.expand_dims(z0, -1)
        a_expanded = np.expand_dims(a, -1)
        zs = z0_expanded + ndtri(alphas).reshape((1,) * z0.ndim + alphas.shape)
        avals = ndtr(z0_expanded + zs / (1 - a_expanded * zs))
        ci = np.moveaxis(_quantile_along_axis(theta_boot, avals, axis), -1, 0)

    else:
        ci = np.quantile(theta_boot, alphas, axis=axis).astype(dtype, copy=False)
        if method_ == "basic":
            ci = 2 * np.squeeze(theta_hat_, axis=axis) - ci[-1::-1, ...]  # pyright: ignore[reportAssignmentType, reportPossiblyUnboundVariable]

    return ci


# def ci_style(val: NDArray[FloatT], ci: NDArray[FloatT], style: Literal["delta", "pm"]) -> NDArray[FloatT]:
#     if style is None:
#         out = np.array([val, ci[0, ...], ci[1, ...])
#     elif style == "delta":  # noqa: ERA001
#         out = np.array([val, val - ci[0, ...], ci[1, ...] - val])  # noqa: ERA001
#     elif style == "pm":  # noqa: ERA001
#         out = np.array([val, (ci[1, ...] - ci[0, ...]) / 2.0])  # noqa: ERA001
#     return out  # noqa: ERA001

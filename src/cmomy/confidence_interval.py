"""
Routines to calculate confidence intervals from resampled data (:mod:`cmomy.confidence_interval`)
=================================================================================================
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr

from cmomy.core.array_utils import select_dtype

from .core.compat import copy_if_needed
from .core.docstrings import docfiller
from .core.missing import MISSING
from .core.prob import ndtr, ndtri
from .core.validate import is_xarray, validate_axis
from .core.xr_utils import factory_apply_ufunc_kwargs, select_axis_dim

if TYPE_CHECKING:
    from collections.abc import Hashable
    from typing import Any

    from numpy.typing import NDArray

    from .core.typing import (
        ApplyUFuncKwargs,
        AxisReduce,
        BootStrapMethod,
        DataT,
        DimsReduce,
        FloatDTypes,
        FloatingT,
        KeepAttrs,
        MissingType,
        NDArrayAny,
    )


@overload
def bootstrap_confidence_interval(
    theta_boot: DataT,
    theta_hat: DataT | None = ...,
    theta_jack: DataT | None = ...,
    *,
    alpha: float | FloatDTypes | Iterable[float | FloatDTypes] = ...,
    axis: AxisReduce | MissingType = ...,
    method: BootStrapMethod = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    ci_dim: str = ...,
    keep_attrs: KeepAttrs = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> DataT: ...
@overload
def bootstrap_confidence_interval(
    theta_boot: NDArray[FloatingT],
    theta_hat: float | FloatDTypes | NDArrayAny | None = ...,
    theta_jack: NDArrayAny | None = ...,
    *,
    alpha: float | FloatDTypes | Iterable[float | FloatDTypes] = ...,
    axis: AxisReduce | MissingType = ...,
    method: BootStrapMethod = ...,
    # xarray specific
    dim: DimsReduce | MissingType = ...,
    ci_dim: str = ...,
    keep_attrs: KeepAttrs = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArray[FloatingT]: ...


@docfiller.decorate
def bootstrap_confidence_interval(
    theta_boot: DataT | NDArray[FloatingT],
    theta_hat: float | FloatDTypes | NDArrayAny | DataT | None = None,
    theta_jack: NDArrayAny | DataT | None = None,
    *,
    alpha: float | FloatDTypes | Iterable[float | FloatDTypes] = 0.05,
    axis: AxisReduce | MissingType = MISSING,
    method: BootStrapMethod = "bca",
    # xarray specific
    dim: DimsReduce | MissingType = MISSING,
    ci_dim: str = "alpha",
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArray[FloatingT] | DataT:
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
        Jackknife resampled data.  Needed for ``method`` ``'bca'``.
        Note that this array should have the same shape as ``theta_boot`` except along ``axis``.
    alphas : float or iterable of float
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
    >>> x = cmomy.default_rng(0).random((20))
    >>> sampler = cmomy.resample.factory_sampler(nrep=50, ndat=20, rng=0)
    >>> theta_boot = np.log(
    ...     cmomy.resample_vals(x, mom=1, axis=0, sampler=sampler)[..., 1]
    ... )
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
    dtype = select_dtype(theta_boot, out=None, dtype=None)
    if isinstance(alpha, Iterable):
        alphas = np.array(list(alpha), dtype=dtype)
    else:
        alphas = np.asarray([alpha * 0.5, 1.0 - alpha * 0.5])

    if is_xarray(theta_boot):
        axis, dim = select_axis_dim(theta_boot, axis=axis, dim=dim)

        if is_xarray(theta_jack):
            theta_jack = theta_jack.rename({dim: "_rep_jack"})
        elif theta_jack is not None:
            theta_jack = np.moveaxis(np.asarray(theta_jack, dtype=dtype), axis, -1)

        def _func(*args: NDArrayAny, **kwargs: Any) -> NDArrayAny:
            out = _bootstrap_confidence_interval(*args, **kwargs)
            # move axis to end for apply_ufunc
            return np.moveaxis(out, 0, -1)

        args: list[DataT] = [theta_boot]
        input_core_dims: list[list[Hashable]] = [[dim]]
        if method in {"basic", "bca"}:
            if not is_xarray(theta_hat):
                msg = "`theta_hat` must be an xarray object"
                raise TypeError(msg)
            args.append(theta_hat)
            input_core_dims.append([])
        if method == "bca":
            if not is_xarray(theta_jack):
                msg = "`theta_jack` must be an xarray object"
                raise TypeError(msg)
            args.append(theta_jack)
            input_core_dims.append(["_rep_jack"])

        xout: DataT = (
            xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
                _func,
                *args,
                input_core_dims=input_core_dims,
                output_core_dims=[[ci_dim]],
                kwargs={
                    "alphas": alphas,
                    "axis": -1,
                    "method": method,
                },
                keep_attrs=keep_attrs,
                **factory_apply_ufunc_kwargs(
                    apply_ufunc_kwargs,
                    dask="parallelized",
                    output_dtypes=dtype or np.float64,
                    output_sizes={ci_dim: len(alphas)},
                ),
            )
            .transpose(ci_dim, ...)
            .assign_coords({ci_dim: alphas})
        )
        return xout

    # Numpy
    assert not is_xarray(theta_hat)  # noqa: S101
    assert not is_xarray(theta_jack)  # noqa: S101

    return _bootstrap_confidence_interval(
        theta_boot=theta_boot,
        theta_hat=theta_hat,
        theta_jack=theta_jack,
        alphas=alphas,
        axis=validate_axis(axis),
        method=method,
    )


def _bootstrap_confidence_interval(
    theta_boot: NDArray[FloatingT],
    theta_hat: float | FloatDTypes | NDArrayAny | None = None,
    theta_jack: NDArrayAny | None = None,
    *,
    alphas: NDArrayAny,
    method: BootStrapMethod,
    axis: int,
) -> NDArray[FloatingT]:
    dtype = theta_boot.dtype
    method_ = method.lower()
    if method_ in {"basic", "bca"}:
        if theta_hat is None:
            msg = f"Must specify theta_hat for {method=}"
            raise ValueError(msg)
        theta_hat_: NDArray[FloatingT] = np.array(
            theta_hat, ndmin=1, dtype=dtype, copy=copy_if_needed(None)
        )

        # expand theta hat?
        if theta_hat_.ndim < theta_boot.ndim:
            theta_hat_ = np.expand_dims(theta_hat_, axis=axis)

    ci: NDArray[FloatingT]
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


# * Utilities -----------------------------------------------------------------
class InstabilityWarning(UserWarning):
    """Issued when results may be unstable."""

    def __init__(self, msg: str | None = None) -> None:  # pragma: no cover
        if msg is None:
            msg = "Data instability encountered; results may not be reliable."
        self.args = (msg,)


def _compute_a(
    theta_jack: NDArray[FloatingT],
    axis: int = -1,
) -> FloatingT | NDArray[FloatingT]:
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
    theta_boot: NDArray[FloatingT],
    axis: int = -1,
) -> NDArray[FloatingT]:
    percentile = _percentile_of_score(theta_boot, theta_hat, axis=axis)
    return ndtri(percentile, dtype=theta_boot.dtype)


# modified from scipy.stats._resampling._percentile_along_axis
def _quantile_along_axis(
    theta_boot: NDArray[FloatingT], alpha: NDArray[FloatingT], axis: int = -1
) -> NDArray[FloatingT]:
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

"""
Calculate central (co)moments from values using classic method (:mod:`~cmomy.central_from_values`)
==================================================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

# pandas needed for autdoc typehints
import pandas as pd  # noqa: F401  # pyright: ignore[reportUnusedImport]
import xarray as xr

from ._compat import xr_dot
from .docstrings import docfiller_central as docfiller
from .utils import axis_expand_broadcast, select_axis_dim

if TYPE_CHECKING:
    from typing import Hashable

    from numpy.typing import ArrayLike, DTypeLike

    from .typing import (
        ArrayOrder,
        MomDims,
        Moments,
        NDArrayAny,
    )


# * numpy array
def _central_moments(
    vals: ArrayLike,
    mom: Moments,
    w: NDArrayAny | None = None,
    axis: int = 0,
    last: bool = True,
    dtype: DTypeLike | None = None,
    order: ArrayOrder | None = None,
    out: NDArrayAny | None = None,
) -> NDArrayAny:
    """Calculate central mom along axis."""
    if isinstance(mom, tuple):  # pragma: no cover
        mom = mom[0]

    x = np.asarray(vals, dtype=dtype, order=order)
    if dtype is None:
        dtype = x.dtype

    if w is None:
        w = np.ones_like(x)
    else:
        w = axis_expand_broadcast(
            w, shape=x.shape, axis=axis, roll=False, dtype=dtype, order=order
        )

    # if axis < 0:
    #     axis += x.ndim
    if axis != 0:
        x = np.moveaxis(x, axis, 0)
        w = np.moveaxis(w, axis, 0)

    shape = (mom + 1,) + x.shape[1:]
    if out is None:
        out = np.empty(shape, dtype=dtype)
    elif out.shape != shape:
        # try rolling
        out = np.moveaxis(out, -1, 0)
        if out.shape != shape:
            raise ValueError

    wsum = w.sum(axis=0)  # pyright: ignore[reportUnknownMemberType]
    wsum_inv = 1.0 / wsum
    xave = np.einsum("r...,r...->...", w, x) * wsum_inv  # pyright: ignore[reportUnknownMemberType]

    shape = (-1,) + (1,) * (x.ndim)
    p = np.arange(2, mom + 1).reshape(*shape)  # pyright: ignore[reportUnknownMemberType]

    dx = (x[None, ...] - xave) ** p

    out[0, ...] = wsum
    out[1, ...] = xave
    out[2:, ...] = np.einsum("r..., mr...->m...", w, dx) * wsum_inv  # pyright: ignore[reportUnknownMemberType]

    if last:
        out = np.moveaxis(out, 0, -1)
    return out


def _central_comoments(  # noqa: C901, PLR0912
    vals: tuple[NDArrayAny, NDArrayAny],
    mom: tuple[int, int],
    w: NDArrayAny | None = None,
    axis: int = 0,
    last: bool = True,
    broadcast: bool = False,
    dtype: DTypeLike | None = None,
    order: ArrayOrder | None = None,
    out: NDArrayAny | None = None,
) -> NDArrayAny:
    """Calculate central co-mom (covariance, etc) along axis."""
    if not isinstance(
        mom, tuple
    ):  # pragma: no cover  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError

    if len(mom) != 2:
        raise ValueError

    # change x to tuple of inputs
    if not isinstance(vals, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError
    if len(vals) != 2:
        raise ValueError
    x, y = vals

    x = np.asarray(x, dtype=dtype, order=order)
    if dtype is None:
        dtype = x.dtype

    y = axis_expand_broadcast(
        y,
        shape=x.shape,
        axis=axis,
        roll=False,
        broadcast=broadcast,
        expand=broadcast,
        dtype=dtype,
        order=order,
    )

    if w is None:
        w = np.ones_like(x)
    else:
        w = axis_expand_broadcast(
            w, shape=x.shape, axis=axis, roll=False, dtype=dtype, order=order
        )

    if w.shape != x.shape or y.shape != x.shape:
        raise ValueError

    if axis != 0:
        x = np.moveaxis(x, axis, 0)
        y = np.moveaxis(y, axis, 0)
        w = np.moveaxis(w, axis, 0)

    shape = tuple(x + 1 for x in mom) + x.shape[1:]
    if out is None:
        out = np.empty(shape, dtype=dtype)
    elif out.shape != shape:
        # try moving axis
        out = np.moveaxis(out, [-2, -1], [0, 1])
        if out.shape != shape:
            raise ValueError

    wsum = w.sum(axis=0)  # pyright: ignore[reportUnknownMemberType]
    wsum_inv = 1.0 / wsum

    xave = np.einsum("r...,r...->...", w, x) * wsum_inv  # pyright: ignore[reportUnknownMemberType]
    yave = np.einsum("r...,r...->...", w, y) * wsum_inv  # pyright: ignore[reportUnknownMemberType]

    shape = (-1,) + (1,) * (x.ndim)
    p0 = np.arange(0, mom[0] + 1).reshape(*shape)  # pyright: ignore[reportUnknownMemberType]
    p1 = np.arange(0, mom[1] + 1).reshape(*shape)  # pyright: ignore[reportUnknownMemberType]

    dx = (x[None, ...] - xave) ** p0
    dy = (y[None, ...] - yave) ** p1

    out[...] = (
        np.einsum("r...,ir...,jr...->ij...", w, dx, dy) * wsum_inv  # pyright: ignore[reportUnknownMemberType]
    )

    out[0, 0, ...] = wsum
    out[1, 0, ...] = xave
    out[0, 1, ...] = yave

    if last:
        out = np.moveaxis(out, [0, 1], [-2, -1])
    return out


@docfiller.decorate
def central_moments(
    x: NDArrayAny | tuple[NDArrayAny, NDArrayAny],
    mom: Moments,
    *,
    w: NDArrayAny | None = None,
    axis: int = 0,
    last: bool = True,
    dtype: DTypeLike | None = None,
    order: ArrayOrder | None = None,
    out: NDArrayAny | None = None,
    broadcast: bool = False,
) -> NDArrayAny:
    """
    Calculate central moments or comoments along axis.

    Parameters
    ----------
    x : array-like or tuple of array-like
        if calculating moments, then this is the input array.
        if calculating comoments, then pass in tuple of values of form (x, y)
    {mom}
    w : array-like, optional
        Weights. If passed, should be able to broadcast to `x`. An exception is if
        w is a 1d array with len(w) == x.shape[axis]. In this case,
        w will be reshaped and broadcast against x
    {axis}
    last : bool, default=True
        if True, put mom as last dimension.
        Otherwise, mom will be in first dimension
    {dtype}
    {broadcast}
    out : array
        if present, use this for output data
        Needs to have shape of either (mom,) + shape or shape + (mom,)
        where shape is the shape of x with axis removed

    Returns
    -------
    output : array
        array of shape=shape + mom_shape or mom_shape + shape depending on
        value of `last`, where `shape` is the shape of `x` with axis removed,
        i.e., shape=x.shape[:axis] + x.shape[axis+1:], and `mom_shape` is the shape of
        the moment part, either (mom+1,) or (mom0+1, mom1+1).  Assuming `last is True`,
        output[...,0] is the total weight (or count), output[...,1] is the mean
        value of x, output[...,n] with n>1 is the nth central moment.

    See Also
    --------
    CentralMoments


    Examples
    --------
    create data:

    >>> from cmomy.random import default_rng
    >>> rng = default_rng(0)
    >>> x = rng.random(10)

    Generate first 2 central moments:

    >>> moments = central_moments(x=x, mom=2)
    >>> print(moments)
    [10.      0.5505  0.1014]

    Generate moments with weights

    >>> w = rng.random(10)
    >>> central_moments(x=x, w=w, mom=2)
    array([4.7419, 0.5877, 0.0818])


    Generate co-moments

    >>> y = rng.random(10)
    >>> central_moments(x=(x, y), w=w, mom=(2, 2))
    array([[ 4.7419e+00,  6.3452e-01,  1.0383e-01],
           [ 5.8766e-01, -5.1403e-03,  6.1079e-03],
           [ 8.1817e-02,  1.5621e-03,  7.7609e-04]])

    """
    if isinstance(mom, int):
        mom = (mom,)

    if len(mom) == 1:
        return _central_moments(
            vals=x,
            mom=mom,
            w=w,
            axis=axis,
            last=last,
            dtype=dtype,
            order=order,
            out=out,
        )
    return _central_comoments(
        vals=x,  # type: ignore[arg-type]
        mom=mom,
        w=w,
        axis=axis,
        last=last,
        dtype=dtype,
        order=order,
        broadcast=broadcast,
        out=out,
    )


# * xcentral moments/comoments
def _xcentral_moments(
    vals: xr.DataArray,
    mom: Moments,
    w: xr.DataArray | None = None,
    axis: int | None = None,
    dim: Hashable | None = None,
    mom_dims: MomDims | None = None,
) -> xr.DataArray:
    x = vals
    if not isinstance(x, xr.DataArray):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError

    if isinstance(mom, tuple):
        mom = mom[0]

    if mom_dims is None:
        mom_dims = ("mom_0",)
    elif isinstance(mom_dims, str):
        mom_dims = (mom_dims,)
    if len(mom_dims) != 1:  # type: ignore[arg-type]
        raise ValueError

    if w is None:
        # fmt: off
        w = xr.ones_like(x)  # pyright: ignore[reportUnknownMemberType] # needed for python=3.8
        # fmt: on
    else:
        w = xr.DataArray(w).broadcast_like(x)

        axis, dim = select_axis_dim(dims=x.dims, axis=axis, dim=dim, default_axis=0)

    if TYPE_CHECKING:
        mom_dims = cast("tuple[Hashable]", mom_dims)
        dim = cast(str, dim)

    wsum = w.sum(dim=dim)
    wsum_inv = 1.0 / wsum

    # fmt: off
    xave = xr_dot(w, x, dim=dim) * wsum_inv  # pyright: ignore[reportUnknownMemberType]
    p = xr.DataArray(
        np.arange(0, mom + 1), dims=mom_dims  # pyright: ignore[reportUnknownMemberType]
    )
    dx = (x - xave) ** p
    out = xr_dot(w, dx, dim=dim) * wsum_inv  # pyright: ignore[reportUnknownMemberType]
    # fmt: on

    out.loc[{mom_dims[0]: 0}] = wsum
    out.loc[{mom_dims[0]: 1}] = xave

    # ensure in correct order
    out = out.transpose(..., *mom_dims)
    return cast(xr.DataArray, out)


def _xcentral_comoments(  # noqa: C901
    vals: tuple[xr.DataArray, xr.DataArray],
    mom: Moments,
    w: xr.DataArray | None = None,
    axis: int | None = None,
    dim: Hashable | None = None,
    broadcast: bool = False,
    mom_dims: tuple[Hashable, Hashable] | None = None,
) -> xr.DataArray:
    """Calculate central co-mom (covariance, etc) along axis."""
    mom = (mom, mom) if isinstance(mom, int) else tuple(mom)  # type: ignore[assignment]

    assert isinstance(mom, tuple)  # noqa: S101  # pragma: no cover

    if len(mom) != 2:
        raise ValueError
    if not isinstance(vals, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError
    if len(vals) != 2:
        raise ValueError

    x, y = vals

    if not isinstance(x, xr.DataArray):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError

    w = xr.ones_like(x) if w is None else xr.DataArray(w).broadcast_like(x)  # pyright: ignore[reportUnknownMemberType] # needed for python=3.8

    if broadcast:
        y = xr.DataArray(y).broadcast_like(x)
    elif not isinstance(y, xr.DataArray):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError
    else:
        y = y.transpose(*x.dims)
        if y.shape != x.shape or x.dims != y.dims:
            raise ValueError

    axis, dim = select_axis_dim(dims=x.dims, axis=axis, dim=dim, default_axis=0)

    if mom_dims is None:
        mom_dims = ("mom_0", "mom_1")

    if len(mom_dims) != 2:
        raise ValueError

    if TYPE_CHECKING:
        dim = cast(str, dim)

    wsum = w.sum(dim=dim)
    wsum_inv = 1.0 / wsum

    xy = (x, y)

    # fmt: off
    xave = [xr_dot(w, xx, dim=dim) * wsum_inv for xx in xy]  # pyright: ignore[reportUnknownMemberType]
    p = [
        xr.DataArray(np.arange(0, mom + 1), dims=dim)  # type: ignore[arg-type, unused-ignore]  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        for mom, dim in zip(mom, mom_dims)
    ]

    dx = [(xx - xxave) ** pp for xx, xxave, pp in zip(xy, xave, p)]
    out = xr_dot(w, dx[0], dx[1], dim=dim) * wsum_inv  # pyright: ignore[reportUnknownMemberType]
    # fmt: on

    out.loc[{mom_dims[0]: 0, mom_dims[1]: 0}] = wsum
    out.loc[{mom_dims[0]: 1, mom_dims[1]: 0}] = xave[0]
    out.loc[{mom_dims[0]: 0, mom_dims[1]: 1}] = xave[1]

    out = out.transpose(..., *mom_dims)
    return cast(xr.DataArray, out)


@docfiller.decorate
def xcentral_moments(
    x: xr.DataArray | tuple[xr.DataArray, xr.DataArray],
    mom: Moments,
    *,
    w: xr.DataArray | None = None,
    axis: int | None = None,
    dim: Hashable | None = None,
    mom_dims: MomDims | None = None,
    broadcast: bool = False,
) -> xr.DataArray:
    """
    Calculate central mom along axis.

    Parameters
    ----------
    x : DataArray or tuple of DataArray
        input data
    {mom}
    w : array-like, optional
        if passed, should be able to broadcast to `x`. An exception is if
        w is a 1d array with len(w) == x.shape[axis]. In this case,
        w will be reshaped and broadcast against x
    {axis}
    {dim}
    {dtype}
    {mom_dims}
    {broadcast}

    Returns
    -------
    output : DataArray
        array of shape shape + (mom,) or (mom,) + shape depending on
        value of `last`, where `shape` is the shape of `x` with axis removed,
        i.e., shape=x.shape[:axis] + x.shape[axis+1:]. Assuming `last is True`,
        output[...,0] is the total weight (or count), output[...,1] is the mean
        value of x, output[...,n] with n>1 is the nth central moment
    """
    if isinstance(mom, int):
        mom = (mom,)

    kwargs = {
        "vals": x,
        "mom": mom,
        "w": w,
        "axis": axis,
        "dim": dim,
        "mom_dims": mom_dims,
    }
    if len(mom) == 1:
        out = _xcentral_moments(**kwargs)  # type: ignore[arg-type]
    else:
        kwargs["broadcast"] = broadcast
        out = _xcentral_comoments(**kwargs)  # type: ignore[arg-type]

    return out

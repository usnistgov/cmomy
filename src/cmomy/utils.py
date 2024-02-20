"""Utilities."""
from __future__ import annotations

# from functools import lru_cache
from typing import TYPE_CHECKING, cast

import numpy as np

from .docstrings import docfiller

if TYPE_CHECKING:
    from typing import Sequence

    from numpy.typing import ArrayLike, DTypeLike

    from .typing import ArrayOrder, Mom_NDim, Moments, MomentsStrict, MyNDArray


def normalize_axis_index(axis: int, ndim: int) -> int:
    """Interface to numpy.core.multiarray.normalize_axis_index"""
    import numpy.core.multiarray as ma

    return ma.normalize_axis_index(axis, ndim)  # type: ignore[no-any-return,attr-defined]


def shape_insert_axis(
    *,
    shape: Sequence[int],
    axis: int | None,
    new_size: int,
) -> tuple[int, ...]:
    """Get new shape, given shape, with size put in position axis."""
    if axis is None:
        msg = "must specify integre axis"
        raise ValueError(msg)

    axis = normalize_axis_index(axis, len(shape) + 1)
    shape = tuple(shape)
    return shape[:axis] + (new_size,) + shape[axis:]


def shape_reduce(*, shape: tuple[int, ...], axis: int) -> tuple[int, ...]:
    """Give shape shape after reducing along axis."""
    shape_list = list(shape)
    shape_list.pop(axis)
    return tuple(shape_list)


def axis_expand_broadcast(
    x: ArrayLike,
    *,
    shape: tuple[int, ...],
    axis: int | None,
    verify: bool = True,
    expand: bool = True,
    broadcast: bool = True,
    roll: bool = True,
    dtype: DTypeLike | None = None,
    order: ArrayOrder = None,
) -> MyNDArray:
    """
    Broadcast x to shape.

    If x is 1d, and shape is n-d, but len(x) is same as shape[axis],
    broadcast x across all dimensions
    """
    if verify is True:
        x = np.asarray(x, dtype=dtype, order=order)
    elif not isinstance(x, np.ndarray):
        msg = f"{type(x)=} must be np.ndarray"
        raise TypeError(msg)
    x = cast("MyNDArray", x)

    # if array, and 1d with size same as shape[axis]
    # broadcast from here
    if expand and x.ndim == 1 and x.ndim != len(shape):
        if axis is None:
            msg = "trying to expand an axis with axis==None"
            raise ValueError(msg)
        if len(x) == shape[axis]:
            # reshape for broadcasting
            reshape = (1,) * (len(shape) - 1)
            reshape = shape_insert_axis(shape=reshape, axis=axis, new_size=-1)
            x = x.reshape(*reshape)

    if broadcast and x.shape != shape:
        x = np.broadcast_to(x, shape)
    if roll and axis is not None and axis != 0:
        x = np.moveaxis(x, axis, 0)
    return x


@docfiller.decorate
def validate_mom_and_mom_ndim(
    *,
    mom: Moments | None = None,
    mom_ndim: Mom_NDim | None = None,
    shape: tuple[int, ...] | None = None,
) -> tuple[MomentsStrict, Mom_NDim]:
    """
    Validate mom and mom_ndim to optional shape.

    Parameters
    ----------
    {mom}
    {mom_ndim}
    shape: target shape, optional
        This can be used to infer the ``mom`` from ``shape[-mom_ndim:]``

    Returns
    -------
    mom : tuple of int
        Moments tuple.
    mom_ndim: int
        moment ndim.

    Examples
    --------
    >>> validate_mom_and_mom_ndim(mom=1)
    ((1,), 1)
    >>> validate_mom_and_mom_ndim(mom=(2, 2))
    ((2, 2), 2)
    >>> validate_mom_and_mom_ndim(mom_ndim=1, shape=(3, 3))
    ((2,), 1)
    """
    if isinstance(mom, int):
        mom_ndim = mom_ndim or 1
        mom = (mom,) * mom_ndim  # type: ignore[assignment]

    elif mom is None:
        if mom_ndim is None or shape is None:
            msg = "Must specify either moments or mom_ndim and shape"
            raise ValueError(msg)
        if len(shape) < mom_ndim:
            raise ValueError
        mom = tuple(x - 1 for x in shape[-mom_ndim:])  # type: ignore[assignment]

    elif mom_ndim is None:
        mom = tuple(mom)  # type: ignore[assignment]
        mom_ndim = len(mom)  # type: ignore[assignment]

    # validate everything:
    if not isinstance(mom, tuple):  # pragma: no cover
        raise TypeError

    if mom_ndim not in {1, 2}:
        msg = f"{mom_ndim=} must be 1 or 2."
        raise ValueError(msg)

    if len(mom) != mom_ndim:
        msg = f"{len(mom)=} != {mom_ndim=}"
        raise ValueError(msg)

    if shape and mom != tuple(x - 1 for x in shape[-mom_ndim:]):
        msg = f"{mom=} does not conform to {shape[-mom_ndim:]}"
        raise ValueError(msg)

    return mom, mom_ndim


@docfiller.decorate
def mom_to_mom_ndim(mom: Moments) -> Mom_NDim:
    """
    Calculate mom_ndim from mom.

    Parameters
    ----------
    {mom}

    Returns
    -------
    {mom_ndim}
    """
    if isinstance(mom, int):
        mom_ndim = 1
    elif isinstance(mom, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]
        if (mom_ndim := len(mom)) not in {1, 2}:
            raise ValueError
    else:
        msg = "mom must be int or tuple"
        raise TypeError(msg)
    return cast("Mom_NDim", mom_ndim)


@docfiller.decorate
def select_mom_ndim(*, mom: Moments | None, mom_ndim: Mom_NDim | None) -> Mom_NDim:
    """
    Select a mom_ndim from mom or mom_ndim

    Parameters
    ----------
    {mom}
    {mom_ndim}

    Returns
    -------
    {mom_ndim}
    """
    if mom is not None:
        mom_ndim_calc = mom_to_mom_ndim(mom)
        if mom_ndim and mom_ndim_calc != mom_ndim:
            msg = f"{mom=} is incompatible with {mom_ndim=}"
            raise ValueError(msg)
        return mom_ndim_calc

    if mom_ndim is None:
        msg = "must specify mom_ndim or mom"
        raise TypeError(msg)

    if mom_ndim not in {1, 2}:
        msg = f"{mom_ndim=} must be 1 or 2."
        raise ValueError(msg)

    return mom_ndim

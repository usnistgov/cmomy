"""Utilities."""

from __future__ import annotations

# from functools import lru_cache
from typing import TYPE_CHECKING, cast

import numpy as np

from ._lib.utils import supports_parallel
from .docstrings import docfiller

if TYPE_CHECKING:
    from typing import Iterable, Sequence

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from ._typing_compat import TypeGuard
    from .typing import ArrayOrder, Mom_NDim, Moments, MomentsStrict, NDArrayAny
    from .typing import T_FloatDType as T_Float


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
) -> NDArrayAny:
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
    x = cast("NDArrayAny", x)

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


# * Moment validation
def is_mom_ndim(mom_ndim: int) -> TypeGuard[Mom_NDim]:
    """Validate mom_ndim."""
    return mom_ndim in {1, 2}


def is_mom_tuple(mom: tuple[int, ...]) -> TypeGuard[MomentsStrict]:
    """Validate moment tuple"""
    return len(mom) in {1, 2}


def validate_mom_ndim(mom_ndim: int) -> Mom_NDim:
    """Raise error if mom_ndim invalid."""
    if is_mom_ndim(mom_ndim):
        return mom_ndim

    msg = f"{mom_ndim=} must be either 1 or 2"
    raise ValueError(msg)


def validate_mom(mom: int | Sequence[int]) -> MomentsStrict:
    """
    Convert to MomentsStrict.

    Raise ValueError mom invalid.

    Integers to length 1 tuple
    """
    if isinstance(mom, int):
        mom = (mom,)
    elif not isinstance(mom, tuple):
        mom = tuple(mom)

    if is_mom_tuple(mom):
        return mom

    msg = f"{len(mom)=} must be either 1 or 2"
    raise ValueError(msg)


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
    if mom is not None and mom_ndim is not None:
        mom_ndim = validate_mom_ndim(mom_ndim)
        mom = validate_mom(mom)
        if len(mom) != mom_ndim:
            msg = f"{len(mom)=} != {mom_ndim=}"
            raise ValueError(msg)
        return mom, mom_ndim

    if mom is None and mom_ndim is not None and shape is not None:
        mom_ndim = validate_mom_ndim(mom_ndim)
        if len(shape) < mom_ndim:
            raise ValueError
        mom = validate_mom(tuple(x - 1 for x in shape[-mom_ndim:]))
        return mom, mom_ndim

    if mom is not None and mom_ndim is None:
        mom = validate_mom(mom)
        mom_ndim = validate_mom_ndim(len(mom))
        return mom, mom_ndim

    msg = "Must specify either mom, mom and mom_ndim, or shape and mom_ndim"
    raise ValueError(msg)


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
    mom_ndim = 1 if isinstance(mom, int) else len(mom)
    return validate_mom_ndim(mom_ndim)


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
        return validate_mom_ndim(mom_ndim_calc)

    if mom_ndim is None:
        msg = "must specify mom_ndim or mom"
        raise TypeError(msg)

    return validate_mom_ndim(mom_ndim)


# * New helpers
def parallel_heuristic(parallel: bool | None, size: int, cutoff: int = 10000) -> bool:
    """Default parallel."""
    if parallel is not None:
        return parallel and supports_parallel()
    return size > cutoff


def _prepare_secondary_value_for_reduction(
    target: NDArray[T_Float],
    x: ArrayLike,
    axis: int,
    move_axis: bool,
    *,
    order: ArrayOrder = None,
) -> NDArray[T_Float]:
    """
    Prepare value array (x1, w) for reduction.

    Here, target input base array with ``axis`` already moved
    to the last position (``target = np.moveaxis(base, axis, -1)``).
    If same number of dimensions, move axis if needed.
    If ndim == 0, broadcast to target.shape[axis]
    If ndim == 1, make sure length == target.shape[axis]
    If ndim == target.ndim, move axis to last and verify
    Otherwise, make sure x.shape == target.shape[-x.ndim:]

    Parameters
    ----------
    target : ndarray
        Target array (e.g. ``x0``).  This should already have been
        passed through `_prepare_target_value_for_reduction`

    """
    out: NDArray[T_Float] = np.asarray(x, dtype=target.dtype, order=order)
    if out.ndim == target.ndim:
        if move_axis:
            out = np.moveaxis(out, axis, -1)
            if order:
                out = np.asarray(out, order=order)
        return out

    if out.ndim == 0:
        return np.broadcast_to(out, target.shape[-1])

    if out.ndim == 1 and len(out) != target.shape[-1]:
        msg = f"For 1D secondary values, {len(out)=} must be same as target.shape[axis]={target.shape[-1]}"
        raise ValueError(msg)

    return out


def prepare_values_for_reduction(
    target: NDArray[T_Float],
    *args: ArrayLike,
    axis: int = -1,
    order: ArrayOrder = None,
    ndim: int | None = None,
) -> tuple[NDArray[T_Float], ...]:
    """Convert input value arrays to correct form for reduction."""
    ndim = ndim or target.ndim
    axis = normalize_axis_index(axis, ndim)

    move_axis = (target.ndim > 1) and (axis != ndim - 1)

    if move_axis:
        target = np.moveaxis(target, axis, -1)

    if order:
        target = np.asarray(target, order=order)

    others: Iterable[NDArray[T_Float]] = (
        _prepare_secondary_value_for_reduction(
            target=target, x=x, axis=axis, move_axis=move_axis, order=order
        )
        for x in args
    )
    return target, *others  # type: ignore[arg-type]


def prepare_data_for_reduction(
    data: NDArray[T_Float],
    axis: int,
    mom_ndim: Mom_NDim,
    order: ArrayOrder = None,
) -> NDArray[T_Float]:
    """Convert central moments array to correct form for reduction."""
    ndim = data.ndim - mom_ndim
    axis = normalize_axis_index(axis, ndim)
    last_dim = ndim - 1

    if (ndim > 1) and axis != last_dim:
        data = np.moveaxis(data, axis, last_dim)

    if order:
        data = np.asarray(data, order=order)
    return data


def raise_if_wrong_shape(
    array: NDArrayAny, shape: tuple[int, ...], name: str | None = None
) -> None:
    """Raise error if array.shape != shape"""
    if array.shape != shape:
        name = "out" if name is None else name
        msg = f"name.shape={array.shape=} != required shape {shape}"
        raise ValueError(msg)

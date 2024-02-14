"""Utilities."""
from __future__ import annotations

# from functools import lru_cache
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from typing import Sequence

    from numpy.typing import ArrayLike, DTypeLike

    from .typing import ArrayOrder, MyNDArray


def normalize_axis_index(axis: int, ndim: int) -> int:
    """Interface to numpy.core.multiarray.normalize_axis_index"""
    import numpy.core.multiarray as ma

    return ma.normalize_axis_index(axis, ndim)  # type: ignore[no-any-return,attr-defined]


def shape_insert_axis(
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


def shape_reduce(shape: tuple[int, ...], axis: int) -> tuple[int, ...]:
    """Give shape shape after reducing along axis."""
    shape_list = list(shape)
    shape_list.pop(axis)
    return tuple(shape_list)


def axis_expand_broadcast(
    x: ArrayLike,
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
        raise ValueError
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
            reshape = shape_insert_axis(reshape, axis, -1)
            x = x.reshape(*reshape)

    if broadcast and x.shape != shape:
        x = np.broadcast_to(x, shape)
    if roll and axis is not None and axis != 0:
        x = np.moveaxis(x, axis, 0)
    return x

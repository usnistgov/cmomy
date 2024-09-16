"""Utilities to work with (numpy) arrays."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, cast, overload

import numpy as np

from .validate import (
    is_dataset,
    is_ndarray,
    validate_mom_ndim,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Sequence,
    )

    import xarray as xr
    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from .typing import (
        ArrayOrder,
        ArrayOrderCF,
        AxesGUFunc,
        Mom_NDim,
        NDArrayAny,
        ScalarT,
    )


# * Array order ---------------------------------------------------------------
def arrayorder_to_arrayorder_cf(order: ArrayOrder) -> ArrayOrderCF:
    """Convert general array order to C/F/None"""
    if order is None:
        return order

    order_ = order.upper()
    if order_ in {"C", "F"}:
        return cast("ArrayOrderCF", order_)

    return None


# * Axis normalizer -----------------------------------------------------------
def _validate_ndim(ndim: int, mom_ndim: int | None) -> int:
    ndim = ndim if mom_ndim is None else ndim - validate_mom_ndim(mom_ndim)
    if ndim <= 0:
        msg = "No dimension to reduce/sample over."
        if mom_ndim:
            msg += " Trying to select moment dimensions."
        raise ValueError(msg)
    return ndim


def normalize_axis_index(
    axis: int,
    ndim: int,
    mom_ndim: Mom_NDim | None = None,
    msg_prefix: str | None = None,
) -> int:
    """Interface to numpy.core.multiarray.normalize_axis_index"""
    from .compat import (
        np_normalize_axis_index,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
    )

    ndim = _validate_ndim(ndim, mom_ndim)
    return np_normalize_axis_index(axis, ndim, msg_prefix)  # type: ignore[no-any-return,unused-ignore]


def normalize_axis_tuple(
    axis: int | Iterable[int] | None,
    ndim: int,
    mom_ndim: Mom_NDim | None = None,
    msg_prefix: str | None = None,
    allow_duplicate: bool = False,
) -> tuple[int, ...]:
    """Interface to numpy.core.multiarray.normalize_axis_index"""
    from .compat import (
        np_normalize_axis_tuple,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
    )

    ndim = _validate_ndim(ndim, mom_ndim)
    if axis is None:
        return tuple(range(ndim))
    return np_normalize_axis_tuple(axis, ndim, msg_prefix, allow_duplicate)  # type: ignore[no-any-return,unused-ignore]


def positive_to_negative_index(index: int, ndim: int) -> int:
    """
    Convert positive index to negative index

    Note that this assumes that axis has been normalized via :func:`normalize_axis_index`.

    Examples
    --------
    >>> positive_to_negative_index(0, 4)
    -4
    >>> positive_to_negative_index(-1, 4)
    -1
    >>> positive_to_negative_index(2, 4)
    -2
    """
    if index < 0:
        return index
    return index - ndim


def get_axes_from_values(*args: NDArrayAny, axis_neg: int) -> AxesGUFunc:
    """Get reduction axes for arrays..."""
    return [(-1,) if a.ndim == 1 else (axis_neg,) for a in args]


# new style preparation for reduction....
_MOM_AXES_TUPLE = {1: (-1,), 2: (-2, -1)}


def axes_data_reduction(
    *inner: int | tuple[int, ...],
    mom_ndim: Mom_NDim,
    axis: int,
    out_has_axis: bool = False,
) -> AxesGUFunc:
    """
    axes for reducing data along axis

    if ``out_has_axis == True``, then treat like resample,
    so output will still have ``axis`` with new size in output.

    It is assumed that `axis` is validated against a moments array,
    (i.e., negative values should be ``< -mom_ndim``)

    Can also pass in "inner" dimensions (elements 1:-1 of output)
    """
    mom_axes = _MOM_AXES_TUPLE[mom_ndim]
    data_axes = (axis, *mom_axes)
    out_axes = data_axes if out_has_axis else mom_axes

    return [
        data_axes,
        *((x,) if isinstance(x, int) else x for x in inner),
        out_axes,
    ]


_ALLOWED_FLOAT_DTYPES = {np.dtype(np.float32), np.dtype(np.float64)}


@overload
def select_dtype(
    x: xr.Dataset,
    *,
    out: NDArrayAny | xr.DataArray | None,
    dtype: DTypeLike,
) -> None | np.dtype[np.float32] | np.dtype[np.float64]: ...
@overload
def select_dtype(
    x: xr.DataArray | ArrayLike,
    *,
    out: NDArrayAny | xr.DataArray | None,
    dtype: DTypeLike,
) -> np.dtype[np.float32] | np.dtype[np.float64]: ...
@overload
def select_dtype(
    x: xr.Dataset | xr.DataArray | ArrayLike,
    *,
    out: NDArrayAny | xr.DataArray | None,
    dtype: DTypeLike,
) -> None | np.dtype[np.float32] | np.dtype[np.float64]: ...


def select_dtype(
    x: ArrayLike | xr.DataArray | xr.Dataset,
    *,
    out: NDArrayAny | xr.DataArray | None,
    dtype: DTypeLike,
) -> None | np.dtype[np.float32] | np.dtype[np.float64]:  # DTypeLikeArg[Any]:
    """
    Select a dtype from, in order, out, dtype, or passed array.

    If pass in a Dataset, return dtype
    """
    if is_dataset(x):
        if dtype is None:
            return dtype
        dtype = np.dtype(dtype)
    elif out is not None:
        dtype = out.dtype  # pyright: ignore[reportUnknownMemberType]
    elif dtype is not None:
        dtype = np.dtype(dtype)
    else:
        dtype = getattr(x, "dtype", np.dtype(np.float64))

    if dtype in _ALLOWED_FLOAT_DTYPES:
        return dtype  # type: ignore[return-value]

    msg = f"{dtype=} not supported.  dtype must be conformable to float32 or float64."
    raise ValueError(msg)


def asarray_maybe_recast(
    data: ArrayLike, dtype: DTypeLike = None, recast: bool = False
) -> NDArrayAny:
    """Perform asarray with optional recast to `dtype` if not already an array."""
    if is_ndarray(data):
        if recast and dtype is not None:
            return np.asarray(data, dtype=dtype)
        return data  # pyright: ignore[reportUnknownVariableType]
    return np.asarray(data, dtype=dtype)


def optional_keepdims(
    x: NDArray[ScalarT],
    *,
    axis: int | Sequence[int],
    keepdims: bool = False,
) -> NDArray[ScalarT]:
    """Optional keep dimensions."""
    if keepdims:
        return np.expand_dims(x, axis)
    return x


# * Dummy arrays --------------------------------------------------------------
def dummy_array(
    shape: int | tuple[int, ...],
    dtype: DTypeLike = None,
    broadcast: bool = True,
) -> NDArrayAny:
    """
    Create a dummy array.

    Intended for cases where array is passed to guvectorized function to
    set a shape.
    """
    # lru_cache doesn't play nice with dtype typing.
    return _dummy_array(shape, dtype, broadcast)  # type: ignore[arg-type]


@lru_cache
def _dummy_array(
    shape: int | tuple[int, ...],
    dtype: DTypeLike = None,
    broadcast: bool = True,
) -> NDArrayAny:
    if broadcast:
        if dtype is None:
            dtype = np.float64
        return np.broadcast_to(np.dtype(dtype).type(0), shape)
    return np.empty(shape, dtype=dtype)

"""Utilities to work with (numpy) arrays."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast, overload

import numpy as np

from .validate import (
    is_dataset,
    is_ndarray,
)

if TYPE_CHECKING:
    from collections.abc import (
        Sequence,
    )
    from typing import Any

    import xarray as xr
    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from .typing import (
        ArrayOrderCF,
        ArrayOrderKACF,
        AxesGUFunc,
        MomNDim,
        NDArrayAny,
    )
    from .typing_compat import TypeIs, TypeVar

    _NDArrayT = TypeVar("_NDArrayT", bound=NDArray[Any])
    _T = TypeVar("_T")
    _ScalarT = TypeVar("_ScalarT", bound=np.generic)


# * Array order ---------------------------------------------------------------
def arrayorder_to_arrayorder_cf(order: ArrayOrderKACF) -> ArrayOrderCF:
    """Convert general array order to C/F/None"""
    if order is None:
        return order

    if (order_ := order.upper()) in {"C", "F"}:
        return cast("ArrayOrderCF", order_)

    return None


# * Axis normalizer -----------------------------------------------------------
def normalize_axis_index(
    axis: complex,
    ndim: int,
    mom_ndim: MomNDim | None = None,
    msg_prefix: str | None = None,
) -> int:
    """Interface to numpy.core.multiarray.normalize_axis_index"""
    from .compat import (
        np_normalize_axis_index,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
    )

    if isinstance(axis, complex):
        axis = int(axis.imag)
        if mom_ndim is not None:
            ndim -= mom_ndim

    # normalize will catch if try to pass a float
    return np_normalize_axis_index(axis, ndim, msg_prefix)  # type: ignore[no-any-return,unused-ignore]  # pyright: ignore[reportUnknownVariableType]


def normalize_axis_tuple(
    axis: complex | Iterable[complex],
    ndim: int,
    mom_ndim: MomNDim | None = None,
    msg_prefix: str | None = None,
    allow_duplicate: bool = False,
) -> tuple[int, ...]:
    """Interface to numpy.core.multiarray.normalize_axis_index"""
    if not isinstance(axis, Iterable):
        axis = (axis,)

    out = tuple(
        normalize_axis_index(a, ndim=ndim, mom_ndim=mom_ndim, msg_prefix=msg_prefix)
        for a in axis
    )

    if not allow_duplicate and len(set(out)) != len(out):
        msg = f"Repeat axis in {out}"
        raise ValueError(msg)
    return out


@overload
def reorder(
    ndim_or_seq: int,
    source: complex | Iterable[complex],
    destination: complex | Iterable[complex],
    *,
    normalize: bool = ...,
) -> list[int]: ...
@overload
def reorder(
    ndim_or_seq: Iterable[_T],
    source: complex | Iterable[complex],
    destination: complex | Iterable[complex],
    *,
    normalize: bool = ...,
) -> list[_T]: ...


def reorder(
    ndim_or_seq: int | Iterable[_T],
    source: complex | Iterable[complex],
    destination: complex | Iterable[complex],
    *,
    normalize: bool = True,
) -> list[int] | list[_T]:
    """
    Reorder sequence.

    Using ``x.transpose(*reorder(x.ndim, source, destination))``
    is equivalent to ``np.moveaxis(x, source, destination)``.

    Useful for keeping track of where indices end up.
    To extract to new position for an axis, use ``order.index(axis)``.
    """
    seq: Sequence[Any] = (
        range(ndim_or_seq) if isinstance(ndim_or_seq, int) else list(ndim_or_seq)
    )

    if normalize:
        source = normalize_axis_tuple(source, len(seq), msg_prefix="source")
        destination = normalize_axis_tuple(
            destination, len(seq), msg_prefix="destination"
        )
    else:
        # white lie.  If normalize=False, already normalized.
        source = cast("Sequence[int]", source)
        destination = cast("Sequence[int]", destination)

    if len(source) != len(destination):
        msg = "source and destination must have same length"
        raise ValueError(msg)

    out: list[int] | list[_T] = [x for n, x in enumerate(seq) if n not in source]
    for dest, src in sorted(zip(destination, (seq[s] for s in source), strict=True)):
        out.insert(dest, src)
    return out


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


_ALLOWED_FLOAT_DTYPES = {np.dtype(np.float32), np.dtype(np.float64)}


def _is_allowed_float_dtype(
    dtype: object,
) -> TypeIs[np.dtype[np.float32] | np.dtype[np.float64]]:
    return dtype in _ALLOWED_FLOAT_DTYPES


@overload
def select_dtype(
    x: xr.Dataset,
    *,
    out: NDArrayAny | xr.DataArray | None,
    dtype: DTypeLike,
    fastpath: bool = ...,
) -> np.dtype[np.float32] | np.dtype[np.float64] | None: ...
@overload
def select_dtype(
    x: xr.DataArray | ArrayLike,
    *,
    out: NDArrayAny | xr.DataArray | None,
    dtype: DTypeLike,
    fastpath: bool = ...,
) -> np.dtype[np.float32] | np.dtype[np.float64]: ...
@overload
def select_dtype(
    x: xr.Dataset | xr.DataArray,
    *,
    out: NDArrayAny | xr.DataArray | None,
    dtype: DTypeLike,
    fastpath: bool = ...,
) -> np.dtype[np.float32] | np.dtype[np.float64] | None: ...
@overload
def select_dtype(
    x: xr.Dataset | xr.DataArray | ArrayLike,
    *,
    out: NDArrayAny | xr.DataArray | None,
    dtype: DTypeLike,
    fastpath: bool = ...,
) -> np.dtype[np.float32] | np.dtype[np.float64] | None: ...


def select_dtype(
    x: ArrayLike | xr.DataArray | xr.Dataset,
    *,
    out: NDArrayAny | xr.DataArray | None,
    dtype: DTypeLike,
    fastpath: bool = False,
) -> np.dtype[np.float32] | np.dtype[np.float64] | None:  # DTypeLikeArg[Any]:
    """
    Select a dtype from, in order, out, dtype, or passed array.

    If pass in a Dataset, return dtype
    """
    if fastpath:
        assert dtype in _ALLOWED_FLOAT_DTYPES  # noqa: S101
        return dtype  # type: ignore[return-value] # pyright: ignore[reportReturnType]

    if is_dataset(x):
        if dtype is None:
            return dtype
        dtype = np.dtype(dtype)
    elif out is not None:
        dtype = out.dtype
    elif dtype is not None:
        dtype = np.dtype(dtype)
    else:
        dtype = getattr(x, "dtype", np.dtype(np.float64))

    if _is_allowed_float_dtype(dtype):
        return dtype

    msg = f"{dtype=} not supported.  dtype must be conformable to float32 or float64."
    raise ValueError(msg)


def asarray_maybe_recast(
    data: ArrayLike, dtype: DTypeLike = None, recast: bool = False
) -> NDArrayAny:
    """Perform asarray with optional recast to `dtype` if not already an array."""
    if is_ndarray(data):
        if recast and dtype is not None:
            return np.asarray(data, dtype=dtype)
        return data
    return np.asarray(data, dtype=dtype)


def optional_keepdims(
    x: NDArray[_ScalarT],
    *,
    axis: int | Sequence[int],
    keepdims: bool = False,
) -> NDArray[_ScalarT]:
    """Optional keep dimensions."""
    if keepdims:
        return np.expand_dims(x, axis)
    return x

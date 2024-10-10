"""Validators"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import xarray as xr

from .docstrings import docfiller
from .missing import MISSING

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Sequence,
    )
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from .typing import (
        DataT,
        MissingType,
        MomAxesStrict,
        MomDimsStrict,
        MomentsStrict,
        MomNDim,
    )
    from .typing_compat import TypeIs, TypeVar

    T = TypeVar("T")


# * TypeGuards ----------------------------------------------------------------
def is_ndarray(x: Any) -> TypeIs[NDArray[Any]]:
    """Typeguard ndarray."""
    return isinstance(x, np.ndarray)


def is_dataarray(x: object) -> TypeIs[xr.DataArray]:
    """Typeguard dataarray."""
    return isinstance(x, xr.DataArray)


def is_dataset(x: object) -> TypeIs[xr.Dataset]:
    """Typeguard dataset"""
    return isinstance(x, xr.Dataset)


def is_xarray(x: object) -> TypeIs[xr.Dataset | xr.DataArray]:
    """Typeguard xarray object"""
    return isinstance(x, (xr.DataArray, xr.Dataset))


def is_xarray_typevar(x: ArrayLike | DataT) -> TypeIs[DataT]:
    """Typeguard ``DataT`` typevar against array-like."""
    return isinstance(x, (xr.DataArray, xr.Dataset))


# * Raises --------------------------------------------------------------------
def raise_if_wrong_value(value: T, expected: T, message: str | None = None) -> None:
    """Raise error if value != expected_value"""
    if value != expected:
        message = message or "Wrong value."
        msg = f"{message} Passed {value}. Expected {expected}."
        raise ValueError(msg)


# * Moment validation ---------------------------------------------------------
def is_mom_ndim(mom_ndim: int | None) -> TypeIs[MomNDim]:
    """Validate mom_ndim."""
    return mom_ndim in {1, 2}


def is_mom_tuple(mom: tuple[int, ...]) -> TypeIs[MomentsStrict]:
    """Validate moment tuple"""
    return len(mom) in {1, 2} and all(m > 0 for m in mom)


def validate_mom_ndim(
    mom_ndim: int | None, mom_ndim_default: MomNDim | None = None
) -> MomNDim:
    """Raise error if mom_ndim invalid."""
    if is_mom_ndim(mom_ndim):
        return mom_ndim

    if mom_ndim is None and mom_ndim_default is not None:
        return mom_ndim_default

    msg = f"{mom_ndim=} must be either 1 or 2"
    raise ValueError(msg)


def validate_mom(mom: int | Iterable[int]) -> MomentsStrict:
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

    msg = f"{mom=} must be an integer, or tuple of length 1 or 2, with positive values."
    raise ValueError(msg)


def validate_mom_axes(mom_axes: int | Iterable[int]) -> MomAxesStrict:
    """Validate mom_axes"""
    if isinstance(mom_axes, int):
        return (mom_axes,)

    if not isinstance(mom_axes, tuple):
        mom_axes = tuple(mom_axes)

    if len(mom_axes) in {1, 2}:
        return cast("MomAxesStrict", mom_axes)

    msg = f"{mom_axes=} must be an integer, or tuple of integers of length 1 or 2."
    raise ValueError(msg)


# * Validate type -------------------------------------------------------------
@docfiller.decorate
def validate_floating_dtype(
    dtype: DTypeLike,
    name: Hashable = "array",
) -> None | np.dtype[np.float32] | np.dtype[np.float64]:
    """
    Validate that dtype is conformable float32 or float64.

    Parameters
    ----------
    {dtype}

    Returns
    -------
    `numpy.dtype` object or None
        Note that if ``dtype == None``, ``None`` will be returned.
        Otherwise, will return ``np.dtype(dtype)``.


    """
    if dtype is None:
        # defaults to np.float64, but can have special properties
        # e.g., to np.asarray(..., dtype=None) means infer...
        return dtype

    dtype = np.dtype(dtype)
    if dtype.type in {np.float32, np.float64}:
        return dtype  # type: ignore[return-value]

    msg = f"{dtype=} not supported for {name}.  dtype must be conformable to float32 or float64."
    raise ValueError(msg)


def validate_not_none(x: T | None, name: str = "value") -> T:
    """
    Raise if value is None

    Use for now to catch any cases where `axis=None` or `dim=None`.
    These values are reserved for total reduction...
    """
    if x is None:
        msg = f"{name}={x} is not supported"
        raise TypeError(msg)
    return x


def validate_mom_ndim_and_mom_axes(
    mom_ndim: int | None,
    mom_axes: int | Sequence[int] | None = None,
    mom_ndim_default: MomNDim | None = None,
) -> tuple[MomNDim, MomAxesStrict]:
    """Validate ``mom_ndim`` and ``mom_axes``."""
    if mom_axes is None:
        mom_ndim = validate_mom_ndim(mom_ndim, mom_ndim_default)
        mom_axes = cast("MomAxesStrict", tuple(range(-mom_ndim, 0)))
        return mom_ndim, mom_axes

    mom_axes = validate_mom_axes(mom_axes)
    if mom_ndim is None:
        mom_ndim = cast("MomNDim", len(mom_axes))
    elif len(mom_axes) != mom_ndim:
        msg = f"{len(mom_axes)=} != {mom_ndim=}"
        raise ValueError(msg)
    return cast("MomNDim", mom_ndim), mom_axes


# * Validate Axis -------------------------------------------------------------
def validate_axis(axis: T | None | MissingType) -> T:
    """
    Validate that axis is an integer

    In the future, will allow axis to be None also.
    """
    if axis is None or axis is MISSING:
        msg = f"Must specify axis. Received {axis=}."
        raise TypeError(msg)
    return axis


def validate_axis_mult(
    axis: T | tuple[T, ...] | None | MissingType,
) -> T | tuple[T, ...] | None:
    """Validate that axis is specified."""
    if axis is MISSING:
        msg = f"Must specify axis. Received {axis=}."
        raise TypeError(msg)
    return axis


# * DataArray -----------------------------------------------------------------
def validate_mom_dims(
    mom_dims: Hashable | Sequence[Hashable] | None,
    mom_ndim: MomNDim,
    out: object = None,
    mom_axes: MomAxesStrict | None = None,
) -> MomDimsStrict:
    """Validate mom_dims to correct form."""
    if mom_dims is None:
        if is_dataset(out):
            # select first array in dataset
            out = out[next(iter(out))]

        if is_dataarray(out):
            _axes = range(-mom_ndim, 0) if mom_axes is None else mom_axes
            return cast("MomDimsStrict", tuple(out.dims[a] for a in _axes))

        if mom_ndim == 1:
            return ("mom_0",)
        return ("mom_0", "mom_1")

    validated: tuple[Hashable, ...]
    if isinstance(mom_dims, str):
        validated = (mom_dims,)
    elif isinstance(mom_dims, (tuple, list)):
        validated = tuple(mom_dims)  # pyright: ignore[reportUnknownArgumentType]
    else:
        msg = f"Unknown {type(mom_dims)=}.  Expected str or Sequence[str]"
        raise TypeError(msg)

    if len(validated) != mom_ndim:
        msg = f"mom_dims={validated} inconsistent with {mom_ndim=}"
        raise ValueError(msg)
    return cast("MomDimsStrict", validated)


def validate_mom_dims_and_mom_ndim(
    mom_dims: Hashable | Sequence[Hashable] | None,
    mom_ndim: int | None,
    out: object = None,
    mom_ndim_default: int | None = None,
    mom_axes: int | Sequence[int] | None = None,
) -> tuple[MomDimsStrict, MomNDim]:
    """Validate mom_dims and mom_ndim."""
    if mom_ndim is not None or mom_axes is not None:
        mom_ndim, mom_axes = validate_mom_ndim_and_mom_axes(mom_ndim, mom_axes)
        mom_dims = validate_mom_dims(mom_dims, mom_ndim, out, mom_axes=mom_axes)
        return mom_dims, mom_ndim

    if mom_dims is not None:
        mom_dims = cast(
            "MomDimsStrict",
            (mom_dims,) if isinstance(mom_dims, str) else tuple(mom_dims),  # type: ignore[arg-type]
        )
        mom_ndim = validate_mom_ndim(len(mom_dims), mom_axes)
        return mom_dims, mom_ndim

    if mom_ndim_default is not None:
        return validate_mom_dims_and_mom_ndim(mom_dims, mom_ndim_default, out)

    msg = "Must specify at least one of mom_dims or mom_ndim"
    raise ValueError(msg)

"""Utilities."""

from __future__ import annotations

import enum
from itertools import chain
from typing import TYPE_CHECKING, cast

import numpy as np
import xarray as xr

from ._lib.utils import supports_parallel as supports_parallel  # noqa: PLC0414
from .docstrings import docfiller

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from ._typing_compat import TypeGuard, TypeVar

    # from .typing import FloatT
    # from .typing import T_FloatDType_co as T_Float_co
    from .typing import (
        ArrayOrder,
        ArrayOrderCF,
        DTypeLikeArg,
        MissingType,
        Mom_NDim,
        MomDims,
        MomDimsStrict,
        Moments,
        MomentsStrict,
        NDArrayAny,
        ScalarT,
    )

    T = TypeVar("T")


# * Missing/no values ---------------------------------------------------------
# taken from https://github.com/python-attrs/attrs/blob/main/src/attr/_make.py
class _Missing(enum.Enum):
    """
    Sentinel to indicate the lack of a value when ``None`` is ambiguous.

    Use `cmomy.typing.MissingType` to type this value.
    """

    MISSING = enum.auto()

    def __repr__(self) -> str:
        return "MISSING"  # pragma: no cover

    def __bool__(self) -> bool:
        return False  # pragma: no cover


MISSING = _Missing.MISSING
"""
Sentinel to indicate the lack of a value when ``None`` is ambiguous.
"""


# * Axis normalizer
def normalize_axis_index(axis: int, ndim: int) -> int:
    """Interface to numpy.core.multiarray.normalize_axis_index"""
    from ._compat import (
        np_normalize_axis_index,  # pyright: ignore[reportAttributeAccessIssue]
    )

    return np_normalize_axis_index(axis, ndim)  # type: ignore[no-any-return,unused-ignore]


# * Array order ---------------------------------------------------------------
def arrayorder_to_arrayorder_cf(order: ArrayOrder) -> ArrayOrderCF:
    """Convert general array order to C/F/None"""
    if order is None:
        return order

    order_ = order.upper()
    if order_ in {"C", "F"}:
        return cast("ArrayOrderCF", order_)

    return None


# * peek at iterable ----------------------------------------------------------
def peek_at(iterable: Iterable[T]) -> tuple[T, Iterator[T]]:
    """Returns the first value from iterable, as well as a new iterator with
    the same content as the original iterable
    """
    gen = iter(iterable)
    peek = next(gen)
    return peek, chain([peek], gen)


# * Moment validation ---------------------------------------------------------
def is_mom_ndim(mom_ndim: int) -> TypeGuard[Mom_NDim]:
    """Validate mom_ndim."""
    return mom_ndim in {1, 2}


def is_mom_tuple(mom: tuple[int, ...]) -> TypeGuard[MomentsStrict]:
    """Validate moment tuple"""
    return len(mom) in {1, 2} and all(m > 0 for m in mom)


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

    msg = f"{mom=} must be an integer, or tuple of length 1 or 2, with positive values."
    raise ValueError(msg)


@docfiller.decorate
def validate_mom_and_mom_ndim(
    *,
    mom: Moments | None = None,
    mom_ndim: int | None = None,
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
def select_mom_ndim(*, mom: Moments | None, mom_ndim: int | None) -> Mom_NDim:
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


# * New helpers ---------------------------------------------------------------
@docfiller.decorate
def validate_floating_dtype(
    dtype: DTypeLike,
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

    msg = f"{dtype=} not supported.  dtype must be conformable to float32 or float64."
    raise ValueError(msg)


def parallel_heuristic(parallel: bool | None, size: int, cutoff: int = 10000) -> bool:
    """Default parallel."""
    if parallel is not None:
        return parallel and supports_parallel()
    return size > cutoff


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


def validate_axis(axis: int | None | MissingType) -> int:
    """
    Validate that axis is an integer.

    In the future, will allow axis to be None also.
    """
    if axis is None or axis is MISSING:
        msg = f"Must specify axis as integer.  Received {axis=}."
        raise TypeError(msg)
    return axis


def select_axis_dim(
    dims: tuple[Hashable, ...],
    axis: int | None | MissingType = MISSING,
    dim: Hashable | None | MissingType = MISSING,
    default_axis: int | None | MissingType = MISSING,
    default_dim: Hashable | None | MissingType = MISSING,
) -> tuple[int, Hashable]:
    """Produce axis/dim from input."""
    # for now, disallow None values
    axis = validate_not_none(axis, "axis")
    dim = validate_not_none(dim, "dim")
    default_axis = validate_not_none(default_axis, "default_axis")
    default_dim = validate_not_none(default_dim, "default_dim")

    if axis is MISSING and dim is MISSING:
        if default_axis is not MISSING and default_dim is MISSING:
            axis = default_axis
        elif default_axis is MISSING and default_dim is not MISSING:
            dim = default_dim
        else:
            msg = "Must specify axis or dim, or one of default_axis or default_dim"
            raise ValueError(msg)

    elif axis is not MISSING and dim is not MISSING:
        msg = "Can only specify one of axis or dim"
        raise ValueError(msg)

    if dim is not MISSING:
        axis = dims.index(dim)
    elif axis is not MISSING:
        if isinstance(axis, str):
            msg = f"Using string value for axis is deprecated.  Please use `dim` option instead.  Passed {axis} of type {type(axis)}"
            raise ValueError(msg)
        dim = dims[axis]  # type: ignore[index]
    else:  # pragma: no cover
        msg = f"Unknown dim {dim} and axis {axis}"
        raise TypeError(msg)

    return axis, dim  # type: ignore[return-value]


# ** Prepare for push/reduction
def _prepare_secondary_value_for_reduction(
    target: NDArray[ScalarT],
    x: ArrayLike,
    axis: int,
    move_axis: bool,
    nsamp: int,
    *,
    order: ArrayOrder = None,
) -> NDArray[ScalarT]:
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
    out: NDArray[ScalarT] = np.asarray(x, dtype=target.dtype, order=order)
    if out.ndim == target.ndim:
        if move_axis:
            out = np.moveaxis(out, axis, -1)
            if order:
                out = np.asarray(out, order=order)
        return out

    if out.ndim == 0:
        out = np.broadcast_to(out, nsamp)
        if order:
            out = np.asarray(out, order=order)
        return out

    if out.ndim == 1 and len(out) != nsamp:
        msg = f"For 1D secondary values, {len(out)=} must be same as target.shape[axis]={nsamp}"
        raise ValueError(msg)

    # At least check that last dimension has correct shape.
    if out.shape[-1] != target.shape[-1]:
        msg = f"{out.shape=} will not broadcast to {target.shape=}"
        raise ValueError(msg)

    return out


def _xprepare_secondary_value_for_reduction(
    target: xr.DataArray,
    x: xr.DataArray,
    *,
    order: ArrayOrder = None,
) -> xr.DataArray:
    # assume ordering will be handled by apply_ufunc
    if order or x.dtype != target.dtype:  # pyright: ignore[reportUnknownMemberType]
        x = x.astype(dtype=target.dtype, copy=False, order=order)  # pyright: ignore[reportUnknownMemberType]
    return x


def prepare_values_for_reduction(
    target: ArrayLike,
    *args: ArrayLike,
    narrays: int,
    axis: int | None | MissingType = MISSING,
    dtype: DTypeLikeArg[ScalarT],
    order: ArrayOrder = None,
) -> tuple[NDArray[ScalarT], ...]:
    """
    Convert input value arrays to correct form for reduction.

    Parameters
    ----------
        narrays : int
        The total number of expected arrays.  len(args) + 1 must equal narrays.
    """
    if len(args) + 1 != narrays:
        msg = f"Number of arrays {len(args) + 1} != {narrays}"
        raise ValueError(msg)

    axis = validate_axis(axis)

    target = np.asarray(target, dtype=dtype)
    axis = normalize_axis_index(axis, target.ndim)
    move_axis = (target.ndim > 1) and (axis != target.ndim - 1)

    if move_axis:
        target = np.moveaxis(target, axis, -1)
    if order:
        target = np.asarray(target, order=order)

    others: Iterable[NDArray[ScalarT]] = (
        _prepare_secondary_value_for_reduction(
            target=target,
            x=x,
            axis=axis,
            move_axis=move_axis,
            order=order,
            nsamp=target.shape[-1],
        )
        for x in args
    )
    return target, *others


def xprepare_values_for_reduction(
    target: xr.DataArray,
    *args: ArrayLike | xr.DataArray,
    narrays: int,
    dim: Hashable | None | MissingType,
    axis: int | None | MissingType,
    dtype: DTypeLike,
    order: ArrayOrder = None,
) -> tuple[list[list[Hashable]], tuple[xr.DataArray | NDArrayAny, ...]]:
    """
    Convert input value arrays to correct form for reduction.

    Parameters
    ----------
        narrays : int
        The total number of expected arrays.  len(args) + 1 must equal narrays.

    Returns
    -------
    input_core_dims : list[list[Hashable]]
    tuple_of_arrays : tuple of DataArray or ndarray
    """
    if len(args) + 1 != narrays:
        msg = f"Number of arrays {len(args) + 1} != {narrays}"
        raise ValueError(msg)

    if not isinstance(target, xr.DataArray):  # pyright: ignore[reportUnnecessaryIsInstance]
        msg = "First value must be DataArray.  Received type {type(target)}"
        raise TypeError(msg)

    axis, dim = select_axis_dim(
        dims=target.dims,
        axis=axis,
        dim=dim,
    )

    move_axis = (target.ndim > 1) and (axis != target.ndim - 1)

    # go ahead and do the move in case we want to order...
    if move_axis:
        target = target.transpose(..., dim)  # pyright: ignore[reportUnknownArgumentType]
    target = target.astype(dtype=dtype, order=order, copy=False)  # pyright: ignore[reportUnknownMemberType]

    nsamp = target.shape[-1]

    # nsamp = target.shape[axis]

    others: list[NDArrayAny | xr.DataArray] = []
    for x in args:
        if isinstance(x, xr.DataArray):
            others.append(
                _xprepare_secondary_value_for_reduction(
                    target=target,
                    x=x,
                    order=order,  # axis=axis, dim=dim, move_axis=move_axis, order=order
                )
            )
        else:
            others.append(
                _prepare_secondary_value_for_reduction(  # pyright: ignore[reportUnknownArgumentType]
                    target=target.values,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                    x=x,
                    axis=axis,
                    move_axis=move_axis,
                    order=order,
                    nsamp=nsamp,
                )
            )

    input_core_dims = [[dim]] * (len(others) + 1)
    return input_core_dims, (target, *others)


def prepare_data_for_reduction(
    data: ArrayLike,
    axis: int | None | MissingType,
    mom_ndim: Mom_NDim,
    dtype: DTypeLikeArg[ScalarT],
    order: ArrayOrder = None,
) -> NDArray[ScalarT]:
    """Convert central moments array to correct form for reduction."""
    data = np.asarray(data, dtype=dtype)
    axis = validate_axis(axis)

    ndim = data.ndim - mom_ndim
    axis = normalize_axis_index(axis, ndim)
    last_dim = ndim - 1

    if (ndim > 1) and axis != last_dim:
        data = np.moveaxis(data, axis, last_dim)
    if order:
        data = np.asarray(data, order=order)
    return data


def xprepare_data_for_reduction(
    data: xr.DataArray,
    axis: int | None | MissingType,
    dim: Hashable | None | MissingType,
    mom_ndim: Mom_NDim,
    order: ArrayOrder = None,
    dtype: DTypeLike = None,
) -> tuple[Hashable, xr.DataArray]:
    """Prepare DataArray for reduction."""
    ndim = data.ndim - mom_ndim
    if isinstance(axis, int):
        axis = normalize_axis_index(axis, ndim)
    axis, dim = select_axis_dim(dims=data.dims, axis=axis, dim=dim)

    if axis >= ndim:
        msg = f"Cannot reduce over moment dimension.  {axis=}, {dim=}."
        raise ValueError(msg)

    last_dim = ndim - 1
    if (ndim > 1) and axis != last_dim:
        data = data.transpose(..., dim, *data.dims[-mom_ndim:])  # pyright: ignore[reportUnknownArgumentType]
    if order or dtype:
        data = data.astype(dtype or data.dtype, order=order, copy=False)  # pyright: ignore[reportUnknownMemberType]

    return dim, data


def prepare_values_for_push_val(
    target: ArrayLike,
    *args: ArrayLike,
    dtype: DTypeLikeArg[ScalarT],
    order: ArrayOrder | None = None,
) -> tuple[NDArray[ScalarT], ...]:
    """Get values ready for push"""
    target = np.asarray(target, order=order, dtype=dtype)
    others: Iterable[NDArray[ScalarT]] = (
        np.asarray(x, dtype=target.dtype, order=order) for x in args
    )
    return target, *others


def raise_if_wrong_shape(
    array: NDArrayAny, shape: tuple[int, ...], name: str | None = None
) -> None:
    """Raise error if array.shape != shape"""
    if array.shape != shape:
        name = "out" if name is None else name
        msg = f"name.shape={array.shape=} != required shape {shape}"
        raise ValueError(msg)


_ALLOWED_FLOAT_DTYPES = {np.dtype(np.float32), np.dtype(np.float64)}


def select_dtype(
    x: xr.DataArray | ArrayLike,
    out: NDArrayAny | None,
    dtype: DTypeLike,
) -> np.dtype[np.float32] | np.dtype[np.float64]:  # DTypeLikeArg[Any]:
    """Select a dtype from, in order, out, dtype, or passed array."""
    if out is not None:
        dtype = out.dtype
    elif dtype is not None:
        dtype = np.dtype(dtype)
    else:
        dtype = getattr(x, "dtype", np.dtype(np.float64))

    if dtype in _ALLOWED_FLOAT_DTYPES:
        return dtype  # type: ignore[return-value]

    msg = f"{dtype=} not supported.  dtype must be conformable to float32 or float64."
    raise ValueError(msg)


# * Xarray utilities ----------------------------------------------------------
def validate_mom_dims(
    mom_dims: Hashable | Sequence[Hashable] | None, mom_ndim: Mom_NDim
) -> MomDimsStrict:
    """Validate mom_dims to correct form."""
    if mom_dims is None:
        if mom_ndim == 1:
            return ("mom_0",)
        return ("mom_0", "mom_1")

    out: tuple[Hashable, ...]
    if isinstance(mom_dims, str):
        out = (mom_dims,)
    elif isinstance(mom_dims, (tuple, list)):
        out = tuple(mom_dims)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
    else:
        msg = f"Unknown {type(mom_dims)=}.  Expected str or Sequence[str]"
        raise TypeError(msg)

    if len(out) != mom_ndim:  # pyright: ignore[reportUnknownArgumentType]
        msg = f"mom_ndim={out} inconsistent with {mom_ndim=}"
        raise ValueError(msg)
    return cast("MomDimsStrict", out)


def move_mom_dims_to_end(
    x: xr.DataArray, mom_dims: MomDims, mom_ndim: Mom_NDim | None = None
) -> xr.DataArray:
    """Move moment dimensions to end"""
    if mom_dims is not None:
        mom_dims = (mom_dims,) if isinstance(mom_dims, str) else tuple(mom_dims)  # type: ignore[arg-type]

        if mom_ndim is not None and len(mom_dims) != mom_ndim:
            msg = f"len(mom_dims)={len(mom_dims)} not equal to mom_ndim={mom_ndim}"
            raise ValueError(msg)

        x = x.transpose(..., *mom_dims)  # pyright: ignore[reportUnknownArgumentType]

    return x


def replace_coords_from_isel(
    da_original: xr.DataArray,
    da_selected: xr.DataArray,
    indexers: Mapping[Any, Any] | None = None,
    drop: bool = False,
    **indexers_kwargs: Any,
) -> xr.DataArray:
    """
    Replace coords in da_selected with coords from coords from da_original.isel(...).

    This assumes that `da_selected` is the result of soe operation, and that indexeding
    ``da_original`` will give the correct coordinates/indexed.

    Useful for adding back coordinates to reduced object.
    """
    from xarray.core.indexes import (
        isel_indexes,  # pyright: ignore[reportUnknownVariableType]
    )
    from xarray.core.indexing import is_fancy_indexer

    # Would prefer to import from actual source by old xarray error.
    from xarray.core.utils import either_dict_or_kwargs  # type: ignore[attr-defined]

    indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")
    if any(is_fancy_indexer(idx) for idx in indexers.values()):  # pragma: no cover
        msg = "no fancy indexers for this"
        raise ValueError(msg)

    indexes, index_variables = isel_indexes(da_original.xindexes, indexers)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

    coords = {}
    for coord_name, coord_value in da_original._coords.items():  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        if coord_name in index_variables:
            coord_value = index_variables[coord_name]  # noqa: PLW2901
        else:
            coord_indexers = {
                k: v for k, v in indexers.items() if k in coord_value.dims
            }
            if coord_indexers:
                coord_value = coord_value.isel(coord_indexers)  # noqa: PLW2901
                if drop and coord_value.ndim == 0:
                    continue
        coords[coord_name] = coord_value

    return da_selected._replace(coords=coords, indexes=indexes)  # pyright: ignore[reportUnknownMemberType, reportPrivateUsage]

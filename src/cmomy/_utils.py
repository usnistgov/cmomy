"""Utilities."""

from __future__ import annotations

import enum
from itertools import chain
from typing import TYPE_CHECKING, cast, overload

import numpy as np
import xarray as xr

from ._lib.utils import supports_parallel as supports_parallel  # noqa: PLC0414
from .docstrings import docfiller

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from ._typing_compat import TypeGuard, TypeVar
    from .typing import (
        ArrayOrder,
        ArrayOrderCF,
        AxesGUFunc,
        AxisReduce,
        AxisReduceMult,
        DimsReduce,
        DimsReduceMult,
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
def normalize_axis_index(
    axis: int,
    ndim: int,
    mom_ndim: Mom_NDim | None = None,
    msg_prefix: str | None = None,
) -> int:
    """Interface to numpy.core.multiarray.normalize_axis_index"""
    from ._compat import (
        np_normalize_axis_index,  # pyright: ignore[reportAttributeAccessIssue]
    )

    ndim = ndim if mom_ndim is None else ndim - validate_mom_ndim(mom_ndim)
    return np_normalize_axis_index(axis, ndim, msg_prefix)  # type: ignore[no-any-return,unused-ignore]


def normalize_axis_tuple(
    axis: int | Iterable[int] | None,
    ndim: int,
    mom_ndim: Mom_NDim | None = None,
    msg_prefix: str | None = None,
    allow_duplicate: bool = False,
) -> tuple[int, ...]:
    """Interface to numpy.core.multiarray.normalize_axis_index"""
    from ._compat import (
        np_normalize_axis_tuple,  # pyright: ignore[reportAttributeAccessIssue]
    )

    ndim = ndim if mom_ndim is None else ndim - validate_mom_ndim(mom_ndim)

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


@docfiller.decorate
def mom_to_mom_shape(mom: int | Iterable[int]) -> MomentsStrict:
    """Convert moments to moments shape."""
    return cast("MomentsStrict", tuple(m + 1 for m in validate_mom(mom)))


@docfiller.decorate
def mom_shape_to_mom(mom_shape: int | Iterable[int]) -> MomentsStrict:
    """Convert moments shape to moments"""
    return cast("MomentsStrict", tuple(m - 1 for m in validate_mom(mom_shape)))


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


def validate_axis(axis: AxisReduce | MissingType) -> int:
    """
    Validate that axis is an integer.

    In the future, will allow axis to be None also.
    """
    if axis is None or axis is MISSING:
        msg = f"Must specify axis. Received {axis=}."
        raise TypeError(msg)
    return axis


def validate_axis_mult(axis: AxisReduceMult | MissingType) -> AxisReduceMult:
    """Validate that axis is specified."""
    if axis is MISSING:
        msg = f"Must specify axis. Received {axis=}."
        raise TypeError(msg)
    return axis


def select_axis_dim(
    data: xr.DataArray,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    default_axis: AxisReduce | MissingType = MISSING,
    default_dim: DimsReduce | MissingType = MISSING,
    mom_ndim: Mom_NDim | None = None,
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
        axis = data.get_axis_num(dim)
        if axis >= data.ndim - (0 if mom_ndim is None else mom_ndim):
            msg = f"Cannot select moment dimension. {axis=}, {dim=}."
            raise ValueError(msg)

    elif axis is not MISSING:
        if isinstance(axis, str):
            msg = f"Using string value for axis is deprecated.  Please use `dim` option instead.  Passed {axis} of type {type(axis)}"
            raise ValueError(msg)
        # wrap axis
        axis = normalize_axis_index(axis, data.ndim, mom_ndim)  # type: ignore[arg-type]
        dim = data.dims[axis]
    else:  # pragma: no cover
        msg = f"Unknown dim {dim} and axis {axis}"
        raise TypeError(msg)

    return axis, dim


def select_axis_dim_mult(
    data: xr.DataArray,
    axis: AxisReduceMult | MissingType = MISSING,
    dim: DimsReduceMult | MissingType = MISSING,
    default_axis: AxisReduceMult | MissingType = MISSING,
    default_dim: DimsReduceMult | MissingType = MISSING,
    mom_ndim: Mom_NDim | None = None,
) -> tuple[tuple[int, ...], tuple[Hashable, ...]]:
    """
    Produce axis/dim tuples from input.

    This is like `select_axis_dim`, but allows multiple values in axis/dim.
    """
    # Allow None, which implies choosing all dimensions...
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

    ndim = data.ndim - (0 if mom_ndim is None else mom_ndim)
    dim_: tuple[Hashable, ...]
    axis_: tuple[int, ...]

    if dim is not MISSING:
        dim_ = (
            data.dims[:ndim]
            if dim is None
            else (dim,)
            if isinstance(dim, str)
            else tuple(dim)  # type: ignore[arg-type]
        )

        axis_ = data.get_axis_num(dim_)
        if any(a >= ndim for a in axis_):
            msg = f"Cannot select moment dimension. {axis_=}, {dim_=}."
            raise ValueError(msg)

    elif axis is not MISSING:
        axis_ = normalize_axis_tuple(axis, data.ndim, mom_ndim)
        dim_ = tuple(data.dims[a] for a in axis_)

    else:  # pragma: no cover
        msg = f"Unknown dim {dim} and axis {axis}"
        raise TypeError(msg)

    return axis_, dim_


@overload
def moveaxis(
    x: NDArray[ScalarT],
    axis: int | tuple[int, ...] | MissingType = ...,
    dest: int | tuple[int, ...] | MissingType = ...,
    *,
    dim: str | Sequence[Hashable] | MissingType = ...,
    dest_dim: str | Sequence[Hashable] | MissingType = ...,
    mom_ndim: Mom_NDim | None = ...,
) -> NDArray[ScalarT]: ...
@overload
def moveaxis(
    x: xr.DataArray,
    axis: int | tuple[int, ...] | MissingType = ...,
    dest: int | tuple[int, ...] | MissingType = ...,
    *,
    dim: str | Sequence[Hashable] | MissingType = ...,
    dest_dim: str | Sequence[Hashable] | MissingType = ...,
    mom_ndim: Mom_NDim | None = ...,
) -> xr.DataArray: ...


@docfiller.decorate
def moveaxis(
    x: NDArray[ScalarT] | xr.DataArray,
    axis: int | tuple[int, ...] | MissingType = MISSING,
    dest: int | tuple[int, ...] | MissingType = MISSING,
    *,
    dim: str | Sequence[Hashable] | MissingType = MISSING,
    dest_dim: str | Sequence[Hashable] | MissingType = MISSING,
    mom_ndim: Mom_NDim | None = None,
) -> NDArray[ScalarT] | xr.DataArray:
    """
    Generalized moveaxis for moments arrays.

    Parameters
    ----------
    x : ndarray or DataArray
        input data
    axis : int or sequence of int
        Original positions of axes to move.
    dest : int or sequence of int
        Destination positions for each original axes.
    dim : str or sequence of hashable
        Original dimensions to move (for DataArray).
    dest_dim : str or sequence of hashable
        Destination of each original dimension.
    {mom_ndim_optional}

    Returns
    -------
    out : ndarray or DataArray
        Same type as ``x`` with moved axis.

    Notes
    -----
    Must specify either ``axis`` or ``dim`` and either ``dest`` or
    ``dest_dim``.

    See Also
    --------
    numpy.moveaxis

    Examples
    --------
    >>> x = np.zeros((2, 3, 4, 5))
    >>> moveaxis(x, 0, -1).shape
    (3, 4, 5, 2)

    Specifying ``mom_ndim`` will result in negative axis relative
    to the moments dimensions.

    >>> moveaxis(x, 0, -1, mom_ndim=1).shape
    (3, 4, 2, 5)

    Multiple axes can also be specified.

    >>> moveaxis(x, (1, 0), (-2, -1), mom_ndim=1).shape
    (4, 3, 2, 5)

    This also works with dataarrays

    >>> dx = xr.DataArray(x, dims=["a", "b", "c", "mom_0"])
    >>> moveaxis(dx, dim="a", dest=-1, mom_ndim=1).dims
    ('b', 'c', 'a', 'mom_0')
    """
    mom_ndim = None if mom_ndim is None else validate_mom_ndim(mom_ndim)

    if isinstance(x, xr.DataArray):
        axes0, dims0 = select_axis_dim_mult(x, axis=axis, dim=dim, mom_ndim=mom_ndim)
        axes1, dims1 = select_axis_dim_mult(
            x, axis=dest, dim=dest_dim, mom_ndim=mom_ndim
        )

        if len(dims0) != len(dims1):
            msg = "`dim` and `dest_dim` must have same length"
            raise ValueError(msg)

        order = [n for n in range(x.ndim) if n not in axes0]
        for dst, src in sorted(zip(axes1, axes0)):
            order.insert(dst, src)
        return x.transpose(*(x.dims[o] for o in order))

    axes0 = normalize_axis_tuple(validate_axis_mult(axis), x.ndim, mom_ndim)
    axes1 = normalize_axis_tuple(validate_axis_mult(dest), x.ndim, mom_ndim)

    return np.moveaxis(x, axes0, axes1)


# ** Prepare for push/reduction
def prepare_secondary_value_for_reduction(
    x: ArrayLike,
    axis: int,
    nsamp: int,
    dtype: DTypeLikeArg[ScalarT],
    *,
    move_axis_to_end: bool = True,
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
    out: NDArray[ScalarT] = np.asarray(x, dtype=dtype)
    if out.ndim == 0:
        return np.broadcast_to(out, nsamp)

    if out.ndim == 1:
        axis_check = -1
    elif move_axis_to_end and axis != -1:
        out = np.moveaxis(out, axis, -1)
        axis_check = -1
    else:
        axis_check = axis

    if out.shape[axis_check] != nsamp:
        msg = f"{out.shape[axis_check]} must be same as target.shape[axis]={nsamp}"
        raise ValueError(msg)

    return out


def prepare_values_for_reduction(
    target: ArrayLike,
    *args: ArrayLike,
    narrays: int,
    axis: AxisReduce | MissingType = MISSING,
    dtype: DTypeLikeArg[ScalarT],
    move_axis_to_end: bool = True,
) -> tuple[int, tuple[NDArray[ScalarT], ...]]:
    """
    Convert input value arrays to correct form for reduction.

    Note: Played around with using axes to handle the different axes, but it's easier to
    just move to end before operation...

    Parameters
    ----------
    narrays : int
        The total number of expected arrays.  len(args) + 1 must equal narrays.
    """
    if len(args) + 1 != narrays:
        msg = f"Number of arrays {len(args) + 1} != {narrays}"
        raise ValueError(msg)

    target = np.asarray(target, dtype=dtype)
    axis = normalize_axis_index(validate_axis(axis), target.ndim)
    nsamp = target.shape[axis]

    axis_neg = positive_to_negative_index(axis, target.ndim)
    if move_axis_to_end and axis_neg != -1:
        target = np.moveaxis(target, axis_neg, -1)

    others: Iterable[NDArray[ScalarT]] = (
        prepare_secondary_value_for_reduction(
            x=x,
            axis=axis_neg,
            nsamp=nsamp,
            dtype=target.dtype,
            move_axis_to_end=move_axis_to_end,
        )
        for x in args
    )

    return -1 if move_axis_to_end else axis_neg, (target, *others)


def xprepare_secondary_value_for_reduction(
    x: xr.DataArray | ArrayLike,
    axis: int,
    nsamp: int,
    dtype: DTypeLikeArg[ScalarT],
) -> xr.DataArray | NDArray[ScalarT]:
    """Prepare secondary values for reduction."""
    if isinstance(x, xr.DataArray):
        return x.astype(dtype=dtype, copy=False)
    return prepare_secondary_value_for_reduction(
        x, axis=axis, nsamp=nsamp, dtype=dtype, move_axis_to_end=True
    )


def xprepare_values_for_reduction(
    target: xr.DataArray,
    *args: ArrayLike | xr.DataArray,
    narrays: int,
    dim: DimsReduce | MissingType,
    axis: AxisReduce | MissingType,
    dtype: DTypeLikeArg[ScalarT],
) -> tuple[list[Sequence[Hashable]], list[xr.DataArray | NDArray[ScalarT]]]:
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

    axis, dim = select_axis_dim(
        target,
        axis=axis,
        dim=dim,
    )

    dtype = target.dtype if dtype is None else dtype
    nsamp = target.sizes[dim]
    axis_neg = positive_to_negative_index(axis, target.ndim)

    arrays = [
        xprepare_secondary_value_for_reduction(
            a, axis=axis_neg, nsamp=nsamp, dtype=dtype
        )
        for a in (target, *args)
    ]
    # NOTE: Go with list[Sequence[...]] here so that `input_core_dims` can
    # be updated later with less restriction...
    input_core_dims: list[Sequence[Hashable]] = [[dim]] * len(arrays)
    return input_core_dims, arrays


def get_axes_from_values(*args: NDArrayAny, axis_neg: int) -> AxesGUFunc:
    """Get reduction axes for arrays..."""
    return [(-1,) if a.ndim == 1 else (axis_neg,) for a in args]


def get_out_from_values(
    *args: NDArray[ScalarT],
    mom: MomentsStrict,
    axis_neg: int,
    axis_new_size: int | None = None,
) -> NDArray[ScalarT]:
    """Pass in axis if this is a reduction and will be removing axis_neg"""
    val_shape: tuple[int, ...] = np.broadcast_shapes(
        args[0].shape, *(a.shape for a in args[1:] if a.ndim > 1)
    )

    # need to normalize
    axis = normalize_axis_index(axis_neg, len(val_shape))
    if axis_new_size is None:
        val_shape = (*val_shape[:axis], *val_shape[axis + 1 :])
    else:
        val_shape = (*val_shape[:axis], axis_new_size, *val_shape[axis + 1 :])

    out_shape = (*val_shape, *mom_to_mom_shape(mom))
    return np.zeros(out_shape, dtype=args[0].dtype)


def raise_if_wrong_shape(
    array: NDArrayAny, shape: tuple[int, ...], name: str | None = None
) -> None:
    """Raise error if array.shape != shape"""
    if array.shape != shape:
        name = "out" if name is None else name
        msg = f"{name}.shape={array.shape=} != required shape {shape}"
        raise ValueError(msg)


_ALLOWED_FLOAT_DTYPES = {np.dtype(np.float32), np.dtype(np.float64)}


def select_dtype(
    x: xr.DataArray | ArrayLike,
    out: NDArrayAny | xr.DataArray | None,
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


def optional_keepdims(
    x: NDArray[ScalarT],
    *,
    axis: int | Sequence[int],
    keepdims: bool = False,
) -> NDArray[ScalarT]:
    if keepdims:
        return np.expand_dims(x, axis)
    return x


# new style preparation for reduction....
def prepare_data_for_reduction(
    data: ArrayLike,
    axis: AxisReduce | MissingType,
    mom_ndim: Mom_NDim,
    dtype: DTypeLikeArg[ScalarT],
    move_axis_to_end: bool = False,
) -> tuple[int, NDArray[ScalarT]]:
    """Convert central moments array to correct form for reduction."""
    data = np.asarray(data, dtype=dtype)
    axis = normalize_axis_index(validate_axis(axis), data.ndim, mom_ndim)

    if move_axis_to_end:
        # make sure this axis is positive in case we want to use it again...
        axis_out = data.ndim - (mom_ndim + 1)
        data = np.moveaxis(data, axis, axis_out)
    else:
        axis_out = axis

    return axis_out, data


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


def xprepare_out_for_resample_vals(
    target: xr.DataArray,
    out: NDArray[ScalarT] | None,
    dim: DimsReduce,
    mom_ndim: Mom_NDim,
    move_axis_to_end: bool,
) -> NDArray[ScalarT] | None:
    if out is None:
        return out

    # if isinstance(out, xr.DataArray):
    #     return out  # noqa: ERA001

    if move_axis_to_end:
        # out should already be in correct order
        return out

    axis_neg = positive_to_negative_index(
        target.get_axis_num(dim),
        ndim=target.ndim,
    )

    return np.moveaxis(out, axis_neg - mom_ndim, -(mom_ndim + 1))


def xprepare_out_for_resample_data(
    out: NDArray[ScalarT] | None,
    *,
    mom_ndim: Mom_NDim | None,
    axis: int,
    move_axis_to_end: bool,
) -> NDArray[ScalarT] | None:
    """Move axis to last dimensions before moment dimensions."""
    if out is None:
        return out

    if move_axis_to_end:
        # out should already be in correct order
        return out

    shift = 0 if mom_ndim is None else mom_ndim
    return np.moveaxis(out, axis, -(shift + 1))


# def optional_move_end_to_axis(
#     out: NDArray[ScalarT],
#     *,
#     mom_ndim: Mom_NDim,
#     axis: int,
# ) -> NDArray[ScalarT]:
#     """
#     Move 'last' axis back to original position

#     Note that this assumes axis is negative (relative to end of array), and relative to `mom_dim`.
#     """
#     if axis != -1:
#         np.moveaxis(out, -(mom_ndim + 1), axis - mom_ndim)  # noqa: ERA001
#     return out  # noqa: ERA001


# * Xarray utilities ----------------------------------------------------------
def validate_mom_dims(
    mom_dims: Hashable | Sequence[Hashable] | None,
    mom_ndim: Mom_NDim,
    out: Any = None,
) -> MomDimsStrict:
    """Validate mom_dims to correct form."""
    if mom_dims is None:
        if isinstance(out, xr.DataArray):
            return cast("MomDimsStrict", out.dims[-mom_ndim:])

        if mom_ndim == 1:
            return ("mom_0",)
        return ("mom_0", "mom_1")

    validated: tuple[Hashable, ...]
    if isinstance(mom_dims, str):
        validated = (mom_dims,)
    elif isinstance(mom_dims, (tuple, list)):
        validated = tuple(mom_dims)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
    else:
        msg = f"Unknown {type(mom_dims)=}.  Expected str or Sequence[str]"
        raise TypeError(msg)

    if len(validated) != mom_ndim:  # pyright: ignore[reportUnknownArgumentType]
        msg = f"mom_dims={validated} inconsistent with {mom_ndim=}"
        raise ValueError(msg)
    return cast("MomDimsStrict", validated)


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

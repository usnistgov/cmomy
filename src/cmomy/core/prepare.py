"""Prepare arrays for operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from .array_utils import (
    asarray_maybe_recast,
    normalize_axis_index,
    positive_to_negative_index,
)
from .missing import MISSING
from .validate import (
    validate_axis,
)
from .xr_utils import (
    raise_if_dataset,
    select_axis_dim,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Sequence,
    )
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from .typing import (
        AxisReduce,
        DimsReduce,
        MissingType,
        Mom_NDim,
        NDArrayAny,
        ScalarT,
    )


# * Data
def prepare_data_for_reduction(
    data: ArrayLike,
    axis: AxisReduce | MissingType,
    mom_ndim: Mom_NDim,
    dtype: DTypeLike,
    recast: bool = True,
    move_axis_to_end: bool = False,
) -> tuple[int, NDArrayAny]:
    """Convert central moments array to correct form for reduction."""
    data = asarray_maybe_recast(data, dtype=dtype, recast=recast)
    axis = normalize_axis_index(validate_axis(axis), data.ndim, mom_ndim)

    if move_axis_to_end:
        # make sure this axis is positive in case we want to use it again...
        axis_out = data.ndim - (mom_ndim + 1)
        data = np.moveaxis(data, axis, axis_out)
    else:
        axis_out = axis

    return axis_out, data


# * Vals
def prepare_values_for_reduction(
    target: ArrayLike,
    *args: ArrayLike | xr.Dataset,
    narrays: int,
    axis: AxisReduce | MissingType = MISSING,
    dtype: DTypeLike,
    recast: bool = True,
    move_axis_to_end: bool = True,
) -> tuple[int, tuple[NDArrayAny, ...]]:
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

    target = asarray_maybe_recast(target, dtype=dtype, recast=recast)
    axis = normalize_axis_index(validate_axis(axis), target.ndim)
    nsamp = target.shape[axis]

    axis_neg = positive_to_negative_index(axis, target.ndim)
    if move_axis_to_end and axis_neg != -1:
        target = np.moveaxis(target, axis_neg, -1)

    others: Iterable[NDArrayAny] = (
        prepare_secondary_value_for_reduction(
            x=x,
            axis=axis_neg,
            nsamp=nsamp,
            dtype=target.dtype,
            recast=recast,
            move_axis_to_end=move_axis_to_end,
        )
        for x in args
    )

    return -1 if move_axis_to_end else axis_neg, (target, *others)


def prepare_secondary_value_for_reduction(
    x: ArrayLike | xr.Dataset,
    axis: int,
    nsamp: int,
    dtype: DTypeLike,
    recast: bool,
    *,
    move_axis_to_end: bool = True,
) -> NDArrayAny:
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
    raise_if_dataset(x, "Passed Dataset as secondary value with array primary value.")

    out: NDArrayAny = asarray_maybe_recast(x, dtype=dtype, recast=recast)  # type: ignore[arg-type]
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


def xprepare_values_for_reduction(
    target: xr.Dataset | xr.DataArray,
    *args: ArrayLike | xr.DataArray | xr.Dataset,
    narrays: int,
    dim: DimsReduce | MissingType,
    axis: AxisReduce | MissingType,
    dtype: DTypeLike,
    recast: bool = True,
) -> tuple[
    Hashable,
    list[Sequence[Hashable]],
    list[xr.Dataset | xr.DataArray | NDArrayAny],
]:
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

    nsamp = target.sizes[dim]
    axis_neg = positive_to_negative_index(
        axis,
        # if dataset, Use first variable as template...
        ndim=(
            target[next(iter(target))] if isinstance(target, xr.Dataset) else target
        ).ndim,
    )

    arrays = [
        xprepare_secondary_value_for_reduction(
            a,
            axis=axis_neg,
            nsamp=nsamp,
            dtype=dtype,
            recast=recast,
        )
        for a in (target, *args)
    ]
    # NOTE: Go with list[Sequence[...]] here so that `input_core_dims` can
    # be updated later with less restriction...
    input_core_dims: list[Sequence[Hashable]] = [[dim]] * len(arrays)
    return dim, input_core_dims, arrays


def xprepare_secondary_value_for_reduction(
    x: xr.Dataset | xr.DataArray | ArrayLike,
    axis: int,
    nsamp: int,
    dtype: DTypeLike,
    recast: bool,
) -> xr.Dataset | xr.DataArray | NDArrayAny:
    """Prepare secondary values for reduction."""
    if isinstance(x, xr.Dataset):
        return x
    if isinstance(x, xr.DataArray):
        return x.astype(dtype, copy=False) if recast and dtype is not None else x  # pyright: ignore[reportUnknownMemberType]
    return prepare_secondary_value_for_reduction(
        x,
        axis=axis,
        nsamp=nsamp,
        dtype=dtype,
        recast=recast,
        move_axis_to_end=True,
    )


# * Out
def xprepare_out_for_resample_vals(
    target: xr.DataArray | xr.Dataset,
    out: NDArray[ScalarT] | None,
    dim: DimsReduce,
    mom_ndim: Mom_NDim,
    move_axis_to_end: bool,
) -> NDArray[ScalarT] | None:
    """Prepare out for resampling"""
    # NOTE: silently ignore out of datasets.
    if out is None or isinstance(target, xr.Dataset):
        return None

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
    data: Any = None,
) -> NDArray[ScalarT] | None:
    """Move axis to last dimensions before moment dimensions."""
    if out is None or isinstance(data, xr.Dataset):
        return None

    if move_axis_to_end:
        # out should already be in correct order
        return out

    shift = 0 if mom_ndim is None else mom_ndim
    return np.moveaxis(out, axis, -(shift + 1))

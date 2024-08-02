"""Prepare arrays for operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ._missing import MISSING
from ._utils import (
    normalize_axis_index,
    positive_to_negative_index,
    select_axis_dim,
)
from ._validate import (
    validate_axis,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Sequence,
    )

    from numpy.typing import ArrayLike, NDArray

    from .typing import (
        AxisReduce,
        DimsReduce,
        DTypeLikeArg,
        MissingType,
        Mom_NDim,
        ScalarT,
    )


# * Data
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


# * Vals
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


# * Out
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

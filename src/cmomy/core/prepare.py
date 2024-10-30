"""Prepare arrays for operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cmomy.core.moment_params import MomParamsXArray, default_mom_params_xarray
from cmomy.core.utils import mom_to_mom_shape

from .array_utils import (
    arrayorder_to_arrayorder_cf,
    asarray_maybe_recast,
    normalize_axis_index,
    positive_to_negative_index,
)
from .missing import MISSING
from .validate import (
    is_dataarray,
    is_dataset,
    validate_axis,
)
from .xr_utils import (
    raise_if_dataset,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Sequence,
    )

    import xarray as xr
    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from cmomy.core.moment_params import MomParamsArray
    from cmomy.core.typing import AxisReduceWrap

    from .typing import (
        ArrayOrder,
        ArrayOrderCF,
        DimsReduce,
        MissingType,
        MomentsStrict,
        MomNDim,
        NDArrayAny,
        ScalarT,
    )


# * Data
def prepare_data_for_reduction(
    data: ArrayLike,
    axis: AxisReduceWrap | MissingType,
    mom_params: MomParamsArray,
    dtype: DTypeLike,
    recast: bool = True,
    axes_to_end: bool = False,
) -> tuple[int, MomParamsArray, NDArrayAny]:
    """Convert central moments array to correct form for reduction."""
    data = asarray_maybe_recast(data, dtype=dtype, recast=recast)
    axis = normalize_axis_index(
        validate_axis(axis),
        data.ndim,
        mom_ndim=mom_params.ndim,
    )

    if axes_to_end:
        axis_out = data.ndim - (mom_params.ndim + 1)
        mom_params_orig, mom_params = mom_params, mom_params.axes_to_end()
        data = np.moveaxis(
            data, (axis, *mom_params_orig.axes), (axis_out, *mom_params.axes)
        )
    else:
        axis_out = axis

    return axis_out, mom_params, data


# * Vals
def prepare_values_for_reduction(
    target: ArrayLike,
    *args: ArrayLike | xr.Dataset,
    narrays: int,
    axis: AxisReduceWrap | MissingType = MISSING,
    dtype: DTypeLike,
    recast: bool = True,
    axes_to_end: bool = True,
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
    if axes_to_end and axis_neg != -1:
        target = np.moveaxis(target, axis_neg, -1)

    others: Iterable[NDArrayAny] = (
        prepare_secondary_value_for_reduction(
            x=x,
            axis=axis_neg,
            nsamp=nsamp,
            dtype=target.dtype,
            recast=recast,
            axes_to_end=axes_to_end,
        )
        for x in args
    )

    return -1 if axes_to_end else axis_neg, (target, *others)


def prepare_secondary_value_for_reduction(
    x: ArrayLike | xr.Dataset,
    axis: int,
    nsamp: int,
    dtype: DTypeLike,
    recast: bool,
    *,
    axes_to_end: bool = True,
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

    out: NDArrayAny = asarray_maybe_recast(x, dtype=dtype, recast=recast)  # type: ignore[arg-type, unused-ignore]
    if out.ndim == 0:
        return np.broadcast_to(out, nsamp)

    if out.ndim == 1:
        axis_check = -1
    elif axes_to_end and axis != -1:
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
    axis: AxisReduceWrap | MissingType,
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

    axis, dim = default_mom_params_xarray.select_axis_dim(
        target,
        axis=axis,
        dim=dim,
    )

    nsamp = target.sizes[dim]
    axis_neg = positive_to_negative_index(
        axis,
        # if dataset, Use first variable as template...
        ndim=(target[next(iter(target))] if is_dataset(target) else target).ndim,
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
    if is_dataset(x):
        return x
    if is_dataarray(x):
        return x.astype(dtype, copy=False) if (recast and dtype is not None) else x  # pyright: ignore[reportUnknownMemberType]
    return prepare_secondary_value_for_reduction(
        x,
        axis=axis,
        nsamp=nsamp,
        dtype=dtype,
        recast=recast,
        axes_to_end=True,
    )


# * Out
def xprepare_out_for_reduce_data(
    target: xr.DataArray | xr.Dataset,
    out: NDArray[ScalarT] | None,
    *,
    dim: tuple[Hashable, ...],
    mom_params: MomParamsXArray,
    keepdims: bool,
    axes_to_end: bool,
) -> NDArray[ScalarT] | None:
    """Prepare out for reduce_data"""
    if out is None or is_dataset(target):
        return None

    if axes_to_end:
        return out

    if keepdims:
        return np.moveaxis(
            out,
            (*target.get_axis_num(dim), *mom_params.get_axes(target)),
            range(-(len(dim) + mom_params.ndim), 0),
        )

    # otherwise need to remove reduction dimensions before move.
    dims = [d for d in target.dims if d not in dim]
    axes0 = [dims.index(d) for d in mom_params.dims]

    return np.moveaxis(out, axes0, mom_params.axes_last)


def xprepare_out_for_transform(
    target: xr.DataArray | xr.Dataset,
    out: NDArray[ScalarT] | None,
    *,
    mom_params: MomParamsXArray,
    axes_to_end: bool,
) -> NDArray[ScalarT] | None:
    """Prepare out for transform."""
    if out is None or is_dataset(target):
        return None

    if axes_to_end:
        return out

    return np.moveaxis(
        out,
        mom_params.get_axes(target),
        mom_params.axes_last,
    )


def xprepare_out_for_resample_vals(
    target: xr.DataArray | xr.Dataset,
    out: NDArray[ScalarT] | None,
    dim: DimsReduce,
    mom_ndim: MomNDim,
    axes_to_end: bool,
) -> NDArray[ScalarT] | None:
    """Prepare out for resampling"""
    # NOTE: silently ignore out of datasets.
    if out is None or is_dataset(target):
        return None

    if axes_to_end:
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
    mom_params: MomParamsXArray,
    axis: int,
    axes_to_end: bool,
    data: xr.DataArray | xr.Dataset,
) -> NDArray[ScalarT] | None:
    """Move axis to last dimensions before moment dimensions."""
    if out is None or is_dataset(data):
        return None

    if axes_to_end:
        # out should already be in correct order
        return out

    axes0 = (axis, *mom_params.get_axes(data))
    axes1 = (-(mom_params.ndim + 1), *mom_params.axes_last)
    return np.moveaxis(out, axes0, axes1)


def prepare_out_from_values(
    out: NDArrayAny | None,
    *args: NDArrayAny,
    mom: MomentsStrict,
    axis_neg: int,
    axis_new_size: int | None = None,
    dtype: DTypeLike,
    order: ArrayOrderCF,
) -> NDArrayAny:
    """Pass in axis if this is a reduction and will be removing axis_neg"""
    if out is not None:
        return out

    val_shape: tuple[int, ...] = np.broadcast_shapes(
        args[0].shape, *(a.shape for a in args[1:] if a.ndim > 1)
    )
    mom_shape = mom_to_mom_shape(mom)

    axis = normalize_axis_index(axis_neg, len(val_shape))
    if axis_new_size is None:
        return np.empty(
            (*val_shape[:axis], *val_shape[axis + 1 :], *mom_shape),
            dtype=dtype,
            order=order,
        )

    if axis_neg == -1 or order is not None:
        # special case, axis is already at the end
        return np.empty(
            (*val_shape[:axis], axis_new_size, *val_shape[axis + 1 :], *mom_shape),
            dtype=dtype,
            order=order,
        )

    # otherwise, make array in calculation order
    out = np.empty(
        (*val_shape[:axis], *val_shape[axis + 1 :], axis_new_size, *mom_shape),
        dtype=dtype,
        order=order,
    )
    return np.moveaxis(out, -(len(mom) + 1), axis)


def prepare_out_for_reduce_data_grouped(
    data: NDArrayAny,
    *,
    mom_params: MomParamsArray,
    axis: int,
    axis_new_size: int,
    order: ArrayOrderCF,
    dtype: DTypeLike,
) -> NDArrayAny:
    """Prepare out with ordering."""
    shape = (*data.shape[:axis], axis_new_size, *data.shape[axis + 1 :])
    if order is None:
        # otherwise, make array in calculation order
        axes0 = (axis, *mom_params.axes)
        axes1 = (data.ndim - (mom_params.ndim + 1), *mom_params.axes_last)
        if axes0 != axes1:
            axes0 = mom_params.normalize_axis_tuple(axes0, data.ndim)
            new_shape = [s for i, s in enumerate(data.shape) if i not in axes0]

            out = np.empty(
                (*new_shape, axis_new_size, *mom_params.get_mom_shape(data)),
                dtype=dtype,
                order=None,
            )
            return np.moveaxis(out, axes1, axes0)

    return np.empty(shape, dtype=dtype, order=order)


def optional_prepare_out_for_resample_data(
    *,
    out: NDArrayAny | None,
    data: NDArrayAny,
    axis: int,
    axis_new_size: int,
    order: ArrayOrder,
    dtype: DTypeLike,
) -> NDArrayAny | None:
    """Prepare out with ordering."""
    if out is not None:
        return out

    order = arrayorder_to_arrayorder_cf(order)
    if order is None:
        return None

    return np.empty(
        (*data.shape[:axis], axis_new_size, *data.shape[axis + 1 :]),
        dtype=dtype,
        order=order,
    )

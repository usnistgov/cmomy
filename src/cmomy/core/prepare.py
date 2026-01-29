"""Prepare arrays for operations."""

# pylint: disable=missing-class-docstring

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from .array_utils import (
    arrayorder_to_arrayorder_cf,
    asarray_maybe_recast,
    normalize_axis_index,
    positive_to_negative_index,
    reorder,
)
from .missing import MISSING
from .moment_params import MomParamsArray, MomParamsXArray, default_mom_params_xarray
from .utils import mom_to_mom_shape
from .validate import (
    is_dataarray,
    is_dataset,
    validate_axis,
    validate_axis_mult,
    validate_mom,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Sequence,
    )

    import xarray as xr
    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from .moment_params import MomParamsType
    from .typing import (
        ArrayOrderCF,
        ArrayOrderKACF,
        AxisReduceMultWrap,
        AxisReduceWrap,
        DimsReduce,
        MissingType,
        MomentsStrict,
        MomNDim,
        NDArrayAny,
    )
    from .typing_compat import Self, TypeVar

    _ScalarT = TypeVar("_ScalarT", bound=np.generic)


# * Prepare classes
@dataclass
class _PrepareBaseArray:
    mom_params: MomParamsArray
    recast: bool = field(default=False)

    @classmethod
    def factory(
        cls,
        mom_params: MomParamsType = None,
        ndim: int | None = None,
        axes: int | Sequence[int] | None = None,
        default_ndim: MomNDim | None = None,
        recast: bool = False,
    ) -> Self:
        mom_params = MomParamsArray.factory(
            mom_params=mom_params, ndim=ndim, axes=axes, default_ndim=default_ndim
        )
        return cls(mom_params=mom_params, recast=recast)


@dataclass
class _PrepareBaseXArray:
    mom_params: MomParamsXArray
    recast: bool = field(default=False)

    @classmethod
    def factory(
        cls,
        mom_params: MomParamsType = None,
        ndim: int | None = None,
        dims: Hashable | Sequence[Hashable] | None = None,
        axes: int | Sequence[int] | None = None,
        data: object = None,
        default_ndim: MomNDim | None = None,
        recast: bool = False,
    ) -> Self:
        mom_params = MomParamsXArray.factory(
            mom_params=mom_params,
            ndim=ndim,
            dims=dims,
            axes=axes,
            data=data,
            default_ndim=default_ndim,
        )
        return cls(mom_params=mom_params, recast=recast)


@dataclass
class PrepareDataArray(_PrepareBaseArray):
    """Prepare data for reductions"""

    def data_for_reduction(
        self,
        data: ArrayLike,
        *,
        axis: AxisReduceWrap | MissingType,
        axes_to_end: bool = False,
        dtype: DTypeLike,
    ) -> tuple[Self, int, NDArrayAny]:
        """Convert central moments array to correct form for reduction."""
        data = asarray_maybe_recast(data, dtype=dtype, recast=self.recast)
        axis = self.mom_params.normalize_axis_index(validate_axis(axis), data.ndim)

        if axes_to_end:
            axis_out = data.ndim - (self.mom_params.ndim + 1)
            data = np.moveaxis(
                data,
                (axis, *self.mom_params.axes),
                (axis_out, *self.mom_params.axes_last),
            )
            return (
                replace(self, mom_params=self.mom_params.axes_to_end()),
                axis_out,
                data,
            )

        return self, axis, data

    def data_for_reduction_multiple(
        self,
        data: ArrayLike,
        *,
        axis: AxisReduceMultWrap | MissingType,
        axes_to_end: bool,
        dtype: DTypeLike,
    ) -> tuple[Self, tuple[int, ...], NDArrayAny]:
        data = asarray_maybe_recast(data, dtype=dtype, recast=self.recast)
        if axis is None:
            mom_axes = self.mom_params.normalize_axis_tuple(
                self.mom_params.axes, data.ndim
            )
            axis = tuple(i for i in range(data.ndim) if i not in mom_axes)
        else:
            axis = self.mom_params.normalize_axis_tuple(
                validate_axis_mult(axis), data.ndim
            )

        if axes_to_end:
            axis_out = tuple(
                range(
                    data.ndim - (self.mom_params.ndim + len(axis)),
                    data.ndim - self.mom_params.ndim,
                )
            )
            data = np.moveaxis(
                data,
                (*axis, *self.mom_params.axes),
                (*axis_out, *self.mom_params.axes_last),
            )
            return (
                replace(self, mom_params=self.mom_params.axes_to_end()),
                axis_out,
                data,
            )
        return self, axis, data

    @staticmethod
    def optional_out_sample(
        out: NDArrayAny | None,
        *,
        data: NDArrayAny,
        axis: int,
        axis_new_size: int,
        order: ArrayOrderKACF,
        dtype: DTypeLike,
    ) -> NDArrayAny | None:
        """Prepare optional out with ordering."""
        if out is not None:
            return out

        if (order := arrayorder_to_arrayorder_cf(order)) is None:
            return None

        return np.empty(
            (*data.shape[:axis], axis_new_size, *data.shape[axis + 1 :]),
            dtype=dtype,
            order=order,
        )

    def out_sample(
        self,
        out: NDArrayAny | None,
        *,
        data: NDArrayAny,
        axis: int,
        axis_new_size: int | None,
        order: ArrayOrderCF,
        dtype: DTypeLike,
    ) -> NDArrayAny:
        """
        Prepare out with ordering.

        Use this when `out` is required (i.e., signature of gufunc doesn't have "->").
        """
        if out is not None:
            return out

        if axis_new_size is None:
            axis_new_size = data.shape[axis]

        shape: tuple[int, ...] = (
            *data.shape[:axis],
            axis_new_size,  # pyright: ignore[reportAssignmentType]
            *data.shape[axis + 1 :],
        )
        if order is None:
            # otherwise, make array in calculation order
            axes0 = (axis, *self.mom_params.axes)
            axes1 = (data.ndim - (self.mom_params.ndim + 1), *self.mom_params.axes_last)
            out = np.empty(
                reorder(shape, axes0, axes1),
                dtype=dtype,
                order=None,
            )
            return np.moveaxis(out, axes1, axes0)

        return np.empty(shape, dtype=dtype, order=order)


@dataclass
class PrepareValsArray(_PrepareBaseArray):
    """Prepare values for reductions"""

    @classmethod
    def factory_mom(
        cls,
        mom: int | Sequence[int],
        mom_params: MomParamsType = None,
        axes: int | Sequence[int] | None = None,
        default_ndim: MomNDim | None = None,
        recast: bool = False,
    ) -> tuple[Self, MomentsStrict]:
        mom = validate_mom(mom)
        return cls.factory(
            mom_params=mom_params,
            ndim=len(mom),
            axes=axes,
            default_ndim=default_ndim,
            recast=recast,
        ), mom

    def values_for_reduction(
        self,
        target: ArrayLike,
        *args: ArrayLike | xr.Dataset,
        narrays: int,
        axis: AxisReduceWrap | MissingType = MISSING,
        axes_to_end: bool,
        dtype: DTypeLike,
    ) -> tuple[Self, int, tuple[NDArrayAny, ...]]:
        obj = (
            replace(self, mom_params=self.mom_params.axes_to_end())
            if axes_to_end
            else self
        )

        return obj, *prepare_array_values_for_reduction(
            target,
            *args,
            narrays=narrays,
            axis=axis,
            axes_to_end=axes_to_end,
            dtype=dtype,
            recast=self.recast,
        )

    def get_axis_sample_out(
        self,
        axis_neg: int,
        axis: int | None,
        axis_new_size: int | MissingType | None,
        out_ndim: int,
    ) -> int:
        if (
            axis_new_size is MISSING
            or self.mom_params.axes == self.mom_params.axes_last
        ):
            return axis_neg - self.mom_params.ndim

        if axis is None:
            axis = normalize_axis_index(axis_neg, out_ndim - self.mom_params.ndim)
        return reorder(out_ndim, self.mom_params.axes_last, self.mom_params.axes).index(
            axis
        )

    @staticmethod
    def get_val_shape(*args: NDArrayAny | xr.DataArray) -> tuple[int, ...]:
        return np.broadcast_shapes(
            args[0].shape, *(a.shape for a in args[1:] if a.ndim > 1)
        )

    def out_from_values(
        self,
        out: NDArrayAny | None,
        *,
        val_shape: Sequence[int],
        mom: MomentsStrict,
        axis_neg: int,
        axis_new_size: int | MissingType | None = MISSING,
        dtype: DTypeLike,
        order: ArrayOrderCF,
    ) -> tuple[NDArrayAny, int]:
        """
        Pass in axis if this is a reduction and will be removing axis_neg

        None value for axis_new_size implies axis_new_size = val_shape[axis_neg]
        """
        if out is not None:
            return out, self.get_axis_sample_out(
                axis_neg, None, axis_new_size, out.ndim
            )

        mom_shape = mom_to_mom_shape(mom)
        axis = normalize_axis_index(axis_neg, len(val_shape))
        if axis_new_size is None:
            axis_new_size = val_shape[axis]
        axis_sample_out = self.get_axis_sample_out(
            axis_neg, axis, axis_new_size, len(val_shape) + self.mom_params.ndim
        )

        if order is None:
            # use calculation order
            shape_calculate = (
                *val_shape[:axis],
                *val_shape[axis + 1 :],
                *(() if axis_new_size is MISSING else (axis_new_size,)),
                *mom_shape,
            )
            out = np.empty(shape_calculate, dtype=dtype, order=order)

            # reorder
            if axis_new_size is MISSING:
                axes0 = self.mom_params.axes_last
                axes1 = self.mom_params.axes
            else:
                axes0 = (
                    out.ndim - (self.mom_params.ndim + 1),
                    *self.mom_params.axes_last,
                )
                axes1 = (axis_sample_out, *self.mom_params.axes)

            return np.moveaxis(out, axes0, axes1), axis_sample_out

        # Use actual order
        shape: list[int] = [
            *val_shape[:axis],
            *(() if axis_new_size is MISSING else (axis_new_size,)),
            *val_shape[axis + 1 :],
            *mom_shape,
        ]

        shape = reorder(shape, self.mom_params.axes_last, self.mom_params.axes)
        return np.empty(shape, dtype=dtype, order=order), axis_sample_out


@dataclass
class PrepareDataXArray(_PrepareBaseXArray):
    """Prepare data for reduction"""

    def to_array(self, data: xr.DataArray | None = None) -> PrepareDataArray:
        return PrepareDataArray(
            mom_params=self.mom_params.to_array(data=data),
            recast=self.recast,
        )

    @cached_property
    def prepare_array(self) -> PrepareDataArray:
        return self.to_array()

    def optional_out_reduce(
        self,
        out: NDArray[_ScalarT] | None,
        *,
        target: xr.DataArray | xr.Dataset,
        dim: tuple[Hashable, ...],
        keepdims: bool,
        axes_to_end: bool,
        order: ArrayOrderKACF,
        dtype: DTypeLike,
    ) -> NDArray[_ScalarT] | None:
        """Prepare out for reduce_data"""
        if is_dataset(target):
            return None

        if axes_to_end:
            # out is None or in correct form
            return out

        axis = target.get_axis_num(dim)
        if out is None:
            if (order_cf := arrayorder_to_arrayorder_cf(order)) is not None:
                if keepdims:
                    shape = tuple(
                        1 if i in axis else s for i, s in enumerate(target.shape)
                    )
                else:
                    shape = tuple(
                        s for i, s in enumerate(target.shape) if i not in axis
                    )
                out = np.empty(shape, dtype=dtype, order=order_cf)
            else:
                return None

        if keepdims:
            axes0 = (*axis, *self.mom_params.get_axes(target))
            axes1 = range(-(len(dim) + self.mom_params.ndim), 0)

        else:
            # otherwise need to remove reduction dimensions before move.
            dims = [d for d in target.dims if d not in dim]
            axes0 = tuple(dims.index(d) for d in self.mom_params.dims)
            axes1 = self.mom_params.axes_last

        return np.moveaxis(out, axes0, axes1)

    def optional_out_transform(
        self,
        out: NDArray[_ScalarT] | None,
        *,
        target: xr.DataArray | xr.Dataset,
        axes_to_end: bool,
        order: ArrayOrderKACF,
        dtype: DTypeLike,
    ) -> NDArray[_ScalarT] | None:
        """Prepare out for transform."""
        if is_dataset(target):
            return None

        if axes_to_end:
            # out is None or in correct order
            return out

        if out is None:
            if (order_cf := arrayorder_to_arrayorder_cf(order)) is not None:
                out = np.empty(target.shape, dtype=dtype, order=order_cf)
            else:
                return None

        return np.moveaxis(
            out,
            self.mom_params.get_axes(target),
            self.mom_params.axes_last,
        )

    def optional_out_sample(
        self,
        out: NDArray[_ScalarT] | None,
        *,
        data: xr.DataArray | xr.Dataset,
        axis: int,
        axis_new_size: int | None = None,
        axes_to_end: bool,
        order: ArrayOrderKACF,
        dtype: DTypeLike,
    ) -> NDArray[_ScalarT] | None:
        """Move axis to last dimensions before moment dimensions."""
        if is_dataset(data):
            return None

        if axes_to_end:
            # out is None or in correct order
            return out

        if out is None:
            if (order_cf := arrayorder_to_arrayorder_cf(order)) is not None:
                if axis_new_size is None:
                    axis_new_size = data.shape[axis]
                out = np.empty(
                    (*data.shape[:axis], axis_new_size, *data.shape[axis + 1 :]),
                    dtype=dtype,
                    order=order_cf,
                )
            else:
                return None

        return np.moveaxis(
            out,
            (axis, *self.mom_params.get_axes(data)),
            (-(self.mom_params.ndim + 1), *self.mom_params.axes_last),
        )


@dataclass
class PrepareValsXArray(_PrepareBaseXArray):
    """Prepare values for reduction"""

    def to_array(self, data: xr.DataArray | None = None) -> PrepareValsArray:
        return PrepareValsArray(
            mom_params=self.mom_params.to_array(data=data),
            recast=self.recast,
        )

    @cached_property
    def prepare_array(self) -> PrepareValsArray:
        return self.to_array()

    @classmethod
    def factory_mom(
        cls,
        mom: int | Sequence[int],
        mom_params: MomParamsType = None,
        dims: Hashable | Sequence[Hashable] | None = None,
        axes: int | Sequence[int] | None = None,
        data: object = None,
        default_ndim: MomNDim | None = None,
        recast: bool = False,
    ) -> tuple[Self, MomentsStrict]:
        mom = validate_mom(mom)
        return cls.factory(
            ndim=len(mom),
            mom_params=mom_params,
            dims=dims,
            axes=axes,
            data=data,
            default_ndim=default_ndim,
            recast=recast,
        ), mom

    def values_for_reduction(
        self,
        target: xr.Dataset | xr.DataArray,
        *args: ArrayLike | xr.DataArray | xr.Dataset,
        narrays: int,
        dim: DimsReduce | MissingType,
        axis: AxisReduceWrap | MissingType,
        dtype: DTypeLike,
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
        return prepare_xarray_values_for_reduction(
            target,
            *args,
            narrays=narrays,
            dim=dim,
            axis=axis,
            dtype=dtype,
            recast=self.recast,
        )

    def optional_out_from_values(
        self,
        out: NDArray[_ScalarT] | None,
        *args: NDArrayAny | xr.DataArray | xr.Dataset,
        target: xr.DataArray | xr.Dataset,
        dim: DimsReduce,
        mom: MomentsStrict,
        axis_new_size: int | MissingType | None = MISSING,
        axes_to_end: bool,
        order: ArrayOrderCF,
        dtype: DTypeLike,
        mom_axes: int | Sequence[int] | None,
        mom_params: MomParamsType,
    ) -> tuple[NDArray[_ScalarT] | None, MomParamsArray]:
        """Prepare out for resampling"""
        # NOTE: silently ignore out of datasets.

        if is_dataset(target):
            return None, self.prepare_array.mom_params

        # special case for if have new mom_axes
        prep_array = PrepareValsArray.factory(
            axes=mom_axes, mom_params=mom_params, ndim=self.mom_params.ndim
        )

        mom_params = prep_array.mom_params
        if out is None and order is None:
            return None, mom_params

        if axes_to_end or (out is None and order is None):
            # axes_to_end -> out is None or in correct order
            # out is None and order is None -> defer to PrepareValsArray
            return out, mom_params

        if any(is_dataset(_) for _ in args):
            msg = "Passed secondary dataset"
            raise TypeError(msg)

        axis_neg = positive_to_negative_index(target.get_axis_num(dim), target.ndim)
        val_shape = reorder(
            prep_array.get_val_shape(*args),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
            -1,
            axis_neg,
        )

        # shape of values with axis in last position
        out, axis_sample_out = prep_array.out_from_values(
            out,
            val_shape=val_shape,
            mom=mom,
            axis_neg=axis_neg,
            axis_new_size=axis_new_size,
            dtype=dtype,
            order=order,
        )

        if axis_new_size is MISSING:
            axes0 = prep_array.mom_params.axes
            axes1 = prep_array.mom_params.axes_last
        else:
            axes0 = (axis_sample_out, *prep_array.mom_params.axes)
            axes1 = range(-(self.mom_params.ndim + 1), 0)
        return np.moveaxis(out, axes0, axes1), mom_params


# * Methods
# These don't require mom_params, so leave them a functions.
def prepare_array_values_for_reduction(
    target: ArrayLike,
    *args: ArrayLike | xr.Dataset,
    narrays: int,
    axis: AxisReduceWrap | MissingType = MISSING,
    axes_to_end: bool,
    dtype: DTypeLike,
    recast: bool = False,
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
        _prepare_array_secondary_value_for_reduction(
            x=x,
            axis=axis_neg,
            nsamp=nsamp,
            axes_to_end=axes_to_end,
            dtype=target.dtype,
            recast=recast,
        )
        for x in args
    )

    return -1 if axes_to_end else axis_neg, (target, *others)


def _prepare_array_secondary_value_for_reduction(
    x: ArrayLike | xr.Dataset,
    *,
    axis: int,
    nsamp: int,
    axes_to_end: bool,
    dtype: DTypeLike,
    recast: bool,
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
    """
    if is_dataset(x):
        msg = "Passed Dataset as secondary value with array primary value."
        raise TypeError(msg)

    out: NDArrayAny = asarray_maybe_recast(x, dtype=dtype, recast=recast)
    if not out.ndim:
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


def prepare_xarray_values_for_reduction(
    target: xr.Dataset | xr.DataArray,
    *args: ArrayLike | xr.DataArray | xr.Dataset,
    narrays: int,
    dim: DimsReduce | MissingType,
    axis: AxisReduceWrap | MissingType,
    dtype: DTypeLike,
    recast: bool = False,
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
        _prepare_xarray_secondary_value_for_reduction(
            a,
            axis=axis_neg,
            dim=dim,
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


def _prepare_xarray_secondary_value_for_reduction(
    x: xr.Dataset | xr.DataArray | ArrayLike,
    axis: int,
    dim: DimsReduce,
    nsamp: int,
    dtype: DTypeLike,
    recast: bool,
) -> xr.Dataset | xr.DataArray | NDArrayAny:
    """Prepare secondary values for reduction."""
    if is_dataset(x):
        return x
    if is_dataarray(x):
        return (
            x.astype(dtype, copy=False)  # pyright: ignore[reportUnknownMemberType]
            if (recast and dtype is not None)
            else x
        ).transpose(..., dim)
    return _prepare_array_secondary_value_for_reduction(
        x,
        axis=axis,
        nsamp=nsamp,
        axes_to_end=True,
        dtype=dtype,
        recast=recast,
    )

"""Thin wrapper around central routines with xarray support."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

import numpy as np

# pandas needed for autdoc typehints
import pandas as pd  # noqa: F401  # pyright: ignore[reportUnusedImport]
import xarray as xr

from ._central_abc import CentralMomentsABC
from ._utils import (
    MISSING,
    # replace_coords_from_isel,
    select_axis_dim,
    validate_mom_and_mom_ndim,
    xprepare_data_for_reduction,
    xprepare_values_for_reduction,
)
from .docstrings import docfiller_xcentral as docfiller

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence
    from typing import (
        Any,
        Callable,
        Literal,
    )

    # Iterable,
    from numpy.typing import ArrayLike, DTypeLike, NDArray
    from xarray.core import types as xr_types
    from xarray.core.coordinates import DataArrayCoordinates
    from xarray.core.indexes import Indexes

    from cmomy.typing import KeepAttrs

    from ._central_numpy import CentralMoments
    from ._typing_compat import Self
    from .typing import (
        ArrayOrder,
        ArrayOrderCF,
        AxisReduce,
        CoordsPolicy,
        DataCasting,
        Dims,
        DimsReduce,
        DTypeLikeArg,
        FloatT2,
        Groups,
        MissingType,
        Mom_NDim,
        MomDims,
        Moments,
        NDArrayAny,
        NDArrayInt,
        XArrayAttrsType,
        XArrayCoordsType,
        XArrayDimsType,
        XArrayIndexesType,
        XArrayNameType,
    )

from .typing import FloatT

# * xCentralMoments -----------------------------------------------------------
docfiller_abc = docfiller.factory_from_parent(CentralMomentsABC)
docfiller_inherit_abc = docfiller.factory_inherit_from_parent(CentralMomentsABC)


@docfiller.inherit(CentralMomentsABC)  # noqa: PLR0904
class xCentralMoments(CentralMomentsABC[FloatT, xr.DataArray]):  # noqa: N801
    """Wrapper of :class:`xarray.DataArray` based central moments data."""

    _xdata: xr.DataArray

    __slots__ = ("_xdata",)

    @overload
    def __init__(
        self,
        data: xr.DataArray,
        *,
        mom_ndim: Mom_NDim = ...,
        copy: bool | None = ...,
        order: ArrayOrder = ...,
        dtype: DTypeLikeArg[FloatT],
        fastpath: bool = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        data: xr.DataArray,
        *,
        mom_ndim: Mom_NDim = ...,
        copy: bool | None = ...,
        order: ArrayOrder = ...,
        dtype: DTypeLike = ...,
        fastpath: bool = ...,
    ) -> None: ...

    def __init__(
        self,
        data: xr.DataArray,
        *,
        mom_ndim: Mom_NDim = 1,
        copy: bool | None = None,
        order: ArrayOrder = None,
        dtype: DTypeLike = None,
        fastpath: bool = False,
    ) -> None:
        if not isinstance(data, xr.DataArray):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = (
                "data must be a xarray.DataArray. "
                "See CentralMoments.to_xcentralmoments for wrapping numpy arrays"
            )
            raise TypeError(msg)

        if fastpath:
            self._cache = {}
            self._xdata = data
            self._data = self._xdata.data
            self._mom_ndim = mom_ndim
        else:
            # Handle None value directly because using astype...
            copy = False if copy is None else copy
            super().__init__(
                data=data.astype(dtype or data.dtype, copy=copy, order=order),  # pyright: ignore[reportUnknownMemberType]
                mom_ndim=mom_ndim,
            )

    def set_values(self, values: xr.DataArray) -> None:
        # Ensure we have a numpy array underlying DataArray.
        self._data = values.to_numpy()  # pyright: ignore[reportUnknownMemberType]
        self._xdata = values.copy(data=self._data)

    def to_values(self) -> xr.DataArray:
        return self._xdata

    # ** DataArray properties -----------------------------------------------------
    def to_dataarray(self) -> xr.DataArray:
        """Underlying :class:`xarray.DataArray`."""
        return self._xdata

    @property
    def attrs(self) -> dict[Any, Any]:
        """Attributes of values."""
        return self._xdata.attrs

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """Dimensions of values."""
        return self._xdata.dims

    @property
    def coords(self) -> DataArrayCoordinates[Any]:
        """Coordinates of values."""
        return self._xdata.coords  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

    @property
    def name(self) -> Hashable:
        """Name of values."""
        return self._xdata.name

    @property
    def indexes(self) -> Indexes[Any]:  # pragma: no cover
        """Indexes of values."""
        return self._xdata.indexes  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

    @property
    def sizes(self) -> Mapping[Hashable, int]:
        """Sizes of values."""
        return self._xdata.sizes

    @property
    def val_dims(self) -> tuple[Hashable, ...]:
        """Names of value dimensions."""
        return self.dims[: -self.mom_ndim]

    @property
    def mom_dims(self) -> tuple[Hashable, ...]:
        """Names of moment dimensions."""
        return self.dims[-self.mom_ndim :]

    # ** top level creation/copy/new ----------------------------------------------
    def _wrap_like(self, x: NDArrayAny) -> xr.DataArray:
        return self._xdata.copy(data=x)

    # def _replace_coords_isel(
    #     self,
    #     da_selected: xr.DataArray,
    #     indexers: Mapping[Any, Any] | None = None,
    #     drop: bool = False,
    #     **indexers_kwargs: Any,
    # ) -> xr.DataArray:
    #     """Update coords when reducing."""
    #     return replace_coords_from_isel(
    #         da_original=self._xdata,
    #         da_selected=da_selected,
    #         indexers=indexers,
    #         drop=drop,
    #         **indexers_kwargs,
    #     )

    # def _check_reduce_axis_dim(
    #     self,
    #     *,
    #     axis: AxisReduce | MissingType = MISSING,
    #     dim: DimsReduce | MissingType = MISSING,
    #     default_axis: AxisReduce | MissingType = MISSING,
    #     default_dim: DimsReduce | MissingType = MISSING,
    # ) -> tuple[int, Hashable]:
    #     self._raise_if_scalar()

    #     axis, dim = select_axis_dim(
    #         dims=self.dims,
    #         axis=axis,
    #         dim=dim,
    #         default_axis=default_axis,
    #         default_dim=default_dim,
    #     )

    #     if dim in self.mom_dims:
    #         msg = f"Can only reduce over value dimensions {self.val_dims}. Passed moment dimension {dim}."
    #         raise ValueError(msg)

    #     return axis, dim

    # def _remove_dim(self, dim: Hashable | Iterable[Hashable]) -> tuple[Hashable, ...]:
    #     """Return self.dims with dim removed"""
    #     dim = {dim} if isinstance(dim, str) else set(dim)  # type: ignore[arg-type]
    #     return tuple(d for d in self.dims if d not in dim)

    @overload
    def new_like(
        self,
        data: NDArray[FloatT2],
        *,
        copy: bool | None = ...,
        order: ArrayOrder = ...,
        verify: bool = ...,
        dtype: None = ...,
    ) -> xCentralMoments[FloatT2]: ...
    @overload
    def new_like(
        self,
        data: xr.DataArray,
        *,
        copy: bool | None = ...,
        order: ArrayOrder = ...,
        verify: bool = ...,
        dtype: None = ...,
    ) -> xCentralMoments[Any]: ...
    @overload
    def new_like(
        self,
        data: Any = ...,
        *,
        copy: bool | None = ...,
        order: ArrayOrder = ...,
        verify: bool = ...,
        dtype: DTypeLikeArg[FloatT2],
    ) -> xCentralMoments[FloatT2]: ...
    @overload
    def new_like(
        self,
        data: None = ...,
        *,
        copy: bool | None = ...,
        order: ArrayOrder = ...,
        verify: bool = ...,
        dtype: None = ...,
    ) -> Self: ...
    @overload
    def new_like(
        self,
        data: Any = ...,
        *,
        copy: bool | None = ...,
        order: ArrayOrder = ...,
        verify: bool = ...,
        dtype: DTypeLike = ...,
    ) -> xCentralMoments[Any]: ...

    @docfiller_abc()
    def new_like(
        self,
        data: NDArrayAny | xr.DataArray | None = None,
        *,
        copy: bool | None = False,
        order: ArrayOrder = None,
        verify: bool = False,
        dtype: DTypeLike = None,
    ) -> xCentralMoments[Any] | Self:
        xdata: xr.DataArray
        if data is None:
            xdata = cast(  # type: ignore[redundant-cast] # needed by pyright...
                xr.DataArray,
                xr.zeros_like(
                    self._xdata,
                    dtype=dtype or self.dtype,  # type: ignore[arg-type]
                ),
            )  # pyright: ignore[reportCallIssue, reportArgumentType]
            copy = False
        else:
            if isinstance(data, xr.DataArray):
                xdata = data
            else:
                xdata = self._xdata.copy(data=data)

            shape: tuple[int, ...] = xdata.shape
            if shape[-self.mom_ndim :] != self.mom_shape:
                # at a minimum, verify that mom_shape is unchanged.
                msg = f"{shape=} has wrong mom_shape={self.mom_shape}"
                raise ValueError(msg)
            if verify and shape != self.shape:
                msg = f"{shape=} != {self.shape=}"
                raise ValueError(msg)

        return type(self)(
            data=xdata,
            mom_ndim=self._mom_ndim,
            copy=copy,
            order=order,
            dtype=dtype,
        )

    @overload
    def astype(
        self,
        dtype: DTypeLikeArg[FloatT2],
        *,
        order: ArrayOrder = None,
        casting: DataCasting = None,
        subok: bool | None = None,
        copy: bool = False,
    ) -> xCentralMoments[FloatT2]: ...
    @overload
    def astype(
        self,
        dtype: None,
        *,
        order: ArrayOrder = None,
        casting: DataCasting = None,
        subok: bool | None = None,
        copy: bool = False,
    ) -> xCentralMoments[np.float64]: ...
    @overload
    def astype(
        self,
        dtype: DTypeLike,
        *,
        order: ArrayOrder = None,
        casting: DataCasting = None,
        subok: bool | None = None,
        copy: bool = False,
    ) -> xCentralMoments[Any]: ...

    @docfiller_abc()
    def astype(
        self,
        dtype: DTypeLike,
        *,
        order: ArrayOrder = None,
        casting: DataCasting = None,
        subok: bool | None = None,
        copy: bool = False,
    ) -> xCentralMoments[Any]:
        return super().astype(
            dtype=dtype, order=order, casting=casting, subok=subok, copy=copy
        )

    @docfiller_inherit_abc()
    def moments_to_comoments(
        self,
        *,
        mom: tuple[int, int],
        mom_dims: MomDims | None = None,
        keep_attrs: KeepAttrs = None,
    ) -> Self:
        """
        Convert moments (mom_ndim=1) to comoments (mom_ndim=2).

        Parameters
        ----------
        {mom_dims}
        {keep_attrs}

        See Also
        --------
        .convert.moments_to_comoments
        """
        self._raise_if_not_mom_ndim_1()

        from . import convert

        return type(self)(
            convert.moments_to_comoments(
                self.to_values(),
                mom=mom,
                dtype=self.dtype,
                mom_dims=mom_dims,
                keep_attrs=keep_attrs,
            ),
            mom_ndim=2,
        )

    @docfiller_inherit_abc()
    def assign_weight(
        self, weight: ArrayLike | xr.DataArray, copy: bool = True
    ) -> Self:
        return super().assign_weight(weight=weight, copy=copy)

    # ** Access to underlying statistics ------------------------------------------
    # TODO(wpk): add overload
    def _single_index_selector(
        self,
        val: int,
        dim_combined: str = "variable",
    ) -> dict[Hashable, Any]:
        idxs = self._single_index(val)[-self.mom_ndim :]
        return {
            dim: (idx if self._mom_ndim == 1 else xr.DataArray(idx, dims=dim_combined))
            for dim, idx in zip(self.mom_dims, idxs)
        }

    def _single_index_dataarray(
        self,
        val: int,
        dim_combined: str = "variable",
        coords_combined: str | Sequence[Hashable] | None = None,
    ) -> xr.DataArray:
        if coords_combined is None:
            coords_combined = self.mom_dims

        selector = self._single_index_selector(
            val=val,
            dim_combined=dim_combined,
        )

        out = self.to_dataarray().isel(selector)
        if self._mom_ndim > 1:
            out = out.assign_coords(coords={dim_combined: list(coords_combined)})  # pyright: ignore[reportUnknownMemberType]
        return out

    def mean(
        self,
        dim_combined: str = "variable",
        coords_combined: str | Sequence[Hashable] | None = None,
    ) -> xr.DataArray:
        """Return mean/first moment(s) of data."""
        return self._single_index_dataarray(
            val=1, dim_combined=dim_combined, coords_combined=coords_combined
        )

    def var(
        self,
        dim_combined: str = "variable",
        coords_combined: str | Sequence[Hashable] | None = None,
    ) -> xr.DataArray:
        """Return variance (second central moment) of data."""
        return self._single_index_dataarray(
            val=2, dim_combined=dim_combined, coords_combined=coords_combined
        )

    # ** xarray specific methods --------------------------------------------------
    def _wrap_xarray_method(self, _method: str, *args: Any, **kwargs: Any) -> Self:
        xdata = getattr(self._xdata, _method)(*args, **kwargs)
        return self.new_like(data=xdata)  # type: ignore[no-any-return]

    def assign_coords(
        self, coords: XArrayCoordsType = None, **coords_kwargs: Any
    ) -> Self:
        """Assign coordinates to data and return new object."""
        return self._wrap_xarray_method("assign_coords", coords=coords, **coords_kwargs)

    def assign_attrs(self, *args: Any, **kwargs: Any) -> Self:
        """Assign attributes to data and return new object."""
        return self._wrap_xarray_method("assign_attrs", *args, **kwargs)

    def rename(
        self,
        new_name_or_name_dict: Hashable | Mapping[Any, Hashable] | None = None,
        **names: Hashable,
    ) -> Self:
        """Rename object."""
        return self._wrap_xarray_method(
            "rename", new_name_or_name_dict=new_name_or_name_dict, **names
        )

    def stack(
        self,
        dimensions: Mapping[Any, Sequence[Hashable]] | None = None,
        *,
        _reorder: bool = True,
        _copy: bool | None = False,
        _order: ArrayOrder = None,
        _verify: bool = False,
        **dimensions_kwargs: Any,
    ) -> Self:
        """
        Stack dimensions.

        Returns
        -------
        output : xCentralMoments
            With dimensions stacked.

        See Also
        --------
        pipe
        xarray.DataArray.stack

        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.random.default_rng(0)
        >>> vals = xr.DataArray(rng.random((10, 2, 3)), dims=["rec", "dim_0", "dim_1"])
        >>> da = xCentralMoments.from_vals(vals, mom=2, dim="rec")
        >>> da
        <xCentralMoments(val_shape=(2, 3), mom=(2,))>
        <xarray.DataArray (dim_0: 2, dim_1: 3, mom_0: 3)> Size: 144B
        array([[[10.    ,  0.5205,  0.0452],
                [10.    ,  0.4438,  0.0734],
                [10.    ,  0.5038,  0.1153]],
        <BLANKLINE>
               [[10.    ,  0.5238,  0.1272],
                [10.    ,  0.628 ,  0.0524],
                [10.    ,  0.412 ,  0.0865]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0
        >>> da_stack = da.stack(z=["dim_0", "dim_1"])
        >>> da_stack
        <xCentralMoments(val_shape=(6,), mom=(2,))>
        <xarray.DataArray (z: 6, mom_0: 3)> Size: 144B
        array([[10.    ,  0.5205,  0.0452],
               [10.    ,  0.4438,  0.0734],
               [10.    ,  0.5038,  0.1153],
               [10.    ,  0.5238,  0.1272],
               [10.    ,  0.628 ,  0.0524],
               [10.    ,  0.412 ,  0.0865]])
        Coordinates:
          * z        (z) object 48B MultiIndex
          * dim_0    (z) int64 48B 0 0 0 1 1 1
          * dim_1    (z) int64 48B 0 1 2 0 1 2
        Dimensions without coordinates: mom_0

        And unstack

        >>> da_stack.unstack("z")
        <xCentralMoments(val_shape=(2, 3), mom=(2,))>
        <xarray.DataArray (dim_0: 2, dim_1: 3, mom_0: 3)> Size: 144B
        array([[[10.    ,  0.5205,  0.0452],
                [10.    ,  0.4438,  0.0734],
                [10.    ,  0.5038,  0.1153]],
        <BLANKLINE>
               [[10.    ,  0.5238,  0.1272],
                [10.    ,  0.628 ,  0.0524],
                [10.    ,  0.412 ,  0.0865]]])
        Coordinates:
          * dim_0    (dim_0) int64 16B 0 1
          * dim_1    (dim_1) int64 24B 0 1 2
        Dimensions without coordinates: mom_0
        """
        return self.pipe(
            "stack",
            dimensions=dimensions,
            _reorder=_reorder,
            _order=_order,
            _copy=_copy,
            _verify=_verify,
            **dimensions_kwargs,
        )

    def unstack(
        self,
        dim: Dims = None,
        fill_value: Any = np.nan,
        *,
        sparse: bool = False,
        _reorder: bool = True,
        _copy: bool | None = False,
        _order: ArrayOrder = None,
        _verify: bool = False,
    ) -> Self:
        """
        Unstack dimensions.

        Returns
        -------
        output : xCentralMoments
            With dimensions unstacked

        See Also
        --------
        stack
        xarray.DataArray.unstack

        """
        return self.pipe(
            "unstack",
            _reorder=_reorder,
            _copy=_copy,
            _order=_order,
            _verify=_verify,
            dim=dim,
            fill_value=fill_value,
            sparse=sparse,
        )

    def set_index(
        self,
        indexes: Mapping[Any, Hashable | Sequence[Hashable]] | None = None,
        append: bool = False,
        *,
        _reorder: bool = True,
        _copy: bool | None = False,
        _order: ArrayOrder = None,
        _verify: bool = False,
        **indexes_kwargs: Hashable | Sequence[Hashable],
    ) -> Self:
        """
        Interface to :meth:`xarray.DataArray.set_index`

        Returns
        -------
        output : xCentralMoments
            With new index.

        See Also
        --------
        stack
        unstack
        reset_index
        """
        return self.pipe(
            "set_index",
            _reorder=_reorder,
            _copy=_copy,
            _order=_order,
            _verify=_verify,
            indexes=indexes,
            append=append,
            **indexes_kwargs,
        )

    def reset_index(
        self,
        dims_or_levels: Hashable | Sequence[Hashable],
        drop: bool = False,
        *,
        _reorder: bool = True,
        _copy: bool | None = False,
        _order: ArrayOrder = None,
        _verify: bool = False,
    ) -> Self:
        """Interface to :meth:`xarray.DataArray.reset_index`."""
        return self.pipe(
            "reset_index",
            _reorder=_reorder,
            _copy=_copy,
            _order=_order,
            _verify=_verify,
            dims_or_levels=dims_or_levels,
            drop=drop,
        )

    def drop_vars(
        self,
        names: str | Iterable[Hashable] | Callable[[Self], str | Iterable[Hashable]],
        *,
        errors: Literal["raise", "ignore"] = "raise",
        _reorder: bool = True,
        _copy: bool | None = False,
        _order: ArrayOrder = None,
        _verify: bool = False,
    ) -> Self:
        """Interface to :meth:`xarray.DataArray.drop_vars`"""
        return self.pipe(
            "drop_vars",
            names=names,
            errors=errors,
            _reorder=_reorder,
            _copy=_copy,
            _order=_order,
            _verify=_verify,
        )

    def swap_dims(
        self,
        dims_dict: Mapping[Any, Hashable] | None = None,
        _reorder: bool = True,
        _copy: bool | None = False,
        _order: ArrayOrder = None,
        _verify: bool = False,
        **dims_kwargs: Any,
    ) -> Self:
        """Interface to :meth:`xarray.DataArray.swap_dims`."""
        return self.pipe(
            "swap_dims",
            dims_dict=dims_dict,
            _reorder=_reorder,
            _copy=_copy,
            _order=_order,
            _verify=_verify,
            **dims_kwargs,
        )

    def sel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        method: str | None = None,
        tolerance: Any = None,
        drop: bool = False,
        *,
        _reorder: bool = False,
        _copy: bool | None = False,
        _order: ArrayOrder = None,
        _verify: bool = False,
        **indexers_kws: Any,
    ) -> Self:
        """
        Select subset of data.

        Returns
        -------
        output : xCentralMoments
            With dimensions unstacked

        See Also
        --------
        xarray.DataArray.sel

        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.random.default_rng(0)
        >>> da = cmomy.CentralMoments.from_vals(
        ...     rng.random((10, 3)), axis=0, mom=2
        ... ).to_x(dims="x", coords=dict(x=list("abc")))
        >>> da
        <xCentralMoments(val_shape=(3,), mom=(2,))>
        <xarray.DataArray (x: 3, mom_0: 3)> Size: 72B
        array([[10.    ,  0.5248,  0.1106],
               [10.    ,  0.5688,  0.0689],
               [10.    ,  0.5094,  0.1198]])
        Coordinates:
          * x        (x) <U1 12B 'a' 'b' 'c'
        Dimensions without coordinates: mom_0

        Select by value

        >>> da.sel(x="a")
        <xCentralMoments(val_shape=(), mom=(2,))>
        <xarray.DataArray (mom_0: 3)> Size: 24B
        array([10.    ,  0.5248,  0.1106])
        Coordinates:
            x        <U1 4B 'a'
        Dimensions without coordinates: mom_0
        >>> da.sel(x=["a", "c"])
        <xCentralMoments(val_shape=(2,), mom=(2,))>
        <xarray.DataArray (x: 2, mom_0: 3)> Size: 48B
        array([[10.    ,  0.5248,  0.1106],
               [10.    ,  0.5094,  0.1198]])
        Coordinates:
          * x        (x) <U1 8B 'a' 'c'
        Dimensions without coordinates: mom_0


        Select by position

        >>> da.isel(x=0)
        <xCentralMoments(val_shape=(), mom=(2,))>
        <xarray.DataArray (mom_0: 3)> Size: 24B
        array([10.    ,  0.5248,  0.1106])
        Coordinates:
            x        <U1 4B 'a'
        Dimensions without coordinates: mom_0
        >>> da.isel(x=[0, 1])
        <xCentralMoments(val_shape=(2,), mom=(2,))>
        <xarray.DataArray (x: 2, mom_0: 3)> Size: 48B
        array([[10.    ,  0.5248,  0.1106],
               [10.    ,  0.5688,  0.0689]])
        Coordinates:
          * x        (x) <U1 8B 'a' 'b'
        Dimensions without coordinates: mom_0

        """
        return self.pipe(
            "sel",
            _reorder=_reorder,
            _copy=_copy,
            _order=_order,
            _verify=_verify,
            indexers=indexers,
            method=method,
            tolerance=tolerance,
            drop=drop,
            **indexers_kws,
        )

    def isel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        drop: bool = False,
        missing_dims: xr_types.ErrorOptionsWithWarn = "raise",
        *,
        _reorder: bool = False,
        _copy: bool | None = False,
        _order: ArrayOrder = None,
        _verify: bool = False,
        **indexers_kws: Any,
    ) -> Self:
        """
        Select subset of data by position.

        Returns
        -------
        output : xCentralMoments
            With dimensions unstacked

        See Also
        --------
        sel
        xarray.DataArray.isel
        """
        return self.pipe(
            "isel",
            _reorder=_reorder,
            _copy=_copy,
            _order=_order,
            _verify=_verify,
            indexers=indexers,
            drop=drop,
            missing_dims=missing_dims,
            **indexers_kws,
        )

    def transpose(
        self,
        *dims: Hashable,
        transpose_coords: bool = True,
        missing_dims: xr_types.ErrorOptionsWithWarn = "raise",
        _copy: bool | None = False,
        _order: ArrayOrder = None,
        _verify: bool = False,
    ) -> Self:
        """
        Transpose dimensions of data.

        Notes
        -----
        ``mom_dims`` will always be put at the end of the array.

        See Also
        --------
        pipe
        xarray.DataArray.transpose
        """
        # make sure dims are last

        dims = tuple(d for d in dims if d not in self.mom_dims) + self.mom_dims

        return self.pipe(
            "transpose",
            *dims,
            transpose_coords=transpose_coords,
            missing_dims=missing_dims,
            _reorder=False,
            _copy=_copy,
            _order=_order,
            _verify=_verify,
        )

    # ** To/from CentralMoments
    def to_centralmoments(self, copy: bool = False) -> CentralMoments[FloatT]:
        """Create a CentralMoments object from xCentralMoments."""
        from ._central_numpy import CentralMoments

        return CentralMoments(
            data=self._data.copy() if copy else self._data,
            mom_ndim=self.mom_ndim,
            fastpath=True,
        )

    def to_c(self, copy: bool = False) -> CentralMoments[FloatT]:
        """Alias to :meth:`to_centralmoments`"""
        return self.to_centralmoments(copy=copy)

    # @cached.prop
    # def centralmoments_view(self) -> CentralMoments[FloatT]:
    #     """
    #     Create CentralMoments view.

    #     This object has the same underlying data as `self`, but no
    #     DataArray attributes.  Useful for some function calls.

    #     See Also
    #     --------
    #     CentralMoments.to_xcentralmoments
    #     xCentralMoments.from_centralmoments
    #     """
    #     return self._to_centralmoments(copy=False)

    @classmethod
    @docfiller.decorate
    def from_centralmoments(
        cls,
        obj: CentralMoments[FloatT2],
        *,
        dims: XArrayDimsType = None,
        attrs: XArrayAttrsType = None,
        coords: XArrayCoordsType = None,
        name: XArrayNameType = None,
        indexes: XArrayIndexesType = None,
        mom_dims: MomDims | None = None,
        template: xr.DataArray | None = None,
        copy: bool = False,
    ) -> xCentralMoments[FloatT2]:
        """
        Create and xCentralMoments object from CentralMoments.

        Parameters
        ----------
        obj : CentralMoments
            Input object to be converted.
        {xr_params}
        {copy}

        Returns
        -------
        output : xCentralMoments

        See Also
        --------
        CentralMoments.to_xcentralmoments
        """
        return obj.to_xcentralmoments(
            dims=dims,
            attrs=attrs,
            coords=coords,
            name=name,
            indexes=indexes,
            mom_dims=mom_dims,
            template=template,
            copy=copy,
        )

    # ** Pushing --------------------------------------------------------------
    def _push_data_dataarray(
        self,
        data: xr.DataArray,
        *,
        order: ArrayOrder = None,
        parallel: bool | None = None,
    ) -> Self:
        def func(_out: NDArrayAny, data: NDArrayAny) -> NDArrayAny:
            if order:
                data = np.asarray(data, order=order)
            self._pusher(parallel).data(data, self._data)
            return self._data

        # NOTE:  Use this pattern here and below
        # pass self._xdata to func for broadcasting/alignment.
        # But then will use `self._data` as is for `out` parameter.
        # Also, put `out` first so things are transposed relative to that...
        _ = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            func,
            self._xdata,
            data,
            input_core_dims=[self.mom_dims, self.mom_dims],
            output_core_dims=[self.mom_dims],
        )
        return self

    def _push_datas_dataarray(
        self,
        datas: xr.DataArray,
        *,
        axis: AxisReduce | MissingType,
        dim: DimsReduce | MissingType,
        parallel: bool | None,
        order: ArrayOrder,
    ) -> Self:
        def func(_out: NDArrayAny, _datas: NDArrayAny) -> NDArrayAny:
            self._pusher(parallel).datas(_datas, self._data)
            return self._data

        dim, datas = xprepare_data_for_reduction(
            data=datas,
            axis=axis,
            dim=dim,
            mom_ndim=self.mom_ndim,
            order=order,
            dtype=self.dtype,
        )

        datas_dims = [*self.val_dims, dim, *self.mom_dims]
        _ = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            func,
            self._xdata,
            datas,
            input_core_dims=[self.dims, datas_dims],
            output_core_dims=[self.dims],
        )

        return self

    def _push_val_dataarray(
        self,
        x: xr.DataArray,
        *y: ArrayLike,
        weight: ArrayLike | None = None,
        order: ArrayOrder = None,  # noqa: ARG002
        parallel: bool | None = None,
    ) -> Self:
        self._check_y(y, self.mom_ndim)

        def func(
            _out: NDArrayAny, x0: NDArrayAny, w: NDArrayAny, *x1: NDArrayAny
        ) -> NDArrayAny:
            self._pusher(parallel).val(x0, *x1, w, self._data)
            return self._data

        weight = 1.0 if weight is None else weight
        core_dims: list[Any] = [self.mom_dims, *([[]] * (2 + len(y)))]
        _ = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            func,
            self._xdata,
            x,
            weight,
            *y,
            input_core_dims=core_dims,
            output_core_dims=[self.mom_dims],
        )
        return self

    def _push_vals_dataarray(
        self,
        x: xr.DataArray,
        *y: ArrayLike,
        weight: ArrayLike | None = None,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        order: ArrayOrder = None,
        parallel: bool | None = None,
    ) -> Self:
        self._check_y(y, self.mom_ndim)

        weight = 1.0 if weight is None else weight

        input_core_dims, (x0, w, *x1) = xprepare_values_for_reduction(
            x,
            weight,
            *y,
            axis=axis,
            dim=dim,
            dtype=self.dtype,
            order=order,
            narrays=self.mom_ndim + 1,
        )

        def func(
            _out: NDArrayAny, x0: NDArrayAny, w: NDArrayAny, *x1: NDArrayAny
        ) -> NDArrayAny:
            self._pusher(parallel).vals(x0, *x1, w, self._data)
            return self._data

        _ = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            func,
            self._xdata,
            x0,
            w,
            *x1,
            input_core_dims=[self.mom_dims, *input_core_dims],  # type: ignore[has-type]
            output_core_dims=[self.mom_dims],
        )
        return self

    @docfiller_inherit_abc()
    def push_data(
        self,
        data: NDArray[FloatT] | xr.DataArray,
        *,
        order: ArrayOrder = None,
        parallel: bool | None = False,
    ) -> Self:
        if isinstance(data, xr.DataArray):
            return self._push_data_dataarray(data, order=order, parallel=parallel)
        return self._push_data_numpy(data, order=order, parallel=parallel)

    @docfiller_inherit_abc()
    def push_datas(
        self,
        datas: NDArray[FloatT] | xr.DataArray,
        *,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        order: ArrayOrder = None,
        parallel: bool | None = None,
    ) -> Self:
        if isinstance(datas, xr.DataArray):
            return self._push_datas_dataarray(
                datas, axis=axis, dim=dim, order=order, parallel=parallel
            )
        return self._push_datas_numpy(datas, axis=axis, order=order, parallel=parallel)

    @docfiller_inherit_abc()
    def push_val(
        self,
        x: ArrayLike | xr.DataArray,
        *y: ArrayLike | xr.DataArray,
        weight: ArrayLike | xr.DataArray | None = None,
        order: ArrayOrder = None,
        parallel: bool | None = False,
    ) -> Self:
        if isinstance(x, xr.DataArray):
            return self._push_val_dataarray(
                x, *y, weight=weight, order=order, parallel=parallel
            )

        return self._push_val_numpy(
            x, *y, weight=weight, order=order, parallel=parallel
        )

    @docfiller_inherit_abc()
    def push_vals(
        self,
        x: ArrayLike | xr.DataArray,
        *y: ArrayLike | xr.DataArray,
        weight: ArrayLike | xr.DataArray | None = None,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        order: ArrayOrder = None,
        parallel: bool | None = None,
    ) -> Self:
        if isinstance(x, xr.DataArray):
            return self._push_vals_dataarray(
                x,
                *y,
                weight=weight,
                axis=axis,
                dim=dim,
                order=order,
                parallel=parallel,
            )

        return self._push_vals_numpy(
            x, *y, weight=weight, axis=axis, order=order, parallel=parallel
        )

    # ** Manipulation
    # ** Reduction -----------------------------------------------------------
    @docfiller_inherit_abc()
    def randsamp_freq(
        self,
        *,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        nrep: int | None = None,
        nsamp: int | None = None,
        indices: ArrayLike | None = None,
        freq: ArrayLike | None = None,
        check: bool = False,
        rng: np.random.Generator | None = None,
    ) -> NDArrayInt:
        """
        Parameters
        ----------
        dim : str
            Dimension that will be resampled.
        """
        from .resample import randsamp_freq

        return randsamp_freq(
            data=self.values,
            mom_ndim=self._mom_ndim,
            axis=axis,
            dim=dim,
            nrep=nrep,
            nsamp=nsamp,
            indices=indices,
            freq=freq,
            check=check,
            rng=rng,
        )

    @docfiller_inherit_abc()
    def resample_and_reduce(
        self,
        *,
        freq: NDArrayInt,
        dim: DimsReduce | MissingType = MISSING,
        axis: AxisReduce | MissingType = MISSING,
        rep_dim: str = "rep",
        parallel: bool | None = None,
        order: ArrayOrder = None,
        keep_attrs: KeepAttrs = None,
    ) -> Self:
        """
        Parameters
        ----------
        {rep_dim}
        {keep_attrs}

        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.random.default_rng(0)
        >>> da = cmomy.CentralMoments.from_vals(
        ...     rng.random((10, 3)),
        ...     mom=3,
        ...     axis=0,
        ... ).to_x(dims="rec")
        >>> da
        <xCentralMoments(val_shape=(3,), mom=(3,))>
        <xarray.DataArray (rec: 3, mom_0: 4)> Size: 96B
        array([[ 1.0000e+01,  5.2485e-01,  1.1057e-01, -4.6282e-03],
               [ 1.0000e+01,  5.6877e-01,  6.8876e-02, -1.2745e-02],
               [ 1.0000e+01,  5.0944e-01,  1.1978e-01, -1.4644e-02]])
        Dimensions without coordinates: rec, mom_0

        Note that for reproducible results, must set numba random
        seed as well

        >>> freq = da.randsamp_freq(dim="rec", nrep=5)
        >>> da_resamp = da.resample_and_reduce(
        ...     dim="rec",
        ...     freq=freq,
        ... )
        >>> da_resamp
        <xCentralMoments(val_shape=(5,), mom=(3,))>
        <xarray.DataArray (rep: 5, mom_0: 4)> Size: 160B
        array([[ 3.0000e+01,  5.0944e-01,  1.1978e-01, -1.4644e-02],
               [ 3.0000e+01,  5.3435e-01,  1.0038e-01, -1.2329e-02],
               [ 3.0000e+01,  5.2922e-01,  1.0360e-01, -1.6009e-02],
               [ 3.0000e+01,  5.5413e-01,  8.3204e-02, -1.1267e-02],
               [ 3.0000e+01,  5.4899e-01,  8.6627e-02, -1.5407e-02]])
        Dimensions without coordinates: rep, mom_0

        Alternatively, we can resample and reduce

        >>> indices = cmomy.resample.freq_to_indices(freq)
        >>> da.sel(rec=xr.DataArray(indices, dims=["rep", "rec"])).reduce(dim="rec")
        <xCentralMoments(val_shape=(5,), mom=(3,))>
        <xarray.DataArray (rep: 5, mom_0: 4)> Size: 160B
        array([[ 3.0000e+01,  5.0944e-01,  1.1978e-01, -1.4644e-02],
               [ 3.0000e+01,  5.3435e-01,  1.0038e-01, -1.2329e-02],
               [ 3.0000e+01,  5.2922e-01,  1.0360e-01, -1.6009e-02],
               [ 3.0000e+01,  5.5413e-01,  8.3204e-02, -1.1267e-02],
               [ 3.0000e+01,  5.4899e-01,  8.6627e-02, -1.5407e-02]])
        Dimensions without coordinates: rep, mom_0

        """
        self._raise_if_scalar()
        from .resample import resample_data

        data = resample_data(
            self._xdata,
            freq=freq,
            mom_ndim=self.mom_ndim,
            axis=axis,
            dim=dim,
            rep_dim=rep_dim,
            order=order,
            parallel=parallel,
            dtype=self.dtype,
            keep_attrs=keep_attrs,
        )

        return type(self)(data=data, mom_ndim=self.mom_ndim)

    @docfiller_inherit_abc()
    def reduce(
        self,
        *,
        by: str | Groups | None = None,
        axis: AxisReduce | MissingType = MISSING,
        order: ArrayOrder = None,
        parallel: bool | None = None,
        # xarray specific
        coords_policy: CoordsPolicy = "first",
        dim: DimsReduce | MissingType = MISSING,
        group_dim: str | None = None,
        groups: Groups | None = None,
        keep_attrs: KeepAttrs = None,
    ) -> Self:
        """
        Parameters
        ----------
        by : ndarray or DataArray or str or iterable of str, optional
            If ``None``, reduce over entire ``dim``. Otherwise, reduce by
            group. If :class:`~numpy.ndarray`, use the unique values. If
            :class:`~xarray.DataArray`, use unique values and rename dimension
            ``by.name``. If str or Iterable of str, Create grouper from these
            named coordinates.
        {coords_policy}
        {group_dim}
        {groups}
        {keep_attrs}


        Notes
        -----
        This is a new feature, and subject to change. In particular, the
        current implementation is not smart about the output coordinates and
        dimensions, and is inconsistent with :meth:`xarray.DataArray.groupby`.
        It is up the the user to manipulate the dimensions/coordinates. output
        dimensions and coords simplistically.

        See Also
        --------
        .reduction.reduce_data
        .reduction.reduce_data_grouped
        .reduction.reduce_data_indexed

        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.random.default_rng(0)
        >>> vals = xr.DataArray(rng.random((10, 2, 3)), dims=["rec", "dim_0", "dim_1"])
        >>> da = xCentralMoments.from_vals(vals, dim="rec", mom=2)
        >>> da.reduce(dim="dim_0")
        <xCentralMoments(val_shape=(3,), mom=(2,))>
        <xarray.DataArray (dim_1: 3, mom_0: 3)> Size: 72B
        array([[20.    ,  0.5221,  0.0862],
               [20.    ,  0.5359,  0.0714],
               [20.    ,  0.4579,  0.103 ]])
        Dimensions without coordinates: dim_1, mom_0
        """
        self._raise_if_scalar()
        if by is None:
            from .reduction import reduce_data

            return type(self)(
                data=reduce_data(
                    self._xdata,
                    mom_ndim=self._mom_ndim,
                    axis=axis,
                    dim=dim,
                    order=order,
                    parallel=parallel,
                    keep_attrs=keep_attrs,
                    dtype=self.dtype,
                ),
                mom_ndim=self._mom_ndim,
                fastpath=True,
            )

        if coords_policy == "group":
            from .reduction import reduce_data_grouped

            codes: ArrayLike
            if isinstance(by, str):
                from .reduction import factor_by

                _groups, codes = factor_by(self._xdata[by].to_numpy())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                # if groups is None:  # have to exclicitly pass groups...
                #     groups = _groups
            else:
                codes = by

            data = reduce_data_grouped(
                self._xdata,
                mom_ndim=self._mom_ndim,
                by=codes,
                axis=axis,
                dim=dim,
                order=order,
                parallel=parallel,
                dtype=self.dtype,
                group_dim=group_dim,
                groups=groups,
                keep_attrs=keep_attrs,
            )

        else:
            from .reduction import factor_by_to_index, reduce_data_indexed

            if isinstance(by, str):
                _groups, index, group_start, group_end = factor_by_to_index(
                    self._xdata[  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                        by
                    ].to_numpy()  # indexes[by] if by in self.indexes else self._xdata[by]
                )
            else:
                _groups, index, group_start, group_end = factor_by_to_index(by)

            data = reduce_data_indexed(
                self._xdata,
                mom_ndim=self._mom_ndim,
                index=index,
                group_start=group_start,
                group_end=group_end,
                axis=axis,
                dim=dim,
                order=order,
                parallel=parallel,
                dtype=self.dtype,
                coords_policy=coords_policy,
                group_dim=group_dim,
                groups=groups,
                keep_attrs=keep_attrs,
            )

        return type(self)(data=data, mom_ndim=self._mom_ndim, fastpath=True)

    @docfiller.decorate
    def block(
        self,
        block_size: int | None,
        *,
        dim: DimsReduce | MissingType = MISSING,
        axis: AxisReduce | MissingType = MISSING,
        block_dim: str | None = None,
        order: ArrayOrder = None,
        parallel: bool | None = None,
        keep_attrs: KeepAttrs = None,
        coords_policy: CoordsPolicy = "first",
    ) -> Self:
        """
        Parameters
        ----------
        block_size : int
            Number of observations to include in a given block.
        {dim}
        {axis}
        block_dim : str, optional
            Name of blocked dimension.  Defaults to ``dim``.
        {order}
        {parallel}
        {keep_attrs}
        {coords_policy}

        Returns
        -------
        output : xCentralMoments
            Object with block averaging.

        See Also
        --------
        CentralMoments.block
        reduce


        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.random.default_rng(0)
        >>> x = rng.random((10, 10))
        >>> da = cmomy.CentralMoments.from_vals(x, mom=2, axis=0).to_x()
        >>> da
        <xCentralMoments(val_shape=(10,), mom=(2,))>
        <xarray.DataArray (dim_0: 10, mom_0: 3)> Size: 240B
        array([[10.    ,  0.6247,  0.0583],
               [10.    ,  0.3938,  0.0933],
               [10.    ,  0.425 ,  0.1003],
               [10.    ,  0.5   ,  0.117 ],
               [10.    ,  0.5606,  0.0446],
               [10.    ,  0.5612,  0.0861],
               [10.    ,  0.531 ,  0.0731],
               [10.    ,  0.8403,  0.0233],
               [10.    ,  0.5097,  0.103 ],
               [10.    ,  0.5368,  0.085 ]])
        Dimensions without coordinates: dim_0, mom_0

        >>> da.block(block_size=5, dim="dim_0")
        <xCentralMoments(val_shape=(2,), mom=(2,))>
        <xarray.DataArray (dim_0: 2, mom_0: 3)> Size: 48B
        array([[50.    ,  0.5008,  0.0899],
               [50.    ,  0.5958,  0.0893]])
        Dimensions without coordinates: dim_0, mom_0

        This is equivalent to

        >>> cmomy.CentralMoments.from_vals(x.reshape(2, 50), mom=2, axis=1).to_x()
        <xCentralMoments(val_shape=(2,), mom=(2,))>
        <xarray.DataArray (dim_0: 2, mom_0: 3)> Size: 48B
        array([[50.    ,  0.5268,  0.0849],
               [50.    ,  0.5697,  0.0979]])
        Dimensions without coordinates: dim_0, mom_0


        The coordinate policy can be useful to keep coordinates:

        >>> da2 = da.assign_coords(dim_0=range(10))
        >>> da2.block(5, dim="dim_0", coords_policy="first")
        <xCentralMoments(val_shape=(2,), mom=(2,))>
        <xarray.DataArray (dim_0: 2, mom_0: 3)> Size: 48B
        array([[50.    ,  0.5008,  0.0899],
               [50.    ,  0.5958,  0.0893]])
        Coordinates:
          * dim_0    (dim_0) int64 16B 0 5
        Dimensions without coordinates: mom_0
        >>> da2.block(5, dim="dim_0", coords_policy="last")
        <xCentralMoments(val_shape=(2,), mom=(2,))>
        <xarray.DataArray (dim_0: 2, mom_0: 3)> Size: 48B
        array([[50.    ,  0.5008,  0.0899],
               [50.    ,  0.5958,  0.0893]])
        Coordinates:
          * dim_0    (dim_0) int64 16B 4 9
        Dimensions without coordinates: mom_0
        >>> da2.block(5, dim="dim_0", coords_policy=None)
        <xCentralMoments(val_shape=(2,), mom=(2,))>
        <xarray.DataArray (dim_0: 2, mom_0: 3)> Size: 48B
        array([[50.    ,  0.5008,  0.0899],
               [50.    ,  0.5958,  0.0893]])
        Dimensions without coordinates: dim_0, mom_0

        """
        axis, dim = select_axis_dim(
            dims=self.dims,
            axis=axis,
            dim=dim,
        )

        n = self.sizes[dim]
        if block_size is None:
            block_size = n
        nblock = n // block_size

        by = np.arange(nblock).repeat(block_size)
        if len(by) != n:
            by = np.pad(by, (0, n - len(by)), mode="constant", constant_values=-1)

        return self.reduce(
            dim=dim,
            by=by,
            order=order,
            parallel=parallel,
            group_dim=block_dim,
            keep_attrs=keep_attrs,
            coords_policy=coords_policy,
            groups=range(nblock),
        )

    # ** Constructors
    @overload  # type: ignore[override]
    @classmethod
    def zeros(  # type: ignore[overload-overlap]
        cls,
        *,
        mom: Moments,
        val_shape: tuple[int, ...] | int | None = ...,
        dtype: None = ...,
        order: ArrayOrderCF | None = ...,
        dims: XArrayDimsType = ...,
        mom_dims: MomDims | None = ...,
        attrs: XArrayAttrsType = ...,
        coords: XArrayCoordsType = ...,
        name: XArrayNameType = ...,
        indexes: XArrayIndexesType = ...,
        template: xr.DataArray | None = ...,
    ) -> xCentralMoments[np.float64]: ...
    @overload
    @classmethod
    def zeros(
        cls,
        *,
        mom: Moments,
        val_shape: tuple[int, ...] | int | None = ...,
        dtype: DTypeLikeArg[FloatT2],
        order: ArrayOrderCF | None = ...,
        dims: XArrayDimsType = ...,
        mom_dims: MomDims | None = ...,
        attrs: XArrayAttrsType = ...,
        coords: XArrayCoordsType = ...,
        name: XArrayNameType = ...,
        indexes: XArrayIndexesType = ...,
        template: xr.DataArray | None = ...,
    ) -> xCentralMoments[FloatT2]: ...
    @overload
    @classmethod
    def zeros(
        cls,
        *,
        mom: Moments,
        val_shape: tuple[int, ...] | int | None = ...,
        dtype: DTypeLike,
        order: ArrayOrderCF | None = ...,
        dims: XArrayDimsType = ...,
        mom_dims: MomDims | None = ...,
        attrs: XArrayAttrsType = ...,
        coords: XArrayCoordsType = ...,
        name: XArrayNameType = ...,
        indexes: XArrayIndexesType = ...,
        template: xr.DataArray | None = ...,
    ) -> Self: ...

    @classmethod
    @docfiller.decorate
    def zeros(
        cls,
        *,
        mom: Moments,
        val_shape: tuple[int, ...] | int | None = None,
        dtype: DTypeLike = None,
        order: ArrayOrderCF | None = None,
        dims: XArrayDimsType = None,
        mom_dims: MomDims | None = None,
        attrs: XArrayAttrsType = None,
        coords: XArrayCoordsType = None,
        name: XArrayNameType = None,
        indexes: XArrayIndexesType = None,
        template: xr.DataArray | None = None,
    ) -> xCentralMoments[Any] | Self:
        """
        Create a new base object.

        Parameters
        ----------
        {mom}
        {val_shape}
        {dtype}
        {order}
        {xr_params}

        Returns
        -------
        output : {klass}
            New instance with zero values.


        See Also
        --------
        numpy.zeros
        """
        from ._central_numpy import CentralMoments

        return CentralMoments.zeros(
            mom=mom,
            val_shape=val_shape,
            dtype=dtype,
            order=order,
        ).to_xcentralmoments(
            dims=dims,
            attrs=attrs,
            coords=coords,
            name=name,
            indexes=indexes,
            mom_dims=mom_dims,
            template=template,
        )

    @overload
    @classmethod
    def from_vals(
        cls,
        x: xr.DataArray,
        *y: ArrayLike | xr.DataArray,
        mom: Moments,
        weight: ArrayLike | xr.DataArray | None = ...,
        axis: AxisReduce | MissingType = ...,
        dim: DimsReduce | MissingType = ...,
        mom_dims: MomDims | None = ...,
        order: ArrayOrder = ...,
        parallel: bool | None = ...,
        dtype: DTypeLikeArg[FloatT2],
        keep_attrs: KeepAttrs = ...,
    ) -> xCentralMoments[FloatT2]: ...
    @overload
    @classmethod
    def from_vals(
        cls,
        x: xr.DataArray,
        *y: ArrayLike | xr.DataArray,
        mom: Moments,
        weight: ArrayLike | xr.DataArray | None = ...,
        axis: AxisReduce | MissingType = ...,
        dim: DimsReduce | MissingType = ...,
        mom_dims: MomDims | None = ...,
        order: ArrayOrder = ...,
        parallel: bool | None = ...,
        dtype: DTypeLike = ...,
        keep_attrs: KeepAttrs = ...,
    ) -> Self: ...

    @classmethod
    @docfiller.decorate
    def from_vals(
        cls,
        x: xr.DataArray,
        *y: ArrayLike | xr.DataArray,
        mom: Moments,
        weight: ArrayLike | xr.DataArray | None = None,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        mom_dims: MomDims | None = None,
        order: ArrayOrder = None,
        parallel: bool | None = None,
        dtype: DTypeLike = None,
        keep_attrs: KeepAttrs = None,
    ) -> Self | xCentralMoments[Any]:
        """
        Create from observations/values.

        Parameters
        ----------
        x : DataArray
            Values to reduce.
        *y : array-like or DataArray
            Additional values (needed if ``len(mom)==2``).
        weight : scalar or array-like or DataArray, optional
            Optional weight.  If scalar or array, attempt to
            broadcast to `x0.shape`
        {axis_and_dim}
        {mom}
        {order}
        {parallel}
        {dtype}
        {mom_dims}
        {keep_attrs}

        Returns
        -------
        output: {klass}

        See Also
        --------
        push_vals
        CentralMoments.from_vals
        CentralMoments.to_x
        """
        from .reduction import reduce_vals

        mom_strict, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)

        return cls(
            data=reduce_vals(
                x,
                *y,
                mom=mom_strict,
                weight=weight,
                axis=axis,
                dim=dim,
                mom_dims=mom_dims,
                order=order,
                parallel=parallel,
                dtype=dtype,
                keep_attrs=keep_attrs,
            ),
            mom_ndim=mom_ndim,
        )

    @overload
    @classmethod
    def from_resample_vals(
        cls,
        x: xr.DataArray,
        *y: ArrayLike | xr.DataArray,
        mom: Moments,
        freq: NDArrayInt,
        weight: ArrayLike | xr.DataArray | None = ...,
        axis: AxisReduce | MissingType = ...,
        dim: DimsReduce | MissingType = ...,
        order: ArrayOrder = ...,
        parallel: bool | None = ...,
        dtype: DTypeLikeArg[FloatT2],
        mom_dims: MomDims | None = ...,
        rep_dim: str = ...,
        keep_attrs: bool = ...,
    ) -> xCentralMoments[FloatT2]: ...
    @overload
    @classmethod
    def from_resample_vals(
        cls,
        x: xr.DataArray,
        *y: ArrayLike | xr.DataArray,
        mom: Moments,
        freq: NDArrayInt,
        weight: ArrayLike | xr.DataArray | None = ...,
        axis: AxisReduce | MissingType = ...,
        dim: DimsReduce | MissingType = ...,
        order: ArrayOrder = ...,
        parallel: bool | None = ...,
        dtype: DTypeLike = ...,
        mom_dims: MomDims | None = ...,
        rep_dim: str = ...,
        keep_attrs: bool = ...,
    ) -> Self: ...

    @classmethod
    @docfiller.decorate
    def from_resample_vals(
        cls,
        x: xr.DataArray,
        *y: ArrayLike | xr.DataArray,
        mom: Moments,
        freq: NDArrayInt,
        weight: ArrayLike | xr.DataArray | None = None,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        order: ArrayOrder = None,
        parallel: bool | None = None,
        dtype: DTypeLike = None,
        mom_dims: MomDims | None = None,
        rep_dim: str = "rep",
        keep_attrs: bool = True,
    ) -> xCentralMoments[Any] | Self:
        """
        Create from resample observations/values.

        This effectively resamples `x`.

        Parameters
        ----------
        x : DataArray
            For moments, pass single array-like objects `x=x0`.
            For comoments, pass tuple of array-like objects `x=(x0, x1)`.
        *y : array-like or DataArray
            Additional values (needed if ``len(mom) > 1``).
        {mom}
        {freq}
        {weight}
        {axis_and_dim}
        {full_output}
        {order}
        {parallel}
        {dtype}

        Returns
        -------
        out : {klass}
            Instance of calling class


        See Also
        --------
        cmomy.resample.resample_vals
        cmomy.resample.randsamp_freq
        cmomy.resample.freq_to_indices
        cmomy.resample.indices_to_freq
        """
        """
        Parameters
        ----------
        x : array, tuple of array, DataArray, or tuple of DataArray
            For moments, `x=x0`.  For comoments, `x=(x0, x1)`.
            If pass DataArray, inherit attributes from `x0`.  If pass
            ndarray, use `dims`, `attrs`, etc to wrap final result
        {dim}
        {rep_dim}
        {mom_dims}
        {keep_attrs}

        See Also
        --------
        CentralMoments.from_resample_vals
        .resample.resample_vals
        """
        mom_strict, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)

        from .resample import resample_vals

        return cls(
            data=resample_vals(
                x,
                *y,
                freq=freq,
                mom=mom_strict,
                weight=weight,
                axis=axis,
                dim=dim,
                order=order,
                parallel=parallel,
                mom_dims=mom_dims,
                rep_dim=rep_dim,
                keep_attrs=keep_attrs,
                dtype=dtype,
            ),
            mom_ndim=mom_ndim,
        )

    @classmethod
    @docfiller_abc()
    def from_raw(
        cls,
        raw: xr.DataArray,
        *,
        mom_ndim: Mom_NDim,
    ) -> Self:
        return super().from_raw(raw=raw, mom_ndim=mom_ndim)

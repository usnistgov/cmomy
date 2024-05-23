"""Thin wrapper around central routines with xarray support."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

# pandas needed for autdoc typehints
import pandas as pd  # noqa: F401  # pyright: ignore[reportUnusedImport]
import xarray as xr
from module_utilities import cached

from .central_abc import CentralMomentsABC
from .docstrings import docfiller_xcentral as docfiller
from .utils import (
    replace_coords_from_isel,
    select_axis_dim,
    validate_mom_and_mom_ndim,
    xprepare_data_for_reduction,
    xprepare_values_for_reduction,
)

if TYPE_CHECKING:
    from typing import (
        Any,
        Callable,
        Hashable,
        Iterable,
        Literal,
        Mapping,
        Sequence,
    )

    # Iterable,
    from numpy.typing import ArrayLike, DTypeLike, NDArray
    from xarray.core import types as xr_types
    from xarray.core.coordinates import DataArrayCoordinates
    from xarray.core.indexes import Indexes

    from cmomy.typing import KeepAttrs

    from ._typing_compat import Self
    from .central_numpy import CentralMoments
    from .typing import (
        DataCasting,
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

from .typing import ArrayOrder, ArrayOrderCF, DTypeLikeArg
from .typing import T_FloatDType as T_Float
from .typing import T_FloatDType2 as T_Float2

# * xCentralMoments -----------------------------------------------------------

docfiller_abc = docfiller.factory_from_parent(CentralMomentsABC)
docfiller_inherit_abc = docfiller.factory_inherit_from_parent(CentralMomentsABC)


@docfiller(CentralMomentsABC)  # noqa: PLR0904
class xCentralMoments(CentralMomentsABC[xr.DataArray, T_Float]):  # noqa: N801
    """
    Notes
    -----
    Most methods are wrapped to accept :class:`xarray.DataArray` object.
    """

    _xdata: xr.DataArray

    __slots__ = ("_xdata",)

    def __init__(
        self,
        data: xr.DataArray,
        mom_ndim: Mom_NDim = 1,
        *,
        fastpath: bool = False,
    ) -> None:
        if fastpath:
            self._cache = {}
            self._xdata = data
            self._data = self._xdata.data
            self._mom_ndim = mom_ndim
        else:
            super().__init__(data=data, mom_ndim=mom_ndim)

    def set_values(self, values: xr.DataArray) -> None:
        if not isinstance(values, xr.DataArray):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = (
                "data must be a xarray.DataArray. "
                "See xCentralMoments.from_data for wrapping numpy arrays"
            )
            raise TypeError(msg)

        # Ensure we have a numpy array underlying DataArray.
        self._data = values.to_numpy()
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

    def _replace_coords_isel(
        self,
        da_selected: xr.DataArray,
        indexers: Mapping[Any, Any] | None = None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ) -> xr.DataArray:
        """Update coords when reducing."""
        return replace_coords_from_isel(
            da_original=self._xdata,
            da_selected=da_selected,
            indexers=indexers,
            drop=drop,
            **indexers_kwargs,
        )

    def _check_reduce_axis_dim(
        self,
        *,
        axis: int | None = None,
        dim: Hashable | None = None,
        default_axis: int | None = None,
        default_dim: Hashable | None = None,
    ) -> tuple[int, Hashable]:
        self._raise_if_scalar()

        axis, dim = select_axis_dim(
            dims=self.dims,
            axis=axis,
            dim=dim,
            default_axis=default_axis,
            default_dim=default_dim,
        )

        if dim in self.mom_dims:
            msg = f"Can only reduce over value dimensions {self.val_dims}. Passed moment dimension {dim}."
            raise ValueError(msg)

        return axis, dim

    def _remove_dim(self, dim: Hashable | Iterable[Hashable]) -> tuple[Hashable, ...]:
        """Return self.dims with dim removed"""
        dim = {dim} if isinstance(dim, str) else set(dim)  # type: ignore[arg-type]
        return tuple(d for d in self.dims if d not in dim)

    @docfiller_abc()
    def new_like(
        self,
        data: NDArray[T_Float] | xr.DataArray | None = None,
        *,
        copy: bool = False,
        order: ArrayOrder = None,
        verify: bool = False,
        dtype: DTypeLike | None = None,
    ) -> Self:
        if isinstance(data, np.ndarray):
            xdata = self._xdata.copy(data=data)
        elif isinstance(data, xr.DataArray):
            xdata = data
        else:
            xdata = xr.zeros_like(self._xdata, dtype=dtype or self.dtype)  # type: ignore[arg-type]  # pyright: ignore[reportUnknownMemberType] # needed for python=3.8
            copy = verify = False

        if verify and xdata.shape != self.shape:
            msg = f"{xdata.shape=} != {self.shape=}"
            raise ValueError(msg)

        return type(self).from_data(
            data=xdata, mom_ndim=self._mom_ndim, copy=copy, order=order, dtype=dtype
        )

    @overload  # type: ignore[override]
    def astype(
        self,
        dtype: DTypeLikeArg[T_Float2],
        *,
        order: ArrayOrder = None,
        casting: DataCasting = None,
        subok: bool | None = None,
        copy: bool = False,
    ) -> xCentralMoments[T_Float2]: ...

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

    @docfiller_abc()  # type: ignore[arg-type]
    def astype(
        self,
        dtype: DTypeLikeArg[T_Float2] | None,
        *,
        order: ArrayOrder = None,
        casting: DataCasting = None,
        subok: bool | None = None,
        copy: bool = False,
    ) -> xCentralMoments[T_Float2] | xCentralMoments[np.float64]:
        return super().astype(  # type: ignore[return-value]
            dtype=dtype, order=order, casting=casting, subok=subok, copy=copy
        )

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
        return self.new_like(data=xdata)

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
        _copy: bool = False,
        _order: ArrayOrder = None,
        _kws: Mapping[str, Any] | None = None,
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
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> da = xCentralMoments.from_vals(rng.random((10, 2, 3)), mom=2, axis=0)
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
            _kws=_kws,
            **dimensions_kwargs,
        )

    def unstack(
        self,
        dim: Hashable | Sequence[Hashable] | None = None,
        fill_value: Any = np.nan,
        *,
        sparse: bool = False,
        _reorder: bool = True,
        _copy: bool = False,
        _order: ArrayOrder = None,
        _kws: Mapping[str, Any] | None = None,
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
            _kws=_kws,
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
        _copy: bool = False,
        _order: ArrayOrder = None,
        _kws: Mapping[str, Any] | None = None,
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
            _kws=_kws,
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
        _copy: bool = False,
        _order: ArrayOrder = None,
        _kws: Mapping[str, Any] | None = None,
    ) -> Self:
        """Interface to :meth:`xarray.DataArray.reset_index`."""
        return self.pipe(
            "reset_index",
            _reorder=_reorder,
            _copy=_copy,
            _order=_order,
            _kws=_kws,
            dims_or_levels=dims_or_levels,
            drop=drop,
        )

    def drop_vars(
        self,
        names: str | Iterable[Hashable] | Callable[[Self], str | Iterable[Hashable]],
        *,
        errors: Literal["raise", "ignore"] = "raise",
        _reorder: bool = True,
        _copy: bool = False,
        _order: ArrayOrder = None,
        _kws: Mapping[str, Any] | None = None,
    ) -> Self:
        """Interface to :meth:`xarray.DataArray.drop_vars`"""
        return self.pipe(
            "drop_vars",
            names=names,
            errors=errors,
            _reorder=_reorder,
            _copy=_copy,
            _order=_order,
            _kws=_kws,
        )

    def swap_dims(
        self,
        dims_dict: Mapping[Any, Hashable] | None = None,
        _reorder: bool = True,
        _copy: bool = False,
        _order: ArrayOrder = None,
        _kws: Mapping[str, Any] | None = None,
        **dims_kwargs: Any,
    ) -> Self:
        """Interface to :meth:`xarray.DataArray.swap_dims`."""
        return self.pipe(
            "drop_vars",
            dims_dict=dims_dict,
            _reorder=_reorder,
            _copy=_copy,
            _order=_order,
            _kws=_kws,
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
        _copy: bool = False,
        _order: ArrayOrder = None,
        _kws: Mapping[str, Any] | None = None,
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
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> da = xCentralMoments.from_vals(
        ...     rng.random((10, 3)), axis=0, dims="x", coords=dict(x=list("abc")), mom=2
        ... )
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
            _kws=_kws,
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
        _copy: bool = False,
        _order: ArrayOrder = None,
        _kws: Mapping[str, Any] | None = None,
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
            _kws=_kws,
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
        _copy: bool = False,
        _order: ArrayOrder = None,
        _kws: Mapping[str, Any] | None = None,
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
            _kws=_kws,
        )

    # ** To/from CentralMoments
    def to_centralmoments(
        self, data: NDArray[T_Float] | None = None
    ) -> CentralMoments[T_Float]:
        """Create a CentralMoments object from xCentralMoments."""
        from .central_numpy import CentralMoments

        if data is None:
            return CentralMoments(
                data=self._data,
                mom_ndim=self.mom_ndim,
                fastpath=True,
            )

        return CentralMoments(data=data, mom_ndim=self.mom_ndim)

    def to_c(self, data: NDArray[T_Float] | None = None) -> CentralMoments[T_Float]:
        """Alias to :meth:`to_centralmoments`"""
        return self.to_centralmoments(data=data)

    @cached.prop
    def centralmoments_view(self) -> CentralMoments[T_Float]:
        """
        Create CentralMoments view.

        This object has the same underlying data as `self`, but no
        DataArray attributes.  Useful for some function calls.

        See Also
        --------
        CentralMoments.to_xcentralmoments
        xCentralMoments.from_centralmoments
        """
        return self.to_centralmoments()

    @classmethod
    @docfiller.decorate
    def from_centralmoments(
        cls,
        obj: CentralMoments[T_Float2],
        *,
        dims: XArrayDimsType = None,
        attrs: XArrayAttrsType = None,
        coords: XArrayCoordsType = None,
        name: XArrayNameType = None,
        indexes: XArrayIndexesType = None,
        mom_dims: MomDims | None = None,
        template: xr.DataArray | None = None,
        copy: bool = False,
    ) -> xCentralMoments[T_Float2]:
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
    def _check_out_shape(self, out: NDArrayAny) -> None:
        if out.shape != self.shape:
            msg = f"Trying to reshape from {self.shape=} to {out.shape=}."
            raise ValueError(msg)

    def _push_data_dataarray(
        self,
        data: xr.DataArray,
        *,
        order: ArrayOrder = None,
        parallel: bool | None = None,
    ) -> Self:
        def func(out: NDArrayAny, data: NDArrayAny) -> NDArrayAny:
            self._check_out_shape(out)
            if order:
                data = np.asarray(data, order=order)
            self._pusher(parallel).data(data, self._data)
            return self._data

        # NOTE:  Use this pattern here and below
        # pass self._data to func as a test for broadcasting,
        # But then will use `self._data` as is for `out` parameter.
        # Also, but `out` first so things are transposed relative to that...
        _ = xr.apply_ufunc(
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
        axis: int | None,
        dim: Hashable | None,
        parallel: bool | None,
        order: ArrayOrder,
    ) -> Self:
        def func(_out: NDArrayAny, _datas: NDArrayAny) -> NDArrayAny:
            self._check_out_shape(_out)
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
        _ = xr.apply_ufunc(
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
            out: NDArrayAny, x0: NDArrayAny, w: NDArrayAny, *x1: NDArrayAny
        ) -> NDArrayAny:
            self._check_out_shape(out)
            self._pusher(parallel).val(x0, *x1, w, self._data)
            return self._data

        weight = 1.0 if weight is None else weight
        core_dims = [self.mom_dims, *([[]] * (2 + len(y)))]
        _ = xr.apply_ufunc(
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
        axis: int | None = None,
        dim: Hashable | None = None,
        order: ArrayOrder = None,
        parallel: bool | None = None,
    ) -> Self:
        self._check_y(y, self.mom_ndim)

        weight = 1.0 if weight is None else weight

        input_core_dims, (x0, w, *x1) = xprepare_values_for_reduction(
            x, weight, *y, axis=axis, dim=dim, order=order, narrays=self.mom_ndim + 1
        )

        def func(
            out: NDArrayAny, x0: NDArrayAny, w: NDArrayAny, *x1: NDArrayAny
        ) -> NDArrayAny:
            self._check_out_shape(out)
            self._pusher(parallel).vals(x0, *x1, w, self._data)
            return self._data

        _ = xr.apply_ufunc(
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
        data: NDArray[T_Float] | xr.DataArray,
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
        datas: NDArray[T_Float] | xr.DataArray,
        *,
        axis: int | None = None,
        dim: Hashable | None = None,
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
        axis: int | None = None,
        dim: Hashable | None = None,
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
    # ** Reducetion -----------------------------------------------------------
    @docfiller.decorate
    def resample_and_reduce(
        self,
        *,
        freq: NDArrayInt,
        dim: Hashable | None = None,
        axis: int | None = None,
        rep_dim: str = "rep",
        parallel: bool | None = None,
        order: ArrayOrder = None,
        keep_attrs: KeepAttrs = None,
    ) -> Self:
        """

        Parameters
        ----------
        {freq}
        {dim}
        {axis}
        {rep_dim}
        {parallel}
        {order}
        {keep_attrs}

        Returns
        -------
        output : {klass}

        See Also
        --------
        .resample.resample_data
        .resample.randsamp_freq

        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> da = xCentralMoments.from_vals(
        ...     rng.random((10, 3)), mom=3, axis=0, dims="rec"
        ... )
        >>> da
        <xCentralMoments(val_shape=(3,), mom=(3,))>
        <xarray.DataArray (rec: 3, mom_0: 4)> Size: 96B
        array([[ 1.0000e+01,  5.2485e-01,  1.1057e-01, -4.6282e-03],
               [ 1.0000e+01,  5.6877e-01,  6.8876e-02, -1.2745e-02],
               [ 1.0000e+01,  5.0944e-01,  1.1978e-01, -1.4644e-02]])
        Dimensions without coordinates: rec, mom_0

        Note that for reproducible results, must set numba random
        seed as well

        >>> da_resamp, freq = da.resample_and_reduce(
        ...     nrep=5,
        ...     dim="rec",
        ...     full_output=True,
        ...     rng=rng,
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

        >>> from cmomy.resample import freq_to_indices
        >>> indices = freq_to_indices(freq)
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

        return type(self)(data=data, mom_ndim=self.mom_ndim)  # type: ignore[arg-type]

    def reduce(
        self,
        *,
        by: ArrayLike | None = None,
        axis: int | None = None,
        order: ArrayOrder = None,
        parallel: bool | None = None,
        # xarray specific
        dim: Hashable | None = None,
        dtype: DTypeLike | None = None,
        group_dim: Hashable | None = None,
        groups: Sequence[Any] | None = None,
        keep_attrs: KeepAttrs = None,
    ) -> Self:
        """
        Parameters
        ----------
        {dim}
        {axis}
        by : ndarray or DataArray or str or iterable of str, optional
            If ``None``, reduce over entire ``dim``. Otherwise, reduce by
            group. If :class:`~numpy.ndarray`, use the unique values. If
            :class:`~xarray.DataArray`, use unique values and rename dimension
            ``by.name``. If str or Iterable of str, Create grouper from these
            named coordinates.
        coords_policy : {{'first', 'last', 'group', None}}
            Policy for handling coordinates along ``dim`` by ``by`` is specified.
            If no coordinates do nothing, otherwise use:

            * 'first': select first value of coordinate for each block.
            * 'last': select last value of coordinate for each block.
            * 'group': Assign unique groups from ``group_idx`` to ``dim``
            * None: drop any coordinates.
        group_name : str, optional
            If supplied, add the unique groups to this coordinate. Ignored if
            ``coords_policy == "group"`.
        rename_dim : hashable, optional
            Optional name for the output dimension (rename ``dim`` to
            ``rename_dim``).
        reduce_kws : mapping, optional
            Optional parameters to :func:`.indexed.reduce_by_index`.
        **kwargs
            Extra arguments to :meth:`from_datas` if ``group_idx`` is ``None``,
            or to :meth:`from_data` otherwise

        Returns
        -------
        output : {klass}
            If ``group_idx`` is ``None``, reduce over all samples in ``dim`` or
            ``axis``. Otherwise, reduce for each unique value of ``group_idx``.

        Notes
        -----
        This is a new feature, and subject to change. In particular, the
        current implementation is not smart about the output coordinates and
        dimensions, and is inconsistent with :meth:`xarray.DataArray.groupby`.
        It is up the the user to manipulate the dimensions/coordinates. output
        dimensions and coords simplistically.

        See Also
        --------
        from_datas
        .indexed.reduce_by_group_idx

        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> da = xCentralMoments.from_vals(rng.random((10, 2, 3)), axis=0, mom=2)
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
            return type(self).from_datas(
                self._xdata,
                mom_ndim=self._mom_ndim,
                axis=axis,
                dim=dim,
                order=order,
                parallel=parallel,
            )

        from .reduction import reduce_data_grouped

        out = reduce_data_grouped(
            self._xdata,
            mom_ndim=self._mom_ndim,
            by=by,
            axis=axis,
            dim=dim,
            order=order,
            parallel=parallel,
            dtype=dtype,
            group_dim=group_dim,
            groups=groups,
            keep_attrs=keep_attrs,
        )

        return type(self)(data=out, mom_ndim=self.mom_ndim)

    @docfiller.decorate
    def block(
        self,
        block_size: int | None,
        dim: Hashable | None = None,
        axis: int | None = None,
        block_dim: str | None = None,
        order: ArrayOrder = None,
        parallel: bool | None = None,
        keep_attrs: KeepAttrs = None,
    ) -> Self:
        """
        Parameters
        ----------
        block_size : int
            Number of observations to include in a given block.
        {dim}
        {axis}
        block_dim : str, default=None,
            Name of blocked dimension.  Defaults to ``dim``.
        coords_policy : {{'first', 'last', None}}
            Policy for handling coordinates along `axis`.
            If no coordinates do nothing, otherwise use:

            * 'first': select first value of coordinate for each block.
            * 'last': select last value of coordinate for each block.
            * None: drop any coordinates.



        {coords_policy}
        **kwargs
            Extra arguments to :meth:`CentralMoments.block`

        Returns
        -------
        output : xCentralMoments
            Object with block averaging.

        See Also
        --------
        CentralMoments.block


        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> x = rng.random((10, 10))
        >>> da = xCentralMoments.from_vals(x, mom=2)
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

        >>> xCentralMoments.from_vals(x.reshape(2, 50), mom=2, axis=1)
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
        axis, dim = self._check_reduce_axis_dim(axis=axis, dim=dim, default_axis=0)

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
        )

    # ** Constructors
    @classmethod
    @docfiller_inherit_abc()
    def zeros(
        cls,
        *,
        mom: Moments,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        order: ArrayOrderCF | None = None,
        dims: XArrayDimsType = None,
        mom_dims: MomDims | None = None,
        attrs: XArrayAttrsType = None,
        coords: XArrayCoordsType = None,
        name: XArrayNameType = None,
        indexes: XArrayIndexesType = None,
        template: xr.DataArray | None = None,
    ) -> Self:
        """
        Parameters
        ----------
        {xr_params}
        {mom_dims}

        See Also
        --------
        CentralMoments.to_xcentralmoments
        """
        from .central_numpy import CentralMoments

        return CentralMoments.zeros(  # type: ignore[return-value]
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

    @classmethod
    @docfiller_inherit_abc()
    def from_data(
        cls,
        data: xr.DataArray,
        *,
        mom_ndim: Mom_NDim,
        copy: bool = False,
        order: ArrayOrder = None,
        dtype: DTypeLike | None = None,
    ) -> Self:
        return cls(
            data=data.astype(dtype or data.dtype, copy=copy, order=order),
            mom_ndim=mom_ndim,
        )

    @classmethod
    @docfiller_inherit_abc()
    def from_datas(
        cls,
        datas: xr.DataArray,
        *,
        mom_ndim: Mom_NDim,
        axis: int | None = None,
        dim: Hashable | None = None,
        order: ArrayOrder = None,
        parallel: bool | None = None,
        keep_attrs: bool = True,
    ) -> Self:
        """

        Parameters
        ----------
        datas : DataArray
            If pass in a DataArray, use it's attributes in new object.
            If ndarray, use `dim`, `attrs`, etc, to wrap resulting data.
        {dim}
        {xr_params}


        See Also
        --------
        CentralMoments.from_datas
        CentralMoments.to_xcentralmoments

        Notes
        -----
        If pass in :class:`xarray.DataArray`, then dims, etc, are ignored.
        Note that here, `dims` does not include the dimension reduced over.
        The dimensions are applied after the fact.
        """
        from .reduction import reduce_data

        return cls(
            data=reduce_data(
                data=datas,
                mom_ndim=mom_ndim,
                axis=axis,
                dim=dim,
                order=order,
                parallel=parallel,
                keep_attrs=keep_attrs,
            ),
            mom_ndim=mom_ndim,
        )

    @classmethod
    @docfiller_inherit_abc()
    def from_vals(
        cls,
        x: xr.DataArray,
        *y: ArrayLike | xr.DataArray,
        mom: Moments,
        weight: ArrayLike | xr.DataArray | None = None,
        axis: int | None = None,
        dim: Hashable | None = None,
        mom_dims: MomDims | None = None,
        order: ArrayOrder = None,
        parallel: bool | None = None,
        keep_attrs: KeepAttrs = None,
    ) -> Self:
        """
        Parameters
        ----------
        x : array, tuple of array, DataArray, or tuple of DataArray
            For moments, `x=x0`.  For comoments, `x=(x0, x1)`.
            If pass DataArray, inherit attributes from `x0`.  If pass
            ndarray, use `dims`, `attrs`, etc to wrap final result
        {dim}
        {mom_dims}
        {keep_attrs}

        See Also
        --------
        CentralMoments.from_vals
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
                keep_attrs=keep_attrs,
            ),
            mom_ndim=mom_ndim,
        )

    @classmethod
    @docfiller_inherit_abc()
    def from_resample_vals(
        cls,
        x: xr.DataArray,
        *y: ArrayLike | xr.DataArray,
        mom: Moments,
        freq: NDArrayInt,
        weight: ArrayLike | xr.DataArray | None = None,
        axis: int | None = None,
        dim: Hashable | None = None,
        order: ArrayOrder = None,
        parallel: bool | None = None,
        mom_dims: MomDims | None = None,
        rep_dim: str | None = "rep",
        keep_attrs: bool = True,
    ) -> Self:
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
        """
        mom_strict, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)

        from .resample import resample_vals

        return cls(
            data=resample_vals(  # type: ignore[arg-type]
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
            ),
            mom_ndim=mom_ndim,
        )

    @classmethod
    @docfiller_inherit_abc()
    def from_raws(
        cls,
        raws: xr.DataArray,
        *,
        mom_ndim: Mom_NDim,
        axis: int | None = None,
        dim: Hashable | None = None,
        order: ArrayOrder = None,
        parallel: bool | None = None,
        keep_attrs: bool = False,
    ) -> Self:
        """
        See Also
        --------
        CentralMoments.from_raw
        """
        return cls.from_raw(raws, mom_ndim=mom_ndim).reduce(
            axis=axis,
            dim=dim,
            order=order,
            parallel=parallel,
            keep_attrs=keep_attrs,
        )

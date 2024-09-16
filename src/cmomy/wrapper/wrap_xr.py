"""Wrapper object for daataarrays"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

import numpy as np
import xarray as xr

from cmomy.core.missing import MISSING
from cmomy.core.prepare import (
    prepare_data_for_reduction,
    prepare_values_for_reduction,
    xprepare_values_for_reduction,
)
from cmomy.core.validate import (
    is_dataarray,
    is_dataset,
    is_xarray,
    raise_if_wrong_value,
    validate_floating_dtype,
    validate_mom_dims,
)
from cmomy.core.xr_utils import (
    astype_dtype_dict,
    contains_dims,
    get_apply_ufunc_kwargs,
    get_mom_shape,
    select_axis_dim,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        ItemsView,
        Iterable,
        Iterator,
        KeysView,
        Mapping,
        Sequence,
        ValuesView,
    )
    from typing import Any, Literal

    from numpy.typing import ArrayLike, DTypeLike
    from xarray.core import types as xr_types
    from xarray.core.coordinates import DataArrayCoordinates, DatasetCoordinates

    from cmomy.core.typing import (
        ApplyUFuncKwargs,
        ArrayOrder,
        ArrayOrderCF,
        AttrsType,
        AxisReduce,
        Casting,
        CentralMomentsDataArray,
        CentralMomentsDataset,
        CoordsPolicy,
        CoordsType,
        Dims,
        DimsReduce,
        DimsType,
        Groups,
        KeepAttrs,
        MissingCoreDimOptions,
        MissingType,
        Mom_NDim,
        MomDims,
        MomDimsStrict,
        Moments,
        MomentsStrict,
        NameType,
        NDArrayAny,
    )
    from cmomy.core.typing_compat import Self
    from cmomy.wrapper.wrap_np import CentralMomentsArray


from cmomy.core.docstrings import docfiller_xcentral as docfiller
from cmomy.core.typing import DataT

from .wrap_abc import CentralMomentsABC

docfiller_abc = docfiller.factory_from_parent(CentralMomentsABC)
docfiller_inherit_abc = docfiller.factory_inherit_from_parent(CentralMomentsABC)


@docfiller.inherit(CentralMomentsABC)  # noqa: PLR0904
class CentralMomentsData(CentralMomentsABC[DataT]):
    """
    Central moments wrapper of {DataArray} or {Dataset} objects.

    Parameters
    ----------
    {mom_dims_data}
    """

    __slots__ = ("_mom_dims",)

    _mom_dims: MomDimsStrict

    def __init__(
        self,
        obj: DataT,
        *,
        mom_ndim: Mom_NDim = 1,
        mom_dims: MomDims | None = None,
        fastpath: bool = False,
    ) -> None:
        if not is_xarray(obj):
            msg = "obj must be a DataArray or Dataset, not {type(obj)}"
            raise TypeError(msg)

        self._mom_dims = (
            cast("MomDimsStrict", mom_dims)
            if fastpath
            else validate_mom_dims(mom_dims, mom_ndim, obj)
        )

        # NOTE: Why this ignore?
        super().__init__(obj=obj, mom_ndim=mom_ndim, fastpath=fastpath)  # type: ignore[arg-type]

    # ** Properties ------------------------------------------------------------
    @property
    @docfiller_abc()
    def mom_shape(self) -> MomentsStrict:
        return get_mom_shape(self._obj, self._mom_dims)

    @property
    def mom_dims(self) -> MomDimsStrict:
        """Moments dimension names."""
        return self._mom_dims

    # Dataset specific dict-view-like
    def as_dict(self: CentralMomentsDataset) -> dict[Hashable, CentralMomentsDataArray]:
        """
        Create a wrapped dictionary view of dataset

        Note that this dict will only contain variables (DataArrays) which
        contain ``mom_dims``.  That is, non central moment arrays will
        be dropped.
        """
        if is_dataarray(self._obj):
            self._raise_not_implemented("dict view")
        return {
            k: type(self)(  # type: ignore[misc]
                obj,  # type: ignore[arg-type]
                mom_ndim=self._mom_ndim,
                mom_dims=self._mom_dims,
                fastpath=True,
            )
            for k, obj in self._obj.items()
            if contains_dims(obj, self._mom_dims)
        }

    def items(
        self: CentralMomentsDataset,
    ) -> ItemsView[Hashable, CentralMomentsDataArray]:
        """
        Dict like items method.

        Note that only DataArrays that contain ``mom_dims`` are returned.
        To iterate over all items, use ``self.obj.items()``
        """
        return self.as_dict().items()

    def keys(
        self: CentralMomentsDataset,
    ) -> KeysView[Hashable]:  # -> Iterator[Hashable]:
        """Dict like keys view"""
        return self.as_dict().keys()  # type: ignore[misc]

    def values(
        self: CentralMomentsDataset,
    ) -> ValuesView[CentralMomentsDataArray]:  # -> Iterator[CentralMomentsDataArray]:
        """Dict like values view"""
        return self.as_dict().values()  # type: ignore[misc]

    @overload
    def iter(self: CentralMomentsDataArray) -> Iterator[CentralMomentsDataArray]: ...
    @overload
    def iter(self: CentralMomentsDataset) -> Iterator[Hashable]: ...
    @overload
    def iter(self) -> Any: ...

    def iter(self) -> Iterator[Hashable] | Iterator[CentralMomentsDataArray]:
        """Need this for proper typing with mypy..."""
        if is_dataarray(self._obj):
            if self.ndim <= self._mom_ndim:
                msg = "Can only iterate over wrapped DataArray with extra dimension."
                raise ValueError(msg)
            for obj in self._obj:
                yield self.new_like(obj)  # noqa: DOC402
        else:
            yield from self.keys()  # type: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

    @overload
    def __iter__(
        self: CentralMomentsDataArray,
    ) -> Iterator[CentralMomentsDataArray]: ...
    @overload
    def __iter__(self: CentralMomentsDataset) -> Iterator[Hashable]: ...
    @overload
    def __iter__(self) -> Any: ...

    def __iter__(self) -> Iterator[Hashable] | Iterator[CentralMomentsDataArray]:
        return self.iter()

    @overload
    def __getitem__(
        self: CentralMomentsDataArray,
        key: Any,
    ) -> CentralMomentsDataArray: ...
    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self: CentralMomentsDataset,
        key: Hashable,
    ) -> CentralMomentsDataArray: ...
    @overload
    def __getitem__(
        self: CentralMomentsDataset,
        key: Any,
    ) -> CentralMomentsDataset: ...

    def __getitem__(self, key: Any) -> CentralMomentsDataArray | CentralMomentsDataset:
        """
        Access variables or coordinates of this wrapped dataset as a DataArray or a
        subset of variables or a indexed dataset.

        For wrapped ``Dataset``, return a subset of variables.

        The selection is wrapped with ``CentralMomentsData``.
        """
        obj: xr.DataArray | xr.Dataset = self._obj[key]  # pyright: ignore[reportUnknownVariableType]
        if not contains_dims(obj, self._mom_dims):  # pyright: ignore[reportUnknownArgumentType]
            msg = f"Cannot select object without {self._mom_dims}"
            raise ValueError(msg)
        self._raise_if_wrong_mom_shape(get_mom_shape(obj, self._mom_dims))  # pyright: ignore[reportUnknownArgumentType]
        return type(self)(
            obj,  # type: ignore[arg-type]
            mom_ndim=self._mom_ndim,
            mom_dims=self._mom_dims,
            fastpath=True,
        )

    # ** Create/copy/new ------------------------------------------------------
    def _new_like(self, obj: DataT) -> Self:
        return type(self)(
            obj=obj,
            mom_ndim=self._mom_ndim,
            mom_dims=self._mom_dims,
            fastpath=True,
        )

    @docfiller_inherit_abc()
    def new_like(  # type: ignore[override]
        self,
        obj: ArrayLike | DataT | Mapping[Any, Any] | None = None,
        *,
        copy: bool | None = None,
        deep: bool = True,
        verify: bool = False,
        dtype: DTypeLike | Mapping[Hashable, DTypeLike] = None,
        fastpath: bool = False,
    ) -> Self:
        """
        Parameters
        ----------
        obj : array-like, DataArray, Dataset, or mapping
            Object for new object to wrap.  Passed to `self.obj.copy`.
        deep : bool
            Parameter to :meth:`~xarray.Dataset.copy` or :meth:`~xarray.DataArray.copy`.

        See Also
        --------
        xarray.DataArray.copy
        xarray.Dataset.copy
        """
        if obj is None:
            # TODO(wpk): different type for dtype in xarray (can be a mapping...)
            # Also can probably speed this up by validating dtype here...
            return type(self)(
                obj=xr.zeros_like(self._obj, dtype=dtype),  # type: ignore[arg-type]
                mom_ndim=self._mom_ndim,
                mom_dims=self._mom_dims,
                fastpath=fastpath,
            )

        # TODO(wpk): edge case of passing in new xarray data with different moment dimensions.
        # For now, this will raise an error.
        obj_: DataT
        if type(self._obj) is type(obj):
            obj_ = cast("DataT", obj)
        else:
            obj_ = self._obj.copy(data=obj)  # type: ignore[arg-type]

        # minimal check on shape and that mom_dims are present....
        if not contains_dims(obj_, self._mom_dims):
            msg = f"Cannot create new from object without {self._mom_dims}"
            raise ValueError(msg)
        self._raise_if_wrong_mom_shape(get_mom_shape(obj_, self._mom_dims))

        if verify:
            raise_if_wrong_value(obj_.sizes, self._obj.sizes, "Wrong `obj.sizes`.")

        if not fastpath:
            copy = False if copy is None else copy
            if dtype:
                obj_ = obj_.astype(astype_dtype_dict(self._obj, dtype), copy=copy)  # pyright: ignore[reportUnknownMemberType]
            elif copy:
                obj_ = obj_.copy(deep=deep)

        return type(self)(
            obj=obj_,
            mom_ndim=self._mom_ndim,
            mom_dims=self._mom_dims,
            fastpath=fastpath,
        )

    @docfiller_inherit_abc()
    def copy(self, deep: bool = True) -> Self:
        """
        Parameters
        ----------
        deep : bool
            Parameters to :meth:`xarray.DataArray.copy` or `xarray.Dataset.copy`
        """
        return type(self)(
            obj=self._obj.copy(deep=deep),
            mom_ndim=self._mom_ndim,
            mom_dims=self._mom_dims,
            fastpath=True,
        )

    @docfiller_inherit_abc()
    def astype(
        self,
        dtype: DTypeLike | Mapping[Hashable, DTypeLike],
        *,
        order: ArrayOrder = None,
        casting: Casting | None = None,
        subok: bool | None = None,
        copy: bool = False,
    ) -> Self:
        """
        Parameters
        ----------
        dtype : dtype or mapping of hashable to dtype
            If ``self.obj`` is a :class:`~xarray.Dataset`, passing a mapping will update only certain variables.
        """
        dtype = astype_dtype_dict(self._obj, dtype)
        # Reimplement with a a full new_like to defer dtype check to __init__
        return self.new_like(
            obj=self._obj.astype(  # pyright: ignore[reportUnknownMemberType]
                dtype, order=order, casting=casting, subok=subok, copy=copy
            )
        )

    # ** Utils ----------------------------------------------------------------
    def _validate_dtype(self) -> None:
        if is_dataarray(self._obj):
            _ = validate_floating_dtype(self._obj)
            return

        for name, val in self._obj.items():
            if contains_dims(val, self._mom_dims):
                _ = validate_floating_dtype(val, name=name)

    # ** Pushing --------------------------------------------------------------
    @property
    def _dtype(self) -> np.dtype[Any] | None:
        if is_dataarray(self._obj):
            return self._obj.dtype  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        return None

    @docfiller_inherit_abc()
    def push_data(
        self,
        data: DataT | ArrayLike,
        *,
        scale: ArrayLike | None = None,
        casting: Casting = "same_kind",
        parallel: bool | None = False,
        keep_attrs: KeepAttrs = True,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self:
        """
        Parameters
        ----------
        {keep_attrs}
        {on_missing_core_dim}
        {apply_ufunc_kwargs}
        """
        if scale is None:

            def func(out: NDArrayAny, data: NDArrayAny) -> NDArrayAny:
                self._pusher(parallel, size=out.size).data(
                    data,
                    out,
                    casting=casting,
                    signature=(out.dtype, out.dtype),
                )
                return out
        else:

            def func(out: NDArrayAny, data: NDArrayAny) -> NDArrayAny:
                self._pusher(parallel, size=out.size).data_scale(
                    data,
                    scale,
                    out,
                    casting=casting,
                    signature=(out.dtype, out.dtype, out.dtype),
                )
                return out

        self._obj = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            func,
            self._obj,
            data,
            input_core_dims=[self._mom_dims, self._mom_dims],
            output_core_dims=[self._mom_dims],
            keep_attrs=keep_attrs,
            **get_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                on_missing_core_dim=on_missing_core_dim,
                dask="parallelized",
                output_dtypes=self._dtype or np.float64,
            ),
        )

        return self

    @docfiller_inherit_abc()
    def push_datas(
        self,
        datas: DataT | ArrayLike,
        *,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        casting: Casting = "same_kind",
        parallel: bool | None = None,
        keep_attrs: KeepAttrs = True,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self:
        """
        Parameters
        ----------
        {keep_attrs}
        {on_missing_core_dim}
        {apply_ufunc_kwargs}
        """

        def func(out: NDArrayAny, datas: NDArrayAny) -> NDArrayAny:
            self._pusher(parallel, size=out.size).datas(
                datas,
                out,
                casting=casting,
                signature=(out.dtype, out.dtype),
            )
            return out

        if is_xarray(datas):
            axis, dim = select_axis_dim(
                datas, axis=axis, dim=dim, mom_ndim=self._mom_ndim
            )

        else:
            # Removed restriction that you could only pass array with wrapped DataArray.
            # Trust the user to do what they want....
            axis, datas = prepare_data_for_reduction(
                datas,
                axis=axis,
                mom_ndim=self._mom_ndim,
                dtype=self._dtype,
                move_axis_to_end=True,
                recast=False,
            )
            dim = "_dummy123"

        self._obj = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            func,
            self._obj,
            datas,
            input_core_dims=[self._mom_dims, [dim, *self._mom_dims]],
            output_core_dims=[self._mom_dims],
            keep_attrs=keep_attrs,
            **get_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                on_missing_core_dim=on_missing_core_dim,
                dask="parallelized",
                output_dtypes=self._dtype or np.float64,
            ),
        )

        return self

    @docfiller_inherit_abc()
    def push_val(
        self,
        x: ArrayLike | DataT,
        *y: ArrayLike | xr.DataArray | DataT,
        weight: ArrayLike | xr.DataArray | DataT | None = None,
        casting: Casting = "same_kind",
        parallel: bool | None = False,
        keep_attrs: KeepAttrs = True,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self:
        """
        Parameters
        ----------
        {keep_attrs}
        {on_missing_core_dim}
        {apply_ufunc_kwargs}
        """
        self._check_y(y, self._mom_ndim)

        def func(
            out: NDArrayAny,
            *args: NDArrayAny,
        ) -> NDArrayAny:
            self._pusher(parallel, size=out.size).val(
                out,
                *args,
                casting=casting,
                signature=(out.dtype,) * (len(args) + 1),
            )
            return out

        core_dims: list[Any] = [self.mom_dims, *([[]] * (2 + len(y)))]
        self._obj = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            func,
            self._obj,
            x,
            1.0 if weight is None else weight,
            *y,
            input_core_dims=core_dims,
            output_core_dims=[self.mom_dims],
            keep_attrs=keep_attrs,
            **get_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                on_missing_core_dim=on_missing_core_dim,
                dask="parallelized",
                output_dtypes=self._dtype or np.float64,
            ),
        )

        return self

    @docfiller_inherit_abc()
    def push_vals(
        self,
        x: ArrayLike | DataT,
        *y: ArrayLike | xr.DataArray | DataT,
        weight: ArrayLike | xr.DataArray | DataT | None = None,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        casting: Casting = "same_kind",
        parallel: bool | None = None,
        keep_attrs: KeepAttrs = True,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self:
        """
        Parameters
        ----------
        {keep_attrs}
        {on_missing_core_dim}
        {apply_ufunc_kwargs}
        """
        weight = 1.0 if weight is None else weight
        xargs: Sequence[ArrayLike | xr.DataArray | xr.Dataset]
        if is_xarray(x):
            dim, input_core_dims, xargs = xprepare_values_for_reduction(
                x,
                weight,
                *y,
                axis=axis,
                dim=dim,
                dtype=self._dtype,
                recast=False,
                narrays=self._mom_ndim + 1,
            )
        else:
            axis, xargs = prepare_values_for_reduction(
                x,
                weight,
                *y,
                axis=axis,
                dtype=self._dtype,
                recast=False,
                narrays=self._mom_ndim + 1,
                move_axis_to_end=True,
            )
            dim = "_dummy123"
            input_core_dims = [[dim]] * len(xargs)

        def func(
            out: NDArrayAny,
            *args: NDArrayAny,
        ) -> NDArrayAny:
            self._pusher(parallel, size=out.size).vals(
                out,
                *args,
                casting=casting,
                signature=(out.dtype,) * (len(args) + 1),
            )
            return out

        self._obj = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            func,
            self._obj,
            *xargs,
            input_core_dims=[self._mom_dims, *input_core_dims],  # type: ignore[has-type]
            output_core_dims=[self._mom_dims],
            keep_attrs=keep_attrs,
            **get_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                on_missing_core_dim=on_missing_core_dim,
                dask="parallelized",
                output_dtypes=self._dtype or np.float64,
            ),
        )

        return self

    # ** Interface to modules -------------------------------------------------
    # *** .reduction ----------------------------------------------------------
    @docfiller_inherit_abc()
    def reduce(  # noqa: PLR0913
        self,
        *,
        by: str | Groups | None = None,
        block: int | None = None,
        axis: AxisReduce | MissingType = MISSING,
        move_axis_to_end: bool = False,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        casting: Casting = "same_kind",
        order: ArrayOrderCF = None,
        keepdims: bool = False,
        parallel: bool | None = None,
        # xarray specific
        coords_policy: CoordsPolicy = "first",
        use_reduce: bool = False,
        dim: DimsReduce | MissingType = MISSING,
        group_dim: str | None = None,
        groups: Groups | None = None,
        keep_attrs: KeepAttrs = None,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
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
        use_reduce : bool
            If ``True``, use ``self.obj.reduce(...)``.
        {coords_policy}
        {group_dim}
        {groups}
        {keep_attrs}
        {on_missing_core_dim}
        {apply_ufunc_kwargs}

        Notes
        -----
        This is a new feature, and subject to change. In particular, the
        current implementation is not smart about the output coordinates and
        dimensions, and is inconsistent with :meth:`xarray.DataArray.groupby`.
        It is up the the user to manipulate the dimensions/coordinates. output
        dimensions and coords simplistically.

        See Also
        --------
        reduction.reduce_data
        reduction.reduce_data_grouped
        reduction.reduce_data_indexed

        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.default_rng(0)
        >>> data = xr.DataArray(rng.random((6, 3)), dims=["rec", "mom"]).assign_coords(
        ...     rec=range(6)
        ... )
        >>> da = CentralMomentsData(data)
        >>> da
        <CentralMomentsData(mom_ndim=1)>
        <xarray.DataArray (rec: 6, mom: 3)> Size: 144B
        array([[0.637 , 0.2698, 0.041 ],
               [0.0165, 0.8133, 0.9128],
               [0.6066, 0.7295, 0.5436],
               [0.9351, 0.8159, 0.0027],
               [0.8574, 0.0336, 0.7297],
               [0.1757, 0.8632, 0.5415]])
        Coordinates:
          * rec      (rec) int64 48B 0 1 2 3 4 5
        Dimensions without coordinates: mom

        Reduce along a single dimension

        >>> da.reduce(dim="rec")
        <CentralMomentsData(mom_ndim=1)>
        <xarray.DataArray (mom: 3)> Size: 24B
        array([3.2283, 0.4867, 0.4535])
        Dimensions without coordinates: mom


        Reduce by group:

        >>> by = [0, 0, 0, 1, 1, 1]
        >>> da.reduce(dim="rec", by=by, coords_policy="first")
        <CentralMomentsData(mom_ndim=1)>
        <xarray.DataArray (rec: 2, mom: 3)> Size: 48B
        array([[1.2601, 0.4982, 0.3478],
               [1.9681, 0.4793, 0.521 ]])
        Coordinates:
          * rec      (rec) int64 16B 0 3
        Dimensions without coordinates: mom

        Note the coordinate ``rec``.  Using the ``coords_policy="first"``
        uses the first coordinate a group is observed.

        >>> da.reduce(dim="rec", by=by, coords_policy="group", groups=["a", "b"])
        <CentralMomentsData(mom_ndim=1)>
        <xarray.DataArray (rec: 2, mom: 3)> Size: 48B
        array([[1.2601, 0.4982, 0.3478],
               [1.9681, 0.4793, 0.521 ]])
        Coordinates:
          * rec      (rec) <U1 8B 'a' 'b'
        Dimensions without coordinates: mom

        Reduce by coord:

        >>> dag = da.assign_coords(group_coord=("rec", by))
        >>> dag
        <CentralMomentsData(mom_ndim=1)>
        <xarray.DataArray (rec: 6, mom: 3)> Size: 144B
        array([[0.637 , 0.2698, 0.041 ],
               [0.0165, 0.8133, 0.9128],
               [0.6066, 0.7295, 0.5436],
               [0.9351, 0.8159, 0.0027],
               [0.8574, 0.0336, 0.7297],
               [0.1757, 0.8632, 0.5415]])
        Coordinates:
          * rec          (rec) int64 48B 0 1 2 3 4 5
            group_coord  (rec) int64 48B 0 0 0 1 1 1
        Dimensions without coordinates: mom

        >>> dag.reduce(dim="rec", by="group_coord")
        <CentralMomentsData(mom_ndim=1)>
        <xarray.DataArray (rec: 2, mom: 3)> Size: 48B
        array([[1.2601, 0.4982, 0.3478],
               [1.9681, 0.4793, 0.521 ]])
        Coordinates:
          * rec          (rec) int64 16B 0 3
            group_coord  (rec) int64 16B 0 1
        Dimensions without coordinates: mom


        Block averaging:

        >>> da.reduce(block=3, dim="rec")
        <CentralMomentsData(mom_ndim=1)>
        <xarray.DataArray (rec: 2, mom: 3)> Size: 48B
        array([[1.2601, 0.4982, 0.3478],
               [1.9681, 0.4793, 0.521 ]])
        Coordinates:
          * rec      (rec) int64 16B 0 3
        Dimensions without coordinates: mom

        Note that this is the same as the group reduction above. For finer
        control of how ``block`` is transformed to ``by``, see
        :func:`~.reduction.block_by`.
        """
        if by is None and block is not None:
            by = self._block_by(block, axis=axis, dim=dim)
            groups = groups or range(by.max() + 1)

        if by is None:
            from cmomy.reduction import reduce_data

            data = reduce_data(
                self._obj,
                mom_ndim=self._mom_ndim,
                axis=axis,
                dim=dim,
                out=out,
                dtype=dtype,
                casting=casting,
                order=order,
                keepdims=keepdims,
                parallel=parallel,
                use_reduce=use_reduce,
                keep_attrs=keep_attrs,
                on_missing_core_dim=on_missing_core_dim,
                apply_ufunc_kwargs=apply_ufunc_kwargs,
            )

        elif coords_policy in {"first", "last"}:
            from cmomy.reduction import factor_by_to_index, reduce_data_indexed

            if isinstance(by, str):
                _groups, index, group_start, group_end = factor_by_to_index(
                    self._obj[by].to_numpy()  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
                )
            else:
                _groups, index, group_start, group_end = factor_by_to_index(by)

            data = reduce_data_indexed(
                self._obj,
                mom_ndim=self._mom_ndim,
                index=index,
                group_start=group_start,
                group_end=group_end,
                axis=axis,
                dim=dim,
                move_axis_to_end=move_axis_to_end,
                parallel=parallel,
                dtype=dtype,
                out=out,
                casting=casting,
                order=order,
                coords_policy=coords_policy,
                group_dim=group_dim,
                groups=groups,
                keep_attrs=keep_attrs,
            )
        else:
            from cmomy.reduction import reduce_data_grouped

            codes: ArrayLike
            if isinstance(by, str):
                from cmomy.reduction import factor_by

                _groups, codes = factor_by(self._obj[by].to_numpy())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            else:
                codes = by

            data = reduce_data_grouped(
                self._obj,
                mom_ndim=self._mom_ndim,
                by=codes,
                axis=axis,
                dim=dim,
                move_axis_to_end=move_axis_to_end,
                parallel=parallel,
                dtype=dtype,
                out=out,
                casting=casting,
                order=order,
                group_dim=group_dim,
                groups=groups,
                keep_attrs=keep_attrs,
                on_missing_core_dim=on_missing_core_dim,
                apply_ufunc_kwargs=apply_ufunc_kwargs,
            )

        return self._new_like(data)

    # ** Constructors ----------------------------------------------------------
    @classmethod
    @docfiller.decorate
    def zeros(  # type: ignore[override]
        cls,
        *,
        mom: Moments,
        val_shape: tuple[int, ...] | int | None = None,
        dtype: DTypeLike = None,
        order: ArrayOrderCF | None = None,
        dims: DimsType = None,
        mom_dims: MomDims | None = None,
        attrs: AttrsType = None,
        coords: CoordsType = None,
        name: NameType = None,
        template: xr.DataArray | None = None,
    ) -> CentralMomentsDataArray:
        """
        Create a new object with specified sizes.

        This can only be used to create a wrapped :class:`~xarray.DataArray` object.
        To create a new wrapped :class:`~xarray.Dataset`, use :func:`xarray.zeros_like`.

        Parameters
        ----------
        {mom}
        {val_shape}
        {dtype}
        {order_cf}
        {xr_params}

        Returns
        -------
        {klass}
            New instance with zero values.

        See Also
        --------
        cmomy.CentralMomentsArray.zeros
        cmomy.CentralMomentsArray.to_x
        """
        from .wrap_np import CentralMomentsArray

        return CentralMomentsArray.zeros(
            mom=mom,
            val_shape=val_shape,
            dtype=dtype,
            order=order,
        ).to_x(
            dims=dims,
            mom_dims=mom_dims,
            attrs=attrs,
            coords=coords,
            name=name,
            template=template,
        )

    # ** Xarray specific stuff ------------------------------------------------
    def to_dataset(
        self,
        dim: Hashable | None = None,
        *,
        name: Hashable | None = None,
        promote_attrs: bool = False,
    ) -> CentralMomentsDataset:
        """
        Convert to Dataset or wrapped Dataset.


        Parameters
        ----------
        dim : Hashable, optional
            Name of the dimension on this array along which to split this array
            into separate variables. If not provided, this array is converted
            into a Dataset of one variable.
        name : Hashable, optional
            Name to substitute for this array's name. Only valid if ``dim`` is
            not provided.
        promote_attrs : bool, default: False
            Set to True to shallow copy attrs of DataArray to returned Dataset.

        Returns
        -------
        CentralMomentsData
            Object wrapping Dataset.

        See Also
        --------
        xarray.DataArray.to_dataset
        """
        if is_dataset(self._obj):
            return self  # pyright: ignore[reportReturnType]

        obj = self._obj.to_dataset(
            dim=dim, name=name, promote_attrs=promote_attrs
        ).transpose(..., *self._mom_dims)

        return type(self)(  # type: ignore[return-value]
            obj=obj,  # type: ignore[arg-type]
            mom_ndim=self._mom_ndim,
            mom_dims=self._mom_dims,
            fastpath=True,
        )

    def to_dataarray(
        self,
        dim: Hashable = "variable",
        *,
        name: Hashable | None = None,
    ) -> CentralMomentsDataArray:
        """
        Convert wrapped Dataset to DataArray

        The data variables of this dataset will be broadcast against each other
        and stacked along the first axis of the new array. All coordinates of
        this dataset will remain coordinates.

        Parameters
        ----------
        dim : Hashable, default: "variable"
            Name of the new dimension.
        name : Hashable or None, optional
            Name of the new data array.

        Returns
        -------
        CentralMomentsData
            New object wrapping DataArray


        See Also
        --------
        xarray.Dataset.to_dataarray
        """
        if is_dataarray(self._obj):
            return self  # pyright: ignore[reportReturnType]

        obj = self._obj.to_array(dim=dim, name=name).transpose(..., *self._mom_dims)
        return type(self)(  # type: ignore[return-value]
            obj=obj,  # type: ignore[arg-type]
            mom_ndim=self._mom_ndim,
            mom_dims=self._mom_dims,
            fastpath=True,
        )

    @property
    def attrs(self) -> dict[Any, Any]:
        """Attributes of values."""
        return self._obj.attrs

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """
        Dimension names.

        Note that for a wrapped Dataset, ``self.obj.dims`` is a mapping, while
        ``self.dims`` is always a tuple of names.
        """
        obj = self._obj
        if is_dataarray(obj):
            return obj.dims
        return tuple(obj.dims)

    @property
    def val_dims(self) -> tuple[Hashable, ...]:
        """Names of value (i.e., not moment) dimensions."""
        if is_dataset(self._obj):
            self._raise_notimplemented_for_dataset()
        return tuple(d for d in self.dims if d not in self.mom_dims)

    @property
    def coords(self) -> DataArrayCoordinates[Any] | DatasetCoordinates:
        """Coordinates of values."""
        return self._obj.coords  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

    @property
    def name(self) -> Hashable:
        """Name of values."""
        if is_dataset(self._obj):
            self._raise_notimplemented_for_dataset()
        return self._obj.name

    @property
    def sizes(self) -> Mapping[Hashable, int]:
        """Sizes of values."""
        return self._obj.sizes

    def compute(self, **kwargs: Any) -> Self:
        """Interface to :meth:`xarray.DataArray.compute` and :meth:`xarray.Dataset.compute`"""
        return self._new_like(self._obj.compute(**kwargs))  # pyright: ignore[reportUnknownMemberType]

    def chunk(self, *args: Any, **kwargs: Any) -> Self:
        """Interface to :meth:`xarray.DataArray.chunk` and :meth:`xarray.Dataset.chunk`"""
        return self._new_like(self._obj.chunk(*args, **kwargs))  # pyright: ignore[reportUnknownMemberType]

    def as_numpy(self) -> Self:
        """Coerces wrapped data and coordinates into numpy arrays."""
        return self._new_like(self._obj.as_numpy())

    def assign(
        self, variables: Mapping[Any, Any] | None = None, **variables_kwargs: Any
    ) -> Self:
        """Assign new variable to and return new object."""
        if is_dataarray(self._obj):
            self._raise_not_implemented("`assign`")
        return self.new_like(obj=self._obj.assign(variables, **variables_kwargs))

    def assign_coords(self, coords: CoordsType = None, **coords_kwargs: Any) -> Self:
        """Assign coordinates to data and return new object."""
        return self._new_like(
            self._obj.assign_coords(coords, **coords_kwargs),  # pyright: ignore[reportUnknownMemberType]
        )

    def assign_attrs(self, *args: Any, **kwargs: Any) -> Self:
        """Assign attributes to data and return new object."""
        return self._new_like(
            self._obj.assign_attrs(*args, **kwargs),
        )

    def rename(
        self,
        new_name_or_name_dict: Hashable | Mapping[Any, Hashable] | None = None,
        **names: Hashable,
    ) -> Self:
        """Rename object."""
        return self._new_like(
            self._obj.rename(new_name_or_name_dict, **names)  # type: ignore[arg-type]
        )

    def stack(
        self,
        dim: Mapping[Any, Sequence[Hashable]] | None = None,
        *,
        _reorder: bool = True,
        _copy: bool | None = False,
        _verify: bool = False,
        _fastpath: bool = False,
        **dimensions_kwargs: Any,
    ) -> Self:
        """
        Stack dimensions.

        Returns
        -------
        output : CentralMomentsData
            With dimensions stacked.

        See Also
        --------
        pipe
        xarray.DataArray.stack

        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.default_rng(0)
        >>> vals = xr.DataArray(rng.random((10, 2, 3)), dims=["rec", "dim_0", "dim_1"])
        >>> da = CentralMomentsData.from_vals(vals, mom=2, dim="rec")
        >>> da
        <CentralMomentsData(mom_ndim=1)>
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
        <CentralMomentsData(mom_ndim=1)>
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
        <CentralMomentsData(mom_ndim=1)>
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
            dim,
            _reorder=_reorder,
            _copy=_copy,
            _verify=_verify,
            _fastpath=_fastpath,
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
        _verify: bool = False,
        _fastpath: bool = False,
    ) -> Self:
        """
        Unstack dimensions.

        Returns
        -------
        output : CentralMomentsData
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
            _verify=_verify,
            _fastpath=_fastpath,
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
        _verify: bool = False,
        _fastpath: bool = False,
        **indexes_kwargs: Hashable | Sequence[Hashable],
    ) -> Self:
        """
        Interface to :meth:`xarray.DataArray.set_index`

        Returns
        -------
        output : CentralMomentsData
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
            _verify=_verify,
            _fastpath=_fastpath,
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
        _verify: bool = False,
        _fastpath: bool = False,
    ) -> Self:
        """Interface to :meth:`xarray.DataArray.reset_index`."""
        return self.pipe(
            "reset_index",
            _reorder=_reorder,
            _copy=_copy,
            _verify=_verify,
            _fastpath=_fastpath,
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
        _verify: bool = False,
        _fastpath: bool = False,
    ) -> Self:
        """Interface to :meth:`xarray.DataArray.drop_vars`"""
        return self.pipe(
            "drop_vars",
            names=names,
            errors=errors,
            _reorder=_reorder,
            _copy=_copy,
            _verify=_verify,
            _fastpath=_fastpath,
        )

    def swap_dims(
        self,
        dims_dict: Mapping[Any, Hashable] | None = None,
        _reorder: bool = True,
        _copy: bool | None = False,
        _verify: bool = False,
        _fastpath: bool = False,
        **dims_kwargs: Any,
    ) -> Self:
        """Interface to :meth:`xarray.DataArray.swap_dims`."""
        return self.pipe(
            "swap_dims",
            dims_dict=dims_dict,
            _reorder=_reorder,
            _copy=_copy,
            _verify=_verify,
            _fastpath=_fastpath,
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
        _verify: bool = False,
        _fastpath: bool = False,
        **indexers_kws: Any,
    ) -> Self:
        """
        Select subset of data.

        Returns
        -------
        output : CentralMomentsData
            With dimensions unstacked

        See Also
        --------
        xarray.DataArray.sel

        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.default_rng(0)
        >>> da = cmomy.CentralMomentsArray.from_vals(
        ...     rng.random((10, 3)), axis=0, mom=2
        ... ).to_x(dims="x", coords=dict(x=list("abc")))
        >>> da
        <CentralMomentsData(mom_ndim=1)>
        <xarray.DataArray (x: 3, mom_0: 3)> Size: 72B
        array([[10.    ,  0.5248,  0.1106],
               [10.    ,  0.5688,  0.0689],
               [10.    ,  0.5094,  0.1198]])
        Coordinates:
          * x        (x) <U1 12B 'a' 'b' 'c'
        Dimensions without coordinates: mom_0

        Select by value

        >>> da.sel(x="a")
        <CentralMomentsData(mom_ndim=1)>
        <xarray.DataArray (mom_0: 3)> Size: 24B
        array([10.    ,  0.5248,  0.1106])
        Coordinates:
            x        <U1 4B 'a'
        Dimensions without coordinates: mom_0
        >>> da.sel(x=["a", "c"])
        <CentralMomentsData(mom_ndim=1)>
        <xarray.DataArray (x: 2, mom_0: 3)> Size: 48B
        array([[10.    ,  0.5248,  0.1106],
               [10.    ,  0.5094,  0.1198]])
        Coordinates:
          * x        (x) <U1 8B 'a' 'c'
        Dimensions without coordinates: mom_0


        Select by position

        >>> da.isel(x=0)
        <CentralMomentsData(mom_ndim=1)>
        <xarray.DataArray (mom_0: 3)> Size: 24B
        array([10.    ,  0.5248,  0.1106])
        Coordinates:
            x        <U1 4B 'a'
        Dimensions without coordinates: mom_0
        >>> da.isel(x=[0, 1])
        <CentralMomentsData(mom_ndim=1)>
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
            _verify=_verify,
            _fastpath=_fastpath,
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
        _verify: bool = False,
        _fastpath: bool = False,
        **indexers_kws: Any,
    ) -> Self:
        """
        Select subset of data by position.

        Returns
        -------
        output : CentralMomentsData
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
            _verify=_verify,
            _fastpath=_fastpath,
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
        _verify: bool = False,
        _fastpath: bool = False,
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
        xarray.Dataset.transpose
        """
        # make sure dims are last

        dims = tuple(d for d in dims if d not in self.mom_dims) + self.mom_dims
        kws: dict[str, bool] = (
            {"transpose_coords": transpose_coords} if is_dataarray(self._obj) else {}
        )

        return self.pipe(
            "transpose",
            *dims,
            **kws,
            missing_dims=missing_dims,
            _reorder=False,
            _copy=_copy,
            _verify=_verify,
            _fastpath=_fastpath,
        )

    # ** To/from CentralMomentsArray ------------------------------------------
    def to_array(self, copy: bool = False) -> CentralMomentsArray[Any]:
        """Convert to :class:`cmomy.CentralMomentsArray` object if possible."""
        from .wrap_np import CentralMomentsArray

        if is_dataset(self.obj):
            self._raise_notimplemented_for_dataset()

        obj = self._obj.to_numpy()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        return CentralMomentsArray(
            obj.copy() if copy else obj,  # pyright: ignore[reportUnknownArgumentType]
            mom_ndim=self._mom_ndim,
            fastpath=True,
        )

    def to_c(self, copy: bool = False) -> CentralMomentsArray[Any]:
        """Alias to :meth:`to_array`."""
        return self.to_array(copy=copy)

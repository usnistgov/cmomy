"""Wrapper object for dataarrays"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr
from module_utilities import cached

from cmomy.core.validate import validate_floating_dtype, validate_mom_and_mom_ndim

from .core.missing import MISSING
from .core.prepare import (
    prepare_data_for_reduction,
    prepare_values_for_reduction,
    xprepare_values_for_reduction,
)
from .core.validate import (
    are_same_type,
    is_dataarray,
    is_dataset,
    is_xarray,
    validate_mom_dims,
)
from .core.xr_utils import (
    contains_dims,
    get_apply_ufunc_kwargs,
    get_mom_shape,
    select_axis_dim,
    select_ndat,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
    from typing import Any, Literal

    from numpy.typing import ArrayLike, DTypeLike
    from xarray.core import types as xr_types
    from xarray.core.coordinates import DataArrayCoordinates
    from xarray.core.indexes import Indexes

    from ._wrapper_numpy import CentralWrapperNumpy
    from .core.typing import (
        ApplyUFuncKwargs,
        ArrayOrderCF,
        AxisReduce,
        CoordsPolicy,
        Dims,
        DimsReduce,
        Groups,
        KeepAttrs,
        MissingCoreDimOptions,
        MissingType,
        Mom_NDim,
        MomDims,
        MomDimsStrict,
        Moments,
        MomentsStrict,
        NDArrayAny,
        XArrayAttrsType,
        XArrayCoordsType,
        XArrayDimsType,
        XArrayIndexesType,
        XArrayNameType,
    )
    from .core.typing_compat import Self


from ._wrapper_abc import CentralWrapperABC
from .core.docstrings import docfiller
from .core.typing import GenXArrayT

docfiller_abc = docfiller.factory_from_parent(CentralWrapperABC)
docfiller_inherit_abc = docfiller.factory_inherit_from_parent(CentralWrapperABC)


@docfiller.inherit(CentralWrapperABC)  # noqa: PLR0904
class CentralWrapperXArray(CentralWrapperABC[GenXArrayT]):
    _mom_dims: MomDimsStrict

    def __init__(
        self,
        obj: GenXArrayT,
        mom_ndim: Mom_NDim = 1,
        mom_dims: MomDims | None = None,
        fastpath: bool = False,
    ) -> None:
        if fastpath:
            if mom_dims is None:
                msg = "Must specify mom_dims with fastpath"
                raise ValueError(msg)
            self._mom_dims = mom_dims  # type: ignore[assignment]
        else:
            self._mom_dims = validate_mom_dims(mom_dims, mom_ndim, obj)

        if not is_xarray(obj):
            msg = "obj must be a DataArray or Dataset, not {type(obj)}"
            raise TypeError(msg)
        # NOTE: Why this ignore?
        super().__init__(obj=obj, mom_ndim=mom_ndim)  # type: ignore[arg-type]

    # ** Properties ------------------------------------------------------------
    @property
    def mom_shape(self) -> MomentsStrict:
        return get_mom_shape(self._obj, self._mom_dims)

    @property
    def mom_dims(self) -> MomDimsStrict:
        return self._mom_dims

    @overload
    def __getitem__(
        self: CentralWrapperXArray[xr.DataArray],
        key: Any,
    ) -> CentralWrapperXArray[xr.DataArray]: ...
    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self: CentralWrapperXArray[xr.Dataset],
        key: Hashable,
    ) -> CentralWrapperXArray[xr.DataArray]: ...
    @overload
    def __getitem__(
        self: CentralWrapperXArray[xr.Dataset],
        key: Any,
    ) -> CentralWrapperXArray[xr.Dataset]: ...

    def __getitem__(
        self, key: Any
    ) -> CentralWrapperXArray[xr.DataArray] | CentralWrapperXArray[xr.Dataset]:
        obj: xr.DataArray | xr.Dataset = self._obj[key]
        if not contains_dims(obj, self._mom_dims):
            msg = f"Cannot select object without {self._mom_dims}"
            raise ValueError(msg)
        self._raise_if_wrong_mom_shape(get_mom_shape(obj, self._mom_dims))
        return type(self)(
            obj, mom_ndim=self._mom_ndim, mom_dims=self._mom_dims, fastpath=True
        )  # type: ignore[arg-type]

    # ** Create/copy/new ------------------------------------------------------
    def _new_like(self, obj: GenXArrayT) -> Self:
        return type(self)(
            obj=obj,
            mom_ndim=self._mom_ndim,
            mom_dims=self._mom_dims,
            fastpath=True,
        )

    @docfiller_inherit_abc()
    def new_like(
        self,
        obj: NDArrayAny | xr.DataArray | xr.Dataset | None = None,
        *,
        copy: bool | None = None,
        deep: bool = True,
        verify: bool = False,
        dtype: DTypeLike = None,
        fastpath: bool = False,
    ) -> Self:
        if obj is None:
            obj: GenXArrayT = xr.zeros_like(self._obj, dtype=dtype)
            copy = False
            dtype = None
        elif isinstance(obj, np.ndarray):
            if is_dataset(self._obj):
                msg = "Can only pass an array for wrapped DataArray."
                raise TypeError(msg)
            obj = self._obj.copy(data=obj)

        assert is_xarray(obj)  # noqa: S101
        if are_same_type(self._obj, obj):
            msg = f"Can only pass in objects conformable to {type(self._obj)}"
            raise TypeError(msg)

        # minimal check on shape and that mom_dims are present....
        self._raise_if_wrong_mom_shape(get_mom_shape(obj, self._mom_dims))
        if not contains_dims(obj, self._mom_dims):
            msg = f"Cannot create new from object without {self._mom_dims}"
            raise ValueError(msg)

        if verify and self._obj.sizes != obj.sizes:
            msg = f"{self.obj.sizes=} != {obj.sizes=}"
            raise ValueError(msg)

        if not fastpath:
            copy = False if copy is None else copy
            if dtype:
                obj = obj.astype(dtype, copy=copy)
            elif copy:
                obj = obj.copy(deep=deep)

        return type(self)(
            obj=obj,
            mom_ndim=self._mom_ndim,
            mom_dims=self._mom_dims,
            fastpath=fastpath,
        )

    @docfiller_inherit_abc()
    def copy(self, deep: bool = True) -> Self:
        return type(self)(
            obj=self._obj.copy(deep=deep),
            mom_ndim=self._mom_ndim,
            mom_dims=self._mom_dims,
            fastpath=True,
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
    @cached.prop
    def _dtype(self) -> np.dtype[Any] | None:
        if is_dataarray(self._obj):
            return self._obj.dtype
        return None

    def _push_data_dataarray(
        self,
        obj: GenXArrayT,
        data: GenXArrayT | ArrayLike,
        *,
        parallel: bool | None = None,
        keep_attrs: KeepAttrs = True,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> GenXArrayT:
        def func(out: NDArrayAny, data: NDArrayAny) -> NDArrayAny:
            self._pusher(parallel).data(data, out)
            return out

        return xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            func,
            obj,
            data,
            input_core_dims=[self.mom_dims, self.mom_dims],
            output_core_dims=[self.mom_dims],
            keep_attrs=keep_attrs,
            **get_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                on_missing_core_dim=on_missing_core_dim,
                dask="parallelized",
                output_dtypes=self._dtype or np.float64,
            ),
        )

    def _push_datas_dataarray(
        self,
        obj: GenXArrayT,
        datas: GenXArrayT | ArrayLike,
        *,
        axis: AxisReduce | MissingType,
        dim: DimsReduce | MissingType,
        parallel: bool | None,
        keep_attrs: KeepAttrs = True,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> GenXArrayT:
        def func(out: NDArrayAny, datas: NDArrayAny) -> NDArrayAny:
            self._pusher(parallel).datas(datas, out)
            return out

        if is_xarray(datas):
            axis, dim = select_axis_dim(
                datas, axis=axis, dim=dim, mom_ndim=self._mom_ndim
            )
            if self._dtype is not None:
                datas = datas.astype(self._dtype, copy=False)

        elif is_dataarray(obj):
            axis, datas = prepare_data_for_reduction(
                datas,
                axis=axis,
                mom_ndim=self._mom_ndim,
                dtype=self._dtype,
                move_axis_to_end=True,
            )
            dim = "_dummy123"
        else:
            msg = "Must pass xarray object of same type as `self.obj` or arraylike if `self.obj` is a dataarray."
            raise ValueError(msg)

        return xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            func,
            obj,
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

    def _push_val_dataarray(
        self,
        obj: GenXArrayT,
        x: ArrayLike | xr.DataArray | xr.Dataset,
        *y: ArrayLike | xr.DataArray | xr.Dataset,
        weight: ArrayLike | xr.DataArray | xr.Dataset | None = None,
        parallel: bool | None = None,
        keep_attrs: KeepAttrs = True,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> GenXArrayT:
        self._check_y(y, self._mom_ndim)

        def func(
            out: NDArrayAny,
            *args: NDArrayAny,
        ) -> NDArrayAny:
            self._pusher(parallel).val(out, *args)
            return out

        weight = 1.0 if weight is None else weight
        core_dims: list[Any] = [self.mom_dims, *([[]] * (2 + len(y)))]
        return xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            func,
            obj,
            x,
            weight,
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

    def _push_vals_dataarray(
        self,
        obj: GenXArrayT,
        x: ArrayLike | xr.DataArray | xr.Dataset,
        *y: ArrayLike | xr.DataArray | xr.Dataset,
        weight: ArrayLike | xr.DataArray | xr.Dataset | None = None,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        parallel: bool | None = None,
        keep_attrs: KeepAttrs = True,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self:
        self._check_y(y, self._mom_ndim)
        weight = 1.0 if weight is None else weight
        if is_xarray(x):
            dim, input_core_dims, xargs = xprepare_values_for_reduction(
                x,
                weight,
                *y,
                axis=axis,
                dim=dim,
                dtype=self._dtype,
                narrays=self._mom_ndim + 1,
            )
        else:
            axis, xargs = prepare_values_for_reduction(
                np.asarray(x, dtype=self._dtype),
                weight,
                *y,
                axis=axis,
                dtype=self._dtype,
                narrays=self._mom_ndim + 1,
                move_axis_to_end=True,
            )
            dim = "_dummy123"
            input_core_dims = [[dim]] * len(xargs)

        def func(
            out: NDArrayAny,
            *args: NDArrayAny,
        ) -> NDArrayAny:
            self._pusher(parallel).vals(out, *args)
            return out

        return xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            func,
            obj,
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

    @docfiller_inherit_abc()
    def push_data(
        self,
        data: GenXArrayT | ArrayLike,
        *,
        parallel: bool | None = False,
    ) -> Self:
        self._obj = self._push_data_dataarray(self._obj, data, parallel=parallel)
        return self

    @docfiller_inherit_abc()
    def push_datas(
        self,
        datas: GenXArrayT | ArrayLike,
        *,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        parallel: bool | None = None,
    ) -> Self:
        self._obj = self._push_datas_dataarray(
            self._obj, datas, axis=axis, dim=dim, parallel=parallel
        )
        return self

    @docfiller_inherit_abc()
    def push_val(
        self,
        x: ArrayLike | xr.DataArray | xr.Dataset,
        *y: ArrayLike | xr.DataArray | xr.Dataset,
        weight: ArrayLike | xr.DataArray | xr.Dataset | None = None,
        parallel: bool | None = False,
    ) -> Self:
        self._obj = self._push_val_dataarray(
            self._obj, x, *y, weight=weight, parallel=parallel
        )
        return self

    @docfiller_inherit_abc()
    def push_vals(
        self,
        x: ArrayLike | xr.DataArray | xr.Dataset,
        *y: ArrayLike | xr.DataArray | xr.Dataset,
        weight: ArrayLike | xr.DataArray | xr.Dataset | None = None,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        parallel: bool | None = None,
    ) -> Self:
        self._obj = self._push_vals_dataarray(
            self._obj,
            x,
            *y,
            weight=weight,
            axis=axis,
            dim=dim,
            parallel=parallel,
        )
        return self

    # ** Operators ------------------------------------------------------------
    def _check_other_conformable(self, other: Self) -> None:
        if self.obj.sizes != other.obj.sizes:
            msg = "shape input={self.obj.shape} != other shape = {other.obj.shape}"
            raise ValueError(msg)

    # ** Access to underlying statistics --------------------------------------
    def std(self, squeeze: bool = True) -> xr.DataArray:
        if is_dataarray(self._obj):
            return np.sqrt(self.var(squeeze=squeeze))
        msg = "Not implemented for Dataset"
        raise NotImplementedError(msg)

    # ** Interface to .reduction ----------------------------------------------
    @docfiller_inherit_abc()
    def reduce(
        self,
        *,
        by: str | Groups | None = None,
        axis: AxisReduce | MissingType = MISSING,
        keepdims: bool = False,
        move_axis_to_end: bool = False,
        parallel: bool | None = None,
        dtype: DTypeLike = None,
        # xarray specific
        use_reduce: bool = False,
        coords_policy: CoordsPolicy = "first",
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
        if by is None:
            from .reduction import reduce_data

            data = reduce_data(
                self._obj,
                mom_ndim=self._mom_ndim,
                axis=axis,
                dim=dim,
                parallel=parallel,
                keep_attrs=bool(keep_attrs),
                dtype=dtype,
                use_reduce=use_reduce,
                keepdims=keepdims,
                on_missing_core_dim=on_missing_core_dim,
                apply_ufunc_kwargs=apply_ufunc_kwargs,
            )

        elif coords_policy == "group":
            from .reduction import reduce_data_grouped

            codes: ArrayLike
            if isinstance(by, str):
                from .reduction import factor_by

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
                group_dim=group_dim,
                groups=groups,
                keep_attrs=keep_attrs,
                on_missing_core_dim=on_missing_core_dim,
                apply_ufunc_kwargs=apply_ufunc_kwargs,
            )

        else:
            from .reduction import factor_by_to_index, reduce_data_indexed

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
                coords_policy=coords_policy,
                group_dim=group_dim,
                groups=groups,
                keep_attrs=keep_attrs,
            )

        return self._new_like(data)

    # ** Constructors ----------------------------------------------------------
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
    ) -> CentralWrapperXArray[xr.DataArray]:
        from ._wrapper_numpy import CentralWrapperNumpy

        return CentralWrapperNumpy.zeros(
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
            indexes=indexes,
            template=template,
        )

    @classmethod
    @docfiller.decorate
    def from_vals(
        cls,
        x: GenXArrayT,
        *y: ArrayLike | xr.DataArray | xr.Dataset,
        mom: Moments,
        weight: ArrayLike | xr.DataArray | xr.Dataset | None = None,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        mom_dims: MomDims | None = None,
        keepdims: bool = False,
        dtype: DTypeLike = None,
        out: NDArrayAny | None = None,
        parallel: bool | None = None,
        keep_attrs: KeepAttrs = None,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self:
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
        {mom}
        {axis_and_dim}
        {mom_dims}
        {keepdims}
        {order}
        {dtype}
        {parallel}
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

        mom, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
        mom_dims = validate_mom_dims(mom_dims, mom_ndim)
        return cls(
            obj=reduce_vals(
                x,
                *y,
                mom=mom,
                weight=weight,
                axis=axis,
                dim=dim,
                mom_dims=mom_dims,
                keepdims=keepdims,
                parallel=parallel,
                dtype=dtype,
                out=out,
                keep_attrs=keep_attrs,
                on_missing_core_dim=on_missing_core_dim,
                apply_ufunc_kwargs=apply_ufunc_kwargs,
            ),
            mom_ndim=mom_ndim,
            mom_dims=mom_dims,
        )

    @classmethod
    @docfiller.decorate
    def from_resample_vals(  # noqa: PLR0913
        cls,
        x: GenXArrayT,
        *y: ArrayLike | xr.DataArray | xr.Dataset,
        mom: Moments,
        weight: ArrayLike | xr.DataArray | xr.Dataset | None = None,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        freq: ArrayLike | xr.DataArray | xr.Dataset | None = None,
        nrep: int | None = None,
        rng: np.random.Generator | None = None,
        move_axis_to_end: bool = True,
        parallel: bool | None = None,
        dtype: DTypeLike = None,
        out: NDArrayAny | None = None,
        mom_dims: MomDims | None = None,
        rep_dim: str = "rep",
        keep_attrs: bool = True,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self:
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
        {weight}
        {axis_and_dim}
        {freq}
        {nrep}
        {rng}
        {move_axis_to_end}
        {order}
        {parallel}
        {dtype}
        {out}
        {mom_dims}
        {rep_dim}
        {keep_attrs}
        {on_missing_core_dim}
        {apply_ufunc_kwargs}

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
        mom, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
        mom_dims = validate_mom_dims(mom_dims, mom_ndim)

        from .resample import resample_vals

        return cls(
            obj=resample_vals(
                x,
                *y,
                freq=freq,
                nrep=nrep,
                rng=rng,
                mom=mom,
                weight=weight,
                axis=axis,
                dim=dim,
                move_axis_to_end=move_axis_to_end,
                parallel=parallel,
                mom_dims=mom_dims,
                rep_dim=rep_dim,
                keep_attrs=keep_attrs,
                on_missing_core_dim=on_missing_core_dim,
                apply_ufunc_kwargs=apply_ufunc_kwargs,
                dtype=dtype,
                out=out,
            ),
            mom_ndim=mom_ndim,
            mom_dims=mom_dims,
        )

    @classmethod
    @docfiller.decorate
    def from_raw(
        cls,
        raw: GenXArrayT,
        *,
        mom_ndim: Mom_NDim = 1,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        keep_attrs: KeepAttrs = None,
        mom_dims: MomDims | None = None,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self:
        from . import convert

        mom_dims = validate_mom_dims(mom_dims, mom_ndim)
        return cls(
            obj=convert.moments_type(
                raw,
                mom_ndim=mom_ndim,
                to="central",
                out=out,
                dtype=dtype,
                keep_attrs=keep_attrs,
                mom_dims=mom_dims,
                on_missing_core_dim=on_missing_core_dim,
                apply_ufunc_kwargs=apply_ufunc_kwargs,
            ),
            mom_dims=mom_dims,
            mom_ndim=mom_ndim,
        )

    # ** Xarray specific stuff ------------------------------------------------
    @property
    def attrs(self) -> dict[Any, Any]:
        """Attributes of values."""
        return self._obj.attrs

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """Dimensions of values."""
        return self._obj.dims

    @property
    def coords(self) -> DataArrayCoordinates[Any]:
        """Coordinates of values."""
        return self._obj.coords  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

    @property
    def name(self) -> Hashable:
        """Name of values."""
        return self._obj.name

    @property
    def indexes(self) -> Indexes[Any]:  # pragma: no cover
        """Indexes of values."""
        return self._obj.indexes  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

    @property
    def sizes(self) -> Mapping[Hashable, int]:
        """Sizes of values."""
        return self._obj.sizes

    def compute(self, **kwargs: Any) -> Self:
        """Interface to :meth:`xarray.DataArray.compute` and :meth:`xarray.Dataset.compute`"""
        return self._new_like(self._obj.compute(**kwargs))

    def chunk(self, *args: Any, **kwargs: Any) -> Self:
        """Interface to :meth:`xarray.DataArray.chunk` and :meth:`xarray.Dataset.chunk`"""
        return self._new_like(self._obj.chunk(*args, **kwargs))

    def assign_coords(
        self, coords: XArrayCoordsType = None, **coords_kwargs: Any
    ) -> Self:
        """Assign coordinates to data and return new object."""
        return self._new_like(
            self._obj.assign_attrs(coords, **coords_kwargs),
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
            dim,
            _reorder=_reorder,
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
        _verify: bool = False,
    ) -> Self:
        """Interface to :meth:`xarray.DataArray.reset_index`."""
        return self.pipe(
            "reset_index",
            _reorder=_reorder,
            _copy=_copy,
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
        _verify: bool = False,
    ) -> Self:
        """Interface to :meth:`xarray.DataArray.drop_vars`"""
        return self.pipe(
            "drop_vars",
            names=names,
            errors=errors,
            _reorder=_reorder,
            _copy=_copy,
            _verify=_verify,
        )

    def swap_dims(
        self,
        dims_dict: Mapping[Any, Hashable] | None = None,
        _reorder: bool = True,
        _copy: bool | None = False,
        _verify: bool = False,
        **dims_kwargs: Any,
    ) -> Self:
        """Interface to :meth:`xarray.DataArray.swap_dims`."""
        return self.pipe(
            "swap_dims",
            dims_dict=dims_dict,
            _reorder=_reorder,
            _copy=_copy,
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
            _verify=_verify,
        )

    # ** To/from CentralWrapperNumpy ------------------------------------------
    def to_c(self, copy: bool = False) -> CentralWrapperNumpy[Any]:
        from ._wrapper_numpy import CentralWrapperNumpy

        if is_dataarray(self._obj):
            obj = self._obj.to_numpy()
            return CentralWrapperNumpy(
                obj.copy() if copy else obj,
                mom_ndim=self._mom_ndim,
                fastpath=True,
            )
        msg = "Can only convert to numpy wrapper from dataarray."
        raise NotImplementedError(msg)

    # ** Manipulation ---------------------------------------------------------
    @docfiller_inherit_abc()
    def moveaxis(
        self,
        axis: int | tuple[int, ...] | MissingType = MISSING,
        dest: int | tuple[int, ...] | MissingType = MISSING,
        dim: str | Sequence[Hashable] | MissingType = MISSING,
        dest_dim: str | Sequence[Hashable] | MissingType = MISSING,
        **kwargs: Any,
    ) -> Self:
        return super().moveaxis(
            axis=axis, dest=dest, dim=dim, dest_dim=dest_dim, **kwargs
        )

    @docfiller_inherit_abc()
    def resample_and_reduce(
        self,
        *,
        freq: ArrayLike | xr.DataArray | xr.Dataset | None = None,
        nrep: int | None = None,
        rng: np.random.Generator | None = None,
        paired: bool = True,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        rep_dim: str = "rep",
        parallel: bool | None = None,
        dtype: DTypeLike = None,
        out: NDArrayAny | None = None,
        keep_attrs: KeepAttrs = None,
        # dask specific...
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
        **kwargs: Any,
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
        return super().resample_and_reduce(
            freq=freq,
            nrep=nrep,
            rng=rng,
            paired=paired,
            axis=axis,
            dim=dim,
            rep_dim=rep_dim,
            parallel=parallel,
            dtype=dtype,
            out=out,
            keep_attrs=keep_attrs,
            mom_dims=self._mom_dims,
            on_missing_core_dim=on_missing_core_dim,
            apply_ufunc_kwargs=apply_ufunc_kwargs,
            **kwargs,
        )

    @docfiller_inherit_abc()
    def jackknife_and_reduce(
        self,
        *,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        data_reduced: Self | GenXArrayT | None = None,
        rep_dim: str | None = "rep",
        parallel: bool | None = None,
        dtype: DTypeLike = None,
        out: NDArrayAny | None = None,
        keep_attrs: KeepAttrs = None,
        # dask specific...
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Parameters
        ----------
        {rep_dim}
        {keep_attrs}
        """
        return super().jackknife_and_reduce(
            axis=axis,
            dim=dim,
            data_reduced=data_reduced,
            rep_dim=rep_dim,
            parallel=parallel,
            dtype=dtype,
            out=out,
            keep_attrs=keep_attrs,
            on_missing_core_dim=on_missing_core_dim,
            apply_ufunc_kwargs=apply_ufunc_kwargs,
            **kwargs,
        )

    @docfiller.decorate
    def block(
        self,
        block_size: int | None,
        *,
        dim: DimsReduce | MissingType = MISSING,
        axis: AxisReduce | MissingType = MISSING,
        block_dim: str | None = None,
        parallel: bool | None = None,
        keep_attrs: KeepAttrs = None,
        coords_policy: CoordsPolicy = "first",
        **kwargs: Any,
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
        n = select_ndat(self._obj, dim=dim, axis=axis, mom_ndim=self._mom_ndim)
        if block_size is None:
            block_size = n
        nblock = n // block_size

        by = np.arange(nblock).repeat(block_size)
        if len(by) != n:
            by = np.pad(by, (0, n - len(by)), mode="constant", constant_values=-1)

        return self.reduce(
            dim=dim,
            by=by,
            parallel=parallel,
            group_dim=block_dim,
            keep_attrs=keep_attrs,
            coords_policy=coords_policy,
            groups=range(nblock),
            **kwargs,
        )

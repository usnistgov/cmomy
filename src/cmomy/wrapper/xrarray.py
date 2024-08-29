"""Wrapper object for daataarrays"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr
from module_utilities import cached

from cmomy.core.missing import MISSING
from cmomy.core.prepare import (
    prepare_data_for_reduction,
    prepare_values_for_reduction,
    xprepare_values_for_reduction,
)
from cmomy.core.validate import (
    are_same_type,
    is_dataarray,
    is_dataset,
    is_xarray,
    validate_floating_dtype,
    validate_mom_and_mom_ndim,
    validate_mom_dims,
)
from cmomy.core.xr_utils import (
    contains_dims,
    get_apply_ufunc_kwargs,
    get_mom_shape,
    select_axis_dim,
    select_ndat,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Sequence
    from typing import Any, Literal, NoReturn

    from numpy.typing import ArrayLike, DTypeLike
    from xarray.core import types as xr_types
    from xarray.core.coordinates import DataArrayCoordinates, DatasetCoordinates

    from cmomy.core.typing import (
        ApplyUFuncKwargs,
        ArrayOrder,
        ArrayOrderCF,
        AxisReduce,
        Casting,
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
    from cmomy.core.typing_compat import Self
    from cmomy.wrapper.nparray import CentralMoments


from cmomy.core.docstrings import docfiller_xcentral as docfiller
from cmomy.core.typing import XArrayT

from .base import CentralMomentsABC

docfiller_abc = docfiller.factory_from_parent(CentralMomentsABC)
docfiller_inherit_abc = docfiller.factory_inherit_from_parent(CentralMomentsABC)


@docfiller.inherit(CentralMomentsABC)  # noqa: PLR0904
class xCentralMoments(CentralMomentsABC[XArrayT]):  # noqa: N801
    """
    Parameters
    ----------
    {mom_dims_data}
    """

    _mom_dims: MomDimsStrict

    def __init__(
        self,
        obj: XArrayT,
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

    @property
    def dtype(self) -> np.dtype[Any]:
        if is_dataset(self._obj):
            self._raise_notimplemented_for_dataset()
        return self._obj.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        if is_dataset(self._obj):
            self._raise_notimplemented_for_dataset()
        return self._obj.shape

    @property
    def val_shape(self) -> tuple[int, ...]:
        if is_dataset(self._obj):
            self._raise_notimplemented_for_dataset()
        return self._obj.shape[: -self._mom_ndim]

    @overload
    def __getitem__(
        self: xCentralMoments[xr.DataArray],
        key: Any,
    ) -> xCentralMoments[xr.DataArray]: ...
    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self: xCentralMoments[xr.Dataset],
        key: Hashable,
    ) -> xCentralMoments[xr.DataArray]: ...
    @overload
    def __getitem__(
        self: xCentralMoments[xr.Dataset],
        key: Any,
    ) -> xCentralMoments[xr.Dataset]: ...

    def __getitem__(
        self, key: Any
    ) -> xCentralMoments[xr.DataArray] | xCentralMoments[xr.Dataset]:
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
    def _new_like(self, obj: XArrayT) -> Self:
        return type(self)(
            obj=obj,
            mom_ndim=self._mom_ndim,
            mom_dims=self._mom_dims,
            fastpath=True,
        )

    @docfiller_inherit_abc()
    def new_like(  # type: ignore[override]
        self,
        obj: NDArrayAny | XArrayT | None = None,
        *,
        copy: bool | None = None,
        deep: bool = True,
        verify: bool = False,
        dtype: DTypeLike = None,
        fastpath: bool = False,
    ) -> Self:
        """
        Parameters
        ----------
        deep : bool
            Parameter to :meth:`~xarray.Dataset.copy` or :meth:`~xarray.DataArray.copy`.
        """
        # TODO(wpk): edge case of passing in new xarray data with different dimensions.
        # For now, this will raise an error.
        obj_: XArrayT
        if obj is None:
            # TODO(wpk): different type for dtype in xarray.
            obj_ = xr.zeros_like(self._obj, dtype=dtype)  # type: ignore[arg-type]
            copy = False
            dtype = None
        elif isinstance(obj, np.ndarray):
            if is_dataarray(self._obj):
                obj_ = self._obj.copy(data=obj)
            else:
                msg = "Can only pass an array for wrapped DataArray."
                raise TypeError(msg)
        else:
            obj_ = obj

        assert is_xarray(obj_)  # noqa: S101
        if are_same_type(self._obj, obj_):
            msg = f"Can only pass in objects conformable to {type(self._obj)}"
            raise TypeError(msg)

        # minimal check on shape and that mom_dims are present....
        if not contains_dims(obj_, self._mom_dims):
            msg = f"Cannot create new from object without {self._mom_dims}"
            raise ValueError(msg)
        self._raise_if_wrong_mom_shape(get_mom_shape(obj_, self._mom_dims))

        if verify and self._obj.sizes != obj_.sizes:
            msg = f"{self.obj.sizes=} != obj.sizes={obj_.sizes}"
            raise ValueError(msg)

        if not fastpath:
            copy = False if copy is None else copy
            if dtype:
                obj_ = obj_.astype(dtype, copy=copy)  # pyright: ignore[reportUnknownMemberType]
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
        if isinstance(dtype, Mapping):
            if is_dataarray(self._obj):
                msg = "Passing a mapping for `dtype` only allowed for Dataset."
                raise ValueError(msg)
            dtype = dict(self._obj.dtypes, **dtype)

        # Reimplement with a a full new_like to defer dtype check to __init__
        return self.new_like(
            obj=self._obj.astype(
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

    @staticmethod
    def _raise_notimplemented_for_dataset() -> NoReturn:
        msg = "Not implemented for Dataset"
        raise NotImplementedError(msg)

    # ** Pushing --------------------------------------------------------------
    @cached.prop
    def _dtype(self) -> np.dtype[Any] | None:
        if is_dataarray(self._obj):
            return self._obj.dtype  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        return None

    def _push_vals_dataarray(
        self,
        obj: XArrayT,
        x: ArrayLike | xr.DataArray | xr.Dataset,
        *y: ArrayLike | xr.DataArray | xr.Dataset,
        weight: ArrayLike | xr.DataArray | xr.Dataset | None = None,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        parallel: bool | None = None,
        keep_attrs: KeepAttrs = True,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> XArrayT:
        self._check_y(y, self._mom_ndim)
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

        return xr.apply_ufunc(  # type: ignore[no-any-return]
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
        data: XArrayT | ArrayLike,
        *,
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

        def func(out: NDArrayAny, data: NDArrayAny) -> NDArrayAny:
            self._pusher(parallel).data(
                data,
                out,
                casting=casting,
                signature=(out.dtype, out.dtype),
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
        datas: XArrayT | ArrayLike,
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
            self._pusher(parallel).datas(
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

        elif is_dataarray(self._obj):
            axis, datas = prepare_data_for_reduction(
                datas,
                axis=axis,
                mom_ndim=self._mom_ndim,
                dtype=self._dtype,
                move_axis_to_end=True,
                recast=False,
            )
            dim = "_dummy123"
        else:
            msg = "Must pass xarray object of same type as `self.obj` or arraylike if `self.obj` is a dataarray."
            raise ValueError(msg)

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
        x: ArrayLike | XArrayT,
        *y: ArrayLike | xr.DataArray | XArrayT,
        weight: ArrayLike | xr.DataArray | XArrayT | None = None,
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
            self._pusher(parallel).val(
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
        x: ArrayLike | XArrayT,
        *y: ArrayLike | xr.DataArray | XArrayT,
        weight: ArrayLike | xr.DataArray | XArrayT | None = None,
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
        self._check_y(y, self._mom_ndim)
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
            self._pusher(parallel).vals(
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

    # ** Operators ------------------------------------------------------------
    def _check_other_conformable(self, other: Self) -> None:
        if self.obj.sizes != other.obj.sizes:
            msg = "shape input={self.obj.shape} != other shape = {other.obj.shape}"
            raise ValueError(msg)

    # ** Interface to modules -------------------------------------------------
    # *** .utils --------------------------------------------------------------
    @docfiller_inherit_abc()
    def moveaxis(
        self,
        axis: int | tuple[int, ...] | MissingType = MISSING,
        dest: int | tuple[int, ...] | MissingType = MISSING,
        dim: str | Sequence[Hashable] | MissingType = MISSING,
        dest_dim: str | Sequence[Hashable] | MissingType = MISSING,
        **kwargs: Any,
    ) -> Self:
        """
        Parameters
        ----------
        dim : str or sequence of hashable
            Original dimensions to move (for DataArray).
        dest_dim : str or sequence of hashable
            Destination of each original dimension.
        """
        return super().moveaxis(
            axis=axis, dest=dest, dim=dim, dest_dim=dest_dim, **kwargs
        )

    # *** .convert ------------------------------------------------------------
    @docfiller_inherit_abc()
    def moments_to_comoments(  # type: ignore[override]
        self,
        *,
        mom: tuple[int, int],
        mom_dims2: MomDims | None = None,
        keep_attrs: KeepAttrs = None,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self:
        """
        Parameters
        ----------
        mom_dims2 : tuple of str
            Moments dimensions for output (``mom_ndim=2``) data.  Defaults to ``("mom_0", "mom_1")``.
        {keep_attrs}
        {on_missing_core_dim}
        {apply_ufunc_kwargs}
        """
        self._raise_if_not_mom_ndim_1()
        from cmomy import convert

        mom_dims2 = validate_mom_dims(mom_dims2, mom_ndim=2)

        return type(self)(
            convert.moments_to_comoments(  # pyright: ignore[reportArgumentType]
                self._obj,
                mom=mom,
                mom_dims=self._mom_dims,
                mom_dims2=mom_dims2,
                keep_attrs=keep_attrs,
                on_missing_core_dim=on_missing_core_dim,
                apply_ufunc_kwargs=apply_ufunc_kwargs,
            ),
            mom_ndim=2,
            mom_dims=mom_dims2,
        )

    # *** .resample -----------------------------------------------------------
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
        parallel: bool | None = None,
        dtype: DTypeLike = None,
        out: NDArrayAny | None = None,
        rep_dim: str = "rep",
        keep_attrs: KeepAttrs = None,
        # dask specific...
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Parameters
        ----------
        {paired}
        {dtype}
        {out}
        {rep_dim}
        {keep_attrs}
        {on_missing_core_dim}
        {apply_ufunc_kwargs}
        **kwargs
            Extra arguments to :func:`.resample.resample_data`

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
        <xCentralMoments(mom_ndim=1)>
        <xarray.DataArray (rec: 3, mom_0: 4)> Size: 96B
        array([[ 1.0000e+01,  5.2485e-01,  1.1057e-01, -4.6282e-03],
               [ 1.0000e+01,  5.6877e-01,  6.8876e-02, -1.2745e-02],
               [ 1.0000e+01,  5.0944e-01,  1.1978e-01, -1.4644e-02]])
        Dimensions without coordinates: rec, mom_0

        Note that for reproducible results, must set numba random
        seed as well

        >>> freq = cmomy.randsamp_freq(data=da.obj, dim="rec", nrep=5)
        >>> da_resamp = da.resample_and_reduce(
        ...     dim="rec",
        ...     freq=freq,
        ... )
        >>> da_resamp
        <xCentralMoments(mom_ndim=1)>
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
        <xCentralMoments(mom_ndim=1)>
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
        data_reduced: Self | XArrayT | None = None,
        parallel: bool | None = None,
        dtype: DTypeLike = None,
        out: NDArrayAny | None = None,
        rep_dim: str | None = "rep",
        keep_attrs: KeepAttrs = None,
        # dask specific...
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Parameters
        ----------
        {dtype}
        {out}
        {rep_dim}
        {keep_attrs}
        {on_missing_core_dim}
        {apply_ufunc_kwargs}
        """
        return super().jackknife_and_reduce(
            axis=axis,
            dim=dim,
            data_reduced=data_reduced,  # type: ignore[arg-type]
            rep_dim=rep_dim,
            parallel=parallel,
            dtype=dtype,
            out=out,
            keep_attrs=keep_attrs,
            on_missing_core_dim=on_missing_core_dim,
            apply_ufunc_kwargs=apply_ufunc_kwargs,
            **kwargs,
        )

    # *** .reduction ----------------------------------------------------------
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
        out: NDArrayAny | None = None,
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
        {dtype}
        {out}
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
        <xCentralMoments(mom_ndim=1)>
        <xarray.DataArray (dim_1: 3, mom_0: 3)> Size: 72B
        array([[20.    ,  0.5221,  0.0862],
               [20.    ,  0.5359,  0.0714],
               [20.    ,  0.4579,  0.103 ]])
        Dimensions without coordinates: dim_1, mom_0
        """
        if by is None:
            from cmomy.reduction import reduce_data

            data = reduce_data(
                self._obj,
                mom_ndim=self._mom_ndim,
                axis=axis,
                dim=dim,
                parallel=parallel,
                keep_attrs=bool(keep_attrs),
                dtype=dtype,
                out=out,
                use_reduce=use_reduce,
                keepdims=keepdims,
                on_missing_core_dim=on_missing_core_dim,
                apply_ufunc_kwargs=apply_ufunc_kwargs,
            )

        elif coords_policy == "group":
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
                group_dim=group_dim,
                groups=groups,
                keep_attrs=keep_attrs,
                on_missing_core_dim=on_missing_core_dim,
                apply_ufunc_kwargs=apply_ufunc_kwargs,
            )

        else:
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
                coords_policy=coords_policy,
                group_dim=group_dim,
                groups=groups,
                keep_attrs=keep_attrs,
            )

        return self._new_like(data)

    # ** Access to underlying statistics --------------------------------------
    def std(self, squeeze: bool = True) -> xr.DataArray:
        if is_dataarray(self._obj):
            return np.sqrt(self.var(squeeze=squeeze))  # type: ignore[return-value]
        self._raise_notimplemented_for_dataset()
        return None

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
        dims: XArrayDimsType = None,
        mom_dims: MomDims | None = None,
        attrs: XArrayAttrsType = None,
        coords: XArrayCoordsType = None,
        name: XArrayNameType = None,
        indexes: XArrayIndexesType = None,
        template: xr.DataArray | None = None,
    ) -> xCentralMoments[xr.DataArray]:
        from .nparray import CentralMoments

        return CentralMoments.zeros(
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
        x: XArrayT,
        *y: ArrayLike | xr.DataArray | XArrayT,
        mom: Moments,
        weight: ArrayLike | xr.DataArray | XArrayT | None = None,
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
        from cmomy.reduction import reduce_vals

        mom, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
        mom_dims = validate_mom_dims(mom_dims, mom_ndim)
        return cls(
            obj=reduce_vals(
                x,  # pyright: ignore[reportArgumentType]
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
        x: XArrayT,
        *y: ArrayLike | xr.DataArray | XArrayT,
        mom: Moments,
        weight: ArrayLike | xr.DataArray | XArrayT | None = None,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        freq: ArrayLike | xr.DataArray | XArrayT | None = None,
        nrep: int | None = None,
        rng: np.random.Generator | None = None,
        move_axis_to_end: bool = True,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        casting: Casting = "same_kind",
        order: ArrayOrderCF = None,
        parallel: bool | None = None,
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
        {out}
        {dtype}
        {casting}
        {order_cf}
        {parallel}
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

        from cmomy.resample import resample_vals

        return cls(
            obj=resample_vals(
                x,  # pyright: ignore[reportArgumentType]
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
                casting=casting,
                order=order,
            ),
            mom_ndim=mom_ndim,
            mom_dims=mom_dims,
        )

    @classmethod
    @docfiller.decorate
    def from_raw(  # type: ignore[override]
        cls,
        raw: XArrayT,
        *,
        mom_ndim: Mom_NDim = 1,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        keep_attrs: KeepAttrs = None,
        mom_dims: MomDims | None = None,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self:
        from cmomy import convert

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
        obj = self._obj
        if is_dataarray(obj):
            return obj.dims
        return tuple(obj.dims)

    @property
    def coords(self) -> DataArrayCoordinates[Any] | DatasetCoordinates:
        """Coordinates of values."""
        return self._obj.coords  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

    @property
    def name(self) -> Hashable:
        """Name of values."""
        if is_dataarray(self._obj):
            return self._obj.name
        self._raise_notimplemented_for_dataset()
        return None

    @property
    def sizes(self) -> Mapping[Hashable, int]:
        """Sizes of values."""
        return self._obj.sizes

    def compute(self, **kwargs: Any) -> Self:
        """Interface to :meth:`xarray.DataArray.compute` and :meth:`xarray.Dataset.compute`"""
        return self._new_like(self._obj.compute(**kwargs))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

    def chunk(self, *args: Any, **kwargs: Any) -> Self:
        """Interface to :meth:`xarray.DataArray.chunk` and :meth:`xarray.Dataset.chunk`"""
        return self._new_like(self._obj.chunk(*args, **kwargs))  # pyright: ignore[reportUnknownMemberType]

    def assign_coords(
        self, coords: XArrayCoordsType = None, **coords_kwargs: Any
    ) -> Self:
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
        <xCentralMoments(mom_ndim=1)>
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
        <xCentralMoments(mom_ndim=1)>
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
        <xCentralMoments(mom_ndim=1)>
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
        <xCentralMoments(mom_ndim=1)>
        <xarray.DataArray (x: 3, mom_0: 3)> Size: 72B
        array([[10.    ,  0.5248,  0.1106],
               [10.    ,  0.5688,  0.0689],
               [10.    ,  0.5094,  0.1198]])
        Coordinates:
          * x        (x) <U1 12B 'a' 'b' 'c'
        Dimensions without coordinates: mom_0

        Select by value

        >>> da.sel(x="a")
        <xCentralMoments(mom_ndim=1)>
        <xarray.DataArray (mom_0: 3)> Size: 24B
        array([10.    ,  0.5248,  0.1106])
        Coordinates:
            x        <U1 4B 'a'
        Dimensions without coordinates: mom_0
        >>> da.sel(x=["a", "c"])
        <xCentralMoments(mom_ndim=1)>
        <xarray.DataArray (x: 2, mom_0: 3)> Size: 48B
        array([[10.    ,  0.5248,  0.1106],
               [10.    ,  0.5094,  0.1198]])
        Coordinates:
          * x        (x) <U1 8B 'a' 'c'
        Dimensions without coordinates: mom_0


        Select by position

        >>> da.isel(x=0)
        <xCentralMoments(mom_ndim=1)>
        <xarray.DataArray (mom_0: 3)> Size: 24B
        array([10.    ,  0.5248,  0.1106])
        Coordinates:
            x        <U1 4B 'a'
        Dimensions without coordinates: mom_0
        >>> da.isel(x=[0, 1])
        <xCentralMoments(mom_ndim=1)>
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
            _fastpath=_fastpath,
        )

    # ** To/from CentralMoments ------------------------------------------
    def to_c(self, copy: bool = False) -> CentralMoments[Any]:
        from .nparray import CentralMoments

        if is_dataarray(self._obj):
            obj = self._obj.to_numpy()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            return CentralMoments(
                obj.copy() if copy else obj,  # pyright: ignore[reportUnknownArgumentType]
                mom_ndim=self._mom_ndim,
                fastpath=True,
            )
        self._raise_notimplemented_for_dataset()
        return None

    # ** Manipulation ---------------------------------------------------------

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
        <xCentralMoments(mom_ndim=1)>
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
        <xCentralMoments(mom_ndim=1)>
        <xarray.DataArray (dim_0: 2, mom_0: 3)> Size: 48B
        array([[50.    ,  0.5008,  0.0899],
               [50.    ,  0.5958,  0.0893]])
        Dimensions without coordinates: dim_0, mom_0

        This is equivalent to

        >>> cmomy.CentralMoments.from_vals(x.reshape(2, 50), mom=2, axis=1).to_x()
        <xCentralMoments(mom_ndim=1)>
        <xarray.DataArray (dim_0: 2, mom_0: 3)> Size: 48B
        array([[50.    ,  0.5268,  0.0849],
               [50.    ,  0.5697,  0.0979]])
        Dimensions without coordinates: dim_0, mom_0


        The coordinate policy can be useful to keep coordinates:

        >>> da2 = da.assign_coords(dim_0=range(10))
        >>> da2.block(5, dim="dim_0", coords_policy="first")
        <xCentralMoments(mom_ndim=1)>
        <xarray.DataArray (dim_0: 2, mom_0: 3)> Size: 48B
        array([[50.    ,  0.5008,  0.0899],
               [50.    ,  0.5958,  0.0893]])
        Coordinates:
          * dim_0    (dim_0) int64 16B 0 5
        Dimensions without coordinates: mom_0
        >>> da2.block(5, dim="dim_0", coords_policy="last")
        <xCentralMoments(mom_ndim=1)>
        <xarray.DataArray (dim_0: 2, mom_0: 3)> Size: 48B
        array([[50.    ,  0.5008,  0.0899],
               [50.    ,  0.5958,  0.0893]])
        Coordinates:
          * dim_0    (dim_0) int64 16B 4 9
        Dimensions without coordinates: mom_0
        >>> da2.block(5, dim="dim_0", coords_policy=None)
        <xCentralMoments(mom_ndim=1)>
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

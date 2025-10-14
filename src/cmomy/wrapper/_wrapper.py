"""Array, DataArray, and Dataset wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, cast, overload

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from cmomy.core.array_utils import arrayorder_to_arrayorder_cf
from cmomy.core.compat import copy_if_needed
from cmomy.core.docstrings import (
    docfiller_central as docfiller_array,
)
from cmomy.core.docstrings import (
    docfiller_xcentral as docfiller_data,
)
from cmomy.core.missing import MISSING
from cmomy.core.moment_params import (
    MomParamsArray,
    MomParamsXArray,
)
from cmomy.core.prepare import (
    PrepareDataArray,
    prepare_array_values_for_reduction,
    prepare_xarray_values_for_reduction,
)
from cmomy.core.typing import DataT, FloatT
from cmomy.core.typing_compat import override
from cmomy.core.utils import mom_to_mom_shape
from cmomy.core.validate import (
    is_dataarray,
    is_dataset,
    is_ndarray,
    is_xarray,
    is_xarray_typevar,
    raise_if_wrong_value,
    validate_axis,
    validate_floating_dtype,
)
from cmomy.core.xr_utils import (
    astype_dtype_dict,
    contains_dims,
    factory_apply_ufunc_kwargs,
)

from ._wrapper_abc import CentralMomentsABC

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

    from cmomy.core._typing_kwargs import (
        ApplyUFuncKwargs,
        WrapNPReduceKwargs,
        WrapNPResampleAndReduceKwargs,
        WrapNPTransformKwargs,
    )
    from cmomy.core.moment_params import MomParamsType
    from cmomy.core.typing import (
        ArrayLikeArg,
        ArrayOrderACF,
        ArrayOrderCF,
        ArrayOrderKACF,
        AttrsType,
        AxisReduce,
        AxisReduceWrap,
        Casting,
        CoordsPolicy,
        CoordsType,
        Dims,
        DimsReduce,
        DimsType,
        DTypeLikeArg,
        FloatT_,
        Groups,
        KeepAttrs,
        MissingType,
        MomAxes,
        MomAxesStrict,
        MomDims,
        MomDimsStrict,
        Moments,
        MomNDim,
        NameType,
        NDArrayAny,
    )
    from cmomy.core.typing_compat import Self, TypeAlias, Unpack
    from cmomy.resample.typing import SamplerType

    CentralMomentsArrayAny: TypeAlias = "CentralMomentsArray[Any]"
    CentralMomentsDataArray: TypeAlias = "CentralMomentsData[xr.DataArray]"
    CentralMomentsDataset: TypeAlias = "CentralMomentsData[xr.Dataset]"


docfiller_array_abc = docfiller_array.factory_from_parent(CentralMomentsABC)
docfiller_array_inherit_abc = docfiller_array.factory_inherit_from_parent(
    CentralMomentsABC
)


@docfiller_array.inherit(CentralMomentsABC)
class CentralMomentsArray(
    CentralMomentsABC[NDArray[FloatT], MomParamsArray],  # type: ignore[type-var]
    Generic[FloatT],
):
    r"""
    Central moments wrapper of {ndarray} arrays.

    Parameters
    ----------
    {mom_ndim_data}
    {mom_axes}
    {copy}
    {order}
    {dtype}

    """

    # pylint: disable=arguments-differ
    _mom_params: MomParamsArray

    @overload
    def __init__(
        self,
        obj: ArrayLikeArg[FloatT],
        *,
        mom_ndim: MomNDim | None = ...,
        mom_axes: MomAxes | None = ...,
        copy: bool | None = ...,
        dtype: None = ...,
        order: ArrayOrderKACF = ...,
        mom_params: MomParamsType = ...,
        fastpath: bool = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        obj: ArrayLike,
        *,
        mom_ndim: MomNDim | None = ...,
        mom_axes: MomAxes | None = ...,
        copy: bool | None = ...,
        dtype: DTypeLikeArg[FloatT],
        order: ArrayOrderKACF = ...,
        mom_params: MomParamsType = ...,
        fastpath: bool = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        obj: ArrayLike,
        *,
        mom_ndim: MomNDim | None = ...,
        mom_axes: MomAxes | None = ...,
        copy: bool | None = ...,
        dtype: DTypeLike = ...,
        order: ArrayOrderKACF = ...,
        mom_params: MomParamsType = ...,
        fastpath: bool = ...,
    ) -> None: ...

    def __init__(
        self,
        obj: ArrayLike,
        *,
        mom_ndim: MomNDim | None = None,
        mom_axes: MomAxes | None = None,
        copy: bool | None = None,
        dtype: DTypeLike = None,
        order: ArrayOrderKACF = None,
        mom_params: MomParamsType = None,
        fastpath: bool = False,
    ) -> None:
        if fastpath:
            if not is_ndarray(obj):
                msg = f"Must pass ndarray as data.  Not {type(obj)=}"
                raise TypeError(msg)
            assert isinstance(mom_params, MomParamsArray)  # noqa: S101

        else:
            obj = np.array(
                obj,
                dtype=dtype,
                copy=copy_if_needed(copy),
                order=order,
            )

            mom_params = MomParamsArray.factory(
                mom_params=mom_params, ndim=mom_ndim, axes=mom_axes, default_ndim=1
            )
            mom_params.check_data(obj)

            # NOTE: too much of a pain to allow arbitrary mom_axes right now.
            # Would have to add special checks to moveaxis, reshape, reduce, etc
            mom_params_end = mom_params.axes_to_end()
            if mom_params.axes != mom_params_end.axes:
                obj = np.moveaxis(obj, mom_params.axes, mom_params_end.axes)
                mom_params = mom_params_end

        super().__init__(
            obj,
            mom_params=mom_params,
            fastpath=fastpath,
        )

    # ** Properties ------------------------------------------------------------
    @property
    def mom_axes(self) -> MomAxesStrict:
        """Axes index corresponding to moment(s)."""
        return self._mom_params.axes

    # Reimplement to get dtype correct
    @property
    @override
    @docfiller_array_abc()
    def dtype(self) -> np.dtype[FloatT]:
        return self._obj.dtype

    def __getitem__(self, key: Any) -> Self:
        """
        Get new object by indexing.

        Note that only objects with the same moment(s) shape are allowed.

        If you want to extract data in general, use `self.to_values()[....]`.
        """
        obj = self.obj[key]
        self._raise_if_wrong_mom_shape(
            self._mom_params.get_mom_shape(obj),
        )
        return self._new_like(obj)

    @override
    def to_numpy(self) -> NDArray[FloatT]:
        return self._obj

    def __iter__(self) -> Iterator[Self]:
        for k in range(self._obj.shape[0]):
            yield self[k]

    # ** Create/copy/new ------------------------------------------------------
    @overload
    def new_like(  # pylint: disable=signature-differs
        self,
        obj: ArrayLikeArg[FloatT_],
        *,
        verify: bool = ...,
        copy: bool | None = ...,
        dtype: None = ...,
        order: ArrayOrderKACF = ...,
        fastpath: bool = ...,
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    def new_like(
        self,
        obj: ArrayLike | None = ...,
        *,
        verify: bool = ...,
        copy: bool | None = ...,
        dtype: DTypeLikeArg[FloatT_],
        order: ArrayOrderKACF = ...,
        fastpath: bool = ...,
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    def new_like(
        self,
        obj: None = ...,
        *,
        verify: bool = ...,
        copy: bool | None = ...,
        dtype: None = ...,
        order: ArrayOrderKACF = ...,
        fastpath: bool = ...,
    ) -> Self: ...
    @overload
    def new_like(
        self,
        obj: ArrayLike | None = ...,
        *,
        verify: bool = ...,
        copy: bool | None = ...,
        dtype: DTypeLike = ...,
        order: ArrayOrderKACF = ...,
        fastpath: bool = ...,
    ) -> CentralMomentsArrayAny: ...

    @override
    @docfiller_array_inherit_abc()
    def new_like(
        self,
        obj: ArrayLike | None = None,
        *,
        verify: bool = False,
        copy: bool | None = None,
        dtype: DTypeLike = None,
        order: ArrayOrderKACF = None,
        fastpath: bool = False,
    ) -> CentralMomentsArrayAny:
        """
        Parameters
        ----------
        {dtype}

        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.default_rng(0)
        >>> da = CentralMomentsArray(rng.random(4))
        >>> da
        <CentralMomentsArray(mom_ndim=1)>
        array([0.637 , 0.2698, 0.041 , 0.0165])

        >>> da2 = da.new_like()
        >>> da2
        <CentralMomentsArray(mom_ndim=1)>
        array([0., 0., 0., 0.])

        >>> da.new_like([1, 2, 3, 4], dtype=np.float32)
        <CentralMomentsArray(mom_ndim=1)>
        array([1., 2., 3., 4.], dtype=float32)
        """
        if obj is None:
            obj = np.zeros_like(self.obj, dtype=dtype)
        else:
            obj = np.asarray(obj, dtype=dtype)
            self._raise_if_wrong_mom_shape(self._mom_params.get_mom_shape(obj))
            if verify:
                raise_if_wrong_value(
                    obj.shape,
                    self.obj.shape,
                    "With `verify`, `obj` shape must equal `self.shape`.",
                )

        return type(self)(
            obj=obj,
            copy=copy,
            dtype=dtype,
            order=order,
            mom_params=self._mom_params,
            fastpath=fastpath,
        )

    @overload
    def astype(
        self,
        dtype: DTypeLikeArg[FloatT_],
        *,
        order: ArrayOrderKACF = ...,
        casting: Casting | None = ...,
        subok: bool | None = ...,
        copy: bool = ...,
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    def astype(
        self,
        dtype: None,
        *,
        order: ArrayOrderKACF = ...,
        casting: Casting | None = ...,
        subok: bool | None = ...,
        copy: bool = ...,
    ) -> CentralMomentsArray[np.float64]: ...
    @overload
    def astype(
        self,
        dtype: DTypeLike,
        *,
        order: ArrayOrderKACF = ...,
        casting: Casting | None = ...,
        subok: bool | None = ...,
        copy: bool = ...,
    ) -> CentralMomentsArrayAny: ...

    @override
    @docfiller_array_abc()
    def astype(
        self,
        dtype: DTypeLike,
        *,
        order: ArrayOrderKACF = None,
        casting: Casting | None = None,
        subok: bool | None = None,
        copy: bool = False,
    ) -> CentralMomentsArrayAny:
        return super().astype(
            dtype=dtype, order=order, casting=casting, subok=subok, copy=copy
        )

    # ** Utils ----------------------------------------------------------------
    @override
    def _validate_dtype(self) -> None:
        _ = validate_floating_dtype(self._obj.dtype)

    # ** Pushing --------------------------------------------------------------
    @override
    @docfiller_array_inherit_abc()
    def push_data(
        self,
        data: ArrayLike,
        *,
        casting: Casting = "same_kind",
        parallel: bool | None = False,
        scale: ArrayLike | None = None,
    ) -> Self:
        """
        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.default_rng(0)
        >>> xs = rng.random((2, 10))
        >>> datas = [cmomy.reduce_vals(x, mom=2, axis=0) for x in xs]
        >>> da = CentralMomentsArray(datas[0], mom_ndim=1)
        >>> da
        <CentralMomentsArray(mom_ndim=1)>
        array([10.    ,  0.5505,  0.1014])


        >>> da.push_data(datas[1])
        <CentralMomentsArray(mom_ndim=1)>
        array([20.    ,  0.5124,  0.1033])


        Which is equivalent to

        >>> cmomy.wrap_reduce_vals(xs.reshape(-1), mom=2, axis=0)
        <CentralMomentsArray(mom_ndim=1)>
        array([20.    ,  0.5124,  0.1033])

        """
        if scale is None:
            self._pusher(parallel, size=self._obj.size).data(
                data,
                self._obj,
                casting=casting,
                signature=(self.dtype, self.dtype),
            )
        else:
            self._pusher(parallel, size=self._obj.size).data_scale(
                data,
                scale,
                self._obj,
                casting=casting,
                signature=(self.dtype, self.dtype, self.dtype),
            )
        return self

    @override
    @docfiller_array_inherit_abc()
    def push_datas(
        self,
        datas: ArrayLike,
        *,
        axis: AxisReduce = -1,
        mom_axes: MomAxes | None = None,
        casting: Casting = "same_kind",
        parallel: bool | None = None,
    ) -> Self:
        """
        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.default_rng(0)
        >>> xs = rng.random((2, 10))
        >>> datas = cmomy.reduce_vals(xs, axis=1, mom=2)
        >>> da = CentralMomentsArray.zeros(mom=2)
        >>> da.push_datas(datas, axis=0)
        <CentralMomentsArray(mom_ndim=1)>
        array([20.    ,  0.5124,  0.1033])


        Which is equivalent to

        >>> cmomy.wrap_reduce_vals(xs.reshape(-1), mom=2, axis=0)
        <CentralMomentsArray(mom_ndim=1)>
        array([20.    ,  0.5124,  0.1033])
        """
        _, axis, datas = PrepareDataArray.factory(
            ndim=self.mom_ndim,
            axes=mom_axes,
            recast=False,
        ).data_for_reduction(
            data=datas,
            axis=axis,
            axes_to_end=True,
            dtype=self.dtype,
        )

        self._pusher(parallel, size=datas.size).datas(
            datas,
            self._obj,
            axes=self._mom_params.axes_data_reduction(axis=axis),
            casting=casting,
            signature=(self.dtype, self.dtype),
        )

        return self

    @override
    @docfiller_array_inherit_abc()
    def push_val(
        self,
        x: ArrayLike,
        *y: ArrayLike,
        weight: ArrayLike | None = None,
        casting: Casting = "same_kind",
        parallel: bool | None = False,
    ) -> Self:
        """
        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.default_rng(0)
        >>> x = rng.random((10, 2))
        >>> y = rng.random(10)
        >>> w = rng.random(10)

        >>> da = CentralMomentsArray.zeros(val_shape=(2,), mom=(2, 2))
        >>> for xx, yy, ww in zip(x, y, w):
        ...     _ = da.push_val(xx, yy, weight=ww)

        >>> da
        <CentralMomentsArray(mom_ndim=2)>
        array([[[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 6.4741e-01,  3.3791e-02, -5.1117e-03],
                [ 5.0888e-02, -1.0060e-02,  7.0290e-03]],
        <BLANKLINE>
               [[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 3.9793e-01,  6.3224e-03, -2.2669e-02],
                [ 9.3979e-02,  9.9433e-04,  6.5765e-03]]])


        Which is the same as

        >>> cmomy.wrap_reduce_vals(x, y, weight=w, mom=(2, 2), axis=0)
        <CentralMomentsArray(mom_ndim=2)>
        array([[[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 6.4741e-01,  3.3791e-02, -5.1117e-03],
                [ 5.0888e-02, -1.0060e-02,  7.0290e-03]],
        <BLANKLINE>
               [[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 3.9793e-01,  6.3224e-03, -2.2669e-02],
                [ 9.3979e-02,  9.9433e-04,  6.5765e-03]]])

        """
        self._check_y(y, self.mom_ndim)
        self._pusher(parallel, size=self._obj.size).val(
            self._obj,
            x,
            1.0 if weight is None else weight,
            *y,
            casting=casting,
            signature=(self.dtype,) * (len(y) + 3),
        )
        return self

    @override
    @docfiller_array_inherit_abc()
    def push_vals(
        self,
        x: ArrayLike,
        *y: ArrayLike,
        axis: AxisReduce = -1,
        weight: ArrayLike | None = None,
        casting: Casting = "same_kind",
        parallel: bool | None = None,
    ) -> Self:
        """
        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.default_rng(0)
        >>> x = rng.random((10, 2))
        >>> y = rng.random(10)
        >>> w = rng.random(10)

        >>> da = CentralMomentsArray.zeros(val_shape=(2,), mom=(2, 2))
        >>> da.push_vals(x, y, weight=w, axis=0)
        <CentralMomentsArray(mom_ndim=2)>
        array([[[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 6.4741e-01,  3.3791e-02, -5.1117e-03],
                [ 5.0888e-02, -1.0060e-02,  7.0290e-03]],
        <BLANKLINE>
               [[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 3.9793e-01,  6.3224e-03, -2.2669e-02],
                [ 9.3979e-02,  9.9433e-04,  6.5765e-03]]])


        Which is the same as

        >>> cmomy.wrap_reduce_vals(x, y, weight=w, mom=(2, 2), axis=0)
        <CentralMomentsArray(mom_ndim=2)>
        array([[[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 6.4741e-01,  3.3791e-02, -5.1117e-03],
                [ 5.0888e-02, -1.0060e-02,  7.0290e-03]],
        <BLANKLINE>
               [[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 3.9793e-01,  6.3224e-03, -2.2669e-02],
                [ 9.3979e-02,  9.9433e-04,  6.5765e-03]]])

        """
        axis, args = prepare_array_values_for_reduction(
            x,
            1.0 if weight is None else weight,
            *y,
            axis=axis,
            dtype=self.dtype,
            narrays=self.mom_ndim + 1,
            axes_to_end=True,
            recast=False,
        )

        self._pusher(parallel, size=self._obj.size).vals(
            self._obj, *args, casting=casting, signature=(self.dtype,) * (len(args) + 1)
        )
        return self

    # ** Interface to modules -------------------------------------------------
    # *** .convert ------------------------------------------------------------
    @overload  # type: ignore[override]
    def cumulative(
        self,
        *,
        out: None = ...,
        dtype: None = ...,
        **kwargs: Unpack[WrapNPTransformKwargs],
    ) -> NDArray[FloatT]: ...
    @overload
    def cumulative(
        self,
        *,
        out: NDArray[FloatT_],
        dtype: DTypeLike = ...,
        **kwargs: Unpack[WrapNPTransformKwargs],
    ) -> NDArray[FloatT_]: ...
    @overload
    def cumulative(
        self,
        *,
        out: None = ...,
        dtype: DTypeLikeArg[FloatT_],
        **kwargs: Unpack[WrapNPTransformKwargs],
    ) -> NDArray[FloatT_]: ...
    @overload
    def cumulative(
        self,
        *,
        out: NDArrayAny | None = ...,
        dtype: DTypeLike = ...,
        **kwargs: Unpack[WrapNPTransformKwargs],
    ) -> NDArrayAny: ...

    @override
    @docfiller_array_inherit_abc()
    def cumulative(  # pyright: ignore[reportIncompatibleMethodOverride]  # pylint: disable=arguments-differ
        self,
        *,
        axis: AxisReduce | MissingType = MISSING,
        axes_to_end: bool = False,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        casting: Casting = "same_kind",
        order: ArrayOrderKACF = None,
        parallel: bool | None = None,
    ) -> NDArrayAny:
        return super().cumulative(
            axis=axis,
            axes_to_end=axes_to_end,
            out=out,
            dtype=dtype,
            casting=casting,
            order=order,
            parallel=parallel,
        )

    @overload  # type: ignore[override]
    def moments_to_comoments(
        self,
        *,
        mom: tuple[int, int],
        dtype: None = ...,
        order: ArrayOrderCF = ...,
    ) -> Self: ...
    @overload
    def moments_to_comoments(
        self,
        *,
        mom: tuple[int, int],
        dtype: DTypeLikeArg[FloatT_],
        order: ArrayOrderCF = ...,
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    def moments_to_comoments(
        self,
        *,
        mom: tuple[int, int],
        dtype: DTypeLike,
        order: ArrayOrderCF = ...,
    ) -> CentralMomentsArrayAny: ...

    @override
    @docfiller_array_inherit_abc()
    def moments_to_comoments(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        mom: tuple[int, int],
        dtype: DTypeLike = None,
        order: ArrayOrderCF = None,
    ) -> Self | CentralMomentsArrayAny:
        return super().moments_to_comoments(
            mom=mom,
            dtype=dtype,
            order=order,
        )

    # *** .resample -----------------------------------------------------------
    @overload  # type: ignore[override]
    def resample_and_reduce(
        self,
        *,
        sampler: SamplerType,
        out: None = ...,
        dtype: None = ...,
        **kwargs: Unpack[WrapNPResampleAndReduceKwargs],
    ) -> Self: ...
    @overload
    def resample_and_reduce(
        self,
        *,
        sampler: SamplerType,
        out: NDArray[FloatT_],
        dtype: DTypeLike = ...,
        **kwargs: Unpack[WrapNPResampleAndReduceKwargs],
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    def resample_and_reduce(
        self,
        *,
        sampler: SamplerType,
        out: None = ...,
        dtype: DTypeLikeArg[FloatT_],
        **kwargs: Unpack[WrapNPResampleAndReduceKwargs],
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    def resample_and_reduce(
        self,
        *,
        sampler: SamplerType,
        out: NDArrayAny | None = ...,
        dtype: DTypeLike = ...,
        **kwargs: Unpack[WrapNPResampleAndReduceKwargs],
    ) -> CentralMomentsArrayAny: ...

    @override
    @docfiller_array_inherit_abc()
    def resample_and_reduce(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        axis: AxisReduce | MissingType = -1,
        sampler: SamplerType,
        axes_to_end: bool = False,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        casting: Casting = "same_kind",
        order: ArrayOrderKACF = None,
        parallel: bool | None = None,
    ) -> Self | CentralMomentsArrayAny:
        """
        See Also
        --------
        CentralMomentsData.resample_and_reduce

        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.default_rng(0)
        >>> c = cmomy.CentralMomentsArray(rng.random((3, 3)), mom_ndim=1)
        >>> c
        <CentralMomentsArray(mom_ndim=1)>
        array([[0.637 , 0.2698, 0.041 ],
               [0.0165, 0.8133, 0.9128],
               [0.6066, 0.7295, 0.5436]])

        >>> c.resample_and_reduce(axis=0, sampler=dict(nrep=5, rng=0))
        <CentralMomentsArray(mom_ndim=1)>
        array([[0.6397, 0.7338, 0.563 ],
               [1.9109, 0.2698, 0.041 ],
               [1.9109, 0.2698, 0.041 ],
               [1.2298, 0.7306, 0.5487],
               [0.6397, 0.7338, 0.563 ]])
        """
        return super().resample_and_reduce(
            axis=axis,
            sampler=sampler,
            axes_to_end=axes_to_end,
            out=out,
            dtype=dtype,
            casting=casting,
            order=order,
            parallel=parallel,
        )

    @overload  # type: ignore[override]
    def jackknife_and_reduce(
        self,
        *,
        data_reduced: ArrayLike | None = ...,
        out: None = ...,
        dtype: None = ...,
        **kwargs: Unpack[WrapNPTransformKwargs],
    ) -> Self: ...
    @overload
    def jackknife_and_reduce(
        self,
        *,
        data_reduced: ArrayLike | None = ...,
        out: NDArray[FloatT_],
        dtype: DTypeLike = ...,
        **kwargs: Unpack[WrapNPTransformKwargs],
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    def jackknife_and_reduce(
        self,
        *,
        data_reduced: ArrayLike | None = ...,
        out: None = ...,
        dtype: DTypeLikeArg[FloatT_],
        **kwargs: Unpack[WrapNPTransformKwargs],
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    def jackknife_and_reduce(
        self,
        *,
        data_reduced: ArrayLike | None = ...,
        out: NDArrayAny | None = ...,
        dtype: DTypeLike = ...,
        **kwargs: Unpack[WrapNPTransformKwargs],
    ) -> CentralMomentsArrayAny: ...

    @override
    @docfiller_array_inherit_abc()
    def jackknife_and_reduce(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        data_reduced: ArrayLike | None = None,
        axis: AxisReduce | MissingType = -1,
        axes_to_end: bool = False,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        casting: Casting = "same_kind",
        order: ArrayOrderKACF = None,
        parallel: bool | None = None,
    ) -> Self | CentralMomentsArrayAny:
        return super().jackknife_and_reduce(
            data_reduced=data_reduced,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
            axis=axis,
            axes_to_end=axes_to_end,
            out=out,
            dtype=dtype,
            casting=casting,
            order=order,
            parallel=parallel,
        )

    # *** .reduction ----------------------------------------------------------
    @overload
    def reduce(
        self,
        *,
        out: None = ...,
        dtype: None = ...,
        **kwargs: Unpack[WrapNPReduceKwargs],
    ) -> Self: ...
    @overload
    def reduce(
        self,
        *,
        out: NDArray[FloatT_],
        dtype: DTypeLike = ...,
        **kwargs: Unpack[WrapNPReduceKwargs],
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    def reduce(
        self,
        *,
        out: None = ...,
        dtype: DTypeLikeArg[FloatT_],
        **kwargs: Unpack[WrapNPReduceKwargs],
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    def reduce(
        self,
        *,
        out: NDArrayAny | None = ...,
        dtype: DTypeLike = ...,
        **kwargs: Unpack[WrapNPReduceKwargs],
    ) -> CentralMomentsArrayAny: ...

    @override
    @docfiller_array_inherit_abc()
    def reduce(
        self,
        axis: AxisReduce | MissingType = MISSING,
        *,
        by: Groups | None = None,
        block: int | None = None,
        axes_to_end: bool = False,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        casting: Casting = "same_kind",
        order: ArrayOrderKACF = None,
        keepdims: bool = False,
        parallel: bool | None = None,
    ) -> Self | CentralMomentsArrayAny:
        if by is None and block is not None:
            by = self._block_by(block, axis=axis)

        if by is None:
            from cmomy.reduction import reduce_data

            obj = reduce_data(
                self._obj,
                mom_params=self._mom_params,
                axis=axis,
                out=out,
                dtype=dtype,
                casting=casting,
                order=order,
                keepdims=keepdims,
                parallel=parallel,
            )
        else:
            from cmomy.grouped import reduce_data_grouped

            obj = reduce_data_grouped(
                self._obj,
                mom_params=self._mom_params,
                by=by,
                axis=axis,
                axes_to_end=axes_to_end,
                out=out,
                dtype=dtype,
                casting=casting,
                order=arrayorder_to_arrayorder_cf(order),
                parallel=parallel,
            )
        return self._new_like(obj)

    # ** Constructors ----------------------------------------------------------
    @overload
    @classmethod
    def zeros(
        cls,
        *,
        mom: Moments,
        val_shape: int | Sequence[int] | None = ...,
        dtype: None = ...,
        order: ArrayOrderCF = ...,
    ) -> CentralMomentsArray[np.float64]: ...
    @overload
    @classmethod
    def zeros(
        cls,
        *,
        mom: Moments,
        val_shape: int | Sequence[int] | None = ...,
        dtype: DTypeLikeArg[FloatT_],
        order: ArrayOrderCF = ...,
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    @classmethod
    def zeros(
        cls,
        *,
        mom: Moments,
        val_shape: int | Sequence[int] | None = ...,
        dtype: DTypeLike = ...,
        order: ArrayOrderCF = ...,
    ) -> CentralMomentsArrayAny: ...

    @override
    @classmethod
    @docfiller_array_abc()
    def zeros(
        cls,
        *,
        mom: Moments,
        val_shape: int | Sequence[int] | None = None,
        dtype: DTypeLike = None,
        order: ArrayOrderCF = None,
    ) -> CentralMomentsArrayAny | Self:
        """
        Parameters
        ----------
        {mom}
        {val_shape}
        {dtype}
        {order}
        """
        mom, mom_params = MomParamsArray.factory_mom(mom)

        if val_shape is None:
            val_shape = ()
        elif isinstance(val_shape, int):
            val_shape = (val_shape,)

        # add in moments
        shape: tuple[int, ...] = (*val_shape, *mom_to_mom_shape(mom))  # type: ignore[misc]

        return cls(np.zeros(shape, dtype=dtype, order=order), mom_params=mom_params)

    # ** Custom for this class -------------------------------------------------
    def _raise_if_scalar(self, message: str | None = None) -> None:
        if self._obj.ndim <= self.mom_ndim:
            if message is None:  # pragma: no cover
                message = "Not implemented for scalar"
            raise ValueError(message)

    @docfiller_array.decorate
    def reshape(
        self,
        shape: int | Sequence[int],
        *,
        order: ArrayOrderACF = None,
    ) -> Self:
        """
        Create a new object with reshaped data.

        Parameters
        ----------
        shape : int or tuple
            shape of values part of data.
        order : {{"C", "F", "A"}}, optional
            Parameter to :func:`numpy.reshape`. Note that this parameter has
            nothing to do with the output data order. Rather, it is how the
            data is read for the reshape.

        Returns
        -------
        output : CentralMomentsArray
            Output object with reshaped data.  This will be a view if possible;
            otherwise, it will be copy.

        See Also
        --------
        numpy.reshape
        new_like

        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.default_rng(0)
        >>> da = cmomy.wrap_reduce_vals(rng.random((10, 2, 3)), mom=2, axis=0)
        >>> da
        <CentralMomentsArray(mom_ndim=1)>
        array([[[10.    ,  0.5205,  0.0452],
                [10.    ,  0.4438,  0.0734],
                [10.    ,  0.5038,  0.1153]],
        <BLANKLINE>
               [[10.    ,  0.5238,  0.1272],
                [10.    ,  0.628 ,  0.0524],
                [10.    ,  0.412 ,  0.0865]]])

        >>> da.reshape(shape=(-1,))
        <CentralMomentsArray(mom_ndim=1)>
        array([[10.    ,  0.5205,  0.0452],
               [10.    ,  0.4438,  0.0734],
               [10.    ,  0.5038,  0.1153],
               [10.    ,  0.5238,  0.1272],
               [10.    ,  0.628 ,  0.0524],
               [10.    ,  0.412 ,  0.0865]])
        """
        self._raise_if_scalar()
        shape = (shape,) if isinstance(shape, int) else shape
        new_shape = (*shape, *self.mom_shape)  # type: ignore[misc]
        obj = self._obj.reshape(new_shape, order=order)
        return self.new_like(obj)  # type: ignore[return-value]  # pyright: ignore[reportReturnType]

    @docfiller_array.decorate
    def resample(
        self,
        indices: ArrayLike,
        *,
        axis: AxisReduceWrap = -1,
        last: bool = False,
    ) -> Self:
        """
        Create a new object sampled from index.

        Parameters
        ----------
        {indices}
        {axis_data_and_dim}
        last : bool, default=True
            If ``True``, and axis != -1, move the axis to last position before moments.
            This makes results similar to resample and reduce
            If `last` False, then resampled array can have odd shape
        {order}

        Returns
        -------
        output : object
            Instance of calling class. The new object will have shape
            ``(..., shape[axis-1], nrep, nsamp, shape[axis+1], ...)``,
            (if ``last=False``) or shape
            ``(..., shape[axis-1], shape[axis+1], ..., nrep, nsamp, mom_0, ...)``
            (if ``last=True``),
            where ``shape=self.obj.shape`` and ``nrep, nsamp = indices.shape``.

        """
        self._raise_if_scalar()

        axis = self._mom_params.normalize_axis_index(
            validate_axis(axis), self._obj.ndim
        )

        obj = self.obj
        last_dim = obj.ndim - self.mom_ndim - 1
        if last and axis != last_dim:
            obj = np.moveaxis(obj, axis, last_dim)
            axis = last_dim

        indices = np.asarray(indices, dtype=np.int64)
        obj = np.take(obj, indices, axis=axis)

        return self.new_like(obj)  # type: ignore[return-value]  # pyright: ignore[reportReturnType]

    def fill(self, value: Any = 0) -> Self:
        """
        Fill data with value.

        Parameters
        ----------
        value : scalar
            Value to insert into `self.data`

        Returns
        -------
        self : object
            Same type as calling class.
            Same object as caller with data filled with `values`
        """
        self._obj.fill(value)
        return self

    def zero(self) -> Self:
        """
        Zero out underlying data.

        Returns
        -------
        self : object
            Same type as calling class.
            Same object with data filled with zeros.

        See Also
        --------
        fill
        """
        return self.fill(value=0.0)

    # ** To/from xarray ------------------------------------------------------------
    @docfiller_array.decorate
    def to_dataarray(
        self,
        *,
        dims: DimsType = None,
        attrs: AttrsType = None,
        coords: CoordsType = None,
        name: NameType = None,
        mom_dims: MomDims | None = None,
        template: xr.DataArray | None = None,
        copy: bool = False,
    ) -> CentralMomentsData[xr.DataArray]:
        r"""
        Create a  :class:`cmomy.CentralMomentsData` object from ``self``.


        Parameters
        ----------
        {xr_params}
        {copy_tf}

        Returns
        -------
        output : DataArray

        See Also
        --------
        CentralMomentsData


        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> c = CentralMomentsArray(rng.random((1, 2, 4)), mom_ndim=1)
        >>> c
        <CentralMomentsArray(mom_ndim=1)>
        array([[[0.637 , 0.2698, 0.041 , 0.0165],
                [0.8133, 0.9128, 0.6066, 0.7295]]])

        Default is to create a :class:`.CentralMomentsData` object

        >>> c.to_dataarray()
        <CentralMomentsData(mom_ndim=1)>
        <xarray.DataArray (dim_0: 1, dim_1: 2, mom_0: 4)> Size: 64B
        array([[[0.637 , 0.2698, 0.041 , 0.0165],
                [0.8133, 0.9128, 0.6066, 0.7295]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0


        To just create a :class:`xarray.DataArray` object, access :attr:`obj`

        >>> c.to_dataarray().obj
        <xarray.DataArray (dim_0: 1, dim_1: 2, mom_0: 4)> Size: 64B
        array([[[0.637 , 0.2698, 0.041 , 0.0165],
                [0.8133, 0.9128, 0.6066, 0.7295]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0


        You can set attributes during construction:

        >>> c.to_dataarray(dims=["a", "b", "mom"])
        <CentralMomentsData(mom_ndim=1)>
        <xarray.DataArray (a: 1, b: 2, mom: 4)> Size: 64B
        array([[[0.637 , 0.2698, 0.041 , 0.0165],
                [0.8133, 0.9128, 0.6066, 0.7295]]])
        Dimensions without coordinates: a, b, mom
        """
        data = self.obj
        if copy:
            data = data.copy()

        out: xr.DataArray
        if template is not None:
            out = template.copy(data=data)
        else:
            val_ndim = self.obj.ndim - self.mom_ndim
            if dims is None:
                dims = tuple(f"dim_{i}" for i in range(val_ndim))
            elif isinstance(dims, str):
                dims = (dims,)
            else:
                # try to convert to tuple
                dims = tuple(dims)

            if len(dims) == self.obj.ndim:
                dims_output = dims

            elif len(dims) == val_ndim:
                if mom_dims is None:
                    mom_dims = tuple(f"mom_{i}" for i in range(self.mom_ndim))
                elif isinstance(mom_dims, str):
                    mom_dims = (mom_dims,)
                else:
                    # try to convert to tuple
                    mom_dims = tuple(mom_dims)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

                raise_if_wrong_value(
                    len(mom_dims),
                    self.mom_ndim,
                    "`len(mom_dims)` must equal `mom_dims`.",
                )

                dims_output = dims + mom_dims

            else:
                msg = f"Problem with {dims}, {mom_dims}.  Total length should be {self.obj.ndim}"
                raise ValueError(msg)
            out = xr.DataArray(
                data, dims=dims_output, coords=coords, attrs=attrs, name=name
            )

        return CentralMomentsData(
            obj=out,
            mom_params=MomParamsXArray.factory(ndim=self.mom_ndim, data=out),
            fastpath=True,
        )

    def to_x(
        self,
        *,
        dims: DimsType = None,
        attrs: AttrsType = None,
        coords: CoordsType = None,
        name: NameType = None,
        mom_dims: MomDims | None = None,
        template: xr.DataArray | None = None,
        copy: bool = False,
    ) -> CentralMomentsData[xr.DataArray]:
        """Alias to :meth:`to_dataarray`"""
        return self.to_dataarray(
            dims=dims,
            attrs=attrs,
            coords=coords,
            name=name,
            mom_dims=mom_dims,
            template=template,
            copy=copy,
        )


docfiller_data_abc = docfiller_data.factory_from_parent(CentralMomentsABC)
docfiller_data_inherit_abc = docfiller_data.factory_inherit_from_parent(
    CentralMomentsABC
)


@docfiller_data.inherit(CentralMomentsABC)  # noqa: PLR0904
class CentralMomentsData(CentralMomentsABC[DataT, MomParamsXArray]):
    """
    Central moments wrapper of {DataArray} or {Dataset} objects.

    Parameters
    ----------
    {mom_ndim_data}
    {mom_axes}
    {mom_dims_data}
    """

    _mom_params: MomParamsXArray

    def __init__(
        self,
        obj: DataT,
        *,
        mom_ndim: MomNDim | None = None,
        mom_axes: MomAxes | None = None,
        mom_dims: MomDims | None = None,
        mom_params: MomParamsType = None,
        fastpath: bool = False,
    ) -> None:
        if not is_xarray(obj):
            msg = "obj must be a DataArray or Dataset, not {type(obj)}"  # pyright: ignore[reportUnreachable]
            raise TypeError(msg)

        if fastpath:
            assert isinstance(mom_params, MomParamsXArray)  # noqa: S101
        else:
            mom_params = MomParamsXArray.factory(
                mom_params=mom_params,
                ndim=mom_ndim,
                dims=mom_dims,
                axes=mom_axes,
                data=obj,
                default_ndim=1,
            )

            mom_params.check_data(obj)

        # NOTE: Why this ignore?
        super().__init__(obj=obj, mom_params=mom_params, fastpath=fastpath)  # type: ignore[arg-type]

    # ** Properties ------------------------------------------------------------
    @property
    def mom_dims(self) -> MomDimsStrict:
        """Moments dimension names."""
        return self._mom_params.dims

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
        return {  # pyright: ignore[reportReturnType]
            k: type(self)(  # type: ignore[misc]
                obj,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
                mom_params=self._mom_params,
                fastpath=True,
            )
            for k, obj in self._obj.items()
            if contains_dims(obj, *self.mom_dims)
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
        return self.as_dict().keys()

    def values(
        self: CentralMomentsDataset,
    ) -> ValuesView[CentralMomentsDataArray]:  # -> Iterator[CentralMomentsDataArray]:
        """Dict like values view"""
        return self.as_dict().values()

    @overload
    def iter(self: CentralMomentsDataArray) -> Iterator[CentralMomentsDataArray]: ...
    @overload
    def iter(self: CentralMomentsDataset) -> Iterator[Hashable]: ...
    @overload
    def iter(self) -> Any: ...

    def iter(self) -> Iterator[Hashable] | Iterator[CentralMomentsDataArray]:
        """Need this for proper typing with mypy..."""  # noqa: DOC402
        if is_dataarray(self._obj):
            if self.ndim <= self.mom_ndim:
                msg = "Can only iterate over wrapped DataArray with extra dimension."
                raise ValueError(msg)
            for obj in self._obj:
                yield self.new_like(obj)
        else:
            yield from self.keys()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

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
        if not contains_dims(obj, *self.mom_dims):  # pyright: ignore[reportUnknownArgumentType]
            msg = f"Cannot select object without {self.mom_dims}"
            raise ValueError(msg)
        self._raise_if_wrong_mom_shape(self._mom_params.get_mom_shape(obj))  # pyright: ignore[reportUnknownArgumentType]

        return type(self)(  # pyright: ignore[reportReturnType]
            obj=obj,  # type: ignore[arg-type]  # pyright: ignore[reportUnknownArgumentType]
            mom_params=self._mom_params,
            fastpath=True,
        )

    # ** Create/copy/new ------------------------------------------------------
    @override
    @docfiller_data_inherit_abc()
    def new_like(  # type: ignore[override]  # pylint: disable=arguments-differ  # pyright: ignore[reportIncompatibleMethodOverride]
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
                obj=xr.zeros_like(self._obj, dtype=dtype),  # type: ignore[arg-type]  # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]
                mom_params=self._mom_params,
                fastpath=fastpath,
            )

        # TODO(wpk): edge case of passing in new xarray data with different moment dimensions.
        # For now, this will raise an error.
        obj_: DataT = (
            cast("DataT", obj)
            if type(self._obj) is type(obj)
            else self._obj.copy(data=obj)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        )

        # minimal check on shape and that mom_dims are present....
        if not contains_dims(obj_, *self.mom_dims):
            msg = f"Cannot create new from object without {self.mom_dims}"
            raise ValueError(msg)
        self._raise_if_wrong_mom_shape(self._mom_params.get_mom_shape(obj_))

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
            mom_params=self._mom_params,
            fastpath=fastpath,
        )

    @override
    @docfiller_data_inherit_abc()
    def copy(self, deep: bool = True) -> Self:
        """
        Parameters
        ----------
        deep : bool
            Parameters to :meth:`xarray.DataArray.copy` or `xarray.Dataset.copy`
        """
        return type(self)(
            obj=self._obj.copy(deep=deep),
            mom_params=self._mom_params,
            fastpath=True,
        )

    @override
    @docfiller_data_inherit_abc()
    def astype(
        self,
        dtype: DTypeLike | Mapping[Hashable, DTypeLike],
        *,
        order: ArrayOrderKACF = None,
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
    @override
    def _validate_dtype(self) -> None:
        if is_dataarray(self._obj):
            _ = validate_floating_dtype(self._obj)
            return

        for name, val in self._obj.items():
            if contains_dims(val, *self.mom_dims):
                _ = validate_floating_dtype(val, name=name)

    # ** Pushing --------------------------------------------------------------
    @property
    def _dtype(self) -> np.dtype[Any] | None:
        if is_dataarray(self._obj):
            return self._obj.dtype
        return None

    @override
    @docfiller_data_inherit_abc()
    def push_data(
        self,
        data: DataT | ArrayLike,
        *,
        scale: ArrayLike | None = None,
        casting: Casting = "same_kind",
        parallel: bool | None = False,
        keep_attrs: KeepAttrs = True,
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self:
        """
        Parameters
        ----------
        {keep_attrs}
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
            input_core_dims=[self.mom_dims, self.mom_dims],
            output_core_dims=[self.mom_dims],
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_dtypes=self._dtype or np.float64,
            ),
        )

        return self

    @override
    @docfiller_data_inherit_abc()
    def push_datas(
        self,
        datas: DataT | ArrayLike,
        *,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        mom_axes: MomAxes | None = None,
        casting: Casting = "same_kind",
        parallel: bool | None = None,
        keep_attrs: KeepAttrs = True,
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self:
        """
        Parameters
        ----------
        {keep_attrs}
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

        if is_xarray_typevar["DataT"].check(datas):
            axis, dim = self._mom_params.select_axis_dim(
                datas,
                axis=axis,
                dim=dim,
            )

        else:
            _, axis, datas = PrepareDataArray.factory(
                ndim=self.mom_ndim,
                axes=mom_axes,
                recast=False,
            ).data_for_reduction(
                data=datas,
                axis=axis,
                axes_to_end=True,
                dtype=self._dtype,
            )
            dim = "_dummy123"

        self._obj = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            func,
            self._obj,
            datas,
            input_core_dims=[self.mom_dims, [dim, *self.mom_dims]],
            output_core_dims=[self.mom_dims],
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_dtypes=self._dtype or np.float64,
            ),
        )

        return self

    @override
    @docfiller_data_inherit_abc()
    def push_val(
        self,
        x: ArrayLike | DataT,
        *y: ArrayLike | xr.DataArray | DataT,
        weight: ArrayLike | xr.DataArray | DataT | None = None,
        casting: Casting = "same_kind",
        parallel: bool | None = False,
        keep_attrs: KeepAttrs = True,
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self:
        """
        Parameters
        ----------
        {keep_attrs}
        {apply_ufunc_kwargs}
        """
        self._check_y(y, self.mom_ndim)

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
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_dtypes=self._dtype or np.float64,
            ),
        )

        return self

    @override
    @docfiller_data_inherit_abc()
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
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self:
        """
        Parameters
        ----------
        {keep_attrs}
        {apply_ufunc_kwargs}
        """
        weight = 1.0 if weight is None else weight
        xargs: Sequence[ArrayLike | xr.DataArray | xr.Dataset]
        if is_xarray_typevar["DataT"].check(x):
            dim, input_core_dims, xargs = prepare_xarray_values_for_reduction(
                x,
                weight,
                *y,
                axis=axis,
                dim=dim,
                narrays=self.mom_ndim + 1,
                dtype=self._dtype,
                recast=False,
            )
        else:
            axis, xargs = prepare_array_values_for_reduction(
                x,
                weight,
                *y,
                axis=axis,
                dtype=self._dtype,
                narrays=self.mom_ndim + 1,
                axes_to_end=True,
                recast=False,
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
            input_core_dims=[self.mom_dims, *input_core_dims],  # type: ignore[has-type]
            output_core_dims=[self.mom_dims],
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_dtypes=self._dtype or np.float64,
            ),
        )

        return self

    # ** Interface to modules -------------------------------------------------
    # *** .reduction ----------------------------------------------------------
    @override
    @docfiller_data_inherit_abc()
    def reduce(  # noqa: PLR0913
        self,
        dim: DimsReduce | MissingType = MISSING,
        *,
        by: str | Groups | None = None,
        block: int | None = None,
        use_map: bool | None = None,
        group_dim: str | None = None,
        groups: Groups | None = None,
        axis: AxisReduce | MissingType = MISSING,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        casting: Casting = "same_kind",
        order: ArrayOrderCF = None,
        keepdims: bool = False,
        parallel: bool | None = None,
        axes_to_end: bool = False,
        # xarray specific
        coords_policy: CoordsPolicy = "first",
        keep_attrs: KeepAttrs = None,
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
        use_map : bool, optional
            See :func:`.reduction.reduce_data`
        {coords_policy}
        {group_dim}
        {groups}
        {keep_attrs}
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
        grouped.reduce_data_grouped
        grouped.reduce_data_indexed

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
        :func:`~.grouped.block_by`.
        """
        if by is None and block is not None:
            by = self._block_by(block, axis=axis, dim=dim)
            groups = groups or range(by.max() + 1)

        if by is None:
            from cmomy.reduction import reduce_data

            data = reduce_data(
                self._obj,
                mom_ndim=self.mom_ndim,
                mom_dims=self.mom_dims,
                axis=axis,
                dim=dim,
                out=out,
                dtype=dtype,
                casting=casting,
                order=order,
                keepdims=keepdims,
                parallel=parallel,
                use_map=use_map,
                keep_attrs=keep_attrs,
                apply_ufunc_kwargs=apply_ufunc_kwargs,
            )

        else:
            from cmomy.grouped import reduce_data_grouped

            codes: ArrayLike
            if isinstance(by, str):
                from cmomy.grouped import factor_by

                codes, groups_ = factor_by(self._obj[by].to_numpy())
                if groups is None:
                    groups = groups_
            else:
                codes = by

            data = reduce_data_grouped(
                self._obj,
                mom_ndim=self.mom_ndim,
                mom_dims=self.mom_dims,
                by=codes,
                axis=axis,
                dim=dim,
                axes_to_end=axes_to_end,
                parallel=parallel,
                dtype=dtype,
                out=out,
                casting=casting,
                order=order,
                coords_policy=coords_policy,
                group_dim=group_dim,
                groups=groups,
                keep_attrs=keep_attrs,
                apply_ufunc_kwargs=apply_ufunc_kwargs,
            )

        return self._new_like(data)

    # ** Constructors ----------------------------------------------------------
    @override
    @classmethod
    @docfiller_data.decorate
    def zeros(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
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
        ).transpose(..., *self.mom_dims)

        return type(self)(  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
            obj=obj,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
            mom_params=self._mom_params,
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

        obj = self._obj.to_array(dim=dim, name=name).transpose(..., *self.mom_dims)
        return type(self)(  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
            obj=obj,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
            mom_params=self._mom_params,
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
            self._obj.rename(new_name_or_name_dict, **names)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
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
        >>> da = cmomy.wrap_reduce_vals(vals, mom=2, dim="rec")
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
        >>> da = cmomy.wrap_reduce_vals(rng.random((10, 3)), axis=0, mom=2).to_x(
        ...     dims="x", coords=dict(x=list("abc"))
        ... )
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
            {"transpose_coords": transpose_coords} if is_dataarray(self._obj) else {}  # type: ignore[redundant-expr]
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
        if is_dataset(self.obj):
            self._raise_notimplemented_for_dataset()

        obj = self._obj.to_numpy()
        return CentralMomentsArray(
            obj.copy() if copy else obj,
            mom_params=self._mom_params.to_array(),
            fastpath=True,
        )

    def to_c(self, copy: bool = False) -> CentralMomentsArray[Any]:
        """Alias to :meth:`to_array`."""
        return self.to_array(copy=copy)

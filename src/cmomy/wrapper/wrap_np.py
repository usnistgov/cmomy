"""Wrapper object for numpy arrays"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, overload

import numpy as np
import xarray as xr

from cmomy.core.array_utils import (
    arrayorder_to_arrayorder_cf,
    axes_data_reduction,
)
from cmomy.core.compat import copy_if_needed
from cmomy.core.missing import MISSING
from cmomy.core.prepare import (
    prepare_data_for_reduction,
    prepare_values_for_reduction,
)
from cmomy.core.utils import mom_to_mom_shape
from cmomy.core.validate import (
    is_ndarray,
    raise_if_wrong_value,
    validate_axis,
    validate_floating_dtype,
    validate_mom_and_mom_ndim,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike

    from cmomy.core.typing import (
        ArrayLikeArg,
        ArrayOrder,
        ArrayOrderCF,
        ArrayOrderCFA,
        AttrsType,
        AxisReduce,
        Casting,
        CentralMomentsArrayAny,
        CentralMomentsDataArray,
        CoordsType,
        DimsType,
        DTypeLikeArg,
        FloatT_,
        Groups,
        MissingType,
        Mom_NDim,
        MomDims,
        Moments,
        MomentsStrict,
        NameType,
        NDArrayAny,
        Sampler,
        WrapNPReduceKwargs,
        WrapNPResampleAndReduceKwargs,
        WrapNPTransform,
    )
    from cmomy.core.typing_compat import Self, Unpack


from numpy.typing import NDArray

from cmomy.core.docstrings import docfiller_central as docfiller
from cmomy.core.typing import FloatT

from .wrap_abc import CentralMomentsABC

docfiller_abc = docfiller.factory_from_parent(CentralMomentsABC)
docfiller_inherit_abc = docfiller.factory_inherit_from_parent(CentralMomentsABC)


@docfiller.inherit(CentralMomentsABC)  # noqa: PLR0904
class CentralMomentsArray(CentralMomentsABC[NDArray[FloatT]], Generic[FloatT]):  # type: ignore[type-var]
    r"""
    Central moments wrapper of {ndarray} arrays.

    Parameters
    ----------
    {copy}
    {order}
    {dtype}

    """

    @overload
    def __init__(
        self,
        obj: ArrayLikeArg[FloatT],
        *,
        mom_ndim: Mom_NDim | None = ...,
        copy: bool | None = ...,
        dtype: None = ...,
        order: ArrayOrder = ...,
        fastpath: bool = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        obj: ArrayLike,
        *,
        mom_ndim: Mom_NDim | None = ...,
        copy: bool | None = ...,
        dtype: DTypeLikeArg[FloatT],
        order: ArrayOrder = ...,
        fastpath: bool = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        obj: ArrayLike,
        *,
        mom_ndim: Mom_NDim | None = ...,
        copy: bool | None = ...,
        dtype: DTypeLike = ...,
        order: ArrayOrder = ...,
        fastpath: bool = ...,
    ) -> None: ...

    def __init__(
        self,
        obj: ArrayLike,
        *,
        mom_ndim: Mom_NDim | None = None,
        copy: bool | None = None,
        dtype: DTypeLike = None,
        order: ArrayOrder = None,
        fastpath: bool = False,
    ) -> None:
        if not fastpath:
            obj = np.array(
                obj,
                dtype=dtype,
                copy=copy_if_needed(copy),
                order=order,
            )
        elif not is_ndarray(obj):
            msg = f"Must pass ndarray as data.  Not {type(obj)=}"
            raise TypeError(msg)

        super().__init__(
            obj,
            mom_ndim=mom_ndim,
            fastpath=fastpath,
        )

    # ** Properties ------------------------------------------------------------
    @property
    @docfiller_abc()
    def mom_shape(self) -> MomentsStrict:
        return self._obj.shape[-self._mom_ndim :]  # type: ignore[return-value]

    # Reimplement to get dtype correct
    @property
    @docfiller_abc()
    def dtype(self) -> np.dtype[FloatT]:
        return self._obj.dtype

    def __getitem__(self, key: Any) -> Self:
        """
        Get new object by indexing.

        Note that only objects with the same moment(s) shape are allowed.

        If you want to extract data in general, use `self.to_values()[....]`.
        """
        obj = self.obj[key]
        self._raise_if_wrong_mom_shape(obj.shape[-self._mom_ndim :])
        return self._new_like(obj)

    def to_numpy(self) -> NDArray[FloatT]:
        return self._obj

    def __iter__(self) -> Iterator[Self]:
        for k in range(self._obj.shape[0]):
            yield self[k]

    # ** Create/copy/new ------------------------------------------------------
    def _new_like(self, obj: NDArray[FloatT]) -> Self:
        return type(self)(
            obj=obj,
            mom_ndim=self._mom_ndim,
            fastpath=True,
        )

    @overload
    def new_like(
        self,
        obj: ArrayLikeArg[FloatT_],
        *,
        verify: bool = ...,
        copy: bool | None = ...,
        dtype: None = ...,
        order: ArrayOrder = ...,
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
        order: ArrayOrder = ...,
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
        order: ArrayOrder = ...,
        fastpath: bool = ...,
    ) -> Self: ...
    @overload
    def new_like(
        self,
        obj: Any = ...,
        *,
        verify: bool = ...,
        copy: bool | None = ...,
        dtype: DTypeLike = ...,
        order: ArrayOrder = ...,
        fastpath: bool = ...,
    ) -> CentralMomentsArrayAny: ...

    @docfiller_inherit_abc()
    def new_like(
        self,
        obj: ArrayLike | None = None,
        *,
        verify: bool = False,
        copy: bool | None = None,
        dtype: DTypeLike = None,
        order: ArrayOrder = None,
        fastpath: bool = False,
    ) -> CentralMomentsArrayAny:
        """
        Parameters
        ----------
        {dtype}

        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> da = CentralMomentsArray.from_vals(rng.random(10), mom=3, axis=0)
        >>> da
        <CentralMomentsArray(mom_ndim=1)>
        array([10.    ,  0.5505,  0.1014, -0.0178])

        >>> da2 = da.new_like().zero()
        >>> da2
        <CentralMomentsArray(mom_ndim=1)>
        array([0., 0., 0., 0.])

        >>> da
        <CentralMomentsArray(mom_ndim=1)>
        array([10.    ,  0.5505,  0.1014, -0.0178])
        """
        if obj is None:
            obj = np.zeros_like(self.obj, dtype=dtype)
        else:
            obj = np.asarray(obj, dtype=dtype)
            self._raise_if_wrong_mom_shape(obj.shape[-self.mom_ndim :])
            if verify:
                raise_if_wrong_value(
                    obj.shape,
                    self.obj.shape,
                    "With `verify`, `obj` shape must equal `self.shape`.",
                )

        return type(self)(
            obj=obj,
            copy=copy,
            mom_ndim=self.mom_ndim,
            dtype=dtype,
            order=order,
            fastpath=fastpath,
        )

    @overload
    def astype(
        self,
        dtype: DTypeLikeArg[FloatT_],
        *,
        order: ArrayOrder = ...,
        casting: Casting | None = ...,
        subok: bool | None = ...,
        copy: bool = ...,
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    def astype(
        self,
        dtype: None,
        *,
        order: ArrayOrder = ...,
        casting: Casting | None = ...,
        subok: bool | None = ...,
        copy: bool = ...,
    ) -> CentralMomentsArray[np.float64]: ...
    @overload
    def astype(
        self,
        dtype: DTypeLike,
        *,
        order: ArrayOrder = ...,
        casting: Casting | None = ...,
        subok: bool | None = ...,
        copy: bool = ...,
    ) -> CentralMomentsArrayAny: ...

    @docfiller_abc()
    def astype(
        self,
        dtype: DTypeLike,
        *,
        order: ArrayOrder = None,
        casting: Casting | None = None,
        subok: bool | None = None,
        copy: bool = False,
    ) -> CentralMomentsArrayAny:
        return super().astype(
            dtype=dtype, order=order, casting=casting, subok=subok, copy=copy
        )

    # ** Utils ----------------------------------------------------------------
    def _validate_dtype(self) -> None:
        _ = validate_floating_dtype(self._obj.dtype)

    # ** Pushing --------------------------------------------------------------
    @docfiller_inherit_abc()
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

        >>> CentralMomentsArray.from_vals(xs.reshape(-1), mom=2, axis=0)
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

    @docfiller_inherit_abc()
    def push_datas(
        self,
        datas: ArrayLike,
        *,
        axis: AxisReduce = -1,
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

        >>> CentralMomentsArray.from_vals(xs.reshape(-1), mom=2, axis=0)
        <CentralMomentsArray(mom_ndim=1)>
        array([20.    ,  0.5124,  0.1033])
        """
        axis, datas = prepare_data_for_reduction(
            data=datas,
            axis=axis,
            mom_ndim=self._mom_ndim,
            dtype=self.dtype,
            recast=False,
        )
        axes = axes_data_reduction(mom_ndim=self._mom_ndim, axis=axis)

        self._pusher(parallel, size=datas.size).datas(
            datas,
            self._obj,
            axes=axes,
            casting=casting,
            signature=(self.dtype, self.dtype),
        )

        return self

    @docfiller_inherit_abc()
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

        >>> CentralMomentsArray.from_vals(x, y, weight=w, mom=(2, 2), axis=0)
        <CentralMomentsArray(mom_ndim=2)>
        array([[[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 6.4741e-01,  3.3791e-02, -5.1117e-03],
                [ 5.0888e-02, -1.0060e-02,  7.0290e-03]],
        <BLANKLINE>
               [[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 3.9793e-01,  6.3224e-03, -2.2669e-02],
                [ 9.3979e-02,  9.9433e-04,  6.5765e-03]]])

        """
        self._check_y(y, self._mom_ndim)
        self._pusher(parallel, size=self._obj.size).val(
            self._obj,
            x,
            1.0 if weight is None else weight,
            *y,
            casting=casting,
            signature=(self.dtype,) * (len(y) + 3),
        )
        return self

    @docfiller_inherit_abc()
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

        >>> CentralMomentsArray.from_vals(x, y, weight=w, mom=(2, 2), axis=0)
        <CentralMomentsArray(mom_ndim=2)>
        array([[[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 6.4741e-01,  3.3791e-02, -5.1117e-03],
                [ 5.0888e-02, -1.0060e-02,  7.0290e-03]],
        <BLANKLINE>
               [[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 3.9793e-01,  6.3224e-03, -2.2669e-02],
                [ 9.3979e-02,  9.9433e-04,  6.5765e-03]]])

        """
        axis, args = prepare_values_for_reduction(
            x,
            1.0 if weight is None else weight,
            *y,
            axis=axis,
            dtype=self.dtype,
            recast=False,
            narrays=self._mom_ndim + 1,
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
        **kwargs: Unpack[WrapNPTransform],
    ) -> NDArray[FloatT]: ...
    @overload
    def cumulative(
        self,
        *,
        out: NDArray[FloatT_],
        dtype: DTypeLike = ...,
        **kwargs: Unpack[WrapNPTransform],
    ) -> NDArray[FloatT_]: ...
    @overload
    def cumulative(
        self,
        *,
        out: None = ...,
        dtype: DTypeLikeArg[FloatT_],
        **kwargs: Unpack[WrapNPTransform],
    ) -> NDArray[FloatT_]: ...
    @overload
    def cumulative(
        self,
        *,
        out: NDArrayAny | None = ...,
        dtype: DTypeLike = ...,
        **kwargs: Unpack[WrapNPTransform],
    ) -> NDArrayAny: ...

    @docfiller_inherit_abc()
    def cumulative(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        axis: AxisReduce | MissingType = MISSING,
        move_axis_to_end: bool = False,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        casting: Casting = "same_kind",
        order: ArrayOrder = None,
        parallel: bool | None = None,
    ) -> NDArrayAny:
        return super().cumulative(
            axis=axis,
            move_axis_to_end=move_axis_to_end,
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

    @docfiller_inherit_abc()
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
        sampler: Sampler,
        out: None = ...,
        dtype: None = ...,
        **kwargs: Unpack[WrapNPResampleAndReduceKwargs],
    ) -> Self: ...
    @overload
    def resample_and_reduce(
        self,
        *,
        sampler: Sampler,
        out: NDArray[FloatT_],
        dtype: DTypeLike = ...,
        **kwargs: Unpack[WrapNPResampleAndReduceKwargs],
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    def resample_and_reduce(
        self,
        *,
        sampler: Sampler,
        out: None = ...,
        dtype: DTypeLikeArg[FloatT_],
        **kwargs: Unpack[WrapNPResampleAndReduceKwargs],
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    def resample_and_reduce(
        self,
        *,
        sampler: Sampler,
        out: NDArrayAny | None = ...,
        dtype: DTypeLike = ...,
        **kwargs: Unpack[WrapNPResampleAndReduceKwargs],
    ) -> CentralMomentsArrayAny: ...

    @docfiller_inherit_abc()
    def resample_and_reduce(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        axis: AxisReduce | MissingType = -1,
        sampler: Sampler,
        move_axis_to_end: bool = False,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        casting: Casting = "same_kind",
        order: ArrayOrder = None,
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
            move_axis_to_end=move_axis_to_end,
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
        **kwargs: Unpack[WrapNPTransform],
    ) -> Self: ...
    @overload
    def jackknife_and_reduce(
        self,
        *,
        data_reduced: ArrayLike | None = ...,
        out: NDArray[FloatT_],
        dtype: DTypeLike = ...,
        **kwargs: Unpack[WrapNPTransform],
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    def jackknife_and_reduce(
        self,
        *,
        data_reduced: ArrayLike | None = ...,
        out: None = ...,
        dtype: DTypeLikeArg[FloatT_],
        **kwargs: Unpack[WrapNPTransform],
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    def jackknife_and_reduce(
        self,
        *,
        data_reduced: ArrayLike | None = ...,
        out: NDArrayAny | None = ...,
        dtype: DTypeLike = ...,
        **kwargs: Unpack[WrapNPTransform],
    ) -> CentralMomentsArrayAny: ...

    @docfiller_inherit_abc()
    def jackknife_and_reduce(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        data_reduced: ArrayLike | None = None,
        axis: AxisReduce | MissingType = -1,
        move_axis_to_end: bool = False,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        casting: Casting = "same_kind",
        order: ArrayOrder = None,
        parallel: bool | None = None,
    ) -> Self | CentralMomentsArrayAny:
        return super().jackknife_and_reduce(
            data_reduced=data_reduced,  # type: ignore[arg-type]
            axis=axis,
            move_axis_to_end=move_axis_to_end,
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

    @docfiller_inherit_abc()
    def reduce(
        self,
        *,
        by: Groups | None = None,
        block: int | None = None,
        axis: AxisReduce | MissingType = MISSING,
        move_axis_to_end: bool = False,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        casting: Casting = "same_kind",
        order: ArrayOrder = None,
        keepdims: bool = False,
        parallel: bool | None = None,
    ) -> Self | CentralMomentsArrayAny:
        if by is None and block is not None:
            by = self._block_by(block, axis=axis)

        if by is None:
            from cmomy.reduction import reduce_data

            obj = reduce_data(
                self._obj,
                mom_ndim=self._mom_ndim,
                axis=axis,
                out=out,
                dtype=dtype,
                casting=casting,
                order=order,
                keepdims=keepdims,
                parallel=parallel,
            )
        else:
            from cmomy.reduction import reduce_data_grouped

            obj = reduce_data_grouped(
                self._obj,
                mom_ndim=self._mom_ndim,
                by=by,
                axis=axis,
                move_axis_to_end=move_axis_to_end,
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
        val_shape: tuple[int, ...] | int | None = ...,
        dtype: None = ...,
        order: ArrayOrderCF = ...,
    ) -> CentralMomentsArray[np.float64]: ...
    @overload
    @classmethod
    def zeros(
        cls,
        *,
        mom: Moments,
        val_shape: tuple[int, ...] | int | None = ...,
        dtype: DTypeLikeArg[FloatT_],
        order: ArrayOrderCF = ...,
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    @classmethod
    def zeros(
        cls,
        *,
        mom: Moments,
        val_shape: tuple[int, ...] | int | None = ...,
        dtype: DTypeLike = ...,
        order: ArrayOrderCF = ...,
    ) -> Self: ...

    @classmethod
    @docfiller_abc()
    def zeros(
        cls,
        *,
        mom: Moments,
        val_shape: tuple[int, ...] | int | None = None,
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
        mom, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)

        vshape: tuple[int, ...]
        if val_shape is None:
            vshape = ()
        elif isinstance(val_shape, int):
            vshape = (val_shape,)
        else:
            vshape = val_shape

        # add in moments
        shape: tuple[int, ...] = (*vshape, *mom_to_mom_shape(mom))

        return cls(np.zeros(shape, dtype=dtype, order=order), mom_ndim=mom_ndim)

    # ** Custom for this class -------------------------------------------------
    def _raise_if_scalar(self, message: str | None = None) -> None:
        if self._obj.ndim <= self.mom_ndim:
            if message is None:  # pragma: no cover
                message = "Not implemented for scalar"
            raise ValueError(message)

    @docfiller.decorate
    def reshape(
        self,
        shape: tuple[int, ...],
        *,
        order: ArrayOrderCFA = None,
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
            Output object with reshaped data.  This will be a view if possilble;
            otherwise, it will be copy.

        See Also
        --------
        numpy.reshape
        new_like

        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> da = CentralMomentsArray.from_vals(rng.random((10, 2, 3)), mom=2, axis=0)
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
        new_shape = (*shape, *self.mom_shape)
        obj = self._obj.reshape(new_shape, order=order)
        return self.new_like(obj)  # type: ignore[return-value]

    @docfiller.decorate
    def resample(
        self,
        indices: ArrayLike,
        *,
        axis: AxisReduce = -1,
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
        axis = validate_axis(axis)

        obj = self.obj
        last_dim = obj.ndim - self.mom_ndim - 1
        if last and axis != last_dim:
            obj = np.moveaxis(obj, axis, last_dim)
            axis = last_dim

        indices = np.asarray(indices, dtype=np.int64)
        obj = np.take(obj, indices, axis=axis)

        return self.new_like(obj)  # type: ignore[return-value]

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
    @docfiller.decorate
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
    ) -> CentralMomentsDataArray:
        r"""
        Create a  :class:`.CentralMomentsData` object from ``self``.


        Parameters
        ----------
        {xr_params}
        {copy_tf}

        Returns
        -------
        output : DataArray


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
                    mom_dims = tuple(mom_dims)  # type: ignore[arg-type]

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

        from cmomy.wrapper.wrap_xr import CentralMomentsData

        return CentralMomentsData(
            obj=out,
            mom_ndim=self._mom_ndim,
            mom_dims=out.dims[-self._mom_ndim :],
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
    ) -> CentralMomentsDataArray:
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

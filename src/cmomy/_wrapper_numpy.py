"""Wrapper object for numpy arrays"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, overload

import numpy as np
import xarray as xr

from cmomy.core.validate import validate_floating_dtype

from .core.compat import copy_if_needed
from .core.utils import mom_to_mom_shape
from .core.validate import (
    validate_axis,
    validate_mom_and_mom_ndim,
)
from .core.xr_utils import select_ndat

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike

    from ._wrapper_xarray import CentralWrapperXArray
    from .core.typing import (
        ArrayLikeArg,
        ArrayOrder,
        ArrayOrderCF,
        ArrayOrderCFA,
        AxisReduce,
        Casting,
        DTypeLikeArg,
        Groups,
        Mom_NDim,
        MomDims,
        Moments,
        MomentsStrict,
        NDArrayAny,
        ScalarT2,
        XArrayAttrsType,
        XArrayCoordsType,
        XArrayDimsType,
        XArrayIndexesType,
        XArrayNameType,
    )
    from .core.typing_compat import Self


from numpy.typing import NDArray

from ._wrapper_abc import CentralWrapperABC
from .core.docstrings import docfiller_wrapper_numpy as docfiller
from .core.typing import ScalarT

docfiller_abc = docfiller.factory_from_parent(CentralWrapperABC)
docfiller_inherit_abc = docfiller.factory_inherit_from_parent(CentralWrapperABC)


@docfiller.inherit(CentralWrapperABC)  # noqa: PLR0904
class CentralWrapperNumpy(CentralWrapperABC[NDArray[ScalarT]], Generic[ScalarT]):  # type: ignore[type-var]
    r"""
    Wrapper to calculate central moments of :class:`~numpy.ndarray` backed arrays.

    Parameters
    ----------
    {copy}
    {order}
    {dtype}

    """

    @overload
    def __init__(
        self,
        obj: ArrayLikeArg[ScalarT],
        *,
        mom_ndim: Mom_NDim = ...,
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
        mom_ndim: Mom_NDim = ...,
        copy: bool | None = ...,
        dtype: DTypeLikeArg[ScalarT],
        order: ArrayOrder = ...,
        fastpath: bool = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        obj: ArrayLike,
        *,
        mom_ndim: Mom_NDim = ...,
        copy: bool | None = ...,
        dtype: DTypeLike = ...,
        order: ArrayOrder = ...,
        fastpath: bool = ...,
    ) -> None: ...

    def __init__(
        self,
        obj: ArrayLike,
        *,
        mom_ndim: Mom_NDim = 1,
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
        elif not isinstance(obj, np.ndarray):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"Must pass ndarray as data.  Not {type(obj)=}"
            raise TypeError(msg)

        super().__init__(
            obj,  # pyright: ignore[reportUnknownArgumentType]
            mom_ndim=mom_ndim,
            fastpath=fastpath,
        )

    # ** Properties ------------------------------------------------------------
    @property
    def mom_shape(self) -> MomentsStrict:
        return self.obj.shape[-self._mom_ndim :]  # type: ignore[return-value]

    def __getitem__(self, key: Any) -> Self:
        """
        Get new object by indexing.

        Note that only objects with the same moment(s) shape are allowed.

        If you want to extract data in general, use `self.to_values()[....]`.
        """
        if self.mom_ndim >= self.obj.ndim:
            msg = "Can only use __getitem__ with extra dimensions."
            raise ValueError(msg)
        return self._new_like(obj=self.obj[key])

    def __iter__(self) -> Iterator[Self]:
        for k in range(self._obj.shape[0]):
            yield self[k]

    # ** Create/copy/new ------------------------------------------------------
    def _new_like(self, obj: NDArray[ScalarT]) -> Self:
        return type(self)(
            obj=obj,
            mom_ndim=self._mom_ndim,
            fastpath=True,
        )

    @overload
    def new_like(
        self,
        obj: NDArray[ScalarT2],
        *,
        verify: bool = ...,
        copy: bool | None = ...,
        dtype: DTypeLike = ...,
        order: ArrayOrder = ...,
        fastpath: bool = ...,
    ) -> CentralWrapperNumpy[ScalarT2]: ...
    @overload
    def new_like(
        self,
        obj: Any = ...,
        *,
        verify: bool = ...,
        copy: bool | None = ...,
        dtype: DTypeLikeArg[ScalarT2],
        order: ArrayOrder = ...,
        fastpath: bool = ...,
    ) -> CentralWrapperNumpy[ScalarT2]: ...
    @overload
    def new_like(
        self,
        obj: None = ...,
        *,
        verify: bool = ...,
        copy: bool | None = ...,
        dtype: DTypeLike = ...,
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
    ) -> CentralWrapperNumpy[Any]: ...

    @docfiller_inherit_abc()
    def new_like(
        self,
        obj: NDArrayAny | None = None,
        *,
        verify: bool = False,
        copy: bool | None = None,
        dtype: DTypeLike = None,
        order: ArrayOrder = None,
        fastpath: bool = False,
    ) -> CentralWrapperNumpy[Any]:
        """
        Parameters
        ----------
        {dtype}

        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> da = CentralWrapperNumpy.from_vals(rng.random(10), mom=3, axis=0)
        >>> da
        <CentralWrapperNumpy(mom_ndim=1)>
        array([10.    ,  0.5505,  0.1014, -0.0178])

        >>> da2 = da.new_like().zero()
        >>> da2
        <CentralWrapperNumpy(mom_ndim=1)>
        array([0., 0., 0., 0.])

        >>> da
        <CentralWrapperNumpy(mom_ndim=1)>
        array([10.    ,  0.5505,  0.1014, -0.0178])
        """
        if obj is None:
            obj = np.zeros_like(self.obj, dtype=dtype)
        else:
            self._raise_if_wrong_mom_shape(obj.shape[-self.mom_ndim :])
            if verify and obj.shape != self.obj.shape:
                msg = f"{obj.shape=} != {self.obj.shape=}"
                raise ValueError(msg)

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
        dtype: DTypeLikeArg[ScalarT2],
        *,
        order: ArrayOrder = ...,
        casting: Casting | None = ...,
        subok: bool | None = ...,
        copy: bool = ...,
    ) -> CentralWrapperNumpy[ScalarT2]: ...
    @overload
    def astype(
        self,
        dtype: None,
        *,
        order: ArrayOrder = ...,
        casting: Casting | None = ...,
        subok: bool | None = ...,
        copy: bool = ...,
    ) -> CentralWrapperNumpy[np.float64]: ...
    @overload
    def astype(
        self,
        dtype: DTypeLike,
        *,
        order: ArrayOrder = ...,
        casting: Casting | None = ...,
        subok: bool | None = ...,
        copy: bool = ...,
    ) -> CentralWrapperNumpy[Any]: ...

    @docfiller_abc()
    def astype(
        self,
        dtype: DTypeLike,
        *,
        order: ArrayOrder = None,
        casting: Casting | None = None,
        subok: bool | None = None,
        copy: bool = False,
    ) -> CentralWrapperNumpy[Any]:
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
        parallel: bool | None = False,
    ) -> Self:
        """
        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.random.default_rng(0)
        >>> xs = rng.random((2, 10))
        >>> datas = [cmomy.reduce_vals(x, mom=2, axis=0) for x in xs]
        >>> da = CentralWrapperNumpy(datas[0], mom_ndim=1)
        >>> da
        <CentralWrapperNumpy(mom_ndim=1)>
        array([10.    ,  0.5505,  0.1014])


        >>> da.push_data(datas[1])
        <CentralWrapperNumpy(mom_ndim=1)>
        array([20.    ,  0.5124,  0.1033])


        Which is equivalent to

        >>> CentralWrapperNumpy.from_vals(xs.reshape(-1), mom=2, axis=0)
        <CentralWrapperNumpy(mom_ndim=1)>
        array([20.    ,  0.5124,  0.1033])

        """
        self._obj = self._push_data_numpy(self._obj, data, parallel=parallel)
        return self

    @docfiller_inherit_abc()
    def push_datas(
        self,
        datas: ArrayLike,
        *,
        axis: AxisReduce = -1,
        parallel: bool | None = None,
    ) -> Self:
        """
        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.random.default_rng(0)
        >>> xs = rng.random((2, 10))
        >>> datas = cmomy.reduce_vals(xs, axis=1, mom=2)
        >>> da = CentralWrapperNumpy.zeros(mom=2)
        >>> da.push_datas(datas, axis=0)
        <CentralWrapperNumpy(mom_ndim=1)>
        array([20.    ,  0.5124,  0.1033])


        Which is equivalent to

        >>> CentralWrapperNumpy.from_vals(xs.reshape(-1), mom=2, axis=0)
        <CentralWrapperNumpy(mom_ndim=1)>
        array([20.    ,  0.5124,  0.1033])
        """
        self._obj = self._push_datas_numpy(
            self._obj, datas, axis=axis, parallel=parallel
        )
        return self

    @docfiller_inherit_abc()
    def push_val(
        self,
        x: ArrayLike,
        *y: ArrayLike,
        weight: ArrayLike | None = None,
        parallel: bool | None = False,
    ) -> Self:
        """
        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.random.default_rng(0)
        >>> x = rng.random((10, 2))
        >>> y = rng.random(10)
        >>> w = rng.random(10)

        >>> da = CentralWrapperNumpy.zeros(val_shape=(2,), mom=(2, 2))
        >>> for xx, yy, ww in zip(x, y, w):
        ...     _ = da.push_val(xx, yy, weight=ww)

        >>> da
        <CentralWrapperNumpy(mom_ndim=2)>
        array([[[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 6.4741e-01,  3.3791e-02, -5.1117e-03],
                [ 5.0888e-02, -1.0060e-02,  7.0290e-03]],
        <BLANKLINE>
               [[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 3.9793e-01,  6.3224e-03, -2.2669e-02],
                [ 9.3979e-02,  9.9433e-04,  6.5765e-03]]])


        Which is the same as

        >>> CentralWrapperNumpy.from_vals(x, y, weight=w, mom=(2, 2), axis=0)
        <CentralWrapperNumpy(mom_ndim=2)>
        array([[[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 6.4741e-01,  3.3791e-02, -5.1117e-03],
                [ 5.0888e-02, -1.0060e-02,  7.0290e-03]],
        <BLANKLINE>
               [[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 3.9793e-01,  6.3224e-03, -2.2669e-02],
                [ 9.3979e-02,  9.9433e-04,  6.5765e-03]]])

        """
        self._obj = self._push_val_numpy(self._obj, x, weight, *y, parallel=parallel)
        return self

    @docfiller_inherit_abc()
    def push_vals(
        self,
        x: ArrayLike,
        *y: ArrayLike,
        axis: AxisReduce = -1,
        weight: ArrayLike | None = None,
        parallel: bool | None = None,
    ) -> Self:
        """
        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.random.default_rng(0)
        >>> x = rng.random((10, 2))
        >>> y = rng.random(10)
        >>> w = rng.random(10)

        >>> da = CentralWrapperNumpy.zeros(val_shape=(2,), mom=(2, 2))
        >>> da.push_vals(x, y, weight=w, axis=0)
        <CentralWrapperNumpy(mom_ndim=2)>
        array([[[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 6.4741e-01,  3.3791e-02, -5.1117e-03],
                [ 5.0888e-02, -1.0060e-02,  7.0290e-03]],
        <BLANKLINE>
               [[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 3.9793e-01,  6.3224e-03, -2.2669e-02],
                [ 9.3979e-02,  9.9433e-04,  6.5765e-03]]])


        Which is the same as

        >>> CentralWrapperNumpy.from_vals(x, y, weight=w, mom=(2, 2), axis=0)
        <CentralWrapperNumpy(mom_ndim=2)>
        array([[[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 6.4741e-01,  3.3791e-02, -5.1117e-03],
                [ 5.0888e-02, -1.0060e-02,  7.0290e-03]],
        <BLANKLINE>
               [[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 3.9793e-01,  6.3224e-03, -2.2669e-02],
                [ 9.3979e-02,  9.9433e-04,  6.5765e-03]]])

        """
        self._obj = self._push_vals_numpy(
            self._obj, x, weight, *y, axis=axis, parallel=parallel
        )
        return self

    # ** Operators ------------------------------------------------------------
    def _check_other_conformable(self, other: Self) -> None:
        if self.obj.shape != other.obj.shape:
            msg = "shape input={self.obj.shape} != other shape = {other.obj.shape}"
            raise ValueError(msg)

    # ** Interface to modules -------------------------------------------------
    # *** .reduction ----------------------------------------------------------
    @docfiller_inherit_abc()
    def reduce(
        self,
        *,
        by: Groups | None = None,
        axis: AxisReduce = -1,
        keepdims: bool = False,
        move_axis_to_end: bool = False,
        parallel: bool | None = None,
    ) -> Self:
        if by is None:
            from .reduction import reduce_data

            obj = reduce_data(
                self._obj,
                mom_ndim=self._mom_ndim,
                axis=axis,
                parallel=parallel,
                keepdims=keepdims,
            )
        else:
            from .reduction import reduce_data_grouped

            obj = reduce_data_grouped(
                self._obj,
                mom_ndim=self._mom_ndim,
                by=by,
                axis=axis,
                move_axis_to_end=move_axis_to_end,
                parallel=parallel,
            )
        return self._new_like(obj)

    # ** Access to underlying statistics --------------------------------------
    def std(self, squeeze: bool = True) -> NDArray[ScalarT]:
        return np.sqrt(self.var(squeeze=squeeze))

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
    ) -> CentralWrapperNumpy[np.float64]: ...
    @overload
    @classmethod
    def zeros(
        cls,
        *,
        mom: Moments,
        val_shape: tuple[int, ...] | int | None = ...,
        dtype: DTypeLikeArg[ScalarT2],
        order: ArrayOrderCF = ...,
    ) -> CentralWrapperNumpy[ScalarT2]: ...
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
    ) -> CentralWrapperNumpy[Any] | Self:
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

    @overload
    @classmethod
    def from_vals(
        cls,
        x: ArrayLikeArg[ScalarT2],
        *y: ArrayLike,
        mom: Moments,
        axis: AxisReduce = -1,
        weight: ArrayLike | None = ...,
        keepdims: bool = ...,
        dtype: None = ...,
        out: None = ...,
        parallel: bool | None = ...,
    ) -> CentralWrapperNumpy[ScalarT2]: ...
    # out
    @overload
    @classmethod
    def from_vals(
        cls,
        x: ArrayLike,
        *y: ArrayLike,
        mom: Moments,
        axis: AxisReduce = -1,
        weight: ArrayLike | None = ...,
        keepdims: bool = ...,
        dtype: DTypeLike = ...,
        out: NDArray[ScalarT2],
        parallel: bool | None = ...,
    ) -> CentralWrapperNumpy[ScalarT2]: ...
    # dtype
    @overload
    @classmethod
    def from_vals(
        cls,
        x: ArrayLike,
        *y: ArrayLike,
        mom: Moments,
        axis: AxisReduce = -1,
        weight: ArrayLike | None = ...,
        keepdims: bool = ...,
        dtype: DTypeLikeArg[ScalarT2],
        out: None = ...,
        parallel: bool | None = ...,
    ) -> CentralWrapperNumpy[ScalarT2]: ...
    # fallback
    @overload
    @classmethod
    def from_vals(
        cls,
        x: ArrayLike,
        *y: ArrayLike,
        mom: Moments,
        axis: AxisReduce = -1,
        weight: ArrayLike | None = ...,
        keepdims: bool = ...,
        dtype: DTypeLike = ...,
        out: NDArrayAny | None = ...,
        parallel: bool | None = ...,
    ) -> Self: ...

    @classmethod
    @docfiller_abc()
    def from_vals(
        cls,
        x: ArrayLike,
        *y: ArrayLike,
        mom: Moments,
        axis: AxisReduce = -1,
        weight: ArrayLike | None = None,
        keepdims: bool = False,
        dtype: DTypeLike = None,
        out: NDArrayAny | None = None,
        parallel: bool | None = None,
    ) -> CentralWrapperNumpy[Any] | Self:
        """
        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> x = rng.random((100, 3))
        >>> da = CentralWrapperNumpy.from_vals(x, axis=0, mom=2)
        >>> da
        <CentralWrapperNumpy(mom_ndim=1)>
        array([[1.0000e+02, 5.5313e-01, 8.8593e-02],
               [1.0000e+02, 5.5355e-01, 7.1942e-02],
               [1.0000e+02, 5.1413e-01, 1.0407e-01]])
        """
        mom_strict, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)

        from .reduction import reduce_vals

        data = reduce_vals(
            x,
            *y,
            mom=mom_strict,
            weight=weight,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            out=out,
            parallel=parallel,
        )
        return cls(obj=data, mom_ndim=mom_ndim)

    @overload
    @classmethod
    def from_resample_vals(
        cls,
        x: ArrayLikeArg[ScalarT2],
        *y: ArrayLike,
        mom: Moments,
        axis: AxisReduce = ...,
        freq: ArrayLike | None = ...,
        nrep: int | None = ...,
        rng: np.random.Generator | None = ...,
        move_axis_to_end: bool = ...,
        weight: ArrayLike | None = ...,
        parallel: bool | None = ...,
        dtype: None = ...,
        out: None = ...,
    ) -> CentralWrapperNumpy[ScalarT2]: ...
    @overload
    @classmethod
    def from_resample_vals(
        cls,
        x: Any,
        *y: ArrayLike,
        mom: Moments,
        axis: AxisReduce = ...,
        freq: ArrayLike | None = ...,
        nrep: int | None = ...,
        rng: np.random.Generator | None = ...,
        move_axis_to_end: bool = ...,
        weight: ArrayLike | None = ...,
        parallel: bool | None = ...,
        dtype: DTypeLike = ...,
        out: NDArray[ScalarT2],
    ) -> CentralWrapperNumpy[ScalarT2]: ...
    @overload
    @classmethod
    def from_resample_vals(
        cls,
        x: Any,
        *y: ArrayLike,
        mom: Moments,
        axis: AxisReduce = ...,
        freq: ArrayLike | None = ...,
        nrep: int | None = ...,
        rng: np.random.Generator | None = ...,
        move_axis_to_end: bool = ...,
        weight: ArrayLike | None = ...,
        parallel: bool | None = ...,
        dtype: DTypeLikeArg[ScalarT2],
        out: None = ...,
    ) -> CentralWrapperNumpy[ScalarT2]: ...
    @overload
    @classmethod
    def from_resample_vals(
        cls,
        x: Any,
        *y: ArrayLike,
        mom: Moments,
        axis: AxisReduce = ...,
        freq: ArrayLike | None = ...,
        nrep: int | None = ...,
        rng: np.random.Generator | None = ...,
        move_axis_to_end: bool = ...,
        weight: ArrayLike | None = ...,
        parallel: bool | None = ...,
        dtype: DTypeLike = ...,
        out: NDArrayAny | None = ...,
    ) -> Self: ...

    @classmethod
    @docfiller_abc()
    def from_resample_vals(
        cls,
        x: ArrayLike,
        *y: ArrayLike,
        mom: Moments,
        axis: AxisReduce = -1,
        freq: ArrayLike | None = None,
        nrep: int | None = None,
        rng: np.random.Generator | None = None,
        move_axis_to_end: bool = True,
        weight: ArrayLike | None = None,
        parallel: bool | None = None,
        dtype: DTypeLike = None,
        out: NDArrayAny | None = None,
    ) -> CentralWrapperNumpy[Any]:
        """
        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> from cmomy.resample import random_freq
        >>> rng = default_rng(0)
        >>> ndat, nrep = 10, 3
        >>> x = rng.random(ndat)
        >>> freq = random_freq(nrep=nrep, ndat=ndat)
        >>> da = CentralWrapperNumpy.from_resample_vals(x, freq=freq, axis=0, mom=2)
        >>> da
        <CentralWrapperNumpy(mom_ndim=1)>
        array([[10.    ,  0.5397,  0.0757],
               [10.    ,  0.5848,  0.0618],
               [10.    ,  0.5768,  0.0564]])

        Note that this is equivalent to (though in general faster than)

        >>> from cmomy.resample import freq_to_indices
        >>> indices = freq_to_indices(freq)
        >>> x_resamp = np.take(x, indices, axis=0)
        >>> da = CentralWrapperNumpy.from_vals(x_resamp, axis=1, mom=2)
        >>> da
        <CentralWrapperNumpy(mom_ndim=1)>
        array([[10.    ,  0.5397,  0.0757],
               [10.    ,  0.5848,  0.0618],
               [10.    ,  0.5768,  0.0564]])

        """
        from .resample import resample_vals

        mom_strict, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
        data = resample_vals(
            x,
            *y,
            freq=freq,
            nrep=nrep,
            rng=rng,
            mom=mom_strict,
            axis=axis,
            move_axis_to_end=move_axis_to_end,
            weight=weight,
            parallel=parallel,
            dtype=dtype,
            out=out,
        )

        return cls(obj=data, mom_ndim=mom_ndim)

    @classmethod
    @docfiller_abc()
    def from_raw(
        cls,
        raw: NDArray[ScalarT],
        *,
        mom_ndim: Mom_NDim,
        **kwargs: Any,
    ) -> Self:
        """
        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> x = rng.random(10)
        >>> raw_x = (x[:, None] ** np.arange(5)).mean(axis=0)

        >>> dx_raw = CentralWrapperNumpy.from_raw(raw_x, mom_ndim=1)
        >>> print(dx_raw.mean())
        0.5505105129032412
        >>> dx_raw.cmom()
        array([ 1.    ,  0.    ,  0.1014, -0.0178,  0.02  ])

        Which is equivalent to creating raw moments from values
        >>> dx_cen = CentralWrapperNumpy.from_vals(x, axis=0, mom=4)
        >>> print(dx_cen.mean())
        0.5505105129032413
        >>> dx_cen.cmom()
        array([ 1.    ,  0.    ,  0.1014, -0.0178,  0.02  ])

        But note that calculating using from_raw can lead to
        numerical issues.  For example

        >>> y = x + 10000
        >>> raw_y = (y[:, None] ** np.arange(5)).mean(axis=0)
        >>> dy_raw = CentralWrapperNumpy.from_raw(raw_y, mom_ndim=1)
        >>> print(dy_raw.mean() - 10000)
        0.5505105129050207

        Note that the central moments don't match!

        >>> np.isclose(dy_raw.cmom(), dx_raw.cmom())
        array([ True,  True,  True, False, False])

        >>> dy_cen = CentralWrapperNumpy.from_vals(y, axis=0, mom=4)
        >>> print(dy_cen.mean() - 10000)
        0.5505105129032017
        >>> dy_cen.cmom()  # this matches above
        array([ 1.    ,  0.    ,  0.1014, -0.0178,  0.02  ])
        """
        return super().from_raw(raw=raw, mom_ndim=mom_ndim, **kwargs)

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
        output : CentralWrapperNumpy
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
        >>> da = CentralWrapperNumpy.from_vals(rng.random((10, 2, 3)), mom=2, axis=0)
        >>> da
        <CentralWrapperNumpy(mom_ndim=1)>
        array([[[10.    ,  0.5205,  0.0452],
                [10.    ,  0.4438,  0.0734],
                [10.    ,  0.5038,  0.1153]],
        <BLANKLINE>
               [[10.    ,  0.5238,  0.1272],
                [10.    ,  0.628 ,  0.0524],
                [10.    ,  0.412 ,  0.0865]]])

        >>> da.reshape(shape=(-1,))
        <CentralWrapperNumpy(mom_ndim=1)>
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
        # TODO(wpk):  figure out how to better implement new_like to make this work correctly...
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
        obj = np.take(obj, indices, axis=axis)  # pyright: ignore[reportUnknownMemberType]

        return self.new_like(obj)  # type: ignore[return-value]

    @docfiller.decorate
    def block(
        self,
        block_size: int | None,
        *,
        axis: AxisReduce = -1,
        parallel: bool | None = None,
        # **kwargs: Any,
    ) -> Self:
        """
        Block average reduction.

        Parameters
        ----------
        block_size : int
            number of consecutive records to combine
        {axis}

        Returns
        -------
        output : object
            Block averaged data of shape
            ``(..., shape[axis-1],nblock, shape[axis+1], ..., mom_0, ...)``
            Where ``shape=self.shape``.  That is, the blocked coordinates are
            last.


        Notes
        -----
        The block averaged `axis` will be moved to the front of the output data.

        See Also
        --------
        reduce
        """
        self._raise_if_scalar()

        n = select_ndat(self._obj, axis=axis, mom_ndim=self._mom_ndim)

        if block_size is None:
            block_size = n

        nblock = n // block_size

        by = np.arange(nblock).repeat(block_size)
        if len(by) != n:
            by = np.pad(by, (0, n - len(by)), mode="constant", constant_values=-1)

        return self.reduce(axis=axis, by=by, parallel=parallel)

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
        dims: XArrayDimsType = None,
        attrs: XArrayAttrsType = None,
        coords: XArrayCoordsType = None,
        name: XArrayNameType = None,
        indexes: XArrayIndexesType = None,  # noqa: ARG002
        mom_dims: MomDims | None = None,
        template: xr.DataArray | None = None,
        copy: bool = False,
    ) -> xr.DataArray:
        """
        Create a :class:`xarray.DataArray` representation of underlying data.

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
        >>> da = CentralWrapperNumpy(rng.random((1, 2, 4)), mom_ndim=1)
        >>> da
        <CentralWrapperNumpy(mom_ndim=1)>
        array([[[0.637 , 0.2698, 0.041 , 0.0165],
                [0.8133, 0.9128, 0.6066, 0.7295]]])

        Default constructor

        >>> da.to_dataarray()
        <xarray.DataArray (dim_0: 1, dim_1: 2, mom_0: 4)> Size: 64B
        array([[[0.637 , 0.2698, 0.041 , 0.0165],
                [0.8133, 0.9128, 0.6066, 0.7295]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0

        Setting attributes

        >>> da.to_dataarray()
        <xarray.DataArray (dim_0: 1, dim_1: 2, mom_0: 4)> Size: 64B
        array([[[0.637 , 0.2698, 0.041 , 0.0165],
                [0.8133, 0.9128, 0.6066, 0.7295]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0
        >>> da
        <CentralWrapperNumpy(mom_ndim=1)>
        array([[[0.637 , 0.2698, 0.041 , 0.0165],
                [0.8133, 0.9128, 0.6066, 0.7295]]])

        """
        data = self.obj
        if copy:
            data = data.copy()

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
                dims = tuple(dims)  # type: ignore[arg-type]

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

                if len(mom_dims) != self.mom_ndim:
                    msg = f"{mom_dims=} != {self.mom_ndim=}"
                    raise ValueError(msg)

                dims_output = dims + mom_dims

            else:
                msg = f"Problem with {dims}, {mom_dims}.  Total length should be {self.obj.ndim}"
                raise ValueError(msg)
            out = xr.DataArray(
                data, dims=dims_output, coords=coords, attrs=attrs, name=name
            )

        return out

    @docfiller.decorate
    def to_x(
        self,
        *,
        dims: XArrayDimsType = None,
        attrs: XArrayAttrsType = None,
        coords: XArrayCoordsType = None,
        name: XArrayNameType = None,
        indexes: XArrayIndexesType = None,
        mom_dims: MomDims | None = None,
        template: xr.DataArray | None = None,
        copy: bool = False,
    ) -> CentralWrapperXArray[xr.DataArray]:
        """
        Create an :class:`xarray.DataArray` representation of underlying data.

        Parameters
        ----------
        {xr_params}
        {copy_tf}

        Returns
        -------
        output : CentralWrapperXArray

        See Also
        --------
        to_dataarray

        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> da = CentralWrapperNumpy.from_vals(rng.random((10, 1, 2)), axis=0, mom=2)
        >>> da
        <CentralWrapperNumpy(mom_ndim=1)>
        array([[[10.    ,  0.6207,  0.0647],
                [10.    ,  0.404 ,  0.1185]]])

        Default constructor

        >>> da.to_x()
        <CentralWrapperXArray(mom_ndim=1)>
        <xarray.DataArray (dim_0: 1, dim_1: 2, mom_0: 3)> Size: 48B
        array([[[10.    ,  0.6207,  0.0647],
                [10.    ,  0.404 ,  0.1185]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0

        Setting attributes

        >>> da.to_x()
        <CentralWrapperXArray(mom_ndim=1)>
        <xarray.DataArray (dim_0: 1, dim_1: 2, mom_0: 3)> Size: 48B
        array([[[10.    ,  0.6207,  0.0647],
                [10.    ,  0.404 ,  0.1185]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0
        >>> da
        <CentralWrapperNumpy(mom_ndim=1)>
        array([[[10.    ,  0.6207,  0.0647],
                [10.    ,  0.404 ,  0.1185]]])

        """
        from ._wrapper_xarray import CentralWrapperXArray

        data = self.to_dataarray(
            dims=dims,
            attrs=attrs,
            coords=coords,
            name=name,
            indexes=indexes,
            mom_dims=mom_dims,
            template=template,
            copy=copy,
        )
        return CentralWrapperXArray(
            obj=data, mom_ndim=self._mom_ndim
        )  # , fastpath=True)

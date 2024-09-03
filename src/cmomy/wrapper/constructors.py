"""
Attempt at a more generic wrapper class

Idea is to Wrap ndarray, xr.DataArray, and xr.Dataset objects...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr

from cmomy.core.compat import copy_if_needed
from cmomy.core.docstrings import docfiller
from cmomy.core.missing import MISSING
from cmomy.core.validate import (
    is_xarray,
    validate_mom_and_mom_ndim,
    validate_mom_dims,
)

from .wrap_np import CentralMomentsArray
from .wrap_xr import CentralMomentsXArray

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from cmomy.core.typing import (
        ApplyUFuncKwargs,
        ArrayLikeArg,
        ArrayOrder,
        ArrayOrderCF,
        AxisReduce,
        Casting,
        CentralMomentsArrayAny,
        CentralMomentsArrayT,
        CentralMomentsDataAny,
        CentralMomentsDataArray,
        CentralMomentsDataset,
        CentralMomentsXArrayT,
        DimsReduce,
        DTypeLikeArg,
        FloatT,
        KeepAttrs,
        MissingCoreDimOptions,
        MissingType,
        Mom_NDim,
        MomDims,
        Moments,
        NDArrayAny,
        RngTypes,
        XArrayT,
    )


# * General wrapper -----------------------------------------------------------
@overload
def wrap(  # pyright: ignore[reportOverlappingOverload]
    obj: XArrayT,
    *,
    mom_ndim: Mom_NDim = ...,
    mom_dims: MomDims | None = ...,
    dtype: DTypeLike | Mapping[str, DTypeLike] = ...,
    copy: bool | None = ...,
    fastpath: bool = ...,
) -> CentralMomentsXArray[XArrayT]: ...
@overload
def wrap(  # type: ignore[misc]
    obj: xr.DataArray | xr.Dataset,
    *,
    mom_ndim: Mom_NDim = ...,
    mom_dims: MomDims | None = ...,
    dtype: DTypeLike | Mapping[str, DTypeLike] = ...,
    copy: bool | None = ...,
    fastpath: bool = ...,
) -> CentralMomentsDataArray | CentralMomentsDataset: ...
@overload
def wrap(
    obj: ArrayLikeArg[FloatT],
    *,
    mom_ndim: Mom_NDim = ...,
    mom_dims: MomDims | None = ...,
    dtype: None = ...,
    copy: bool | None = ...,
    fastpath: bool = ...,
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap(
    obj: ArrayLike,
    *,
    mom_ndim: Mom_NDim = ...,
    mom_dims: MomDims | None = ...,
    dtype: DTypeLikeArg[FloatT],
    copy: bool | None = ...,
    fastpath: bool = ...,
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap(
    obj: ArrayLike,
    *,
    mom_ndim: Mom_NDim = ...,
    mom_dims: MomDims | None = ...,
    dtype: DTypeLike = ...,
    copy: bool | None = ...,
    fastpath: bool = ...,
) -> CentralMomentsArrayAny: ...
@overload
def wrap(
    obj: Any,
    *,
    mom_ndim: Mom_NDim = ...,
    mom_dims: MomDims | None = ...,
    dtype: Any = ...,
    copy: bool | None = ...,
    fastpath: bool = ...,
) -> Any: ...


def wrap(  # type: ignore[misc]
    obj: ArrayLike | XArrayT,
    *,
    mom_ndim: Mom_NDim = 1,
    mom_dims: MomDims | None = None,
    dtype: DTypeLike | Mapping[str, DTypeLike] = None,
    copy: bool | None = False,
    fastpath: bool = False,
) -> CentralMomentsArray[Any] | CentralMomentsXArray[XArrayT]:
    """Wrap object with central moments class."""
    if is_xarray(obj):
        if not fastpath:
            copy = copy_if_needed(copy)
            if dtype is not None:
                obj = obj.astype(dtype, copy=copy)  # pyright: ignore[reportUnknownMemberType]
            elif copy:
                obj = obj.copy(deep=True)

        return CentralMomentsXArray(
            obj=obj,  # type: ignore[arg-type]
            mom_ndim=mom_ndim,
            mom_dims=mom_dims,
            fastpath=fastpath,
        )

    return CentralMomentsArray(
        obj=obj,  # type: ignore[arg-type]
        mom_ndim=mom_ndim,
        fastpath=fastpath,
        dtype=dtype,  # type: ignore[arg-type]
        copy=copy,
    )


# * Zeros like ----------------------------------------------------------------
@overload
def zeros_like(
    c: CentralMomentsXArrayT,
    *,
    mom_ndim: Mom_NDim = ...,
    mom_dims: MomDims | None = ...,
    dtype: DTypeLike | Mapping[Any, DTypeLike] = ...,
    order: ArrayOrder = ...,
    subok: bool = ...,
    **kwargs: Any,
) -> CentralMomentsXArrayT: ...
@overload
def zeros_like(
    c: CentralMomentsArrayT,
    *,
    mom_ndim: Mom_NDim = ...,
    mom_dims: MomDims | None = ...,
    dtype: None = ...,
    order: ArrayOrder = ...,
    subok: bool = ...,
    **kwargs: Any,
) -> CentralMomentsArrayT: ...
@overload
def zeros_like(
    c: CentralMomentsArrayAny,
    *,
    mom_ndim: Mom_NDim = ...,
    mom_dims: MomDims | None = ...,
    dtype: DTypeLikeArg[FloatT],
    order: ArrayOrder = ...,
    subok: bool = ...,
    **kwargs: Any,
) -> CentralMomentsArray[FloatT]: ...
@overload
def zeros_like(
    c: CentralMomentsArrayAny | CentralMomentsDataAny,
    *,
    mom_ndim: Mom_NDim = ...,
    mom_dims: MomDims | None = ...,
    dtype: DTypeLike | Mapping[Any, DTypeLike] = ...,
    order: ArrayOrder = ...,
    subok: bool = ...,
    **kwargs: Any,
) -> CentralMomentsArrayAny | CentralMomentsDataAny: ...


def zeros_like(
    c: CentralMomentsArrayAny | CentralMomentsDataAny,
    *,
    mom_ndim: Mom_NDim = 1,
    mom_dims: MomDims | None = None,
    dtype: DTypeLike | Mapping[Any, DTypeLike] = None,
    order: ArrayOrder = None,
    subok: bool = True,
    **kwargs: Any,
) -> CentralMomentsArrayAny | CentralMomentsDataAny:
    """
    Create new wrapped object like given object.

    Parameters
    ----------
    fill_value : scalar or dict-like
        Value to fill new object with.
    {mom_ndim}
    {mom_dims}
    {dtype}
    {order}
    {subok}
    **kwargs
        Extra arguments to :func:`xarray.zeros_like`
    """
    if isinstance(c, CentralMomentsXArray):
        return wrap(
            xr.zeros_like(
                c.obj,
                dtype=dtype,  # type: ignore[arg-type]
                **kwargs,
            ),
            mom_ndim=mom_ndim,
            mom_dims=mom_dims,
        )
    return wrap(  # type: ignore[no-any-return]
        np.zeros_like(
            c.obj,
            dtype=dtype,  # type: ignore[arg-type]
            order=order,
            subok=subok,
            **kwargs,
        ),
        mom_ndim=mom_ndim,
    )


# * From vals -----------------------------------------------------------------
@overload
def wrap_reduce_vals(  # pyright: ignore[reportOverlappingOverload]
    x: XArrayT,
    *y: ArrayLike | xr.DataArray | XArrayT,
    mom: Moments,
    weight: ArrayLike | xr.DataArray | XArrayT | None = ...,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    casting: Casting = ...,
    order: ArrayOrderCF = ...,
    keepdims: bool = False,
    parallel: bool | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> CentralMomentsXArray[XArrayT]: ...
@overload
def wrap_reduce_vals(  # type: ignore[misc]
    x: xr.DataArray | xr.Dataset,
    *y: ArrayLike | xr.DataArray | xr.Dataset,
    mom: Moments,
    weight: ArrayLike | xr.DataArray | xr.Dataset | None = ...,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    casting: Casting = ...,
    order: ArrayOrderCF = ...,
    keepdims: bool = False,
    parallel: bool | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> CentralMomentsDataArray | CentralMomentsDataset: ...
@overload
def wrap_reduce_vals(
    x: ArrayLikeArg[FloatT],
    *y: ArrayLike,
    mom: Moments,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    out: None = ...,
    dtype: None = ...,
    casting: Casting = ...,
    order: ArrayOrderCF = ...,
    keepdims: bool = False,
    parallel: bool | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap_reduce_vals(
    x: ArrayLike,
    *y: ArrayLike,
    mom: Moments,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    casting: Casting = ...,
    order: ArrayOrderCF = ...,
    keepdims: bool = False,
    parallel: bool | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap_reduce_vals(
    x: ArrayLike,
    *y: ArrayLike,
    mom: Moments,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    casting: Casting = ...,
    order: ArrayOrderCF = ...,
    keepdims: bool = False,
    parallel: bool | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap_reduce_vals(
    x: ArrayLike,
    *y: ArrayLike,
    mom: Moments,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    casting: Casting = ...,
    order: ArrayOrderCF = ...,
    keepdims: bool = False,
    parallel: bool | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> CentralMomentsArrayAny: ...
@overload
def wrap_reduce_vals(
    x: Any,
    *y: Any,
    mom: Moments,
    weight: Any = ...,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    mom_dims: MomDims | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    casting: Casting = ...,
    order: ArrayOrderCF = ...,
    keepdims: bool = False,
    parallel: bool | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> Any: ...


@docfiller.decorate  # type: ignore[misc]
def wrap_reduce_vals(  # pyright: ignore[reportInconsistentOverload]
    x: ArrayLike | XArrayT,
    *y: ArrayLike | xr.DataArray | XArrayT,
    mom: Moments,
    weight: ArrayLike | xr.DataArray | XArrayT | None = None,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_dims: MomDims | None = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderCF = None,
    keepdims: bool = False,
    parallel: bool | None = None,
    keep_attrs: KeepAttrs = None,
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> CentralMomentsArrayAny | CentralMomentsXArray[XArrayT]:
    """
    Create from observations/values.

    Parameters
    ----------
    x : array-like
        Values to reduce.
    *y : array-like
        Additional values (needed if ``len(mom)==2``).
    weight : scalar or array-like
        Optional weight.  If scalar or array, attempt to
        broadcast to `x0.shape`
    {mom}
    {axis}
    {dim}
    {mom_dims}
    {keepdims}
    {out}
    {dtype}
    {casting}
    {order}
    {keepdims}
    {parallel}
    {keep_attrs}
    {on_missing_core_dim}
    {apply_ufunc_kwargs}

    Returns
    -------
    object
        Instance of calling class.

    See Also
    --------
    push_vals
    ~.reduction.reduce_vals

    Examples
    --------
    >>> import cmomy
    >>> rng = cmomy.random.default_rng(0)
    >>> x = rng.random((100, 3))
    >>> da = cmomy.wrapped.wrap_reduce_vals(x, axis=0, mom=2)
    >>> da
    <CentralMomentsArray(mom_ndim=1)>
    array([[1.0000e+02, 5.5313e-01, 8.8593e-02],
            [1.0000e+02, 5.5355e-01, 7.1942e-02],
            [1.0000e+02, 5.1413e-01, 1.0407e-01]])

    """
    from cmomy.reduction import reduce_vals

    mom, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
    kws = _get_mom_dims_kws(x, mom_dims, mom_ndim)
    obj = reduce_vals(  # type: ignore[type-var, misc]
        x,  # pyright: ignore[reportArgumentType]
        *y,
        mom=mom,
        weight=weight,
        axis=axis,
        dim=dim,
        **kws,
        keepdims=keepdims,
        parallel=parallel,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        keep_attrs=keep_attrs,
        on_missing_core_dim=on_missing_core_dim,
        apply_ufunc_kwargs=apply_ufunc_kwargs,
    )

    return wrap(  # type: ignore[no-any-return]
        obj=obj,  # pyright: ignore[reportUnknownArgumentType]
        mom_ndim=mom_ndim,
        **kws,
        fastpath=True,
    )


# * resample vals -------------------------------------------------------------
@overload
def wrap_resample_vals(  # pyright: ignore[reportOverlappingOverload]
    x: XArrayT,
    *y: xr.DataArray | XArrayT,
    mom: Moments,
    weight: xr.DataArray | XArrayT | None = ...,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    freq: ArrayLike | xr.DataArray | XArrayT | None = ...,
    nrep: int | None = ...,
    rng: RngTypes | None = ...,
    move_axis_to_end: bool = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    casting: Casting = ...,
    order: ArrayOrderCF = ...,
    parallel: bool | None = ...,
    mom_dims: MomDims | None = ...,
    rep_dim: str = "rep",
    keep_attrs: bool = True,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> CentralMomentsXArray[XArrayT]: ...
@overload
def wrap_resample_vals(  # type: ignore[misc]
    x: xr.DataArray | xr.Dataset,
    *y: xr.DataArray | XArrayT,
    mom: Moments,
    weight: xr.DataArray | XArrayT | None = ...,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    freq: ArrayLike | xr.DataArray | XArrayT | None = ...,
    nrep: int | None = ...,
    rng: RngTypes | None = ...,
    move_axis_to_end: bool = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    casting: Casting = ...,
    order: ArrayOrderCF = ...,
    parallel: bool | None = ...,
    mom_dims: MomDims | None = ...,
    rep_dim: str = "rep",
    keep_attrs: bool = True,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> CentralMomentsDataArray | CentralMomentsDataset: ...
@overload
def wrap_resample_vals(
    x: ArrayLikeArg[FloatT],
    *y: ArrayLike,
    mom: Moments,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    freq: ArrayLike | None = ...,
    nrep: int | None = ...,
    rng: RngTypes | None = ...,
    move_axis_to_end: bool = ...,
    out: None = ...,
    dtype: None = ...,
    casting: Casting = ...,
    order: ArrayOrderCF = ...,
    parallel: bool | None = ...,
    mom_dims: MomDims | None = ...,
    rep_dim: str = "rep",
    keep_attrs: bool = True,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap_resample_vals(
    x: ArrayLike,
    *y: ArrayLike,
    mom: Moments,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    freq: ArrayLike | None = ...,
    nrep: int | None = ...,
    rng: RngTypes | None = ...,
    move_axis_to_end: bool = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    casting: Casting = ...,
    order: ArrayOrderCF = ...,
    parallel: bool | None = ...,
    mom_dims: MomDims | None = ...,
    rep_dim: str = "rep",
    keep_attrs: bool = True,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap_resample_vals(
    x: ArrayLike,
    *y: ArrayLike,
    mom: Moments,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    freq: ArrayLike | None = ...,
    nrep: int | None = ...,
    rng: RngTypes | None = ...,
    move_axis_to_end: bool = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    casting: Casting = ...,
    order: ArrayOrderCF = ...,
    parallel: bool | None = ...,
    mom_dims: MomDims | None = ...,
    rep_dim: str = "rep",
    keep_attrs: bool = True,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap_resample_vals(
    x: ArrayLike,
    *y: ArrayLike,
    mom: Moments,
    weight: ArrayLike | None = ...,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    freq: ArrayLike | None = ...,
    nrep: int | None = ...,
    rng: RngTypes | None = ...,
    move_axis_to_end: bool = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    casting: Casting = ...,
    order: ArrayOrderCF = ...,
    parallel: bool | None = ...,
    mom_dims: MomDims | None = ...,
    rep_dim: str = "rep",
    keep_attrs: bool = True,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> CentralMomentsArrayAny: ...
@overload
def wrap_resample_vals(
    x: Any,
    *y: Any,
    mom: Moments,
    weight: Any = ...,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    freq: Any = ...,
    nrep: int | None = ...,
    rng: RngTypes | None = ...,
    move_axis_to_end: bool = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    casting: Casting = ...,
    order: ArrayOrderCF = ...,
    parallel: bool | None = ...,
    mom_dims: MomDims | None = ...,
    rep_dim: str = "rep",
    keep_attrs: bool = True,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> Any: ...


@docfiller.decorate  # type: ignore[misc]
def wrap_resample_vals(  # pyright: ignore[reportInconsistentOverload] # noqa: PLR0913
    x: ArrayLike | XArrayT,
    *y: ArrayLike | xr.DataArray | XArrayT,
    mom: Moments,
    weight: ArrayLike | xr.DataArray | XArrayT | None = None,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    freq: ArrayLike | xr.DataArray | XArrayT | None = None,
    nrep: int | None = None,
    rng: RngTypes | None = None,
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
) -> CentralMomentsArrayAny | CentralMomentsXArray[XArrayT]:
    """
    Create from resample observations/values.

    This effectively resamples `x`.

    Parameters
    ----------
    x : array-like
        Observations.
    *y : array-like,
        Additional values (needed if ``len(mom) > 1``).
    {mom}
    {weight}
    {axis}
    {dim}
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
    object
        Instance of calling class


    See Also
    --------
    ~.resample.resample_vals
    ~.resample.randsamp_freq
    ~.resample.freq_to_indices
    ~.resample.indices_to_freq
    """
    mom, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
    kws = _get_mom_dims_kws(x, mom_dims, mom_ndim)

    from cmomy.resample import resample_vals

    obj = resample_vals(  # type: ignore[type-var, misc]
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
        **kws,
        rep_dim=rep_dim,
        keep_attrs=keep_attrs,
        on_missing_core_dim=on_missing_core_dim,
        apply_ufunc_kwargs=apply_ufunc_kwargs,
        dtype=dtype,
        out=out,
        casting=casting,
        order=order,
    )
    return wrap(  # type: ignore[no-any-return]
        obj=obj,  # pyright: ignore[reportUnknownArgumentType]
        mom_ndim=mom_ndim,
        **kws,
        fastpath=True,
    )


# * From raw ------------------------------------------------------------------
@overload
def wrap_raw(  # pyright: ignore[reportOverlappingOverload]
    raw: XArrayT,
    *,
    mom_ndim: Mom_NDim,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    casting: Casting = ...,
    order: ArrayOrder = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> CentralMomentsXArray[XArrayT]: ...
@overload
def wrap_raw(  # type: ignore[misc]
    raw: xr.DataArray | xr.Dataset,
    *,
    mom_ndim: Mom_NDim,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    casting: Casting = ...,
    order: ArrayOrder = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> CentralMomentsDataArray | CentralMomentsDataset: ...
@overload
def wrap_raw(
    raw: ArrayLikeArg[FloatT],
    *,
    mom_ndim: Mom_NDim,
    out: None = ...,
    dtype: None = ...,
    casting: Casting = ...,
    order: ArrayOrder = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap_raw(
    raw: ArrayLike,
    *,
    mom_ndim: Mom_NDim,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    casting: Casting = ...,
    order: ArrayOrder = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap_raw(
    raw: ArrayLike,
    *,
    mom_ndim: Mom_NDim,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    casting: Casting = ...,
    order: ArrayOrder = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap_raw(
    raw: ArrayLike,
    *,
    mom_ndim: Mom_NDim,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    casting: Casting = ...,
    order: ArrayOrder = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> CentralMomentsArrayAny: ...
@overload
def wrap_raw(
    raw: Any,
    *,
    mom_ndim: Mom_NDim,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    casting: Casting = ...,
    order: ArrayOrder = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> Any: ...


@docfiller.decorate  # type: ignore[misc]
def wrap_raw(  # pyright: ignore[reportInconsistentOverload]
    raw: ArrayLike | XArrayT,
    *,
    mom_ndim: Mom_NDim,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrder = None,
    keep_attrs: KeepAttrs = None,
    mom_dims: MomDims | None = None,
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> CentralMomentsArrayAny | CentralMomentsXArray[XArrayT]:
    """
    Create object from raw moment data.

    raw[..., i, j] = <x**i y**j>.
    raw[..., 0, 0] = `weight`


    Parameters
    ----------
    raw : {t_array}
        Raw moment array.
    {mom_ndim}
    {out}
    {dtype}
    {casting}
    {order}
    {keep_attrs}
    {mom_dims}
    {on_missing_core_dim}
    {apply_ufunc_kwargs}

    Returns
    -------
    output : {klass}

    See Also
    --------
    to_raw
    rmom
    .convert.moments_type

    Notes
    -----
    Weights are taken from ``raw[...,0, 0]``.
    Using raw moments can result in numerical issues, especially for higher moments.  Use with care.

    Examples
    --------
    >>> import cmomy
    >>> rng = cmomy.random.default_rng(0)
    >>> x = rng.random(10)
    >>> raw_x = (x[:, None] ** np.arange(5)).mean(axis=0)

    >>> dx_raw = cmomy.CentralMomentsArray.wrap_raw(raw_x, mom_ndim=1)
    >>> print(dx_raw.mean())
    0.5505105129032412
    >>> dx_raw.cmom()
    array([ 1.    ,  0.    ,  0.1014, -0.0178,  0.02  ])

    Which is equivalent to creating raw moments from values
    >>> dx_cen = cmomy.CentralMomentsArray.wrap_reduce_vals(x, axis=0, mom=4)
    >>> print(dx_cen.mean())
    0.5505105129032413
    >>> dx_cen.cmom()
    array([ 1.    ,  0.    ,  0.1014, -0.0178,  0.02  ])

    But note that calculating using wrap_raw can lead to
    numerical issues.  For example

    >>> y = x + 10000
    >>> raw_y = (y[:, None] ** np.arange(5)).mean(axis=0)
    >>> dy_raw = cmomy.CentralMomentsArray.wrap_raw(raw_y, mom_ndim=1)
    >>> print(dy_raw.mean() - 10000)
    0.5505105129050207

    Note that the central moments don't match!

    >>> np.isclose(dy_raw.cmom(), dx_raw.cmom())
    array([ True,  True,  True, False, False])

    >>> dy_cen = cmomy.CentralMomentsArray.wrap_reduce_vals(y, axis=0, mom=4)
    >>> print(dy_cen.mean() - 10000)
    0.5505105129032017
    >>> dy_cen.cmom()  # this matches above
    array([ 1.    ,  0.    ,  0.1014, -0.0178,  0.02  ])
    """
    from cmomy import convert

    kws = _get_mom_dims_kws(raw, mom_dims, mom_ndim, raw)
    return wrap(  # type: ignore[no-any-return]
        obj=convert.moments_type(
            raw,
            mom_ndim=mom_ndim,
            to="central",
            out=out,
            dtype=dtype,
            casting=casting,
            order=order,
            keep_attrs=keep_attrs,
            on_missing_core_dim=on_missing_core_dim,
            apply_ufunc_kwargs=apply_ufunc_kwargs,
            **kws,
        ),
        mom_ndim=mom_ndim,
        **kws,
        fastpath=True,
    )


# * Utilities -----------------------------------------------------------------
def _get_mom_dims_kws(
    target: ArrayLike | xr.DataArray | xr.Dataset,
    mom_dims: MomDims | None,
    mom_ndim: Mom_NDim,
    out: Any = None,
) -> dict[str, Any]:
    return (
        {"mom_dims": validate_mom_dims(mom_dims, mom_ndim, out)}
        if is_xarray(target)
        else {}
    )

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
from .wrap_xr import CentralMomentsData

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
        CentralMomentsDataT,
        DataT,
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
        ReduceValsKwargs,
        ResampleValsKwargs,
        RngTypes,
        WrapKwargs,
        WrapRawKwargs,
        ZerosLikeKwargs,
    )
    from cmomy.core.typing_compat import Unpack


# * General wrapper -----------------------------------------------------------
@overload
def wrap(  # pyright: ignore[reportOverlappingOverload]
    obj: DataT,
    *,
    dtype: DTypeLike | Mapping[str, DTypeLike] = ...,
    **kwargs: Unpack[WrapKwargs],
) -> CentralMomentsData[DataT]: ...
@overload
def wrap(  # type: ignore[misc]
    obj: xr.DataArray | xr.Dataset,
    *,
    dtype: DTypeLike | Mapping[str, DTypeLike] = ...,
    **kwargs: Unpack[WrapKwargs],
) -> CentralMomentsDataArray | CentralMomentsDataset: ...
@overload
def wrap(
    obj: ArrayLikeArg[FloatT],
    *,
    dtype: None = ...,
    **kwargs: Unpack[WrapKwargs],
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap(
    obj: ArrayLike,
    *,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[WrapKwargs],
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap(
    obj: ArrayLike,
    *,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[WrapKwargs],
) -> CentralMomentsArrayAny: ...


@docfiller.decorate  # type: ignore[misc]
def wrap(  # pyright: ignore[reportInconsistentOverload]
    obj: ArrayLike | DataT,
    *,
    mom_ndim: Mom_NDim = 1,
    mom_dims: MomDims | None = None,
    dtype: DTypeLike | Mapping[str, DTypeLike] = None,
    copy: bool | None = False,
    fastpath: bool = False,
) -> CentralMomentsArray[Any] | CentralMomentsData[DataT]:
    """
    Wrap object with central moments class.

    This will choose the correct wrapper class given the
    type of array (:class:`~numpy.ndarray` or :mod:`xarray` object).

    Parameters
    ----------
    obj : array-like or DataArray or Dataset
        Central Moments array.
    {mom_ndim}
    {mom_dims}
    {dtype}
    {copy}
    {fastpath}

    Returns
    -------
    {wrapped_out}

    See Also
    --------
    CentralMomentsArray
    CentralMomentsData


    Examples
    --------
    >>> data = [10.0, 2.0, 3.0]
    >>> wrap(data)
    <CentralMomentsArray(mom_ndim=1)>
    array([10.,  2.,  3.])


    >>> xdata = xr.DataArray(data, dims="mom")
    >>> wrap(xdata)
    <CentralMomentsData(mom_ndim=1)>
    <xarray.DataArray (mom: 3)> Size: 24B
    array([10.,  2.,  3.])
    Dimensions without coordinates: mom

    """
    if is_xarray(obj):
        if not fastpath:
            copy = copy_if_needed(copy)
            if dtype is not None:
                obj = obj.astype(dtype, copy=copy)  # pyright: ignore[reportUnknownMemberType]
            elif copy:
                obj = obj.copy(deep=True)

        return CentralMomentsData(
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
    c: CentralMomentsDataT,
    *,
    dtype: DTypeLike | Mapping[Any, DTypeLike] = ...,
    **kwargs: Unpack[ZerosLikeKwargs],
) -> CentralMomentsDataT: ...
@overload
def zeros_like(
    c: CentralMomentsArrayT,
    *,
    dtype: None = ...,
    **kwargs: Unpack[ZerosLikeKwargs],
) -> CentralMomentsArrayT: ...
@overload
def zeros_like(
    c: CentralMomentsArrayAny,
    *,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[ZerosLikeKwargs],
) -> CentralMomentsArray[FloatT]: ...
@overload
def zeros_like(
    c: CentralMomentsArrayAny,
    *,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ZerosLikeKwargs],
) -> CentralMomentsArrayAny: ...
@overload
def zeros_like(
    c: CentralMomentsArrayAny | CentralMomentsDataAny,
    *,
    dtype: DTypeLike | Mapping[Any, DTypeLike] = ...,
    **kwargs: Unpack[ZerosLikeKwargs],
) -> CentralMomentsArrayAny | CentralMomentsDataAny: ...


@docfiller.decorate
def zeros_like(
    c: CentralMomentsArrayAny | CentralMomentsDataAny,
    *,
    dtype: DTypeLike | Mapping[Any, DTypeLike] = None,
    order: ArrayOrder = None,
    subok: bool = True,
    chunks: Any = None,
    chunked_array_type: str | None = None,
    from_array_kwargs: dict[str, Any] | None = None,
) -> CentralMomentsArrayAny | CentralMomentsDataAny:
    r"""
    Create new wrapped object with zeros like given wrapped object.

    Parameters
    ----------
    c : CentralMomentsArray or CentralMomentsData
        Wrapped instance to create new object like.
    fill_value : scalar or dict-like
        Value to fill new object with.
    {dtype}
    {order}
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise the
        returned array will be forced to be a base-class array.
    chunks : int, "auto", tuple of int or mapping of Hashable to int, optional
        Chunk sizes along each dimension, e.g., ``5``, ``"auto"``, ``(5, 5)``
        or ``{{"x": 5, "y": 5}}``.
    chunked_array_type: str, optional
        Which chunked array type to coerce the underlying data array to.
        Defaults to 'dask' if installed, else whatever is registered via the
        `ChunkManagerEnetryPoint` system. Experimental API that should not be
        relied upon.
    from_array_kwargs: dict, optional
        Additional keyword arguments passed on to the
        `ChunkManagerEntrypoint.from_array` method used to create chunked
        arrays, via whichever chunk manager is specified through the
        `chunked_array_type` kwarg. For example, with dask as the default
        chunked array type, this method would pass additional kwargs to
        :py:func:`dask.array.from_array`. Experimental API that should not be
        relied upon.

    Returns
    -------
    {wrapped_out}

    See Also
    --------
    numpy.zeros_like
    xarray.zeros_like
    """
    if isinstance(c, CentralMomentsData):
        return wrap(
            xr.zeros_like(  # type: ignore[misc]
                c.obj,
                dtype=dtype,  # type: ignore[arg-type]
                chunks=chunks,
                chunked_array_type=chunked_array_type,
                from_array_kwargs=from_array_kwargs,
            ),
            mom_ndim=c.mom_ndim,
            mom_dims=c.mom_dims,
        )
    return wrap(  # type: ignore[no-any-return]
        np.zeros_like(
            c.obj,
            dtype=dtype,  # type: ignore[arg-type]
            order=order,
            subok=subok,
        ),
        mom_ndim=c.mom_ndim,
    )


# * From vals -----------------------------------------------------------------
@overload
def wrap_reduce_vals(  # pyright: ignore[reportOverlappingOverload]
    x: DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    weight: ArrayLike | xr.DataArray | DataT | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceValsKwargs],
) -> CentralMomentsData[DataT]: ...
@overload
def wrap_reduce_vals(  # type: ignore[misc]
    x: xr.DataArray | xr.Dataset,
    *y: ArrayLike | xr.DataArray | xr.Dataset,
    weight: ArrayLike | xr.DataArray | xr.Dataset | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceValsKwargs],
) -> CentralMomentsDataArray | CentralMomentsDataset: ...
@overload
def wrap_reduce_vals(
    x: ArrayLikeArg[FloatT],
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[ReduceValsKwargs],
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap_reduce_vals(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceValsKwargs],
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap_reduce_vals(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[ReduceValsKwargs],
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap_reduce_vals(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceValsKwargs],
) -> CentralMomentsArrayAny: ...


@docfiller.decorate  # type: ignore[misc]
def wrap_reduce_vals(  # pyright: ignore[reportInconsistentOverload]
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    mom: Moments,
    weight: ArrayLike | xr.DataArray | DataT | None = None,
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
) -> CentralMomentsArrayAny | CentralMomentsData[DataT]:
    """
    Create wrapped object from values.

    Parameters
    ----------
    {x_genarray}
    {y_genarray}
    {mom}
    {weight_genarray}
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
    {wrapped_out}

    See Also
    --------
    ~.reduction.reduce_vals
    wrap

    Examples
    --------
    >>> import cmomy
    >>> rng = cmomy.default_rng(0)
    >>> x = rng.random((100, 3))
    >>> da = cmomy.wrap_reduce_vals(x, axis=0, mom=2)
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

    return wrap(  # pyright: ignore[reportUnknownVariableType]
        obj=obj,  # pyright: ignore[reportUnknownArgumentType]
        mom_ndim=mom_ndim,
        **kws,
        fastpath=True,
    )


# * resample vals -------------------------------------------------------------
@overload
def wrap_resample_vals(  # pyright: ignore[reportOverlappingOverload]
    x: DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    weight: ArrayLike | xr.DataArray | DataT | None = ...,
    freq: ArrayLike | xr.DataArray | DataT | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ResampleValsKwargs],
) -> CentralMomentsData[DataT]: ...
@overload
def wrap_resample_vals(  # type: ignore[misc]
    x: xr.DataArray | xr.Dataset,
    *y: ArrayLike | xr.DataArray | DataT,
    weight: ArrayLike | xr.DataArray | DataT | None = ...,
    freq: ArrayLike | xr.DataArray | DataT | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ResampleValsKwargs],
) -> CentralMomentsDataArray | CentralMomentsDataset: ...
@overload
def wrap_resample_vals(
    x: ArrayLikeArg[FloatT],
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    freq: ArrayLike | None = ...,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[ResampleValsKwargs],
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap_resample_vals(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    freq: ArrayLike | None = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ResampleValsKwargs],
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap_resample_vals(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    freq: ArrayLike | None = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[ResampleValsKwargs],
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap_resample_vals(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    freq: ArrayLike | None = ...,
    out: Any = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ResampleValsKwargs],
) -> CentralMomentsArrayAny: ...


@docfiller.decorate  # type: ignore[misc]
def wrap_resample_vals(  # pyright: ignore[reportInconsistentOverload] # noqa: PLR0913
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    mom: Moments,
    weight: ArrayLike | xr.DataArray | DataT | None = None,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    freq: ArrayLike | xr.DataArray | DataT | None = None,
    nrep: int | None = None,
    rng: RngTypes | None = None,
    paired: bool = True,
    move_axis_to_end: bool = True,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderCF = None,
    parallel: bool | None = None,
    mom_dims: MomDims | None = None,
    rep_dim: str = "rep",
    keep_attrs: KeepAttrs = None,
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> CentralMomentsArrayAny | CentralMomentsData[DataT]:
    """
    Create wrapped object from resampled values.


    Parameters
    ----------
    {x_genarray}
    {y_genarray}
    {mom}
    {weight_genarray}
    {axis}
    {dim}
    {freq}
    {nrep}
    {rng}
    {paired}
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
    {wrapped_out}

    See Also
    --------
    wrap
    ~.resample.resample_vals
    ~.resample.randsamp_freq
    ~.resample.freq_to_indices
    ~.resample.indices_to_freq


    Examples
    --------
    >>> import cmomy
    >>> rng = cmomy.default_rng(0)
    >>> x = rng.random(10)
    >>> cmomy.wrap_resample_vals(x, mom=2, axis=0, nrep=5)
    <CentralMomentsArray(mom_ndim=1)>
    array([[10.    ,  0.5397,  0.0757],
           [10.    ,  0.5848,  0.0618],
           [10.    ,  0.5768,  0.0564],
           [10.    ,  0.6138,  0.1081],
           [10.    ,  0.5808,  0.0685]])
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
        paired=paired,
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
    return wrap(  # pyright: ignore[reportUnknownVariableType]
        obj=obj,  # pyright: ignore[reportUnknownArgumentType]
        mom_ndim=mom_ndim,
        **kws,
        fastpath=True,
    )


# * From raw ------------------------------------------------------------------
@overload
def wrap_raw(  # pyright: ignore[reportOverlappingOverload]
    raw: DataT,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[WrapRawKwargs],
) -> CentralMomentsData[DataT]: ...
@overload
def wrap_raw(  # type: ignore[misc]
    raw: xr.DataArray | xr.Dataset,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[WrapRawKwargs],
) -> CentralMomentsDataArray | CentralMomentsDataset: ...
@overload
def wrap_raw(
    raw: ArrayLikeArg[FloatT],
    *,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[WrapRawKwargs],
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap_raw(
    raw: ArrayLike,
    *,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[WrapRawKwargs],
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap_raw(
    raw: ArrayLike,
    *,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[WrapRawKwargs],
) -> CentralMomentsArray[FloatT]: ...
@overload
def wrap_raw(
    raw: ArrayLike,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[WrapRawKwargs],
) -> CentralMomentsArrayAny: ...


@docfiller.decorate  # type: ignore[misc]
def wrap_raw(  # pyright: ignore[reportInconsistentOverload]
    raw: ArrayLike | DataT,
    *,
    mom_ndim: Mom_NDim = 1,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrder = None,
    keep_attrs: KeepAttrs = None,
    mom_dims: MomDims | None = None,
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> CentralMomentsArrayAny | CentralMomentsData[DataT]:
    """
    Create object from raw moment data.

    raw[..., i, j] = <x**i y**j>.
    raw[..., 0, 0] = `weight`


    Parameters
    ----------
    raw : array-like or DataArray or Dataset
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
    {wrapped_out}

    See Also
    --------
    wrap
    .convert.moments_type

    Notes
    -----
    Weights are taken from ``raw[...,0, 0]``.
    Using raw moments can result in numerical issues, especially for higher moments.  Use with care.

    Examples
    --------
    >>> import cmomy
    >>> rng = cmomy.default_rng(0)
    >>> x = rng.random(10)
    >>> raw_x = (x[:, None] ** np.arange(5)).mean(axis=0)

    >>> dx_raw = cmomy.wrap_raw(raw_x, mom_ndim=1)
    >>> print(dx_raw.mean())
    0.5505105129032412
    >>> dx_raw.cmom()
    array([ 1.    ,  0.    ,  0.1014, -0.0178,  0.02  ])

    Which is equivalent to creating raw moments from values
    >>> dx_cen = cmomy.wrap_reduce_vals(x, axis=0, mom=4)
    >>> print(dx_cen.mean())
    0.5505105129032413
    >>> dx_cen.cmom()
    array([ 1.    ,  0.    ,  0.1014, -0.0178,  0.02  ])

    But note that calculating using wrap_raw can lead to
    numerical issues.  For example

    >>> y = x + 10000
    >>> raw_y = (y[:, None] ** np.arange(5)).mean(axis=0)
    >>> dy_raw = cmomy.wrap_raw(raw_y, mom_ndim=1)
    >>> print(dy_raw.mean() - 10000)
    0.5505105129050207

    Note that the central moments don't match!

    >>> np.isclose(dy_raw.cmom(), dx_raw.cmom())
    array([ True,  True,  True, False, False])

    >>> dy_cen = cmomy.wrap_reduce_vals(y, axis=0, mom=4)
    >>> print(dy_cen.mean() - 10000)
    0.5505105129032017
    >>> dy_cen.cmom()  # this matches above
    array([ 1.    ,  0.    ,  0.1014, -0.0178,  0.02  ])
    """
    from cmomy import convert

    kws = _get_mom_dims_kws(raw, mom_dims, mom_ndim, raw)
    return wrap(  # pyright: ignore[reportUnknownVariableType]
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

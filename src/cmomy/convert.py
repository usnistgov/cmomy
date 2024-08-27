"""
Conversion routines (:mod:`~cmomy.convert`)
===========================================
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast, overload

import numpy as np
import xarray as xr

from ._lib.factory import (
    factory_convert,
    factory_cumulative,
    parallel_heuristic,
)
from .core.array_utils import (
    axes_data_reduction,
    select_dtype,
)
from .core.docstrings import docfiller
from .core.missing import MISSING
from .core.prepare import (
    prepare_data_for_reduction,
    xprepare_out_for_resample_data,
)
from .core.utils import (
    mom_to_mom_shape,
    peek_at,
)
from .core.validate import (
    validate_mom_dims,
    validate_mom_ndim,
)
from .core.xr_utils import (
    get_apply_ufunc_kwargs,
    select_axis_dim,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
    )
    from typing import (
        Any,
    )

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from ._central_dataarray import xCentralMoments
    from ._central_numpy import CentralMoments
    from .core.typing import (
        ApplyUFuncKwargs,
        ArrayLikeArg,
        AxisReduce,
        ConvertStyle,
        DimsReduce,
        DTypeLikeArg,
        FloatT,
        GenXArrayT,
        KeepAttrs,
        MissingCoreDimOptions,
        MissingType,
        Mom_NDim,
        MomDims,
        NDArrayAny,
        ScalarT,
    )


# * Convert between raw and central moments
@overload
def moments_type(
    values_in: GenXArrayT,
    *,
    mom_ndim: Mom_NDim,
    to: ConvertStyle = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> GenXArrayT: ...
# array
@overload
def moments_type(
    values_in: ArrayLikeArg[FloatT],
    *,
    mom_ndim: Mom_NDim,
    to: ConvertStyle = ...,
    out: None = ...,
    dtype: None = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArray[FloatT]: ...
# out
@overload
def moments_type(
    values_in: ArrayLike,
    *,
    mom_ndim: Mom_NDim,
    to: ConvertStyle = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def moments_type(
    values_in: ArrayLike,
    *,
    mom_ndim: Mom_NDim,
    to: ConvertStyle = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArray[FloatT]: ...
# fallback
@overload
def moments_type(
    values_in: ArrayLike,
    *,
    mom_ndim: Mom_NDim,
    to: ConvertStyle = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArrayAny: ...


@docfiller.decorate
def moments_type(
    values_in: ArrayLike | GenXArrayT,
    *,
    mom_ndim: Mom_NDim,
    to: ConvertStyle = "central",
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    keep_attrs: KeepAttrs = None,
    mom_dims: MomDims | None = None,
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | GenXArrayT:
    r"""
    Convert between central and raw moments type.

    Parameters
    ----------
    values_in : array-like, DataArray, or Dataset
        The moments array to convert from.
    {mom_ndim}
    to : {{"raw", "central"}}
        The style of the ``values_in`` to convert to. If ``"raw"``, convert from central to raw.
        If ``"central"`` convert from raw to central moments.
    {out}
    {dtype}
    {move_axis_to_end}
    {keep_attrs}
    {mom_dims_data}
    {on_missing_core_dim}
    {apply_ufunc_kwargs}

    Returns
    -------
    output : ndarray or DataArray or Dataset
        Moments array converted from ``input_style`` to opposite format.

    Notes
    -----
    The structure of arrays are as follow.
    Central moments:

    * ``values_in[..., 0]`` : weight
    * ``values_in[..., 1]`` : mean
    * ``values_in[..., k]`` : kth central moment

    Central comoments of variables `a` and `b`:

    * ``values_in[..., 0, 0]`` : weight
    * ``values_in[..., 1, 0]`` : mean of `a`
    * ``values_in[....,0, 1]`` : mean of `b`
    * ``values_in[..., i, j]`` : :math:`\langle (\delta a)^i (\delta b)^j \rangle`,

    where `a` and `b` are the variables being considered.

    Raw moments array:

    * ``values_in[..., 0]`` : weight
    * ``values_in[..., k]`` : kth moment :math:`\langle a^k \rangle`

    Raw comoments array of variables `a` and `b`:

    * ``values_in[..., 0, 0]`` : weight
    * ``values_in[..., i, j]`` : :math:`\langle a^i b^j \rangle`,

    """
    dtype = select_dtype(values_in, out=out, dtype=dtype)
    mom_ndim = validate_mom_ndim(mom_ndim)
    if isinstance(values_in, (xr.DataArray, xr.Dataset)):
        mom_dims = validate_mom_dims(mom_dims, mom_ndim, values_in)
        xout: GenXArrayT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            moments_type,
            values_in,
            input_core_dims=[mom_dims],
            output_core_dims=[mom_dims],
            kwargs={
                "mom_ndim": mom_ndim,
                "to": to,
                "out": None if isinstance(values_in, xr.Dataset) else out,
                "dtype": dtype,
            },
            keep_attrs=keep_attrs,
            **get_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                on_missing_core_dim=on_missing_core_dim,
                dask="parallelized",
                output_dtypes=dtype or np.float64,
            ),
        )
        return xout

    values_in = np.asarray(values_in, dtype=dtype)
    return factory_convert(mom_ndim=mom_ndim, to=to)(values_in, out=out)


# * Moments to Cumulative moments
@overload
def cumulative(  # pyright: ignore[reportOverlappingOverload]
    values_in: GenXArrayT,
    *,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    mom_ndim: Mom_NDim = ...,
    inverse: bool = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> GenXArrayT: ...
# array
@overload
def cumulative(
    values_in: ArrayLikeArg[FloatT],
    *,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    mom_ndim: Mom_NDim = ...,
    inverse: bool = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    out: None = ...,
    dtype: None = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArray[FloatT]: ...
# out
@overload
def cumulative(
    values_in: ArrayLike,
    *,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    mom_ndim: Mom_NDim = ...,
    inverse: bool = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def cumulative(
    values_in: ArrayLike,
    *,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    mom_ndim: Mom_NDim = ...,
    inverse: bool = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArray[FloatT]: ...
# fallback
@overload
def cumulative(
    values_in: ArrayLike,
    *,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    mom_ndim: Mom_NDim = ...,
    inverse: bool = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    out: NDArrayAny | None = ...,
    dtype: Any = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArrayAny: ...


@docfiller.decorate
def cumulative(  # pyright: ignore[reportOverlappingOverload]
    values_in: ArrayLike | GenXArrayT,
    *,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: Mom_NDim = 1,
    inverse: bool = False,
    move_axis_to_end: bool = False,
    parallel: bool | None = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    keep_attrs: KeepAttrs = None,
    mom_dims: MomDims | None = None,
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | GenXArrayT:
    """
    Convert between moments array and cumulative moments array.

    Parameters
    ----------
    values_in : array-like, DataArray, or Dataset
    {axis}
    {dim}
    {mom_ndim}
    inverse : bool, optional
        Default is to create a cumulative moments array.  Pass ``inverse=True`` to convert from
        cumulative moments array back to normal moments.
    {move_axis_to_end}
    {parallel}
    {out}
    {dtype}
    {keep_attrs}
    {mom_dims_data}
    {on_missing_core_dim}
    {apply_ufunc_kwargs}

    Returns
    -------
    out : ndarray or DataArray or Dataset
        Same type as ``values_in``, with moments accumulated over ``axis``.

    Examples
    --------
    >>> import cmomy
    >>> x = cmomy.random.default_rng(0).random((10, 3))
    >>> data = cmomy.reduce_vals(x, mom=2, axis=0)
    >>> data
    array([[10.    ,  0.5248,  0.1106],
           [10.    ,  0.5688,  0.0689],
           [10.    ,  0.5094,  0.1198]])

    >>> cdata = cumulative(data, axis=0, mom_ndim=1)
    >>> cdata
    array([[10.    ,  0.5248,  0.1106],
           [20.    ,  0.5468,  0.0902],
           [30.    ,  0.5344,  0.1004]])

    To get the original data back, pass ``inverse=True``

    >>> cumulative(cdata, axis=0, mom_ndim=1, inverse=True)
    array([[10.    ,  0.5248,  0.1106],
           [10.    ,  0.5688,  0.0689],
           [10.    ,  0.5094,  0.1198]])


    """
    mom_ndim = validate_mom_ndim(mom_ndim)
    dtype = select_dtype(values_in, out=out, dtype=dtype)
    if isinstance(values_in, (xr.DataArray, xr.Dataset)):
        dtype = select_dtype(values_in, out=out, dtype=dtype)
        axis, dim = select_axis_dim(values_in, axis=axis, dim=dim, mom_ndim=mom_ndim)
        core_dims = [[dim, *validate_mom_dims(mom_dims, mom_ndim, values_in)]]

        xout: GenXArrayT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            cumulative,
            values_in,
            input_core_dims=core_dims,
            output_core_dims=core_dims,
            kwargs={
                "mom_ndim": mom_ndim,
                "inverse": inverse,
                "axis": -1,
                "out": xprepare_out_for_resample_data(
                    out,
                    mom_ndim=mom_ndim,
                    axis=axis,
                    move_axis_to_end=move_axis_to_end,
                    data=values_in,
                ),
                "dtype": dtype,
                "move_axis_to_end": False,
            },
            keep_attrs=keep_attrs,
            **get_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                on_missing_core_dim=on_missing_core_dim,
                dask="parallelized",
                output_dtypes=dtype or np.float64,
            ),
        )

        if not move_axis_to_end and isinstance(xout, xr.DataArray):
            xout = xout.transpose(*values_in.dims)
        return xout

    # Numpy
    axis, values_in = prepare_data_for_reduction(
        values_in,
        axis=axis,
        mom_ndim=mom_ndim,
        dtype=dtype,  # type: ignore[arg-type]
        move_axis_to_end=move_axis_to_end,
    )
    axes = axes_data_reduction(mom_ndim=mom_ndim, axis=axis, out_has_axis=True)

    return factory_cumulative(
        mom_ndim=mom_ndim,
        inverse=inverse,
        parallel=parallel_heuristic(parallel, values_in.size),
    )(values_in, out=out, axes=axes)


# * Moments to  Comoments
def _validate_mom_moments_to_comoments(
    mom: Sequence[int], mom_orig: int
) -> tuple[int, int]:
    if not isinstance(mom, Sequence) or len(mom) != 2:  # pyright: ignore[reportUnnecessaryIsInstance]
        msg = "Must supply length 2 sequence for `mom`."
        raise ValueError(msg)

    if mom[0] < 0:
        mom[1]
        out = (mom_orig - mom[1], mom[1])
    elif mom[1] < 0:
        out = (mom[0], mom_orig - mom[0])
    else:
        out = (mom[0], mom[1])

    if any(m < 1 for m in out) or sum(out) > mom_orig:
        msg = f"{mom=} inconsistent with original moments={mom_orig}"
        raise ValueError(msg)

    return out


@overload
def moments_to_comoments(  # pyright: ignore[reportOverlappingOverload]
    values: GenXArrayT,
    *,
    mom: tuple[int, int],
    dtype: DTypeLike = ...,
    mom_dims: MomDims | None = ...,
    mom_dims2: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> GenXArrayT: ...
# array
@overload
def moments_to_comoments(
    values: ArrayLikeArg[FloatT],
    *,
    mom: tuple[int, int],
    dtype: None = ...,
    mom_dims: MomDims | None = ...,
    mom_dims2: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def moments_to_comoments(
    values: ArrayLike,
    *,
    mom: tuple[int, int],
    dtype: DTypeLikeArg[FloatT],
    mom_dims: MomDims | None = ...,
    mom_dims2: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArray[FloatT]: ...
# fallback
@overload
def moments_to_comoments(
    values: ArrayLike,
    *,
    mom: tuple[int, int],
    dtype: DTypeLike = ...,
    mom_dims: MomDims | None = ...,
    mom_dims2: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
    on_missing_core_dim: MissingCoreDimOptions = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
) -> NDArrayAny: ...


@docfiller.decorate
def moments_to_comoments(  # pyright: ignore[reportOverlappingOverload]
    values: ArrayLike | GenXArrayT,
    *,
    mom: tuple[int, int],
    dtype: DTypeLike = None,
    mom_dims: MomDims | None = None,
    mom_dims2: MomDims | None = None,
    keep_attrs: KeepAttrs = None,
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | GenXArrayT:
    """
    Convert from moments to comoments data.

    Parameters
    ----------
    values : array-like or DataArray
        Moments array with ``mom_ndim==1``. It is assumed that the last
        dimension is the moments dimension.
    {mom_moments_to_comoments}
    {dtype}
    mom_dims : str or tuple of str
        Optional name of moment dimension of input (``mom_ndim=1``) data.  Defaults to
        ``first.dims[-mom_ndim]`` where ``first`` is either ``values`` if a DataArray
        or the first variable of ``values`` if a Dataset.  You may need to pass this value
        if ``values`` is a Dataset.
    mom_dims2 : tuple of str
        Moments dimensions for output (``mom_ndim=2``) data.  Defaults to ``("mom_0", "mom_1")``.
    {keep_attrs}
    {on_missing_core_dim}
    {apply_ufunc_kwargs}

    Returns
    -------
    output : ndarray or DataArray
        Co-moments array.  Same type as ``values``.


    Examples
    --------
    >>> import cmomy
    >>> x = cmomy.random.default_rng(0).random(10)
    >>> data1 = cmomy.reduce_vals(x, mom=4, axis=0)
    >>> data1
    array([10.    ,  0.5505,  0.1014, -0.0178,  0.02  ])

    >>> moments_to_comoments(data1, mom=(2, -1))
    array([[10.    ,  0.5505,  0.1014],
           [ 0.5505,  0.1014, -0.0178],
           [ 0.1014, -0.0178,  0.02  ]])


    Which is identical to

    >>> cmomy.reduction.reduce_vals(x, x, mom=(2, 2), axis=0)
    array([[10.    ,  0.5505,  0.1014],
           [ 0.5505,  0.1014, -0.0178],
           [ 0.1014, -0.0178,  0.02  ]])


    This also works for :class:`~xarray.DataArray` data

    >>> xdata = xr.DataArray(data1, dims="mom")
    >>> xdata
    <xarray.DataArray (mom: 5)> Size: 40B
    array([10.    ,  0.5505,  0.1014, -0.0178,  0.02  ])
    Dimensions without coordinates: mom


    >>> moments_to_comoments(xdata, mom=(2, -1))
    <xarray.DataArray (mom_0: 3, mom_1: 3)> Size: 72B
    array([[10.    ,  0.5505,  0.1014],
           [ 0.5505,  0.1014, -0.0178],
           [ 0.1014, -0.0178,  0.02  ]])
    Dimensions without coordinates: mom_0, mom_1


    Note that this also works for raw moments.

    """
    dtype = select_dtype(values, out=None, dtype=dtype)
    if isinstance(values, (xr.DataArray, xr.Dataset)):
        mom_dim_in, *_ = validate_mom_dims(mom_dims, mom_ndim=1, out=values)
        mom_dims2 = validate_mom_dims(mom_dims2, mom_ndim=2)

        if mom_dim_in in mom_dims2:
            # give this a temporary name for simplicity:
            old_name, mom_dim_in = mom_dim_in, f"_tmp_{mom_dim_in}"
            values = values.rename({old_name: mom_dim_in})

        xout: GenXArrayT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            moments_to_comoments,
            values,
            input_core_dims=[[mom_dim_in]],
            output_core_dims=[mom_dims2],
            kwargs={"mom": mom, "dtype": dtype},
            keep_attrs=keep_attrs,
            **get_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                on_missing_core_dim=on_missing_core_dim,
                dask="parallelized",
                output_sizes=dict(
                    zip(
                        mom_dims2,
                        mom_to_mom_shape(
                            _validate_mom_moments_to_comoments(
                                mom, values.sizes[mom_dim_in] - 1
                            )
                        ),
                    )
                ),
                output_dtypes=dtype or np.float64,
            ),
        )

        return xout

    # numpy
    values = np.asarray(values, dtype=dtype)
    mom = _validate_mom_moments_to_comoments(mom, values.shape[-1] - 1)
    out = np.empty((*values.shape[:-1], *mom_to_mom_shape(mom)), dtype=dtype)  # type: ignore[union-attr]
    for i, j in np.ndindex(*out.shape[-2:]):
        out[..., i, j] = values[..., i + j]
    return out


# * concat
@overload
def concat(
    arrays: Iterable[GenXArrayT],
    *,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    **kwargs: Any,
) -> GenXArrayT: ...
@overload
def concat(
    arrays: Iterable[CentralMoments[FloatT]],
    *,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    **kwargs: Any,
) -> CentralMoments[FloatT]: ...
@overload
def concat(
    arrays: Iterable[xCentralMoments[FloatT]],
    *,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    **kwargs: Any,
) -> xCentralMoments[FloatT]: ...
@overload
def concat(
    arrays: Iterable[NDArray[ScalarT]],
    *,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    **kwargs: Any,
) -> NDArray[ScalarT]: ...


@docfiller.decorate
def concat(
    arrays: Iterable[GenXArrayT]
    | Iterable[CentralMoments[Any]]
    | Iterable[xCentralMoments[Any]]
    | Iterable[NDArrayAny],
    *,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    **kwargs: Any,
) -> GenXArrayT | CentralMoments[Any] | xCentralMoments[Any] | NDArrayAny:
    """
    Concatenate moments objects.

    Parameters
    ----------
    arrays : Iterable of ndarray or DataArray or CentralMoments or xCentralMoments
        Central moments objects to combine.
    axis : int, optional
        Axis to concatenate along. If specify axis for
        :class:`~xarray.DataArray` or :class:`~.xCentralMoments` input objects
        with out ``dim``, then determine ``dim`` from ``dim =
        first.dims[axis]`` where ``first`` is the first item in ``arrays``.
    dim : str, optional
        Dimension to concatenate along (used for :class:`~xarray.DataArray` and
        :class:`~.xCentralMoments` objects only)
    **kwargs
        Extra arguments to :func:`numpy.concatenate` or :func:`xarray.concat`.

    Returns
    -------
    output : ndarray or DataArray or CentralMoments or xCentralMoments
        Concatenated object.  Type is the same as the elements of ``arrays``.

    Examples
    --------
    >>> import cmomy
    >>> shape = (2, 1, 2)
    >>> x = np.arange(np.prod(shape)).reshape(shape).astype(np.float64)
    >>> y = -x
    >>> out = concat((x, y), axis=1)
    >>> out.shape
    (2, 2, 2)
    >>> out
    array([[[ 0.,  1.],
            [-0., -1.]],
    <BLANKLINE>
           [[ 2.,  3.],
            [-2., -3.]]])

    >>> dx = xr.DataArray(x, dims=["a", "b", "mom"])
    >>> dy = xr.DataArray(y, dims=["a", "b", "mom"])
    >>> concat((dx, dy), dim="b")
    <xarray.DataArray (a: 2, b: 2, mom: 2)> Size: 64B
    array([[[ 0.,  1.],
            [-0., -1.]],
    <BLANKLINE>
           [[ 2.,  3.],
            [-2., -3.]]])
    Dimensions without coordinates: a, b, mom

    For :class:`~xarray.DataArray` objects, you can specify a new dimension

    >>> concat((dx, dy), dim="new")
    <xarray.DataArray (new: 2, a: 2, b: 1, mom: 2)> Size: 64B
    array([[[[ 0.,  1.]],
    <BLANKLINE>
            [[ 2.,  3.]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[-0., -1.]],
    <BLANKLINE>
            [[-2., -3.]]]])
    Dimensions without coordinates: new, a, b, mom


    You can also concatenate :class:`~.CentralMoments` and :class:`~.xCentralMoments` objects

    >>> cx = cmomy.CentralMoments(x)
    >>> cy = cmomy.CentralMoments(y)
    >>> concat((cx, cy), axis=1)
    <CentralMoments(val_shape=(2, 2), mom=(1,))>
    array([[[ 0.,  1.],
            [-0., -1.]],
    <BLANKLINE>
           [[ 2.,  3.],
            [-2., -3.]]])

    >>> dcx = cmomy.xCentralMoments(dx)
    >>> dcy = cmomy.xCentralMoments(dy)
    >>> concat((dcx, dcy), dim="new")
    <xCentralMoments(val_shape=(2, 2, 1), mom=(1,))>
    <xarray.DataArray (new: 2, a: 2, b: 1, mom: 2)> Size: 64B
    array([[[[ 0.,  1.]],
    <BLANKLINE>
            [[ 2.,  3.]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[-0., -1.]],
    <BLANKLINE>
            [[-2., -3.]]]])
    Dimensions without coordinates: new, a, b, mom



    """
    first, arrays_iter = peek_at(arrays)

    if isinstance(first, np.ndarray):
        axis = 0 if axis is MISSING else axis
        return np.concatenate(
            tuple(arrays_iter),  # type: ignore[arg-type]
            axis=axis,
            dtype=first.dtype,
            **kwargs,
        )

    if isinstance(first, (xr.DataArray, xr.Dataset)):
        if dim is MISSING or dim is None or dim in first.dims:
            axis, dim = select_axis_dim(first, axis=axis, dim=dim, default_axis=0)
        # otherwise, assume adding a new dimension...
        return cast("GenXArrayT", xr.concat(tuple(arrays_iter), dim=dim, **kwargs))  # type: ignore[type-var]

    return type(first)(  # type: ignore[call-arg, return-value]
        concat(
            (c.to_values() for c in arrays_iter),  # type: ignore[attr-defined]
            axis=axis,
            dim=dim,
            **kwargs,
        ),
        mom_ndim=first.mom_ndim,  # type: ignore[attr-defined]
    )

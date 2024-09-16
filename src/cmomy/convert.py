"""
Conversion routines (:mod:`~cmomy.convert`)
===========================================
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast, overload

import numpy as np
import xarray as xr

from .core.array_utils import (
    asarray_maybe_recast,
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
    is_dataarray,
    is_dataset,
    is_ndarray,
    is_xarray,
    validate_mom_dims,
    validate_mom_ndim,
)
from .core.xr_utils import (
    get_apply_ufunc_kwargs,
    select_axis_dim,
)
from .factory import (
    factory_convert,
    factory_cumulative,
    parallel_heuristic,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
    )
    from typing import (
        Any,
    )

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from .core.typing import (
        ApplyUFuncKwargs,
        ArrayLikeArg,
        ArrayOrder,
        ArrayOrderCF,
        AxisReduce,
        Casting,
        CentralMomentsT,
        ConvertStyle,
        CumulativeKwargs,
        DataT,
        DimsReduce,
        DTypeLikeArg,
        FloatT,
        KeepAttrs,
        MissingCoreDimOptions,
        MissingType,
        Mom_NDim,
        MomDims,
        MomentsToComomentsKwargs,
        MomentsTypeKwargs,
        NDArrayAny,
        NDArrayT,
    )
    from .core.typing_compat import Unpack


# * Convert between raw and central moments
@overload
def moments_type(
    values_in: DataT,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[MomentsTypeKwargs],
) -> DataT: ...
# array
@overload
def moments_type(
    values_in: ArrayLikeArg[FloatT],
    *,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[MomentsTypeKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def moments_type(
    values_in: ArrayLike,
    *,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[MomentsTypeKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def moments_type(
    values_in: ArrayLike,
    *,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[MomentsTypeKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def moments_type(
    values_in: ArrayLike,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[MomentsTypeKwargs],
) -> NDArrayAny: ...


@docfiller.decorate
def moments_type(
    values_in: ArrayLike | DataT,
    *,
    mom_ndim: Mom_NDim = 1,
    to: ConvertStyle = "central",
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrder = None,
    keep_attrs: KeepAttrs = None,
    mom_dims: MomDims | None = None,
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
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
    {casting}
    {order}
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
    if is_xarray(values_in):
        mom_dims = validate_mom_dims(mom_dims, mom_ndim, values_in)
        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _moments_type,
            values_in,
            input_core_dims=[mom_dims],
            output_core_dims=[mom_dims],
            kwargs={
                "mom_ndim": mom_ndim,
                "to": to,
                "out": None if is_dataset(values_in) else out,
                "dtype": dtype,
                "casting": casting,
                "order": order,
                "fastpath": is_dataarray(values_in),
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

    return _moments_type(
        values_in,
        out=out,
        mom_ndim=mom_ndim,
        to=to,
        dtype=dtype,
        casting=casting,
        order=order,
        fastpath=True,
    )


def _moments_type(
    values_in: ArrayLike,
    out: NDArrayAny | None,
    mom_ndim: Mom_NDim,
    to: ConvertStyle,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrder,
    fastpath: bool,
) -> NDArrayAny:
    if not fastpath:
        dtype = select_dtype(values_in, out=out, dtype=dtype)

    return factory_convert(mom_ndim=mom_ndim, to=to)(
        values_in,  # type: ignore[arg-type]
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
    )


# * Moments to Cumulative moments
@overload
def cumulative(
    values_in: DataT,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[CumulativeKwargs],
) -> DataT: ...
# array
@overload
def cumulative(
    values_in: ArrayLikeArg[FloatT],
    *,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[CumulativeKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def cumulative(
    values_in: ArrayLike,
    *,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[CumulativeKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def cumulative(
    values_in: ArrayLike,
    *,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[CumulativeKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def cumulative(
    values_in: ArrayLike,
    *,
    out: NDArrayAny | None = ...,
    dtype: Any = ...,
    **kwargs: Unpack[CumulativeKwargs],
) -> NDArrayAny: ...


@docfiller.decorate
def cumulative(
    values_in: ArrayLike | DataT,
    *,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: Mom_NDim = 1,
    inverse: bool = False,
    move_axis_to_end: bool = False,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrder = None,
    parallel: bool | None = None,
    keep_attrs: KeepAttrs = None,
    mom_dims: MomDims | None = None,
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
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
    {out}
    {dtype}
    {casting}
    {order}
    {parallel}
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
    >>> x = cmomy.default_rng(0).random((10, 3))
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
    if is_xarray(values_in):
        axis, dim = select_axis_dim(values_in, axis=axis, dim=dim, mom_ndim=mom_ndim)
        core_dims = [[dim, *validate_mom_dims(mom_dims, mom_ndim, values_in)]]

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _cumulative,
            values_in,
            input_core_dims=core_dims,
            output_core_dims=core_dims,
            kwargs={
                "mom_ndim": mom_ndim,
                "inverse": inverse,
                "axis": -(mom_ndim + 1),
                "out": xprepare_out_for_resample_data(
                    out,
                    mom_ndim=mom_ndim,
                    axis=axis,
                    move_axis_to_end=move_axis_to_end,
                    data=values_in,
                ),
                "parallel": parallel,
                "dtype": dtype,
                "casting": casting,
                "order": order,
                "fastpath": is_dataarray(values_in),
            },
            keep_attrs=keep_attrs,
            **get_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                on_missing_core_dim=on_missing_core_dim,
                dask="parallelized",
                output_dtypes=dtype or np.float64,
            ),
        )

        if not move_axis_to_end and is_dataarray(xout):
            xout = xout.transpose(*values_in.dims)
        return xout

    # Numpy
    axis, values_in = prepare_data_for_reduction(
        values_in,
        axis=axis,
        mom_ndim=mom_ndim,
        dtype=None,
        recast=False,
        move_axis_to_end=move_axis_to_end,
    )
    return _cumulative(
        values_in,
        out=out,
        axis=axis,
        mom_ndim=mom_ndim,
        inverse=inverse,
        parallel=parallel,
        dtype=dtype,
        casting=casting,
        order=order,
        fastpath=True,
    )


def _cumulative(
    values_in: NDArrayAny,
    out: NDArrayAny | None,
    axis: int,
    mom_ndim: Mom_NDim,
    inverse: bool,
    parallel: bool | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrder,
    fastpath: bool,
) -> NDArrayAny:
    if not fastpath:
        dtype = select_dtype(values_in, out=out, dtype=dtype)

    axes = axes_data_reduction(mom_ndim=mom_ndim, axis=axis, out_has_axis=True)
    return factory_cumulative(
        mom_ndim=mom_ndim,
        inverse=inverse,
        parallel=parallel_heuristic(parallel, values_in.size),
    )(
        values_in,
        out=out,
        axes=axes,
        dtype=dtype,
        casting=casting,
        order=order,
    )


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
def moments_to_comoments(
    values: DataT,
    *,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[MomentsToComomentsKwargs],
) -> DataT: ...
# array
@overload
def moments_to_comoments(
    values: ArrayLikeArg[FloatT],
    *,
    dtype: None = ...,
    **kwargs: Unpack[MomentsToComomentsKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def moments_to_comoments(
    values: ArrayLike,
    *,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[MomentsToComomentsKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def moments_to_comoments(
    values: ArrayLike,
    *,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[MomentsToComomentsKwargs],
) -> NDArrayAny: ...


@docfiller.decorate
def moments_to_comoments(
    values: ArrayLike | DataT,
    *,
    mom: tuple[int, int],
    dtype: DTypeLike = None,
    order: ArrayOrderCF = None,
    mom_dims: MomDims | None = None,
    mom_dims2: MomDims | None = None,
    keep_attrs: KeepAttrs = None,
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Convert from moments to comoments data.

    Parameters
    ----------
    values : array-like or DataArray
        Moments array with ``mom_ndim==1``. It is assumed that the last
        dimension is the moments dimension.
    {mom_moments_to_comoments}
    {dtype}
    {order_cf}
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
    >>> x = cmomy.default_rng(0).random(10)
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
    if is_xarray(values):
        mom_dim_in, *_ = validate_mom_dims(mom_dims, mom_ndim=1, out=values)
        mom_dims2 = validate_mom_dims(mom_dims2, mom_ndim=2)

        if mom_dim_in in mom_dims2:
            # give this a temporary name for simplicity:
            old_name, mom_dim_in = mom_dim_in, f"_tmp_{mom_dim_in}"
            values = values.rename({old_name: mom_dim_in})

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
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
    values = asarray_maybe_recast(values, dtype=dtype, recast=False)
    mom = _validate_mom_moments_to_comoments(mom, values.shape[-1] - 1)
    out = np.empty(
        (*values.shape[:-1], *mom_to_mom_shape(mom)),  # type: ignore[union-attr]
        dtype=dtype,
        order=order,
    )
    for i, j in np.ndindex(*out.shape[-2:]):
        out[..., i, j] = values[..., i + j]
    return out


# * concat
@overload
def concat(
    arrays: Iterable[CentralMomentsT],
    *,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    **kwargs: Any,
) -> CentralMomentsT: ...
@overload
def concat(
    arrays: Iterable[DataT],
    *,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    **kwargs: Any,
) -> DataT: ...
@overload
def concat(
    arrays: Iterable[NDArrayT],
    *,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    **kwargs: Any,
) -> NDArrayT: ...


@docfiller.decorate  # type: ignore[arg-type]
def concat(
    arrays: Iterable[NDArrayT] | Iterable[DataT] | Iterable[CentralMomentsT],
    *,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    **kwargs: Any,
) -> NDArrayT | DataT | CentralMomentsT:
    """
    Concatenate moments objects.

    Parameters
    ----------
    arrays : Iterable of ndarray or DataArray or CentralMomentsArray or CentralMomentsData
        Central moments objects to combine.
    axis : int, optional
        Axis to concatenate along. If specify axis for
        :class:`~xarray.DataArray` or :class:`~.CentralMomentsData` input objects
        with out ``dim``, then determine ``dim`` from ``dim =
        first.dims[axis]`` where ``first`` is the first item in ``arrays``.
    dim : str, optional
        Dimension to concatenate along (used for :class:`~xarray.DataArray` and
        :class:`~.CentralMomentsData` objects only)
    **kwargs
        Extra arguments to :func:`numpy.concatenate` or :func:`xarray.concat`.

    Returns
    -------
    output : ndarray or DataArray or CentralMomentsArray or CentralMomentsData
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


    You can also concatenate :class:`~.CentralMomentsArray` and :class:`~.CentralMomentsData` objects

    >>> cx = cmomy.CentralMomentsArray(x)
    >>> cy = cmomy.CentralMomentsArray(y)
    >>> concat((cx, cy), axis=1)
    <CentralMomentsArray(mom_ndim=1)>
    array([[[ 0.,  1.],
            [-0., -1.]],
    <BLANKLINE>
           [[ 2.,  3.],
            [-2., -3.]]])

    >>> dcx = cmomy.CentralMomentsData(dx)
    >>> dcy = cmomy.CentralMomentsData(dy)
    >>> concat((dcx, dcy), dim="new")
    <CentralMomentsData(mom_ndim=1)>
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

    if is_ndarray(first):
        axis = 0 if axis is MISSING else axis
        return np.concatenate(  # type: ignore[return-value]
            tuple(arrays_iter),  # type: ignore[arg-type]
            axis=axis,
            dtype=first.dtype,
            **kwargs,
        )

    if is_xarray(first):
        if dim is MISSING or dim is None or dim in first.dims:
            axis, dim = select_axis_dim(first, axis=axis, dim=dim, default_axis=0)
        # otherwise, assume adding a new dimension...
        return cast("DataT", xr.concat(tuple(arrays_iter), dim=dim, **kwargs))  # type: ignore[type-var]

    return type(first)(  # type: ignore[call-arg, return-value]
        concat(
            (c.obj for c in arrays_iter),  # type: ignore[attr-defined]
            axis=axis,
            dim=dim,
            **kwargs,
        ),
        mom_ndim=first.mom_ndim,  # type: ignore[attr-defined]
    )

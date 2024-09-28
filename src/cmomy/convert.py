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
    validate_mom_dims_and_mom_ndim,
    validate_mom_ndim,
)
from .core.xr_utils import (
    factory_apply_ufunc_kwargs,
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


@docfiller.decorate  # type: ignore[arg-type, unused-ignore]
def moments_type(
    values_in: ArrayLike | DataT,
    *,
    mom_ndim: Mom_NDim | None = None,
    to: ConvertStyle = "central",
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrder = None,
    keep_attrs: KeepAttrs = None,
    mom_dims: MomDims | None = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    r"""
    Convert between central and raw moments type.

    Parameters
    ----------
    values_in : array-like, DataArray, or Dataset
        The moments array to convert from.
    {mom_ndim_data}
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
    if is_xarray(values_in):
        mom_dims, mom_ndim = validate_mom_dims_and_mom_ndim(
            mom_dims, mom_ndim, values_in, mom_ndim_default=1
        )
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
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_dtypes=dtype or np.float64,
            ),
        )
        return xout

    return _moments_type(
        values_in,
        out=out,
        mom_ndim=validate_mom_ndim(mom_ndim, mom_ndim_default=1),
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


@docfiller.decorate  # type: ignore[arg-type,unused-ignore]
def cumulative(
    values_in: ArrayLike | DataT,
    *,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: Mom_NDim | None = None,
    inverse: bool = False,
    move_axis_to_end: bool = False,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrder = None,
    parallel: bool | None = None,
    keep_attrs: KeepAttrs = None,
    mom_dims: MomDims | None = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Convert between moments array and cumulative moments array.

    Parameters
    ----------
    values_in : array-like, DataArray, or Dataset
    {axis}
    {dim}
    {mom_ndim_data}
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
    dtype = select_dtype(values_in, out=out, dtype=dtype)
    if is_xarray(values_in):
        mom_dims, mom_ndim = validate_mom_dims_and_mom_ndim(
            mom_dims, mom_ndim, values_in, mom_ndim_default=1
        )
        axis, dim = select_axis_dim(values_in, axis=axis, dim=dim, mom_dims=mom_dims)
        core_dims = [[dim, *mom_dims]]  # type: ignore[misc]

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
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_dtypes=dtype or np.float64,
            ),
        )

        if not move_axis_to_end and is_dataarray(xout):
            xout = xout.transpose(*values_in.dims)
        return xout

    # Numpy
    mom_ndim = validate_mom_ndim(mom_ndim, mom_ndim_default=1)
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
    data: DataT,
    *,
    mom: tuple[int, int],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[MomentsToComomentsKwargs],
) -> DataT: ...
# array
@overload
def moments_to_comoments(
    data: ArrayLikeArg[FloatT],
    *,
    mom: tuple[int, int],
    dtype: None = ...,
    **kwargs: Unpack[MomentsToComomentsKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def moments_to_comoments(
    data: ArrayLike,
    *,
    mom: tuple[int, int],
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[MomentsToComomentsKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def moments_to_comoments(
    data: ArrayLike,
    *,
    mom: tuple[int, int],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[MomentsToComomentsKwargs],
) -> NDArrayAny: ...


@docfiller.decorate  # type: ignore[arg-type,unused-ignore]
def moments_to_comoments(
    data: ArrayLike | DataT,
    *,
    mom: tuple[int, int],
    dtype: DTypeLike = None,
    order: ArrayOrderCF = None,
    mom_dims: MomDims | None = None,
    mom_dims_out: MomDims | None = None,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Convert from moments to comoments data.

    Parameters
    ----------
    data : array-like or DataArray or Dataset
        Moments array with ``mom_ndim==1``. It is assumed that the last
        dimension is the moments dimension.
    {mom_moments_to_comoments}
    {dtype}
    {order_cf}
    {mom_dims_data}
    mom_dims_out : tuple of str
        Moments dimensions for output (``mom_ndim=2``) data.  Defaults to ``("mom_0", "mom_1")``.
    {keep_attrs}
    {apply_ufunc_kwargs}

    Returns
    -------
    output : ndarray or DataArray
        Co-moments array.  Same type as ``data``.


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
    dtype = select_dtype(data, out=None, dtype=dtype)
    if is_xarray(data):
        mom_dim_in, *_ = validate_mom_dims(mom_dims, mom_ndim=1, out=data)
        mom_dims_out = validate_mom_dims(mom_dims_out, mom_ndim=2)

        if mom_dim_in in mom_dims_out:
            # give this a temporary name for simplicity:
            old_name, mom_dim_in = mom_dim_in, f"_tmp_{mom_dim_in}"
            data = data.rename({old_name: mom_dim_in})

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            moments_to_comoments,
            data,
            input_core_dims=[[mom_dim_in]],
            output_core_dims=[mom_dims_out],
            kwargs={"mom": mom, "dtype": dtype},
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_sizes=dict(
                    zip(
                        mom_dims_out,
                        mom_to_mom_shape(
                            _validate_mom_moments_to_comoments(
                                mom, data.sizes[mom_dim_in] - 1
                            )
                        ),
                    )
                ),
                output_dtypes=dtype or np.float64,
            ),
        )

        return xout

    # numpy
    data = asarray_maybe_recast(data, dtype=dtype, recast=False)
    mom = _validate_mom_moments_to_comoments(mom, data.shape[-1] - 1)
    out = np.empty(
        (*data.shape[:-1], *mom_to_mom_shape(mom)),  # type: ignore[union-attr]
        dtype=dtype,
        order=order,
    )
    for i, j in np.ndindex(*out.shape[-2:]):
        out[..., i, j] = data[..., i + j]
    return out


@overload
def comoments_to_moments(
    data: DataT,
    *,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[MomentsToComomentsKwargs],
) -> DataT: ...
# array
@overload
def comoments_to_moments(
    data: ArrayLikeArg[FloatT],
    *,
    dtype: None = ...,
    **kwargs: Unpack[MomentsToComomentsKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def comoments_to_moments(
    data: ArrayLike,
    *,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[MomentsToComomentsKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def comoments_to_moments(
    data: ArrayLike,
    *,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[MomentsToComomentsKwargs],
) -> NDArrayAny: ...


@docfiller.decorate  # type: ignore[arg-type,unused-ignore]
def comoments_to_moments(
    data: ArrayLike | DataT,
    *,
    dtype: DTypeLike = None,
    order: ArrayOrderCF = None,
    mom_dims: MomDims | None = None,
    mom_dims_out: MomDims | None = None,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Convert comoments of same variable to moments array

    Inverse of :func:`moments_to_comoments`.  Use with caution.
    This is intended only to use for symmetric comoment arrays (i.e.,
    one created from :func:`comoments_to_moments`).


    Parameters
    ----------
    data : array-like or DataArray or Dataset
        Moments array with ``mom_ndim==1``. It is assumed that the last
        dimension is the moments dimension.
    {dtype}
    {order_cf}
    {mom_dims_data}
    mom_dims_out : tuple of str
        Moments dimensions for output (``mom_ndim=1``) data.  Defaults to ``("mom_0",)``.
    {keep_attrs}
    {apply_ufunc_kwargs}

    Returns
    -------
    output : ndarray or DataArray
        Co-moments array.  Same type as ``data``.


    Examples
    --------
    >>> import cmomy
    >>> x = cmomy.default_rng(0).random(10)
    >>> data2 = cmomy.reduce_vals(x, x, mom=(1, 2), axis=0)
    >>> data2
    array([[10.    ,  0.5505,  0.1014],
           [ 0.5505,  0.1014, -0.0178]])
    >>> comoments_to_moments(data2)
    array([10.    ,  0.5505,  0.1014, -0.0178])


    Note that this is identical to the following:

    >>> cmomy.reduce_vals(x, mom=3, axis=0)
    array([10.    ,  0.5505,  0.1014, -0.0178])
    """
    dtype = select_dtype(data, out=None, dtype=dtype)
    if is_xarray(data):
        mom_dims = validate_mom_dims(mom_dims, mom_ndim=2, out=data)
        mom_dim_out, *_ = validate_mom_dims(mom_dims_out, mom_ndim=1)
        if mom_dim_out in mom_dims:
            # give this a temporary name for simplicity:
            new_name = f"_tmp_{mom_dim_out}"
            data = data.rename({mom_dim_out: new_name})
            mom_dims = tuple(new_name if d == mom_dim_out else d for d in mom_dims)

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            comoments_to_moments,
            data,
            input_core_dims=[mom_dims],
            output_core_dims=[[mom_dim_out]],
            kwargs={"dtype": dtype},
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_sizes={mom_dim_out: sum(data.sizes[k] for k in mom_dims) - 1},
                output_dtypes=dtype or np.float64,
            ),
        )
        return xout

    data = asarray_maybe_recast(data, dtype, recast=False)

    new_mom_len = sum(data.shape[-2:]) - 1
    val_shape: tuple[int, ...] = data.shape[:-2]
    out = np.empty((*val_shape, new_mom_len), dtype=dtype, order=order)

    sampled = [False] * (new_mom_len + 1)
    for i, j in np.ndindex(*data.shape[-2:]):
        k = i + j
        if not sampled[k]:
            out[..., k] = data[..., i, j]
            sampled[k] = True

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

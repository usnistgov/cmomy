"""
Conversion routines (:mod:`~cmomy.convert`)
===========================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr

from .core.array_utils import (
    arrayorder_to_arrayorder_cf,
    asarray_maybe_recast,
    reorder,
    select_dtype,
)
from .core.docstrings import docfiller
from .core.missing import MISSING
from .core.moment_params import (
    MomParamsArray,
    MomParamsXArray,
)
from .core.prepare import (
    PrepareDataArray,
    PrepareDataXArray,
)
from .core.utils import (
    mom_to_mom_shape,
)
from .core.validate import (
    is_dataarray,
    is_dataset,
    is_xarray_typevar,
)
from .core.xr_utils import (
    factory_apply_ufunc_kwargs,
    transpose_like,
)
from .factory import (
    factory_convert,
    factory_cumulative,
    parallel_heuristic,
)

if TYPE_CHECKING:
    from collections.abc import (
        Sequence,
    )
    from typing import (
        Any,
    )

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from .core._typing_kwargs import (
        ApplyUFuncKwargs,
        CumulativeKwargs,
        MomentsToComomentsKwargs,
        MomentsTypeKwargs,
    )
    from .core.moment_params import MomParamsType
    from .core.typing import (
        ArrayLikeArg,
        ArrayOrderCF,
        ArrayOrderKACF,
        AxisReduceWrap,
        Casting,
        ConvertStyle,
        DataT,
        DimsReduce,
        DTypeLikeArg,
        FloatT,
        KeepAttrs,
        MissingType,
        MomAxes,
        MomDims,
        MomNDim,
        NDArrayAny,
    )
    from .core.typing_compat import TypeVar, Unpack

    _NDArrayT = TypeVar("_NDArrayT", bound="NDArray[Any]")


# * Convert between raw and central moments
@overload
def moments_type(  # pyright: ignore[reportOverlappingOverload]
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
# arraylike or DataT
@overload
def moments_type(
    values_in: ArrayLike | DataT,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[MomentsTypeKwargs],
) -> NDArrayAny | DataT: ...


@docfiller.decorate  # type: ignore[arg-type, unused-ignore]
def moments_type(
    values_in: ArrayLike | DataT,
    *,
    mom_ndim: MomNDim | None = None,
    mom_axes: MomAxes | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsType = None,
    to: ConvertStyle = "central",
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderKACF = None,
    axes_to_end: bool = False,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    r"""
    Convert between central and raw moments type.

    Parameters
    ----------
    values_in : array-like, DataArray, or Dataset
        The moments array to convert from.
    {mom_ndim_data}
    {mom_axes}
    {mom_dims_data}
    {mom_params}
    to : {{"raw", "central"}}
        The style of the ``values_in`` to convert to. If ``"raw"``, convert from central to raw.
        If ``"central"`` convert from raw to central moments.
    {out}
    {dtype}
    {casting}
    {order}
    {axes_to_end}
    {keep_attrs}
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
    # TODO(wpk): add axes_to_end like parameter...
    dtype = select_dtype(values_in, out=out, dtype=dtype)
    if is_xarray_typevar["DataT"].check(values_in):
        prep = PrepareDataXArray.factory(
            mom_params=mom_params,
            ndim=mom_ndim,
            dims=mom_dims,
            axes=mom_axes,
            data=values_in,
            default_ndim=1,
        )

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _moments_type,
            values_in,
            input_core_dims=[prep.mom_params.dims],
            output_core_dims=[prep.mom_params.dims],
            kwargs={
                "mom_params": prep.mom_params.to_array(),
                "to": to,
                "out": prep.optional_out_transform(
                    target=values_in,
                    out=out,
                    axes_to_end=axes_to_end,
                    order=order,
                    dtype=dtype,
                ),
                "dtype": dtype,
                "casting": casting,
                "order": order,
                "fastpath": is_dataarray(values_in),
                "axes_to_end": False,
            },
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_dtypes=dtype if dtype is not None else np.float64,  # type: ignore[redundant-expr]
            ),
        )
        if not axes_to_end:
            xout = transpose_like(xout, template=values_in)
        return xout

    return _moments_type(
        values_in,
        out=out,
        mom_params=MomParamsArray.factory(
            mom_params=mom_params, ndim=mom_ndim, axes=mom_axes, default_ndim=1
        ),
        to=to,
        dtype=dtype,
        casting=casting,
        order=order,
        axes_to_end=axes_to_end,
        fastpath=True,
    )


def _moments_type(
    values_in: ArrayLike,
    out: NDArrayAny | None,
    mom_params: MomParamsArray,
    to: ConvertStyle,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrderKACF,
    axes_to_end: bool,
    fastpath: bool,
) -> NDArrayAny:
    dtype = select_dtype(values_in, out=out, dtype=dtype, fastpath=fastpath)

    values_in = asarray_maybe_recast(values_in, dtype=dtype, recast=False)
    axes_out = mom_params.axes_last if axes_to_end else mom_params.axes

    # out, by default, will be in calculation order.
    if (
        out is None
        and (order_cf := arrayorder_to_arrayorder_cf(order)) is not None
        and axes_out != mom_params.axes_last
    ):
        shape = tuple(
            values_in.shape[o]
            for o in reorder(values_in.ndim, mom_params.axes, axes_out)
        )
        out = np.empty(shape, dtype=dtype, order=order_cf)

    return factory_convert(mom_ndim=mom_params.ndim, to=to)(
        values_in,
        out=out,
        axes=[mom_params.axes, axes_out],
        dtype=dtype,
        casting=casting,
        order=order,
    )


# * Moments to Cumulative moments
@overload
def cumulative(  # pyright: ignore[reportOverlappingOverload]
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
# arraylike or DataT
@overload
def cumulative(
    values_in: ArrayLike | DataT,
    *,
    out: NDArrayAny | None = ...,
    dtype: Any = ...,
    **kwargs: Unpack[CumulativeKwargs],
) -> NDArrayAny | DataT: ...


@docfiller.decorate  # type: ignore[arg-type,unused-ignore]
def cumulative(  # noqa: PLR0913
    values_in: ArrayLike | DataT,
    *,
    axis: AxisReduceWrap | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: MomNDim | None = None,
    mom_axes: MomAxes | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsType = None,
    inverse: bool = False,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderKACF = None,
    parallel: bool | None = None,
    axes_to_end: bool = False,
    keep_attrs: KeepAttrs = None,
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
    {mom_axes}
    {mom_dims_data}
    {mom_params}
    inverse : bool, optional
        Default is to create a cumulative moments array.  Pass ``inverse=True`` to convert from
        cumulative moments array back to normal moments.
    {out}
    {dtype}
    {casting}
    {order}
    {parallel}
    {axes_to_end}
    {keep_attrs}
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
    if is_xarray_typevar["DataT"].check(values_in):
        prep = PrepareDataXArray.factory(
            mom_params=mom_params,
            ndim=mom_ndim,
            axes=mom_axes,
            dims=mom_dims,
            data=values_in,
            default_ndim=1,
        )
        axis, dim = prep.mom_params.select_axis_dim(
            values_in,
            axis=axis,
            dim=dim,
        )
        core_dims = [prep.mom_params.core_dims(dim)]

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _cumulative,
            values_in,
            input_core_dims=core_dims,
            output_core_dims=core_dims,
            kwargs={
                "prep": prep.prepare_array,
                "inverse": inverse,
                "axis": -(prep.mom_params.ndim + 1),
                "out": prep.optional_out_sample(
                    out,
                    data=values_in,
                    axis=axis,
                    axes_to_end=axes_to_end,
                    order=order,
                    dtype=dtype,
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
                output_dtypes=dtype if dtype is not None else np.float64,  # type: ignore[redundant-expr]
            ),
        )

        if not axes_to_end:
            xout = transpose_like(
                xout,
                template=values_in,
            )
        elif is_dataset(xout):
            xout = xout.transpose(
                ..., dim, *prep.mom_params.dims, missing_dims="ignore"
            )

        return xout

    # Numpy
    prep, axis, values_in = PrepareDataArray.factory(
        mom_params=mom_params,
        ndim=mom_ndim,
        axes=mom_axes,
        default_ndim=1,
    ).data_for_reduction(
        data=values_in,
        axis=axis,
        axes_to_end=axes_to_end,
        dtype=None,  # NOTE(wpk): what?
    )
    return _cumulative(
        values_in,
        out=out,
        axis=axis,
        prep=prep,
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
    prep: PrepareDataArray,
    inverse: bool,
    parallel: bool | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrderKACF,
    fastpath: bool,
) -> NDArrayAny:
    dtype = select_dtype(values_in, out=out, dtype=dtype, fastpath=fastpath)

    axes = prep.mom_params.axes_data_reduction(
        axis=axis,
        out_has_axis=True,
    )

    out = prep.optional_out_sample(
        data=values_in,
        out=out,
        axis=axis,
        axis_new_size=values_in.shape[axis],
        order=order,
        dtype=dtype,
    )

    return factory_cumulative(
        mom_ndim=prep.mom_params.ndim,
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
    if len(mom) != 2:
        msg = "Must supply length 2 sequence for `mom`."
        raise ValueError(msg)

    if mom[0] < 0:
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
# arraylike or DataT
@overload
def moments_to_comoments(
    data: ArrayLike | DataT,
    *,
    mom: tuple[int, int],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[MomentsToComomentsKwargs],
) -> NDArrayAny | DataT: ...


@docfiller.decorate  # type: ignore[arg-type,unused-ignore]
def moments_to_comoments(
    data: ArrayLike | DataT,
    *,
    mom: tuple[int, int],
    mom_axes: MomAxes | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsType = None,
    mom_dims_out: MomDims | None = None,
    dtype: DTypeLike = None,
    order: ArrayOrderCF = None,
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
    {mom_axes}
    {mom_dims_data}
    {mom_params}
    mom_dims_out : tuple of str
        Moments dimensions for output (``mom_ndim=2``) data.  Defaults to ``("mom_0", "mom_1")``.
    {dtype}
    {order_cf}
    {keep_attrs}
    {apply_ufunc_kwargs}

    Returns
    -------
    output : ndarray or DataArray
        Co-moments array.  Same type as ``data``.  Note that new moment dimensions are
        always the last dimensions.

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

    Note that new moment dimensions are always last.

    >>> data = cmomy.default_rng(0).random((6, 1, 2))
    >>> a = moments_to_comoments(data, mom_axes=0, mom=(2, -1))
    >>> a.shape
    (1, 2, 3, 4)
    >>> b = moments_to_comoments(np.moveaxis(data, 0, -1), mom=(2, -1))
    >>> np.testing.assert_equal(a, b)



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
    if is_xarray_typevar["DataT"].check(data):
        mom_params = MomParamsXArray.factory(
            mom_params=mom_params, ndim=1, dims=mom_dims, axes=mom_axes, data=data
        )
        mom_params_out = MomParamsXArray.factory(ndim=2, dims=mom_dims_out)
        if (mom_dim_in := mom_params.dims[0]) in mom_params_out.dims:
            # give this a temporary name for simplicity:
            old_name, mom_dim_in = mom_dim_in, f"_tmp_{mom_dim_in}"
            data = data.rename({old_name: mom_dim_in})

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            moments_to_comoments,
            data,
            input_core_dims=[[mom_dim_in]],
            output_core_dims=[mom_params_out.dims],
            kwargs={"mom": mom, "dtype": dtype},
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_sizes=dict(
                    zip(
                        mom_params_out.dims,
                        mom_to_mom_shape(
                            _validate_mom_moments_to_comoments(
                                mom, data.sizes[mom_dim_in] - 1
                            )
                        ),
                        strict=True,
                    )
                ),
                output_dtypes=dtype if dtype is not None else np.float64,  # type: ignore[redundant-expr]
            ),
        )

        return xout

    # numpy
    data = asarray_maybe_recast(data, dtype=dtype, recast=False)
    # make sure mom_axes are at the end...
    mom_params = MomParamsArray.factory(mom_params=mom_params, ndim=1, axes=mom_axes)
    data = np.moveaxis(data, mom_params.axes, -1)

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
def comoments_to_moments(  # pyright: ignore[reportOverlappingOverload]
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
# arraylike or DataT
@overload
def comoments_to_moments(
    data: ArrayLike | DataT,
    *,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[MomentsToComomentsKwargs],
) -> NDArrayAny | DataT: ...


@docfiller.decorate  # type: ignore[arg-type,unused-ignore]
def comoments_to_moments(
    data: ArrayLike | DataT,
    *,
    mom_axes: MomAxes | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsType = None,
    mom_dims_out: MomDims | None = None,
    dtype: DTypeLike = None,
    order: ArrayOrderCF = None,
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
    {mom_axes}
    {mom_params}
    {mom_dims_data}
    mom_dims_out : tuple of str
        Moments dimensions for output (``mom_ndim=1``) data.  Defaults to ``("mom_0",)``.
    {dtype}
    {order_cf}
    {keep_attrs}
    {apply_ufunc_kwargs}

    Returns
    -------
    output : ndarray or DataArray
        Co-moments array. Same type as ``data``. Note that the output moments
        are in the last dimension.


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
    if is_xarray_typevar["DataT"].check(data):
        mom_params = MomParamsXArray.factory(
            mom_params=mom_params, ndim=2, axes=mom_axes, dims=mom_dims, data=data
        )
        mom_params_out = MomParamsXArray.factory(ndim=1, dims=mom_dims_out)
        if (mom_dim_out := mom_params_out.dims[0]) in mom_params.dims:
            # give this a temporary name for simplicity:
            new_name = f"_tmp_{mom_dim_out}"
            data = data.rename({mom_dim_out: new_name})
            mom_params = mom_params.new_like(
                dims=tuple(new_name if d == mom_dim_out else d for d in mom_params.dims)
            )

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            comoments_to_moments,
            data,
            input_core_dims=[mom_params.dims],
            output_core_dims=[[mom_dim_out]],
            kwargs={"dtype": dtype},
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_sizes={
                    mom_dim_out: sum(data.sizes[k] for k in mom_params.dims) - 1
                },
                output_dtypes=dtype if dtype is not None else np.float64,  # type: ignore[redundant-expr]
            ),
        )
        return xout

    data = asarray_maybe_recast(data, dtype, recast=False)
    # make sure mom_axes at end
    mom_params = MomParamsArray.factory(mom_params=mom_params, ndim=2, axes=mom_axes)
    data = np.moveaxis(data, mom_params.axes, (-2, -1))

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

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr

from cmomy.core.array_utils import (
    asarray_maybe_recast,
    select_dtype,
)
from cmomy.core.docstrings import docfiller
from cmomy.core.missing import MISSING
from cmomy.core.moment_params import (
    MomParamsArray,
    MomParamsXArray,
)
from cmomy.core.prepare import (
    optional_prepare_out_for_resample_data,
    prepare_data_for_reduction,
    prepare_out_for_reduce_data_grouped,
    xprepare_out_for_resample_data,
)
from cmomy.core.validate import (
    is_dataarray,
    is_dataset,
    is_xarray_typevar,
    raise_if_wrong_value,
)
from cmomy.core.xr_utils import (
    factory_apply_ufunc_kwargs,
    replace_coords_from_isel,
    transpose_like,
)
from cmomy.factory import (
    factory_reduce_data_grouped,
    factory_reduce_data_indexed,
    parallel_heuristic,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from cmomy.core.typing import (
        ApplyUFuncKwargs,
        ArrayLikeArg,
        ArrayOrder,
        ArrayOrderCF,
        ArrayT,
        AxisReduceWrap,
        Casting,
        CoordsPolicy,
        DataT,
        DimsReduce,
        DTypeLikeArg,
        FloatT,
        Groups,
        KeepAttrs,
        MissingType,
        MomAxes,
        MomDims,
        MomNDim,
        MomParamsInput,
        NDArrayAny,
        NDArrayInt,
        ReduceDataGroupedKwargs,
        ReduceDataIndexedKwargs,
        Sampler,
    )
    from cmomy.core.typing_compat import Unpack


# * Grouped
@overload
def reduce_data_grouped(
    data: DataT,
    by: ArrayLike,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceDataGroupedKwargs],
) -> DataT: ...
# Array no output or dtype
@overload
def reduce_data_grouped(
    data: ArrayLikeArg[FloatT],
    by: ArrayLike,
    *,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[ReduceDataGroupedKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def reduce_data_grouped(
    data: ArrayLike,
    by: ArrayLike,
    *,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceDataGroupedKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def reduce_data_grouped(
    data: ArrayLike,
    by: ArrayLike,
    *,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[ReduceDataGroupedKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def reduce_data_grouped(
    data: ArrayLike,
    by: ArrayLike,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceDataGroupedKwargs],
) -> NDArrayAny: ...


# ** public
@docfiller.decorate  # type: ignore[arg-type, unused-ignore]
def reduce_data_grouped(  # noqa: PLR0913
    data: ArrayLike | DataT,
    by: ArrayLike,
    *,
    axis: AxisReduceWrap | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: MomNDim | None = None,
    mom_axes: MomAxes | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsInput = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderCF = None,
    parallel: bool | None = None,
    axes_to_end: bool = False,
    group_dim: str | None = None,
    groups: Groups | None = None,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Reduce data by group.


    Parameters
    ----------
    {data_numpy_or_dataarray_or_dataset}
    {by}
    {axis_data}
    {dim}
    {mom_ndim_data}
    {mom_axes}
    {mom_dims_data}
    {mom_params}
    {out}
    {dtype}
    {casting}
    {order_cf}
    {axes_to_end}
    {parallel}
    {group_dim}
    {groups}
    {keep_attrs}
    {apply_ufunc_kwargs}

    Returns
    -------
    out : ndarray or DataArray or Dataset
        Reduced data of same type as input ``data``, with shape
        ``out.shape = (..., shape[axis-1], ngroup, shape[axis+1], ..., mom0, ...)``
        where ``shape = data.shape`` and ngroups = ``by.max() + 1``.


    See Also
    --------
    cmomy.grouped.factor_by

    Examples
    --------
    >>> import cmomy
    >>> data = np.ones((5, 3))
    >>> by = [0, 0, -1, 1, -1]
    >>> reduce_data_grouped(data, mom_ndim=1, axis=0, by=by)
    array([[2., 1., 1.],
           [1., 1., 1.]])

    This also works for :class:`~xarray.DataArray` objects.  In this case,
    the groups are added as coordinates to ``group_dim``

    >>> xout = xr.DataArray(data, dims=["rec", "mom"])
    >>> reduce_data_grouped(xout, mom_ndim=1, dim="rec", by=by, group_dim="group")
    <xarray.DataArray (group: 2, mom: 3)> Size: 48B
    array([[2., 1., 1.],
           [1., 1., 1.]])
    Dimensions without coordinates: group, mom

    Note that if ``by`` skips some groups, they will still be included in
    The output.  For example the following ``by`` skips the value `0`.

    >>> by = [1, 1, -1, 2, 2]
    >>> reduce_data_grouped(xout, mom_ndim=1, dim="rec", by=by)
    <xarray.DataArray (rec: 3, mom: 3)> Size: 72B
    array([[0., 0., 0.],
           [2., 1., 1.],
           [2., 1., 1.]])
    Dimensions without coordinates: rec, mom

    If you want to ensure that only included groups are used, use :func:`cmomy.grouped.factor_by`.
    This has the added benefit of working with non integer groups as well

    >>> by = ["a", "a", None, "b", "b"]
    >>> groups, codes = cmomy.grouped.factor_by(by)
    >>> reduce_data_grouped(xout, mom_ndim=1, dim="rec", by=codes, groups=groups)
    <xarray.DataArray (rec: 2, mom: 3)> Size: 48B
    array([[2., 1., 1.],
           [2., 1., 1.]])
    Coordinates:
      * rec      (rec) <U1 8B 'a' 'b'
    Dimensions without coordinates: mom


    """
    dtype = select_dtype(data, out=out, dtype=dtype)
    by = np.asarray(by, dtype=np.int64)
    if is_xarray_typevar(data):
        mom_params = MomParamsXArray.factory(
            mom_params=mom_params,
            ndim=mom_ndim,
            dims=mom_dims,
            axes=mom_axes,
            data=data,
            default_ndim=1,
        )
        axis, dim = mom_params.select_axis_dim(data, axis=axis, dim=dim)
        core_dims = mom_params.core_dims(dim)

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _reduce_data_grouped,
            data,
            by,
            input_core_dims=[core_dims, [dim]],
            output_core_dims=[core_dims],
            exclude_dims={dim},
            kwargs={
                "mom_params": mom_params.to_array(),
                # Need total axis here...
                "axis": -(mom_params.ndim + 1),
                "dtype": dtype,
                "out": xprepare_out_for_resample_data(
                    out,
                    mom_params=mom_params,
                    axis=axis,
                    axes_to_end=axes_to_end,
                    data=data,
                ),
                "casting": casting,
                "order": order,
                "parallel": parallel,
                "fastpath": is_dataarray(data),
            },
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_sizes={dim: by.max() + 1},
                output_dtypes=dtype or np.float64,
            ),
        )

        if not axes_to_end:
            xout = transpose_like(
                xout,
                template=data,
            )
        elif is_dataset(xout):
            xout = xout.transpose(..., dim, *mom_params.dims, missing_dims="ignore")  # pyright: ignore[reportUnknownArgumentType]

        if groups is not None:
            xout = xout.assign_coords({dim: (dim, groups)})  # pyright: ignore[reportUnknownMemberType]
        if group_dim:
            xout = xout.rename({dim: group_dim})

        return xout

    # Numpy
    axis, mom_params, data = prepare_data_for_reduction(
        data=data,
        axis=axis,
        mom_params=MomParamsArray.factory(
            mom_params=mom_params, ndim=mom_ndim, axes=mom_axes, default_ndim=1
        ),
        dtype=dtype,
        recast=False,
        axes_to_end=axes_to_end,
    )
    return _reduce_data_grouped(
        data,
        by=by,
        mom_params=mom_params,
        axis=axis,
        dtype=dtype,
        out=out,
        casting=casting,
        order=order,
        parallel=parallel,
        fastpath=True,
    )


def _reduce_data_grouped(
    data: NDArrayAny,
    by: NDArrayInt,
    *,
    mom_params: MomParamsArray,
    axis: int,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrderCF,
    out: NDArrayAny | None,
    parallel: bool | None,
    fastpath: bool,
) -> NDArrayAny:
    if not fastpath:
        dtype = select_dtype(data, out=out, dtype=dtype)

    # include inner core dims for by
    axes = mom_params.axes_data_reduction(
        (-1,),
        axis=axis,
        out_has_axis=True,
    )
    raise_if_wrong_value(len(by), data.shape[axis], "Wrong length of `by`.")

    if out is None:
        out = prepare_out_for_reduce_data_grouped(
            data,
            mom_params=mom_params,
            axis=axis,
            axis_new_size=by.max() + 1,
            order=order,
            dtype=dtype,
        )
    out.fill(0.0)

    # pylint: disable=unexpected-keyword-arg
    factory_reduce_data_grouped(
        mom_ndim=mom_params.ndim,
        parallel=parallel_heuristic(parallel, size=data.size),
    )(
        data,
        by,
        out,
        axes=axes,
        casting=casting,
        signature=(dtype, np.int64, dtype),
    )
    return out


# * Indexed
def _validate_index(
    ndat: int,
    index: ArrayLike,
    group_start: ArrayLike,
    group_end: ArrayLike,
) -> tuple[NDArrayInt, NDArrayInt, NDArrayInt]:
    def _validate(name: str, x: NDArrayInt, upper: int) -> None:
        min_: int = x.min()
        max_: int = x.max()
        if min_ < 0:
            msg = f"min({name}) = {min_} < 0"
            raise ValueError(msg)
        if max_ >= upper:
            msg = f"max({name}) = {max_} > {upper}"
            raise ValueError(msg)

    index = np.asarray(index, dtype=np.int64)
    group_start = np.asarray(group_start, dtype=np.int64)
    group_end = np.asarray(group_end, dtype=np.int64)

    # TODO(wpk): if nindex == 0, should just get out of here?
    nindex = len(index)
    if nindex == 0:
        if (group_start != 0).any() or (group_end != 0).any():
            msg = "With zero length index, group_start = group_stop = 0"
            raise ValueError(msg)
    else:
        _validate("index", index, ndat)
        _validate("group_start", group_start, nindex)
        _validate("group_end", group_end, nindex + 1)

    raise_if_wrong_value(
        len(group_start),
        len(group_end),
        "`group_start` and `group_end` must have same length",
    )

    if (group_end < group_start).any():
        msg = "Found end < start"
        raise ValueError(msg)

    return index, group_start, group_end


@overload
def reduce_data_indexed(
    data: DataT,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceDataIndexedKwargs],
) -> DataT: ...
# Array no out or dtype
@overload
def reduce_data_indexed(
    data: ArrayLikeArg[FloatT],
    *,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[ReduceDataIndexedKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def reduce_data_indexed(
    data: ArrayLike,
    *,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceDataIndexedKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def reduce_data_indexed(
    data: ArrayLike,
    *,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[ReduceDataIndexedKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def reduce_data_indexed(
    data: ArrayLike,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceDataIndexedKwargs],
) -> NDArrayAny: ...


# ** public
@docfiller.decorate  # type: ignore[arg-type, unused-ignore]
def reduce_data_indexed(  # noqa: PLR0913
    data: ArrayLike | DataT,
    *,
    index: ArrayLike,
    group_start: ArrayLike,
    group_end: ArrayLike,
    scale: ArrayLike | None = None,
    axis: AxisReduceWrap | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: MomNDim | None = None,
    mom_axes: MomAxes | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsInput = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrder = None,
    parallel: bool | None = None,
    axes_to_end: bool = False,
    coords_policy: CoordsPolicy = "first",
    group_dim: str | None = None,
    groups: Groups | None = None,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Reduce data by index

    Parameters
    ----------
    {data_numpy_or_dataarray_or_dataset}
    index : ndarray
        Index into `data.shape[axis]`.
    group_start, group_end : ndarray
        Start, end of index for a group.
        ``index[group_start[group]:group_end[group]]`` are the indices for
        group ``group``.
    scale : ndarray, optional
        Weights of same size as ``index``.
    {axis_data}
    {dim}
    {mom_ndim_data}
    {mom_axes}
    {mom_dims_data}
    {mom_params}
    {out}
    {dtype}
    {casting}
    {order}
    {parallel}
    {axes_to_end}
    {coords_policy}
    {group_dim}
    {groups}
    {keep_attrs}
    {apply_ufunc_kwargs}

    Returns
    -------
    out : ndarray or DataArray or Dataset
        Reduced data of same type as input ``data``, with shape
        ``out.shape = (..., shape[axis-1], ngroup, shape[axis+1], ..., mom0, ...)``,
        where ``shape = data.shape`` and ``ngroup = len(group_start)``.

    See Also
    --------
    cmomy.grouped.factor_by_to_index


    Examples
    --------
    This is a more general reduction than :func:`reduce_data_grouped`, but
    it can be used similarly.

    >>> import cmomy
    >>> data = np.ones((5, 3))
    >>> by = ["a", "a", "b", "b", "c"]
    >>> groups, index, start, end = cmomy.grouped.factor_by_to_index(by)
    >>> reduce_data_indexed(
    ...     data, mom_ndim=1, axis=0, index=index, group_start=start, group_end=end
    ... )
    array([[2., 1., 1.],
           [2., 1., 1.],
           [1., 1., 1.]])

    This also works for :class:`~xarray.DataArray` objects

    >>> xout = xr.DataArray(data, dims=["rec", "mom"])
    >>> reduce_data_indexed(
    ...     xout,
    ...     mom_ndim=1,
    ...     dim="rec",
    ...     index=index,
    ...     group_start=start,
    ...     group_end=end,
    ...     group_dim="group",
    ...     groups=groups,
    ...     coords_policy="group",
    ... )
    <xarray.DataArray (group: 3, mom: 3)> Size: 72B
    array([[2., 1., 1.],
           [2., 1., 1.],
           [1., 1., 1.]])
    Coordinates:
      * group    (group) <U1 12B 'a' 'b' 'c'
    Dimensions without coordinates: mom
    """
    dtype = select_dtype(data, out=out, dtype=dtype)

    if is_xarray_typevar(data):
        mom_params = MomParamsXArray.factory(
            mom_params=mom_params,
            ndim=mom_ndim,
            dims=mom_dims,
            axes=mom_axes,
            data=data,
            default_ndim=1,
        )
        axis, dim = mom_params.select_axis_dim(data, axis=axis, dim=dim)
        core_dims = mom_params.core_dims(dim)

        # Yes, doing this here and in nmpy section.
        index, group_start, group_end = _validate_index(
            ndat=data.sizes[dim],
            index=index,
            group_start=group_start,
            group_end=group_end,
        )

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _reduce_data_indexed,
            data,
            input_core_dims=[core_dims],
            output_core_dims=[core_dims],
            exclude_dims={dim},
            kwargs={
                "axis": -(mom_params.ndim + 1),
                "mom_params": mom_params.to_array(),
                "index": index,
                "group_start": group_start,
                "group_end": group_end,
                "scale": scale,
                "out": xprepare_out_for_resample_data(
                    out,
                    mom_params=mom_params,
                    axis=axis,
                    axes_to_end=axes_to_end,
                    data=data,
                ),
                "dtype": dtype,
                "casting": casting,
                "order": order,
                "parallel": parallel,
                "fastpath": is_dataarray(data),
            },
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_sizes={dim: len(group_start)},
                output_dtypes=dtype or np.float64,
            ),
        )

        if not axes_to_end:
            xout = transpose_like(
                xout,
                template=data,
            )
        elif is_dataset(xout):
            xout = xout.transpose(..., dim, *mom_params.dims, missing_dims="ignore")  # pyright: ignore[reportUnknownArgumentType]

        if coords_policy in {"first", "last"} and is_dataarray(data):
            # in case we passed in index, group_start, group_end as non-arrays
            # these will be processed correctly in the numpy call
            # but just make sure here....
            index, group_start, group_end = (
                np.asarray(_, dtype=np.int64) for _ in (index, group_start, group_end)
            )
            dim_select = index[
                group_end - 1 if coords_policy == "last" else group_start
            ]

            xout = replace_coords_from_isel(  # type: ignore[assignment, unused-ignore]  # error with python3.12
                da_original=data,
                da_selected=xout,  # type: ignore[arg-type, unused-ignore]  # error python3.12 and pyright
                indexers={dim: dim_select},
                drop=False,
            )
        elif coords_policy == "group" and groups is not None:
            xout = xout.assign_coords({dim: groups})  # pyright: ignore[reportUnknownMemberType]
        if group_dim:
            xout = xout.rename({dim: group_dim})

        return xout

    # Numpy
    axis, mom_params, data = prepare_data_for_reduction(
        data=data,
        axis=axis,
        mom_params=MomParamsArray.factory(
            mom_params=mom_params, ndim=mom_ndim, axes=mom_axes, default_ndim=1
        ),
        dtype=dtype,
        recast=False,
        axes_to_end=axes_to_end,
    )

    index, group_start, group_end = _validate_index(
        ndat=data.shape[axis],
        index=index,
        group_start=group_start,
        group_end=group_end,
    )

    return _reduce_data_indexed(
        data,
        axis=axis,
        mom_params=mom_params,
        index=index,
        group_start=group_start,
        group_end=group_end,
        scale=scale,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        parallel=parallel,
        fastpath=True,
    )


def _reduce_data_indexed(
    data: NDArrayAny,
    *,
    axis: int,
    mom_params: MomParamsArray,
    index: NDArrayAny,
    group_start: NDArrayAny,
    group_end: NDArrayAny,
    scale: ArrayLike | None,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrder,
    parallel: bool | None,
    fastpath: bool = False,
) -> NDArrayAny:
    if not fastpath:
        dtype = select_dtype(data, out=out, dtype=dtype)

    if scale is None:
        scale = np.broadcast_to(np.dtype(dtype).type(1), index.shape)
    else:
        scale = asarray_maybe_recast(scale, dtype=dtype, recast=False)
        raise_if_wrong_value(
            len(scale), len(index), "`scale` and `index` must have same length."
        )

    # include inner dims for index, start, end, scale
    axes = mom_params.axes_data_reduction(
        *(-1,) * 4,
        axis=axis,
        out_has_axis=True,
    )

    # optional out with correct ordering
    out = optional_prepare_out_for_resample_data(
        out=out,
        data=data,
        axis=axis,
        axis_new_size=len(group_start),
        order=order,
        dtype=dtype,
    )

    # pylint: disable=unexpected-keyword-arg
    return factory_reduce_data_indexed(
        mom_ndim=mom_params.ndim,
        parallel=parallel_heuristic(parallel, size=data.size),
    )(
        data,
        index,
        group_start,
        group_end,
        scale,  # pyright: ignore[reportArgumentType]
        out=out,
        axes=axes,
        casting=casting,
        order=order,
        signature=(dtype, np.int64, np.int64, np.int64, dtype, dtype),
    )


# * For testing purposes
def resample_data_indexed(  # noqa: PLR0913
    data: ArrayT,
    sampler: Sampler,
    *,
    mom_ndim: MomNDim | None = None,
    axis: AxisReduceWrap | MissingType = MISSING,
    mom_axes: MomAxes | None = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrder = None,
    parallel: bool = True,
    axes_to_end: bool = False,
    # xarray specific
    dim: DimsReduce | MissingType = MISSING,
    mom_dims: MomDims | None = None,
    coords_policy: CoordsPolicy = "first",
    rep_dim: str = "rep",
    groups: Groups | None = None,
    keep_attrs: KeepAttrs = None,
) -> ArrayT:
    """Resample using indexed reduction."""
    from cmomy.resample import factory_sampler

    sampler = factory_sampler(
        sampler,
        data=data,
        axis=axis,
        mom_axes=mom_axes,
        dim=dim,
        mom_ndim=mom_ndim,
        mom_dims=mom_dims,
        rep_dim=rep_dim,
        parallel=parallel,
    )

    from cmomy._lib.utils import freq_to_index_start_end_scales

    index, start, end, scales = freq_to_index_start_end_scales(sampler.freq)

    return reduce_data_indexed(  # pyright: ignore[reportReturnType]
        data=data,
        mom_ndim=mom_ndim,
        index=index,
        group_start=start,
        group_end=end,
        scale=scales,
        axis=axis,
        mom_axes=mom_axes,
        axes_to_end=axes_to_end,
        parallel=parallel,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        dim=dim,
        mom_dims=mom_dims,
        coords_policy=coords_policy,
        group_dim=rep_dim,
        groups=groups,
        keep_attrs=keep_attrs,
    )

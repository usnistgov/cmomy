from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr

from cmomy.core.array_utils import (
    asarray_maybe_recast,
    get_axes_from_values,
    select_dtype,
)
from cmomy.core.docstrings import docfiller
from cmomy.core.missing import MISSING
from cmomy.core.prepare import (
    PrepareDataArray,
    PrepareDataXArray,
    PrepareValsArray,
    PrepareValsXArray,
)
from cmomy.core.utils import mom_to_mom_shape
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
    factory_reduce_vals_grouped,
    factory_reduce_vals_indexed,
    parallel_heuristic,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from cmomy.core._typing_kwargs import (
        ApplyUFuncKwargs,
        ReduceDataGroupedKwargs,
        ReduceDataIndexedKwargs,
        ReduceValsGroupedKwargs,
        ReduceValsIndexedKwargs,
    )
    from cmomy.core.moment_params import MomParamsType
    from cmomy.core.typing import (
        ArrayLikeArg,
        ArrayOrderCF,
        ArrayOrderKACF,
        AxesGUFunc,
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
        Moments,
        MomentsStrict,
        MomNDim,
        NDArrayAny,
        NDArrayInt,
    )
    from cmomy.core.typing_compat import Unpack


# * Utils
def _apply_coords_policy_indexed(
    *,
    selected: DataT,
    template: DataT,
    dim: Hashable,
    coords_policy: CoordsPolicy,
    index: NDArrayInt,
    group_start: NDArrayInt,
    group_end: NDArrayInt,
    groups: Groups | None,
) -> DataT:
    if coords_policy in {"first", "last"} and is_dataarray(selected):
        # in case we passed in index, group_start, group_end as non-arrays
        dim_select = index[group_end - 1 if coords_policy == "last" else group_start]

        return replace_coords_from_isel(  # type: ignore[assignment, unused-ignore]  # error with python3.12
            template=template,
            selected=selected,
            indexers={dim: dim_select},
            drop=False,
        )
    if coords_policy == "group" and groups is not None:
        return selected.assign_coords({dim: groups})  # pyright: ignore[reportUnknownMemberType]

    return selected


def _apply_coords_policy_grouped(
    *,
    selected: DataT,
    template: DataT,
    dim: Hashable,
    coords_policy: CoordsPolicy,
    by: NDArrayInt,
    groups: Groups | None,
) -> DataT:
    if coords_policy == "group" and groups is not None:
        return selected.assign_coords({dim: groups})  # pyright: ignore[reportUnknownMemberType]

    from ._factorize import factor_by_to_index

    index, start, end, _ = factor_by_to_index(by)

    return _apply_coords_policy_indexed(
        selected=selected,
        template=template,
        dim=dim,
        coords_policy=coords_policy,
        index=index,
        group_start=start,
        group_end=end,
        groups=None,
    )


def _optional_group_dim(
    data: DataT, dim: Hashable, group_dim: str | None = None
) -> DataT:
    if group_dim:
        return data.rename({dim: group_dim})
    return data


# * Data ----------------------------------------------------------------------
# ** Grouped
@overload
def reduce_data_grouped(  # pyright: ignore[reportOverlappingOverload]
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
# arraylike or DataT
@overload
def reduce_data_grouped(
    data: ArrayLike | DataT,
    by: ArrayLike,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceDataGroupedKwargs],
) -> NDArrayAny | DataT: ...


# *** public
@docfiller.decorate  # type: ignore[arg-type, unused-ignore]  # error with python3.13.  Flags passed != expected, but they're the same...
def reduce_data_grouped(  # noqa: PLR0913
    data: ArrayLike | DataT,
    by: ArrayLike,
    *,
    axis: AxisReduceWrap | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: MomNDim | None = None,
    mom_axes: MomAxes | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsType = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderCF = None,
    parallel: bool | None = None,
    axes_to_end: bool = False,
    coords_policy: CoordsPolicy = "group",
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
    {coords_policy}
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
    >>> codes, groups = cmomy.grouped.factor_by(by)
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
    if is_xarray_typevar["DataT"].check(data):
        prep = PrepareDataXArray.factory(
            mom_params=mom_params,
            ndim=mom_ndim,
            axes=mom_axes,
            dims=mom_dims,
            data=data,
            default_ndim=1,
        )
        axis, dim = prep.mom_params.select_axis_dim(data, axis=axis, dim=dim)
        core_dims = prep.mom_params.core_dims(dim)
        axis_new_size = by.max() + 1

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _reduce_data_grouped,
            data,
            by,
            input_core_dims=[core_dims, [dim]],
            output_core_dims=[core_dims],
            exclude_dims={dim},
            kwargs={
                "prep": prep.prepare_array,
                # Need total axis here...
                "axis": -(prep.mom_params.ndim + 1),
                "dtype": dtype,
                "out": prep.optional_out_sample(
                    out,
                    data=data,
                    axis=axis,
                    axis_new_size=axis_new_size,
                    axes_to_end=axes_to_end,
                    order=order,
                    dtype=dtype,
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
                output_sizes={dim: axis_new_size},
                output_dtypes=dtype if dtype is not None else np.float64,  # type: ignore[redundant-expr]
            ),
        )

        xout = _apply_coords_policy_grouped(
            selected=xout,
            template=data,
            dim=dim,
            coords_policy=coords_policy,
            by=by,
            groups=groups,
        )

        if not axes_to_end:
            xout = transpose_like(
                xout,
                template=data,
            )
        elif is_dataset(xout):
            xout = xout.transpose(
                ..., dim, *prep.mom_params.dims, missing_dims="ignore"
            )

        return _optional_group_dim(xout, dim, group_dim)

    # Numpy
    prep, axis, data = PrepareDataArray.factory(
        mom_params=mom_params,
        ndim=mom_ndim,
        axes=mom_axes,
        default_ndim=1,
    ).data_for_reduction(
        data=data,
        axis=axis,
        axes_to_end=axes_to_end,
        dtype=dtype,
    )
    return _reduce_data_grouped(
        data,
        by=by,
        prep=prep,
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
    prep: PrepareDataArray,
    axis: int,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrderCF,
    out: NDArrayAny | None,
    parallel: bool | None,
    fastpath: bool,
) -> NDArrayAny:
    dtype = select_dtype(data, out=out, dtype=dtype, fastpath=fastpath)

    # include inner core dims for by
    axes = prep.mom_params.axes_data_reduction(
        (-1,),
        axis=axis,
        out_has_axis=True,
    )
    raise_if_wrong_value(len(by), data.shape[axis], "Wrong length of `by`.")

    out = prep.out_sample(
        out,
        data=data,
        axis=axis,
        axis_new_size=by.max() + 1,
        order=order,
        dtype=dtype,
    )
    out.fill(0.0)

    _ = factory_reduce_data_grouped(
        mom_ndim=prep.mom_params.ndim,
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


# ** Indexed
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
def reduce_data_indexed(  # pyright: ignore[reportOverlappingOverload]
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
# arraylike or DataT
@overload
def reduce_data_indexed(
    data: ArrayLike | DataT,
    *,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceDataIndexedKwargs],
) -> NDArrayAny | DataT: ...


# *** public
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
    mom_params: MomParamsType = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderKACF = None,
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
    >>> index, start, end, groups = cmomy.grouped.factor_by_to_index(by)
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

    if is_xarray_typevar["DataT"].check(data):
        prep = PrepareDataXArray.factory(
            mom_params=mom_params,
            ndim=mom_ndim,
            axes=mom_axes,
            dims=mom_dims,
            data=data,
            default_ndim=1,
        )
        axis, dim = prep.mom_params.select_axis_dim(data, axis=axis, dim=dim)
        core_dims = prep.mom_params.core_dims(dim)

        # Yes, doing this here and in numpy section.
        index, group_start, group_end = _validate_index(
            ndat=data.sizes[dim],
            index=index,
            group_start=group_start,
            group_end=group_end,
        )
        axis_new_size = len(group_start)

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _reduce_data_indexed,
            data,
            input_core_dims=[core_dims],
            output_core_dims=[core_dims],
            exclude_dims={dim},
            kwargs={
                "axis": -(prep.mom_params.ndim + 1),
                "prep": prep.prepare_array,
                "index": index,
                "group_start": group_start,
                "group_end": group_end,
                "scale": scale,
                "out": prep.optional_out_sample(
                    out,
                    data=data,
                    axis=axis,
                    axis_new_size=axis_new_size,
                    axes_to_end=axes_to_end,
                    order=order,
                    dtype=dtype,
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
                output_sizes={dim: axis_new_size},
                output_dtypes=dtype if dtype is not None else np.float64,  # type: ignore[redundant-expr]
            ),
        )

        xout = _apply_coords_policy_indexed(
            selected=xout,
            template=data,
            dim=dim,
            coords_policy=coords_policy,
            index=index,
            group_start=group_start,
            group_end=group_end,
            groups=groups,
        )

        if not axes_to_end:
            xout = transpose_like(
                xout,
                template=data,
            )
        elif is_dataset(xout):
            xout = xout.transpose(
                ..., dim, *prep.mom_params.dims, missing_dims="ignore"
            )

        return _optional_group_dim(xout, dim, group_dim)

    # Numpy
    prep, axis, data = PrepareDataArray.factory(
        mom_params=mom_params,
        ndim=mom_ndim,
        axes=mom_axes,
        default_ndim=1,
    ).data_for_reduction(
        data=data,
        axis=axis,
        axes_to_end=axes_to_end,
        dtype=dtype,
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
        prep=prep,
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
    prep: PrepareDataArray,
    index: NDArrayAny,
    group_start: NDArrayAny,
    group_end: NDArrayAny,
    scale: ArrayLike | None,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrderKACF,
    parallel: bool | None,
    fastpath: bool = False,
) -> NDArrayAny:
    dtype = select_dtype(data, out=out, dtype=dtype, fastpath=fastpath)

    if scale is None:
        scale = np.broadcast_to(np.dtype(dtype).type(1), index.shape)
    else:
        scale = asarray_maybe_recast(scale, dtype=dtype, recast=False)
        raise_if_wrong_value(
            len(scale), len(index), "`scale` and `index` must have same length."
        )

    # include inner dims for index, start, end, scale
    axes = prep.mom_params.axes_data_reduction(
        *(-1,) * 4,
        axis=axis,
        out_has_axis=True,
    )

    # optional out with correct ordering
    out = prep.optional_out_sample(
        out=out,
        data=data,
        axis=axis,
        axis_new_size=len(group_start),
        order=order,
        dtype=dtype,
    )

    return factory_reduce_data_indexed(
        mom_ndim=prep.mom_params.ndim,
        parallel=parallel_heuristic(parallel, size=data.size),
    )(
        data,
        index,
        group_start,
        group_end,
        scale,
        out=out,
        axes=axes,
        casting=casting,
        order=order,
        signature=(dtype, np.int64, np.int64, np.int64, dtype, dtype),
    )


# * Vals
# ** Grouped
@overload
def reduce_vals_grouped(  # pyright: ignore[reportOverlappingOverload]
    x: DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    by: ArrayLike,
    weight: ArrayLike | xr.DataArray | DataT | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceValsGroupedKwargs],
) -> DataT: ...
# array
@overload
def reduce_vals_grouped(
    x: ArrayLikeArg[FloatT],
    *y: ArrayLike,
    by: ArrayLike,
    weight: ArrayLike | None = ...,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[ReduceValsGroupedKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def reduce_vals_grouped(
    x: ArrayLike,
    *y: ArrayLike,
    by: ArrayLike,
    weight: ArrayLike | None = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceValsGroupedKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def reduce_vals_grouped(
    x: ArrayLike,
    *y: ArrayLike,
    by: ArrayLike,
    weight: ArrayLike | None = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[ReduceValsGroupedKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def reduce_vals_grouped(
    x: ArrayLike,
    *y: ArrayLike,
    by: ArrayLike,
    weight: ArrayLike | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceValsGroupedKwargs],
) -> NDArrayAny: ...
# arraylike or DataT
@overload
def reduce_vals_grouped(
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    by: ArrayLike,
    weight: ArrayLike | xr.DataArray | DataT | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceValsGroupedKwargs],
) -> NDArrayAny | DataT: ...


@docfiller.decorate  # type: ignore[arg-type, unused-ignore]
def reduce_vals_grouped(  # noqa: PLR0913
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    by: ArrayLike,
    mom: Moments,
    axis: AxisReduceWrap | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    weight: ArrayLike | xr.DataArray | DataT | None = None,
    mom_dims: MomDims | None = None,
    mom_axes: MomAxes | None = None,
    mom_params: MomParamsType = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderCF = None,
    parallel: bool | None = None,
    axes_to_end: bool = False,
    coords_policy: CoordsPolicy = "group",
    group_dim: str | None = None,
    groups: Groups | None = None,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Reduce value with grouping.

    Parameters
    ----------
    {x_genarray}
    {y_genarray}
    {by}
    {mom}
    {axis}
    {dim}
    {weight_genarray}
    {mom_dims}
    {mom_axes}
    {mom_params}
    {out}
    {dtype}
    {casting}
    {order_cf}
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
        Reduced data of same type as input arrays.

    See Also
    --------
    .reduce_vals
    reduce_data_grouped
    """
    weight = 1.0 if weight is None else weight
    dtype = select_dtype(x, out=out, dtype=dtype)
    by = np.asarray(by, dtype=np.int64)
    axis_new_size = by.max() + 1

    if is_xarray_typevar["DataT"].check(x):
        prep, mom = PrepareValsXArray.factory_mom(
            mom=mom, mom_params=mom_params, dims=mom_dims, recast=False
        )
        dim, input_core_dims, xargs = prep.values_for_reduction(
            x,
            weight,
            *y,
            axis=axis,
            dim=dim,
            narrays=prep.mom_params.ndim + 1,
            dtype=dtype,
        )

        out, mom_params_axes = prep.optional_out_from_values(
            out,
            *xargs,
            target=x,
            dim=dim,
            mom=mom,
            axis_new_size=axis_new_size,
            axes_to_end=axes_to_end,
            order=order,
            dtype=dtype,
            mom_axes=mom_axes,
            mom_params=prep.mom_params,
        )

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _reduce_vals_grouped,
            *xargs,
            by,
            input_core_dims=[*input_core_dims, [dim]],  # type: ignore[has-type]
            output_core_dims=[prep.mom_params.core_dims(dim)],
            exclude_dims={dim},
            kwargs={
                "mom": mom,
                "prep": prep.prepare_array,
                "axis_neg": -1,
                "out": out,
                "dtype": dtype,
                "casting": casting,
                "order": order,
                "parallel": parallel,
                "fastpath": is_dataarray(x),
                "axis_new_size": axis_new_size,
            },
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_sizes={
                    dim: axis_new_size,
                    **dict(
                        zip(prep.mom_params.dims, mom_to_mom_shape(mom), strict=True)
                    ),
                },
                output_dtypes=dtype if dtype is not None else np.float64,  # type: ignore[redundant-expr]
            ),
        )

        xout = _apply_coords_policy_grouped(
            selected=xout,
            template=x,
            dim=dim,
            coords_policy=coords_policy,
            by=by,
            groups=groups,
        )

        if not axes_to_end:
            xout = transpose_like(
                xout,
                template=x,
                append=prep.mom_params.dims,
                mom_params_axes=mom_params_axes,
            )
        elif is_dataset(x):
            xout = xout.transpose(
                ...,
                dim,
                *prep.mom_params.dims,
                missing_dims="ignore",
            )

        return _optional_group_dim(xout, dim, group_dim)

    # Numpy
    prep, mom = PrepareValsArray.factory_mom(
        mom=mom, axes=mom_axes, mom_params=mom_params, recast=False
    )
    prep, axis_neg, args = prep.values_for_reduction(
        x,
        weight,
        *y,
        axis=axis,
        narrays=prep.mom_params.ndim + 1,
        axes_to_end=axes_to_end,
        dtype=dtype,
    )

    return _reduce_vals_grouped(
        *args,
        by,
        mom=mom,
        prep=prep,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        axis_neg=axis_neg,
        axis_new_size=axis_new_size,
        parallel=parallel,
        fastpath=True,
    )


def _reduce_vals_grouped(
    # x, w, *y, by
    *args: NDArrayAny,
    mom: MomentsStrict,
    prep: PrepareValsArray,
    axis_neg: int,
    axis_new_size: int,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrderCF,
    parallel: bool | None,
    fastpath: bool = False,
) -> NDArrayAny:
    dtype = select_dtype(args[0], out=out, dtype=dtype, fastpath=fastpath)

    args, by = args[:-1], args[-1]

    out, axis_sample_out = prep.out_from_values(
        out,
        val_shape=prep.get_val_shape(*args),
        mom=mom,
        axis_neg=axis_neg,
        axis_new_size=axis_new_size,
        dtype=dtype,
        order=order,
    )
    out.fill(0.0)

    axes: AxesGUFunc = [
        # out
        (axis_sample_out, *prep.mom_params.axes),
        # by
        (-1,),
        # x, weight, *y
        *get_axes_from_values(*args, axis_neg=axis_neg),
    ]

    _ = factory_reduce_vals_grouped(
        mom_ndim=prep.mom_params.ndim,
        parallel=parallel_heuristic(parallel, size=args[0].size * prep.mom_params.ndim),
    )(
        out,
        by,
        *args,
        axes=axes,
        casting=casting,
        order=order,
        signature=(dtype, np.int64) + (dtype,) * len(args),
    )

    return out


# ** Indexed
@overload
def reduce_vals_indexed(  # pyright: ignore[reportOverlappingOverload]
    x: DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    weight: ArrayLike | xr.DataArray | DataT | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceValsIndexedKwargs],
) -> DataT: ...
# array
@overload
def reduce_vals_indexed(
    x: ArrayLikeArg[FloatT],
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[ReduceValsIndexedKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def reduce_vals_indexed(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceValsIndexedKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def reduce_vals_indexed(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
    **kwargs: Unpack[ReduceValsIndexedKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def reduce_vals_indexed(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceValsIndexedKwargs],
) -> NDArrayAny: ...
# arraylike or DataT
@overload
def reduce_vals_indexed(
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    weight: ArrayLike | xr.DataArray | DataT | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ReduceValsIndexedKwargs],
) -> NDArrayAny | DataT: ...


@docfiller.decorate  # type: ignore[arg-type, unused-ignore]
def reduce_vals_indexed(  # noqa: PLR0913
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    index: ArrayLike,
    group_start: ArrayLike,
    group_end: ArrayLike,
    scale: ArrayLike | None = None,
    mom: Moments,
    axis: AxisReduceWrap | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    weight: ArrayLike | xr.DataArray | DataT | None = None,
    mom_dims: MomDims | None = None,
    mom_axes: MomAxes | None = None,
    mom_params: MomParamsType = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
    casting: Casting = "same_kind",
    order: ArrayOrderCF = None,
    parallel: bool | None = None,
    axes_to_end: bool = False,
    coords_policy: CoordsPolicy = "first",
    group_dim: str | None = None,
    groups: Groups | None = None,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
    """
    Reduce value with grouping.

    Parameters
    ----------
    {x_genarray}
    {y_genarray}
    index : ndarray
        Index into `data.shape[axis]`.
    group_start, group_end : ndarray
        Start, end of index for a group.
        ``index[group_start[group]:group_end[group]]`` are the indices for
        group ``group``.
    scale : ndarray, optional
        Weights of same size as ``index``.
    {mom}
    {axis}
    {dim}
    {weight_genarray}
    {mom_dims}
    {mom_axes}
    {mom_params}
    {out}
    {dtype}
    {casting}
    {order_cf}
    {parallel}
    {axes_to_end}
    {coords_policy}
    {group_dim}
    {groups}
    {keep_attrs}
    {apply_ufunc_kwargs}

    See Also
    --------
    .reduce_vals
    reduce_data_indexed
    reduce_vals_grouped
    """
    weight = 1.0 if weight is None else weight
    dtype = select_dtype(x, out=out, dtype=dtype)

    if is_xarray_typevar["DataT"].check(x):
        prep, mom = PrepareValsXArray.factory_mom(
            mom=mom, mom_params=mom_params, dims=mom_dims, recast=False
        )
        dim, input_core_dims, xargs = prep.values_for_reduction(
            x,
            weight,
            *y,
            axis=axis,
            dim=dim,
            narrays=prep.mom_params.ndim + 1,
            dtype=dtype,
        )

        index, group_start, group_end = _validate_index(
            ndat=x.sizes[dim],
            index=index,
            group_start=group_start,
            group_end=group_end,
        )

        out, mom_params_axes = prep.optional_out_from_values(
            out,
            *xargs,
            target=x,
            dim=dim,
            mom=mom,
            axis_new_size=len(group_start),
            axes_to_end=axes_to_end,
            order=order,
            dtype=dtype,
            mom_axes=mom_axes,
            mom_params=prep.mom_params,
        )

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _reduce_vals_indexed,
            *xargs,
            input_core_dims=input_core_dims,
            output_core_dims=[prep.mom_params.core_dims(dim)],
            exclude_dims={dim},
            kwargs={
                "mom": mom,
                "prep": prep.prepare_array,
                "index": index,
                "group_start": group_start,
                "group_end": group_end,
                "scale": scale,
                "axis_neg": -1,
                "out": out,
                "dtype": dtype,
                "casting": casting,
                "order": order,
                "parallel": parallel,
                "fastpath": is_dataarray(x),
            },
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_sizes={
                    dim: len(group_start),
                    **dict(
                        zip(prep.mom_params.dims, mom_to_mom_shape(mom), strict=True)
                    ),
                },
                output_dtypes=dtype if dtype is not None else np.float64,  # type: ignore[redundant-expr]
            ),
        )

        xout = _apply_coords_policy_indexed(
            selected=xout,
            template=x,
            dim=dim,
            coords_policy=coords_policy,
            index=index,
            group_start=group_start,
            group_end=group_end,
            groups=groups,
        )

        if not axes_to_end:
            xout = transpose_like(
                xout,
                template=x,
                append=prep.mom_params.dims,
                mom_params_axes=mom_params_axes,
            )
        elif is_dataset(x):
            xout = xout.transpose(
                ...,
                dim,
                *prep.mom_params.dims,
                missing_dims="ignore",
            )

        return _optional_group_dim(xout, dim, group_dim)

    # Numpy
    prep, mom = PrepareValsArray.factory_mom(
        mom=mom, axes=mom_axes, mom_params=mom_params, recast=False
    )
    prep, axis_neg, args = prep.values_for_reduction(
        x,
        weight,
        *y,
        axis=axis,
        narrays=prep.mom_params.ndim + 1,
        axes_to_end=axes_to_end,
        dtype=dtype,
    )

    index, group_start, group_end = _validate_index(
        ndat=args[0].shape[axis_neg],
        index=index,
        group_start=group_start,
        group_end=group_end,
    )

    return _reduce_vals_indexed(
        *args,
        index=index,
        group_start=group_start,
        group_end=group_end,
        scale=scale,
        mom=mom,
        prep=prep,
        out=out,
        dtype=dtype,
        casting=casting,
        order=order,
        axis_neg=axis_neg,
        parallel=parallel,
        fastpath=True,
    )


def _reduce_vals_indexed(
    # x, w, *y
    *args: NDArrayAny,
    index: NDArrayAny,
    group_start: NDArrayAny,
    group_end: NDArrayAny,
    scale: ArrayLike | None,
    mom: MomentsStrict,
    prep: PrepareValsArray,
    axis_neg: int,
    out: NDArrayAny | None,
    dtype: DTypeLike,
    casting: Casting,
    order: ArrayOrderCF,
    parallel: bool | None,
    fastpath: bool = False,
) -> NDArrayAny:
    dtype = select_dtype(args[0], out=out, dtype=dtype, fastpath=fastpath)

    if scale is None:
        scale = np.broadcast_to(np.dtype(dtype).type(1), index.shape)
    else:
        scale = asarray_maybe_recast(scale, dtype=dtype, recast=False)
        raise_if_wrong_value(
            len(scale), len(index), "`scale` and `index` must have same length."
        )

    out, axis_sample_out = prep.out_from_values(
        out,
        val_shape=prep.get_val_shape(*args),
        mom=mom,
        axis_neg=axis_neg,
        axis_new_size=len(group_start),
        dtype=dtype,
        order=order,
    )
    out.fill(0.0)

    axes: AxesGUFunc = [
        # out
        (axis_sample_out, *prep.mom_params.axes),
        # index,start,end,scale,
        *((-1,),) * 4,
        # x, weight, *y
        *get_axes_from_values(*args, axis_neg=axis_neg),
    ]

    _ = factory_reduce_vals_indexed(
        mom_ndim=prep.mom_params.ndim,
        parallel=parallel_heuristic(parallel, size=args[0].size * prep.mom_params.ndim),
    )(
        out,
        index,
        group_start,
        group_end,
        scale,
        *args,
        axes=axes,
        casting=casting,
        order=order,
        signature=(dtype, np.int64, np.int64, np.int64) + (dtype,) * (len(args) + 1),
    )

    return out

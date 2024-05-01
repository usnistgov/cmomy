"""
Routines to perform indexed reduction (:mod:`~cmomy.indexed`)
=============================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import ArrayLike

from cmomy.typing import Moments

from .docstrings import docfiller

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike

    from .typing import ArrayOrder, Moments, NDArrayAny


@docfiller.decorate
def group_idx_to_groups_index_start_end(
    group_idx: NDArrayAny,
) -> tuple[NDArrayAny, NDArrayAny, NDArrayAny, NDArrayAny]:
    """
    Transform group_idx to iterable quantities.

    Parameters
    ----------
    {group_idx}


    Returns
    -------
    groups : ndarray
        Unique groups in `group_idx`.
    index : ndarray
        Indexing array. ``index[start[k]:end[k]]`` are the index with group
        ``groups[k]``.
    start : ndarray
        See ``index``
    end : ndarray
        See ``index``.
    """
    indexes_sorted = np.argsort(group_idx)

    group_idx_sorted = np.asarray(group_idx, dtype=np.int64)[indexes_sorted]
    groups, n_start, count = np.unique(
        group_idx_sorted, return_index=True, return_counts=True
    )
    n_end = n_start + count

    return groups, indexes_sorted, n_start, n_end


def _verify_index(
    ndat: int,
    index: NDArrayAny,
    group_start: NDArrayAny,
    group_end: NDArrayAny,
) -> None:
    def _verify(name: str, x: NDArrayAny, upper: int) -> None:
        min_ = cast(int, x.min())
        max_ = cast(int, x.max())
        if min_ < 0:
            msg = f"min({name}) = {min_} < 0"
            raise ValueError(msg)
        if max_ >= upper:
            msg = f"max({name}) = {max_} > {upper}"
            raise ValueError(msg)

    nindex = len(index)
    if len(index) == 0:
        if (group_start != 0).any() or (group_end != 0).any():
            msg = "no index start must equal end"
            raise ValueError(msg)
    else:
        _verify("index", index, ndat)
        _verify("group_start", group_start, nindex)
        _verify("group_end", group_end, nindex + 1)

    if len(group_start) != len(group_end):
        msg = "len(start) != len(end)"
        raise ValueError(msg)

    if (group_end < group_start).any():
        msg = "Found end < start"
        raise ValueError(msg)


def reduce_by_index(  # noqa: PLR0914
    data: NDArrayAny,
    mom: Moments,
    index: NDArrayAny,
    group_start: NDArrayAny,
    group_end: NDArrayAny,
    scales: NDArrayAny | None = None,
    verify: bool = True,
    axis: int = 0,
    dtype: DTypeLike | None = None,
    order: ArrayOrder = None,
    parallel: bool = True,
    out: NDArrayAny | None = None,
) -> NDArrayAny:
    """
    Low level reduce by index.

    This assumes reducing along first dimension of ``data``.
    """
    from ._lib.indexed import factory_indexed_push_data

    # TODO(wpk): Can refactor this...
    # Overlaps with resample.resample_data

    dtype = dtype or data.dtype

    if isinstance(mom, int):
        mom = (mom,)

    # get data in correct form
    ndim = data.ndim - len(mom)
    if axis < 0:
        axis += ndim
    if not 0 <= axis < ndim:  # pragma: no cover
        raise ValueError
    if axis != 0:
        data = np.moveaxis(data, axis, 0)

    if verify:
        _verify_index(data.shape[0], index, group_start, group_end)

    # shapes
    ngroups = len(group_start)
    meta_shape: tuple[int, ...] = data.shape[1 : -len(mom)]
    mom_shape: tuple[int, ...] = tuple(x + 1 for x in mom)
    out_shape: tuple[int, ...] = (ngroups, *data.shape[1:])

    # default to using "C" order
    order = order or "C"
    if out is None:
        out = np.empty(out_shape, dtype=dtype, order="C")
    elif out.shape != out_shape:
        msg = f"{out.shape=} != {out_shape}"
        raise ValueError(msg)
    else:
        out = np.asarray(out, dtype=dtype, order="C")

    meta_reshape = (np.prod(meta_shape),) if meta_shape else ()
    datar = data.reshape((data.shape[0], *meta_reshape, *mom_shape))
    outr = out.reshape((ngroups, *meta_reshape, *mom_shape))

    if scales is None:
        scales = np.ones_like(index, dtype=dtype or data.dtype)

    outr.fill(0.0)
    factory_indexed_push_data(
        cov=len(mom) > 1,
        vec=len(meta_shape) > 0,
        parallel=parallel,
    )(datar, index, group_start, group_end, scales, outr)

    return outr.reshape(out.shape)


def reduce_by_group_idx(
    data: NDArrayAny,
    mom: Moments,
    group_idx: ArrayLike,
    scales: NDArrayAny | None = None,
    axis: int = 0,
    dtype: DTypeLike | None = None,
    order: ArrayOrder = None,
    parallel: bool = True,
    out: NDArrayAny | None = None,
) -> tuple[NDArrayAny, NDArrayAny]:
    """
    Reduce central moments data by group.

    Parameters
    ----------
    data : array-like
        Central moments array.
    {group_idx}
    {mom}
    axis : int
        Axis to reduce along.
    """
    group_idx = np.asarray(group_idx, dtype=np.int64, order=order)
    if len(group_idx) != data.shape[axis]:
        msg = f"{len(group_idx)=} != {data.shape[axis]=}"
        raise ValueError(msg)

    groups, index, group_start, group_end = group_idx_to_groups_index_start_end(
        group_idx
    )

    out = reduce_by_index(
        data=data,
        mom=mom,
        index=index,
        group_start=group_start,
        group_end=group_end,
        scales=scales,
        axis=axis,
        dtype=dtype,
        order=order,
        parallel=parallel,
        out=out,
    )

    return groups, out


# * For testing purposes
def resample_data(
    data: NDArrayAny,
    freq: NDArrayAny,
    mom: Moments,
    axis: int = 0,
    dtype: DTypeLike | None = None,
    order: ArrayOrder = None,
    parallel: bool = True,
    out: NDArrayAny | None = None,
) -> NDArrayAny:
    """Resample using indexed reduction."""
    from ._lib.indexed import freq_to_index_start_end_scales

    index, start, end, scales = freq_to_index_start_end_scales(freq)

    return reduce_by_index(
        data=data,
        mom=mom,
        index=index,
        group_start=start,
        group_end=end,
        scales=scales,
        axis=axis,
        dtype=dtype,
        order=order,
        parallel=parallel,
        out=out,
    )

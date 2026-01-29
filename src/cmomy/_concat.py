from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

import numpy as np
import xarray as xr

from .core.docstrings import docfiller
from .core.missing import MISSING
from .core.moment_params import (
    default_mom_params_xarray,
)
from .core.utils import (
    peek_at,
)
from .core.validate import (
    is_ndarray,
    is_xarray_typevar,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
    )
    from typing import (
        Any,
    )

    from numpy.typing import NDArray

    from .core.typing import (
        AxisReduce,
        DataT,
        DimsReduce,
        MissingType,
    )
    from .core.typing_compat import TypeVar
    from .wrapper._wrapper_abc import CentralMomentsABC

    _CentralMomentsT = TypeVar("_CentralMomentsT", bound=CentralMomentsABC[Any, Any])
    _NDArrayT = TypeVar("_NDArrayT", bound=NDArray[Any])


@overload
def concat(
    arrays: Iterable[_CentralMomentsT],
    *,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    **kwargs: Any,
) -> _CentralMomentsT: ...
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
    arrays: Iterable[_NDArrayT],
    *,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    **kwargs: Any,
) -> _NDArrayT: ...


@docfiller.decorate
def concat(
    arrays: Iterable[_NDArrayT] | Iterable[DataT] | Iterable[_CentralMomentsT],
    *,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    **kwargs: Any,
) -> _NDArrayT | DataT | _CentralMomentsT:
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
        return np.concatenate(  # type: ignore[return-value]  # pylint: disable=unexpected-keyword-arg  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
            tuple(arrays_iter),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
            axis=axis,
            dtype=first.dtype,
            **kwargs,
        )

    if is_xarray_typevar["DataT"].check(first):
        if dim is MISSING or dim is None or dim in first.dims:
            axis, dim = default_mom_params_xarray.select_axis_dim(
                first, axis=axis, dim=dim, default_axis=0
            )
        # otherwise, assume adding a new dimension...
        return cast("DataT", xr.concat(tuple(arrays_iter), dim=dim, **kwargs))  # type: ignore[arg-type]  # pyright: ignore[reportCallIssue,reportArgumentType]

    return type(first)(  # type: ignore[call-arg, return-value]  # pyright: ignore[reportCallIssue]
        concat(
            (c.obj for c in arrays_iter),  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue, reportUnknownArgumentType, reportUnknownMemberType]
            axis=axis,
            dim=dim,
            **kwargs,
        ),
        mom_ndim=first.mom_ndim,  # type: ignore[attr-defined]  # pyright: ignore[reportCallIssue]
    )

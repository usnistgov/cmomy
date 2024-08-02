"""
Conversion routines (:mod:`~cmomy.convert`)
===========================================
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast, overload

import numpy as np
import xarray as xr

from .core.docstrings import docfiller
from .core.missing import MISSING
from .core.prepare import (
    prepare_data_for_reduction,
)
from .core.utils import (
    axes_data_reduction,
    mom_to_mom_shape,
    peek_at,
    select_axis_dim,
    select_dtype,
)
from .core.validate import (
    validate_mom_and_mom_ndim,
    validate_mom_dims,
    validate_mom_ndim,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from ._central_dataarray import xCentralMoments
    from ._central_numpy import CentralMoments
    from .core.typing import (
        ArrayLikeArg,
        AxisReduce,
        ConvertStyle,
        DimsReduce,
        DTypeLikeArg,
        FloatT,
        KeepAttrs,
        MissingType,
        Mom_NDim,
        MomDims,
        Moments,
        NDArrayAny,
        ScalarT,
    )


# * Convert between raw and central moments
@overload
def moments_type(
    values_in: xr.DataArray,
    *,
    mom_ndim: Mom_NDim,
    to: ConvertStyle = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
) -> xr.DataArray: ...
# array
@overload
def moments_type(
    values_in: ArrayLikeArg[FloatT],
    *,
    mom_ndim: Mom_NDim,
    to: ConvertStyle = ...,
    out: None = ...,
    dtype: None = ...,
) -> NDArray[FloatT]: ...
# out
@overload
def moments_type(
    values_in: Any,
    *,
    mom_ndim: Mom_NDim,
    to: ConvertStyle = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def moments_type(
    values_in: Any,
    *,
    mom_ndim: Mom_NDim,
    to: ConvertStyle = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
) -> NDArray[FloatT]: ...
# fallback
@overload
def moments_type(
    values_in: Any,
    *,
    mom_ndim: Mom_NDim,
    to: ConvertStyle = ...,
    out: None = ...,
    dtype: DTypeLike = ...,
) -> NDArrayAny: ...


@docfiller.decorate
def moments_type(
    values_in: ArrayLike | xr.DataArray,
    *,
    mom_ndim: Mom_NDim,
    to: ConvertStyle = "central",
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
) -> NDArrayAny | xr.DataArray:
    r"""
    Convert between central and raw moments type.

    Parameters
    ----------
    values_in : array-like or DataArray
        The moments array to convert from.
    {mom_ndim}
    to : {{"raw", "central"}}
        The style of the ``values_in`` to convert to. If ``"raw"``, convert from central to raw.
        If ``"central"`` convert from raw to central moments.
    {out}
    {dtype}

    Returns
    -------
    ndarray
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
    if isinstance(values_in, xr.DataArray):
        return values_in.copy(
            data=moments_type(
                values_in.to_numpy(),  # pyright: ignore[reportUnknownMemberType]
                mom_ndim=mom_ndim,
                to=to,
                out=out,
                dtype=dtype,
            )
        )

    mom_ndim = validate_mom_ndim(mom_ndim)
    dtype = select_dtype(values_in, out=out, dtype=dtype)
    values_in = np.asarray(values_in, dtype=dtype)

    from ._lib.factory import factory_convert

    return factory_convert(mom_ndim=mom_ndim, to=to)(values_in, out=out)


# * Vals -> Data
@overload
def vals_to_data(
    x: xr.DataArray,
    *y: ArrayLike | xr.DataArray,
    mom: Moments,
    weight: ArrayLike | xr.DataArray | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | xr.DataArray | None = ...,
    mom_dims: MomDims | None = ...,
) -> xr.DataArray: ...
# Array
@overload
def vals_to_data(
    x: ArrayLikeArg[FloatT],
    *y: ArrayLike,
    mom: Moments,
    weight: ArrayLike | None = ...,
    dtype: None = ...,
    out: None = ...,
    mom_dims: MomDims | None = ...,
) -> NDArray[FloatT]: ...
# out
@overload
def vals_to_data(
    x: ArrayLike,
    *y: ArrayLike,
    mom: Moments,
    weight: ArrayLike | None = ...,
    dtype: DTypeLike = ...,
    out: NDArray[FloatT],
    mom_dims: MomDims | None = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def vals_to_data(
    x: ArrayLike,
    *y: ArrayLike,
    mom: Moments,
    weight: ArrayLike | None = ...,
    dtype: DTypeLikeArg[FloatT],
    out: None = ...,
    mom_dims: MomDims | None = ...,
) -> NDArray[FloatT]: ...
# fallback
@overload
def vals_to_data(
    x: ArrayLike,
    *y: ArrayLike,
    mom: Moments,
    weight: ArrayLike | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
    mom_dims: MomDims | None = ...,
) -> NDArrayAny: ...


def vals_to_data(
    x: ArrayLike | xr.DataArray,
    *y: ArrayLike | xr.DataArray,
    mom: Moments,
    weight: ArrayLike | xr.DataArray | None = None,
    dtype: DTypeLike = None,
    out: NDArrayAny | xr.DataArray | None = None,
    mom_dims: MomDims | None = None,
) -> NDArrayAny | xr.DataArray:
    """
    Convert `values` to `central moments array`.

    This allows passing `values` based observations to `data` routines.
    See examples below for more details

    Parameters
    ----------
    x : array-like or DataArray
        First value.
    *y : array-like or DataArray
        Secondary value (if comoments).
    {mom}
    {weight}
    {dtype}
    {out}

    Returns
    -------
    data : ndarray or DataArray

    Notes
    -----
    Values ``x``, ``y`` and ``weight`` must be broadcastable.

    Examples
    --------
    >>> w = np.full((2), 0.1)
    >>> x = np.full((1, 2), 0.2)
    >>> out = vals_to_data(x, weight=w, mom=2)
    >>> out.shape
    (1, 2, 3)
    >>> print(out[..., 0])
    [[0.1 0.1]]
    >>> print(out[..., 1])
    [[0.2 0.2]]
    >>> y = np.full((2, 1, 2), 0.3)
    >>> out = vals_to_data(x, y, weight=w, mom=(2, 2))
    >>> out.shape
    (2, 1, 2, 3, 3)
    >>> print(out[..., 0, 0])
    [[[0.1 0.1]]
    <BLANKLINE>
     [[0.1 0.1]]]
    >>> print(out[..., 1, 0])
    [[[0.2 0.2]]
    <BLANKLINE>
     [[0.2 0.2]]]
    >>> print(out[..., 0, 1])
    [[[0.3 0.3]]
    <BLANKLINE>
     [[0.3 0.3]]]
    """
    mom, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
    dtype = select_dtype(x, out=out, dtype=dtype)
    weight = 1.0 if weight is None else weight
    if len(y) != mom_ndim - 1:
        msg = "Supply single value for ``y`` if and only if ``mom_ndim==2``."
        raise ValueError(msg)

    if isinstance(x, xr.DataArray):
        if isinstance(out, xr.DataArray) and mom_dims is None:
            mom_dims = out.dims[-mom_ndim:]
        else:
            mom_dims = validate_mom_dims(mom_dims=mom_dims, mom_ndim=mom_ndim)
        # Do this for consistency with numpy version
        # In numpy version, ``x`` sets the right hand most dimensions.
        # xr.apply_ufunc (which calls xr.broadcast) acts left to right.
        # that is, if call a  func(x, y, z), the dimensions in ``x`` will
        # be the left most dimensions
        return xr.apply_ufunc(  # type: ignore[no-any-return]
            lambda out, weight, *args, **kwargs: vals_to_data(  # pyright: ignore[reportUnknownLambdaType]
                *args[-1::-1],  # pyright: ignore[reportUnknownArgumentType]
                weight=weight,  # pyright: ignore[reportUnknownArgumentType]
                out=out,  # pyright: ignore[reportUnknownArgumentType]
                **kwargs,  # pyright: ignore[reportUnknownArgumentType]
            ),
            out,
            weight,
            *y,
            x,
            input_core_dims=[mom_dims, *((),) * (mom_ndim + 1)],
            output_core_dims=[mom_dims],
            kwargs={
                "dtype": dtype,
                "mom": mom,
            },
        )

    x, weight, *y = (np.asarray(a, dtype=dtype) for a in (x, weight, *y))  # type: ignore[assignment]
    if out is None:
        val_shape: tuple[int, ...] = np.broadcast_shapes(
            *(_.shape for _ in (x, *y, weight))  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportAttributeAccessIssue, reportUnknownArgumentType]
        )
        out = np.zeros((*val_shape, *mom_to_mom_shape(mom)), dtype=dtype)
    else:
        out[...] = 0.0

    if mom_ndim == 1:
        out[..., 0] = weight
        out[..., 1] = x
    else:
        out[..., 0, 0] = weight
        out[..., 1, 0] = x
        out[..., 0, 1] = y[0]

    return out


# * Moments to Cumulative moments
@overload
def cumulative(  # pyright: ignore[reportOverlappingOverload]
    values_in: xr.DataArray,
    *,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    mom_ndim: Mom_NDim = ...,
    inverse: bool = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
) -> xr.DataArray: ...
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
) -> NDArray[FloatT]: ...
# out
@overload
def cumulative(
    values_in: Any,
    *,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    mom_ndim: Mom_NDim = ...,
    inverse: bool = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def cumulative(
    values_in: Any,
    *,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    mom_ndim: Mom_NDim = ...,
    inverse: bool = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
) -> NDArray[FloatT]: ...
# fallback
@overload
def cumulative(
    values_in: Any,
    *,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    mom_ndim: Mom_NDim = ...,
    inverse: bool = ...,
    move_axis_to_end: bool = ...,
    parallel: bool | None = ...,
    out: Any = ...,
    dtype: Any = ...,
) -> NDArrayAny: ...


@docfiller.decorate
def cumulative(  # pyright: ignore[reportOverlappingOverload]
    values_in: ArrayLike | xr.DataArray,
    *,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: Mom_NDim = 1,
    inverse: bool = False,
    move_axis_to_end: bool = False,
    parallel: bool | None = None,
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
) -> NDArrayAny | xr.DataArray:
    """
    Convert between moments array and cumulative moments array.

    Parameters
    ----------
    values_in : array-like or DataArray
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
    if isinstance(values_in, xr.DataArray):
        axis, dim = select_axis_dim(values_in, axis=axis, dim=dim, mom_ndim=mom_ndim)

        if move_axis_to_end:
            axis = -1
            values_in = values_in.transpose(..., dim, *values_in.dims[-mom_ndim:])

        return values_in.copy(
            data=cumulative(
                values_in.to_numpy(),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
                mom_ndim=mom_ndim,
                inverse=inverse,
                axis=axis,
                out=out,
                dtype=dtype,
                move_axis_to_end=False,
            )
        )

    mom_ndim = validate_mom_ndim(mom_ndim)
    dtype = select_dtype(values_in, out=out, dtype=dtype)

    axis, values_in = prepare_data_for_reduction(
        values_in,
        axis=axis,
        mom_ndim=mom_ndim,
        dtype=dtype,
        move_axis_to_end=move_axis_to_end,
    )

    axes = axes_data_reduction(mom_ndim=mom_ndim, axis=axis, out_has_axis=True)

    from ._lib.factory import factory_cumulative

    return factory_cumulative(
        mom_ndim=mom_ndim,
        inverse=inverse,
        parallel=parallel,
        size=values_in.size,
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
    values: xr.DataArray,
    *,
    mom: tuple[int, int],
    dtype: DTypeLike = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> xr.DataArray: ...
# array
@overload
def moments_to_comoments(
    values: ArrayLikeArg[FloatT],
    *,
    mom: tuple[int, int],
    dtype: None = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# dtype
@overload
def moments_to_comoments(
    values: Any,
    *,
    mom: tuple[int, int],
    dtype: DTypeLikeArg[FloatT],
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
# fallback
@overload
def moments_to_comoments(
    values: Any,
    *,
    mom: tuple[int, int],
    dtype: DTypeLike = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArrayAny: ...


@docfiller.decorate
def moments_to_comoments(  # pyright: ignore[reportOverlappingOverload]
    values: ArrayLike | xr.DataArray,
    *,
    mom: tuple[int, int],
    dtype: DTypeLike = None,
    mom_dims: MomDims | None = None,
    keep_attrs: KeepAttrs = None,
) -> NDArrayAny | xr.DataArray:
    """
    Convert from moments to comoments data.

    Parameters
    ----------
    values : array-like or DataArray
        Moments array with ``mom_ndim==1``. It is assumed that the last
        dimension is the moments dimension.
    {mom_moments_to_comoments}
    {dtype}
    {mom_dims}
    {keep_attrs}

    Returns
    -------
    output : ndarray or DataArray
        Co-moments array.  Same type as ``values``.


    Notes
    -----
    ``mom_dims`` and ``keep_attrs`` are used only if ``values`` is a
    :class:`~xarray.DataArray`.

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
    if isinstance(values, xr.DataArray):
        mom_dims = validate_mom_dims(mom_dims=mom_dims, mom_ndim=2)
        return xr.apply_ufunc(  # type: ignore[no-any-return]
            moments_to_comoments,
            values,
            input_core_dims=[values.dims],
            output_core_dims=[[*values.dims[:-1], *mom_dims]],
            exclude_dims={values.dims[-1]},
            kwargs={"mom": mom, "dtype": dtype},
            keep_attrs=keep_attrs,
        )

    dtype = select_dtype(values, out=None, dtype=dtype)
    values = np.asarray(values, dtype=dtype)

    mom = _validate_mom_moments_to_comoments(mom, values.shape[-1] - 1)
    out = np.empty((*values.shape[:-1], *mom_to_mom_shape(mom)), dtype=dtype)
    for i, j in np.ndindex(*out.shape[-2:]):
        out[..., i, j] = values[..., i + j]
    return out


# * Update weights
@overload
def assign_weight(
    data: xr.DataArray,
    weight: ArrayLike | xr.DataArray,
    *,
    mom_ndim: Mom_NDim,
    copy: bool = ...,
) -> xr.DataArray: ...
@overload
def assign_weight(
    data: NDArray[FloatT],
    weight: ArrayLike,
    *,
    mom_ndim: Mom_NDim,
    copy: bool = ...,
) -> NDArray[FloatT]: ...


@docfiller.decorate
def assign_weight(
    data: NDArray[FloatT] | xr.DataArray,
    weight: ArrayLike | xr.DataArray,
    *,
    mom_ndim: Mom_NDim,
    copy: bool = True,
) -> NDArray[FloatT] | xr.DataArray:
    """
    Update weights of moments array.

    Parameters
    ----------
    data : ndarray or DataArray
        Moments array.
    {weight}
    {mom_ndim}
    copy : bool, default=True
        If ``True`` (the default), return new array with updated weights.
        Otherwise, return the original array with weights updated inplace.
    """
    out = data.copy() if copy else data
    if mom_ndim == 1:
        out[..., 0] = weight
    else:
        out[..., 0, 0] = weight
    return out


# * concat
@overload
def concat(
    arrays: Iterable[xr.DataArray],
    *,
    axis: AxisReduce | MissingType = ...,
    dim: DimsReduce | MissingType = ...,
    **kwargs: Any,
) -> xr.DataArray: ...
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
    arrays: Iterable[xr.DataArray]
    | Iterable[CentralMoments[Any]]
    | Iterable[xCentralMoments[Any]]
    | Iterable[NDArrayAny],
    *,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    **kwargs: Any,
) -> xr.DataArray | CentralMoments[Any] | xCentralMoments[Any] | NDArrayAny:
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

    if isinstance(first, xr.DataArray):
        if dim is MISSING or dim is None or dim in first.dims:
            axis, dim = select_axis_dim(first, axis=axis, dim=dim, default_axis=0)
        # otherwise, assume adding a new dimension...
        return cast("xr.DataArray", xr.concat(tuple(arrays_iter), dim=dim, **kwargs))  # type: ignore[type-var]

    return type(first)(  # type: ignore[call-arg, return-value]
        concat(
            (c.to_values() for c in arrays_iter),  # type: ignore[attr-defined]
            axis=axis,
            dim=dim,
            **kwargs,
        ),
        mom_ndim=first.mom_ndim,  # type: ignore[attr-defined]
    )

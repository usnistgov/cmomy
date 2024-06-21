"""
Conversion routines (:mod:`~cmomy.convert`)
===========================================
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast, overload

import numpy as np
import xarray as xr

from ._utils import (
    MISSING,
    peek_at,
    select_axis_dim,
    select_dtype,
    validate_mom_dims,
    validate_mom_ndim,
)
from .docstrings import docfiller

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from ._central_dataarray import xCentralMoments
    from ._central_numpy import CentralMoments
    from .typing import (
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
        NDArrayAny,
        ScalarT,
    )


# * Convert between raw and central moments
@overload
def moments_type(  # type: ignore[overload-overlap]
    values_in: xr.DataArray,
    *,
    mom_ndim: Mom_NDim,
    to: ConvertStyle = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
) -> xr.DataArray: ...
@overload
def moments_type(
    values_in: ArrayLikeArg[FloatT],
    *,
    mom_ndim: Mom_NDim,
    to: ConvertStyle = ...,
    out: None = ...,
    dtype: None = ...,
) -> NDArray[FloatT]: ...
@overload
def moments_type(
    values_in: ArrayLike,
    *,
    mom_ndim: Mom_NDim,
    to: ConvertStyle = ...,
    out: None = ...,
    dtype: None = ...,
) -> NDArrayAny: ...
@overload
def moments_type(
    values_in: Any,
    *,
    mom_ndim: Mom_NDim,
    to: ConvertStyle = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
) -> NDArray[FloatT]: ...
@overload
def moments_type(
    values_in: Any,
    *,
    mom_ndim: Mom_NDim,
    to: ConvertStyle = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
) -> NDArray[FloatT]: ...
@overload
def moments_type(
    values_in: Any,
    *,
    mom_ndim: Mom_NDim,
    to: ConvertStyle = ...,
    out: None = ...,
    dtype: DTypeLike,
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


    Parameters
    ----------
    values_in : ndarray
        The moments array to convert from.
    {mom_ndim}
    to : {{"raw", "central"}}
        The style of the ``values_in`` to convert to. If ``"raw"``, convert from central to raw.
        If ``"central"`` convert from raw to central moments.
    {out}

    Returns
    -------
    ndarray
        Moments array converted from ``input_style`` to opposite format.
    """
    from ._lib.factory import factory_convert

    mom_ndim = validate_mom_ndim(mom_ndim)

    _input_to_output = factory_convert(mom_ndim=mom_ndim, to=to)

    dtype = select_dtype(values_in, out=out, dtype=dtype)

    if isinstance(values_in, xr.DataArray):
        return values_in.copy(
            data=moments_type(
                values_in.to_numpy().astype(dtype),  # pyright: ignore[reportUnknownMemberType]
                mom_ndim=mom_ndim,
                to=to,
                out=out,
            )
        )

    values_in = np.asarray(values_in, dtype=dtype)
    if out is not None:
        return _input_to_output(values_in, out)
    return _input_to_output(values_in)


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


def _moments_to_comoments(
    values: NDArrayAny,
    mom: tuple[int, int],
    dtype: DTypeLikeArg[FloatT],
) -> NDArray[FloatT]:
    mom = _validate_mom_moments_to_comoments(mom, values.shape[-1] - 1)
    out = np.empty((*values.shape[:-1], *(m + 1 for m in mom)), dtype=dtype)
    for i, j in np.ndindex(*out.shape[-2:]):
        out[..., i, j] = values[..., i + j]
    return out


@overload
def moments_to_comoments(  # type: ignore[overload-overlap]
    values: xr.DataArray,
    *,
    mom: tuple[int, int],
    dtype: DTypeLike = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> xr.DataArray: ...
@overload
def moments_to_comoments(
    values: ArrayLikeArg[FloatT],
    *,
    mom: tuple[int, int],
    dtype: None = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
@overload
def moments_to_comoments(
    values: ArrayLike,
    *,
    mom: tuple[int, int],
    dtype: None = ...,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArrayAny: ...
@overload
def moments_to_comoments(
    values: Any,
    *,
    mom: tuple[int, int],
    dtype: DTypeLikeArg[FloatT],
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArray[FloatT]: ...
@overload
def moments_to_comoments(
    values: Any,
    *,
    mom: tuple[int, int],
    dtype: DTypeLike,
    mom_dims: MomDims | None = ...,
    keep_attrs: KeepAttrs = ...,
) -> NDArrayAny: ...


@docfiller.decorate
def moments_to_comoments(
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
        dtype = values.dtype if dtype is None else dtype  # pyright: ignore[reportUnknownMemberType]

        return xr.apply_ufunc(  # type: ignore[no-any-return]
            _moments_to_comoments,
            values,
            input_core_dims=[values.dims],
            output_core_dims=[[*values.dims[:-1], *mom_dims]],
            exclude_dims={values.dims[-1]},
            kwargs={"mom": mom, "dtype": dtype},
            keep_attrs=keep_attrs,
        )

    values = np.asarray(values)
    dtype = values.dtype if dtype is None else dtype
    return _moments_to_comoments(values, mom, dtype)  # type: ignore[arg-type]


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
            axis, dim = select_axis_dim(first.dims, axis=axis, dim=dim, default_axis=0)
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

"""
Conversion routines (:mod:`~cmomy.convert`)
===========================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, overload

import numpy as np
import xarray as xr

from ._utils import select_dtype, validate_mom_dims, validate_mom_ndim
from .docstrings import docfiller

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from cmomy.typing import NDArrayAny

    from .typing import (
        ArrayLikeArg,
        ConvertStyle,
        DTypeLikeArg,
        FloatT,
        KeepAttrs,
        Mom_NDim,
        MomDims,
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

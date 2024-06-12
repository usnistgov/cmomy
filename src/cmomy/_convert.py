"""
Convert between raw and central moments
=======================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr

from ._utils import select_dtype, validate_mom_ndim
from .docstrings import docfiller

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from cmomy.typing import NDArrayAny

    from .typing import ArrayLikeArg, ConvertStyle, DTypeLikeArg, FloatT


@overload
def convert(  # type: ignore[overload-overlap]
    values_in: xr.DataArray,
    *,
    mom_ndim: int,
    to: ConvertStyle = ...,
    out: NDArrayAny | None = ...,
    dtype: DTypeLike = ...,
) -> xr.DataArray: ...
@overload
def convert(
    values_in: ArrayLikeArg[FloatT],
    *,
    mom_ndim: int,
    to: ConvertStyle = ...,
    out: None = ...,
    dtype: None = ...,
) -> NDArray[FloatT]: ...
@overload
def convert(
    values_in: ArrayLike,
    *,
    mom_ndim: int,
    to: ConvertStyle = ...,
    out: None = ...,
    dtype: None = ...,
) -> NDArrayAny: ...
@overload
def convert(
    values_in: Any,
    *,
    mom_ndim: int,
    to: ConvertStyle = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
) -> NDArray[FloatT]: ...
@overload
def convert(
    values_in: Any,
    *,
    mom_ndim: int,
    to: ConvertStyle = ...,
    out: None = ...,
    dtype: DTypeLikeArg[FloatT],
) -> NDArray[FloatT]: ...
@overload
def convert(
    values_in: Any,
    *,
    mom_ndim: int,
    to: ConvertStyle = ...,
    out: None = ...,
    dtype: DTypeLike,
) -> NDArrayAny: ...


@docfiller.decorate
def convert(
    values_in: ArrayLike | xr.DataArray,
    *,
    mom_ndim: int,
    to: ConvertStyle = "central",
    out: NDArrayAny | None = None,
    dtype: DTypeLike = None,
) -> NDArrayAny | xr.DataArray:
    r"""
    Convert between central and raw moments.

    The structure of arrays are as follow.
    Central moments:

    * ``x[..., 0]`` : weight
    * ``x[..., 1]`` : mean
    * ``x[..., k]`` : kth central moment

    Central comoments of variables `a` and `b`:

    * ``x[..., 0, 0]`` : weight
    * ``x[..., 1, 0]`` : mean of `a`
    * ``x[....,0, 1]`` : mean of `b`
    * ``x[..., i, j]`` : :math:`\langle (\delta a)^i (\delta b)^j \rangle`,

    where `a` and `b` are the variables being considered.

    Raw moments array:

    * ``x[..., 0]`` : weight
    * ``x[..., k]`` : kth moment :math:`\langle a^k \rangle`

    Raw comoments array of variables `a` and `b`:

    * ``x[..., 0, 0]`` : weight
    * ``x[..., i, j]`` : :math:`\langle a^i b^j \rangle`,


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
            data=convert(
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

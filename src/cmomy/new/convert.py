"""
Routines to perform central moments reduction (:mod:`~cmomy.reduce)
===================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from cmomy.new.utils import validate_mom_ndim

from .docstrings import docfiller

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .typing import ConvertStyle, Mom_NDim, T_Array
    from .typing import T_FloatDType as T_Float


@docfiller.decorate
def convert(
    values_in: T_Array,
    mom_ndim: Mom_NDim,
    to: ConvertStyle = "central",
    out: NDArray[T_Float] | None = None,
) -> T_Array:
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

    if isinstance(values_in, xr.DataArray):
        return values_in.copy(
            data=convert(values_in.to_numpy(), mom_ndim=mom_ndim, to=to, out=out)  # type: ignore[arg-type]
        )

    if out is not None:
        return _input_to_output(values_in, out)  # type: ignore[arg-type]
    return _input_to_output(values_in)

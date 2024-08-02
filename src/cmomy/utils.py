"""
Interface to utility functions (:mod:`cmomy.utils`)
===================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr

from ._missing import MISSING
from ._utils import mom_shape_to_mom as mom_shape_to_mom  # noqa: PLC0414
from ._utils import mom_to_mom_shape as mom_to_mom_shape  # noqa: PLC0414
from ._utils import (
    normalize_axis_tuple,
    select_axis_dim_mult,
)
from ._validate import (
    validate_axis_mult,
    validate_mom_ndim,
)
from .docstrings import docfiller

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Sequence,
    )

    from numpy.typing import NDArray

    from .typing import (
        MissingType,
        Mom_NDim,
        ScalarT,
    )


@overload
def moveaxis(
    x: NDArray[ScalarT],
    axis: int | tuple[int, ...] | MissingType = ...,
    dest: int | tuple[int, ...] | MissingType = ...,
    *,
    dim: str | Sequence[Hashable] | MissingType = ...,
    dest_dim: str | Sequence[Hashable] | MissingType = ...,
    mom_ndim: Mom_NDim | None = ...,
) -> NDArray[ScalarT]: ...
@overload
def moveaxis(
    x: xr.DataArray,
    axis: int | tuple[int, ...] | MissingType = ...,
    dest: int | tuple[int, ...] | MissingType = ...,
    *,
    dim: str | Sequence[Hashable] | MissingType = ...,
    dest_dim: str | Sequence[Hashable] | MissingType = ...,
    mom_ndim: Mom_NDim | None = ...,
) -> xr.DataArray: ...


@docfiller.decorate
def moveaxis(
    x: NDArray[ScalarT] | xr.DataArray,
    axis: int | tuple[int, ...] | MissingType = MISSING,
    dest: int | tuple[int, ...] | MissingType = MISSING,
    *,
    dim: str | Sequence[Hashable] | MissingType = MISSING,
    dest_dim: str | Sequence[Hashable] | MissingType = MISSING,
    mom_ndim: Mom_NDim | None = None,
) -> NDArray[ScalarT] | xr.DataArray:
    """
    Generalized moveaxis for moments arrays.

    Parameters
    ----------
    x : ndarray or DataArray
        input data
    axis : int or sequence of int
        Original positions of axes to move.
    dest : int or sequence of int
        Destination positions for each original axes.
    dim : str or sequence of hashable
        Original dimensions to move (for DataArray).
    dest_dim : str or sequence of hashable
        Destination of each original dimension.
    {mom_ndim_optional}

    Returns
    -------
    out : ndarray or DataArray
        Same type as ``x`` with moved axis.

    Notes
    -----
    Must specify either ``axis`` or ``dim`` and either ``dest`` or
    ``dest_dim``.

    See Also
    --------
    numpy.moveaxis

    Examples
    --------
    >>> x = np.zeros((2, 3, 4, 5))
    >>> moveaxis(x, 0, -1).shape
    (3, 4, 5, 2)

    Specifying ``mom_ndim`` will result in negative axis relative
    to the moments dimensions.

    >>> moveaxis(x, 0, -1, mom_ndim=1).shape
    (3, 4, 2, 5)

    Multiple axes can also be specified.

    >>> moveaxis(x, (1, 0), (-2, -1), mom_ndim=1).shape
    (4, 3, 2, 5)

    This also works with dataarrays

    >>> dx = xr.DataArray(x, dims=["a", "b", "c", "mom_0"])
    >>> moveaxis(dx, dim="a", dest=-1, mom_ndim=1).dims
    ('b', 'c', 'a', 'mom_0')
    """
    mom_ndim = None if mom_ndim is None else validate_mom_ndim(mom_ndim)

    if isinstance(x, xr.DataArray):
        axes0, dims0 = select_axis_dim_mult(x, axis=axis, dim=dim, mom_ndim=mom_ndim)
        axes1, dims1 = select_axis_dim_mult(
            x, axis=dest, dim=dest_dim, mom_ndim=mom_ndim
        )

        if len(dims0) != len(dims1):
            msg = "`dim` and `dest_dim` must have same length"
            raise ValueError(msg)

        order = [n for n in range(x.ndim) if n not in axes0]
        for dst, src in sorted(zip(axes1, axes0)):
            order.insert(dst, src)
        return x.transpose(*(x.dims[o] for o in order))

    axes0 = normalize_axis_tuple(validate_axis_mult(axis), x.ndim, mom_ndim)
    axes1 = normalize_axis_tuple(validate_axis_mult(dest), x.ndim, mom_ndim)

    return np.moveaxis(x, axes0, axes1)

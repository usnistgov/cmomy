"""
Interface to utility functions (:mod:`cmomy.utils`)
===================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr

from .core.array_utils import normalize_axis_tuple
from .core.docstrings import docfiller
from .core.missing import MISSING
from .core.utils import mom_shape_to_mom as mom_shape_to_mom  # noqa: PLC0414
from .core.utils import mom_to_mom_shape as mom_to_mom_shape  # noqa: PLC0414
from .core.validate import (
    validate_axis_mult,
    validate_mom_ndim,
)
from .core.xr_utils import select_axis_dim_mult

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Sequence,
    )

    from numpy.typing import NDArray

    from .core.typing import (
        MissingType,
        Mom_NDim,
        ScalarT,
        SelectMoment,
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


# * Selecting subsets of data -------------------------------------------------
@overload
def select_moment(
    data: xr.DataArray,
    name: SelectMoment,
    mom_ndim: Mom_NDim,
    *,
    dim_combined: str = ...,
    coords_combined: str | Sequence[Hashable] | None = ...,
    keep_attrs: bool | None = ...,
) -> xr.DataArray: ...
@overload
def select_moment(
    data: NDArray[ScalarT],
    name: SelectMoment,
    mom_ndim: Mom_NDim,
    *,
    dim_combined: str = ...,
    coords_combined: str | Sequence[Hashable] | None = ...,
    keep_attrs: bool | None = ...,
) -> NDArray[ScalarT]: ...


def select_moment(  # noqa: C901, PLR0912
    data: xr.DataArray | NDArray[ScalarT],
    name: SelectMoment,
    mom_ndim: Mom_NDim,
    *,
    squeeze: bool = False,
    dim_combined: str = "variable",
    coords_combined: str | Sequence[Hashable] | None = None,
    keep_attrs: bool | None = None,
) -> xr.DataArray | NDArray[ScalarT]:
    """
    Select specific moments for a central moments array.

    Parameters
    ----------
    {data}
    {mom_ndim}
    name : {"weight", "ave", "var", "cov", "xave", "xvar", "yave", "yvar"}
        Name of moment(s) to select.

        - ``"weight"`` : weights
        - ``"ave"`` : Averages.
        - ``"var"``: Variance. The last dimension is of size ``mom_ndim``.
        - ``"cov"``: Covariance if ``mom_ndim == 2``, or variace if ``mom_ndim == 1``.
        - ``"xave"``: Average of first variable.
        - ``"yave"``: Average of second variable (if ``mom_ndim == 2``).
        - ``"xvar"``: Variance of first variable.
        - ``"yvar"``: Variace of second variable (if ``mom_ndim == 2``).

        Moments ``"weight", "xave", "yave", "xvar", "yvar", "cov"`` will have
        shape ``data.shape[:-mom_ndim]``. Moments ``"ave", "var"`` result in
        output of shape ``(*data.shape[:-mom_ndim], mom_ndim)``, unless
        ``mom_ndim == 1`` and ``squeeze = True``.

    squeeze : bool, default=False
        If True, squeeze last dimension if ``name`` is one of ``ave`` or ``var`` and ``mom_ndim == 1``.
    dim_combined: str, optional
        Name of new dimension for options ``name`` that can select multiple dimensions.
    coords_combined: str or sequence of str, optional
        Coordates to assign to ``dim_combined``.  Defaults to names of moments dimension(s)
    {keep_attrs}

    Returns
    -------
    output : ndarray or DataArray
        Same type as ``data``. If ``name`` is ``ave`` or ``var``, the last
        dimensions of ``output`` has shape ``mom_ndim`` with each element
        corresponding to the `ith` variable. Otherwise, ``output.shape ==
        data.shape[:-mom_ndim]``.  Note that ``output`` may be a view of ``data``.


    Examples
    --------
    >>> data = np.arange(2 * 3).reshape(2, 3)
    >>> select_moment(data, "weight", 1)
    array([0, 3])
    >>> select_moment(data, "ave", 1)
    array([[1],
           [4]])

    Note that by default, selecting ``ave`` and ``var`` will result in the last dimension having
    size ``mom_ndim``.  If ``squeeze = True`` is passed and ``mom_ndim==1``, this dimension will be removed

    >>> select_moment(data, "ave", 1, squeeze=True)
    array([1, 4])

    >>> select_moment(data, "xave", 2)
    array(3)
    >>> select_moment(data, "cov", 2)
    array(4)
    """
    if isinstance(data, xr.DataArray):
        input_core_dims = [data.dims]
        if name in {"ave", "var"} and (mom_ndim != 1 or not squeeze):
            output_core_dims = [(*data.dims[:-mom_ndim], dim_combined)]
            if coords_combined is None:
                coords_combined = data.dims[-mom_ndim:]
            elif isinstance(coords_combined, str):
                coords_combined = [coords_combined]
            if len(coords_combined) != mom_ndim:
                msg = f"{len(coords_combined)=} must equal {mom_ndim=}"
                raise ValueError(msg)

        else:
            output_core_dims = [data.dims[:-mom_ndim]]
            coords_combined = None

        xout: xr.DataArray = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            select_moment,
            data,
            input_core_dims=input_core_dims,
            output_core_dims=output_core_dims,
            kwargs={
                "name": name,
                "mom_ndim": mom_ndim,
                "squeeze": squeeze,
            },
            keep_attrs=keep_attrs,
        )

        if coords_combined is not None and dim_combined in xout.dims:
            xout = xout.assign_coords(  # pyright: ignore[reportUnknownMemberType]
                {dim_combined: (dim_combined, list(coords_combined))}
            )
        return xout

    mom_ndim = validate_mom_ndim(mom_ndim)
    if data.ndim < mom_ndim:
        msg = f"{data.ndim=} must be >= {mom_ndim=}"
        raise ValueError(msg)

    idx: tuple[int, ...] | tuple[list[int], ...]
    if name == "weight":
        idx = (0,) if mom_ndim == 1 else (0, 0)
    elif name == "ave":
        idx = ((1,) if squeeze else ([1],)) if mom_ndim == 1 else ([1, 0], [0, 1])
    elif name == "var":
        idx = ((2,) if squeeze else ([2],)) if mom_ndim == 1 else ([2, 0], [0, 2])
    elif name == "cov":
        idx = (2,) if mom_ndim == 1 else (1, 1)
    elif name == "xave":
        idx = (1,) if mom_ndim == 1 else (1, 0)
    elif name == "xvar":
        idx = (2,) if mom_ndim == 1 else (2, 0)

    else:
        if name == "yave":
            idx = (0, 1)
        elif name == "yvar":
            idx = (0, 2)
        else:
            msg = f"Unknown option {name}."
            raise ValueError(msg)
        if mom_ndim != 2:
            msg = f"{name} requires mom_ndim == 2"
            raise ValueError(msg)

    return data[(..., *idx)]

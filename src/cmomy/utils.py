"""
Interface to utility functions (:mod:`cmomy.utils`)
===================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr

from .core.array_utils import normalize_axis_tuple, select_dtype
from .core.docstrings import docfiller
from .core.missing import MISSING
from .core.utils import mom_shape_to_mom as mom_shape_to_mom  # noqa: PLC0414
from .core.utils import mom_to_mom_shape as mom_to_mom_shape  # noqa: PLC0414
from .core.validate import (
    validate_axis_mult,
    validate_mom_and_mom_ndim,
    validate_mom_dims,
    validate_mom_ndim,
)
from .core.xr_utils import select_axis_dim_mult

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Sequence,
    )

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from .core.typing import (
        ArrayLikeArg,
        DTypeLikeArg,
        FloatT,
        KeepAttrs,
        MissingType,
        Mom_NDim,
        MomDims,
        Moments,
        NDArrayAny,
        ScalarT,
        SelectMoment,
    )
    from .core.typing_compat import EllipsisType


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
@docfiller.decorate
def moment_indexer(
    name: SelectMoment | str, mom_ndim: Mom_NDim, squeeze: bool = True
) -> tuple[EllipsisType | int | list[int], ...]:
    """
    Get indexer for moments

    Parameters
    ----------
    {select_name}
    {mom_ndim}
    {select_squeeze}

    Returns
    -------
    indexer : tuple
    """
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
    return (..., *idx)


@overload
def select_moment(
    data: xr.DataArray,
    name: SelectMoment,
    *,
    mom_ndim: Mom_NDim,
    dim_combined: str = ...,
    coords_combined: str | Sequence[Hashable] | None = ...,
    keep_attrs: bool | None = ...,
) -> xr.DataArray: ...
@overload
def select_moment(
    data: NDArray[ScalarT],
    name: SelectMoment,
    *,
    mom_ndim: Mom_NDim,
    dim_combined: str = ...,
    coords_combined: str | Sequence[Hashable] | None = ...,
    keep_attrs: bool | None = ...,
) -> NDArray[ScalarT]: ...


@docfiller.decorate
def select_moment(
    data: xr.DataArray | NDArray[ScalarT],
    name: SelectMoment,
    *,
    mom_ndim: Mom_NDim,
    squeeze: bool = True,
    dim_combined: str = "variable",
    coords_combined: str | Sequence[Hashable] | None = None,
    keep_attrs: KeepAttrs = None,
) -> xr.DataArray | NDArray[ScalarT]:
    """
    Select specific moments for a central moments array.

    Parameters
    ----------
    {data}
    {mom_ndim}
    {select_name}
    {select_squeeze}
    {select_dim_combined}
    {select_coords_combined}
    {keep_attrs}

    Returns
    -------
    output : ndarray or DataArray
        Same type as ``data``. If ``name`` is ``ave`` or ``var``, the last
        dimensions of ``output`` has shape ``mom_ndim`` with each element
        corresponding to the `ith` variable. If ``squeeze=True`` and
        `mom_ndim==1`, this last dimension is removed. For all other ``name``
        options ``output.shape == data.shape[:-mom_ndim]``. Note that
        ``output`` may be a view of ``data``.


    Examples
    --------
    >>> data = np.arange(2 * 3).reshape(2, 3)
    >>> select_moment(data, "weight", mom_ndim=1)
    array([0, 3])
    >>> select_moment(data, "ave", mom_ndim=1)
    array([1, 4])

    Note that with ``squeeze = False ``, selecting ``ave`` and ``var`` will
    result in the last dimension having size ``mom_ndim``. If ``squeeze =
    True`` (the default) and ``mom_ndim==1``, this dimension will be removed

    >>> select_moment(data, "ave", mom_ndim=1, squeeze=False)
    array([[1],
           [4]])


    >>> select_moment(data, "xave", mom_ndim=2)
    array(3)
    >>> select_moment(data, "cov", mom_ndim=2)
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

    idx = moment_indexer(name, mom_ndim, squeeze)

    return data[idx]


# * Assign value(s)
@overload
def assign_moment(
    data: xr.DataArray,
    name: SelectMoment,
    value: ArrayLike | xr.DataArray,
    *,
    mom_ndim: Mom_NDim,
    squeeze: bool = True,
    copy: bool = True,
) -> xr.DataArray: ...
@overload
def assign_moment(
    data: NDArray[ScalarT],
    name: SelectMoment,
    value: ArrayLike | xr.DataArray,
    *,
    mom_ndim: Mom_NDim,
    squeeze: bool = True,
    copy: bool = True,
) -> NDArray[ScalarT]: ...


@docfiller.decorate
def assign_moment(
    data: xr.DataArray | NDArray[ScalarT],
    name: SelectMoment,
    value: ArrayLike | xr.DataArray,
    *,
    mom_ndim: Mom_NDim,
    squeeze: bool = True,
    copy: bool = True,
) -> xr.DataArray | NDArray[ScalarT]:
    """
    Update weights of moments array.

    Parameters
    ----------
    data : ndarray or DataArray
        Moments array.
    {select_name}
    value : array-like
        Value to assign to moment ``name``.
    {mom_ndim}
    {select_squeeze}
    copy : bool, default=True
        If ``True`` (the default), return new array with updated weights.
        Otherwise, return the original array with weights updated inplace.

    Returns
    -------
    output : ndarray or DataArray
        Same type as ``data`` with moment ``name`` updated to ``value``.

    See Also
    --------
    select_moment

    Examples
    --------
    >>> data = np.arange(3)
    >>> data
    array([0, 1, 2])

    >>> assign_moment(data, "weight", -1, mom_ndim=1)
    array([-1,  1,  2])

    >>> assign_moment(data, "ave", -1, mom_ndim=1)
    array([ 0, -1,  2])

    >>> assign_moment(data, "var", -1, mom_ndim=1)
    array([ 0,  1, -1])


    For multidimensional data, the passed ``value`` must conform to the
    selected data


    >>> data = np.arange(2 * 3).reshape(2, 3)

    Selecting ``ave`` for this data with ``mom_ndim=1`` and ``squeeze=False`` would have shape ``(2, 1)``.

    >>> assign_moment(data, "ave", np.ones((2, 1)), mom_ndim=1, squeeze=False)
    array([[0, 1, 2],
           [3, 1, 5]])

    The ``squeeze`` parameter has the same meaning as for :func:`select_moment`

    >>> assign_moment(data, "ave", np.ones(2), mom_ndim=1, squeeze=True)
    array([[0, 1, 2],
           [3, 1, 5]])

    """
    out = data.copy() if copy else data
    out[moment_indexer(name, validate_mom_ndim(mom_ndim), squeeze)] = value

    return out


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


@docfiller.decorate
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

    out = assign_moment(out, "weight", weight, mom_ndim=mom_ndim, copy=False)
    out = assign_moment(out, "xave", x, mom_ndim=mom_ndim, copy=False)

    if mom_ndim == 2:
        out = assign_moment(out, "yave", y[0], mom_ndim=mom_ndim, copy=False)

    return out

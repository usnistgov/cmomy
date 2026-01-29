"""
Interface to utility functions (:mod:`cmomy.utils`)
===================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr
from xarray.namedarray.utils import either_dict_or_kwargs

from .core.array_utils import reorder, select_dtype
from .core.docstrings import docfiller
from .core.missing import MISSING
from .core.moment_params import (
    MomParamsArray,
    MomParamsArrayOptional,
    MomParamsXArray,
    MomParamsXArrayOptional,
)
from .core.utils import (  # pylint: disable=useless-import-alias,unused-import
    mom_shape_to_mom as mom_shape_to_mom,  # noqa: PLC0414
)
from .core.utils import (  # pylint: disable=useless-import-alias
    mom_to_mom_shape as mom_to_mom_shape,  # noqa: PLC0414
)
from .core.validate import (
    is_dataarray,
    is_dataset,
    is_xarray,
    is_xarray_typevar,
    raise_if_wrong_value,
    validate_axis_mult,
)
from .core.xr_utils import (
    factory_apply_ufunc_kwargs,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Mapping,
        Sequence,
    )
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from .core._typing_kwargs import (
        ApplyUFuncKwargs,
        SelectMomentKwargs,
        ValsToDataKwargs,
    )
    from .core.moment_params import MomParamsType
    from .core.typing import (
        ArrayLikeArg,
        AxesWrap,
        DataT,
        DTypeLikeArg,
        FloatT,
        KeepAttrs,
        MissingType,
        MomAxes,
        MomDims,
        Moments,
        MomentsStrict,
        MomNDim,
        NDArrayAny,
        SelectMoment,
    )
    from .core.typing_compat import EllipsisType, TypeVar, Unpack

    _ScalarT = TypeVar("_ScalarT", bound=np.generic)


# * moveaxis ------------------------------------------------------------------
def moveaxis_order(
    x: NDArray[_ScalarT] | xr.DataArray,
    axis: AxesWrap | MissingType = MISSING,
    dest: AxesWrap | MissingType = MISSING,
    *,
    dim: str | Sequence[Hashable] | MissingType = MISSING,
    dest_dim: str | Sequence[Hashable] | MissingType = MISSING,
    mom_ndim: MomNDim | None = None,
    mom_axes: MomAxes | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsType = None,
    axes_to_end: bool = False,
    allow_select_mom_axes: bool = False,
) -> list[int]:
    """Get new integer order for transpose corresponding to `moveaxis` parameters."""
    if is_dataarray(x):
        mom_params = MomParamsXArrayOptional.factory(
            mom_params=mom_params, ndim=mom_ndim, dims=mom_dims, axes=mom_axes, data=x
        )

        if axes_to_end:
            if axis is not MISSING or dim is not MISSING:
                axis, _ = mom_params.select_axis_dim_mult(
                    x,
                    axis=axis,
                    dim=dim,
                    allow_select_mom_axes=False,
                )
                dest = tuple(a * 1j for a in range(-len(axis), 0))
            else:
                axis = dest = ()

            if mom_params.dims is not None:
                axis = (*axis, *mom_params.get_axes(x))
                dest = (*dest, *mom_params.axes_last)

            return moveaxis_order(
                x,
                axis=axis,
                dest=dest,
                mom_params=mom_params,
                allow_select_mom_axes=True,
                axes_to_end=False,
            )

        axes0, axes1 = (
            mom_params.select_axis_dim_mult(
                x,
                axis=a,
                dim=d,
                allow_select_mom_axes=allow_select_mom_axes,
            )[0]
            for a, d in zip((axis, dest), (dim, dest_dim), strict=True)
        )
        return reorder(x.ndim, axes0, axes1, normalize=False)

    # numpy
    mom_params = MomParamsArrayOptional.factory(
        mom_params=mom_params, ndim=mom_ndim, axes=mom_axes
    ).normalize_axes(x.ndim)

    if axes_to_end:
        if axis is not MISSING:
            axis = mom_params.normalize_axis_tuple(validate_axis_mult(axis), x.ndim)
            dest = tuple(a * 1j for a in range(-len(axis), 0))
        else:
            axis = dest = ()

        if mom_params.axes is not None:
            axis = (*axis, *mom_params.axes)
            dest = (*dest, *mom_params.axes_last)
        return moveaxis_order(
            x,
            axis=axis,
            dest=dest,
            mom_params=mom_params,
            allow_select_mom_axes=True,
            axes_to_end=False,
        )

    axes0, axes1 = (
        mom_params.normalize_axis_tuple(validate_axis_mult(a), x.ndim)
        for a in (axis, dest)
    )
    if not allow_select_mom_axes:
        mom_params.raise_if_in_mom_axes(*axes0, *axes1)

    return reorder(x.ndim, axes0, axes1, normalize=False)


@overload
def moveaxis(
    x: NDArray[_ScalarT],
    axis: AxesWrap | MissingType = ...,
    dest: AxesWrap | MissingType = ...,
    *,
    dim: str | Sequence[Hashable] | MissingType = ...,
    dest_dim: str | Sequence[Hashable] | MissingType = ...,
    mom_ndim: MomNDim | None = ...,
    mom_axes: MomAxes | None = ...,
    mom_dims: MomDims | None = ...,
    mom_params: MomParamsType = ...,
    axes_to_end: bool = ...,
    allow_select_mom_axes: bool = ...,
) -> NDArray[_ScalarT]: ...
@overload
def moveaxis(
    x: xr.DataArray,
    axis: AxesWrap | MissingType = ...,
    dest: AxesWrap | MissingType = ...,
    *,
    dim: str | Sequence[Hashable] | MissingType = ...,
    dest_dim: str | Sequence[Hashable] | MissingType = ...,
    mom_ndim: MomNDim | None = ...,
    mom_axes: MomAxes | None = ...,
    mom_dims: MomDims | None = ...,
    mom_params: MomParamsType = ...,
    axes_to_end: bool = ...,
    allow_select_mom_axes: bool = ...,
) -> xr.DataArray: ...


@docfiller.decorate
def moveaxis(
    x: NDArray[_ScalarT] | xr.DataArray,
    axis: AxesWrap | MissingType = MISSING,
    dest: AxesWrap | MissingType = MISSING,
    *,
    dim: str | Sequence[Hashable] | MissingType = MISSING,
    dest_dim: str | Sequence[Hashable] | MissingType = MISSING,
    mom_ndim: MomNDim | None = None,
    mom_axes: MomAxes | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsType = None,
    axes_to_end: bool = False,
    allow_select_mom_axes: bool = False,
) -> NDArray[_ScalarT] | xr.DataArray:
    """
    Generalized moveaxis for moments arrays.

    Parameters
    ----------
    x : ndarray or DataArray
        input data
    axis : complex or sequence of complex
        Original positions of axes to move.  Complex values wrapped relative to ``mom_ndim``.
    dest : complex or sequence of complex
        Destination positions for each original axes.
    dim : str or sequence of hashable
        Original dimensions to move (for DataArray).
    dest_dim : str or sequence of hashable
        Destination of each original dimension.
    {mom_ndim_optional}
    {mom_axes}
    {mom_dims_data}
    {mom_params}
    {axes_to_end}
    allow_select_mom_axes : bool, default=False
        If True, allow moving moment axes.  Otherwise, raise ``ValueError`` if try to move ``mom_axes``.

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

    Specifying ``mom_ndim`` will result in negative complex axis relative
    to the moments dimensions.

    >>> moveaxis(x, 0, -1j, mom_ndim=1).shape
    (3, 4, 2, 5)

    Note that if ``mom_ndim`` is not specified, then complex axis are equivalent to
    integer axis.

    >>> moveaxis(x, 0, -1j).shape
    (3, 4, 5, 2)

    Multiple axes can also be specified.

    >>> moveaxis(x, (1, 0), (-2j, -1j), mom_ndim=1).shape
    (4, 3, 2, 5)

    This also works with dataarrays

    >>> dx = xr.DataArray(x, dims=["a", "b", "c", "mom_0"])
    >>> moveaxis(dx, dim="a", dest=-1j, mom_ndim=1).dims
    ('b', 'c', 'a', 'mom_0')

    All the routines in ``cmomy`` accept `moment dimensions` in arbitrary locations.  However, it is often easiest
    to work with these dimensions as the last dimensions in an array.  You can easily achieve this with ``moveaxis`` with the parameter ``axes_to_end=True``.
    For example:

    >>> moveaxis(x, mom_axes=(0, 1), axes_to_end=True).shape
    (4, 5, 2, 3)

    >>> moveaxis(x, axis=-1, mom_axes=(1, 2), axes_to_end=True).shape
    (2, 5, 3, 4)

    >>> moveaxis(x, axis=-2j, mom_ndim=2, axes_to_end=True).shape
    (3, 2, 4, 5)
    """
    order = moveaxis_order(
        x,
        axis=axis,
        dest=dest,
        dim=dim,
        dest_dim=dest_dim,
        mom_ndim=mom_ndim,
        mom_axes=mom_axes,
        mom_dims=mom_dims,
        mom_params=mom_params,
        axes_to_end=axes_to_end,
        allow_select_mom_axes=allow_select_mom_axes,
    )

    if is_dataarray(x):
        return x.transpose(*(x.dims[o] for o in order))
    return x.transpose(order)


# * Selecting subsets of data -------------------------------------------------
# pylint: disable=consider-using-namedtuple-or-dataclass
_MOMENT_INDEXER_1: dict[str, tuple[int | slice, ...]] = {
    "weight": (0,),
    "cov": (2,),
    "xave": (1,),
    "xvar": (2,),
}
_MOMENT_INDEXER_2: dict[str, tuple[int | slice, ...]] = {
    "weight": (0, 0),
    "cov": (1, 1),
    "xave": (1, 0),
    "xvar": (2, 0),
    "yave": (0, 1),
    "yvar": (0, 2),
    "xmom_0": (0, slice(None)),
    "xmom_1": (1, slice(None)),
    "ymom_0": (slice(None), 0),
    "ymom_1": (slice(None), 1),
}


@docfiller.decorate
def moment_indexer(
    name: SelectMoment | str, mom_ndim: MomNDim, squeeze: bool = True
) -> tuple[EllipsisType | int | list[int] | slice, ...]:
    """
    Get indexer for moments

    Parameters
    ----------
    {select_moment_name}
    {mom_ndim}
    {select_squeeze}

    Returns
    -------
    indexer : tuple
    """
    idx: tuple[int, ...] | tuple[list[int], ...] | tuple[int | slice, ...]

    # special indexers
    if name == "all":
        idx = ()
    elif name == "ave":
        idx = ((1,) if squeeze else ([1],)) if mom_ndim == 1 else ([1, 0], [0, 1])
    elif name == "var":
        idx = ((2,) if squeeze else ([2],)) if mom_ndim == 1 else ([2, 0], [0, 2])
    else:
        indexer: dict[str, tuple[int | slice, ...]] = (
            _MOMENT_INDEXER_1 if mom_ndim == 1 else _MOMENT_INDEXER_2
        )
        if name not in indexer:
            msg = f"Unknown option {name} for {mom_ndim=}"
            raise ValueError(msg)
        idx = indexer[name]

    return (..., *idx)


@overload
def select_moment(
    data: DataT,
    name: SelectMoment,
    **kwargs: Unpack[SelectMomentKwargs],
) -> DataT: ...
@overload
def select_moment(
    data: NDArray[_ScalarT],
    name: SelectMoment,
    **kwargs: Unpack[SelectMomentKwargs],
) -> NDArray[_ScalarT]: ...


@docfiller.decorate
def select_moment(
    data: NDArray[_ScalarT] | DataT,
    name: SelectMoment,
    *,
    mom_ndim: MomNDim | None = None,
    mom_axes: MomAxes | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsType = None,
    squeeze: bool = True,
    dim_combined: str = "variable",
    coords_combined: str | Sequence[Hashable] | None = None,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArray[_ScalarT] | DataT:
    """
    Select specific moments for a central moments array.

    Parameters
    ----------
    {data}
    {mom_ndim}
    {mom_axes}
    {mom_dims_data}
    {mom_params}
    {select_moment_name}
    {select_squeeze}
    {select_dim_combined}
    {select_coords_combined}
    {keep_attrs}
    {apply_ufunc_kwargs}

    Returns
    -------
    output : ndarray or DataArray or Dataset
        Same type as ``data``. If ``name`` is ``ave`` or ``var``, the last
        dimensions of ``output`` has shape ``mom_ndim`` with each element
        corresponding to the `ith` variable. If ``squeeze=True`` and
        `mom_ndim==1`, this last dimension is removed. For all other ``name``
        options, output has shape of input with moment dimensions removed.  In all cases, ``mom_dims`` or ``mom_axes``
        are first moved to the last dimensions.


    Examples
    --------
    >>> data = np.arange(2 * 3).reshape(2, 3)
    >>> data
    array([[0, 1, 2],
           [3, 4, 5]])
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

    Use options ``"xmom_0"``, to select all values with first
    moment equal ``0``, etc

    >>> select_moment(data, "xmom_0", mom_ndim=2)
    array([0, 1, 2])
    >>> select_moment(data, "ymom_1", mom_ndim=2)
    array([1, 4])

    """
    if is_xarray_typevar["DataT"].check(data):
        if name == "all":
            return data

        mom_params = MomParamsXArray.factory(
            mom_params=mom_params,
            ndim=mom_ndim,
            dims=mom_dims,
            axes=mom_axes,
            data=data,
            default_ndim=1,
        )

        # input/output dimensions
        input_core_dims = [mom_params.dims]
        output_core_dims: list[Sequence[Hashable]]
        output_sizes: dict[Hashable, int] | None
        if name in {"ave", "var"} and (mom_params.ndim != 1 or not squeeze):
            output_core_dims = [[dim_combined]]
            output_sizes = {dim_combined: mom_params.ndim}
            if coords_combined is None:
                coords_combined = mom_params.dims
            elif isinstance(coords_combined, str):
                coords_combined = [coords_combined]

            raise_if_wrong_value(
                len(coords_combined),
                mom_params.ndim,
                "`len(coords_combined)` must equal `mom_ndim`.",
            )
        else:
            output_sizes = None
            coords_combined = None
            if name.startswith("xmom_"):
                output_core_dims = [mom_params.dims[1:]]
            elif name.startswith("ymom_"):
                output_core_dims = [mom_params.dims[:1]]
            else:
                output_core_dims = [[]]

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _select_moment,
            data,
            input_core_dims=input_core_dims,
            output_core_dims=output_core_dims,
            kwargs={
                "mom_params": mom_params.to_array(),
                "name": name,
                "squeeze": squeeze,
            },
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_sizes=output_sizes,
                output_dtypes=data.dtype
                if is_dataarray(data)  # type: ignore[redundant-expr]
                else np.float64,
            ),
        )
        if coords_combined is not None and dim_combined in xout.dims:
            xout = xout.assign_coords(  # pyright: ignore[reportUnknownMemberType]
                {dim_combined: (dim_combined, list(coords_combined))}
            )
        return xout

    assert isinstance(data, np.ndarray)  # noqa: S101
    mom_params = MomParamsArray.factory(
        mom_params=mom_params, ndim=mom_ndim, axes=mom_axes, default_ndim=1
    )
    mom_params_end = mom_params.axes_to_end()
    if mom_params.axes != mom_params_end.axes:
        data = np.moveaxis(data, mom_params.axes, mom_params_end.axes)
        mom_params = mom_params_end

    return _select_moment(
        data,
        mom_params=mom_params,
        name=name,
        squeeze=squeeze,
    )


def _select_moment(
    data: NDArray[_ScalarT],
    *,
    mom_params: MomParamsArray,
    name: SelectMoment,
    squeeze: bool,
) -> NDArray[_ScalarT]:
    if data.ndim < mom_params.ndim:
        msg = f"{data.ndim=} must be >= {mom_params.ndim=}"
        raise ValueError(msg)
    idx = moment_indexer(name, mom_params.ndim, squeeze)
    return data[idx]


# * Assign value(s) -----------------------------------------------------------
# NOTE: Can't do kwargs trick used elsewhere, because want to be
# able to use **moments_kwargs....
@overload
def assign_moment(
    data: DataT,
    moment: Mapping[SelectMoment, ArrayLike | xr.DataArray | DataT] | None = None,
    *,
    mom_ndim: MomNDim | None = ...,
    mom_axes: MomAxes | None = ...,
    mom_dims: MomDims | None = ...,
    mom_params: MomParamsType = ...,
    squeeze: bool = ...,
    copy: bool = ...,
    keep_attrs: KeepAttrs = ...,
    dim_combined: Hashable | None = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
    **moment_kwargs: ArrayLike | xr.DataArray | xr.Dataset,
) -> DataT: ...
@overload
def assign_moment(
    data: NDArray[_ScalarT],
    moment: Mapping[SelectMoment, ArrayLike] | None = None,
    *,
    mom_ndim: MomNDim | None = ...,
    mom_axes: MomAxes | None = ...,
    mom_dims: MomDims | None = ...,
    mom_params: MomParamsType = ...,
    squeeze: bool = ...,
    copy: bool = ...,
    keep_attrs: KeepAttrs = ...,
    dim_combined: Hashable | None = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
    **moment_kwargs: ArrayLike | xr.DataArray | xr.Dataset,
) -> NDArray[_ScalarT]: ...


# TODO(wpk): dtype parameter?
@docfiller.decorate
def assign_moment(
    data: NDArray[_ScalarT] | DataT,
    moment: Mapping[SelectMoment, ArrayLike | xr.DataArray | DataT] | None = None,
    *,
    mom_ndim: MomNDim | None = None,
    mom_axes: MomAxes | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsType = None,
    squeeze: bool = True,
    copy: bool = True,
    dim_combined: Hashable | None = None,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    **moment_kwargs: ArrayLike | xr.DataArray | xr.Dataset,
) -> NDArray[_ScalarT] | DataT:
    r"""
    Update weights of moments array.

    Parameters
    ----------
    data : ndarray or DataArray
        Moments array.
    {assign_moment_mapping}
    {mom_ndim}
    {mom_axes}
    {mom_dims_data}
    {mom_params}
    {select_squeeze}
    copy : bool, default=True
        If ``True`` (the default), return new array with updated weights.
        Otherwise, return the original array with weights updated inplace.
        Note that a copy is always created for a ``dask`` backed object.
    dim_combined : str, optional
        Name of dimensions for multiple values. Must supply if passing in
        multiple values for ``name="ave"`` etc.
    {keep_attrs}
    {apply_ufunc_kwargs}
    **moment_kwargs
        Keyword argument form of ``moment``.  Must provide either ``moment`` or ``moment_kwargs``.

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

    >>> assign_moment(data, weight=-1, mom_ndim=1)
    array([-1,  1,  2])

    >>> assign_moment(data, ave=-1, mom_ndim=1)
    array([ 0, -1,  2])

    >>> assign_moment(data, var=-1, mom_ndim=1)
    array([ 0,  1, -1])

    Note that you can chain these together.  Note that the order of evaluation
    is the order values are passed.


    >>> assign_moment(data, weight=-1, ave=-2, mom_ndim=1)
    array([-1, -2,  2])


    For multidimensional data, the passed ``value`` must conform to the
    selected data


    >>> data = np.arange(2 * 3).reshape(2, 3)

    Selecting ``ave`` for this data with ``mom_ndim=1`` and ``squeeze=False`` would have shape ``(2, 1)``.

    >>> assign_moment(data, ave=np.ones((2, 1)), mom_ndim=1, squeeze=False)
    array([[0, 1, 2],
           [3, 1, 5]])

    The ``squeeze`` parameter has the same meaning as for :func:`select_moment`

    >>> assign_moment(data, ave=np.ones(2), mom_ndim=1, squeeze=True)
    array([[0, 1, 2],
           [3, 1, 5]])

    """
    # get names and values
    moment_kwargs = either_dict_or_kwargs(  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
        moment if moment is None else dict(moment),
        moment_kwargs,
        "assign_moment",
    )

    if is_xarray_typevar["DataT"].check(data):
        mom_params = MomParamsXArray.factory(
            mom_params=mom_params,
            ndim=mom_ndim,
            axes=mom_axes,
            dims=mom_dims,
            data=data,
            default_ndim=1,
        )
        # figure out values shape...
        input_core_dims: list[Sequence[Hashable]] = [mom_params.dims]
        for name, value in moment_kwargs.items():
            if not is_xarray(value) or np.isscalar(value):
                input_core_dims.append([])
            elif (
                name in {"ave", "var"}
                and (mom_params.ndim != 1 or not squeeze)
                and dim_combined
            ):
                input_core_dims.append([dim_combined])
            elif name.startswith("xmom_"):
                input_core_dims.append(mom_params.dims[1:])
            elif name.startswith("ymom_"):
                input_core_dims.append(mom_params.dims[:1])
            elif name == "all":
                input_core_dims.append(mom_params.dims)
            else:
                # fallback
                input_core_dims.append([])

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _assign_moment,
            data,
            *moment_kwargs.values(),
            input_core_dims=input_core_dims,
            output_core_dims=[mom_params.dims],
            kwargs={
                "names": moment_kwargs.keys(),
                "mom_params": mom_params.to_array(),
                "squeeze": squeeze,
                "copy": copy,
            },
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_dtypes=data.dtype
                if is_dataarray(data)  # type: ignore[redundant-expr]
                else np.float64,
            ),
        )
        return xout

    assert isinstance(data, np.ndarray)  # noqa: S101
    return _assign_moment(
        data,
        *moment_kwargs.values(),
        names=moment_kwargs.keys(),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        mom_params=MomParamsArray.factory(
            mom_params=mom_params, ndim=mom_ndim, axes=mom_axes, default_ndim=1
        ),
        squeeze=squeeze,
        copy=copy,
    )


def _assign_moment(
    data: NDArray[_ScalarT],
    *values: ArrayLike | xr.DataArray | xr.Dataset,
    names: Iterable[SelectMoment],
    mom_params: MomParamsArray,
    squeeze: bool,
    copy: bool,
) -> NDArray[_ScalarT]:
    out = data.copy() if copy else data

    mom_params_end = mom_params.axes_to_end()
    if moved := mom_params.axes != mom_params_end.axes:
        out = np.moveaxis(out, mom_params.axes, mom_params_end.axes)

    for name, value in zip(names, values, strict=True):
        out[moment_indexer(name, mom_params.ndim, squeeze)] = value  # pyright: ignore[reportArgumentType]

    if moved:
        out = np.moveaxis(out, mom_params_end.axes, mom_params.axes)
    return out


# * Vals -> Data --------------------------------------------------------------
# TODO(wpk): move this to convert?
@overload
def vals_to_data(  # pyright: ignore[reportOverlappingOverload]
    x: DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    weight: ArrayLike | xr.DataArray | DataT | None = ...,
    out: NDArrayAny | xr.DataArray | None = ...,
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ValsToDataKwargs],
) -> DataT: ...
# TODO(wpk): remove out as xr.DataArray.  Makes typing too weird...
# out
@overload
def vals_to_data(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    dtype: DTypeLike = ...,
    out: xr.DataArray,
    **kwargs: Unpack[ValsToDataKwargs],
) -> xr.DataArray: ...
# Array
@overload
def vals_to_data(
    x: ArrayLikeArg[FloatT],
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: None = ...,
    dtype: None = ...,
    **kwargs: Unpack[ValsToDataKwargs],
) -> NDArray[FloatT]: ...
# out
@overload
def vals_to_data(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    out: NDArray[FloatT],
    dtype: DTypeLike = ...,
    **kwargs: Unpack[ValsToDataKwargs],
) -> NDArray[FloatT]: ...
# dtype
@overload
def vals_to_data(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    dtype: DTypeLikeArg[FloatT],
    out: None = ...,
    **kwargs: Unpack[ValsToDataKwargs],
) -> NDArray[FloatT]: ...
# fallback
@overload
def vals_to_data(
    x: ArrayLike,
    *y: ArrayLike,
    weight: ArrayLike | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
    **kwargs: Unpack[ValsToDataKwargs],
) -> NDArrayAny: ...
# arraylike or DataT
@overload
def vals_to_data(
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    weight: ArrayLike | xr.DataArray | DataT | None = ...,
    dtype: DTypeLike = ...,
    out: NDArrayAny | None = ...,
    **kwargs: Unpack[ValsToDataKwargs],
) -> NDArrayAny | DataT: ...


@docfiller.decorate  # type: ignore[arg-type,unused-ignore]
def vals_to_data(
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    mom: Moments,
    weight: ArrayLike | xr.DataArray | DataT | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsType = None,
    dtype: DTypeLike = None,
    out: NDArrayAny | xr.DataArray | None = None,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArrayAny | DataT:
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
    {mom_dims}
    {mom_params}
    {dtype}
    {out}
    {keep_attrs}
    {apply_ufunc_kwargs}

    Returns
    -------
    data : ndarray or DataArray


    Notes
    -----
    Values ``x``, ``y`` and ``weight`` must be broadcastable.

    Note that if input data is a chunked dask array, then ``out`` is ignored, and a new
    array is always created.

    Also, broadcasting is different here then :func:`~.reduction.reduce_vals`,
    :func:`~.resample.resample_vals`, etc. For these reduction functions, you
    can, for example, pass a `1-D`` array for ``y`` with `N-D` ``x`` regardless
    if the axis is not the last axis. This is because those reduction functions
    have a ``axis`` parameter that can be used to infer the intended shape of
    ``y`` (and ``weight``). In this function, you have to specify ``y`` and
    ``weight`` such that they broadcast to ``x``.

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

    def _check_y(mom_ndim: int) -> None:
        raise_if_wrong_value(
            len(y), mom_ndim - 1, "`len(y)` should equal `mom_ndim -1`."
        )

    dtype = select_dtype(x, out=out, dtype=dtype)
    weight = 1.0 if weight is None else weight
    args: list[Any] = [x, weight, *y]

    if is_xarray_typevar["DataT"].check(x):
        mom, mom_params = MomParamsXArray.factory_mom(
            mom_params=mom_params, mom=mom, dims=mom_dims, data=out
        )
        _check_y(mom_params.ndim)

        # Explicitly select type depending o out
        # This is needed to make apply_ufunc work with dask data
        # can't pass None value in that case...
        out = None if is_dataset(x) else out  # type: ignore[redundant-expr]
        input_core_dims: list[Sequence[Hashable]] = [[]] * (mom_params.ndim + 1)
        if out is None:

            def _func(*args: Any, **kwargs: Any) -> Any:
                return _vals_to_data(*args, out=None, **kwargs)
        else:
            args = [out, *args]
            input_core_dims = [mom_params.dims, *input_core_dims]

            def _func(*args: Any, **kwargs: Any) -> Any:
                out_, *args_ = args
                return _vals_to_data(*args_, out=out_, **kwargs)

        return xr.apply_ufunc(  # type: ignore[no-any-return]  # pyright: ignore[reportUnknownMemberType]
            _func,
            *args,
            input_core_dims=input_core_dims,
            output_core_dims=[mom_params.dims],
            kwargs={
                "mom": mom,
                "mom_params": mom_params.to_array(),
                "dtype": dtype,
                "fastpath": False,
            },
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_sizes=dict(
                    zip(mom_params.dims, mom_to_mom_shape(mom), strict=True)
                )
                if out is None  # type: ignore[redundant-expr,unused-ignore]
                else None,
                output_dtypes=dtype if dtype is not None else np.float64,  # type: ignore[redundant-expr]
            ),
        )

    mom, mom_params = MomParamsArray.factory_mom(mom=mom, mom_params=mom_params)
    _check_y(mom_params.ndim)
    return _vals_to_data(  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
        *args,
        mom=mom,
        mom_params=mom_params,
        out=out,
        dtype=dtype,
        fastpath=True,
    )


def _vals_to_data(
    x: ArrayLike,
    weight: ArrayLike | xr.DataArray | xr.Dataset,
    *y: ArrayLike | xr.DataArray | xr.Dataset,
    mom: MomentsStrict,
    mom_params: MomParamsArray,
    out: NDArrayAny | xr.DataArray | None,
    dtype: DTypeLike,
    fastpath: bool,
) -> NDArrayAny | xr.DataArray:
    dtype = select_dtype(x, out=out, dtype=dtype, fastpath=fastpath)

    x_, w, *y_ = (np.asarray(a, dtype=dtype) for a in (x, weight, *y))
    if out is None:
        val_shape: tuple[int, ...] = np.broadcast_shapes(
            *(_.shape for _ in (x_, *y_, w))
        )
        out = np.zeros((*val_shape, *mom_to_mom_shape(mom)), dtype=dtype)
    else:
        out[...] = 0.0

    moment_kwargs: dict[SelectMoment, NDArrayAny | xr.DataArray] = {
        "weight": w,
        "xave": x_,
    }
    if mom_params.ndim == 2:
        moment_kwargs["yave"] = y_[0]
    return assign_moment(
        out, moment_kwargs, mom_ndim=mom_params.ndim, mom_params=mom_params, copy=False
    )

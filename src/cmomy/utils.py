"""
Interface to utility functions (:mod:`cmomy.utils`)
===================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import xarray as xr
from xarray.namedarray.utils import either_dict_or_kwargs

from .core.array_utils import normalize_axis_tuple, select_dtype
from .core.docstrings import docfiller
from .core.missing import MISSING
from .core.utils import mom_shape_to_mom as mom_shape_to_mom  # noqa: PLC0414
from .core.utils import mom_to_mom_shape as mom_to_mom_shape  # noqa: PLC0414
from .core.validate import (
    is_dataarray,
    is_dataset,
    is_xarray,
    raise_if_wrong_value,
    validate_axis_mult,
    validate_mom_and_mom_ndim,
    validate_mom_dims,
    validate_mom_dims_and_mom_ndim,
    validate_mom_ndim,
    validate_optional_mom_dims_and_mom_ndim,
)
from .core.xr_utils import (
    factory_apply_ufunc_kwargs,
    select_axis_dim_mult,
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

    from .core.typing import (
        ApplyUFuncKwargs,
        ArrayLikeArg,
        DataT,
        DTypeLikeArg,
        FloatT,
        KeepAttrs,
        MissingType,
        Mom_NDim,
        MomDims,
        Moments,
        MomentsStrict,
        NDArrayAny,
        ScalarT,
        SelectMoment,
        SelectMomentKwargs,
        ValsToDataKwargs,
    )
    from .core.typing_compat import EllipsisType, Unpack


# * moveaxis ------------------------------------------------------------------
@overload
def moveaxis(
    x: NDArray[ScalarT],
    axis: int | tuple[int, ...] | MissingType = ...,
    dest: int | tuple[int, ...] | MissingType = ...,
    *,
    dim: str | Sequence[Hashable] | MissingType = ...,
    dest_dim: str | Sequence[Hashable] | MissingType = ...,
    mom_ndim: Mom_NDim | None = ...,
    mom_dims: MomDims | None = ...,
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
    mom_dims: MomDims | None = ...,
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
    mom_dims: MomDims | None = None,
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
    {mom_dims_data}

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
    if is_dataarray(x):
        mom_dims, mom_ndim = validate_optional_mom_dims_and_mom_ndim(
            mom_dims, mom_ndim, x
        )
        axes0, dims0 = select_axis_dim_mult(x, axis=axis, dim=dim, mom_dims=mom_dims)
        axes1, dims1 = select_axis_dim_mult(
            x, axis=dest, dim=dest_dim, mom_dims=mom_dims
        )

        raise_if_wrong_value(
            len(dims0), len(dims1), "`dim` and `dest_dim` must have same length."
        )

        order = [n for n in range(x.ndim) if n not in axes0]
        for dst, src in sorted(zip(axes1, axes0)):
            order.insert(dst, src)
        return x.transpose(*(x.dims[o] for o in order))

    mom_ndim = None if mom_ndim is None else validate_mom_ndim(mom_ndim)
    axes0 = normalize_axis_tuple(validate_axis_mult(axis), x.ndim, mom_ndim)
    axes1 = normalize_axis_tuple(validate_axis_mult(dest), x.ndim, mom_ndim)

    return np.moveaxis(x, axes0, axes1)


# * Selecting subsets of data -------------------------------------------------
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
    name: SelectMoment | str, mom_ndim: Mom_NDim, squeeze: bool = True
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

    return (..., *idx)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]


@overload
def select_moment(
    data: DataT,
    name: SelectMoment,
    **kwargs: Unpack[SelectMomentKwargs],
) -> DataT: ...
@overload
def select_moment(
    data: NDArray[ScalarT],
    name: SelectMoment,
    **kwargs: Unpack[SelectMomentKwargs],
) -> NDArray[ScalarT]: ...


@docfiller.decorate
def select_moment(
    data: NDArray[ScalarT] | DataT,
    name: SelectMoment,
    *,
    mom_ndim: Mom_NDim | None = None,
    squeeze: bool = True,
    dim_combined: str = "variable",
    coords_combined: str | Sequence[Hashable] | None = None,
    mom_dims: MomDims | None = None,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
) -> NDArray[ScalarT] | DataT:
    """
    Select specific moments for a central moments array.

    Parameters
    ----------
    {data}
    {mom_ndim}
    {select_moment_name}
    {select_squeeze}
    {select_dim_combined}
    {select_coords_combined}
    {keep_attrs}
    {mom_dims_data}
    {apply_ufunc_kwargs}

    Returns
    -------
    output : ndarray or DataArray or Dataset
        Same type as ``data``. If ``name`` is ``ave`` or ``var``, the last
        dimensions of ``output`` has shape ``mom_ndim`` with each element
        corresponding to the `ith` variable. If ``squeeze=True`` and
        `mom_ndim==1`, this last dimension is removed. For all other ``name``
        options, output has shape of input with moment dimensions removed.


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
    if is_xarray(data):
        if name == "all":
            return data

        mom_dims, mom_ndim = validate_mom_dims_and_mom_ndim(
            mom_dims, mom_ndim, data, mom_ndim_default=1
        )
        # input/output dimensions
        input_core_dims = [mom_dims]
        output_core_dims: list[Sequence[Hashable]]
        output_sizes: dict[Hashable, int] | None
        if name in {"ave", "var"} and (mom_ndim != 1 or not squeeze):
            output_core_dims = [[dim_combined]]
            output_sizes = {dim_combined: mom_ndim}
            if coords_combined is None:
                coords_combined = mom_dims
            elif isinstance(coords_combined, str):
                coords_combined = [coords_combined]

            raise_if_wrong_value(
                len(coords_combined),
                mom_ndim,
                "`len(coords_combined)` must equal `mom_ndim`.",
            )
        else:
            output_sizes = None
            coords_combined = None
            if name.startswith("xmom_"):
                output_core_dims = [mom_dims[1:]]
            elif name.startswith("ymom_"):
                output_core_dims = [mom_dims[:1]]
            else:
                output_core_dims = [[]]

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _select_moment,
            data,
            input_core_dims=input_core_dims,
            output_core_dims=output_core_dims,
            kwargs={
                "mom_ndim": mom_ndim,
                "name": name,
                "squeeze": squeeze,
            },
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_sizes=output_sizes,
                output_dtypes=data.dtype  # pyright: ignore[reportUnknownMemberType]
                if is_dataarray(data)
                else np.float64,
            ),
        )
        if coords_combined is not None and dim_combined in xout.dims:
            xout = xout.assign_coords(  # pyright: ignore[reportUnknownMemberType]
                {dim_combined: (dim_combined, list(coords_combined))}
            )
        return xout

    return _select_moment(
        data,
        mom_ndim=validate_mom_ndim(mom_ndim, mom_ndim_default=1),
        name=name,
        squeeze=squeeze,
    )


def _select_moment(
    data: NDArray[ScalarT],
    *,
    mom_ndim: Mom_NDim,
    name: SelectMoment,
    squeeze: bool,
) -> NDArray[ScalarT]:
    if data.ndim < mom_ndim:
        msg = f"{data.ndim=} must be >= {mom_ndim=}"
        raise ValueError(msg)
    idx = moment_indexer(name, mom_ndim, squeeze)
    return data[idx]


# * Assign value(s) -----------------------------------------------------------
# NOTE: Can't do kwargs trick used elsewhere, because want to be
# able to use **moments_kwargs....
@overload
def assign_moment(
    data: DataT,
    moment: Mapping[SelectMoment, ArrayLike | xr.DataArray | DataT] | None = None,
    *,
    mom_ndim: Mom_NDim | None = ...,
    squeeze: bool = ...,
    copy: bool = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    dim_combined: Hashable | None = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
    **moment_kwargs: ArrayLike | xr.DataArray | xr.Dataset,
) -> DataT: ...
@overload
def assign_moment(
    data: NDArray[ScalarT],
    moment: Mapping[SelectMoment, ArrayLike] | None = None,
    *,
    mom_ndim: Mom_NDim | None = ...,
    squeeze: bool = ...,
    copy: bool = ...,
    keep_attrs: KeepAttrs = ...,
    mom_dims: MomDims | None = ...,
    dim_combined: Hashable | None = ...,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = ...,
    **moment_kwargs: ArrayLike | xr.DataArray | xr.Dataset,
) -> NDArray[ScalarT]: ...


# TODO(wpk): dtype parameter?
@docfiller.decorate
def assign_moment(
    data: NDArray[ScalarT] | DataT,
    moment: Mapping[SelectMoment, ArrayLike | xr.DataArray | DataT] | None = None,
    *,
    mom_ndim: Mom_NDim | None = None,
    squeeze: bool = True,
    copy: bool = True,
    dim_combined: Hashable | None = None,
    mom_dims: MomDims | None = None,
    keep_attrs: KeepAttrs = None,
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    **moment_kwargs: ArrayLike | xr.DataArray | xr.Dataset,
) -> NDArray[ScalarT] | DataT:
    r"""
    Update weights of moments array.

    Parameters
    ----------
    data : ndarray or DataArray
        Moments array.
    {assign_moment_mapping}
    {mom_ndim}
    {select_squeeze}
    copy : bool, default=True
        If ``True`` (the default), return new array with updated weights.
        Otherwise, return the original array with weights updated inplace.
        Note that a copy is always created for a ``dask`` backed object.
    dim_combined : str, optional
        Name of dimensions for multiple values. Must supply if passing in
        multiple values for ``name="ave"`` etc.
    {mom_dims_data}
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
    # get names ands values
    moment_kwargs = either_dict_or_kwargs(  # type: ignore[assignment]
        moment if moment is None else dict(moment),
        moment_kwargs,
        "assign_moment",
    )

    if is_xarray(data):
        mom_dims, mom_ndim = validate_mom_dims_and_mom_ndim(
            mom_dims, mom_ndim, data, mom_ndim_default=1
        )
        # figure out values shape...
        input_core_dims: list[Sequence[Hashable]] = [mom_dims]
        for name, value in moment_kwargs.items():
            if not is_xarray(value) or np.isscalar(value):
                input_core_dims.append([])
            elif (
                name in {"ave", "var"}
                and (mom_ndim != 1 or not squeeze)
                and dim_combined
            ):
                input_core_dims.append([dim_combined])
            elif name.startswith("xmom_"):
                input_core_dims.append(mom_dims[1:])
            elif name.startswith("ymom_"):
                input_core_dims.append(mom_dims[:1])
            elif name == "all":
                input_core_dims.append(mom_dims)
            else:
                # fallback
                input_core_dims.append([])

        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            _assign_moment,
            data,
            *moment_kwargs.values(),
            input_core_dims=input_core_dims,
            output_core_dims=[mom_dims],
            kwargs={
                "names": moment_kwargs.keys(),
                "mom_ndim": mom_ndim,
                "squeeze": squeeze,
                "copy": copy,
            },
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_dtypes=data.dtype  # pyright: ignore[reportUnknownMemberType]
                if is_dataarray(data)
                else np.float64,
            ),
        )
        return xout

    return _assign_moment(
        data,
        *moment_kwargs.values(),
        names=moment_kwargs.keys(),  # type: ignore[arg-type]
        mom_ndim=validate_mom_ndim(mom_ndim),
        squeeze=squeeze,
        copy=copy,
    )


def _assign_moment(
    data: NDArray[ScalarT],
    *values: ArrayLike | xr.DataArray | xr.Dataset,
    names: Iterable[SelectMoment],
    mom_ndim: Mom_NDim,
    squeeze: bool,
    copy: bool,
) -> NDArray[ScalarT]:
    out = data.copy() if copy else data

    for name, value in zip(names, values):
        out[moment_indexer(name, mom_ndim, squeeze)] = value
    return out


# * Vals -> Data --------------------------------------------------------------
# TODO(wpk): move this to convert?
@overload
def vals_to_data(
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


@docfiller.decorate  # type: ignore[arg-type,unused-ignore]
def vals_to_data(
    x: ArrayLike | DataT,
    *y: ArrayLike | xr.DataArray | DataT,
    mom: Moments,
    weight: ArrayLike | xr.DataArray | DataT | None = None,
    dtype: DTypeLike = None,
    out: NDArrayAny | xr.DataArray | None = None,
    mom_dims: MomDims | None = None,
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
    {dtype}
    {out}

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
    mom, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
    dtype = select_dtype(x, out=out, dtype=dtype)
    weight = 1.0 if weight is None else weight

    raise_if_wrong_value(len(y), mom_ndim - 1, "`len(y)` should equal `mom_ndim -1`.")

    args: list[Any] = [x, weight, *y]
    if is_xarray(x):
        if is_dataarray(out) and mom_dims is None:
            mom_dims = out.dims[-mom_ndim:]
        else:
            mom_dims = validate_mom_dims(mom_dims=mom_dims, mom_ndim=mom_ndim)

        # Explicitly select type depending o out
        # This is needed to make apply_ufunc work with dask data
        # can't pass None value in that case...
        out = None if is_dataset(x) else out
        input_core_dims: list[Sequence[Hashable]] = [[]] * (mom_ndim + 1)
        if out is None:

            def _func(*args: Any, **kwargs: Any) -> Any:
                return _vals_to_data(*args, out=None, **kwargs)
        else:
            args = [out, *args]
            input_core_dims = [mom_dims, *input_core_dims]

            def _func(*args: Any, **kwargs: Any) -> Any:
                _out, *_args = args
                return _vals_to_data(*_args, out=_out, **kwargs)  # type: ignore[has-type]

        return xr.apply_ufunc(  # type: ignore[no-any-return]
            _func,
            *args,
            input_core_dims=input_core_dims,
            output_core_dims=[mom_dims],
            kwargs={
                "mom": mom,
                "mom_ndim": mom_ndim,
                "dtype": dtype,
                "fastpath": False,
            },
            keep_attrs=keep_attrs,
            **factory_apply_ufunc_kwargs(
                apply_ufunc_kwargs,
                dask="parallelized",
                output_sizes=dict(zip(mom_dims, mom_to_mom_shape(mom)))
                if out is None
                else None,
                output_dtypes=dtype or np.float64,
            ),
        )

    return _vals_to_data(  # type: ignore[return-value]
        *args,
        mom=mom,
        mom_ndim=mom_ndim,
        out=out,
        dtype=dtype,
        fastpath=True,
    )


def _vals_to_data(
    x: ArrayLike,
    weight: ArrayLike | xr.DataArray | xr.Dataset,
    *y: ArrayLike | xr.DataArray | xr.Dataset,
    mom: MomentsStrict,
    mom_ndim: Mom_NDim,
    out: NDArrayAny | xr.DataArray | None,
    dtype: DTypeLike,
    fastpath: bool,
) -> NDArrayAny | xr.DataArray:
    if not fastpath:
        dtype = select_dtype(x, out=out, dtype=dtype)

    _x, _w, *_y = (np.asarray(a, dtype=dtype) for a in (x, weight, *y))
    if out is None:
        val_shape: tuple[int, ...] = np.broadcast_shapes(
            *(_.shape for _ in (_x, *_y, _w))
        )
        out = np.zeros((*val_shape, *mom_to_mom_shape(mom)), dtype=dtype)
    else:
        out[...] = 0.0

    moment_kwargs: dict[SelectMoment, NDArrayAny | xr.DataArray] = {
        "weight": _w,
        "xave": _x,
    }
    if mom_ndim == 2:
        moment_kwargs["yave"] = _y[0]
    return assign_moment(out, moment_kwargs, mom_ndim=mom_ndim, copy=False)

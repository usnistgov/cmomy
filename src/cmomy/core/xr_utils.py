"""Utilities to work with xr.DataArray objects."""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from typing import TYPE_CHECKING, cast

from cmomy.core.typing import MomentsStrict

from .array_utils import normalize_axis_index, normalize_axis_tuple
from .missing import MISSING
from .validate import (
    is_dataset,
    validate_mom_dims,
    validate_not_none,
)

if TYPE_CHECKING:
    from collections.abc import (
        Collection,
        Hashable,
    )
    from typing import Any

    import xarray as xr
    from numpy.typing import DTypeLike

    from .typing import (
        ApplyUFuncKwargs,
        AxisReduce,
        AxisReduceMult,
        DimsReduce,
        DimsReduceMult,
        MissingCoreDimOptions,
        MissingType,
        Mom_NDim,
        MomDims,
        MomDimsStrict,
        MomentsStrict,
    )


# * apply_ufunc_kws
def get_apply_ufunc_kwargs(
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    on_missing_core_dim: MissingCoreDimOptions | None = None,
    dask: str = "forbidden",
    dask_gufunc_kwargs: Mapping[str, Any] | None = None,
    output_sizes: Mapping[Hashable, int] | None = None,
    output_dtypes: Any = None,
) -> dict[str, Any]:
    """Create apply_ufunc_kwargs dict."""
    out: dict[str, Any] = {} if apply_ufunc_kwargs is None else dict(apply_ufunc_kwargs)
    if on_missing_core_dim:
        out["on_missing_core_dim"] = on_missing_core_dim

    out.setdefault("dask", dask)
    out.setdefault(
        "dask_gufunc_kwargs",
        {} if dask_gufunc_kwargs is None else dict(dask_gufunc_kwargs),
    )

    if output_sizes:
        out["dask_gufunc_kwargs"].setdefault("output_sizes", dict(output_sizes))
    if output_dtypes:
        out.setdefault("output_dtypes", output_dtypes)
    return out


# * Select axis/dim -----------------------------------------------------------
def select_axis_dim(
    data: xr.DataArray | xr.Dataset,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    default_axis: AxisReduce | MissingType = MISSING,
    default_dim: DimsReduce | MissingType = MISSING,
    mom_ndim: Mom_NDim | None = None,
) -> tuple[int, Hashable]:
    """Produce axis/dim from input."""
    # for now, disallow None values
    axis = validate_not_none(axis, "axis")
    dim = validate_not_none(dim, "dim")

    if is_dataset(data):
        if axis is not MISSING or dim is MISSING:
            msg = "For Dataset, must specify ``dim`` value other than ``None`` only."
            raise ValueError(msg)
        return 0, dim

    default_axis = validate_not_none(default_axis, "default_axis")
    default_dim = validate_not_none(default_dim, "default_dim")

    if axis is MISSING and dim is MISSING:
        if default_axis is not MISSING and default_dim is MISSING:
            axis = default_axis
        elif default_axis is MISSING and default_dim is not MISSING:
            dim = default_dim
        else:
            msg = "Must specify axis or dim, or one of default_axis or default_dim"
            raise ValueError(msg)

    elif axis is not MISSING and dim is not MISSING:
        msg = "Can only specify one of axis or dim"
        raise ValueError(msg)

    if dim is not MISSING:
        axis = data.get_axis_num(dim)
        if axis >= data.ndim - (0 if mom_ndim is None else mom_ndim):
            msg = f"Cannot select moment dimension. {axis=}, {dim=}."
            raise ValueError(msg)

    elif axis is not MISSING:
        if isinstance(axis, str):
            msg = f"Using string value for axis is deprecated.  Please use `dim` option instead.  Passed {axis} of type {type(axis)}"
            raise ValueError(msg)
        # wrap axis
        axis = normalize_axis_index(axis, data.ndim, mom_ndim)  # type: ignore[arg-type]
        dim = data.dims[axis]
    else:  # pragma: no cover
        msg = f"Unknown dim {dim} and axis {axis}"
        raise TypeError(msg)

    return axis, dim


def select_axis_dim_mult(  # noqa: C901
    data: xr.DataArray | xr.Dataset,
    axis: AxisReduceMult | MissingType = MISSING,
    dim: DimsReduceMult | MissingType = MISSING,
    default_axis: AxisReduceMult | MissingType = MISSING,
    default_dim: DimsReduceMult | MissingType = MISSING,
    mom_ndim: Mom_NDim | None = None,
    mom_dims: MomDimsStrict | None = None,
) -> tuple[tuple[int, ...], tuple[Hashable, ...]]:
    """
    Produce axis/dim tuples from input.

    This is like `select_axis_dim`, but allows multiple values in axis/dim.
    """
    if is_dataset(data):
        if axis is not MISSING or dim is MISSING:
            msg = "For Dataset, must specify ``dim`` value only."
            raise ValueError(msg)

        if dim is None:
            dim = tuple(data.dims)

            if mom_dims is None and mom_ndim is not None:
                mom_dims = validate_mom_dims(None, mom_ndim, data)

            if mom_dims is not None:
                dim = tuple(d for d in dim if d not in mom_dims)
        return (), (dim,) if isinstance(dim, str) else dim  # type: ignore[return-value]

    # Allow None, which implies choosing all dimensions...
    if axis is MISSING and dim is MISSING:
        if default_axis is not MISSING and default_dim is MISSING:
            axis = default_axis
        elif default_axis is MISSING and default_dim is not MISSING:
            dim = default_dim
        else:
            msg = "Must specify axis or dim, or one of default_axis or default_dim"
            raise ValueError(msg)

    elif axis is not MISSING and dim is not MISSING:
        msg = "Can only specify one of axis or dim"
        raise ValueError(msg)

    ndim = data.ndim - (0 if mom_ndim is None else mom_ndim)
    dim_: tuple[Hashable, ...]
    axis_: tuple[int, ...]

    if dim is not MISSING:
        dim_ = (
            data.dims[:ndim]
            if dim is None
            else (dim,)
            if isinstance(dim, str)
            else tuple(dim)  # type: ignore[arg-type]
        )

        axis_ = data.get_axis_num(dim_)
        if any(a >= ndim for a in axis_):
            msg = f"Cannot select moment dimension. {axis_=}, {dim_=}."
            raise ValueError(msg)

    elif axis is not MISSING:
        axis_ = normalize_axis_tuple(axis, data.ndim, mom_ndim)
        dim_ = tuple(data.dims[a] for a in axis_)

    else:  # pragma: no cover
        msg = f"Unknown dim {dim} and axis {axis}"
        raise TypeError(msg)

    return axis_, dim_


def move_mom_dims_to_end(
    x: xr.DataArray, mom_dims: MomDims, mom_ndim: Mom_NDim | None = None
) -> xr.DataArray:
    """Move moment dimensions to end"""
    if mom_dims is not None:
        mom_dims = (mom_dims,) if isinstance(mom_dims, str) else tuple(mom_dims)  # type: ignore[arg-type]

        if mom_ndim is not None and len(mom_dims) != mom_ndim:
            msg = f"len(mom_dims)={len(mom_dims)} not equal to mom_ndim={mom_ndim}"
            raise ValueError(msg)

        x = x.transpose(..., *mom_dims)

    return x


def replace_coords_from_isel(
    da_original: xr.DataArray,
    da_selected: xr.DataArray,
    indexers: Mapping[Any, Any] | None = None,
    drop: bool = False,
    **indexers_kwargs: Any,
) -> xr.DataArray:
    """
    Replace coords in da_selected with coords from coords from da_original.isel(...).

    This assumes that `da_selected` is the result of soe operation, and that indexeding
    ``da_original`` will give the correct coordinates/indexed.

    Useful for adding back coordinates to reduced object.
    """
    from xarray.core.indexes import (
        isel_indexes,
    )
    from xarray.core.indexing import is_fancy_indexer

    # Would prefer to import from actual source by old xarray error.
    from xarray.core.utils import either_dict_or_kwargs  # type: ignore[attr-defined]

    indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")
    if any(is_fancy_indexer(idx) for idx in indexers.values()):  # pragma: no cover
        msg = "no fancy indexers for this"
        raise ValueError(msg)

    indexes, index_variables = isel_indexes(da_original.xindexes, indexers)

    coords = {}
    for coord_name, coord_value in da_original._coords.items():  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        if coord_name in index_variables:
            coord_value = index_variables[coord_name]  # noqa: PLW2901
        else:
            coord_indexers = {
                k: v for k, v in indexers.items() if k in coord_value.dims
            }
            if coord_indexers:
                coord_value = coord_value.isel(coord_indexers)  # noqa: PLW2901
                if drop and coord_value.ndim == 0:
                    continue
        coords[coord_name] = coord_value

    return da_selected._replace(coords=coords, indexes=indexes)  # pyright: ignore[reportUnknownMemberType, reportPrivateUsage]


def raise_if_dataset(*args: Any, msg: str = "Dataset not allowed.") -> None:
    """Raise TypeError if value is a Dataset."""
    if any(is_dataset(x) for x in args):
        raise TypeError(msg)


def get_mom_shape(
    data: xr.DataArray | xr.Dataset,
    mom_dims: MomDimsStrict,
) -> MomentsStrict:
    """Extract moments shape from xarray object."""
    mom_shape = tuple(data.sizes[m] for m in mom_dims)
    return cast("MomentsStrict", mom_shape)


def contains_dims(
    data: xr.DataArray | xr.Dataset, dims: str | Collection[Hashable]
) -> bool:
    """Wheater data contains `dims`."""
    return all(d in data.dims for d in dims)


def astype_dtype_dict(
    obj: xr.DataArray | xr.Dataset,
    dtype: DTypeLike | Mapping[Hashable, DTypeLike],
) -> DTypeLike | dict[Hashable, DTypeLike]:
    """Get a dtype dict for obj."""
    if isinstance(dtype, Mapping):
        if is_dataset(obj):
            return dict(obj.dtypes, **dtype)  # pyright: ignore[reportCallIssue, reportUnknownMemberType, reportUnknownVariableType]

        msg = "Passing a mapping for `dtype` only allowed for Dataset."
        raise ValueError(msg)

    return dtype

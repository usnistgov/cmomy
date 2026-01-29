# ruff: noqa: SLF001
# pyright: reportPrivateUsage=false
"""Utilities to work with xr.DataArray objects."""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from typing import TYPE_CHECKING, cast

from .validate import (
    is_dataarray,
    is_dataset,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Sequence,
    )
    from typing import Any

    import xarray as xr
    from numpy.typing import DTypeLike

    from ._typing_kwargs import (
        ApplyUFuncKwargs,
    )
    from .moment_params import MomParamsArrayOptional
    from .typing import (
        DataT,
        MissingCoreDimOptions,
        MomDims,
        MomDimsStrict,
        MomentsStrict,
        MomNDim,
    )
    from .typing_compat import EllipsisType


# * apply_ufunc_kws
def factory_apply_ufunc_kwargs(
    apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    on_missing_core_dim: MissingCoreDimOptions = "copy",
    dask: str = "parallel",
    dask_gufunc_kwargs: Mapping[str, Any] | None = None,
    output_sizes: Mapping[Hashable, int] | None = None,
    output_dtypes: Any = None,
) -> dict[str, Any]:
    """
    Create kwargs to pass to :func:`xarray.apply_ufunc`

    Pass in options with ``apply_ufunc_kwargs``.  The other options set defaults of that parameter.
    """
    out: dict[str, Any] = {} if apply_ufunc_kwargs is None else dict(apply_ufunc_kwargs)

    out.setdefault("on_missing_core_dim", on_missing_core_dim)
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
def move_mom_dims_to_end(
    x: xr.DataArray, mom_dims: MomDims, mom_ndim: MomNDim | None = None
) -> xr.DataArray:
    """Move moment dimensions to end"""
    if mom_dims is not None:
        mom_dims = (mom_dims,) if isinstance(mom_dims, str) else tuple(mom_dims)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

        if mom_ndim is not None and len(mom_dims) != mom_ndim:
            msg = f"len(mom_dims)={len(mom_dims)} not equal to mom_ndim={mom_ndim}"
            raise ValueError(msg)

        x = x.transpose(..., *mom_dims)

    return x


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


def astype_dtype_dict(
    obj: xr.DataArray | xr.Dataset,
    dtype: DTypeLike | Mapping[Hashable, DTypeLike],
) -> DTypeLike | dict[Hashable, DTypeLike]:
    """Get a dtype dict for obj."""
    if isinstance(dtype, Mapping):
        if is_dataset(obj):
            return dict(obj.dtypes, **dtype)  # type: ignore[arg-type, return-value]  # pyright: ignore[reportCallIssue, reportUnknownVariableType]

        msg = "Passing a mapping for `dtype` only allowed for Dataset."
        raise ValueError(msg)

    return dtype


def contains_dims(data: xr.DataArray | xr.Dataset, *dims: Hashable) -> bool:
    """Check if xarray object contains dimensions"""
    return all(d in data.dims for d in dims)


# * Transpose like ------------------------------------------------------------
def transpose_like(
    data_out: DataT,
    template: xr.DataArray | xr.Dataset,
    replace: Mapping[Hashable, Hashable] | None = None,
    remove: str | Sequence[Hashable] | None = None,
    keep_attrs: bool | None = True,
    prepend: Sequence[Hashable] | EllipsisType | None = ...,
    append: Sequence[Hashable] | EllipsisType | None = None,
    mom_params_axes: MomParamsArrayOptional | None = None,
) -> DataT:
    """Transpose ``data_out`` like ``template``."""
    replace = {} if replace is None else dict(replace)
    remove_: set[Hashable] = (
        set()
        if remove is None
        else set([remove])  # noqa: C405
        if isinstance(remove, str)
        else set(remove)
    )

    prepend = [] if prepend is None else [prepend] if prepend is ... else prepend
    append = [] if append is None else [append] if append is ... else append

    if is_dataset(data_out):
        return data_out.map(  # pyright: ignore[reportUnknownMemberType]
            _transpose_like,
            keep_attrs=keep_attrs,
            template=template,
            replace=replace,
            remove=remove_,
            prepend=prepend,
            append=append,
        )
    out = _transpose_like(
        data_out,
        template=template,
        replace=replace,
        remove=remove_,
        prepend=prepend,
        append=append,
    )

    if mom_params_axes is not None:
        return mom_params_axes.maybe_reorder_dataarray(out)
    return out


def _transpose_like(
    data_out: DataT,
    template: xr.DataArray | xr.Dataset,
    replace: dict[Hashable, Hashable],
    remove: set[Hashable],
    prepend: Sequence[Hashable],
    append: Sequence[Hashable],
) -> DataT:
    if is_dataset(template):
        template = template[data_out.name]

    order: list[Hashable] = list(template.dims)
    if remove:
        for r in remove:
            if r in order:
                order.remove(r)

    if replace:
        order = [replace.get(o, o) for o in order]

    if (order := [*prepend, *order, *append]) != list(
        data_out.dims
    ):  # pragma: no cover
        data_out = data_out.transpose(*order, missing_dims="ignore")
    return data_out


# * Assign coords to selected -------------------------------------------------
def _replace_coords_from_isel_dataarray(
    template: xr.DataArray,
    selected: xr.DataArray,
    indexers: Mapping[Any, Any],
    drop: bool,
) -> xr.DataArray:
    # Taken from xarray.DataArray.isel
    from xarray.core.indexes import isel_indexes

    indexes, index_variables = isel_indexes(template.xindexes, indexers)
    coords = {}
    for coord_name, coord_value in template._coords.items():
        if coord_name in index_variables:
            value = index_variables[coord_name]
        else:
            coord_indexers = {
                k: v for k, v in indexers.items() if k in coord_value.dims
            }
            if coord_indexers:
                value = coord_value.isel(coord_indexers)
                if drop and not value.ndim:
                    continue
            else:
                value = coord_value

        coords[coord_name] = value

    return selected._replace(coords=coords, indexes=indexes)  # pyright: ignore[reportUnknownMemberType]


def _replace_coords_from_isel_dataset(
    template: xr.Dataset,
    selected: xr.Dataset,
    indexers: Mapping[Any, Any],
    drop: bool,
) -> xr.Dataset:
    # Taken from xarray.Dataset.isel
    from xarray.core.indexes import isel_indexes
    from xarray.core.utils import drop_dims_from_indexers

    indexers = drop_dims_from_indexers(indexers, template.dims, "raise")
    variables: dict[Any, xr.Variable] = {}
    dims: dict[Hashable, int] = {}
    coord_names = template._coord_names.copy()

    indexes, index_variables = isel_indexes(template.xindexes, indexers)

    for name, var in template._variables.items():
        # preserve variable order
        if name in selected._variables:
            new_var = selected._variables[name]
        elif name in index_variables:
            new_var = index_variables[name]
        else:
            var_indexers = {k: v for k, v in indexers.items() if k in var.dims}
            if var_indexers:
                new_var = var.isel(var_indexers)
                if drop and new_var.ndim == 0 and name in coord_names:
                    coord_names.remove(name)
                    continue
            else:
                new_var = var

        variables[name] = new_var
        if len(new_var.dims) != len(new_var.shape):  # pragma: no cover
            msg = "dims and shape have different size"
            raise ValueError(msg)
        dims.update(zip(new_var.dims, new_var.shape, strict=True))

    return template._construct_direct(  # pyright: ignore[reportUnknownMemberType]
        variables=variables,
        coord_names=coord_names,
        dims=dims,
        attrs=template._attrs,
        indexes=indexes,
        encoding=template._encoding,
        close=template._close,
    )


def replace_coords_from_isel(
    template: DataT,
    selected: DataT,
    indexers: Mapping[Any, Any] | None = None,
    drop: bool = False,
    **indexers_kwargs: Any,
) -> DataT:
    """
    Replace coords in selected with coords from coords from template.isel(...).

    This assumes that `selected` is the result of soe operation, and that indexeding
    ``template`` will give the correct coordinates/indexed.

    Useful for adding back coordinates to reduced object.
    """
    from xarray.core.indexing import is_fancy_indexer

    # Would prefer to import from actual source by old xarray error.
    from xarray.core.utils import (  # type: ignore[attr-defined]
        either_dict_or_kwargs,  # pyright: ignore[reportPrivateImportUsage]
    )

    indexers = either_dict_or_kwargs(
        indexers, indexers_kwargs, "replace_coords_from_isel"
    )
    if any(is_fancy_indexer(idx) for idx in indexers.values()):  # pragma: no cover
        msg = "no fancy indexers for this"
        raise ValueError(msg)

    if is_dataset(template) and is_dataset(selected):  # type: ignore[redundant-expr]
        return _replace_coords_from_isel_dataset(
            template=template, selected=selected, indexers=indexers, drop=drop
        )
    if is_dataarray(template) and is_dataarray(selected):  # type: ignore[redundant-expr]
        return _replace_coords_from_isel_dataarray(
            template=template, selected=selected, indexers=indexers, drop=drop
        )
    msg = "template and selected must have same type."
    raise TypeError(msg)

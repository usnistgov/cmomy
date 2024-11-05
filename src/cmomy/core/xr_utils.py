"""Utilities to work with xr.DataArray objects."""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from typing import TYPE_CHECKING, cast

from .validate import (
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

    from .typing import (
        ApplyUFuncKwargs,
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
        mom_dims = (mom_dims,) if isinstance(mom_dims, str) else tuple(mom_dims)  # type: ignore[arg-type]

        if mom_ndim is not None and len(mom_dims) != mom_ndim:
            msg = f"len(mom_dims)={len(mom_dims)} not equal to mom_ndim={mom_ndim}"
            raise ValueError(msg)

        x = x.transpose(..., *mom_dims)  # pyright: ignore[reportUnknownArgumentType]  # python3.9

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
) -> DataT:
    """Transpose ``data_out`` like ``template``."""
    replace = {} if replace is None else dict(replace)
    _remove: set[Hashable] = (
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
            remove=_remove,
            prepend=prepend,
            append=append,
        )
    return _transpose_like(
        data_out,
        template=template,
        replace=replace,
        remove=_remove,
        prepend=prepend,  # pyright: ignore[reportArgumentType]
        append=append,  # pyright: ignore[reportArgumentType]
    )


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

    order = list(template.dims)
    if remove:
        for r in remove:
            if r in order:
                order.remove(r)

    if replace:
        order = [replace.get(o, o) for o in order]

    order = [*prepend, *order, *append]  # type: ignore[has-type]

    if order != list(data_out.dims):  # pragma: no cover
        data_out = data_out.transpose(*order, missing_dims="ignore")
    return data_out

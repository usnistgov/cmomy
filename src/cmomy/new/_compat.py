"""Version compatibility code should go here."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Hashable, Iterable


def xr_dot(
    *arrays: Any,
    dim: str | Iterable[Hashable] | "ellipsis" | None = None,  # noqa: F821, UP037
    **kwargs: Any,
) -> Any:
    """
    Interface to xarray.dot.

    Remove a deprecation warning for older xarray versions.
    xarray deprecated `dims` keyword.  Use `dim` instead.
    """
    import xarray as xr

    try:
        return xr.dot(*arrays, dim=dim, **kwargs)  # type: ignore[arg-type,unused-ignore]
    except TypeError:
        return xr.dot(*arrays, dims=dim, **kwargs)  # type: ignore[arg-type,unused-ignore]

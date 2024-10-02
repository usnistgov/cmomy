"""Helper class to work with moment parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, TypedDict, cast

from .array_utils import normalize_axis_index, normalize_axis_tuple
from .validate import (
    validate_mom_axes,
    validate_mom_dims_and_mom_ndim,
    validate_mom_ndim,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Sequence
    from typing import Any

    import xarray as xr

    from .typing import (
        MomAxesStrict,
        MomDimsStrict,
        MomNDim,
    )
    from .typing_compat import Self, Unpack


class MomParamsDict(TypedDict, total=False):
    """
    Input moment parameters.

    Parameters
    ----------
    axes : int or sequence of int
        Axes/Axis for moment dimension(s).
    dims : str or sequence of hashable
        Name of moment dimensions (for :mod:`xarray` objects).
    """

    ndim: int | None
    axes: int | Sequence[int] | None
    dims: Hashable | Sequence[Hashable] | None


class _Mixin:
    def asdict(self) -> MomParamsDict:
        return cast("MomParamsDict", asdict(self))  # type: ignore[call-overload]


@dataclass
class MomParams:
    """Dataclass for moment parameters"""

    ndim: int | None = None
    axes: int | Sequence[int] | None = None
    dims: Hashable | Sequence[Hashable] | None = None

    def asdict(self) -> MomParamsDict:
        return cast("MomParamsDict", asdict(self))

    def new_like(self, **kwargs: Unpack[MomParamsDict]) -> Self:
        return replace(self, **kwargs)


@dataclass
class MomParamsArray(_Mixin):
    """Array Mom Params."""

    ndim: MomNDim
    axes: MomAxesStrict

    @classmethod
    def from_params(
        cls,
        ndim: int | None = None,
        axes: int | Sequence[int] | None = None,
        default_ndim: MomNDim | None = None,
        dims: Any = None,  # noqa: ARG003  # in case pass in dims parameter, it will be ignored.
    ) -> Self:
        if axes is None:
            ndim = validate_mom_ndim(ndim, default_ndim)
            axes = cast("MomAxesStrict", tuple(range(-ndim, 0)))
            return cls(ndim=ndim, axes=axes)

        axes = validate_mom_axes(axes)
        if ndim is None:
            ndim = len(axes)
        elif len(axes) != ndim:
            msg = f"{len(axes)=} != {ndim=}"
            raise ValueError(msg)
        return cls(ndim=cast("MomNDim", ndim), axes=axes)

    @classmethod
    def factory(
        cls,
        ndim: int | None = None,
        mom_kws: MomParamsDict | None = None,
        axes: int | Sequence[int] | None = None,
        default_ndim: MomNDim | None = None,
    ) -> Self:
        mom_kws = {} if mom_kws is None else MomParamsDict(mom_kws)  # type: ignore[misc]
        if ndim is not None:
            mom_kws["ndim"] = ndim
        mom_kws.setdefault("axes", axes)
        return cls.from_params(**mom_kws, default_ndim=default_ndim)

    def move_axes_to_end(self) -> Self:
        return replace(self, axes=cast("MomAxesStrict", tuple(range(-self.ndim, 0))))

    def normalize_axis_index(
        self, axis: complex, data_ndim: int, msg_prefix: str | None = None
    ) -> int:
        return normalize_axis_index(
            axis=axis, ndim=data_ndim, mom_ndim=self.ndim, msg_prefix=msg_prefix
        )

    def normalize_axis_tuple(
        self,
        axis: complex | Iterable[complex] | None,
        data_ndim: int,
        msg_prefix: str | None = None,
        allow_duplicate: bool = False,
    ) -> tuple[int, ...]:
        return normalize_axis_tuple(
            axis=axis,
            ndim=data_ndim,
            mom_ndim=self.ndim,
            msg_prefix=msg_prefix,
            allow_duplicate=allow_duplicate,
        )

    def normalize_axes(self, data_ndim: int) -> Self:
        """Normalize self.axes in new object."""
        if self.axes is None:  # pyright: ignore[reportUnnecessaryComparison]
            return self

        return replace(
            self,
            axes=normalize_axis_tuple(  # type: ignore[arg-type]
                self.axes, data_ndim, msg_prefix="normalize_axes"
            ),
        )


@dataclass
class MomParamsArrayOptional(MomParamsArray):
    """Optional array mom params."""

    ndim: MomNDim | None  # type: ignore[assignment]
    axes: MomAxesStrict | None  # type: ignore[assignment]

    @classmethod
    def from_params(
        cls,
        ndim: int | None = None,
        axes: int | Sequence[int] | None = None,
        default_ndim: MomNDim | None = None,
        dims: Any = None,  # noqa: ARG003  # in case pass dims parameter, it will be ignored.
    ) -> Self:
        if ndim is None and axes is None and default_ndim is None:
            return cls(None, None)
        return super().from_params(ndim, axes, default_ndim)


@dataclass
class MomParamsXArray(_Mixin):
    """XArray mom parameters."""

    ndim: MomNDim
    dims: MomDimsStrict

    @classmethod
    def from_params(
        cls,
        ndim: int | None = None,
        dims: Hashable | Sequence[Hashable] | None = None,
        axes: int | Sequence[int] | None = None,
        data: xr.DataArray | xr.Dataset | None = None,
        default_ndim: MomNDim | None = None,
    ) -> Self:
        dims, ndim = validate_mom_dims_and_mom_ndim(
            mom_dims=dims,
            mom_ndim=ndim,
            out=data,
            mom_ndim_default=default_ndim,
            mom_axes=axes,
        )
        return cls(ndim=ndim, dims=dims)

    @classmethod
    def factory(
        cls,
        ndim: int | None = None,
        mom_kws: MomParamsDict | None = None,
        dims: Hashable | Sequence[Hashable] | None = None,
        axes: int | Sequence[int] | None = None,
        data: xr.DataArray | xr.Dataset | None = None,
        default_ndim: MomNDim | None = None,
    ) -> Self:
        mom_kws = {} if mom_kws is None else MomParamsDict(mom_kws)  # type: ignore[misc]
        if ndim is not None:
            mom_kws["ndim"] = ndim
        mom_kws.setdefault("dims", dims)
        mom_kws.setdefault("axes", axes)
        return cls.from_params(**mom_kws, data=data, default_ndim=default_ndim)

    @property
    def axes(self) -> MomAxesStrict:
        """
        Moment axes.

        Because this is intended to be used with apply_gufunc, the moment axes are moved to the end.
        """
        return cast("MomAxesStrict", tuple(range(-self.ndim, 0)))

    def to_array(self) -> MomParamsArray:
        return MomParamsArray(ndim=self.ndim, axes=self.axes)


@dataclass
class MomParamsXArrayOptional(MomParamsXArray):
    """Optional"""

    ndim: MomNDim | None  # type: ignore[assignment]
    dims: MomDimsStrict | None  # type: ignore[assignment]

    @classmethod
    def from_params(
        cls,
        ndim: int | None = None,
        dims: Hashable | Sequence[Hashable] | None = None,
        axes: int | Sequence[int] | None = None,
        data: xr.DataArray | xr.Dataset | None = None,
        default_ndim: MomNDim | None = None,
    ) -> Self:
        if ndim is None and dims is None and axes is None and default_ndim is None:
            return cls(None, None)
        return super().from_params(ndim, dims, axes, data, default_ndim)

    def to_array(self) -> MomParamsArrayOptional:
        return MomParamsArrayOptional(
            ndim=self.ndim,
            axes=None if self.ndim is None else self.axes,
        )

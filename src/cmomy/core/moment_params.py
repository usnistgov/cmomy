"""Helper class to work with moment parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, TypedDict, cast

from .array_utils import normalize_axis_index, normalize_axis_tuple
from .missing import MISSING
from .validate import (
    is_dataset,
    validate_mom,
    validate_mom_axes,
    validate_mom_dims_and_mom_ndim,
    validate_mom_ndim,
    validate_not_none,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Sequence
    from typing import Any

    import xarray as xr

    from .typing import (
        AxisReduceMultWrap,
        AxisReduceWrap,
        DimsReduce,
        DimsReduceMult,
        MissingType,
        MomAxesStrict,
        MomDimsStrict,
        MomentsStrict,
        MomNDim,
    )
    from .typing_compat import Self, TypeVar, Unpack

    _Axis = TypeVar("_Axis")
    _Dim = TypeVar("_Dim")


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

    def new_like(self, **kwargs: Any) -> Self:
        return replace(self, **kwargs)  # type: ignore[type-var]


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
class MomParamsBase:
    """Base class for moment parameters."""

    ndim: MomNDim


@dataclass
class MomParamsArray(MomParamsBase, _Mixin):
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
        mom_params: Self | MomParamsDict | None = None,
        ndim: int | None = None,
        axes: int | Sequence[int] | None = None,
        default_ndim: MomNDim | None = None,
    ) -> Self:
        if isinstance(mom_params, cls):
            return mom_params
        mom_params = {} if mom_params is None else MomParamsDict(mom_params)  # type: ignore[misc]
        if ndim is not None:
            mom_params["ndim"] = ndim
        mom_params.setdefault("axes", axes)
        return cls.from_params(**mom_params, default_ndim=default_ndim)

    @classmethod
    def factory_mom(
        cls,
        mom: int | Sequence[int],
        mom_params: Self | MomParamsDict | None = None,
        axes: int | Sequence[int] | None = None,
        default_ndim: MomNDim | None = None,
    ) -> tuple[MomentsStrict, Self]:
        mom = validate_mom(mom)
        return mom, cls.factory(
            ndim=len(mom), mom_params=mom_params, axes=axes, default_ndim=default_ndim
        )

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

    def raise_if_in_mom_axes(self, *axes: int) -> None:
        if self.axes is not None and any(a in self.axes for a in axes):  # pyright: ignore[reportUnnecessaryComparison]
            msg = f"provided axis/axes cannot overlap mom_axes={self.axes}."
            raise ValueError(msg)


@dataclass
class MomParamsArrayOptional(MomParamsArray):
    """Optional array mom params."""

    ndim: MomNDim | None = None  # type: ignore[assignment]
    axes: MomAxesStrict | None = None  # type: ignore[assignment]

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
class MomParamsXArray(MomParamsBase, _Mixin):
    """XArray mom parameters."""

    dims: MomDimsStrict

    @classmethod
    def from_params(
        cls,
        ndim: int | None = None,
        dims: Hashable | Sequence[Hashable] | None = None,
        axes: int | Sequence[int] | None = None,
        data: object = None,
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
        mom_params: Self | MomParamsDict | None = None,
        ndim: int | None = None,
        dims: Hashable | Sequence[Hashable] | None = None,
        axes: int | Sequence[int] | None = None,
        data: object = None,
        default_ndim: MomNDim | None = None,
    ) -> Self:
        if isinstance(mom_params, cls):
            return mom_params

        mom_params = {} if mom_params is None else MomParamsDict(mom_params)  # type: ignore[misc]
        if ndim is not None:
            mom_params["ndim"] = ndim
        mom_params.setdefault("dims", dims)
        mom_params.setdefault("axes", axes)
        return cls.from_params(**mom_params, data=data, default_ndim=default_ndim)

    @classmethod
    def factory_mom(
        cls,
        mom: int | Sequence[int],
        mom_params: Self | MomParamsDict | None = None,
        dims: Hashable | Sequence[Hashable] | None = None,
        axes: int | Sequence[int] | None = None,
        data: object = None,
        default_ndim: MomNDim | None = None,
    ) -> tuple[MomentsStrict, Self]:
        mom = validate_mom(mom)
        return mom, cls.factory(
            ndim=len(mom),
            mom_params=mom_params,
            dims=dims,
            axes=axes,
            data=data,
            default_ndim=default_ndim,
        )

    @property
    def axes(self) -> MomAxesStrict:
        """
        Moment axes.

        Because this is intended to be used with apply_gufunc, the moment axes are moved to the end.
        """
        return cast("MomAxesStrict", tuple(range(-self.ndim, 0)))

    def to_array(self) -> MomParamsArray:
        return MomParamsArray(ndim=self.ndim, axes=self.axes)

    def _check_dim_in_mom_dims(
        self,
        *,
        axis: int | None = None,
        dim: Hashable,
    ) -> None:
        if self.dims is not None and dim in self.dims:  # pyright: ignore[reportUnnecessaryComparison]
            axis_msg = f", {axis=}" if axis is not None else ""
            msg = f"Cannot select moment dimension. {dim=}{axis_msg}."
            raise ValueError(msg)

    @staticmethod
    def _axis_dim_defaults(
        *,
        axis: _Axis,
        dim: _Dim,
        default_axis: _Axis,
        default_dim: _Dim,
    ) -> tuple[_Axis, _Dim]:
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
        return axis, dim  # pyright: ignore[reportReturnType]

    def select_axis_dim(
        self,
        data: xr.DataArray | xr.Dataset,
        axis: AxisReduceWrap | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        *,
        default_axis: AxisReduceWrap | MissingType = MISSING,
        default_dim: DimsReduce | MissingType = MISSING,
    ) -> tuple[int, Hashable]:
        axis = validate_not_none(axis, "axis")
        dim = validate_not_none(dim, "dim")

        if is_dataset(data):
            if axis is not MISSING or dim is MISSING:
                msg = (
                    "For Dataset, must specify ``dim`` value other than ``None`` only."
                )
                raise ValueError(msg)
            self._check_dim_in_mom_dims(dim=dim)
            return 0, dim

        default_axis = validate_not_none(default_axis, "default_axis")
        default_dim = validate_not_none(default_dim, "default_dim")
        axis, dim = self._axis_dim_defaults(
            axis=axis, dim=dim, default_axis=default_axis, default_dim=default_dim
        )  # pyright: ignore[reportAssignmentType]

        if dim is not MISSING:
            axis = data.get_axis_num(dim)
        elif axis is not MISSING:
            axis = self.to_array().normalize_axis_index(
                axis=axis,  # type: ignore[arg-type]
                data_ndim=data.ndim,
            )
            dim = data.dims[axis]
        else:  # pragma: no cover
            msg = f"Unknown dim {dim} and axis {axis}"
            raise TypeError(msg)

        self._check_dim_in_mom_dims(dim=dim, axis=axis)
        return axis, dim

    def select_axis_dim_mult(
        self,
        data: xr.DataArray | xr.Dataset,
        axis: AxisReduceMultWrap | MissingType = MISSING,
        dim: DimsReduceMult | MissingType = MISSING,
        *,
        default_axis: AxisReduceMultWrap | MissingType = MISSING,
        default_dim: DimsReduceMult | MissingType = MISSING,
    ) -> tuple[tuple[int, ...], tuple[Hashable, ...]]:
        def _get_dim_none() -> tuple[Hashable, ...]:
            dim_ = tuple(data.dims)
            if self.dims is not None:  # pyright: ignore[reportUnnecessaryComparison]
                dim_ = tuple(d for d in dim_ if d not in self.dims)
            return dim_

        def _check_dim(dim_: tuple[Hashable, ...]) -> None:
            if self.dims is not None:  # pyright: ignore[reportUnnecessaryComparison]
                for d in dim_:
                    self._check_dim_in_mom_dims(dim=d)

        dim_: tuple[Hashable, ...]
        axis_: tuple[int, ...]
        if is_dataset(data):
            if axis is not MISSING or dim is MISSING:
                msg = "For Dataset, must specify ``dim`` value only."
                raise ValueError(msg)

            if dim is None:
                dim_ = _get_dim_none()
            else:
                dim_ = (dim,) if isinstance(dim, str) else tuple(dim)  # type: ignore[arg-type]
                _check_dim(dim_)
            return (), dim_

        axis, dim = self._axis_dim_defaults(
            axis=axis, dim=dim, default_axis=default_axis, default_dim=default_dim
        )  # pyright: ignore[reportAssignmentType]

        if dim is not MISSING:
            if dim is None:
                dim_ = _get_dim_none()
            else:
                dim_ = (dim,) if isinstance(dim, str) else tuple(dim)  # type: ignore[arg-type]
            axis_ = data.get_axis_num(dim_)
        elif axis is not MISSING:
            axis_ = self.to_array().normalize_axis_tuple(
                axis,
                data.ndim,
            )
            dim_ = tuple(data.dims[a] for a in axis_)
        else:  # pragma: no cover
            msg = f"Unknown dim {dim} and axis {axis}"
            raise TypeError(msg)

        _check_dim(dim_)
        return axis_, dim_


@dataclass
class MomParamsXArrayOptional(MomParamsXArray):
    """Optional"""

    ndim: MomNDim | None = None  # type: ignore[assignment]
    dims: MomDimsStrict | None = None  # type: ignore[assignment]

    @classmethod
    def from_params(
        cls,
        ndim: int | None = None,
        dims: Hashable | Sequence[Hashable] | None = None,
        axes: int | Sequence[int] | None = None,
        data: object = None,
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


default_mom_params_xarray = MomParamsXArrayOptional()

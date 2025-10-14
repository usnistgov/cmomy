"""Helper class to work with moment parameters."""

# pylint: disable=missing-class-docstring
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, ClassVar, TypedDict, cast, overload

from module_utilities.docfiller import DocFiller

from .array_utils import normalize_axis_index, normalize_axis_tuple
from .docstrings import docfiller as _docfiller
from .missing import MISSING
from .typing_compat import override
from .validate import (
    is_dataarray,
    is_dataset,
    is_xarray,
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
        AxesGUFunc,
        AxisReduceMultWrap,
        AxisReduceWrap,
        DataT,
        DimsReduce,
        DimsReduceMult,
        MissingType,
        MomAxesStrict,
        MomDimsStrict,
        MomentsStrict,
        MomNDim,
        NDArrayAny,
    )
    from .typing_compat import Self, TypeAlias, TypeVar

    _Axis = TypeVar("_Axis")
    _Dim = TypeVar("_Dim")

#: Moment parameter input types
MomParamsType: TypeAlias = "MomParams | MomParamsBase | MomParamsDict | None"


_docstring_local = """
Parameters
----------
axes : int or sequence of int
    Axes/Axis for moment dimension(s) (equivalent to ``mom_ndim``).
dims : str or sequence of hashable
    Name of moment dimensions for :mod:`xarray` objects (equivalent to ``mom_dims``).
ndim : {1, 2}
    Number of moment dimensions.
default_ndim : {1, 2}
    Fallback value for ``ndim``.
"""

docfiller = _docfiller.append(
    DocFiller.from_docstring(_docstring_local, combine_keys="parameters")
)


@docfiller.decorate
class MomParamsDict(TypedDict, total=False):
    """
    Input dict for moment parameters.

    Parameters
    ----------
    {axes}
    {dims}
    {ndim}
    """

    ndim: int | None
    axes: int | Sequence[int] | None
    dims: Hashable | Sequence[Hashable] | None


@dataclass(frozen=True)
class _MixinDataclass:
    def asdict(self) -> MomParamsDict:
        return cast("MomParamsDict", asdict(self))

    def new_like(self, **kwargs: Any) -> Self:
        """Create new object from key, value pairs."""
        return replace(self, **kwargs)


@dataclass(frozen=True)
@docfiller.decorate
class MomParams(_MixinDataclass):
    """
    Dataclass for moment parameters input

    Parameters
    ----------
    {ndim}
    {axes}
    {ndim}
    """

    ndim: int | None = None
    axes: int | Sequence[int] | None = None
    dims: Hashable | Sequence[Hashable] | None = None


_MOM_AXES_LAST: dict[MomNDim, MomAxesStrict] = {
    1: (-1,),
    2: (-2, -1),
}


@dataclass(frozen=True)
@docfiller.decorate
class MomParamsBase(ABC, _MixinDataclass):
    """Base class for moment parameters."""

    ndim: MomNDim | None

    def normalize_axis_index(
        self, axis: complex, data_ndim: int, msg_prefix: str | None = None
    ) -> int:
        """Normalize axis relative to ``self.ndim``"""
        return normalize_axis_index(
            axis=axis, ndim=data_ndim, mom_ndim=self.ndim, msg_prefix=msg_prefix
        )

    def normalize_axis_tuple(
        self,
        axis: complex | Iterable[complex],
        data_ndim: int,
        msg_prefix: str | None = None,
        allow_duplicate: bool = False,
    ) -> tuple[int, ...]:
        """Normalize axis tuple relative to ``self.ndim``."""
        return normalize_axis_tuple(
            axis=axis,
            ndim=data_ndim,
            mom_ndim=self.ndim,
            msg_prefix=msg_prefix,
            allow_duplicate=allow_duplicate,
        )

    @property
    def _validated_ndim(self) -> MomNDim:
        if self.ndim is None:
            msg = "Must set ndim"
            raise ValueError(msg)
        return self.ndim

    @property
    def axes_last(self) -> MomAxesStrict:
        return _MOM_AXES_LAST[self._validated_ndim]

    @abstractmethod
    def get_mom_shape(self, data: Any) -> MomentsStrict:
        pass  # pragma: no cover

    @abstractmethod
    def get_mom(self, data: Any) -> MomentsStrict:
        pass  # pragma: no cover

    @abstractmethod
    def get_val_shape(self, data: Any) -> tuple[int, ...]:
        pass  # pragma: no cover

    @abstractmethod
    def axes_to_end(self) -> Self:
        pass  # pragma: no cover

    def check_data(self, data: Any) -> None:
        # NOTE: Not an ideal solution, but get a bunch
        # of possible errors if don't do this.
        try:
            _ = self.get_mom(data)
        except Exception as e:
            msg = f"{self} inconsistent with wrapped object"
            raise ValueError(msg) from e


# ArrayOptional -> normalize_axes, normalize_axis_tuple, axes, axes_last, raise_if_in_mom_axes, normalize_axis_index
@dataclass(frozen=True)
class MomParamsArrayOptional(MomParamsBase):
    """Array moment parameters."""

    ndim: MomNDim | None
    axes: MomAxesStrict | None
    _OPTIONAL: ClassVar[bool] = True

    @classmethod
    @docfiller.decorate
    def factory(
        cls,
        mom_params: MomParamsType = None,
        ndim: int | None = None,
        axes: int | Sequence[int] | None = None,
        default_ndim: MomNDim | None = None,
    ) -> Self:
        """
        Factory method.

        Parameters
        ----------
        {mom_params}
        {ndim}
        {axes}
        {default_ndim}
        """
        if isinstance(mom_params, cls):
            if ndim is not None:
                assert mom_params.ndim == ndim  # noqa: S101
            return mom_params

        if isinstance(mom_params, (MomParams, MomParamsBase)):
            mom_params = mom_params.asdict()
        elif mom_params is None:
            mom_params = MomParamsDict()

        if ndim is not None:
            mom_params["ndim"] = ndim
        _ = mom_params.setdefault("axes", axes)
        return cls.from_params(**mom_params, default_ndim=default_ndim)

    @classmethod
    def from_params(
        cls,
        ndim: int | None = None,
        axes: int | Sequence[int] | None = None,
        default_ndim: MomNDim | None = None,
        dims: Any = None,  # noqa: ARG003  # in case pass dims parameter, it will be ignored.
    ) -> Self:
        """Create from parameters."""
        if cls._OPTIONAL and ndim is None and axes is None and default_ndim is None:
            return cls(None, None)

        if axes is None:
            ndim = validate_mom_ndim(ndim, default_ndim)
            axes = _MOM_AXES_LAST[ndim]
            return cls(ndim=ndim, axes=axes)

        axes = validate_mom_axes(axes)
        if ndim is None:
            ndim = len(axes)
        elif len(axes) != ndim:
            msg = f"{len(axes)=} != {ndim=}"
            raise ValueError(msg)
        return cls(ndim=cast("MomNDim", ndim), axes=axes)

    def raise_if_in_mom_axes(self, *axes: int) -> None:
        """Raise ``ValueError`` if any ``axes`` in ``self.axes``."""
        # these ignores needed because inherit into optional classes below
        if self.axes is not None and any(a in self.axes for a in axes):
            msg = f"provided axis/axes cannot overlap mom_axes={self.axes}."
            raise ValueError(msg)

    @property
    def _validated_axes(self) -> MomAxesStrict:
        if self.axes is None:
            msg = "Must set axes"
            raise ValueError(msg)
        return self.axes

    @override
    def get_mom_shape(self, data: NDArrayAny) -> MomentsStrict:
        """Calculate moment shape from data shape"""
        try:
            return cast(
                "MomentsStrict", tuple(data.shape[a] for a in self._validated_axes)
            )
        except Exception as e:
            msg = "Could not extract moment shape from data"
            raise ValueError(msg) from e

    @override
    def axes_to_end(self) -> Self:
        """Create new object with ``self.axes`` at end."""
        return replace(self, axes=self.axes_last)

    @override
    def get_mom(self, data: NDArrayAny) -> MomentsStrict:
        from .utils import mom_shape_to_mom

        return mom_shape_to_mom(data.shape[a] for a in self._validated_axes)

    @override
    def get_val_shape(self, data: NDArrayAny) -> tuple[int, ...]:
        axes = self.normalize_axis_tuple(self._validated_axes, data.ndim)
        return tuple(s for i, s in enumerate(data.shape) if i not in axes)

    def normalize_axes(self, data_ndim: int) -> Self:
        """Normalize self.axes in new object relative to ``data_ndim``."""
        if self.axes is None:
            return self

        return replace(
            self,
            axes=normalize_axis_tuple(  # type: ignore[arg-type]
                self.axes, data_ndim, msg_prefix="normalize_axes"
            ),
        )

    def maybe_reorder_dataarray(self, x: DataT) -> DataT:
        """Reorder DataArray mom_axes_last to mom_axes."""
        if is_dataarray(x) and self._validated_axes != self.axes_last:  # type: ignore[redundant-expr]
            from .array_utils import reorder

            return x.transpose(*reorder(x.dims, self.axes_last, self._validated_axes))
        return x


@dataclass(frozen=True)
class MomParamsArray(MomParamsArrayOptional):
    """Array Mom Params"""

    ndim: MomNDim
    axes: MomAxesStrict
    _OPTIONAL: ClassVar[bool] = False

    @property
    @override
    def _validated_ndim(self) -> MomNDim:
        return self.ndim

    @property
    @override
    def _validated_axes(self) -> MomAxesStrict:
        return self.axes

    @classmethod
    @docfiller.decorate
    def factory_mom(
        cls,
        mom: int | Sequence[int],
        mom_params: MomParamsType = None,
        axes: int | Sequence[int] | None = None,
        default_ndim: MomNDim | None = None,
    ) -> tuple[MomentsStrict, Self]:
        """
        Factory method to validate ``mom`` and create ``mom_params`` object.

        Parameters
        ----------
        {mom}
        {mom_params}
        {axes}
        {default_ndim}
        """
        mom = validate_mom(mom)
        return mom, cls.factory(
            ndim=len(mom), mom_params=mom_params, axes=axes, default_ndim=default_ndim
        )

    def axes_data_reduction(
        self,
        *inner: int | tuple[int, ...],
        axis: int,
        out_has_axis: bool = False,
    ) -> AxesGUFunc:
        """
        Axes for reducing data along axis

        if ``out_has_axis == True``, then treat like resample,
        so output will still have ``axis`` with new size in output.

        It is assumed that `axis` is validated against a moments array,
        (i.e., negative values should be ``< -mom_ndim``)

        Can also pass in "inner" dimensions (elements 1:-1 of output)
        """
        data_axes = (axis, *self.axes)
        out_axes = data_axes if out_has_axis else self.axes

        return [
            data_axes,
            *((x,) if isinstance(x, int) else x for x in inner),
            out_axes,
        ]


# XArrayOptional -> select_axis_dim_mult, get_axes, axes_last, dims
@dataclass(frozen=True)
class MomParamsXArrayOptional(MomParamsBase):
    """Optional moment parameters for xarray objects."""

    ndim: MomNDim | None
    dims: MomDimsStrict | None
    _OPTIONAL: ClassVar[bool] = True

    @classmethod
    @docfiller.decorate
    def factory(
        cls,
        mom_params: MomParamsType = None,
        ndim: int | None = None,
        dims: Hashable | Sequence[Hashable] | None = None,
        axes: int | Sequence[int] | None = None,
        data: object = None,
        default_ndim: MomNDim | None = None,
    ) -> Self:
        """
        Factory create object

        Parameters
        ----------
        {mom_params}
        {ndim}
        {dims}
        {axes}
        {data}
        {default_ndim}
        """
        if isinstance(mom_params, cls):
            if ndim is not None:
                assert mom_params.ndim == ndim  # noqa: S101
            return mom_params

        if isinstance(mom_params, (MomParams, MomParamsBase)):
            mom_params = mom_params.asdict()
        elif mom_params is None:
            mom_params = MomParamsDict()

        if ndim is not None:
            mom_params["ndim"] = ndim
        _ = mom_params.setdefault("dims", dims)
        _ = mom_params.setdefault("axes", axes)
        return cls.from_params(**mom_params, data=data, default_ndim=default_ndim)

    @classmethod
    def from_params(
        cls,
        ndim: int | None = None,
        dims: Hashable | Sequence[Hashable] | None = None,
        axes: int | Sequence[int] | None = None,
        data: object = None,
        default_ndim: MomNDim | None = None,
    ) -> Self:
        """Create from parameters."""
        if (
            cls._OPTIONAL
            and ndim is None
            and dims is None
            and axes is None
            and default_ndim is None
        ):
            return cls(None, None)

        dims, ndim = validate_mom_dims_and_mom_ndim(
            mom_dims=dims,
            mom_ndim=ndim,
            out=data,
            mom_ndim_default=default_ndim,
            mom_axes=axes,
        )
        return cls(ndim=ndim, dims=dims)

    @property
    def _validated_dims(self) -> MomDimsStrict:
        if self.dims is None:
            msg = "Must set dims"
            raise ValueError(msg)
        return self.dims

    @override
    def axes_to_end(self) -> Self:
        """Create new object with ``self.axes`` at end."""
        _ = self._validated_ndim
        return self

    def get_axes(self, data: xr.DataArray | None = None) -> MomAxesStrict:
        """
        Moment axes.

        If pass data, return ``data.get_axis_num(self.dims)``.  Otherwise, return ``self.axes_last``.
        """
        if data is None:
            return self.axes_last
        return cast("MomAxesStrict", data.get_axis_num(self._validated_dims))

    @override
    def get_mom_shape(self, data: xr.DataArray | xr.Dataset) -> MomentsStrict:
        """Calculate moment shape from data shape"""
        try:
            return cast(
                "MomentsStrict", tuple(data.sizes[d] for d in self._validated_dims)
            )
        except Exception as e:
            msg = "Could not extract moment shape from data"
            raise ValueError(msg) from e

    @override
    def get_mom(self, data: xr.DataArray | xr.Dataset) -> MomentsStrict:
        from .utils import mom_shape_to_mom

        return mom_shape_to_mom(data.sizes[d] for d in self._validated_dims)

    @override
    def get_val_shape(self, data: xr.DataArray) -> tuple[int, ...]:
        return tuple(data.sizes[d] for d in data.dims if d not in self._validated_dims)

    # * Select
    def _raise_if_dim_in_mom_dims(
        self,
        *,
        axis: int | None = None,
        dim: Hashable | None,
    ) -> None:
        # these ignores needed because inherit into optional classes below
        if self.dims is not None and dim in self.dims:
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

    def select_axis_dim_mult(  # noqa: C901
        self,
        data: xr.DataArray | xr.Dataset,
        axis: AxisReduceMultWrap | MissingType = MISSING,
        dim: DimsReduceMult | MissingType = MISSING,
        *,
        default_axis: AxisReduceMultWrap | MissingType = MISSING,
        default_dim: DimsReduceMult | MissingType = MISSING,
        allow_select_mom_axes: bool = False,
    ) -> tuple[tuple[int, ...], tuple[Hashable, ...]]:
        def _get_dim_none() -> tuple[Hashable, ...]:
            dim_ = tuple(data.dims)
            if self.dims is not None:
                dim_ = tuple(d for d in dim_ if d not in self.dims)
            return dim_

        def _check_dim(dim_: tuple[Hashable, ...]) -> None:
            if allow_select_mom_axes:
                return
            if self.dims is not None:
                for d in dim_:
                    self._raise_if_dim_in_mom_dims(dim=d)

        dim_: tuple[Hashable, ...]
        axis_: tuple[int, ...]
        if is_dataset(data):
            if axis is not MISSING or dim is MISSING:
                msg = "For Dataset, must specify ``dim`` value only."
                raise ValueError(msg)

            if dim is None:
                dim_ = _get_dim_none()
            else:
                dim_ = (dim,) if isinstance(dim, str) else tuple(dim)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
                _check_dim(dim_)
            return (), dim_

        axis, dim = self._axis_dim_defaults(
            axis=axis, dim=dim, default_axis=default_axis, default_dim=default_dim
        )

        if axis is None or dim is None:
            dim_ = _get_dim_none()
            axis_ = data.get_axis_num(dim_)
        elif dim is not MISSING:
            dim_ = (dim,) if isinstance(dim, str) else tuple(dim)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
            axis_ = data.get_axis_num(dim_)
        elif axis is not MISSING:
            axis_ = self.normalize_axis_tuple(
                axis,
                data.ndim,
            )
            dim_ = tuple(data.dims[a] for a in axis_)
        else:  # pragma: no cover
            msg = f"Unknown dim {dim} and axis {axis}"
            raise TypeError(msg)

        _check_dim(dim_)
        return axis_, dim_

    def select_axis_dim(
        self,
        data: xr.DataArray | xr.Dataset,
        axis: AxisReduceWrap | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        *,
        default_axis: AxisReduceWrap | MissingType = MISSING,
        default_dim: DimsReduce | MissingType = MISSING,
        allow_select_mom_axes: bool = False,
    ) -> tuple[int, Hashable]:
        axis = validate_not_none(axis, "axis")
        dim = validate_not_none(dim, "dim")

        if is_dataset(data):
            if axis is not MISSING or dim is MISSING:
                msg = (
                    "For Dataset, must specify ``dim`` value other than ``None`` only."
                )
                raise ValueError(msg)
            if not allow_select_mom_axes:
                self._raise_if_dim_in_mom_dims(dim=dim)
            return 0, dim

        default_axis = validate_not_none(default_axis, "default_axis")
        default_dim = validate_not_none(default_dim, "default_dim")
        axis, dim = self._axis_dim_defaults(
            axis=axis, dim=dim, default_axis=default_axis, default_dim=default_dim
        )

        if dim is not MISSING:
            axis = data.get_axis_num(dim)
        elif axis is not MISSING:
            axis = self.normalize_axis_index(
                axis=axis,  # type: ignore[arg-type]
                data_ndim=data.ndim,
            )
            dim = data.dims[axis]
        else:  # pragma: no cover
            msg = f"Unknown dim {dim} and axis {axis}"
            raise TypeError(msg)

        if not allow_select_mom_axes:
            self._raise_if_dim_in_mom_dims(dim=dim, axis=axis)
        return axis, dim


@dataclass(frozen=True)
class MomParamsXArray(MomParamsXArrayOptional):
    """Moment parameters for xarray objects"""

    ndim: MomNDim
    dims: MomDimsStrict
    _OPTIONAL: ClassVar[bool] = False

    @property
    @override
    def _validated_ndim(self) -> MomNDim:
        return self.ndim

    @property
    @override
    def _validated_dims(self) -> MomDimsStrict:
        return self.dims

    @classmethod
    def factory_mom(
        cls,
        mom: int | Sequence[int],
        mom_params: MomParamsType = None,
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

    def to_array(self, data: xr.DataArray | None = None) -> MomParamsArray:
        """
        Convert to MomParamsArray object.

        Axes is ``self.get_axes(data)``.
        """
        return MomParamsArray(ndim=self.ndim, axes=self.get_axes(data))

    def core_dims(self, *dims: Hashable) -> tuple[Hashable, ...]:
        """Core dimensions (*dims, *self.dims)"""
        return (*dims, *self.dims)


default_mom_params_xarray = MomParamsXArrayOptional(None, None)


@overload
def factory_mom_params(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    target: xr.DataArray | xr.Dataset,
    *,
    mom_params: MomParamsType = ...,
    ndim: int | None = ...,
    axes: int | Sequence[int] | None = ...,
    dims: Hashable | Sequence[Hashable] | None = ...,
    data: object = ...,
    default_ndim: MomNDim | None = ...,
) -> MomParamsXArray: ...
@overload
def factory_mom_params(
    target: object,
    *,
    mom_params: MomParamsType = ...,
    ndim: int | None = ...,
    axes: int | Sequence[int] | None = ...,
    dims: Hashable | Sequence[Hashable] | None = ...,
    data: object = ...,
    default_ndim: MomNDim | None = ...,
) -> MomParamsArray: ...


@docfiller.decorate
def factory_mom_params(
    target: object | xr.DataArray | xr.Dataset,
    *,
    mom_params: MomParamsType = None,
    ndim: int | None = None,
    axes: int | Sequence[int] | None = None,
    dims: Hashable | Sequence[Hashable] | None = None,
    data: object = None,
    default_ndim: MomNDim | None = None,
) -> MomParamsArray | MomParamsXArray:
    """
    Factory method to create mom_params.

    Parameters
    ----------
    target : array-like or DataArray or Dataset
        Return object corresponding to data type of ``target``.
    {mom_params}
    {ndim}
    {axes}
    {dims}
    data : array-like or DataArray or Dataset, optional
        Optional data to use as template to extract ``ndim`` using ``axes`` or ``ndim``.
    {default_ndim}

    Returns
    -------
    MomParamsArray or MomParamsXArray
        Moment parameters object.
    """
    if is_xarray(target):
        return MomParamsXArray.factory(
            mom_params=mom_params,
            ndim=ndim,
            axes=axes,
            dims=dims,
            data=data,
            default_ndim=default_ndim,
        )
    return MomParamsArray.factory(
        mom_params=mom_params, ndim=ndim, axes=axes, default_ndim=default_ndim
    )

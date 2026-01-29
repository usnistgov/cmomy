"""
Typing aliases (:mod:`cmomy.core.typing`)
=========================================
"""
# pyright: deprecateTypingAliases=false
# pylint: disable=missing-class-docstring,consider-alternative-union-syntax

from __future__ import annotations

from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    Union,
    runtime_checkable,
)

import numpy as np
import pandas as pd  # noqa: F401

# put outside to get autodoc typehints working...
import xarray as xr
from numpy.typing import NDArray

from .typing_compat import EllipsisType, TypeVar

if TYPE_CHECKING:
    from .missing import _Missing  # pyright: ignore[reportPrivateUsage]
    from .typing_compat import TypeAlias
    from .typing_nested_sequence import (
        _NestedSequence,  # pyright: ignore[reportPrivateUsage]  # noqa: F401
    )

    # Missing value type
    MissingType: TypeAlias = Literal[_Missing.MISSING]

# * TypeVars ------------------------------------------------------------------
#: DataArray or Dataset
DataT = TypeVar("DataT", xr.DataArray, xr.Dataset)
DataArrayOrSetT = TypeVar("DataArrayOrSetT", bound="xr.DataArray | xr.Dataset")

#: TypeVar of floating point precision (np.float32, np.float64, default=Any)
FloatT = TypeVar(
    "FloatT",
    np.float64,
    np.float32,
)
FloatT_ = TypeVar("FloatT_", np.float64, np.float32)

#: TypeVar of types wrapped by IndexSampler
SamplerArrayT = TypeVar(
    "SamplerArrayT",
    "NDArrayAny",
    xr.DataArray,
    xr.Dataset,
    "xr.DataArray | xr.Dataset",
)


# * Numpy ---------------------------------------------------------------------
# Axis/Dim reduction type
# TODO(wpk): convert int -> SupportsIndex?
#: Axes type
Axes: TypeAlias = int | tuple[int, ...]
#: Axes type (with wrapping)
AxesWrap: TypeAlias = complex | tuple[complex, ...]

#: Reduction axes type
AxisReduce: TypeAlias = int | None
#: Reduction axes type (with wrapping)
AxisReduceWrap: TypeAlias = complex | None

AxesGUFunc: TypeAlias = list[tuple[int, ...]]

#: Reduction axes type (multiple)
AxisReduceMult: TypeAlias = int | tuple[int, ...] | None
#: Reduction axes type (multiple, with wrapping)
AxisReduceMultWrap: TypeAlias = complex | tuple[complex, ...] | None

#: Random number generator types
RngTypes: TypeAlias = (
    int
    | Sequence[int]
    | np.random.SeedSequence
    | np.random.BitGenerator
    | np.random.Generator
)

# Types
#: Any dtype
DTypeAny: TypeAlias = Any
#: Floating dtype
FloatDTypes: TypeAlias = np.float32 | np.float64
#: Long integer dtype
LongIntDType: TypeAlias = np.int64
#: Array type (any dtype)
NDArrayAny: TypeAlias = NDArray[DTypeAny]
#: Array type (int dtype)
NDArrayInt = NDArray[np.int64]
#: Array type (float dtype)
NDArrayFloats = NDArray[FloatDTypes]
#: Array type (bool dtype)
NDArrayBool = NDArray[np.bool_]
#: Integer dtype
IntDTypeT: TypeAlias = np.int64
#: Float or array of float
NDGeneric: TypeAlias = FloatT | NDArray[FloatT]


# ** Dtype
_ScalarT = TypeVar("_ScalarT", bound=np.generic)
_DTypeT_co = TypeVar("_DTypeT_co", covariant=True, bound="np.dtype[Any]")


@runtime_checkable
class _SupportsDType(Protocol[_DTypeT_co]):
    @property
    def dtype(self) -> _DTypeT_co: ...  # pragma: no cover


DTypeLikeArg = Union[
    "np.dtype[_ScalarT]",
    "type[_ScalarT]",
    "_SupportsDType[np.dtype[_ScalarT]]",
]


# ** ArrayLike
@runtime_checkable
class _SupportsArray(Protocol[_DTypeT_co]):
    def __array__(self) -> np.ndarray[Any, _DTypeT_co]: ...  # noqa: PLW3201  # pragma: no cover


ArrayLikeArg: TypeAlias = (
    _SupportsArray[np.dtype[_ScalarT]]
    | "_NestedSequence[_SupportsArray[np.dtype[_ScalarT]]]"
)


# * Moments -------------------------------------------------------------------
# NOTE: using the strict form for Moments
# Passing in integer or Sequence[int] will work in almost all cases,
# but will be flagged by typechecker...
#: Moments type
Moments: TypeAlias = int | tuple[int] | tuple[int, int]
#: Strict moments type
MomentsStrict: TypeAlias = tuple[int] | tuple[int, int]
#: Number of moment dimensions
MomNDim = Literal[1, 2]

#: Axes containing moment(s).
MomAxes = Moments
#: Axes containing moment(s).
MomAxesStrict = MomentsStrict

# * Xarray specific stuff -----------------------------------------------------
# fix if using autodoc typehints...
#: Reduction dimension
DimsReduce: TypeAlias = Hashable | None
#: Reduction dimension(s)
DimsReduceMult: TypeAlias = Union[Hashable, "Collection[Hashable]", None]
#: Dimensions
Dims: TypeAlias = str | Collection[Hashable] | EllipsisType | None

#: Dimensions containing moment(s)
MomDims: TypeAlias = Hashable | tuple[Hashable] | tuple[Hashable, Hashable]
#: Dimensions containing moment(s)
MomDimsStrict: TypeAlias = tuple[Hashable] | tuple[Hashable, Hashable]

#: Index
IndexAny: TypeAlias = "pd.Index[Any]"
#: Coordinates
CoordsType: TypeAlias = Mapping[Any, Any] | None
AttrsType: TypeAlias = Mapping[Any, Any] | None
NameType: TypeAlias = Hashable | None
DimsType: TypeAlias = str | Iterable[Hashable] | None
KeepAttrs: TypeAlias = (
    Literal["drop", "identical", "no_conflicts", "drop_conflicts", "override"]
    | bool
    | None
)
Groups: TypeAlias = "Sequence[Any] | NDArrayAny | IndexAny | pd.MultiIndex"


# * Literals ------------------------------------------------------------------
#: Order parameters
ArrayOrderCF: TypeAlias = Literal["C", "F"] | None
#: Order parameters
ArrayOrderACF: TypeAlias = Literal["A", "C", "F"] | None
#: Order parameters
ArrayOrderKACF: TypeAlias = Literal["K", "A", "C", "F"] | None
#: Casting rules
Casting = Literal["no", "equiv", "safe", "same_kind", "unsafe"]
#: What to do if missing a core dimensions.
MissingCoreDimOptions = Literal["raise", "copy", "drop"]
#: Moment names.
SelectMoment = Literal[
    "weight",
    "ave",
    "cov",
    "var",
    "xave",
    "yave",
    "xvar",
    "yvar",
    "all",
    "xmom_0",
    "xmom_1",
    "ymom_0",
    "ymom_1",
]
#: Style to convert to
ConvertStyle = Literal["central", "raw"]
VerifyValuesStyles = Literal["val", "vals", "data", "datas", "var", "vars"]
CoordsPolicy: TypeAlias = Literal["first", "last", "group"] | None
BootStrapMethod = Literal["percentile", "basic", "bca"]
BlockByModes = Literal["drop_first", "drop_last", "expand_first", "expand_last"]

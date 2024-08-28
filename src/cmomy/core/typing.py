"""
Typing aliases (:mod:`cmomy.core.typing`)
=========================================
"""

from __future__ import annotations

from collections.abc import Collection, Hashable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

import numpy as np

# put outside to get autodoc typehints working...
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from .typing_compat import EllipsisType, TypeVar

if TYPE_CHECKING:
    from .missing import _Missing  # pyright: ignore[reportPrivateUsage]
    from .typing_compat import TypeAlias
    from .typing_nested_sequence import (
        _NestedSequence,  # pyright: ignore[reportPrivateUsage]
    )

    # Missing value type
    MissingType: TypeAlias = Literal[_Missing.MISSING]

# * TypeVars
GenArrayT = TypeVar("GenArrayT", NDArray[Any], xr.DataArray, xr.Dataset)
GenXArrayT = TypeVar("GenXArrayT", xr.DataArray, xr.Dataset)
GenXArrayT2 = TypeVar("GenXArrayT2", xr.DataArray, xr.Dataset)
DataArrayOrSetT = TypeVar("DataArrayOrSetT", bound=Union[xr.DataArray, xr.Dataset])

#: TypeVar of array types with restriction
ArrayT = TypeVar(  # type: ignore[misc]
    "ArrayT",
    NDArray[np.float32],
    NDArray[np.float64],
    xr.DataArray,
    default=NDArray[np.float64],
)

FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)

#: TypeVar of floating point precision (np.float32, np.float64, default=Any)
FloatT = TypeVar(  # type: ignore[misc]
    "FloatT",
    np.float32,
    np.float64,
    default=Any,  # pyright: ignore[reportGeneralTypeIssues]
)
FloatT2 = TypeVar("FloatT2", np.float32, np.float64)

#: TypeVar of for np.generic dtype.
ScalarT = TypeVar("ScalarT", bound=np.generic)
ScalarT2 = TypeVar("ScalarT2", bound=np.generic)

FloatingT = TypeVar("FloatingT", bound="np.floating[Any]")
DTypeT_co = TypeVar("DTypeT_co", covariant=True, bound="np.dtype[Any]")
NDArrayT = TypeVar("NDArrayT", bound="NDArray[Any]")


# * Aliases
# ** Numpy
# Axis/Dim reduction type
# TODO(wpk): convert int -> SupportsIndex?
AxisReduce: TypeAlias = Union[int, None]
DimsReduce: TypeAlias = Union[Hashable, None]
AxesGUFunc: TypeAlias = "list[tuple[int, ...]]"
AxisReduceMult: TypeAlias = Union[int, "tuple[int, ...]", None]
DimsReduceMult: TypeAlias = Union[Hashable, "Collection[Hashable]", None]

# Types
DTypeAny: TypeAlias = Any
FloatDTypes = Union[np.float32, np.float64]
LongIntDType: TypeAlias = np.int64
NDArrayAny: TypeAlias = NDArray[DTypeAny]
NDArrayFloats = NDArray[FloatDTypes]
NDArrayBool = NDArray[np.bool_]
NDArrayInt = NDArray[LongIntDType]
IntDTypeT: TypeAlias = np.int64
NDGeneric: TypeAlias = Union[FloatT, NDArray[FloatT]]


# ** Dtype
@runtime_checkable
class _SupportsDType(Protocol[DTypeT_co]):
    @property
    def dtype(self) -> DTypeT_co: ...  # pragma: no cover


DTypeLikeArg = Union[
    "np.dtype[ScalarT]",
    "type[ScalarT]",
    "_SupportsDType[np.dtype[ScalarT]]",
]


# ** ArrayLike
@runtime_checkable
class _SupportsArray(Protocol[DTypeT_co]):
    def __array__(self) -> np.ndarray[Any, DTypeT_co]: ...  # noqa: PLW3201  # pragma: no cover


ArrayLikeArg = Union[
    "_SupportsArray[np.dtype[ScalarT]]",
    "_NestedSequence[_SupportsArray[np.dtype[ScalarT]]]",
]


# * Numba types
# NumbaType = Union[nb.typing.Integer, nb.typing.Array]  # noqa: ERA001
# The above isn't working for pyright.  Just using any for now...
NumbaType = Any

# * Moments
# NOTE: using the strict form for Moments
# Passing in integer or Sequence[int] will work in almost all cases,
# but will be flagged by typechecker...
#: Moments type
Moments: TypeAlias = Union[int, "tuple[int]", "tuple[int, int]"]
MomentsStrict: TypeAlias = Union["tuple[int]", "tuple[int, int]"]
Mom_NDim = Literal[1, 2]

# * Xarray specific stuff
# fix if using autodoc typehints...

MomDims = Union[Hashable, "tuple[Hashable]", "tuple[Hashable, Hashable]"]
MomDimsStrict = Union["tuple[Hashable]", "tuple[Hashable, Hashable]"]

IndexAny: TypeAlias = "pd.Index[Any]"
XArrayCoordsType: TypeAlias = Union[
    Mapping[Any, Any],
    None,
]
XArrayAttrsType: TypeAlias = Optional[Mapping[Any, Any]]
XArrayNameType: TypeAlias = Optional[Hashable]
XArrayDimsType: TypeAlias = Union[Hashable, Sequence[Hashable], None]
XArrayIndexesType: TypeAlias = Any
Dims = Union[str, Collection[Hashable], EllipsisType, None]  # pyright: ignore[reportGeneralTypeIssues]
KeepAttrs: TypeAlias = Union[
    Literal["drop", "identical", "no_conflicts", "drop_conflicts", "override"],
    bool,
    None,
]
Groups: TypeAlias = Union[Sequence[Any], NDArrayAny, IndexAny, pd.MultiIndex]
ApplyUFuncKwargs: TypeAlias = Mapping[str, Any]

# * Literals
ArrayOrder = Literal["C", "F", "A", "K", None]
ArrayOrderCFA = Literal["C", "F", "A", None]
ArrayOrderCF = Literal["C", "F", None]
Casting = Literal["no", "equiv", "safe", "same_kind", "unsafe"]
#: What to do if missing a core dimensions.
MissingCoreDimOptions = Literal["raise", "copy", "drop"]
#: Selectable moment names.
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
]
ConvertStyle = Literal["central", "raw"]
VerifyValuesStyles: TypeAlias = Literal["val", "vals", "data", "datas", "var", "vars"]
CoordsPolicy: TypeAlias = Literal["first", "last", "group", None]
BootStrapMethod: TypeAlias = Literal["percentile", "basic", "bca"]

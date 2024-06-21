"""
Typing aliases (:mod:`cmomy.typing`)
====================================
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

# import numba as nb
# put outside to get autodoc typehints working...
import pandas as pd  # noqa: TCH002
import xarray as xr
from numpy.typing import NDArray

from ._typing_compat import TypeVar

if TYPE_CHECKING:
    from ._typing_compat import TypeAlias
    from ._typing_nested_sequence import (
        _NestedSequence,  # pyright: ignore[reportPrivateUsage]
    )
    from ._utils import _Missing  # pyright: ignore[reportPrivateUsage]

    # Missing value type
    MissingType: TypeAlias = Literal[_Missing.MISSING]


# Axis/Dim reduction type
AxisReduce = Union[int, None]
DimsReduce = Union[Hashable, None]

# * Numpy Arrays
# ** Aliases
DTypeAny: TypeAlias = Any
FloatDTypes = Union[np.float32, np.float64]
LongIntDType: TypeAlias = np.int64
NDArrayAny: TypeAlias = NDArray[DTypeAny]
NDArrayFloats = NDArray[FloatDTypes]
NDArrayInt = NDArray[LongIntDType]
# ** Types
FloatT = TypeVar(  # type: ignore[misc]
    "FloatT",
    np.float32,
    np.float64,
    default=Any,  # pyright: ignore[reportGeneralTypeIssues]
)
"""TypeVar of floating point precision (np.float32, np.float64, default=Any)"""

FloatT2 = TypeVar("FloatT2", np.float32, np.float64)
DTypeT_co = TypeVar("DTypeT_co", covariant=True, bound="np.dtype[Any]")
ScalarT = TypeVar("ScalarT", bound=np.generic)
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


# ** Literals
ArrayOrder = Literal["C", "F", "A", "K", None]
ArrayOrderCFA = Literal["C", "F", "A", None]
ArrayOrderCF = Literal["C", "F", None]
DataCasting = Literal["no", "equiv", "safe", "same_kind", "unsafe", None]


# * Numba types
# NumbaType = Union[nb.typing.Integer, nb.typing.Array]
# The above isn't working for pyright.  Just using any for now...
NumbaType = Any

# * Moments
# NOTE: using the strict form for Moments
# Passing in integer or Sequence[int] will work in almost all cases,
# but will be flagged by typechecker...
if TYPE_CHECKING:
    Moments: TypeAlias = Union[int, tuple[int], tuple[int, int]]
    MomentsStrict: TypeAlias = Union[tuple[int], tuple[int, int]]

Mom_NDim = Literal[1, 2]

# * Generic array
ArrayT = TypeVar(  # type: ignore[misc]
    "ArrayT",
    NDArray[np.float32],
    NDArray[np.float64],
    xr.DataArray,
    default=NDArray[np.float64],
)
"""TypeVar of array type."""

# * Dummy function
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)

# * Xarray specific stuff
# fix if using autodoc typehints...

if TYPE_CHECKING:
    MomDims = Union[Hashable, tuple[Hashable], tuple[Hashable, Hashable]]
    MomDimsStrict = Union[tuple[Hashable], tuple[Hashable, Hashable]]

    IndexAny: TypeAlias = "pd.Index[Any]"
    XArrayCoordsType: TypeAlias = Union[
        Sequence[Union[Sequence[Any], IndexAny, xr.DataArray]],
        Mapping[Any, Any],
        None,
    ]

    XArrayAttrsType: TypeAlias = Optional[Mapping[Any, Any]]
    XArrayNameType: TypeAlias = Optional[Hashable]
    XArrayDimsType: TypeAlias = Union[Hashable, Sequence[Hashable], None]
    XArrayIndexesType: TypeAlias = Any

    Dims = Union[str, Collection[Hashable], ellipsis, None]  # noqa: F821  # pyright: ignore[reportGeneralTypeIssues]

    # literals
    VerifyValuesStyles: TypeAlias = Literal[
        "val", "vals", "data", "datas", "var", "vars"
    ]
    CoordsPolicy: TypeAlias = Literal["first", "last", "group", None]
    KeepAttrs: TypeAlias = Union[
        Literal["drop", "identical", "no_conflicts", "drop_conflicts", "override"],
        bool,
        None,
    ]

    ConvertStyle = Literal["central", "raw"]

    # For indexed reducetion
    Groups: TypeAlias = Union[Sequence[Any], NDArrayAny, IndexAny, pd.MultiIndex]

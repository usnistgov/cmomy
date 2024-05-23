"""
Typing aliases (:mod:`cmomy.typing`)
====================================
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np

# import numba as nb
# put outside to get autodoc typehints working...
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike, NDArray

from ._typing_compat import TypeVar

if TYPE_CHECKING:
    from ._typing_compat import TypeAlias
    # from .central_abc import CentralMomentsABC


# Arrays
DTypeAny: TypeAlias = Any
NDArrayAny: TypeAlias = NDArray[DTypeAny]
ArrayOrder = Literal["C", "F", "A", "K", None]
ArrayOrderCFA = Literal["C", "F", "A", None]
ArrayOrderCF = Literal["C", "F", None]
DataCasting = Literal["no", "equiv", "safe", "same_kind", "unsafe", None]

T_FloatDType = TypeVar("T_FloatDType", np.float32, np.float64, default=np.float64)  # type: ignore[misc]  # something off with default
T_FloatDType_co = TypeVar(  # type: ignore[misc]  # something off with default
    "T_FloatDType_co", np.float32, np.float64, covariant=True, default=np.float64
)
T_FloatDType2 = TypeVar("T_FloatDType2", np.float32, np.float64, default=np.float64)  # type: ignore[misc]  # something off with default]

# Dtype magic
# need these to pass in dtype
T_NPScalar = TypeVar("T_NPScalar", bound=np.generic)
DType_co = TypeVar("DType_co", covariant=True, bound=np.dtype[Any])


@runtime_checkable
class _SupportsDType(Protocol[DType_co]):
    @property
    def dtype(self) -> DType_co: ...


DTypeLikeArg = Union[
    np.dtype[T_NPScalar],
    type[T_NPScalar],
    _SupportsDType[np.dtype[T_NPScalar]],
]

DTypeFloatArg = Union[
    np.dtype[T_FloatDType],
    type[T_FloatDType],
    _SupportsDType[np.dtype[T_FloatDType]],
]


# T_FloatArray = TypeVar("T_FloatArray", NDArray[np.float32], NDArray[np.float64])
# T_NDArray = TypeVar("T_NDArray", bound=NDArray[Any])
T_Scalar = TypeVar("T_Scalar", bound=np.generic)
# Note: At least for now, only use np.int64...
# T_IntDType = TypeVar("T_IntDType", np.int32, np.int64)  # array of ints
T_IntDType: TypeAlias = np.int64
FloatDTypes = Union[np.float32, np.float64]
LongIntDType: TypeAlias = np.int64

NDArrayFloats = NDArray[FloatDTypes]
NDArrayInt = NDArray[LongIntDType]
NDGeneric: TypeAlias = Union[T_FloatDType, NDArray[T_FloatDType]]

# Numba types
# NumbaType = Union[nb.typing.Integer, nb.typing.Array]
# The above isn't working for pyright.  Just using any for now...
NumbaType = Any

# Moments
Moments: TypeAlias = Union[int, "tuple[int]", "tuple[int, int]"]
MomentsStrict: TypeAlias = Union["tuple[int]", "tuple[int, int]"]
Mom_NDim = Literal[1, 2]

# Generic array
T_Array = TypeVar(  # type: ignore[misc]  # something off with default
    "T_Array",
    NDArray[np.float32],
    NDArray[np.float64],
    xr.DataArray,
    default=NDArray[np.float64],
)
T_Array2 = TypeVar(  # type: ignore[misc]  # something off with default
    "T_Array2",
    NDArray[np.float32],
    NDArray[np.float64],
    xr.DataArray,
    default=NDArray[np.float64],
)
# T_CentralMoments = TypeVar("T_CentralMoments", bound="CentralMomentsABC[DTypeAny]")

# Dummy function
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)

# xarray specific stuff
MomDims = Union[Hashable, Tuple[Hashable], Tuple[Hashable, Hashable]]
MomDimsStrict = Union[Tuple[Hashable], Tuple[Hashable, Hashable]]

# fix if using autodoc typehints...
if TYPE_CHECKING:
    _Index: TypeAlias = "pd.Index[Any]"  # type: ignore[type-arg,unused-ignore]  # py38 type error
else:
    _Index: TypeAlias = pd.Index

XArrayCoordsType: TypeAlias = Union[
    Sequence[Union[Sequence[Any], _Index, xr.DataArray]],
    Mapping[Any, Any],
    None,
]

XArrayAttrsType: TypeAlias = Optional[Mapping[Any, Any]]
XArrayNameType: TypeAlias = Optional[Hashable]
XArrayDimsType: TypeAlias = Union[Hashable, Sequence[Hashable], None]
XArrayIndexesType: TypeAlias = Any


# literals
VerifyValuesStyles: TypeAlias = Literal["val", "vals", "data", "datas", "var", "vars"]
CoordsPolicy: TypeAlias = Literal["first", "last", None]
KeepAttrs: TypeAlias = Union[
    Literal["drop", "identical", "no_conflicts", "drop_conflicts", "override"],
    bool,
    None,
]

ConvertStyle = Literal["central", "raw"]


# pushing arrays
MultiArray = Union[float, ArrayLike, T_Array]
"""Generic array."""
MultiArrayVals = Union[ArrayLike, T_Array]
"""Generic value array."""

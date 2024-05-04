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
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

# import numba as nb
# put outside to get autodoc typehints working...
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from ._typing_compat import TypeAlias
    from .abstract_central import CentralMomentsABC


# Arrays
DTypeAny: TypeAlias = Any
NDArrayAny: TypeAlias = NDArray[DTypeAny]
ArrayOrder = Literal["C", "F", "A", "K", None]

T_FloatDType = TypeVar("T_FloatDType", np.float32, np.float64)
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
T_Array = TypeVar("T_Array", NDArrayAny, xr.DataArray)
T_CentralMoments = TypeVar("T_CentralMoments", bound="CentralMomentsABC[DTypeAny]")

# Dummy function
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)

# xarray specific stuff
MomDims = Union[Hashable, Tuple[Hashable], Tuple[Hashable, Hashable]]

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

"""Useful typing stuff."""

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

from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

    from ._typing_compat import TypeAlias
    from .abstract_central import CentralMomentsABC

MyDType: TypeAlias = Any
MyNDArray: TypeAlias = NDArray[MyDType]

Moments: TypeAlias = Union[int, "tuple[int]", "tuple[int, int]"]
MomentsStrict: TypeAlias = Union["tuple[int]", "tuple[int, int]"]
XvalStrict: TypeAlias = Union[MyNDArray, "tuple[MyNDArray, MyNDArray]"]
ArrayOrder = Literal["C", "F", "A", "K", None]

T_Array = TypeVar("T_Array", "MyNDArray", "xr.DataArray")
T_CentralMoments = TypeVar("T_CentralMoments", bound="CentralMomentsABC[MyDType]")

MomDims = Union[Hashable, Tuple[Hashable], Tuple[Hashable, Hashable]]
Mom_NDim = Literal[1, 2]

FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)

# xarray specific stuff
XArrayCoordsType: TypeAlias = Union[
    Sequence[Union[Sequence[Any], "pd.Index[Any]", "xr.DataArray"]],
    Mapping[Any, Any],
    None,
]

XArrayAttrsType: TypeAlias = Optional[Mapping[Any, Any]]
XArrayNameType: TypeAlias = Optional[Hashable]
XArrayDimsType: TypeAlias = Union[Hashable, Sequence[Hashable], None]
XArrayIndexesType: TypeAlias = Any


# literals
VerifyValuesStyles = Literal["val", "vals", "data", "datas", "var", "vars"]

# pushing arrays
MultiArray = Union[float, ArrayLike, T_Array]
MultiArrayVals = Union[ArrayLike, T_Array]

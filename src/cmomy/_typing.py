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

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr
    from numpy.typing import NDArray
    from typing_extensions import TypeAlias

    from .abstract_central import CentralMomentsABC


MyDType: TypeAlias = Any
MyNDArray: TypeAlias = "NDArray[MyDType]"

Moments: TypeAlias = "int | tuple[int] | tuple[int, int]"
XvalStrict: TypeAlias = "MyNDArray | tuple[MyNDArray, MyNDArray]"
ArrayOrder = Literal["C", "F", "A", "K", None]

T_Array = TypeVar("T_Array", "MyNDArray", "xr.DataArray")
T_CentralMoments = TypeVar("T_CentralMoments", bound="CentralMomentsABC[MyDType]")

MomDims = Union[Hashable, Tuple[Hashable], Tuple[Hashable, Hashable]]
Mom_NDim = Literal[1, 2]

FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)

# xarray specific stuff
XArrayCoordsType: TypeAlias = Union[
    Sequence[Union[Sequence[Any], "pd.Index", "xr.DataArray"]],
    Mapping[Any, Any],
    None,
]

XArrayAttrsType: TypeAlias = Optional[Mapping[Any, Any]]
XArrayNameType: TypeAlias = Optional[Hashable]
XArrayDimsType: TypeAlias = Union[Hashable, Sequence[Hashable], None]
XArrayIndexesType: TypeAlias = Any


# literals
VerifyValuesStyles = Literal["val", "vals", "data", "datas", "var", "vars"]
# _VerifyValuesStyles_list = list(get_args(VerifyValuesStyles))


# def is_verifyvaluesstyle(style: str) -> TypeGuard[VerifyValuesStyles]:
#     return style in _VerifyValuesStyles_list

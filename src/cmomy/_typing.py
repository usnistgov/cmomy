"""Useful typing stuff."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Literal,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from ._lazy_imports import np, xr

if TYPE_CHECKING:
    from .abstract_central import CentralMomentsABC

Moments = Union[int, Tuple[int], Tuple[int, int]]
XvalStrict = Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
ArrayOrder = Literal["C", "F", "A", "K", None]


T_CentralMoments = TypeVar("T_CentralMoments", bound="CentralMomentsABC")
T_Array = TypeVar("T_Array", np.ndarray, xr.DataArray)


MomDims = Union[Hashable, Tuple[Hashable], Tuple[Hashable, Hashable]]
Dims = Union[Hashable, Sequence[Hashable]]

Mom_NDim = Literal[1, 2]

FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)

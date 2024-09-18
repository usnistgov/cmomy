"""
Typing aliases (:mod:`cmomy.core.typing`)
=========================================
"""

from __future__ import annotations

from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Protocol,
    TypedDict,
    Union,
    runtime_checkable,
)

import numpy as np

# put outside to get autodoc typehints working...
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike, NDArray

from .typing_compat import EllipsisType, TypeVar

if TYPE_CHECKING:
    from cmomy.wrapper import CentralMomentsArray, CentralMomentsData  # noqa: F401
    from cmomy.wrapper.wrap_abc import CentralMomentsABC  # noqa: F401

    from .missing import _Missing  # pyright: ignore[reportPrivateUsage]
    from .typing_compat import Required, TypeAlias
    from .typing_nested_sequence import (
        _NestedSequence,  # pyright: ignore[reportPrivateUsage]
    )

    # Missing value type
    MissingType: TypeAlias = Literal[_Missing.MISSING]


# * CentralMoments types ------------------------------------------------------

CentralMomentsDataArray: TypeAlias = "CentralMomentsData[xr.DataArray]"
CentralMomentsDataset: TypeAlias = "CentralMomentsData[xr.Dataset]"
CentralMomentsDataAny: TypeAlias = "CentralMomentsData[Any]"
CentralMomentsArrayAny: TypeAlias = "CentralMomentsArray[Any]"


CentralMomentsT = TypeVar("CentralMomentsT", bound="CentralMomentsABC[Any]")
CentralMomentsArrayT = TypeVar("CentralMomentsArrayT", bound="CentralMomentsArray[Any]")
CentralMomentsDataT = TypeVar("CentralMomentsDataT", bound="CentralMomentsData[Any]")


# * TypeVars ------------------------------------------------------------------
#: General data set/array
GenArrayT = TypeVar("GenArrayT", NDArray[Any], xr.DataArray, xr.Dataset)
GenArrayT_ = TypeVar("GenArrayT_", NDArray[Any], xr.DataArray, xr.Dataset)
#: DataArray or Dataset
DataT = TypeVar("DataT", xr.DataArray, xr.Dataset)
DataT_ = TypeVar("DataT_", xr.DataArray, xr.Dataset)
DataArrayOrSetT = TypeVar("DataArrayOrSetT", bound=Union[xr.DataArray, xr.Dataset])

#: TypeVar of array types with restriction
ArrayT = TypeVar(  # type: ignore[misc]
    "ArrayT",
    NDArray[np.float32],
    NDArray[np.float64],
    xr.DataArray,
    default=NDArray[np.float64],
)

FuncT = TypeVar("FuncT", bound=Callable[..., Any])

#: TypeVar of floating point precision (np.float32, np.float64, default=Any)
FloatT = TypeVar(  # type: ignore[misc]
    "FloatT",
    np.float32,
    np.float64,
    default=Any,  # pyright: ignore[reportGeneralTypeIssues]
)
FloatT_ = TypeVar("FloatT_", np.float32, np.float64)

#: TypeVar of for np.generic dtype.
ScalarT = TypeVar("ScalarT", bound=np.generic)
ScalarT_ = TypeVar("ScalarT_", bound=np.generic)

#: TypeVar of floating point precision (all)
FloatingT = TypeVar("FloatingT", bound="np.floating[Any]")
FloatingT_ = TypeVar("FloatingT_", bound="np.floating[Any]")

DTypeT_co = TypeVar("DTypeT_co", covariant=True, bound="np.dtype[Any]")
NDArrayT = TypeVar("NDArrayT", bound="NDArray[Any]")
NDArrayFloatingT = TypeVar("NDArrayFloatingT", bound="NDArray[np.floating[Any]]")


# * Numpy ---------------------------------------------------------------------
# Axis/Dim reduction type
# TODO(wpk): convert int -> SupportsIndex?
AxisReduce: TypeAlias = Union[int, None]
AxesGUFunc: TypeAlias = "list[tuple[int, ...]]"
AxisReduceMult: TypeAlias = Union[int, "tuple[int, ...]", None]

# Rng
RngTypes: TypeAlias = Union[
    int,
    Sequence[int],
    np.random.SeedSequence,
    np.random.BitGenerator,
    np.random.Generator,
]


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


# * Numba types ---------------------------------------------------------------
# NumbaType = Union[nb.typing.Integer, nb.typing.Array]  # noqa: ERA001
# The above isn't working for pyright.  Just using any for now...
NumbaType = Any

# * Moments -------------------------------------------------------------------
# NOTE: using the strict form for Moments
# Passing in integer or Sequence[int] will work in almost all cases,
# but will be flagged by typechecker...
#: Moments type
Moments: TypeAlias = Union[int, "tuple[int]", "tuple[int, int]"]
MomentsStrict: TypeAlias = Union["tuple[int]", "tuple[int, int]"]
Mom_NDim = Literal[1, 2]

# * Xarray specific stuff -----------------------------------------------------
# fix if using autodoc typehints...
DimsReduce: TypeAlias = Union[Hashable, None]
DimsReduceMult: TypeAlias = Union[Hashable, "Collection[Hashable]", None]
# This is what xarray uses for reduction/sampling dimensions
Dims = Union[str, Collection[Hashable], EllipsisType, None]


MomDims = Union[Hashable, "tuple[Hashable]", "tuple[Hashable, Hashable]"]
MomDimsStrict = Union["tuple[Hashable]", "tuple[Hashable, Hashable]"]

IndexAny: TypeAlias = "pd.Index[Any]"
CoordsType: TypeAlias = Union[
    Mapping[Any, Any],
    None,
]
AttrsType: TypeAlias = Optional[Mapping[Any, Any]]
NameType: TypeAlias = Optional[Hashable]
DimsType: TypeAlias = Union[str, Iterable[Hashable], None]
KeepAttrs: TypeAlias = Union[
    Literal["drop", "identical", "no_conflicts", "drop_conflicts", "override"],
    bool,
    None,
]
Groups: TypeAlias = Union[Sequence[Any], NDArrayAny, IndexAny, pd.MultiIndex]
ApplyUFuncKwargs: TypeAlias = Mapping[str, Any]

# * Literals ------------------------------------------------------------------
ArrayOrderCF = Literal["C", "F", None]
ArrayOrderCFA = Literal["C", "F", "A", None]
ArrayOrder = Literal["C", "F", "A", "K", None]
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
    "xmom_0",
    "xmom_1",
    "ymom_0",
    "ymom_1",
]
ConvertStyle = Literal["central", "raw"]
VerifyValuesStyles: TypeAlias = Literal["val", "vals", "data", "datas", "var", "vars"]
CoordsPolicy: TypeAlias = Literal["first", "last", "group", None]
BootStrapMethod: TypeAlias = Literal["percentile", "basic", "bca"]
BlockByModes: TypeAlias = Literal[
    "drop_first", "drop_last", "expand_first", "expand_last"
]


# * Keyword args --------------------------------------------------------------
# ** Common
class _ApplyUFuncKwargs(TypedDict, total=False):
    mom_dims: MomDims | None
    keep_attrs: KeepAttrs
    on_missing_core_dim: MissingCoreDimOptions
    apply_ufunc_kwargs: ApplyUFuncKwargs | None


class _ReductionKwargs(_ApplyUFuncKwargs, total=False):
    casting: Casting


class _ParallelKwargs(TypedDict, total=False):
    parallel: bool | None


class _MomNDimKwargs(TypedDict, total=False):
    mom_ndim: Mom_NDim


class _MomKwargs(TypedDict, total=False):
    mom: Required[Moments]


class _AxisKwargs(TypedDict, total=False):
    axis: AxisReduce | MissingType
    dim: DimsReduce | MissingType


class _AxisMultKwargs(TypedDict, total=False):
    axis: AxisReduceMult | MissingType
    dim: DimsReduceMult | MissingType


class _MoveAxisToEndKwargs(TypedDict, total=False):
    move_axis_to_end: bool


class _OrderKwargs(TypedDict, total=False):
    order: ArrayOrder


class _OrderCFKwargs(TypedDict, total=False):
    order: ArrayOrderCF


class _KeepDimsKwargs(TypedDict, total=False):
    keepdims: bool


class _ResampleKwargs(TypedDict, total=False):
    nrep: int | None
    rng: RngTypes | None


class _ResamplePairedKwargs(_ResampleKwargs, total=False):
    paired: bool
    rep_dim: str


class _DataKwargs(
    _MomNDimKwargs, _AxisKwargs, _ReductionKwargs, _ParallelKwargs, total=False
):
    pass


class _ValsKwargs(
    _MomKwargs, _AxisKwargs, _ReductionKwargs, _ParallelKwargs, total=False
):
    pass


# ** Reduction
class ReduceDataKwargs(
    _MomNDimKwargs,
    _AxisMultKwargs,
    _ReductionKwargs,
    _OrderKwargs,
    _ParallelKwargs,
    _KeepDimsKwargs,
    total=False,
):
    """Extra parameters to :func:`.reduction.reduce_data`"""

    use_reduce: bool


class ReduceValsKwargs(
    _ValsKwargs,
    _OrderCFKwargs,
    _KeepDimsKwargs,
    total=False,
):
    """Extra parameters to :func:`.reduction.reduce_vals`"""


class ReduceDataGroupedKwargs(
    _DataKwargs,
    _MoveAxisToEndKwargs,
    _OrderCFKwargs,
    total=False,
):
    """Extra parameters to :func:`.reduction.reduce_data_grouped`"""

    group_dim: str | None
    groups: Groups | None


class ReduceDataIndexedKwargs(
    _DataKwargs,
    _MoveAxisToEndKwargs,
    _OrderKwargs,
    total=False,
):
    """Extra parameters to :func:`.reduction.reduce_data_indexed`"""

    index: Required[ArrayLike]
    group_start: Required[ArrayLike]
    group_end: Required[ArrayLike]
    scale: ArrayLike | None

    coords_policy: CoordsPolicy
    group_dim: str | None
    groups: Groups | None


# ** Resample
class ResampleDataKwargs(
    _DataKwargs,
    _ResamplePairedKwargs,
    _MoveAxisToEndKwargs,
    _OrderKwargs,
    total=False,
):
    """Extra parameters to :func:`.resample.resample_data`"""


class ResampleValsKwargs(
    _ValsKwargs,
    _ResamplePairedKwargs,
    _MoveAxisToEndKwargs,
    _OrderCFKwargs,
    total=False,
):
    """Extra parameters for :func:`.resample.resample_vals`"""


class JackknifeDataKwargs(
    _DataKwargs,
    _MoveAxisToEndKwargs,
    _OrderKwargs,
    total=False,
):
    """Extra parameters for :func:`.resample.jackknife_data`"""

    rep_dim: str | None


class JackknifeValsKwargs(
    _ValsKwargs,
    _MoveAxisToEndKwargs,
    _OrderKwargs,
    total=False,
):
    """Extra parameters for :func:`.resample.jackknife_data`"""

    rep_dim: str | None


# ** Convert
class WrapRawKwargs(
    _MomNDimKwargs,
    _ReductionKwargs,
    _OrderKwargs,
    total=False,
):
    """Extra parameters for :func:`.wrap_raw`"""


class MomentsTypeKwargs(
    WrapRawKwargs,
    total=False,
):
    """Extra parameters for :func:`.convert.moments_type`"""

    to: ConvertStyle


class CumulativeKwargs(
    _DataKwargs,
    _MoveAxisToEndKwargs,
    _OrderKwargs,
    total=False,
):
    """Extra parameters for :func:`.convert.cumulative`"""

    inverse: bool


class MomentsToComomentsKwargs(
    _OrderCFKwargs,
    total=False,
):
    """Extra parameters for :func:`.convert.moments_to_comoments`"""

    mom: Required[tuple[int, int]]
    mom_dims: MomDims | None
    mom_dims2: MomDims | None
    keep_attrs: KeepAttrs
    on_missing_core_dim: MissingCoreDimOptions
    apply_ufunc_kwargs: ApplyUFuncKwargs | None


# ** Utils
class SelectMomentKwargs(
    _MomNDimKwargs,
    _ApplyUFuncKwargs,
    total=False,
):
    """Extra parameters to :func:`.utils.select_moment`"""

    squeeze: bool
    dim_combined: str
    coords_combined: str | Sequence[Hashable] | None


class ValsToDataKwargs(
    _MomKwargs,
    _ApplyUFuncKwargs,
    total=False,
):
    """Extra parameters to :func:`.utils.vals_to_data`"""


# ** Rolling
class _RollingCommonKwargs(TypedDict, total=False):
    min_periods: int | None
    zero_missing_weights: bool


class _RollingKwargs(_RollingCommonKwargs, total=False):
    window: Required[int]
    center: bool


class _RollingExpKwargs(_RollingCommonKwargs, total=False):
    adjust: bool


class RollingDataKwargs(
    _DataKwargs, _RollingKwargs, _MoveAxisToEndKwargs, _OrderKwargs, total=False
):
    """Extra parameters to :func:`.rolling.rolling_data`"""


class RollingValsKwargs(
    _ValsKwargs,
    _RollingKwargs,
    _MoveAxisToEndKwargs,
    _OrderCFKwargs,
):
    """Extra parameters to :func:`.rolling.rolling_vals`"""


class RollingExpDataKwargs(
    _DataKwargs,
    _RollingExpKwargs,
    _MoveAxisToEndKwargs,
    _OrderKwargs,
    total=False,
):
    """Extra parameters to :func:`.rolling.rolling_exp_data`"""


class RollingExpValsKwargs(
    _ValsKwargs,
    _RollingExpKwargs,
    _MoveAxisToEndKwargs,
    _OrderCFKwargs,
    total=False,
):
    """Extra parameters to :func:`.rolling.rolling_exp_vals`"""


# ** Wrap
class WrapKwargs(
    _MomNDimKwargs,
    total=False,
):
    """Extra parameters to :func:`cmomy.wrap`"""

    mom_dims: MomDims | None
    copy: bool | None
    fastpath: bool


class ZerosLikeKwargs(
    _OrderKwargs,
    total=False,
):
    """Extra parameters to :func:`cmomy.zeros_like`"""

    subok: bool
    chunks: Any
    chunked_array_type: str | None
    from_array_kwargs: dict[str, Any] | None


# *** Wrap_np
class WrapNPTransform(
    _MoveAxisToEndKwargs,
    _OrderKwargs,
    _ParallelKwargs,
    total=False,
):
    """Extra parameters to :meth:`.CentralMomentsArray.cumulative`"""

    axis: AxisReduce | MissingType
    casting: Casting


class WrapNPResampleAndReduceKwargs(
    _ResampleKwargs,
    WrapNPTransform,
    total=False,
):
    """Extra parameters to :meth:`.CentralMomentsArray.resample_and_reduce`"""


class WrapNPReduce(
    WrapNPTransform,
    _KeepDimsKwargs,
    total=False,
):
    """Extra parameters to :meth:`.CentralMomentsArray.reduce`"""

    by: Groups
    block: int

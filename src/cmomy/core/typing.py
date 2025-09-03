"""
Typing aliases (:mod:`cmomy.core.typing`)
=========================================
"""
# pylint: disable=missing-class-docstring,consider-alternative-union-syntax,duplicate-bases

from __future__ import annotations

from collections.abc import Callable, Collection, Hashable, Iterable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
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
from numpy.typing import ArrayLike, NDArray

from .docstrings import docfiller
from .typing_compat import EllipsisType, TypedDict, TypeVar

if TYPE_CHECKING:
    from cmomy.resample.sampler import IndexSampler
    from cmomy.wrapper import CentralMomentsArray, CentralMomentsData  # noqa: F401
    from cmomy.wrapper.wrap_abc import CentralMomentsABC  # noqa: F401

    from .missing import _Missing  # pyright: ignore[reportPrivateUsage]
    from .moment_params import (
        MomParams,
        MomParamsArray,  # noqa: F401
        MomParamsBase,
        MomParamsDict,
        MomParamsXArray,  # noqa: F401
    )
    from .typing_compat import Required, TypeAlias
    from .typing_nested_sequence import (
        _NestedSequence,  # pyright: ignore[reportPrivateUsage]
    )

    # Missing value type
    MissingType: TypeAlias = Literal[_Missing.MISSING]


# * CentralMoments types ------------------------------------------------------

#: DataArray wrapper
CentralMomentsDataArray: TypeAlias = "CentralMomentsData[xr.DataArray]"
#: Dataset wrapper
CentralMomentsDataset: TypeAlias = "CentralMomentsData[xr.Dataset]"
#: DataArray or Dataset wrapper
CentralMomentsDataAny: TypeAlias = "CentralMomentsData[Any]"
#: Any Array wrapper
CentralMomentsArrayAny: TypeAlias = "CentralMomentsArray[Any]"

#: Generic wrapper TypeVar
CentralMomentsT = TypeVar("CentralMomentsT", bound="CentralMomentsABC[Any, Any]")
#: Array wrapper TypeVar
CentralMomentsArrayT = TypeVar("CentralMomentsArrayT", bound="CentralMomentsArray[Any]")
#: xarray object wrapper TypeVar
CentralMomentsDataT = TypeVar("CentralMomentsDataT", bound="CentralMomentsData[Any]")


# * MomParams
#: Moment parameters input
MomParamsInput = Union["MomParams", "MomParamsBase", "MomParamsDict", None]
#: Moment parameters TypeVar
MomParamsT = TypeVar("MomParamsT", "MomParamsArray", "MomParamsXArray")


# * TypeVars ------------------------------------------------------------------
#: General data set/array
GenArrayT = TypeVar("GenArrayT", NDArray[Any], xr.DataArray, xr.Dataset)
GenArrayT_ = TypeVar("GenArrayT_", NDArray[Any], xr.DataArray, xr.Dataset)
#: DataArray or Dataset
DataT = TypeVar("DataT", xr.DataArray, xr.Dataset)
DataT_ = TypeVar("DataT_", xr.DataArray, xr.Dataset)
DataArrayOrSetT = TypeVar("DataArrayOrSetT", bound=Union[xr.DataArray, xr.Dataset])

#: TypeVar of array types with restriction
ArrayT = TypeVar(
    "ArrayT",
    NDArray[np.float64],
    NDArray[np.float32],
    xr.DataArray,
)

#: TypeVar of types wrapped by IndexSampler
SamplerArrayT = TypeVar(
    "SamplerArrayT",
    NDArray[Any],
    xr.DataArray,
    xr.Dataset,
    Union[xr.DataArray, xr.Dataset],
)

FuncT = TypeVar("FuncT", bound=Callable[..., Any])

#: TypeVar of floating point precision (np.float32, np.float64, default=Any)
FloatT = TypeVar(
    "FloatT",
    np.float64,
    np.float32,
)
FloatT_ = TypeVar("FloatT_", np.float64, np.float32)

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
#: Axes type
Axes = Union[int, "tuple[int, ...]"]
#: Axes type (with wrapping)
AxesWrap = Union[complex, "tuple[complex, ...]"]

#: Reduction axes type
AxisReduce: TypeAlias = Optional[int]
#: Reduction axes type (with wrapping)
AxisReduceWrap: TypeAlias = Optional[complex]

AxesGUFunc: TypeAlias = "list[tuple[int, ...]]"

#: Reduction axes type (multiple)
AxisReduceMult: TypeAlias = Union[int, "tuple[int, ...]", None]
#: Reduction axes type (multiple, with wrapping)
AxisReduceMultWrap: TypeAlias = Union[complex, "tuple[complex, ...]", None]

#: Random number generator types
RngTypes: TypeAlias = Union[
    int,
    Sequence[int],
    np.random.SeedSequence,
    np.random.BitGenerator,
    np.random.Generator,
]


# Types
#: Any dtype
DTypeAny: TypeAlias = Any
#: Floating dtype
FloatDTypes = Union[np.float32, np.float64]
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
#: Strict moments type
MomentsStrict: TypeAlias = Union["tuple[int]", "tuple[int, int]"]
#: Number of moment dimensions
MomNDim = Literal[1, 2]

#: Axes containing moment(s).
MomAxes = Moments
#: Axes containing moment(s).
MomAxesStrict = MomentsStrict

# * Xarray specific stuff -----------------------------------------------------
# fix if using autodoc typehints...
#: Reduction dimension
DimsReduce: TypeAlias = Optional[Hashable]
#: Reduction dimension(s)
DimsReduceMult: TypeAlias = Union[Hashable, "Collection[Hashable]", None]
#: Dimensions
Dims = Union[str, Collection[Hashable], EllipsisType, None]

#: Dimensions containing moment(s)
MomDims = Union[Hashable, "tuple[Hashable]", "tuple[Hashable, Hashable]"]
#: Dimensions containing moment(s)
MomDimsStrict = Union["tuple[Hashable]", "tuple[Hashable, Hashable]"]

#: Index
IndexAny: TypeAlias = "pd.Index[Any]"
#: Coordinates
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

# * Literals ------------------------------------------------------------------
#: Order parameters
ArrayOrderCF = Optional[Literal["C", "F"]]
#: Order parameters
ArrayOrderACF = Optional[Literal["A", "C", "F"]]
#: Order parameters
ArrayOrderKACF = Optional[Literal["K", "A", "C", "F"]]
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
VerifyValuesStyles: TypeAlias = Literal["val", "vals", "data", "datas", "var", "vars"]
CoordsPolicy: TypeAlias = Optional[Literal["first", "last", "group"]]
BootStrapMethod: TypeAlias = Literal["percentile", "basic", "bca"]
BlockByModes: TypeAlias = Literal[
    "drop_first", "drop_last", "expand_first", "expand_last"
]


# * Keyword args --------------------------------------------------------------
# ** Common
ApplyUFuncKwargs: TypeAlias = Mapping[str, Any]


class _MomDimsAndApplyUFuncKwargs(TypedDict, total=False):
    mom_dims: MomDims | None
    keep_attrs: KeepAttrs
    apply_ufunc_kwargs: ApplyUFuncKwargs | None


class _ReductionKwargs(_MomDimsAndApplyUFuncKwargs, total=False):
    casting: Casting


class _ParallelKwargs(TypedDict, total=False):
    parallel: bool | None


class _MomNDimKwargs(TypedDict, total=False):
    mom_ndim: MomNDim | None


class _MomAxesKwargs(TypedDict, total=False):
    mom_axes: MomAxes | None


class _MomKwargs(TypedDict, total=False):
    mom: Required[Moments]


class _MomParamsKwargs(TypedDict, total=False):
    mom_params: MomParamsInput


class _AxisKwargs(TypedDict, total=False):
    axis: AxisReduceWrap | MissingType
    dim: DimsReduce | MissingType


class _AxisMultKwargs(TypedDict, total=False):
    axis: AxisReduceMultWrap | MissingType
    dim: DimsReduceMult | MissingType


class _MoveAxisToEndKwargs(TypedDict, total=False):
    axes_to_end: bool


class _OrderKACFKwargs(TypedDict, total=False):
    order: ArrayOrderKACF


class _OrderCFKwargs(TypedDict, total=False):
    order: ArrayOrderCF


class _KeepDimsKwargs(TypedDict, total=False):
    keepdims: bool


class _RepDimKwargs(TypedDict, total=False):
    rep_dim: str


class _GroupsKwargs(TypedDict, total=False):
    group_dim: str | None
    groups: Groups | None


class _IndexedKwargs(TypedDict, total=False):
    index: Required[ArrayLike]
    group_start: Required[ArrayLike]
    group_end: Required[ArrayLike]
    scale: ArrayLike | None

    coords_policy: CoordsPolicy


class _DataKwargs(
    _MomParamsKwargs,
    _MomNDimKwargs,
    _AxisKwargs,
    _ReductionKwargs,
    _ParallelKwargs,
    _MomAxesKwargs,
    total=False,
):
    pass


class _DataCFKwargs(
    _DataKwargs,
    _OrderCFKwargs,
    total=False,
):
    pass


class _DataKACFKwargs(
    _DataKwargs,
    _OrderKACFKwargs,
    total=False,
):
    pass


class _ValsKwargs(
    _MomParamsKwargs,
    _MomKwargs,
    _AxisKwargs,
    _ReductionKwargs,
    _ParallelKwargs,
    total=False,
):
    pass


class _ValsCFKwargs(
    _ValsKwargs,
    _OrderCFKwargs,
    total=False,
):
    pass


# ** Reduction
class ReduceDataKwargs(  # type: ignore[call-arg]
    _MomParamsKwargs,
    _MomNDimKwargs,
    _AxisMultKwargs,
    _ReductionKwargs,
    _ParallelKwargs,
    _OrderKACFKwargs,
    _MomAxesKwargs,
    _KeepDimsKwargs,
    _MoveAxisToEndKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.reduction.reduce_data`"""

    use_map: bool | None


class ReduceValsKwargs(  # type: ignore[call-arg]
    _ValsCFKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.reduction.reduce_vals`"""


class ReduceDataGroupedKwargs(  # type: ignore[call-arg]
    _DataCFKwargs,
    _MoveAxisToEndKwargs,
    _GroupsKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.grouped.reduce_data_grouped`"""


class ReduceDataIndexedKwargs(  # type: ignore[call-arg]
    _DataKACFKwargs,
    _MoveAxisToEndKwargs,
    _GroupsKwargs,
    _IndexedKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.grouped.reduce_data_indexed`"""


class ReduceValsGroupedKwargs(  # type: ignore[call-arg]
    _ValsCFKwargs,
    _MoveAxisToEndKwargs,
    _GroupsKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.grouped.reduce_vals_grouped`"""


class ReduceValsIndexedKwargs(  # type: ignore[call-arg]
    _ValsCFKwargs,
    _MoveAxisToEndKwargs,
    _GroupsKwargs,
    _IndexedKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.grouped.reduce_vals_indexed`"""


# ** Resample
class ResampleDataKwargs(  # type: ignore[call-arg]
    _DataKACFKwargs,
    _MoveAxisToEndKwargs,
    _RepDimKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.resample.resample_data`"""


class ResampleValsKwargs(  # type: ignore[call-arg]
    _ValsCFKwargs,
    _MoveAxisToEndKwargs,
    _RepDimKwargs,
    total=False,
    closed=True,
):
    """Extra parameters for :func:`.resample.resample_vals`"""


class JackknifeDataKwargs(  # type: ignore[call-arg]
    _DataKACFKwargs,
    _MoveAxisToEndKwargs,
    total=False,
    closed=True,
):
    """Extra parameters for :func:`.resample.jackknife_data`"""

    rep_dim: str | None
    mom_axes_reduced: MomAxes | None


class JackknifeValsKwargs(  # type: ignore[call-arg]
    _ValsCFKwargs,
    _MoveAxisToEndKwargs,
    total=False,
    closed=True,
):
    """Extra parameters for :func:`.resample.jackknife_data`"""

    rep_dim: str | None


# ** Convert
class _WrapRawKwargs(
    _MomNDimKwargs,
    _ReductionKwargs,
    _OrderKACFKwargs,
    _MomAxesKwargs,
    _MomParamsKwargs,
    total=False,
):
    pass


class WrapRawKwargs(  # type: ignore[call-arg]
    _WrapRawKwargs,
    total=False,
    closed=True,
):
    """Extra parameters for :func:`.wrap_raw`"""


class MomentsTypeKwargs(  # type: ignore[call-arg]
    _WrapRawKwargs,
    _MoveAxisToEndKwargs,
    total=False,
    closed=True,
):
    """Extra parameters for :func:`.convert.moments_type`"""

    to: ConvertStyle


class CumulativeKwargs(  # type: ignore[call-arg]
    _DataKACFKwargs,
    _MoveAxisToEndKwargs,
    total=False,
    closed=True,
):
    """Extra parameters for :func:`.convert.cumulative`"""

    inverse: bool


class MomentsToComomentsKwargs(  # type: ignore[call-arg]
    _MomDimsAndApplyUFuncKwargs,
    _OrderCFKwargs,
    _MomAxesKwargs,
    _MomParamsKwargs,
    total=False,
    closed=True,
):
    """Extra parameters for :func:`.convert.moments_to_comoments`"""

    mom_dims_out: MomDims | None


# ** Utils
class SelectMomentKwargs(  # type: ignore[call-arg]
    _MomNDimKwargs,
    _MomDimsAndApplyUFuncKwargs,
    _MomAxesKwargs,
    _MomParamsKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.utils.select_moment`"""

    squeeze: bool
    dim_combined: str
    coords_combined: str | Sequence[Hashable] | None


class ValsToDataKwargs(  # type: ignore[call-arg]
    _MomKwargs,
    _MomDimsAndApplyUFuncKwargs,
    _MomParamsKwargs,
    total=False,
    closed=True,
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


class RollingDataKwargs(  # type: ignore[call-arg]
    _DataKACFKwargs,
    _RollingKwargs,
    _MoveAxisToEndKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.rolling.rolling_data`"""


class RollingValsKwargs(  # type: ignore[call-arg]
    _ValsCFKwargs,
    _RollingKwargs,
    _MoveAxisToEndKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.rolling.rolling_vals`"""


class RollingExpDataKwargs(  # type: ignore[call-arg]
    _DataKACFKwargs,
    _RollingExpKwargs,
    _MoveAxisToEndKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.rolling.rolling_exp_data`"""

    alpha_axis: AxisReduceWrap | MissingType


class RollingExpValsKwargs(  # type: ignore[call-arg]
    _ValsCFKwargs,
    _RollingExpKwargs,
    _MoveAxisToEndKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.rolling.rolling_exp_vals`"""


# ** Wrap
class WrapKwargs(  # type: ignore[call-arg]
    _MomNDimKwargs,
    _MomAxesKwargs,
    _MomParamsKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`cmomy.wrap`"""

    mom_dims: MomDims | None
    copy: bool | None
    fastpath: bool


class ZerosLikeKwargs(  # type: ignore[call-arg]
    _OrderKACFKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`cmomy.zeros_like`"""

    subok: bool
    chunks: Any
    chunked_array_type: str | None
    from_array_kwargs: dict[str, Any] | None


# *** Wrap_np
class _WrapNPTransform(
    _MoveAxisToEndKwargs,
    _OrderKACFKwargs,
    _ParallelKwargs,
    total=False,
):
    axis: AxisReduce | MissingType
    casting: Casting


class WrapNPTransform(  # type: ignore[call-arg]
    _WrapNPTransform,
    total=False,
    closed=True,
):
    """Extra parameters to :meth:`.CentralMomentsArray.cumulative`"""


class WrapNPResampleAndReduceKwargs(  # type: ignore[call-arg]
    _WrapNPTransform,
    total=False,
    closed=True,
):
    """Extra parameters to :meth:`.CentralMomentsArray.resample_and_reduce`"""


class WrapNPReduceKwargs(  # type: ignore[call-arg]
    _WrapNPTransform,
    _KeepDimsKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :meth:`.CentralMomentsArray.reduce`"""

    by: Groups
    block: int


class IndexSamplerFromDataKwargs(  # type: ignore[call-arg]
    _AxisKwargs,
    _MomAxesKwargs,
    _MomParamsKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :meth:`.resample.IndexSampler.from_data`"""

    nrep: Required[int]
    nsamp: int | None
    mom_ndim: MomNDim | None
    mom_dims: MomDims | None
    rep_dim: str
    rng: RngTypes | None
    replace: bool
    parallel: bool | None


@docfiller.decorate
class FactoryIndexSamplerKwargs(  # type: ignore[call-arg]
    TypedDict,
    total=False,
    closed=True,
):
    """
    Extra parameters to :func:`.resample.factory_sampler`

    Parameters
    ----------
    {indices}
    {freq_xarray}
    {ndat}
    {nrep}
    {nsamp}
    {paired}
    {rng}
    {resample_replace}
    {shuffle}
    """

    freq: NDArrayAny | xr.DataArray | xr.Dataset | None
    indices: NDArrayAny | xr.DataArray | xr.Dataset | None
    ndat: int | None
    nrep: int | None
    nsamp: int | None
    paired: bool
    rng: RngTypes | None
    replace: bool
    shuffle: bool


#: IndexSampler or mapping which can be converted to IndexSampler
Sampler: TypeAlias = Union[
    int,
    NDArrayAny,
    xr.DataArray,
    xr.Dataset,
    "IndexSampler[Any]",
    FactoryIndexSamplerKwargs,
]

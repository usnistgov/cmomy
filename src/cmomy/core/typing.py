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

from .docstrings import docfiller
from .typing_compat import EllipsisType, TypeVar

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

CentralMomentsDataArray: TypeAlias = "CentralMomentsData[xr.DataArray]"
CentralMomentsDataset: TypeAlias = "CentralMomentsData[xr.Dataset]"
CentralMomentsDataAny: TypeAlias = "CentralMomentsData[Any]"
CentralMomentsArrayAny: TypeAlias = "CentralMomentsArray[Any]"


CentralMomentsT = TypeVar("CentralMomentsT", bound="CentralMomentsABC[Any, Any]")
CentralMomentsArrayT = TypeVar("CentralMomentsArrayT", bound="CentralMomentsArray[Any]")
CentralMomentsDataT = TypeVar("CentralMomentsDataT", bound="CentralMomentsData[Any]")


# * MomParams

MomParamsInput = Union["MomParams", "MomParamsBase", "MomParamsDict", None]
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
ArrayT = TypeVar(  # type: ignore[misc]
    "ArrayT",
    NDArray[np.float32],
    NDArray[np.float64],
    xr.DataArray,
    default=NDArray[np.float64],
)

#: TypeVar of types wrapped by IndexSampler
SamplerArrayT = TypeVar(  # type: ignore[misc]
    "SamplerArrayT",
    NDArray[Any],
    xr.DataArray,
    xr.Dataset,
    Union[xr.DataArray, xr.Dataset],
    default=NDArray[Any],
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
Axes = Union[int, "tuple[int, ...]"]
AxesWrap = Union[complex, "tuple[complex, ...]"]


AxisReduce: TypeAlias = Optional[int]
AxisReduceWrap: TypeAlias = Optional[complex]

AxesGUFunc: TypeAlias = "list[tuple[int, ...]]"

AxisReduceMult: TypeAlias = Union[int, "tuple[int, ...]", None]
AxisReduceMultWrap: TypeAlias = Union[complex, "tuple[complex, ...]", None]

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
NDArrayInt = NDArray[np.int64]
NDArrayFloats = NDArray[FloatDTypes]
NDArrayBool = NDArray[np.bool_]
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
MomNDim = Literal[1, 2]


MomAxes = Moments
MomAxesStrict = MomentsStrict

# * Xarray specific stuff -----------------------------------------------------
# fix if using autodoc typehints...
DimsReduce: TypeAlias = Optional[Hashable]
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


class _OrderKwargs(TypedDict, total=False):
    order: ArrayOrder


class _OrderCFKwargs(TypedDict, total=False):
    order: ArrayOrderCF


class _KeepDimsKwargs(TypedDict, total=False):
    keepdims: bool


class _RepDimKwargs(TypedDict, total=False):
    rep_dim: str


class _DataKwargs(
    _MomNDimKwargs,
    _AxisKwargs,
    _ReductionKwargs,
    _ParallelKwargs,
    _MomAxesKwargs,
    total=False,
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
    _ParallelKwargs,
    _OrderKwargs,
    _KeepDimsKwargs,
    _MomAxesKwargs,
    _MomParamsKwargs,
    _MoveAxisToEndKwargs,
    total=False,
):
    """Extra parameters to :func:`.reduction.reduce_data`"""

    use_map: bool | None


class ReduceValsKwargs(
    _ValsKwargs,
    _OrderCFKwargs,
    _MomParamsKwargs,
    total=False,
):
    """Extra parameters to :func:`.reduction.reduce_vals`"""


class ReduceDataGroupedKwargs(
    _DataKwargs,
    _MoveAxisToEndKwargs,
    _OrderCFKwargs,
    _MomParamsKwargs,
    total=False,
):
    """Extra parameters to :func:`.reduction.reduce_data_grouped`"""

    group_dim: str | None
    groups: Groups | None


class ReduceDataIndexedKwargs(
    _DataKwargs,
    _MoveAxisToEndKwargs,
    _OrderKwargs,
    _MomParamsKwargs,
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
    _RepDimKwargs,
    _MoveAxisToEndKwargs,
    _OrderKwargs,
    _MomParamsKwargs,
    total=False,
):
    """Extra parameters to :func:`.resample.resample_data`"""


class ResampleValsKwargs(
    _ValsKwargs,
    _RepDimKwargs,
    _MoveAxisToEndKwargs,
    _OrderCFKwargs,
    _MomParamsKwargs,
    total=False,
):
    """Extra parameters for :func:`.resample.resample_vals`"""


class JackknifeDataKwargs(
    _DataKwargs,
    _MoveAxisToEndKwargs,
    _OrderKwargs,
    _MomParamsKwargs,
    total=False,
):
    """Extra parameters for :func:`.resample.jackknife_data`"""

    rep_dim: str | None
    mom_axes_reduced: MomAxes | None


class JackknifeValsKwargs(
    _ValsKwargs,
    _MoveAxisToEndKwargs,
    _OrderCFKwargs,
    _MomParamsKwargs,
    total=False,
):
    """Extra parameters for :func:`.resample.jackknife_data`"""

    rep_dim: str | None


# ** Convert
class WrapRawKwargs(
    _MomNDimKwargs,
    _ReductionKwargs,
    _OrderKwargs,
    _MomAxesKwargs,
    _MomParamsKwargs,
    total=False,
):
    """Extra parameters for :func:`.wrap_raw`"""


class MomentsTypeKwargs(
    WrapRawKwargs,
    _MoveAxisToEndKwargs,
    total=False,
):
    """Extra parameters for :func:`.convert.moments_type`"""

    to: ConvertStyle


class CumulativeKwargs(
    _DataKwargs,
    _MoveAxisToEndKwargs,
    _OrderKwargs,
    _MomParamsKwargs,
    total=False,
):
    """Extra parameters for :func:`.convert.cumulative`"""

    inverse: bool


class MomentsToComomentsKwargs(
    _MomDimsAndApplyUFuncKwargs,
    _OrderCFKwargs,
    _MomAxesKwargs,
    _MomParamsKwargs,
    total=False,
):
    """Extra parameters for :func:`.convert.moments_to_comoments`"""

    mom_dims_out: MomDims | None


# ** Utils
class SelectMomentKwargs(
    _MomNDimKwargs,
    _MomDimsAndApplyUFuncKwargs,
    _MomAxesKwargs,
    _MomParamsKwargs,
    total=False,
):
    """Extra parameters to :func:`.utils.select_moment`"""

    squeeze: bool
    dim_combined: str
    coords_combined: str | Sequence[Hashable] | None


class ValsToDataKwargs(
    _MomKwargs,
    _MomDimsAndApplyUFuncKwargs,
    _MomParamsKwargs,
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
    _DataKwargs,
    _RollingKwargs,
    _MoveAxisToEndKwargs,
    _OrderKwargs,
    _MomParamsKwargs,
    total=False,
):
    """Extra parameters to :func:`.rolling.rolling_data`"""


class RollingValsKwargs(
    _ValsKwargs,
    _RollingKwargs,
    _MoveAxisToEndKwargs,
    _OrderCFKwargs,
    _MomParamsKwargs,
):
    """Extra parameters to :func:`.rolling.rolling_vals`"""


class RollingExpDataKwargs(
    _DataKwargs,
    _RollingExpKwargs,
    _MoveAxisToEndKwargs,
    _OrderKwargs,
    _MomParamsKwargs,
    total=False,
):
    """Extra parameters to :func:`.rolling.rolling_exp_data`"""

    alpha_axis: AxisReduceWrap | MissingType


class RollingExpValsKwargs(
    _ValsKwargs,
    _RollingExpKwargs,
    _MoveAxisToEndKwargs,
    _OrderCFKwargs,
    _MomParamsKwargs,
    total=False,
):
    """Extra parameters to :func:`.rolling.rolling_exp_vals`"""


# ** Wrap
class WrapKwargs(
    _MomNDimKwargs,
    _MomAxesKwargs,
    _MomParamsKwargs,
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
    WrapNPTransform,
    total=False,
):
    """Extra parameters to :meth:`.CentralMomentsArray.resample_and_reduce`"""


class WrapNPReduceKwargs(
    WrapNPTransform,
    _KeepDimsKwargs,
    total=False,
):
    """Extra parameters to :meth:`.CentralMomentsArray.reduce`"""

    by: Groups
    block: int


class IndexSamplerFromDataKwargs(
    _AxisKwargs,
    _MomAxesKwargs,
    _MomParamsKwargs,
    total=False,
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
class FactoryIndexSamplerKwargs(
    TypedDict,
    total=False,
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

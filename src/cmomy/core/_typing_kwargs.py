"""
Typed keyword arguments (:mod:`cmomy.core.typing_kwargs`)
=========================================================
"""

# pyright: deprecateTypingAliases=false
# pylint: disable=missing-class-docstring, too-many-ancestors

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .typing_compat import TypedDict

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence
    from typing import Required

    from numpy.typing import ArrayLike

    from .moment_params import MomParamsType
    from .typing import (
        ArrayOrderCF,
        ArrayOrderKACF,
        AxisReduce,
        AxisReduceMultWrap,
        AxisReduceWrap,
        Casting,
        ConvertStyle,
        CoordsPolicy,
        DimsReduce,
        DimsReduceMult,
        Groups,
        KeepAttrs,
        MissingType,
        MomAxes,
        MomDims,
        Moments,
        MomNDim,
        RngTypes,
    )
    from .typing_compat import TypeAlias


# * Keyword args --------------------------------------------------------------
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
    mom_params: MomParamsType


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


class _CoordsPolicyKwargs(TypedDict, total=False):
    coords_policy: CoordsPolicy


class _GroupsKwargs(TypedDict, total=False):
    group_dim: str | None
    groups: Groups | None


class _IndexedKwargs(TypedDict, total=False):
    index: Required[ArrayLike]
    group_start: Required[ArrayLike]
    group_end: Required[ArrayLike]
    scale: ArrayLike | None


class _DataKwargs(
    _MomParamsKwargs,
    _MomNDimKwargs,
    _MomAxesKwargs,
    _MoveAxisToEndKwargs,
    _AxisKwargs,
    _ReductionKwargs,
    _ParallelKwargs,
    total=False,
):
    pass


class _DataCFKwargs(
    _DataKwargs,
    _OrderCFKwargs,
    total=False,
):
    pass


class DataKACFKwargs(
    _DataKwargs,
    _OrderKACFKwargs,
    total=False,
):
    pass


class _ValsKwargs(
    _MomParamsKwargs,
    _MomKwargs,
    _MomAxesKwargs,
    _MoveAxisToEndKwargs,
    _AxisKwargs,
    _ReductionKwargs,
    _ParallelKwargs,
    total=False,
):
    pass


class ValsCFKwargs(
    _ValsKwargs,
    _OrderCFKwargs,
    total=False,
):
    pass


# ** Reduction
# NOTE: special case with multiple axes
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
    ValsCFKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.reduction.reduce_vals`"""


class ReduceDataGroupedKwargs(  # type: ignore[call-arg]
    _DataCFKwargs,
    _GroupsKwargs,
    _CoordsPolicyKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.grouped.reduce_data_grouped`"""


class ReduceDataIndexedKwargs(  # type: ignore[call-arg]
    DataKACFKwargs,
    _GroupsKwargs,
    _IndexedKwargs,
    _CoordsPolicyKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.grouped.reduce_data_indexed`"""


class ReduceValsGroupedKwargs(  # type: ignore[call-arg]
    ValsCFKwargs,
    _GroupsKwargs,
    _CoordsPolicyKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.grouped.reduce_vals_grouped`"""


class ReduceValsIndexedKwargs(  # type: ignore[call-arg]
    ValsCFKwargs,
    _GroupsKwargs,
    _IndexedKwargs,
    _CoordsPolicyKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.grouped.reduce_vals_indexed`"""


class JackknifeDataKwargs(  # type: ignore[call-arg]
    DataKACFKwargs,
    total=False,
    closed=True,
):
    """Extra parameters for :func:`.resample.jackknife_data`"""

    rep_dim: str | None
    mom_axes_reduced: MomAxes | None


class JackknifeValsKwargs(  # type: ignore[call-arg]
    ValsCFKwargs,
    total=False,
    closed=True,
):
    """Extra parameters for :func:`.resample.jackknife_data`"""

    rep_dim: str | None
    mom_axes_reduced: MomAxes | None


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
    DataKACFKwargs,
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
    DataKACFKwargs,
    _RollingKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.rolling.rolling_data`"""


class RollingValsKwargs(  # type: ignore[call-arg]
    ValsCFKwargs,
    _RollingKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.rolling.rolling_vals`"""


class RollingExpDataKwargs(  # type: ignore[call-arg]
    DataKACFKwargs,
    _RollingExpKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.rolling.rolling_exp_data`"""

    alpha_axis: AxisReduceWrap | MissingType


class RollingExpValsKwargs(  # type: ignore[call-arg]
    ValsCFKwargs,
    _RollingExpKwargs,
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
class _WrapNPTransformKwargs(
    _MoveAxisToEndKwargs,
    _OrderKACFKwargs,
    _ParallelKwargs,
    total=False,
):
    axis: AxisReduce | MissingType
    casting: Casting


class WrapNPTransformKwargs(  # type: ignore[call-arg]
    _WrapNPTransformKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :meth:`.CentralMomentsArray.cumulative`"""


class WrapNPResampleAndReduceKwargs(  # type: ignore[call-arg]
    _WrapNPTransformKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :meth:`.CentralMomentsArray.resample_and_reduce`"""


class WrapNPReduceKwargs(  # type: ignore[call-arg]
    _WrapNPTransformKwargs,
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

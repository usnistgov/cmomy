"""Typing aliases for :mod:`cmomy.resample`"""
# NOTE: These are separate from `cmomy.core.typing` because of import cycles.

from __future__ import annotations

from typing import TYPE_CHECKING

from cmomy.core.docstrings import docfiller
from cmomy.core.typing_compat import TypedDict
from cmomy.core.typing_kwargs import (
    _DataKACFKwargs,  # pyright: ignore[reportPrivateUsage]
    _ValsCFKwargs,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from typing import (
        Any,  # noqa: F401
        Required,
    )

    import xarray as xr

    from cmomy.core.typing import (
        NDArrayAny,
        RngTypes,
    )
    from cmomy.core.typing_compat import TypeAlias

    from ._sampler import IndexSampler  # noqa: F401


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
Sampler: TypeAlias = "int | NDArrayAny | xr.DataArray | xr.Dataset | IndexSampler[Any] | FactoryIndexSamplerKwargs"


class _SamplerKwargs(TypedDict, total=False):
    sampler: Required[Sampler]
    rep_dim: str


class ResampleDataKwargs(  # type: ignore[call-arg]
    _DataKACFKwargs,
    _SamplerKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.resample.resample_data`"""


class ResampleValsKwargs(  # type: ignore[call-arg]
    _ValsCFKwargs,
    _SamplerKwargs,
    total=False,
    closed=True,
):
    """Extra parameters for :func:`.resample.resample_vals`"""

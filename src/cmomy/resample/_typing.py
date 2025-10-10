from __future__ import annotations

from typing import TYPE_CHECKING

from cmomy.core.docstrings import docfiller
from cmomy.core.typing_compat import TypedDict

if TYPE_CHECKING:
    from typing import Any  # noqa: F401

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

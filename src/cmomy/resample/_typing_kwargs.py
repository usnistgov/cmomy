from __future__ import annotations

from typing import TYPE_CHECKING

from cmomy.core._typing_kwargs import (
    DataKACFKwargs,
    ValsCFKwargs,
)
from cmomy.core.typing_compat import TypedDict

if TYPE_CHECKING:
    from typing import (
        Required,
    )

    from .typing import SamplerType


class _SamplerKwargs(TypedDict, total=False):
    """Sampler data."""

    sampler: Required[SamplerType]
    rep_dim: str


class ResampleDataKwargs(  # type: ignore[call-arg]
    DataKACFKwargs,
    _SamplerKwargs,
    total=False,
    closed=True,
):
    """Extra parameters to :func:`.resample.resample_data`"""


class ResampleValsKwargs(  # type: ignore[call-arg]  # pylint: disable=duplicate-bases
    ValsCFKwargs,
    _SamplerKwargs,
    total=False,
    closed=True,
):
    """Extra parameters for :func:`.resample.resample_vals`"""

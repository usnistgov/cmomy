from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from cmomy.core.docstrings import docfiller
from cmomy.core.missing import MISSING

from ._sampler import IndexSampler

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike

    from cmomy.core.moment_params import MomParamsType
    from cmomy.core.typing import (
        AxisReduceWrap,
        DimsReduce,
        MissingType,
        MomAxes,
        MomDims,
        MomNDim,
        NDArrayAny,
        RngTypes,
    )

    from .typing import SamplerType


@docfiller.decorate
def factory_sampler(  # noqa: PLR0913
    sampler: SamplerType | None = None,
    *,
    # factory sampler parameters
    freq: NDArrayAny | xr.DataArray | xr.Dataset | None = None,
    indices: NDArrayAny | xr.DataArray | xr.Dataset | None = None,
    nrep: int | None = None,
    ndat: int | None = None,
    nsamp: int | None = None,
    paired: bool = True,
    rng: RngTypes | None = None,
    replace: bool = True,
    shuffle: bool = False,
    # other parameters
    data: ArrayLike | xr.DataArray | xr.Dataset | None = None,
    axis: AxisReduceWrap | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: MomNDim | None = None,
    mom_axes: MomAxes | None = None,
    mom_dims: MomDims | None = None,
    mom_params: MomParamsType = None,
    rep_dim: str = "rep",
    parallel: bool | None = None,
) -> IndexSampler[Any]:
    """
    Factory method to create sampler.

    The main intent of the function is to be called by other functions/method
    that need a sampler. For example, it is used in `.resample_data`. You can
    pass in a frequency array, an :class:`IndexSampler`, or a mapping to create an
    :class:`IndexSampler`. The order of evaluation is as follows:

    #. ``sampler`` is a :class:`IndexSampler`: return ``sampler``.
    #. ``sampler`` is ``None``:
        - if specify ``ndat``: return ``IndexSampler.from_param(...)``
        - if specify ``data``: return ``IndexSampler.from_data(...)``
    #. ``sampler`` is array-like: return ``IndexSampler(freq=sampler, ...)``
    #. ``sampler`` is an int, return ``IndexSampler.from_data(..., nrep=sampler)``
    #. ``sampler`` is a mapping: return ``factory_sampler(**sampler, data=data, axis=axis, dim=dims, mom_ndim=mom_ndim, mom_dims=mom_dims, rep_dim=rep_dim)``.


    Parameters
    ----------
    {sampler}
    {freq_xarray}
    {indices}
    {nrep}
    {ndat}
    {nsamp}
    {paired}
    {rng}
    {resample_replace}
    shuffle : bool
    data : array-like
        If needed, extract ``ndat`` from data.  Also used if ``paired = True``.
    {axis}
    {dim}
    {mom_ndim_optional}
    {mom_axes}
    {mom_dims_data}
    {mom_params}
    {rep_dim}
    {parallel}

    Returns
    -------
    IndexSampler

    See Also
    --------
    IndexSampler.from_params
    .from_data


    Examples
    --------
    >>> a = factory_sampler(nrep=3, ndat=2, rng=0)

    >>> b = factory_sampler(dict(nrep=3, ndat=2, rng=0))
    >>> c = factory_sampler(dict(freq=a.freq))
    >>> d = factory_sampler(a)
    >>> for other in [b, c, d]:
    ...     np.testing.assert_equal(a.freq, other.freq)
    >>> assert d is a

    To instead just pass indices, use:

    >>> e = factory_sampler(dict(indices=a.indices))
    >>> assert a.indices is e.indices

    """
    if isinstance(sampler, IndexSampler):
        return sampler

    if sampler is None:
        if indices is not None or freq is not None:
            return IndexSampler(
                indices=indices,  # pyright: ignore[reportArgumentType]
                freq=freq,  # pyright: ignore[reportArgumentType]
                ndat=ndat,
                parallel=parallel,
                shuffle=False,
                rng=rng,
                fastpath=False,
            )

        if nrep is not None:
            if ndat is not None:
                return IndexSampler.from_params(
                    nrep=nrep,
                    ndat=ndat,
                    nsamp=nsamp,
                    rng=rng,
                    replace=replace,
                    parallel=parallel,
                )
            if data is not None:
                return IndexSampler.from_data(
                    data=data,
                    nrep=nrep,
                    nsamp=nsamp,
                    axis=axis,
                    dim=dim,
                    mom_ndim=mom_ndim,
                    mom_axes=mom_axes,
                    mom_dims=mom_dims,
                    mom_params=mom_params,
                    rep_dim=rep_dim,
                    paired=paired,
                    rng=rng,
                    replace=replace,
                    parallel=parallel,
                )
        msg = "Must specify nrep and data if not passing indices or freq"
        raise ValueError(msg)

    if isinstance(sampler, (np.ndarray, xr.DataArray, xr.Dataset)):
        # fallback to freq
        return IndexSampler(
            freq=sampler,  # pyright: ignore[reportArgumentType]
            ndat=ndat,
            parallel=parallel,
            shuffle=shuffle,
            rng=rng,
            fastpath=False,
        )

    if isinstance(sampler, int):
        sampler = {"nrep": sampler}

    return factory_sampler(
        sampler=None,
        **sampler,
        data=data,
        axis=axis,
        dim=dim,
        mom_ndim=mom_ndim,
        mom_axes=mom_axes,
        mom_dims=mom_dims,
        mom_params=mom_params,
        rep_dim=rep_dim,
        parallel=parallel,
    )

"""Class to handle resampling indices/freqs"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, overload

import numpy as np
import xarray as xr

from cmomy.core.docstrings import docfiller
from cmomy.core.missing import MISSING
from cmomy.core.typing import (
    SamplerArrayT,
)
from cmomy.core.validate import (
    is_dataarray,
    is_dataset,
    is_xarray,
    validate_optional_mom_dims_and_mom_ndim,
)
from cmomy.core.xr_utils import select_axis_dim
from cmomy.factory import (
    factory_freq_to_indices,
    factory_indices_to_freq,
    parallel_heuristic,
)
from cmomy.random import validate_rng

if TYPE_CHECKING:
    from collections.abc import Hashable
    from typing import Any, Literal

    from numpy.typing import ArrayLike

    from cmomy.core.typing import (
        AxisReduce,
        DataT,
        DimsReduce,
        IndexSamplerFromDataKwargs,
        MissingType,
        Mom_NDim,
        MomDims,
        NDArrayAny,
        RngTypes,
        Sampler,
    )
    from cmomy.core.typing_compat import Unpack


# * Class interface -----------------------------------------------------------
@docfiller.decorate
class IndexSampler(Generic[SamplerArrayT]):
    """
    Wrapper around `indices` and `freq` resample arrays

    This a convenience wrapper class to make working with resampling indices
    straightforward.  :mod:`cmomy` primarily performs resampling using
    `frequency` tables instead of the more standard resampling `indices` arrays.  This class keeps track
    of both.

    Parameters
    ----------
    indices : ndarray, DataArray, or Dataset
        Indices resampling array.
    freq: ndarray, DataArray, or Dataset
        Frequency resampling table.
    {ndat}
    {parallel}
    {shuffle}
    {rng}
    {fastpath}
    """

    def __init__(
        self,
        *,
        indices: SamplerArrayT | None = None,
        freq: SamplerArrayT | None = None,
        ndat: int | None = None,
        parallel: bool | None = None,
        shuffle: bool = False,
        rng: RngTypes | None = None,
        fastpath: bool = False,
    ) -> None:
        self._indices: SamplerArrayT | None = indices
        self._freq: SamplerArrayT | None = freq
        self._ndat: int | None = ndat

        self._parallel = parallel
        self._shuffle = shuffle
        self._rng = rng

        if not fastpath:
            self._check()

    def _check(self) -> None:
        if self._indices is None and self._freq is None:
            msg = "Must specify indices or freq"
            raise ValueError(msg)

        if self._ndat is not None and self._freq is not None:
            freq_ndat = self._first_freq.shape[-1]
            if freq_ndat != self._ndat:
                msg = f"ndat={self._ndat} != freq.shape[-1]={freq_ndat}"
                raise ValueError(msg)
        # check indices?

    @property
    def freq(self) -> SamplerArrayT:
        if self._freq is None:
            self._freq = indices_to_freq(
                self.indices, ndat=self.ndat, parallel=self._parallel
            )
        return self._freq

    @property
    def indices(self) -> SamplerArrayT:
        if self._indices is None:
            self._indices = freq_to_indices(
                self.freq, shuffle=self._shuffle, rng=self._rng, parallel=self._parallel
            )
        return self._indices

    @property
    def _first_indices(self) -> NDArrayAny | xr.DataArray:
        if is_dataset(self.indices):
            return next(iter(self.indices.values()))
        return self.indices

    @property
    def _first_freq(self) -> NDArrayAny | xr.DataArray:
        if is_dataset(self.freq):
            return next(iter(self.freq.values()))
        return self.freq

    @property
    def _first(self) -> NDArrayAny | xr.DataArray:
        if self._freq is not None:
            return self._first_freq
        return self._first_indices

    @property
    def ndat(self) -> int:
        if self._ndat is None:
            self._ndat = self._first.shape[-1]
        return self._ndat

    @property
    def nrep(self) -> int:
        return self._first.shape[0]

    def __repr__(self) -> str:
        return f"<IndexSampler(nrep: {self.nrep}, ndat: {self.ndat})>"

    # * constructors
    @classmethod
    @docfiller.decorate
    def from_params(
        cls: type[IndexSampler[Any]],
        nrep: int,
        ndat: int,
        nsamp: int | None = None,
        rng: RngTypes | None = None,
        replace: bool = True,
        parallel: bool | None = None,
    ) -> IndexSampler[NDArrayAny]:
        """
        Create sampler from parameters

        Parameters
        ----------
        {nrep}
        {ndat}
        {nsamp}
        {rng}
        {resample_replace}
        {parallel}

        Returns
        -------
        resample : IndexSampler
            Wrapped object will be an :class:`~numpy.ndarray` of integers.
        """
        indices: NDArrayAny = random_indices(
            nrep=nrep, ndat=ndat, nsamp=nsamp, rng=rng, replace=replace
        )
        return cls(indices=indices, ndat=ndat, parallel=parallel, fastpath=True)

    @overload
    @classmethod
    def from_data(
        cls: type[IndexSampler[Any]],
        data: xr.DataArray,
        *,
        paired: bool = ...,
        **kwargs: Unpack[IndexSamplerFromDataKwargs],
    ) -> IndexSampler[xr.DataArray]: ...
    @overload
    @classmethod
    def from_data(  # pyright: ignore[reportOverlappingOverload]
        cls: type[IndexSampler[Any]],
        data: xr.Dataset,
        *,
        paired: Literal[True] = ...,
        **kwargs: Unpack[IndexSamplerFromDataKwargs],
    ) -> IndexSampler[xr.DataArray]: ...
    @overload
    @classmethod
    def from_data(
        cls: type[IndexSampler[Any]],
        data: xr.Dataset,
        *,
        paired: bool = ...,
        **kwargs: Unpack[IndexSamplerFromDataKwargs],
    ) -> IndexSampler[xr.DataArray | xr.Dataset]: ...
    @overload
    @classmethod
    def from_data(
        cls: type[IndexSampler[Any]],
        data: ArrayLike,
        *,
        paired: bool = ...,
        **kwargs: Unpack[IndexSamplerFromDataKwargs],
    ) -> IndexSampler[NDArrayAny]: ...

    @classmethod
    @docfiller.decorate
    def from_data(
        cls: type[IndexSampler[Any]],
        data: ArrayLike | xr.DataArray | xr.Dataset,
        *,
        nrep: int,
        nsamp: int | None = None,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        mom_ndim: Mom_NDim | None = None,
        mom_dims: MomDims | None = None,
        rep_dim: str = "rep",
        paired: bool = True,
        rng: RngTypes | None = None,
        replace: bool = True,
        parallel: bool | None = None,
    ) -> (
        IndexSampler[NDArrayAny]
        | IndexSampler[xr.DataArray]
        | IndexSampler[xr.DataArray | xr.Dataset]
    ):
        """
        Create sampler for ``data``.

        Parameters
        ----------
        data : ndarray, DataArray, or Dataset
        {nrep}
        {nsamp}
        {axis}
        {dim}
        {mom_ndim_optional}
        {mom_dims_data}
        {rep_dim}
        {paired}
        {rng}
        {resample_replace}
        {parallel}

        Returns
        -------
        sampler : IndexSampler
            Type of wrapped array depends on the passed parameters. In all cases,
            if ``data`` is an array, ``sampler`` will wrap an array, if ``data``
            is an :class:`~xarray.DataArray`, ``sampler`` will wrap an
            :class:`~xarray.DataArray`. If ``data`` is an :class:`~xarray.Dataset`,
            return a wrapped :class:`~xarray.DataArray` if ``paired=True`` or if
            the resulting Dataset has only one variable, and a
            :class:`~xarray.Dataset` otherwise.
        """
        ndat = select_ndat(
            data, axis=axis, dim=dim, mom_ndim=mom_ndim, mom_dims=mom_dims
        )

        indices: NDArrayAny | xr.DataArray | xr.Dataset
        if is_xarray(data):
            indices = _randsamp_indices_dataarray_or_dataset(  # type: ignore[type-var]
                data=data,  # pyright: ignore[reportArgumentType]
                nrep=nrep,
                axis=axis,
                dim=dim,
                nsamp=nsamp,
                ndat=ndat,
                rep_dim=rep_dim,
                paired=paired,
                mom_ndim=mom_ndim,
                mom_dims=mom_dims,
                rng=rng,
            )
        else:
            indices = random_indices(
                nrep=nrep,
                ndat=ndat,
                nsamp=nsamp,
                rng=rng,
                replace=replace,
            )

        return cls(indices=indices, ndat=ndat, parallel=parallel, fastpath=True)


@docfiller.decorate
def factory_sampler(  # noqa: PLR0913
    sampler: Sampler | None = None,
    *,
    freq: NDArrayAny | xr.DataArray | xr.Dataset | None = None,
    indices: NDArrayAny | xr.DataArray | xr.Dataset | None = None,
    ndat: int | None = None,
    nrep: int | None = None,
    nsamp: int | None = None,
    paired: bool = True,
    rng: RngTypes | None = None,
    replace: bool = True,
    shuffle: bool = False,
    data: ArrayLike | xr.DataArray | xr.Dataset | None = None,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: Mom_NDim | None = None,
    mom_dims: MomDims | None = None,
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
    #. ``sampler`` is a mapping: return ``factory_sampler(**sampler, data=data, axis=axis, dim=dims, mom_ndim=mom_ndim, mom_dims=mom_dims, rep_dim=rep_dim)``.


    Parameters
    ----------
    {sampler}
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
    {mom_dims_data}
    {rep_dim}
    {parallel}

    Returns
    -------
    IndexSampler

    See Also
    --------
    IndexSampler.from_params
    IndexSampler.from_data


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
                    mom_dims=mom_dims,
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

    return factory_sampler(
        data=data,
        axis=axis,
        dim=dim,
        mom_ndim=mom_ndim,
        mom_dims=mom_dims,
        rep_dim=rep_dim,
        parallel=parallel,
        **sampler,
    )


# * Create random indices -----------------------------------------------------
@docfiller.decorate
def random_indices(
    nrep: int,
    ndat: int,
    nsamp: int | None = None,
    rng: RngTypes | None = None,
    replace: bool = True,
) -> NDArrayAny:
    """
    Create indices for random resampling (bootstrapping).

    Parameters
    ----------
    {nrep}
    {ndat}
    {nsamp}
    {rng}
    replace :
        Whether to allow replacement.

    Returns
    -------
    indices : ndarray
        Index array of integers of shape ``(nrep, nsamp)``.
    """
    nsamp = ndat if nsamp is None else nsamp
    return validate_rng(rng).choice(ndat, size=(nrep, nsamp), replace=replace)


@docfiller.inherit(random_indices)
def random_freq(
    nrep: int,
    ndat: int,
    nsamp: int | None = None,
    rng: RngTypes | None = None,
    replace: bool = True,
    parallel: bool | None = None,
) -> NDArrayAny:
    """
    Create frequencies for random resampling (bootstrapping).

    Returns
    -------
    freq : ndarray
        Frequency array. ``freq[rep, k]`` is the number of times to sample from the `k`th
        observation for replicate `rep`.
    {parallel}


    See Also
    --------
    random_indices
    """
    return indices_to_freq(
        indices=random_indices(
            nrep=nrep, ndat=ndat, nsamp=nsamp, rng=rng, replace=replace
        ),
        ndat=ndat,
        parallel=parallel,
    )


# * DataArray/set of indices --------------------------------------------------
def _randsamp_indices_dataarray_or_dataset(
    data: DataT,
    *,
    nrep: int,
    ndat: int,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    nsamp: int | None = None,
    rep_dim: str = "rep",
    paired: bool = True,
    mom_ndim: Mom_NDim | None = None,
    mom_dims: MomDims | None = None,
    rng: RngTypes | None = None,
    replace: bool = True,
) -> xr.DataArray | DataT:
    """Create a resampling DataArray or Dataset."""
    mom_dims, mom_ndim = validate_optional_mom_dims_and_mom_ndim(
        mom_dims, mom_ndim, data
    )
    dim = select_axis_dim(data, axis=axis, dim=dim, mom_dims=mom_dims)[1]
    # make sure to validate rng here in case call multiple have dataset
    rng = validate_rng(rng)

    def _get_unique_indices() -> xr.DataArray:
        return xr.DataArray(
            random_indices(nrep=nrep, ndat=ndat, nsamp=nsamp, rng=rng, replace=replace),
            dims=[rep_dim, dim],
        )

    if is_dataarray(data) or paired:
        return _get_unique_indices()

    # generate non-paired dataset
    dims = {dim, *(() if mom_dims is None else mom_dims)}  # type: ignore[misc]
    out: dict[Hashable, xr.DataArray] = {}
    for name, da in data.items():
        if dims.issubset(da.dims):
            out[name] = _get_unique_indices()
    if len(out) == 1:
        # return just a dataarray in this case
        return next(iter(out.values()))
    return xr.Dataset(out)  # pyright: ignore[reportReturnType]


# * select ndat ---------------------------------------------------------------
@docfiller.decorate
def select_ndat(
    data: ArrayLike | xr.DataArray | xr.Dataset,
    *,
    axis: AxisReduce | MissingType = MISSING,
    dim: DimsReduce | MissingType = MISSING,
    mom_ndim: Mom_NDim | None = None,
    mom_dims: MomDims | None = None,
) -> int:
    """
    Determine ndat from array.

    Parameters
    ----------
    data : ndarray, DataArray, Dataset
    {axis}
    {dim}
    {mom_ndim_optional}
    {mom_dims_data}

    Returns
    -------
    int
        size of ``data`` along specified ``axis`` or ``dim``

    Examples
    --------
    >>> data = np.zeros((2, 3, 4))
    >>> select_ndat(data, axis=1)
    3
    >>> select_ndat(data, axis=-1, mom_ndim=2)
    2


    >>> xdata = xr.DataArray(data, dims=["x", "y", "mom"])
    >>> select_ndat(xdata, dim="y")
    3
    >>> select_ndat(xdata, dim="mom", mom_ndim=1)
    Traceback (most recent call last):
    ...
    ValueError: Cannot select moment dimension. dim='mom', axis=2.
    """
    from cmomy.core.array_utils import normalize_axis_index

    if is_xarray(data):
        mom_dims, mom_ndim = validate_optional_mom_dims_and_mom_ndim(
            mom_dims, mom_ndim, data
        )
        axis, dim = select_axis_dim(data, axis=axis, dim=dim, mom_dims=mom_dims)
        return data.sizes[dim]

    if isinstance(axis, int):
        data = np.asarray(data)
        axis = normalize_axis_index(axis, data.ndim, mom_ndim=mom_ndim)
        return data.shape[axis]
    msg = "Must specify integer axis for array input."
    raise TypeError(msg)


# * Convert -------------------------------------------------------------------
@overload
def freq_to_indices(
    freq: DataT,
    *,
    shuffle: bool = ...,
    rng: RngTypes | None = ...,
    parallel: bool | None = ...,
) -> DataT: ...
@overload
def freq_to_indices(
    freq: ArrayLike,
    *,
    shuffle: bool = ...,
    rng: RngTypes | None = ...,
    parallel: bool | None = ...,
) -> NDArrayAny: ...


@docfiller.decorate  # type: ignore[arg-type, unused-ignore]
def freq_to_indices(
    freq: ArrayLike | DataT,
    *,
    shuffle: bool = False,
    rng: RngTypes | None = None,
    parallel: bool | None = None,
) -> NDArrayAny | DataT:
    """
    Convert a frequency array to indices array.

    This creates an "indices" array that is compatible with "freq" array.
    Note that by default, the indices for a single sample (along output[k, :])
    are in sorted order (something like [[0, 0, ..., 1, 1, ...], ...]).
    Pass ``shuffle = True`` to randomly shuffle indices along ``axis=1``.

    Parameters
    ----------
    {freq_xarray}
    {shuffle}
    {rng}
    {parallel}

    Returns
    -------
    ndarray :
        Indices array of shape ``(nrep, nsamp)`` where ``nsamp = freq[k,
        :].sum()`` where `k` is any row.
    """
    if is_xarray(freq):
        rep_dim, dim = freq.dims
        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            freq_to_indices,
            freq,
            input_core_dims=[[rep_dim, dim]],
            output_core_dims=[[rep_dim, dim]],
            # possible for data dimension to change size.
            exclude_dims={dim},
            kwargs={"shuffle": shuffle, "rng": rng, "parallel": parallel},
        )
        return xout

    freq = np.asarray(freq, np.int64)
    nsamps = freq.sum(axis=-1)
    nsamp = nsamps[0]
    if np.any(nsamps != nsamp):
        msg = "Inconsistent number of samples from freq array"
        raise ValueError(msg)

    indices = np.empty((freq.shape[0], nsamp), dtype=np.int64)
    factory_freq_to_indices(parallel=parallel_heuristic(parallel, size=freq.size))(
        freq,
        indices,
    )

    if shuffle:
        rng = validate_rng(rng)
        rng.shuffle(indices, axis=1)
    return indices


@overload
def indices_to_freq(
    indices: DataT,
    *,
    ndat: int | None = ...,
    parallel: bool | None = ...,
) -> DataT: ...
@overload
def indices_to_freq(
    indices: ArrayLike,
    *,
    ndat: int | None = ...,
    parallel: bool | None = ...,
) -> NDArrayAny: ...


def indices_to_freq(
    indices: ArrayLike | DataT,
    *,
    ndat: int | None = None,
    parallel: bool | None = None,
) -> NDArrayAny | DataT:
    """
    Convert indices to frequency array.

    It is assumed that ``indices.shape == (nrep, nsamp)`` with ``nsamp ==
    ndat``. For cases that ``nsamp != ndat``, pass in ``ndat`` explicitl.
    """
    if is_xarray(indices):
        # assume dims are in order (rep, dim)
        rep_dim, dim = indices.dims
        ndat = ndat or indices.sizes[dim]
        xout: DataT = xr.apply_ufunc(  # pyright: ignore[reportUnknownMemberType]
            indices_to_freq,
            indices,
            input_core_dims=[[rep_dim, dim]],
            output_core_dims=[[rep_dim, dim]],
            # allow for possibility that dim will change size...
            exclude_dims={dim},
            kwargs={"ndat": ndat, "parallel": parallel},
        )
        return xout

    indices = np.asarray(indices, np.int64)
    ndat = indices.shape[1] if ndat is None else ndat

    if indices.max() >= ndat:
        msg = f"Cannot have values >= {ndat=}"
        raise ValueError(msg)

    freq = np.zeros((indices.shape[0], ndat), dtype=indices.dtype)

    factory_indices_to_freq(parallel=parallel_heuristic(parallel, size=freq.size))(
        indices, freq
    )

    return freq


# * jackknife frequency
@docfiller.decorate
def jackknife_freq(
    ndat: int,
) -> NDArrayAny:
    r"""
    Frequency array for jackknife resampling.

    Use this frequency array to perform jackknife [1]_ resampling

    Parameters
    ----------
    {ndat}

    Returns
    -------
    freq : ndarray
        Frequency array for jackknife resampling.

    See Also
    --------
    jackknife_vals
    jackknife_data
    .reduction.reduce_vals
    .reduction.reduce_data

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Jackknife_resampling

    Examples
    --------
    >>> jackknife_freq(4)
    array([[0, 1, 1, 1],
           [1, 0, 1, 1],
           [1, 1, 0, 1],
           [1, 1, 1, 0]])

    """
    freq = np.ones((ndat, ndat), dtype=np.int64)
    np.fill_diagonal(freq, 0.0)
    return freq

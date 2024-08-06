"""Base class for central moments calculations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, cast

import numpy as np
from module_utilities import cached

from ._lib.factory import factory_pusher, parallel_heuristic
from .core.array_utils import (
    axes_data_reduction,
    normalize_axis_index,
)
from .core.docstrings import docfiller
from .core.prepare import (
    prepare_data_for_reduction,
    prepare_values_for_reduction,
)
from .core.validate import (
    validate_axis,
    validate_floating_dtype,
    validate_mom_ndim,
)
from .utils import assign_moment, moment_indexer

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence
    from typing import Any, Callable

    import xarray as xr
    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from ._lib.factory import Pusher
    from .core.typing import (
        ArrayOrder,
        ArrayOrderCF,
        AxisReduce,
        DataCasting,
        Groups,
        MissingType,
        Mom_NDim,
        Moments,
        MomentsStrict,
        NDArrayAny,
        NDArrayInt,
        SelectMoment,
    )
    from .core.typing_compat import Self

from .core.typing import ArrayT, FloatT


# * Main class ----------------------------------------------------------------
@docfiller.decorate  # noqa: PLR0904
class CentralMomentsABC(ABC, Generic[FloatT, ArrayT]):
    r"""
    Wrapper to calculate central moments.

    Parameters
    ----------
    data : {t_array}
        Central moments array.
    {mom_ndim}
    {copy}
    {order}
    {dtype}
    fastpath : bool
        For internal use.

    Notes
    -----
    Base data has the form

    .. math::

        {{\rm data}}[..., i, j] = \begin{{cases}}
             \text{{weight}} & i = j = 0 \\
            \langle x \rangle & i = 1, j = 0 \\
            \langle (x - \langle x \rangle^i) (y - \langle y \rangle^j)
            \rangle & i + j > 0
        \end{{cases}}
    """

    _cache: dict[str, Any]
    _data: NDArray[FloatT]
    _mom_ndim: Mom_NDim

    __slots__ = (
        "_cache",
        "_data",
        "_mom_ndim",
    )

    def __init__(
        self,
        data: ArrayT,
        *,
        mom_ndim: Mom_NDim = 1,
        copy: bool | None = None,  # noqa: ARG002
        order: ArrayOrder = None,  # noqa: ARG002
        dtype: DTypeLike = None,  # noqa: ARG002
        fastpath: bool = False,  # noqa: ARG002
    ) -> None:  # pragma: no cover
        self._cache = {}
        self._mom_ndim = validate_mom_ndim(mom_ndim)
        self.set_values(data)

        # other checks
        if self.ndim < self.mom_ndim:
            msg = f"{self.ndim=} < {self.mom_ndim=}"
            raise ValueError(msg)

        # must have positive moments
        if any(m <= 0 for m in self.mom):
            msg = "moments must be positive"
            raise ValueError(msg)

        # only float32 or float64 allowed
        _ = validate_floating_dtype(self.dtype)

    # ** Basic access -------------------------------------------------------------
    @property
    def values(self) -> ArrayT:
        """Access underlying central moments array."""
        return self.to_values()

    @abstractmethod
    def set_values(self, values: ArrayT) -> None:
        """Set values."""

    @abstractmethod
    def to_values(self) -> ArrayT:
        """Access underlying values"""

    @property
    def data(self) -> NDArray[FloatT]:
        """
        Accessor to numpy array underlying data.

        By convention data has the following meaning for the moments indexes

        * `data[...,i=0,j=0]`, weight
        * `data[...,i=1,j=0]]`, if only one moment index is one and all others zero,
          then this is the average value of the variable with unit index.
        * all other cases, the central moments `<(x0-<x0>)**i0 * (x1 - <x1>)**i1 * ...>`

        """
        return self._data

    def to_numpy(self) -> NDArray[FloatT]:
        """Access to numpy array underlying class."""
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        """self.data.shape."""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """self.data.ndim."""
        return self._data.ndim

    @property
    def dtype(self) -> np.dtype[FloatT]:
        """self.data.dtype."""
        # Not sure why I have to cast
        return self._data.dtype

    @property
    def mom_ndim(self) -> Mom_NDim:
        """
        Length of moments.

        if `mom_ndim` == 1, then single variable
        moments if `mom_ndim` == 2, then co-moments.
        """
        return self._mom_ndim

    @property
    def mom_shape(self) -> MomentsStrict:
        """Shape of moments part."""
        return cast("MomentsStrict", self._data.shape[-self._mom_ndim :])

    @property
    def mom(self) -> MomentsStrict:
        """Number of moments."""  # D401
        return tuple(x - 1 for x in self.mom_shape)  # type: ignore[return-value]

    @property
    def val_shape(self) -> tuple[int, ...]:
        """
        Shape of values dimensions.

        That is shape less moments dimensions.
        """
        return self._data.shape[: -self.mom_ndim]

    @property
    def val_ndim(self) -> int:
        """Number of value dimensions."""  # D401
        return len(self.val_shape)

    def __repr__(self) -> str:
        """Repr for class."""
        name = self.__class__.__name__
        s = f"<{name}(val_shape={self.val_shape}, mom={self.mom})>\n"
        return s + repr(self.to_values())

    def _repr_html_(self) -> str:  # noqa: PLW3201
        from .core.formatting import (
            repr_html,  # pyright: ignore[reportUnknownVariableType]
        )

        return repr_html(self)  # type: ignore[no-any-return,no-untyped-call]

    def _check_array_mom_shape(self, data: NDArrayAny | xr.DataArray) -> None:
        """Check that new data has same mom_shape as `self`."""
        if data.shape[-self._mom_ndim :] != self.mom_shape:
            # at a minimum, verify that mom_shape is unchanged.
            msg = f"{data.shape=} has wrong mom_shape={self.mom_shape}"
            raise ValueError(msg)

    def __getitem__(self, key: Any) -> Self:
        """
        Get new object by indexing.

        Note that only objects with the same moment(s) shape are allowed.

        If you want to extract data in general, use `self.to_values()[....]`.
        """
        data = self.to_values()[key]
        self._check_array_mom_shape(data)
        return type(self)(
            data=data,
            mom_ndim=self._mom_ndim,
            fastpath=True,
        )

    def __array__(  # noqa: PLW3201
        self, dtype: DTypeLike = None, copy: bool | None = None
    ) -> NDArray[FloatT]:
        """Used by np.array(self)."""  # D401
        return np.asarray(self._data, dtype=dtype)

    # ** Utils ----------------------------------------------------------------
    def _wrap_axis(
        self,
        axis: int | None,
        ndim: int | None = None,
    ) -> int:
        """Wrap axis to positive value and check."""
        if ndim is None:
            ndim = self.val_ndim

        return normalize_axis_index(validate_axis(axis), ndim)

    # @staticmethod
    # def _set_default_axis(axis: int | None, default: int = -1) -> int:
    #     return default if axis is None else axis  # noqa: ERA001

    @property
    def _is_vector(self) -> bool:
        return self.val_ndim > 0

    def _raise_if_scalar(self, message: str | None = None) -> None:
        if not self._is_vector:
            if message is None:  # pragma: no cover
                message = "Not implemented for scalar"
            raise ValueError(message)

    def _raise_if_not_mom_ndim_1(self) -> None:
        if self._mom_ndim != 1:
            msg = "Only implemented for `mom_ndim==1`"
            raise ValueError(msg)

    # ** top level creation/copy/new ----------------------------------------------
    @abstractmethod
    @docfiller.decorate
    def new_like(
        self,
        data: ArrayT | None = None,
        *,
        copy: bool | None = None,
        order: ArrayOrder | None = None,
        verify: bool = False,
        dtype: DTypeLike = None,
    ) -> Self:
        """
        Create new object like self, with new data.

        Parameters
        ----------
        data : {t_array}
            data for new object
        {copy}
        {order}
        {verify}
        {dtype}

        Returns
        -------
        {klass}
            New {klass} object with zerod out data.

        """

    @docfiller.decorate
    def astype(
        self,
        dtype: DTypeLike,
        *,
        order: ArrayOrder = None,
        casting: DataCasting = None,
        subok: bool | None = None,
        copy: bool = False,
    ) -> Self:
        """
        Underlying data cast to specified type

        Parameters
        ----------
        dtype : str or dtype
            Typecode of data-type to cast the array data.  Note that a value of None will
            upcast to ``np.float64``.  This is the same behaviour as :func:`~numpy.asarray`.
        {order}
        casting : {{'no', 'equiv', 'safe', 'same_kind', 'unsafe'}}, optional
            Controls what kind of data casting may occur.

            - 'no' means the data types should not be cast at all.
            - 'equiv' means only byte-order changes are allowed.
            - 'safe' means only casts which can preserve values are allowed.
            - 'same_kind' means only safe casts or casts within a kind,
              like float64 to float32, are allowed.
            - 'unsafe' (default) means any data conversions may be done.

        subok : bool, optional
            If True, then sub-classes will be passed-through, otherwise the
            returned array will be forced to be a base-class array.
        copy : bool, optional
            By default, astype always returns a newly allocated array. If this
            is set to False and the `dtype` requirement is satisfied, the input
            array is returned insteadof a copy.


        Notes
        -----
        Only ``numpy.float32`` and ``numpy.float64`` dtypes are supported.


        See Also
        --------
        numpy.ndarray.astype
        """
        validate_floating_dtype(dtype)

        kwargs = {"order": order, "casting": casting, "subok": subok, "copy": copy}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        return type(self)(
            data=self.to_values().astype(dtype=dtype, **kwargs),  # type: ignore[arg-type]
            mom_ndim=self._mom_ndim,
            # Already validated dtype, so can use fastpath
            fastpath=True,
        )

    @docfiller.decorate
    def zeros_like(self) -> Self:
        """
        Create new empty object like self.

        Returns
        -------
        output : {klass}
            Object with same attributes as caller, but with data set to zero.

        See Also
        --------
        new_like
        """
        return self.new_like()

    @docfiller.decorate
    def copy(self) -> Self:
        """
        Create a new object with copy of data.

        Parameters
        ----------
        **copy_kws
            passed to parameter ``copy_kws`` in method :meth:`new_like`

        Returns
        -------
        output : {klass}
            Same type as calling class.
            Object with same attributes as caller, but with new underlying data.


        See Also
        --------
        new_like
        zeros_like
        """
        return self.new_like(
            data=self.to_values(),
            verify=False,
            copy=True,
        )

    @docfiller.decorate
    def moments_to_comoments(self, *, mom: tuple[int, int]) -> Self:
        """
        Convert moments (mom_ndim=1) to comoments (mom_ndim=2).

        Parameters
        ----------
        {mom_moments_to_comoments}

        See Also
        --------
        .convert.moments_to_comoments
        """
        self._raise_if_not_mom_ndim_1()

        from . import convert

        return type(self)(
            convert.moments_to_comoments(  # pyright: ignore[reportArgumentType]
                self.to_values(),
                mom=mom,
                dtype=self.dtype,
            ),
            mom_ndim=2,
        )

    @docfiller.decorate
    def assign_moment(
        self,
        name: SelectMoment,
        value: ArrayLike,
        copy: bool = True,
        squeeze: bool = True,
    ) -> Self:
        """
        Create object with update weight, average, etc.

        Parameters
        ----------
        {select_name}
        value : array-like
        copy : bool, default=True
            If ``True`` (default), copy the underlying moments data before update.
            Otherwise, update weights underlying data in place.

        Returns
        -------
        output : object
            Same type as ``self`` with updated data.

        Returns
        -------
        output : object
            Same type as ``self``
        """
        return type(self)(
            data=assign_moment(
                data=self.to_values(),
                name=name,
                value=value,
                mom_ndim=self._mom_ndim,
                copy=copy,
                squeeze=squeeze,
            ),
            mom_ndim=self._mom_ndim,
            fastpath=True,
        )

    # ** Access to underlying statistics ------------------------------------------
    def select_moment(
        self,
        name: SelectMoment,
        *,
        squeeze: bool = True,
        dim_combined: str = "variable",
        coords_combined: str | Sequence[Hashable] | None = None,
        keep_attrs: bool | None = None,
    ) -> ArrayT:
        """
        Select specific moments.

        See Also
        --------
        .utils.select_moment
        """
        from .utils import select_moment

        return cast(
            "ArrayT",
            select_moment(  # type: ignore[call-overload]
                self.to_values(),
                name=name,
                mom_ndim=self._mom_ndim,
                dim_combined=dim_combined,
                coords_combined=coords_combined,
                squeeze=squeeze,
                keep_attrs=keep_attrs,
            ),
        )

    def weight(self) -> ArrayT:
        """Weight data."""
        return self.select_moment("weight")

    def mean(self, squeeze: bool = True) -> ArrayT:
        """Mean (first moment)."""
        return self.select_moment("ave", squeeze=squeeze)

    def var(self, squeeze: bool = True) -> ArrayT:
        """Variance (second central moment)."""
        return self.select_moment("var", squeeze=squeeze)

    def std(self, squeeze: bool = True) -> ArrayT:
        """Standard deviation."""  # D401
        return np.sqrt(self.var(squeeze))  # type: ignore[return-value]

    def cov(self) -> ArrayT:
        """Covariance (or variance if ``mom_ndim==1``)."""
        return self.select_moment("cov")

    def cmom(self) -> ArrayT:
        r"""
        Central moments.

        Strict central moments of the form

        .. math::

            \text{cmom[..., n, m]} =
            \langle (x - \langle x \rangle)^n (y - \langle y \rangle)^m \rangle

        where

        .. math::
            \langle x \rangle = \sum_i w_i x_i / \sum_i w_i

        Returns
        -------
        output : ndarray or DataArray
        """
        # Set weight to 1
        out = assign_moment(
            self.to_values(), name="weight", value=1, mom_ndim=self._mom_ndim, copy=True
        )

        # Set first central moment to zero
        return assign_moment(
            out,
            name="ave",
            value=0,
            mom_ndim=self._mom_ndim,
            copy=False,
        )

    def to_raw(self, *, weight: float | NDArrayAny | None = None) -> ArrayT:
        r"""
        Raw moments accumulation array.

        .. math::

            \text{raw[..., n, m]} = \begin{cases}
            \text{weight} & n = m = 0 \\
            \langle x^n y ^m \rangle & \text{otherwise}
            \end{cases}

        where

        .. math::
            \langle x \rangle = \sum_i w_i x_i / \sum_i w_i


        Returns
        -------
        raw : ndarray or DataArray

        See Also
        --------
        from_raw
        .convert.moments_type
        """
        from . import convert

        out = convert.moments_type(self.to_values(), mom_ndim=self._mom_ndim, to="raw")
        if weight is not None:
            out = assign_moment(
                out, "weight", weight, mom_ndim=self._mom_ndim, copy=False
            )
        return out  # pyright: ignore[reportReturnType]

    def rmom(self) -> ArrayT:
        r"""
        Raw moments.

        .. math::
            \text{rmom[..., n, m]} = \langle x^n y^m  \rangle

        where

        .. math::
            \langle x \rangle = \sum_i w_i x_i / \sum_i w_i


        Returns
        -------
        raw_moments : ndarray or DataArray

        See Also
        --------
        to_raw
        .convert.moments_type
        cmom
        """
        return self.to_raw(weight=1.0)

    # ** Fill ---------------------------------------------------------------------
    def fill(self, value: Any = 0) -> Self:
        """
        Fill data with value.

        Parameters
        ----------
        value : scalar
            Value to insert into `self.data`

        Returns
        -------
        self : object
            Same type as calling class.
            Same object as caller with data filled with `values`
        """
        self._data.fill(value)
        return self

    def zero(self) -> Self:
        """
        Zero out underlying data.

        Returns
        -------
        self : object
            Same type as calling class.
            Same object with data filled with zeros.

        See Also
        --------
        fill
        """
        return self.fill(value=0.0)

    # ** pushing routines ---------------------------------------------------------
    @cached.prop
    def _use_parallel(self) -> bool:
        return parallel_heuristic(None, self.data.size * self.mom_ndim)

    def _pusher(self, parallel: bool | None = None) -> Pusher:
        return factory_pusher(
            mom_ndim=self._mom_ndim,
            parallel=self._use_parallel if parallel is None else parallel,
        )

    # Low level pushers
    def _push_data_numpy(
        self,
        data: ArrayLike,
        *,
        parallel: bool | None = None,
    ) -> Self:
        data = np.asarray(data, dtype=self.dtype)
        self._pusher(parallel).data(data, self._data)
        return self

    def _push_datas_numpy(
        self,
        datas: ArrayLike,
        *,
        axis: AxisReduce | MissingType,
        parallel: bool | None = None,
    ) -> Self:
        axis, datas = prepare_data_for_reduction(
            data=datas,
            axis=axis,
            mom_ndim=self.mom_ndim,
            dtype=self.dtype,
        )
        axes = axes_data_reduction(mom_ndim=self.mom_ndim, axis=axis)

        self._pusher(parallel).datas(
            datas,
            self._data,
            axes=axes,
        )

        return self

    @staticmethod
    def _check_y(y: tuple[ArrayLike, ...], mom_ndim: int) -> None:
        if len(y) + 1 != mom_ndim:
            msg = f"Number of arrays {len(y) + 1} != {mom_ndim=}"
            raise ValueError(msg)

    def _push_val_numpy(
        self,
        x: ArrayLike,
        *y: ArrayLike,
        weight: ArrayLike | None = None,
        parallel: bool | None = None,
    ) -> Self:
        self._check_y(y, self.mom_ndim)

        weight = 1.0 if weight is None else weight

        x = np.asarray(x, dtype=self.dtype)
        self._pusher(parallel).val(
            self._data,
            x,
            *(np.asarray(a, dtype=self.dtype) for a in (weight, *y)),
        )
        return self

    def _push_vals_numpy(
        self,
        x: ArrayLike,
        *y: ArrayLike,
        axis: AxisReduce | MissingType,
        weight: ArrayLike | None = None,
        parallel: bool | None = None,
    ) -> Self:
        self._check_y(y, self.mom_ndim)

        weight = 1.0 if weight is None else weight
        axis, args = prepare_values_for_reduction(
            x,
            weight,
            *y,
            axis=axis,
            dtype=self.dtype,
            narrays=self._mom_ndim + 1,
        )

        self._pusher(parallel).vals(
            self._data,
            *args,
        )
        return self

    @abstractmethod
    @docfiller.decorate
    def push_data(
        self,
        data: Any,
        *,
        parallel: bool | None = False,
    ) -> Self:
        """
        Push data object to moments.

        Parameters
        ----------
        data : array-like
            Accumulation array of same form as ``self.data``
        {order}
        {parallel}

        Returns
        -------
        output : {klass}
            Same object with pushed data.
        """

    @abstractmethod
    @docfiller.decorate
    def push_datas(
        self,
        datas: Any,
        *,
        axis: AxisReduce = -1,
        parallel: bool | None = None,
    ) -> Self:
        """
        Push and reduce multiple average central moments.

        Parameters
        ----------
        datas : array-like, {t_array}
            Collection of accumulation arrays to push onto ``self``.
            This should have shape like `(nrec,) + self.shape`
            if `axis=0`, where `nrec` is the number of data objects to sum.
        {axis_data_and_dim}
        {parallel}
        {order}

        Returns
        -------
        output : {klass}
            Same object with pushed data.
        """

    @abstractmethod
    @docfiller.decorate
    def push_val(
        self,
        x: ArrayLike,
        *y: ArrayLike,
        weight: ArrayLike | None = None,
        parallel: bool | None = False,
    ) -> Self:
        """
        Push single sample to central moments.

        Parameters
        ----------
        x : array
            Values to push onto ``self``.
        *y : array-like, optional
            Additional Values (needed if ``mom_ndim > 1``)
        weight : int, float, array-like, optional
            Weight of each sample.  If scalar, broadcast `w.shape` to `x0.shape`.
        {order}
        {parallel}

        Returns
        -------
        output : {klass}
            Same object with pushed data.

        Notes
        -----
        Array `x0` should have same shape as `self.val_shape`.
        """

    @abstractmethod
    @docfiller.decorate
    def push_vals(
        self,
        x: ArrayLike,
        *y: ArrayLike,
        axis: AxisReduce = -1,
        weight: ArrayLike | None = None,
        parallel: bool | None = None,
    ) -> Self:
        """
        Push multiple samples to central moments.

        Parameters
        ----------
        x : array
            Value to reduce.
        *y : array-like
            Additional array (if ``self.mom_ndim == 2``).
        weight : int, float, array-like, optional
            Weight of each sample.  If scalar, broadcast to `x0.shape`
        {axis_and_dim}
        {order}
        {parallel}

        Returns
        -------
        output : {klass}
            Same object with pushed data.
        """

    # * Reduction -----------------------------------------------------------------
    def randsamp_freq(
        self,
        *,
        axis: AxisReduce | MissingType = -1,
        nrep: int | None = None,
        nsamp: int | None = None,
        indices: ArrayLike | None = None,
        freq: ArrayLike | None = None,
        check: bool = False,
        rng: np.random.Generator | None = None,
        **kwargs: Any,
    ) -> NDArrayInt:
        """
        Interface to :func:`.resample.randsamp_freq`


        Parameters
        ----------
        axis : int
            Axis that will be resampled.  This is used to
            calculate ``ndat``.
        {nrep}
        {nsamp}
        {freq}
        {indices}
        check : bool, default=False
            If ``True``, perform sanity checks.
        {rng}

        Returns
        -------
        freq : ndarray
            Frequency array

        """
        from .resample import randsamp_freq

        return randsamp_freq(
            data=self.values,
            mom_ndim=self._mom_ndim,
            axis=axis,
            nrep=nrep,
            nsamp=nsamp,
            indices=indices,
            freq=freq,
            check=check,
            rng=rng,
            **kwargs,
        )

    @docfiller.decorate
    def resample_and_reduce(
        self,
        *,
        freq: NDArrayInt,
        axis: AxisReduce | MissingType = -1,
        # freq: NDArrayInt | None = None,  # noqa: ERA001
        # indices: NDArrayInt | None = None,  # noqa: ERA001
        # nrep: NDArrayInt | None = None,  # noqa: ERA001
        # rng: np.random.Generator | None = None,  # noqa: ERA001
        parallel: bool | None = None,
        order: ArrayOrder = None,
        **kwargs: Any,
    ) -> Self:
        """
        Bootstrap resample and reduce.

        Parameters
        ----------
        {freq}
        {axis_data_and_dim}
        {parallel}
        {order}

        Returns
        -------
        output : object
            Instance of calling class. Note that new object will have
            ``(...,shape[axis-1], nrep, shape[axis+1], ...)``,
            where ``nrep = freq.shape[0]``.


        See Also
        --------
        reduce
        ~cmomy.resample.randsamp_freq : random frequency sample
        ~cmomy.resample.freq_to_indices : convert frequency sample to index sample
        ~cmomy.resample.indices_to_freq : convert index sample to frequency sample
        ~cmomy.resample.resample_data : method to perform resampling
        """
        self._raise_if_scalar()
        from .resample import resample_data

        data: ArrayT = resample_data(
            self.to_values(),  # pyright: ignore[reportAssignmentType]
            freq=freq,
            mom_ndim=self._mom_ndim,
            axis=axis,
            parallel=parallel,
            dtype=self.dtype,
            **kwargs,
        )
        return type(self)(data=data, mom_ndim=self._mom_ndim, order=order)

    @docfiller.decorate
    def jackknife_and_reduce(
        self,
        *,
        axis: AxisReduce | MissingType = -1,
        parallel: bool | None = None,
        order: ArrayOrder = None,
        data_reduced: xr.DataArray | ArrayLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Jackknife resample and reduce

        Parameters
        ----------
        {axis_data_and_dim}
        {parallel}
        {order}

        Returns
        -------
        output : {klass}
            Instance of calling class with jackknife resampling along ``axis``.
        """
        self._raise_if_scalar()
        from .resample import jackknife_data

        data: ArrayT = jackknife_data(  # pyright: ignore[reportAssignmentType]
            self.to_values(),
            mom_ndim=self._mom_ndim,
            axis=axis,
            data_reduced=data_reduced,
            parallel=parallel,
            **kwargs,
        )

        return type(self)(data=data, mom_ndim=self._mom_ndim, order=order)

    @abstractmethod
    @docfiller.decorate
    def reduce(
        self,
        *,
        by: Groups | None = None,
        axis: AxisReduce = -1,
        keepdims: bool = False,
        move_axis_to_end: bool = False,
        order: ArrayOrder = None,
        parallel: bool | None = None,
    ) -> Self:
        """
        Create new object reduce along axis.

        Parameters
        ----------
        {by}
        {axis_data_and_dim}
        {keepdims}
        {move_axis_to_end}
        {order}
        {parallel}

        Returns
        -------
        output : {klass}
            If ``by`` is ``None``, reduce all samples along ``axis``,
            optionally keeping ``axis`` with size ``1`` if ``keepdims=True``.
            Otherwise, reduce for each unique value of ``by``. In this case,
            output will have shape ``(..., shape[axis-1], ngroup,
            shape[axis+1], ...)`` where ``ngroups = np.max(by) + 1``
            is the number of unique positive values in ``by``.

        See Also
        --------
        ~.reduction.reduce_data
        ~.reduction.reduce_data_grouped
        """

    # ** Manipulation -------------------------------------------------------------
    def pipe(
        self,
        func_or_method: Callable[..., Any] | str,
        *args: Any,
        _reorder: bool = True,
        _copy: bool | None = None,
        _order: ArrayOrder = None,
        _verify: bool = False,
        **kwargs: Any,
    ) -> Self:
        """
        Apply `func_or_method` to underlying data and wrap results in
        class:`cmomy.xCentralMoments` object.

        This is useful for calling any not implemented methods on :class:`numpy.ndarray` or
        :class:`xarray.DataArray` data.

        Note that a ``ValueError`` will be raised if last dimension(s) do not have the
        same shape as ``self.mom_shape``.

        Parameters
        ----------
        func_or_method : str or callable
            If callable, then apply ``values = func_or_method(self.to_values(),
            *args, **kwargs)``. If string is passed, then ``values =
            getattr(self.to_values(), func_or_method)(*args, **kwargs)``.
        *args
            Extra positional arguments to `func_or_method`
        _reorder : bool, default=True
            If True, reorder the data such that ``mom_dims`` are last.  Only
            applicable for ``DataArray`` like underlying data.
        _copy : bool, default=False
            If True, copy the resulting data.  Otherwise, try to use a view.
        _order : str, optional
            Array order to apply to output.
        _verify : bool, default=False
        **kwargs
            Extra keyword arguments to `func_or_method`

        Returns
        -------
        output : object
            New object after `func_or_method` is
            applies to `self.to_values()`


        Notes
        -----
        Use leading underscore for `_order`, `_copy`, etc to avoid possible name
        clashes with ``func_or_method``.


        """
        if isinstance(func_or_method, str):
            values = getattr(self.to_values(), func_or_method)(*args, **kwargs)
        else:
            values = func_or_method(self.to_values(), *args, **kwargs)

        if _reorder and hasattr(self, "mom_dims"):
            values = values.transpose(..., *self.mom_dims)  # pyright: ignore[reportUnknownMemberType,reportGeneralTypeIssues, reportAttributeAccessIssue]

        return self.new_like(
            data=values,
            copy=_copy,
            order=_order,
            verify=_verify,
        )

    @abstractmethod
    @docfiller.decorate
    def moveaxis(
        self,
        axis: int | tuple[int, ...],
        dest: int | tuple[int, ...],
    ) -> Self:
        """
        Generalized moveaxis

        Parameters
        ----------
        {axis}
        axis : int or sequence of int
            Original positions of axes to move.
        dest : int or sequence of int
            Destination positions for each original axes.

        Returns
        -------
        output : {klass}
            Object with moved axes.  This is a view of the original data.

        See Also
        --------
        .utils.moveaxis
        numpy.moveaxis
        """

    # ** Operators ----------------------------------------------------------------
    def _check_other(self, b: Self) -> None:
        """Check other object."""
        if type(self) is not type(b):
            raise TypeError
        if self.mom_ndim != b.mom_ndim or self.shape != b.shape:
            raise ValueError

    def __iadd__(self, b: Self) -> Self:  # noqa: PYI034
        """Self adder."""
        self._check_other(b)
        return self.push_data(b.to_values())

    def __add__(self, b: Self) -> Self:
        """Add objects to new object."""
        self._check_other(b)
        return self.copy().push_data(b.to_values())

    def __isub__(self, b: Self) -> Self:  # noqa: PYI034
        """Inplace subtraction."""
        # NOTE: consider implementint push_data_scale routine to make this cleaner
        self._check_other(b)
        if not np.all(self.weight() >= b.weight()):
            raise ValueError
        data = b.to_values().copy()
        data[moment_indexer("weight", self._mom_ndim)] *= -1
        return self.push_data(data)

    def __sub__(self, b: Self) -> Self:
        """Subtract objects."""
        self._check_other(b)
        if not np.all(self.weight() >= b.weight()):
            raise ValueError
        new = b.copy()
        new._data[moment_indexer("weight", self._mom_ndim)] *= -1
        return new.push_data(self.to_values())

    def __mul__(self, scale: float) -> Self:
        """New object with weight scaled by scale."""  # D401
        scale = float(scale)
        new = self.copy()
        new._data[moment_indexer("weight", self._mom_ndim)] *= scale
        return new

    def __imul__(self, scale: float) -> Self:  # noqa: PYI034
        """Inplace multiply."""
        scale = float(scale)
        self._data[moment_indexer("weight", self._mom_ndim)] *= scale
        return self

    # ** Constructors -------------------------------------------------------------
    @classmethod
    @abstractmethod
    @docfiller.decorate
    def zeros(
        cls,
        *,
        mom: Moments,
        val_shape: tuple[int, ...] | int | None = None,
        dtype: DTypeLike = None,
        order: ArrayOrderCF = None,
    ) -> Self:
        """
        Create a new base object.

        Parameters
        ----------
        {mom}
        {val_shape}
        {dtype}
        {order}

        Returns
        -------
        output : {klass}
            New instance with zero values.


        See Also
        --------
        numpy.zeros
        """

    @classmethod
    @abstractmethod
    @docfiller.decorate
    def from_vals(
        cls,
        x: ArrayT,
        *y: ArrayLike,
        mom: Moments,
        axis: AxisReduce = -1,
        weight: ArrayLike | None = None,
        keepdims: bool = False,
        order: ArrayOrder = None,
        dtype: DTypeLike = None,
        out: NDArrayAny | None = None,
        parallel: bool | None = None,
        # **kwargs: Any,
    ) -> Self:
        """
        Create from observations/values.

        Parameters
        ----------
        x : ndarray
            Values to reduce.
        *y : array-like
            Additional values (needed if ``len(mom)==2``).
        weight : scalar or array-like, optional
            Optional weight.  If scalar or array, attempt to
            broadcast to `x0.shape`
        {axis_and_dim}
        {mom}
        {keepdims}
        {order}
        {dtype}
        {out}
        {parallel}

        Returns
        -------
        output: {klass}

        See Also
        --------
        push_vals
        """

    @classmethod
    @abstractmethod
    @docfiller.decorate
    def from_resample_vals(
        cls,
        x: ArrayT,
        *y: ArrayLike,
        mom: Moments,
        freq: NDArrayInt,
        weight: ArrayLike | None = None,
        axis: AxisReduce = -1,
        move_axis_to_end: bool = True,
        order: ArrayOrder = None,
        parallel: bool | None = None,
        dtype: DTypeLike = None,
        out: NDArrayAny | None = None,
    ) -> Self:
        """
        Create from resample observations/values.

        This effectively resamples `x`.

        Parameters
        ----------
        x : array
            For moments, pass single array-like objects `x=x0`.
            For comoments, pass tuple of array-like objects `x=(x0, x1)`.
        *y : array-like
            Additional values (needed if ``len(mom) > 1``).
        {mom}
        {freq}
        {weight}
        {axis_and_dim}
        {move_axis_to_end}
        {full_output}
        {order}
        {parallel}
        {dtype}
        {out}

        Returns
        -------
        out : {klass}
            Instance of calling class


        See Also
        --------
        cmomy.resample.resample_vals
        cmomy.resample.randsamp_freq
        cmomy.resample.freq_to_indices
        cmomy.resample.indices_to_freq
        """

    @classmethod
    @docfiller.decorate
    def from_raw(
        cls,
        raw: ArrayT,
        *,
        mom_ndim: Mom_NDim,
    ) -> Self:
        """
        Create object from raw moment data.

        raw[..., i, j] = <x**i y**j>.
        raw[..., 0, 0] = `weight`


        Parameters
        ----------
        raw : {t_array}
            Raw moment array.
        {mom_ndim}

        Returns
        -------
        output : {klass}

        See Also
        --------
        to_raw
        rmom
        cmomy.convert.moments_type

        Notes
        -----
        Weights are taken from ``raw[...,0, 0]``.
        Using raw moments can result in numerical issues, especially for higher moments.  Use with care.
        """
        from . import convert

        return cls(
            data=convert.moments_type(raw, mom_ndim=mom_ndim, to="central"),  # pyright: ignore[reportArgumentType]
            mom_ndim=mom_ndim,
        )

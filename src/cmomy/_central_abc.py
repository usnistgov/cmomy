"""Base class for central moments calculations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, cast

import numpy as np
from module_utilities import cached

from ._lib.factory import factory_pusher
from ._utils import (
    normalize_axis_index,
    parallel_heuristic,
    prepare_data_for_reduction,
    prepare_values_for_push_val,
    prepare_values_for_reduction,
    validate_axis,
    validate_floating_dtype,
    validate_mom_ndim,
)
from .docstrings import docfiller

if TYPE_CHECKING:
    from typing import Any, Callable

    import xarray as xr
    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from ._lib.factory import Pusher
    from ._typing_compat import Self
    from .typing import (
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
    )

from .typing import ArrayT, FloatT


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
        from ._formatting import repr_html  # pyright: ignore[reportUnknownVariableType]

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
    #     return default if axis is None else axis

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
        # return type(self)(data=self.to_values(), copy=True)
        return self.new_like(
            data=self.to_values(),
            verify=False,
            copy=True,
            # copy_kws=copy_kws,
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
    def assign_weight(self, weight: ArrayLike, copy: bool = True) -> Self:
        """
        Create object with updated weights

        Parameters
        ----------
        {weight}
        copy : bool, default=True
            If ``True`` (default), copy the underlying moments data before update.
            Otherwise, update weights in in place.

        Returns
        -------
        output : object
            Same type as ``self``
        """
        from . import convert

        return type(self)(
            data=convert.assign_weight(
                data=self.values,
                weight=weight,
                mom_ndim=self._mom_ndim,
                copy=copy,
            ),
            mom_ndim=self._mom_ndim,
            fastpath=True,
        )

    # ** Access to underlying statistics ------------------------------------------
    @cached.prop
    def _weight_index(self) -> tuple[int | ellipsis, ...]:  # noqa: F821
        index: tuple[int, ...] = (0,) * len(self.mom)
        if self.val_ndim > 0:
            return (..., *index)  # pyright: ignore[reportUnknownVariableType]
        return index

    @cached.meth
    def _single_index(self, val: int) -> tuple[ellipsis | int | list[int], ...]:  # noqa: F821
        # index with things like data[..., 1,0] data[..., 0,1]
        # index = (...,[1,0],[0,1])
        dims = len(self.mom)

        index: list[int] | list[list[int]]
        if dims == 1:
            index = [val]

        else:
            # this is a bit more complicated
            index = [[0] * dims for _ in range(dims)]
            for i in range(dims):
                index[i][i] = val

        if self.val_ndim > 0:
            return (..., *index)  # pyright: ignore[reportUnknownVariableType]
        return tuple(index)

    def _wrap_like(self, x: NDArrayAny) -> ArrayT:  # noqa: PLR6301
        return x  # type: ignore[return-value]

    def weight(self) -> float | ArrayT:
        """Weight data."""
        return cast(
            "float | ArrayT",
            self.to_values()[self._weight_index],  # pyright: ignore[reportGeneralTypeIssues, reportIndexIssue]
        )

    def mean(self) -> float | ArrayT:
        """Mean (first moment)."""
        return cast(
            "float | ArrayT",
            self.to_values()[self._single_index(1)],  # pyright: ignore[reportGeneralTypeIssues, reportIndexIssue]
        )

    def var(self) -> float | ArrayT:
        """Variance (second central moment)."""
        return cast(
            "float | ArrayT",
            self.to_values()[self._single_index(2)],  # pyright: ignore[reportGeneralTypeIssues, reportIndexIssue]
        )

    def std(self) -> float | ArrayT:
        """Standard deviation."""  # D401
        return cast("float | ArrayT", np.sqrt(self.var()))

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
        out = self.data.copy()
        # zeroth central moment
        out[self._weight_index] = 1
        # first central moment
        out[self._single_index(1)] = 0
        return self._wrap_like(out)

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

        out = convert.moments_type(self._data, mom_ndim=self.mom_ndim, to="raw")
        if weight is not None:
            out[self._weight_index] = weight
        return self._wrap_like(out)

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
        order: ArrayOrder = None,
        parallel: bool | None = None,
    ) -> Self:
        data = np.asarray(data, order=order, dtype=self.dtype)
        self._pusher(parallel).data(data, self._data)
        return self

    def _push_datas_numpy(
        self,
        datas: ArrayLike,
        *,
        axis: AxisReduce | MissingType,
        parallel: bool | None = None,
        order: ArrayOrder = None,
    ) -> Self:
        datas = prepare_data_for_reduction(
            data=datas,
            axis=axis,
            mom_ndim=self.mom_ndim,
            dtype=self.dtype,
            order=order,
        )

        self._pusher(parallel).datas(
            datas,
            self._data,
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
        order: ArrayOrder = None,
        parallel: bool | None = None,
    ) -> Self:
        self._check_y(y, self.mom_ndim)

        weight = 1.0 if weight is None else weight

        x = np.asarray(x, dtype=self.dtype, order=order)
        x0, *x1, weight = prepare_values_for_push_val(
            x, *y, weight, dtype=self.dtype, order=order
        )

        self._pusher(parallel).val(
            x0,
            *x1,
            weight,
            self._data,
        )
        return self

    def _push_vals_numpy(
        self,
        x: ArrayLike,
        *y: ArrayLike,
        axis: AxisReduce | MissingType,
        weight: ArrayLike | None = None,
        order: ArrayOrder = None,
        parallel: bool | None = None,
    ) -> Self:
        self._check_y(y, self.mom_ndim)

        weight = 1.0 if weight is None else weight
        x = np.asarray(x, dtype=self.dtype, order=order)
        x0, *x1, weight = prepare_values_for_reduction(
            x,
            *y,
            weight,
            axis=axis,
            dtype=self.dtype,
            order=order,
            narrays=self._mom_ndim + 1,
        )

        self._pusher(parallel).vals(
            x0,
            *x1,
            weight,
            self._data,
        )
        return self

    @abstractmethod
    @docfiller.decorate
    def push_data(
        self,
        data: Any,
        *,
        order: ArrayOrder = None,
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
        order: ArrayOrder = None,
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
        order: ArrayOrder = None,
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
        order: ArrayOrder = None,
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
        axis: AxisReduce = -1,
        nrep: int | None = None,
        nsamp: int | None = None,
        indices: ArrayLike | None = None,
        freq: ArrayLike | None = None,
        check: bool = False,
        rng: np.random.Generator | None = None,
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
        )

    @abstractmethod
    @docfiller.decorate
    def resample_and_reduce(
        self,
        *,
        freq: NDArrayInt,
        axis: AxisReduce = -1,
        parallel: bool | None = None,
        order: ArrayOrder = None,
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
            ``(...,shape[axis-1], shape[axis+1], ..., nrep, mom0, ...)``,
            where ``nrep = freq.shape[0]``.


        See Also
        --------
        reduce
        ~cmomy.resample.randsamp_freq : random frequency sample
        ~cmomy.resample.freq_to_indices : convert frequency sample to index sample
        ~cmomy.resample.indices_to_freq : convert index sample to frequency sample
        ~cmomy.resample.resample_data : method to perform resampling
        """

    @abstractmethod
    @docfiller.decorate
    def reduce(
        self,
        *,
        axis: AxisReduce = -1,
        by: Groups | None = None,
        order: ArrayOrder = None,
        parallel: bool | None = None,
    ) -> Self:
        """
        Create new object reduce along axis.

        Parameters
        ----------
        {axis_data_and_dim}
        {by}
        {order}
        {parallel}

        Returns
        -------
        output : {klass}
            If ``by`` is ``None``, reduce all samples along ``axis``.
            Otherwise, reduce for each unique value of ``by``. In this case,
            output will have shape
            ``(..., shape[axis-1], shape[axis+1], ..., ngroup, mom0, ...)``
            where ``ngroups = np.max(by) + 1`` is the number of unique positive
            values in ``by``.

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
            # else:
            #     msg = "to specify `_reorder`, must have attribute `mom_dims`"
            #     raise AttributeError(msg)

        return self.new_like(
            data=values,
            copy=_copy,
            order=_order,
            verify=_verify,
        )

    # ** Operators ----------------------------------------------------------------
    def _check_other(self, b: Self) -> None:
        """Check other object."""
        if type(self) != type(b):
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
        # new = self.copy()
        # new.push_data(b.data)
        # return new
        return self.copy().push_data(b.to_values())

    def __isub__(self, b: Self) -> Self:  # noqa: PYI034
        """Inplace subtraction."""
        # NOTE: consider implementint push_data_scale routine to make this cleaner
        self._check_other(b)
        if not np.all(self.weight() >= b.weight()):
            raise ValueError
        data = b.to_values().copy()
        data[self._weight_index] *= -1
        return self.push_data(data)

    def __sub__(self, b: Self) -> Self:
        """Subtract objects."""
        self._check_other(b)
        if not np.all(self.weight() >= b.weight()):
            raise ValueError
        new = b.copy()
        new._data[self._weight_index] *= -1
        # new.push_data(self.data)
        # return new
        return new.push_data(self.to_values())

    def __mul__(self, scale: float) -> Self:
        """New object with weight scaled by scale."""  # D401
        scale = float(scale)
        new = self.copy()
        new._data[self._weight_index] *= scale
        return new

    def __imul__(self, scale: float) -> Self:  # noqa: PYI034
        """Inplace multiply."""
        scale = float(scale)
        self._data[self._weight_index] *= scale
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
        order: ArrayOrder = None,
        parallel: bool | None = None,
        dtype: DTypeLike = None,
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
        {order}
        {parallel}
        {dtype}

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
        axis: AxisReduce = -1,
        weight: ArrayLike | None = None,
        order: ArrayOrder = None,
        parallel: bool | None = None,
        dtype: DTypeLike = None,
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
        {full_output}
        {order}
        {parallel}
        {dtype}

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

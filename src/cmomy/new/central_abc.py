"""Base class for central moments calculations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, cast

import numpy as np
from module_utilities import cached

from ._lib.factory import factory_pusher
from .docstrings import docfiller
from .typing import ArrayOrderCF, NDArrayAny, T_Array
from .typing import T_FloatDType as T_Float
from .utils import (
    normalize_axis_index,
    parallel_heuristic,
    prepare_data_for_reduction,
    prepare_values_for_push_val,
    prepare_values_for_reduction,
    validate_mom_ndim,
)

if TYPE_CHECKING:
    from typing import Any, Callable, Mapping

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from ._lib.factory import Pusher
    from ._typing_compat import Self
    from .typing import (
        ArrayOrder,
        Mom_NDim,
        Moments,
        MomentsStrict,
        NDArrayInt,
    )


# * Main class ----------------------------------------------------------------
@docfiller.decorate  # noqa: PLR0904
class CentralMomentsABC(ABC, Generic[T_Array, T_Float]):
    r"""
    Wrapper to calculate central moments.

    Base data has the form

    .. math::

        {{\rm data}}[..., i, j] = \begin{{cases}}
             \text{{weight}} & i = j = 0 \\
            \langle x \rangle & i = 1, j = 0 \\
            \langle (x - \langle x \rangle^i) (y - \langle y \rangle^j)
            \rangle & i + j > 0
        \end{{cases}}

    Parameters
    ----------
    data : {t_array}
        Moment collection array
    {mom_ndim}
    fastpath : bool
        For internal use.
    data_flat : ndarray
        For internal use.

    """

    _cache: dict[str, Any]
    _data: NDArray[T_Float]
    _mom_ndim: Mom_NDim

    __slots__ = (
        "_cache",
        "_data",
        "_mom_ndim",
    )

    def __init__(
        self,
        data: T_Array,
        mom_ndim: Mom_NDim = 1,
    ) -> None:  # pragma: no cover
        self._mom_ndim = validate_mom_ndim(mom_ndim)
        self._cache = {}
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
        if self.dtype.type not in {np.float32, np.float64}:
            msg = f"{self.dtype=} not supported. Must be float32 or float64"
            raise ValueError(msg)

    # ** Basic access -------------------------------------------------------------
    @property
    def values(self) -> T_Array:
        """Access underlying central moments array."""
        return self.to_values()

    @values.setter
    def values(self, values: T_Array) -> None:
        self.set_values(values)

    @abstractmethod
    def set_values(self, values: T_Array) -> None:
        """Set values."""

    @abstractmethod
    def to_values(self) -> T_Array:
        """Access underlying values"""

    @property
    def data(self) -> NDArray[T_Float]:
        """
        Accessor to numpy array underlying data.

        By convention data has the following meaning for the moments indexes

        * `data[...,i=0,j=0]`, weights
        * `data[...,i=1,j=0]]`, if only one moment index is one and all others zero,
          then this is the average value of the variable with unit index.
        * all other cases, the central moments `<(x0-<x0>)**i0 * (x1 - <x1>)**i1 * ...>`

        """
        return self._data

    def to_numpy(self) -> NDArray[T_Float]:
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
    def dtype(self) -> np.dtype[Any]:
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
        return cast("MomentsStrict", self.data.shape[-self.mom_ndim :])

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
        return self.data.shape[: -self.mom_ndim]

    @property
    def val_ndim(self) -> int:
        """Number of value dimensions."""  # D401
        return len(self.val_shape)

    @property
    def mom_shape_var(self) -> tuple[int, ...]:
        """Shape of moment part of variance."""
        return tuple(x - 1 for x in self.mom)

    @property
    def shape_var(self) -> tuple[int, ...]:
        """Total variance shape."""
        return self.val_shape + self.mom_shape_var

    def __repr__(self) -> str:
        """Repr for class."""
        name = self.__class__.__name__
        s = f"<{name}(val_shape={self.val_shape}, mom={self.mom})>\n"
        return s + repr(self.to_values())

    def _repr_html_(self) -> str:  # noqa: PLW3201
        from ._formatting import repr_html  # pyright: ignore[reportUnknownVariableType]

        return repr_html(self)  # type: ignore[no-any-return,no-untyped-call]

    def __array__(self, dtype: DTypeLike | None = None) -> NDArray[T_Float]:  # noqa: PLW3201
        """Used by np.array(self)."""  # D401
        return np.asarray(self.data, dtype=dtype)

    # ** top level creation/copy/new ----------------------------------------------
    @abstractmethod
    @docfiller.decorate
    def new_like(
        self,
        data: T_Array | None = None,
        *,
        copy: bool = False,
        order: ArrayOrder | None = None,
        verify: bool = False,
        dtype: DTypeLike | None = None,
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

        See Also
        --------
        from_data
        """

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
        from_data
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
            # copy_kws=copy_kws,
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

    def _wrap_like(self, x: NDArrayAny) -> T_Array:  # noqa: PLR6301
        return x  # type: ignore[return-value]

    def weight(self) -> float | T_Array:
        """Weight data."""
        return cast(
            "float | T_Array",
            self.to_values()[self._weight_index],  # pyright: ignore[reportGeneralTypeIssues, reportIndexIssue]
        )

    def mean(self) -> float | T_Array:
        """Mean (first moment)."""
        return cast(
            "float | T_Array",
            self.to_values()[self._single_index(1)],  # pyright: ignore[reportGeneralTypeIssues, reportIndexIssue]
        )

    def var(self) -> float | T_Array:
        """Variance (second central moment)."""
        return cast(
            "float | T_Array",
            self.to_values()[self._single_index(2)],  # pyright: ignore[reportGeneralTypeIssues, reportIndexIssue]
        )

    def std(self) -> float | T_Array:
        """Standard deviation."""  # D401
        return cast("float | T_Array", np.sqrt(self.var()))

    def cmom(self) -> T_Array:
        r"""
        Central moments.

        Strict central moments of the form

        .. math::

            \text{cmom[..., n, m]} =
            \langle (x - \langle x \rangle^n) (y - \langle y \rangle^m) \rangle

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

    def to_raw(self, *, weights: float | NDArrayAny | None = None) -> T_Array:
        r"""
        Raw moments accumulation array.

        .. math::

            \text{raw[..., n, m]} = \begin{cases}
            \text{weight} & n = m = 0 \\
            \langle x^n y ^m \rangle & \text{otherwise}
            \end{cases}

        Returns
        -------
        raw : ndarray or DataArray

        See Also
        --------
        from_raw
        """
        from .convert import convert

        out = convert(self.data, mom_ndim=self.mom_ndim, to="raw")
        if weights is not None:
            out[self._weight_index] = weights
        return self._wrap_like(out)

    def rmom(self) -> T_Array:
        r"""
        Raw moments.

        .. math::
            \text{rmom[..., n, m]} = \langle x^n y^m \rangle

        Returns
        -------
        raw_moments : ndarray or DataArray

        See Also
        --------
        to_raw
        cmom
        """
        return self.to_raw(weights=1.0)

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

    # def _verify_arrays(
    #         *args: MultiArray[T_Array]
    # ) -> tuple[T_Array, ...]: ...

    # def _check_weights(
    #     self,
    #     *,
    #     w: MultiArray[T_Array] | None,
    #     target: NDArrayAny | T_Array,
    #     axis: int | None = None,
    #     dim: Hashable | None = None,
    # ) -> NDArrayAny:
    #     if w is None:
    #         w = 1.0
    #     return self._verify_value(
    #         x=w,
    #         target=target,
    #         shape_flat=self.val_shape_flat,
    #         axis=axis,
    #         dim=dim,
    #         broadcast=True,
    #         expand=True,
    #     )[0]

    # def _check_val(
    #     self,
    #     *,
    #     x: MultiArray[T_Array],
    #     target: str | NDArrayAny | T_Array,
    #     broadcast: bool = False,
    # ) -> tuple[NDArrayAny, NDArrayAny | T_Array]:
    #     return self._verify_value(
    #         x=x,
    #         target=target,
    #         shape_flat=self.val_shape_flat,
    #         broadcast=broadcast,
    #         expand=False,
    #     )

    # def _check_vals(
    #     self,
    #     *,
    #     x: MultiArrayVals[T_Array],
    #     target: str | NDArrayAny | T_Array,
    #     axis: int | None,
    #     broadcast: bool = False,
    #     dim: Hashable | None = None,
    # ) -> tuple[NDArrayAny, NDArrayAny | T_Array]:
    #     return self._verify_value(
    #         x=x,
    #         target=target,
    #         shape_flat=self.val_shape_flat,
    #         axis=axis,
    #         dim=dim,
    #         broadcast=broadcast,
    #         expand=broadcast,
    #     )

    # def _check_var(
    #     self, *, v: MultiArray[T_Array], broadcast: bool = False
    # ) -> NDArrayAny:
    #     return self._verify_value(
    #         x=v,
    #         target="var",
    #         shape_flat=self.shape_flat_var,
    #         broadcast=broadcast,
    #         expand=False,
    #     )[0]

    # def _check_vars(
    #     self,
    #     *,
    #     v: MultiArrayVals[T_Array],
    #     target: NDArrayAny | T_Array,
    #     axis: int | None,
    #     broadcast: bool = False,
    #     dim: Hashable | None = None,
    # ) -> NDArrayAny:
    #     # assert isinstance(target, np.ndarray)
    #     if not isinstance(target, np.ndarray):  # pragma: no cover
    #         msg = f"{type(target)=} must be numpy.ndarray."
    #         raise TypeError(msg)
    #     return self._verify_value(
    #         x=v,
    #         target="vars",
    #         shape_flat=self.shape_flat_var,
    #         axis=axis,
    #         dim=dim,
    #         broadcast=broadcast,
    #         expand=broadcast,
    #         other=target,
    #     )[0]

    # def _check_data(self, *, data: MultiArrayVals[T_Array]) -> NDArrayAny:
    #     return self._verify_value(
    #         x=data,
    #         target="data",
    #         shape_flat=self.shape_flat,
    #     )[0]

    # def _check_datas(
    #     self,
    #     *,
    #     datas: MultiArrayVals[T_Array],
    #     axis: int | None = None,
    #     dim: Hashable | None = None,
    # ) -> NDArrayAny:
    #     if axis is not None:
    #         axis = normalize_axis_index(axis, self.val_ndim + 1)

    #     return self._verify_value(
    #         x=datas,
    #         target="datas",
    #         shape_flat=self.shape_flat,
    #         axis=axis,
    #         dim=dim,
    #     )[0]

    # @staticmethod
    # def _class_array_to_numpy(*args: T_Array) -> tuple[NDArrayAny, ...]:
    #     """Convert T_Array to numpy array"""
    #     return cast("tuple[NDArrayAny, ...]", args)

    @staticmethod
    def _set_default_axis(axis: int | None, default: int = -1) -> int:
        return default if axis is None else axis

    # use interface here so can override in xCentral
    # @staticmethod
    # def _prepare_values_for_reduction(
    #     target: T_Array,
    #     *args: ArrayLike,
    #     mom_ndim: int,
    #     axis: int = -1,
    #     order: ArrayOrder = None,
    # ) -> tuple[NDArrayAny, ...]:
    #     return prepare_values_for_reduction(
    #         target,
    #         *args,
    #         mom_ndim=mom_ndim,
    #         axis=axis, order=order
    #     )

    # @staticmethod
    # def _prepare_data_for_reduction(
    #     target: T_Array,
    #     axis: int,
    #     mom_ndim: Mom_NDim,
    # )

    # Low level pushers
    def _push_data_numpy(
        self,
        data: NDArray[T_Float],
        *,
        order: ArrayOrder = None,
        parallel: bool | None = None,
    ) -> Self:
        if order:
            data = np.asarray(data, order=order)
        self._pusher(parallel).data(data, self._data)
        return self

    def _push_datas_numpy(
        self,
        datas: NDArray[T_Float],
        *,
        axis: int | None = None,
        parallel: bool | None = None,
        order: ArrayOrder = None,
    ) -> Self:
        datas = prepare_data_for_reduction(
            data=datas,
            axis=self._set_default_axis(axis),
            mom_ndim=self.mom_ndim,
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
        x0, *x1, weight = prepare_values_for_push_val(x, *y, weight, order=order)

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
        weight: ArrayLike | None = None,
        axis: int | None = None,
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
            axis=self._set_default_axis(axis),
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
        data: T_Array,
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
        datas: T_Array,
        *,
        axis: int | None = None,
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
        return self

    @abstractmethod
    @docfiller.decorate
    def push_vals(
        self,
        x: ArrayLike,
        *y: ArrayLike,
        weight: ArrayLike | None = None,
        axis: int | None = None,
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
    # @abstractmethod
    # @docfiller.decorate
    # def resample(
    #     self,
    #     indices: NDArrayInt,
    #     *,
    #     axis: int | None = None,
    #     # # *,
    #     # last: bool = True,
    #     # order: ArrayOrder = None,
    #     # # first: bool = True,
    #     # # verify: bool = False,
    #     # **kwargs: Any,
    # ) -> Self:
    #     """
    #     Create a new object sampled from index.

    #     Parameters
    #     ----------
    #     {indices}
    #     {axis_data_and_dim}

    #     Returns
    #     -------
    #     output : object
    #         Instance of calling class. The new object will have shape
    #         ``(..., shape[axis-1], nrep, shape[axis], ...)``.

    #     See Also
    #     --------
    #     from_data
    #     """
    #     # self._raise_if_scalar()
    #     # axis = self._set_default_axis(axis)

    #     # data = self.data
    #     # last_dim = self.val_ndim - 1
    #     # if last and axis != last_dim:
    #     #     data = np.moveaxis(data, axis, last_dim)
    #     #     axis = last_dim

    #     # out = np.take(data, indices, axis=axis)  # pyright: ignore[reportUnknownMemberType]

    #     # return type(self).from_data(
    #     #     data=out,
    #     #     mom_ndim=self.mom_ndim,
    #     #     copy=False,  # pyright: ignore[reportUnknownMemberType]
    #     #     order=order,
    #     #     **kwargs,
    #     # )

    @abstractmethod
    @docfiller.decorate
    def resample_and_reduce(
        self,
        *,
        freq: NDArrayInt,
        axis: int | None = None,
        parallel: bool = True,
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
        resample
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
        axis: int | None = None,
        by: ArrayLike | None = None,
        order: ArrayOrder = None,
        parallel: bool | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Create new object reduce along axis.

        Parameters
        ----------
        {axis_data_and_dim}
        {by}
        {order}
        {parallel}
        **kwargs
            Extra arguments.

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
        from_datas
        .indexed.reduce_by_group_idx
        """

    # ** Manipulation -------------------------------------------------------------
    def pipe(
        self,
        func_or_method: Callable[..., Any] | str,
        *args: Any,
        _reorder: bool = True,
        _copy: bool = False,
        _order: ArrayOrder = None,
        _check_mom: bool = True,
        _kws: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Apply `func_or_method` to underlying data and wrap results in
        class:`cmomy.xCentralMoments` object.

        This is useful for calling any not implemented methods on :class:`numpy.ndarray` or
        :class:`xarray.DataArray` data.

        Parameters
        ----------
        func_or_method : str or callable
            If callable, then apply ``values = func_or_method(self.to_values(),
            *args, **kwargs)``. If string is passed, then ``values =
            getattr(self.to_values(), func_or_method)(*args, **kwargs)``.
        _reorder : bool, default=True
            If True, reorder the data such that ``mom_dims`` are last.
        _copy : bool, default=False
            If True, copy the resulting data.  Otherwise, try to use a view.
            This is passed as ``copy=_copy`` to :meth:`from_data`.
        _order : str, optional
            Array order to apply to output.
        _check_mom: bool, default=True
            If True, check the resulting object has the same moment shape as the
            current object.
        _kws : Mapping, optional
            Extra arguments to :meth:`from_data`.
        *args
            Extra positional arguments to `func_or_method`
        **kwargs
            Extra keyword arguments to `func_or_method`

        Returns
        -------
        output : :class:`cmomy.xCentralMoments`
            New :class:`cmomy.xCentralMoments` object after `func_or_method` is
            applies to `self.to_values()`


        Notes
        -----
        Use leading underscore for `_order`, `_copy` to avoid name possible name
        clashes.


        See Also
        --------
        from_data
        """
        if isinstance(func_or_method, str):
            values = getattr(self.to_values(), func_or_method)(*args, **kwargs)
        else:
            values = func_or_method(self.to_values(), *args, **kwargs)

        if _reorder:
            if hasattr(self, "mom_dims"):
                values = values.transpose(..., *self.mom_dims)  # pyright: ignore[reportUnknownMemberType,reportGeneralTypeIssues, reportAttributeAccessIssue]
            else:
                msg = "to specify order, must have attribute `mom_dims`"
                raise AttributeError(msg)

        if _check_mom and values.shape[-self.mom_ndim :] != self.mom_shape:
            msg = f"{values.shape[-self.mom_ndim:]=} != {self.mom_shape=}"
            raise ValueError(msg)

        _kws = {} if _kws is None else dict(_kws)
        _kws.setdefault("copy", _copy)
        _kws.setdefault("order", _order)
        _kws.setdefault("mom_ndim", self.mom_ndim)

        return type(self).from_data(data=values, **_kws)

    @property
    def _is_vector(self) -> bool:
        return self.val_ndim > 0

    def _raise_if_scalar(self, message: str | None = None) -> None:
        if not self._is_vector:
            if message is None:  # pragma: no cover
                message = "Not implemented for scalar"
            raise ValueError(message)

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
        """New object with weights scaled by scale."""  # D401
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
    # *** Utils
    # @staticmethod
    # def _datas_axis_to_first(
    #     datas: NDArrayAny,
    #     axis: int,
    #     mom_ndim: Mom_NDim,
    # ) -> tuple[NDArrayAny, int]:
    #     """Move axis to first first position."""
    #     axis = normalize_axis_index(axis, datas.ndim - mom_ndim)
    #     if axis != 0:
    #         datas = np.moveaxis(datas, axis, 0)
    #     return datas, axis

    def _wrap_axis(
        self, axis: int | None, default: int = -1, ndim: int | None = None
    ) -> int:
        """Wrap axis to positive value and check."""
        if axis is None:
            axis = default
        if ndim is None:
            ndim = self.val_ndim

        return normalize_axis_index(axis, ndim)

    # *** Core
    @classmethod
    @abstractmethod
    @docfiller.decorate
    def zeros(
        cls,
        *,
        mom: Moments,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        order: ArrayOrderCF = None,
    ) -> Self:
        """
        Create a new base object.

        Parameters
        ----------
        {mom}
        {val_shape}
        {dtype}

        Returns
        -------
        output : {klass}
            New instance with zero values.


        See Also
        --------
        from_data
        numpy.zeros
        """

    @classmethod
    @abstractmethod
    @docfiller.decorate
    def from_data(
        cls,
        data: T_Array,
        *,
        mom_ndim: Mom_NDim,
        copy: bool = True,
        order: ArrayOrder = None,
        dtype: DTypeLike | None = None,
    ) -> Self:
        """
        Create new object from `data` array with additional checks.

        Parameters
        ----------
        data : array-like
            central moments accumulation array.
        {mom_ndim}
        {copy}
        {copy_kws}

        Returns
        -------
        out : {klass}
            Same type as calling class.
        """

    @classmethod
    @abstractmethod
    @docfiller.decorate
    def from_datas(
        cls,
        datas: T_Array,
        *,
        mom_ndim: Mom_NDim,
        axis: int | None = None,
        order: ArrayOrder = None,
        parallel: bool | None = None,
        # **kwargs: Any,
    ) -> Self:
        """
        Create object from multiple data arrays.

        Parameters
        ----------
        datas : ndarray
            Array of multiple Moment arrays.
            datas[..., i, ...] is the ith data array, where i is
            in position `axis`.
        {mom_ndim}
        {axis_data_and_dim}
        {order}
        {parallel}

        Returns
        -------
        output : {klass}

        See Also
        --------
        from_data
        """

    @classmethod
    @abstractmethod
    @docfiller.decorate
    def from_vals(
        cls,
        x: T_Array,
        *y: ArrayLike,
        mom: Moments,
        weight: ArrayLike | None = None,
        axis: int | None = None,
        order: ArrayOrder = None,
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
            Optional weights.  If scalar or array, attempt to
            broadcast to `x0.shape`
        {axis_and_dim}
        {mom}
        {order}
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
        x: T_Array,
        *y: ArrayLike,
        mom: Moments,
        freq: NDArrayInt,
        weight: ArrayLike | None = None,
        axis: int | None = None,
        order: ArrayOrder = None,
        parallel: bool | None = None,
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
        weight : scalar or array-like, optional
            Optional weights.  If scalar or array, attempt to
            broadcast to `x0.shape`
        {axis_and_dim}
        {full_output}
        {order}
        {parallel}
        {resample_kws}
        **kwargs
            Extra arguments to CentralMoments.from_data

        Returns
        -------
        out : {klass}
            Instance of calling class
        freq : ndarray, optional
            If `full_output` is True, also return `freq` array


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
        raw: T_Array,
        *,
        mom_ndim: Mom_NDim,
    ) -> Self:
        """
        Create object from raw moment data.

        raw[..., i, j] = <x**i y**j>.
        raw[..., 0, 0] = `weight`


        Parameters
        ----------
        raw : ndarray
            Raw moment array.
        {mom_ndim}

        Returns
        -------
        output : {klass}

        See Also
        --------
        to_raw
        rmom
        ~cmomy.convert.convert
        ~cmomy.convert.convert

        Notes
        -----
        Weights are taken from ``raw[...,0, 0]``.
        Using raw moments can result in numerical issues, especially for higher moments.  Use with care.
        """
        from .convert import convert

        return cls(
            data=convert(raw, mom_ndim=mom_ndim, to="central"), mom_ndim=mom_ndim
        )

    @classmethod
    @abstractmethod
    @docfiller.decorate
    def from_raws(
        cls,
        raws: T_Array,
        *,
        mom_ndim: Mom_NDim,
        axis: int | None = None,
        order: ArrayOrder = None,
        parallel: bool | None = None,
    ) -> Self:
        """
        Create object from multiple `raw` moment arrays.

        Parameters
        ----------
        raws : ndarray
            raws[...,i,...] is the ith sample of a `raw` array,
            Note that raw[...,i,j] = <x0**i, x1**j>
            where `i` is in position `axis`
        {mom_ndim}
        {axis_and_dim}
        {order}
        {parallel}
        **kwargs
            Extra arguments to :meth:`from_datas`

        Returns
        -------
        output : {klass}

        See Also
        --------
        from_raw
        from_datas
        ~cmomy.convert.to_central_moments  : convert raw to central moments
        ~cmomy.convert.to_central_comoments : convert raw to central comoments

        Notes
        -----
        Weights are taken from ``raw[...,0, 0]``.
        Using raw moments can result in numerical issues, especially for higher moments.  Use with care.
        """

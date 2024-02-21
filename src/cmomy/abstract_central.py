"""Base class for central moments calculations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from math import prod
from typing import TYPE_CHECKING, Generic, cast, overload

import numpy as np
from module_utilities import cached

from . import convert
from .docstrings import docfiller
from .typing import MyNDArray, T_Array
from .utils import normalize_axis_index

if TYPE_CHECKING:
    from typing import Any, Callable, Hashable, Literal, Mapping

    from numpy.typing import DTypeLike

    from ._lib.pushers import Pusher
    from ._typing_compat import Self
    from .typing import Mom_NDim, Moments, MomentsStrict, MultiArray, MultiArrayVals

# * Main
# TODO(wpk): Total rework is called for to handle typing correctly.


@docfiller.decorate  # noqa: PLR0904
class CentralMomentsABC(ABC, Generic[T_Array]):
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
    """

    __slots__ = (
        "_cache",
        "_data",
        "_data_flat",
        "_mom_ndim",
    )

    def __init__(
        self, data: T_Array, mom_ndim: Mom_NDim = 1
    ) -> None:  # pragma: no cover
        self._data = cast("MyNDArray", data)  # type: ignore[redundant-cast]
        self._data_flat = self._data

        if mom_ndim not in {1, 2}:
            msg = f"{mom_ndim=} must be 1 or 2."
            raise ValueError(msg)

        self._mom_ndim = mom_ndim
        self._cache: dict[str, Any] = {}

        self._validate_data()

    def _validate_data(self) -> None:  # pragma: no cover
        if np.shares_memory(self._data_flat, self._data):
            return

        raise ValueError

    @property
    def data(self) -> MyNDArray:
        """
        Accessor to numpy array underlying data.

        By convention data has the following meaning for the moments indexes

        * `data[...,i=0,j=0]`, weights
        * `data[...,i=1,j=0]]`, if only one moment index is one and all others zero,
          then this is the average value of the variable with unit index.
        * all other cases, the central moments `<(x0-<x0>)**i0 * (x1 - <x1>)**i1 * ...>`

        """
        return self._data

    @abstractmethod
    def to_values(self) -> T_Array:
        """Access underlying central moments array."""

    @property
    def values(self) -> T_Array:
        """Access underlying central moments array."""
        return self.to_values()

    def to_numpy(self) -> MyNDArray:
        """Access to numpy array underlying class."""
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        """self.data.shape."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """self.data.ndim."""
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype[Any]:
        """self.data.dtype."""
        # Not sure why I have to cast
        return self.data.dtype

    @property
    def mom_ndim(self) -> Mom_NDim:
        """
        Length of moments.

        if `mom_ndim` == 1, then single variable
        moments if `mom_ndim` == 2, then co-moments.
        """
        return cast("Mom_NDim", self._mom_ndim)

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
    def val_shape_flat(self) -> tuple[int, ...]:
        """Shape of values part flattened."""
        if self.val_shape == ():
            return ()
        return (prod(self.val_shape),)

    @property
    def shape_flat(self) -> tuple[int, ...]:
        """Shape of flattened data."""
        return self.val_shape_flat + self.mom_shape

    @property
    def mom_shape_var(self) -> tuple[int, ...]:
        """Shape of moment part of variance."""
        return tuple(x - 1 for x in self.mom)

    @property
    def shape_var(self) -> tuple[int, ...]:
        """Total variance shape."""
        return self.val_shape + self.mom_shape_var

    @property
    def shape_flat_var(self) -> tuple[int, ...]:
        """Shape of flat variance."""
        return self.val_shape_flat + self.mom_shape_var

    @cached.prop
    def _push(self) -> Pusher:
        from ._lib.pushers import factory_pushers

        vec = len(self.val_shape) > 0
        cov = self.mom_ndim == 2
        return factory_pushers(cov=cov, vec=vec)

    def __repr__(self) -> str:
        """Repr for class."""
        name = self.__class__.__name__
        s = f"<{name}(val_shape={self.val_shape}, mom={self.mom})>\n"
        return s + repr(self.to_values())

    def _repr_html_(self) -> str:  # noqa: PLW3201
        from ._formatting import repr_html  # pyright: ignore[reportUnknownVariableType]

        return repr_html(self)  # type: ignore[no-any-return,no-untyped-call]

    def __array__(self, dtype: DTypeLike | None = None) -> MyNDArray:  # noqa: PLW3201
        """Used by np.array(self)."""  # D401
        return np.asarray(self.data, dtype=dtype)

    ###########################################################################
    # ** top level creation/copy/new
    ###########################################################################
    @abstractmethod
    @docfiller.decorate
    def new_like(
        self,
        data: MyNDArray | T_Array | None = None,
        *,
        copy: bool = False,
        copy_kws: Mapping[str, Any] | None = None,
        verify: bool = False,
        strict: bool = False,
        **kwargs: Any,
    ) -> Self:
        """
        Create new object like self, with new data.

        Parameters
        ----------
        data : {t_array}
            data for new object
        {copy}
        {copy_kws}
        {verify}
        strict : bool, default=False
            If True, verify that `data` has correct shape
        {kwargs}
            arguments to classmethod :meth:`from_data`

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
    def copy(self, **copy_kws: Any) -> Self:
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
            copy_kws=copy_kws,
        )

    ###########################################################################
    # ** Access to underlying statistics
    ###########################################################################

    @cached.prop
    def _weight_index(self) -> tuple[int | ellipsis, ...]:  # noqa: F821
        index: tuple[int, ...] = (0,) * len(self.mom)
        if self.val_ndim > 0:
            return (..., *index)
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
            return (..., *index)
        return tuple(index)

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

    def _wrap_like(self, x: MyNDArray) -> T_Array:  # noqa: PLR6301
        return x  # type: ignore[return-value]

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

    def to_raw(self, *, weights: float | MyNDArray | None = None) -> T_Array:
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
        if self.mom_ndim == 1:
            out = convert.to_raw_moments(x=self.data)
        elif self.mom_ndim == 2:
            out = convert.to_raw_comoments(x=self.data)
        else:  # pragma: no cover
            msg = f"bad mom_ndim={self.mom_ndim}"
            raise ValueError(msg)

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

    ###########################################################################
    # ** pushing routines
    ###########################################################################
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

    @abstractmethod
    def _verify_value(  # pragma: no cover
        self,
        *,
        x: MultiArray[T_Array],
        target: str | MyNDArray | T_Array,
        shape_flat: tuple[int, ...],
        axis: int | None = None,
        dim: Hashable | None = None,  # included here for consistency
        broadcast: bool = False,
        expand: bool = False,
        other: MyNDArray | None = None,
    ) -> tuple[MyNDArray, MyNDArray | T_Array]:
        pass

    def _check_weight(
        self,
        *,
        w: MultiArray[T_Array] | None,
        target: MyNDArray | T_Array,
    ) -> MyNDArray:
        if w is None:
            w = 1.0
        return self._verify_value(
            x=w,
            target=target,
            shape_flat=self.val_shape_flat,
            broadcast=True,
            expand=True,
        )[0]

    def _check_weights(
        self,
        *,
        w: MultiArray[T_Array] | None,
        target: MyNDArray | T_Array,
        axis: int | None = None,
        dim: Hashable | None = None,
    ) -> MyNDArray:
        if w is None:
            w = 1.0
        return self._verify_value(
            x=w,
            target=target,
            shape_flat=self.val_shape_flat,
            axis=axis,
            dim=dim,
            broadcast=True,
            expand=True,
        )[0]

    def _check_val(
        self,
        *,
        x: MultiArray[T_Array],
        target: str | MyNDArray | T_Array,
        broadcast: bool = False,
    ) -> tuple[MyNDArray, MyNDArray | T_Array]:
        return self._verify_value(
            x=x,
            target=target,
            shape_flat=self.val_shape_flat,
            broadcast=broadcast,
            expand=False,
        )

    def _check_vals(
        self,
        *,
        x: MultiArrayVals[T_Array],
        target: str | MyNDArray | T_Array,
        axis: int | None,
        broadcast: bool = False,
        dim: Hashable | None = None,
    ) -> tuple[MyNDArray, MyNDArray | T_Array]:
        return self._verify_value(
            x=x,
            target=target,
            shape_flat=self.val_shape_flat,
            axis=axis,
            dim=dim,
            broadcast=broadcast,
            expand=broadcast,
        )

    def _check_var(
        self, *, v: MultiArray[T_Array], broadcast: bool = False
    ) -> MyNDArray:
        return self._verify_value(
            x=v,
            target="var",
            shape_flat=self.shape_flat_var,
            broadcast=broadcast,
            expand=False,
        )[0]

    def _check_vars(
        self,
        *,
        v: MultiArrayVals[T_Array],
        target: MyNDArray | T_Array,
        axis: int | None,
        broadcast: bool = False,
        dim: Hashable | None = None,
    ) -> MyNDArray:
        # assert isinstance(target, np.ndarray)
        if not isinstance(target, np.ndarray):  # pragma: no cover
            msg = f"{type(target)=} must be numpy.ndarray."
            raise TypeError(msg)
        return self._verify_value(
            x=v,
            target="vars",
            shape_flat=self.shape_flat_var,
            axis=axis,
            dim=dim,
            broadcast=broadcast,
            expand=broadcast,
            other=target,
        )[0]

    def _check_data(self, *, data: MultiArrayVals[T_Array]) -> MyNDArray:
        return self._verify_value(
            x=data,
            target="data",
            shape_flat=self.shape_flat,
        )[0]

    def _check_datas(
        self,
        *,
        datas: MultiArrayVals[T_Array],
        axis: int | None = None,
        dim: Hashable | None = None,
    ) -> MyNDArray:
        if axis is not None:
            axis = normalize_axis_index(axis, self.val_ndim + 1)

        return self._verify_value(
            x=datas,
            target="datas",
            shape_flat=self.shape_flat,
            axis=axis,
            dim=dim,
        )[0]

    @docfiller.decorate
    def push_data(self, data: MultiArrayVals[T_Array]) -> Self:
        """
        Push data object to moments.

        Parameters
        ----------
        data : array-like
            Accumulation array of same form as ``self.data``

        Returns
        -------
        output : {klass}
            Same object with pushed data.
        """
        data = self._check_data(data=data)
        self._push.data(self._data_flat, data)
        return self

    @docfiller.decorate
    def push_datas(
        self,
        datas: MultiArrayVals[T_Array],
        axis: int | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Push and reduce multiple average central moments.

        Parameters
        ----------
        datas : array-like, {t_array}
            Collection of accumulation arrays to push onto ``self``.
            This should have shape like `(nrec,) + self.shape`
            if `axis=0`, where `nrec` is the number of data objects to sum.
        {axis_and_dim}
        {kwargs}

        Returns
        -------
        output : {klass}
            Same object with pushed data.
        """
        datas = self._check_datas(datas=datas, axis=axis, **kwargs)
        self._push.datas(self._data_flat, datas)
        return self

    @docfiller.decorate
    def push_val(
        self,
        x: MultiArray[T_Array] | tuple[MultiArray[T_Array], MultiArray[T_Array]],
        w: MultiArray[T_Array] | None = None,
        broadcast: bool = False,
    ) -> Self:
        """
        Push single sample to central moments.

        Parameters
        ----------
        x : array-like or tuple of array-like
            Pass single array `x=x0`if accumulating moments.
            Pass tuple of arrays `(x0, x1)` if accumulating comoments.
            `x0.shape == self.val_shape`
        w : int, float, array-like, optional
            Weight of each sample.  If scalar, broadcast `w.shape` to `x0.shape`.
        broadcast : bool, optional
            If True, and `x1` present, attempt to broadcast `x1.shape` to `x0.shape`

        Returns
        -------
        output : {klass}
            Same object with pushed data.

        Notes
        -----
        Array `x0` should have same shape as `self.val_shape`.
        """
        if self.mom_ndim == 1:
            ys = ()
        elif not isinstance(x, tuple):
            raise TypeError
        elif len(x) != self.mom_ndim:
            raise ValueError
        else:
            x, *ys = x

        xr, target = self._check_val(x=x, target="val")  # type: ignore[arg-type]
        yr = tuple(  # type: ignore[var-annotated]
            self._check_val(
                x=y,
                target=target,
                broadcast=broadcast,
            )[0]
            for y in ys
        )
        wr = self._check_weight(w=w, target=target)
        self._push.val(self._data_flat, *((wr, xr, *yr)))  # type: ignore[has-type]
        return self

    @docfiller.decorate
    def push_vals(
        self,
        x: MultiArrayVals[T_Array]
        | tuple[MultiArrayVals[T_Array], MultiArrayVals[T_Array]],
        w: MultiArray[T_Array] | None = None,
        axis: int | None = None,
        broadcast: bool = False,
        **kwargs: Any,
    ) -> Self:
        """
        Push multiple samples to central moments.

        Parameters
        ----------
        x : array-like or tuple of array-like
            Pass single array `x=x0`if accumulating moments.
            Pass tuple of arrays `(x0, x1)` if accumulating comoments.
            `x0.shape` less axis should be same as `self.val_shape`.
        w : int, float, array-like, optional
            Weight of each sample.  If scalar, broadcast to `x0.shape`
        {broadcast}
        {axis_and_dim}
        {kwargs}

        Returns
        -------
        output : {klass}
            Same object with pushed data.
        """
        if self.mom_ndim == 1:
            ys = ()
        elif not isinstance(x, tuple):
            raise TypeError
        elif len(x) != self.mom_ndim:
            raise ValueError
        else:
            x, *ys = x

        # fmt: off
        xr, target = self._check_vals(x=x, axis=axis, target="vals", **kwargs)  # type: ignore[arg-type]
        yr = tuple(  # type: ignore[var-annotated]
            self._check_vals(x=y, target=target, axis=axis, broadcast=broadcast, **kwargs)[0]
            for y in ys
        )
        # fmt: on
        wr = self._check_weights(w=w, target=target, axis=axis, **kwargs)

        self._push.vals(self._data_flat, *((wr, xr, *yr)))  # type: ignore[has-type]
        return self

    ###########################################################################
    # ** Manipulation
    ###########################################################################
    def pipe(
        self,
        func_or_method: Callable[..., Any] | str,
        *args: Any,
        _order: bool = True,
        _copy: bool = False,
        _verify: bool = False,
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
        _order : bool, default=True
            If True, reorder the data such that ``mom_dims`` are last.
        _copy : bool, default=False
            If True, copy the resulting data.  Otherwise, try to use a view.
            This is passed as ``copy=_copy`` to :meth:`from_data`.
        _verify: bool, default=False
            If True, ensure underlying data is contiguous. Passed as
            ``verify=_verify`` to :meth:`from_data`
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

        if _order:
            if hasattr(self, "mom_dims"):
                values = values.transpose(..., *self.mom_dims)  # pyright: ignore[reportUnknownMemberType,reportGeneralTypeIssues, reportAttributeAccessIssue]
            else:
                msg = "to specify order, must have attribute `mom_dims`"
                raise AttributeError(msg)

        _kws = {} if _kws is None else dict(_kws)
        _kws.setdefault("copy", _copy)
        _kws.setdefault("verify", _verify)
        _kws.setdefault("mom_ndim", self.mom_ndim)
        if _check_mom:
            _kws["mom"] = self.mom

        return type(self).from_data(data=values, **_kws)

    @property
    def _is_vector(self) -> bool:
        return self.val_ndim > 0

    def _raise_if_scalar(self, message: str | None = None) -> None:
        if not self._is_vector:
            if message is None:  # pragma: no cover
                message = "not implemented for scalar"
            raise ValueError(message)

    # * Universal reducers

    @docfiller.decorate
    def resample(
        self,
        indices: MyNDArray,
        axis: int | None = 0,
        *,
        first: bool = True,
        verify: bool = False,
        **kwargs: Any,
    ) -> Self:
        """
        Create a new object sampled from index.

        Parameters
        ----------
        {indices}
        {axis_and_dim}
        first : bool, default=True
            if True, and axis != 0, the move the axis to first position.
            This makes results similar to resample and reduce
            If `first` False, then resampled array can have odd shape
        {kwargs}

        Returns
        -------
        output : object
            Instance of calling class
            The new object will have shape
            `(nrep, ndat, ...) + self.shape[:axis] + self.shape[axis+1:]`


        See Also
        --------
        from_data
        """
        axis = axis or 0
        self._raise_if_scalar()
        axis = self._wrap_axis(axis)

        data = self.data
        if first and axis != 0:
            data = np.moveaxis(data, axis, 0)
            axis = 0

        out = np.take(data, indices, axis=axis)  # pyright: ignore[reportUnknownMemberType]

        return type(self).from_data(
            data=out,
            mom_ndim=self.mom_ndim,
            mom=self.mom,
            copy=False,  # pyright: ignore[reportUnknownMemberType]
            verify=verify,
            **kwargs,
        )

    ###########################################################################
    # ** Operators
    ###########################################################################
    def _check_other(self, b: Self) -> None:
        """Check other object."""
        if type(self) != type(b):
            raise TypeError
        if self.mom_ndim != b.mom_ndim or self.shape != b.shape:
            raise ValueError

    def __iadd__(self, b: Self) -> Self:  # noqa: PYI034
        """Self adder."""
        self._check_other(b)
        return self.push_data(b.data)

    def __add__(self, b: Self) -> Self:
        """Add objects to new object."""
        self._check_other(b)
        # new = self.copy()
        # new.push_data(b.data)
        # return new
        return self.copy().push_data(b.data)

    def __isub__(self, b: Self) -> Self:  # noqa: PYI034
        """Inplace subtraction."""
        # NOTE: consider implementint push_data_scale routine to make this cleaner
        self._check_other(b)
        if not np.all(self.weight() >= b.weight()):
            raise ValueError
        data = b.data.copy()
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
        return new.push_data(self.data)

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

    ###########################################################################
    # ** Constructors
    ###########################################################################
    # *** Utils
    @staticmethod
    def _datas_axis_to_first(
        datas: MyNDArray,
        axis: int,
        mom_ndim: Mom_NDim,
    ) -> tuple[MyNDArray, int]:
        """Move axis to first first position."""
        axis = normalize_axis_index(axis, datas.ndim - mom_ndim)
        if axis != 0:
            datas = np.moveaxis(datas, axis, 0)
        return datas, axis

    def _wrap_axis(
        self, axis: int | None, default: int = 0, ndim: int | None = None
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
        mom: Moments | None = None,
        mom_ndim: Mom_NDim | None = None,
        val_shape: tuple[int, ...] | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        zeros_kws: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Create a new base object.

        Parameters
        ----------
        {mom}
        {mom_ndim}
        {val_shape}
        {shape}
        {dtype}
        {zeros_kws}
        **kwargs
            extra arguments to :meth:`from_data`

        Returns
        -------
        output : {klass}
            New instance with zero values.

        Notes
        -----
        The resulting total shape of data is shape + (mom + 1).
        Must specify either `mom` or `mom_ndim`

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
        data: Any,
        *,
        mom: Moments | None = None,
        mom_ndim: Mom_NDim | None = None,
        val_shape: tuple[int, ...] | None = None,
        copy: bool = True,
        copy_kws: Mapping[str, Any] | None = None,
        verify: bool = False,
        dtype: DTypeLike | None = None,
    ) -> Self:
        """
        Create new object from `data` array with additional checks.

        Parameters
        ----------
        data : array-like
            central moments accumulation array.
        {mom_ndim}
        {mom}
        {val_shape}
        {copy}
        {copy_kws}
        {verify}
        {dtype}

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
        datas: Any,
        *,
        mom: Moments | None = None,
        mom_ndim: Mom_NDim | None = None,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        verify: bool = False,
        axis: int | None = 0,
        **kwargs: Any,
    ) -> Self:
        """
        Create object from multiple data arrays.

        Parameters
        ----------
        datas : ndarray
            Array of multiple Moment arrays.
            datas[..., i, ...] is the ith data array, where i is
            in position `axis`.
        {mom}
        {mom_ndim}
        {val_shape}
        {dtype}
        {verify}
        {axis_and_dim}
        **kwargs
            Extra arguments.

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
        x: Any,
        mom: Moments,
        *,
        w: Any = None,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        broadcast: bool = False,
        axis: int | None = 0,
        **kwargs: Any,
    ) -> Self:
        """
        Create from observations/values.

        Parameters
        ----------
        x : array-like or tuple of array-like
            For moments, pass single array-like objects `x=x0`.
            For comoments, pass tuple of array-like objects `x=(x0, x1)`.
        w : scalar or array-like, optional
            Optional weights.  If scalar or array, attempt to
            broadcast to `x0.shape`
        {axis_and_dim}
        {mom}
        {val_shape}
        {broadcast}
        {kwargs}

        Returns
        -------
        output: {klass}

        See Also
        --------
        push_vals
        """

    @overload
    @classmethod
    @abstractmethod
    def from_resample_vals(
        cls,
        x: MyNDArray | tuple[MyNDArray, MyNDArray] | T_Array | tuple[T_Array, T_Array],
        mom: Moments,
        *,
        full_output: Literal[False] = ...,
        **kwargs: Any,
    ) -> Self:
        ...

    @overload
    @classmethod
    @abstractmethod
    def from_resample_vals(
        cls,
        x: MyNDArray | tuple[MyNDArray, MyNDArray] | T_Array | tuple[T_Array, T_Array],
        mom: Moments,
        *,
        full_output: Literal[True],
        **kwargs: Any,
    ) -> tuple[Self, MyNDArray]:
        ...

    @overload
    @classmethod
    @abstractmethod
    def from_resample_vals(
        cls,
        x: MyNDArray | tuple[MyNDArray, MyNDArray] | T_Array | tuple[T_Array, T_Array],
        mom: Moments,
        *,
        full_output: bool,
        **kwargs: Any,
    ) -> Self | tuple[Self, MyNDArray]:
        ...

    @classmethod
    @abstractmethod
    @docfiller.decorate
    def from_resample_vals(
        cls,
        x: MyNDArray | tuple[MyNDArray, MyNDArray] | T_Array | tuple[T_Array, T_Array],
        mom: Moments,
        *,
        full_output: bool = False,
        **kwargs: Any,
    ) -> Self | tuple[Self, MyNDArray]:
        """
        Create from resample observations/values.

        This effectively resamples `x`.

        Parameters
        ----------
        x : array-like or tuple of array-like
            For moments, pass single array-like objects `x=x0`.
            For comoments, pass tuple of array-like objects `x=(x0, x1)`.
        {mom}
        {full_output}
        {nrep}
        {freq}
        {indices}
        w : scalar or array-like, optional
            Optional weights.  If scalar or array, attempt to
            broadcast to `x0.shape`
        {axis_and_dim}
        {dtype}
        {broadcast}
        parallel : bool, default=True
            If True, perform resampling in parallel.
        {resample_kws}
        {rng}
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
    @abstractmethod
    @docfiller.decorate
    def from_raw(
        cls,
        raw: Any,
        *,
        mom: Moments | None = None,
        mom_ndim: Mom_NDim | None = None,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        convert_kws: Mapping[str, Any] | None = None,
        **kwargs: Any,
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
        {mom}
        {val_shape}
        {dtype}
        {convert_kws}
        **kwargs
            Extra arguments to :meth:`from_data`

        Returns
        -------
        output : {klass}

        See Also
        --------
        to_raw
        rmom
        ~cmomy.convert.to_central_moments
        ~cmomy.convert.to_central_comoments

        Notes
        -----
        Weights are taken from ``raw[...,0, 0]``.
        Using raw moments can result in numerical issues, especially for higher moments.  Use with care.

        """

    @classmethod
    @abstractmethod
    @docfiller.decorate
    def from_raws(
        cls,
        raws: Any,
        *,
        mom: Moments | None = None,
        mom_ndim: Mom_NDim | None = None,
        val_shape: tuple[int, ...] | None = None,
        axis: int | None = 0,
        dtype: DTypeLike | None = None,
        convert_kws: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Create object from multiple `raw` moment arrays.

        Parameters
        ----------
        raws : ndarray
            raws[...,i,...] is the ith sample of a `raw` array,
            Note that raw[...,i,j] = <x0**i, x1**j>
            where `i` is in position `axis`
        {axis_and_dim}
        {mom_ndim}
        {mom}
        {val_shape}
        {dtype}
        {convert_kws}
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

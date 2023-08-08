"""Base class for central moments calculations."""


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Hashable,
    Mapping,
    Tuple,
    Union,
    cast,
)

from module_utilities import cached

from . import convert
from ._formatting import repr_html  # pyright: ignore
from ._lazy_imports import np
from ._typing import Mom_NDim, Moments, MyNDArray, T_Array
from .docstrings import docfiller
from .pushers import Pusher, factory_pushers

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike
    from typing_extensions import Self


# * TODO Main
# TODO: Total rework is called for to handle typing correctly.


@docfiller.decorate
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
        "_mom_ndim",
        "_cache",
        "_data",
        "_data_flat",
    )

    # Override __new__ to make signature correct
    # Better to do this in subclasses.
    # otherwise, signature for data will be 'T_Array``
    # def __new__(cls, data: T_Array, mom_ndim: Literal[1, 2] = 1):  # noqa: D102
    #     return super().__new__(cls)  # , data=data, mom_ndim=mom_ndim)

    def __init__(self, data: T_Array, mom_ndim: Mom_NDim = 1) -> None:
        self._data = cast(MyNDArray, data)  # type: ignore
        self._data_flat: MyNDArray = self._data

        assert mom_ndim in [1, 2]

        self._mom_ndim = mom_ndim
        self._cache: dict[str, Any] = {}

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

    @property
    @abstractmethod
    def values(self) -> T_Array:
        """Access underlying central moments array."""

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
        return cast(Mom_NDim, self._mom_ndim)

    @property
    def mom_shape(self) -> tuple[int] | tuple[int, int]:
        """Shape of moments part."""
        return cast(
            Union[Tuple[int], Tuple[int, int]], self.data.shape[-self.mom_ndim :]
        )

    @property
    def mom(self) -> tuple[int] | tuple[int, int]:
        """Number of moments."""  # noqa D401
        return tuple(x - 1 for x in self.mom_shape)  # type: ignore

    @property
    def val_shape(self) -> tuple[int, ...]:
        """
        Shape of values dimensions.

        That is shape less moments dimensions.
        """
        return self.data.shape[: -self.mom_ndim]

    @property
    def val_ndim(self) -> int:
        """Number of value dimensions."""  # noqa D401
        return len(self.val_shape)

    @property
    def val_shape_flat(self) -> tuple[int, ...]:
        """Shape of values part flattened."""
        if self.val_shape == ():
            return ()
        else:
            return (int(np.prod(self.val_shape)),)  # pyright: ignore

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
        vec = len(self.val_shape) > 0
        cov = self.mom_ndim == 2
        return factory_pushers(cov=cov, vec=vec)

    def __repr__(self) -> str:
        """Repr for class."""
        name = self.__class__.__name__
        s = f"<{name}(val_shape={self.val_shape}, mom={self.mom})>\n"
        return s + repr(self.values)

    def _repr_html_(self) -> str:
        return repr_html(self)  # type: ignore

    def __array__(self, dtype: DTypeLike | None = None) -> MyNDArray:
        """Used by np.array(self)."""  # noqa D401
        return np.asarray(self.data, dtype=dtype)

    ###########################################################################
    # ** top level creation/copy/new
    ###########################################################################
    @abstractmethod
    @docfiller.decorate
    def new_like(
        self,
        *,
        data: T_Array | None = None,
        copy: bool = False,
        copy_kws: Mapping[str, Any] | None = None,
        verify: bool = True,
        check_shape: bool = False,
        strict: bool = False,
        **kws: Any,
    ) -> Self:
        """
        Create new object like self, with new data.

        Parameters
        ----------
        data : array-like,optional
            data for new object
        {copy}
        {copy_kws}
        {verify}
        {check_shape}
        strict : bool, default=False
            If True, verify that `data` has correct shape
        **kws
            arguments to classmethod :meth:`from_data`

        Returns
        -------
        output : {klass}

        See Also
        --------
        from_data
        """

    def zeros_like(self) -> Self:
        """
        Create new empty object like self.

        Returns
        -------
        output : object
            Same type as calling class.
            Object with same attributes as caller, but with data set to zero.

        See Also
        --------
        new_like
        from_data
        """
        return self.new_like()

    def copy(self, **copy_kws: Any) -> Self:
        """
        Create a new object with copy of data.

        Parameters
        ----------
        **copy_kws
            passed to parameter ``copy_kws`` in method :meth:`new_like`

        Returns
        -------
        output : object
            Same type as calling class.
            Object with same attributes as caller, but with new underlying data.


        See Also
        --------
        new_like
        zeros_like
        """
        return self.new_like(
            data=self.values,
            verify=False,
            check_shape=False,
            copy=True,
            copy_kws=copy_kws,
        )

    ###########################################################################
    # ** Access to underlying statistics
    ###########################################################################

    @cached.prop
    def _weight_index(self) -> "tuple[int | ellipsis, ...]":  # noqa: UP037, F821
        index = (0,) * len(self.mom)
        if self.val_ndim > 0:
            return (...,) + index
        else:
            return index

    @cached.meth
    def _single_index(
        self, val: int
    ) -> "tuple[ellipsis | int | list[int], ...]":  # noqa: UP037, F821
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
            return tuple([...] + index)  # type: ignore
        else:
            return tuple(index)

    def weight(self) -> float | T_Array:
        """Weight data."""
        return cast(Union[float, T_Array], self.values[self._weight_index])

    def mean(self) -> float | T_Array:
        """Mean (first moment)."""
        return cast(Union[float, T_Array], self.values[self._single_index(1)])

    def var(self) -> float | T_Array:
        """Variance (second central moment)."""
        return cast(Union[float, T_Array], self.values[self._single_index(2)])

    def std(self) -> float | T_Array:
        """Standard deviation."""  # noqa D401
        return cast(Union[float, T_Array], np.sqrt(self.var()))

    def _wrap_like(self, x: MyNDArray, *args: Any, **kwargs: Any) -> T_Array:
        return cast(T_Array, x)  # type: ignore

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

    def to_raw(self, weights: float | MyNDArray | None = None) -> T_Array:
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
        else:
            raise ValueError(f"bad mom_ndim={self.mom_ndim}")

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
    def _verify_value(
        self,
        *,
        x: float | ArrayLike | T_Array,
        target: str | MyNDArray | T_Array,
        shape_flat: tuple[int, ...],
        axis: int | None = None,
        dim: Hashable | None = None,  # included here for consistency
        broadcast: bool = False,
        expand: bool = False,
        other: MyNDArray | None = None,
    ) -> tuple[MyNDArray, MyNDArray]:
        pass

    def _check_weight(
        self,
        *,
        w: float | ArrayLike | T_Array | None,
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

    def _check_weights(  # type: ignore
        self,
        *,
        w: float | ArrayLike | T_Array | None,
        target: MyNDArray,
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

    def _check_val(  # type: ignore
        self,
        *,
        x: float | ArrayLike | T_Array,
        target: str | MyNDArray,
        broadcast: bool = False,
    ) -> tuple[MyNDArray, MyNDArray]:
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
        x: float | ArrayLike | T_Array,
        target: str | MyNDArray,
        axis: int | None,
        broadcast: bool = False,
        dim: Hashable | None = None,
    ) -> tuple[MyNDArray, MyNDArray]:
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
        self, *, v: float | ArrayLike | T_Array, broadcast: bool = False
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
        v: float | ArrayLike | T_Array,
        target: MyNDArray,
        axis: int | None,
        broadcast: bool = False,
        dim: Hashable | None = None,
    ) -> MyNDArray:
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

    def _check_data(self, *, data: ArrayLike | T_Array) -> MyNDArray:
        return self._verify_value(
            x=data,
            target="data",
            shape_flat=self.shape_flat,
        )[0]

    def _check_datas(
        self,
        *,
        datas: ArrayLike | T_Array,
        axis: int | None,
        dim: Hashable | None = None,
    ) -> MyNDArray:
        return self._verify_value(
            x=datas,
            target="datas",
            shape_flat=self.shape_flat,
            axis=axis,
            dim=dim,
        )[0]

    @docfiller.decorate
    def push_data(self, data: ArrayLike | T_Array) -> Self:
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
        datas: ArrayLike | T_Array,
        axis: int | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Push and reduce multiple average central moments.

        Parameters
        ----------
        datas : array-like
            Collection of accumulation arrays to push onto ``self``.
            This should have shape like `(nrec,) + self.shape`
            if `axis=0`, where `nrec` is the number of data objects to sum.
        {axis}

        Returns
        -------
        output : {klass}
            Same object with pushed data.
        """

        datas = self._check_datas(datas=datas, axis=axis, **kwargs)  # type: ignore
        self._push.datas(self._data_flat, datas)
        return self

    @docfiller.decorate
    def push_val(
        self,
        x: float | ArrayLike | T_Array,
        w: float | ArrayLike | T_Array | None = None,
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
        else:
            assert isinstance(x, tuple) and len(x) == self.mom_ndim
            x, *ys = x

        xr, target = self._check_val(x=x, target="val")
        yr = tuple(self._check_val(x=y, target=target, broadcast=broadcast)[0] for y in ys)  # type: ignore
        wr = self._check_weight(w=w, target=target)
        self._push.val(self._data_flat, *((wr, xr) + yr))
        return self

    @docfiller.decorate
    def push_vals(
        self,
        x: ArrayLike | T_Array,
        w: float | ArrayLike | T_Array | None,
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
        {axis}

        Returns
        -------
        output : {klass}
            Same object with pushed data.
        """
        if self.mom_ndim == 1:
            ys = ()
        else:
            assert len(x) == self.mom_ndim
            x, *ys = x

        # fmt: off
        xr, target = self._check_vals(x=x, axis=axis, target="vals", **kwargs)
        yr = tuple( # type: ignore
            self._check_vals(x=y, target=target, axis=axis, broadcast=broadcast, **kwargs)[0]
            for y in ys
        )
        # fmt: on
        wr = self._check_weights(w=w, target=target, axis=axis, **kwargs)

        self._push.vals(self._data_flat, *((wr, xr) + yr))
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
            If callable, then apply ``values = func_or_method(self.values,
            *args, **kwargs)``. If string is passed, then ``values =
            getattr(self.values, func_or_method)(*args, **kwargs)``.
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
            applies to `self.values`


        Notes
        -----
        Use leading underscore for `_order`, `_copy` to avoid name possible name
        clashes.


        See Also
        --------
        from_data
        """

        if isinstance(func_or_method, str):
            values = getattr(self.values, func_or_method)(*args, **kwargs)
        else:
            values = func_or_method(self.values, *args, **kwargs)

        if _order:
            if hasattr(self, "mom_dims"):
                values = values.transpose(..., *self.mom_dims)
            else:
                raise AttributeError("to specify order, must have attribute `mom_dims`")

        if _kws is None:
            _kws = {}
        else:
            _kws = dict(_kws)
        _kws.setdefault("copy", _copy)
        _kws.setdefault("verify", _verify)
        _kws.setdefault("mom_ndim", self.mom_ndim)
        if _check_mom:
            _kws["mom"] = self.mom
            _kws["check_shape"] = True

        out = type(self).from_data(data=values, **_kws)

        return out

    @property
    def _is_vector(self) -> bool:
        return self.val_ndim > 0

    def _raise_if_scalar(self, message: str | None = None) -> None:
        if not self._is_vector:
            if message is None:
                message = "not implemented for scalar"
            raise ValueError(message)

    # * Universal reducers

    @docfiller.decorate
    def resample(
        self,
        indices: MyNDArray,
        axis: int = 0,
        first: bool = True,
        **kws: Any,
    ) -> Self:
        """
        Create a new object sampled from index.

        Parameters
        ----------
        {indices}
        {axis}
        first : bool, default=True
            if True, and axis != 0, the move the axis to first position.
            This makes results similar to resample and reduce
            If `first` False, then resampled array can have odd shape

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
        self._raise_if_scalar()
        axis = self._wrap_axis(axis, **kws)

        data = self.data
        if first and axis != 0:
            data = np.moveaxis(data, axis, 0)
            axis = 0

        out = np.take(data, indices, axis=axis)

        return type(self).from_data(
            data=out,
            mom_ndim=self.mom_ndim,
            mom=self.mom,
            copy=False,
            verify=True,
            **kws,
        )

    ###########################################################################
    # ** Operators
    ###########################################################################
    def _check_other(self, b: Self) -> None:
        """Check other object."""
        assert type(self) == type(b)
        assert self.mom_ndim == b.mom_ndim
        assert self.shape == b.shape

    def __iadd__(
        self,
        b: Self,
    ) -> Self:  # noqa D105
        """Self adder."""
        self._check_other(b)
        # self.push_data(b.data)
        # return self
        return self.push_data(b.data)

    def __add__(
        self,
        b: Self,
    ) -> Self:
        """Add objects to new object."""
        self._check_other(b)
        # new = self.copy()
        # new.push_data(b.data)
        # return new
        return self.copy().push_data(b.data)

    def __isub__(
        self,
        b: Self,
    ) -> Self:
        """Inplace subtraction."""
        # NOTE: consider implementint push_data_scale routine to make this cleaner
        self._check_other(b)
        assert np.all(self.weight() >= b.weight())
        data = b.data.copy()
        data[self._weight_index] *= -1
        # self.push_data(data)
        # return self
        return self.push_data(data)

    def __sub__(
        self,
        b: Self,
    ) -> Self:
        """Subtract objects."""
        self._check_other(b)
        assert np.all(self.weight() >= b.weight())
        new = b.copy()
        new._data[self._weight_index] *= -1
        # new.push_data(self.data)
        # return new
        return new.push_data(self.data)

    def __mul__(self, scale: float | int) -> Self:
        """New object with weights scaled by scale."""  # noqa D401
        scale = float(scale)
        new = self.copy()
        new._data[self._weight_index] *= scale
        return new

    def __imul__(self, scale: float | int) -> Self:
        """Inplace multiply."""
        scale = float(scale)
        self._data[self._weight_index] *= scale
        return self

    ###########################################################################
    # ** Constructors
    ###########################################################################
    # *** Utils
    @classmethod
    def _check_mom(
        cls,
        moments: Moments | None,
        mom_ndim: Mom_NDim | None,
        shape: tuple[int, ...] | None = None,
    ) -> tuple[int] | tuple[int, int]:
        """
        Check moments for correct shape.

        If moments is None, infer from
        shape[-mom_ndim:] if integer, convert to tuple.
        """

        if isinstance(moments, int):
            mom_ndim = mom_ndim or 1
            moments = (moments,) * mom_ndim  # type: ignore

        elif moments is None:
            if mom_ndim is None:
                raise ValueError("must specify either moments or mom_ndim and shape")
            elif shape is None:
                raise ValueError("Must pass shape if infering moments")

            assert len(shape) >= mom_ndim and mom_ndim in [1, 2]
            moments = tuple(x - 1 for x in shape[-mom_ndim:])  # type: ignore

        elif mom_ndim is None:
            moments = tuple(moments)  # type: ignore
            mom_ndim = len(moments)  # type: ignore
            assert mom_ndim in [1, 2]

        assert len(moments) == mom_ndim  # type: ignore
        return cast(Union[Tuple[int], Tuple[int, int]], moments)

    @staticmethod
    def _datas_axis_to_first(
        datas: MyNDArray, axis: int, mom_ndim: Mom_NDim, **kws: Any
    ) -> tuple[MyNDArray, int]:
        """Move axis to first first position."""
        # NOTE: removinvg this. should be handles elsewhere
        # datas = np.asarray(datas)
        # ndim = datas.ndim - mom_ndim
        # if axis < 0:
        #     axis += ndim
        # assert 0 <= axis < ndim
        from numpy.core.numeric import normalize_axis_index  # type: ignore

        axis = normalize_axis_index(axis, datas.ndim - mom_ndim)
        if axis != 0:
            datas = np.moveaxis(datas, axis, 0)
        return datas, axis

    def _wrap_axis(
        self, axis: int | None, default: int = 0, ndim: int | None = None, **kws: Any
    ) -> int:
        """Wrap axis to positive value and check."""
        from numpy.core.numeric import normalize_axis_index  # type: ignore

        if axis is None:
            axis = default
        if ndim is None:
            ndim = self.val_ndim

        return cast(int, normalize_axis_index(axis, ndim))

    @classmethod
    def _mom_ndim_from_mom(cls, mom: Moments) -> Mom_NDim:
        if isinstance(mom, int):
            out = 1
        elif isinstance(mom, tuple):
            out = len(mom)
            assert out in [1, 2]
        else:
            raise ValueError("mom must be int or tuple")
        return cast(Mom_NDim, out)

    @classmethod
    def _choose_mom_ndim(
        cls,
        mom: Moments | None,
        mom_ndim: Mom_NDim | None,
    ) -> Mom_NDim:
        if mom is not None:
            mom_ndim = cls._mom_ndim_from_mom(mom)

        if mom_ndim is None:
            raise ValueError("must specify mom_ndim or mom")

        return mom_ndim

    # *** Core
    @classmethod
    @abstractmethod
    @docfiller.decorate
    def zeros(
        cls,
        mom: Moments | None = None,
        mom_ndim: Mom_NDim | None = None,
        val_shape: tuple[int, ...] | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        zeros_kws: Mapping[str, Any] | None = None,
        **kws: Any,
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
        **kws
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
        mom: Moments | None = None,
        mom_ndim: Mom_NDim | None = None,
        val_shape: tuple[int, ...] | None = None,
        copy: bool = True,
        copy_kws: Mapping[str, Any] | None = None,
        verify: bool = True,
        check_shape: bool = True,
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
        {check_shape}
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
        mom_ndim: Mom_NDim | None = None,
        axis: int | None = 0,
        mom: Moments | None = None,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        verify: bool = True,
        check_shape: bool = True,
        **kws: Any,
    ) -> Self:
        """
        Create object from multiple data arrays.

        Parameters
        ----------
        datas : ndarray
            Array of multiple Moment arrays.
            datas[..., i, ...] is the ith data array, where i is
            in position `axis`.
        {axis}
        {mom}
        {mom_ndim}
        {val_shape}
        {dtype}
        {verify}
        {check_shape}
        **kws
            Extra arguments

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
        w: Any = None,
        axis: int | None = 0,
        mom: Moments = 2,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        broadcast: bool = False,
        **kws: Any,
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
        {axis}
        {mom}
        {val_shape}
        {broadcast}
        **kws
            Optional arguments passed to :meth:`zeros`

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
        *args: Any,
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
        w : scalar or array-like, optional
            Optional weights.  If scalar or array, attempt to
            broadcast to `x0.shape`
        {freq}
        {indices}
        {nrep}
        {axis}
        {mom}
        {dtype}
        {broadcast}
        parallel : bool, default=True
            If True, perform resampling in parallel.
        {resample_kws}
        {full_output}
        **kws
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
        mom_ndim: Mom_NDim | None = None,
        mom: Moments | None = None,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        convert_kws: Mapping[str, Any] | None = None,
        **kws: Any,
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
        **kws
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
        mom_ndim: Mom_NDim | None = None,
        mom: Moments | None = None,
        axis: int | None = 0,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        convert_kws: Mapping[str, Any] | None = None,
        **kws: Any,
    ) -> Self:
        """
        Create object from multiple `raw` moment arrays.

        Parameters
        ----------
        raws : ndarray
            raws[...,i,...] is the ith sample of a `raw` array,
            Note that raw[...,i,j] = <x0**i, x1**j>
            where `i` is in position `axis`
        {axis}
        {mom_ndim}
        {mom}
        {val_shape}
        {dtype}
        {convert_kws}
        **kws
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

    # --------------------------------------------------
    # mom_ndim == 1 specific
    # --------------------------------------------------

    # @staticmethod
    # def _raise_if_not_1d(mom_ndim: Mom_NDim) -> None:
    #     if mom_ndim != 1:
    #         raise NotImplementedError("only available for mom_ndim == 1")

    # special, 1d only methods
    # def push_stat(
    #     self: T_CentralMoments,
    #     a: MyNDArray | float,
    #     v: MyNDArray | float = 0.0,
    #     w: MyNDArray | float | None = None,
    #     broadcast: bool = True,
    # ) -> T_CentralMoments:
    #     """Push statistics onto self."""
    #     self._raise_if_not_1d(self.mom_ndim)

    #     ar, target = self._check_val(a, target="val")
    #     vr = self._check_var(v, broadcast=broadcast)
    #     wr = self._check_weight(w, target=target)
    #     self._push.stat(self._data_flat, wr, ar, vr)
    #     return self

    # def push_stats(
    #     self: T_CentralMoments,
    #     a: MyNDArray,
    #     v: MyNDArray | float = 0.0,
    #     w: MyNDArray | float | None = None,
    #     axis: int = 0,
    #     broadcast: bool = True,
    # ) -> T_CentralMoments:
    #     """Push multiple statistics onto self."""
    #     self._raise_if_not_1d(self.mom_ndim)

    #     ar, target = self._check_vals(a, target="vals", axis=axis)
    #     vr = self._check_vars(v, target=target, axis=axis, broadcast=broadcast)
    #     wr = self._check_weights(w, target=target, axis=axis)
    #     self._push.stats(self._data_flat, wr, ar, vr)
    #     return self

    # @classmethod
    # def from_stat(
    #     cls: Type[T_CentralMoments],
    #     a: ArrayLike | float,
    #     v: MyNDArray | float = 0.0,
    #     w: MyNDArray | float | None = None,
    #     mom: Moments = 2,
    #     val_shape: Tuple[int, ...] | None = None,
    #     dtype: DTypeLike | None = None,
    #     order: ArrayOrder | None = None,
    #     **kws,
    # ) -> T_CentralMoments:
    #     """Create object from single weight, average, variance/covariance."""
    #     mom_ndim = cls._mom_ndim_from_mom(mom)
    #     cls._raise_if_not_1d(mom_ndim)

    #     a = np.asarray(a, dtype=dtype, order=order)

    #     if val_shape is None and isinstance(a, MyNDArray):
    #         val_shape = a.shape
    #     if dtype is None:
    #         dtype = a.dtype

    #     return cls.zeros(val_shape=val_shape, mom=mom, dtype=dtype, **kws).push_stat(
    #         w=w, a=a, v=v
    #     )

    # @classmethod
    # def from_stats(
    #     cls: Type[T_CentralMoments],
    #     a: MyNDArray,
    #     v: MyNDArray,
    #     w: MyNDArray | float | None = None,
    #     axis: int = 0,
    #     mom: Moments = 2,
    #     val_shape: Tuple[int, ...] = None,
    #     dtype: DTypeLike | None = None,
    #     order: ArrayOrder | None = None,
    #     **kws,
    # ) -> T_CentralMoments:
    #     """Create object from several statistics.

    #     Weights, averages, variances/covarainces along
    #     axis.
    #     """

    #     mom_ndim = cls._mom_ndim_from_mom(mom)
    #     cls._raise_if_not_1d(mom_ndim)

    #     a = np.asarray(a, dtype=dtype, order=order)

    #     # get val_shape
    #     if val_shape is None:
    #         val_shape = shape_reduce(a.shape, axis)
    #     return cls.zeros(val_shape=val_shape, dtype=dtype, mom=mom, **kws).push_stats(
    #         a=a, v=v, w=w, axis=axis
    #     )

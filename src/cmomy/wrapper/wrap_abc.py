"""
Attempt at a more generic wrapper class

Idea is to Wrap ndarray, xr.DataArray, and xr.Dataset objects...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, cast, overload

import numpy as np

from cmomy.core.docstrings import docfiller
from cmomy.core.missing import MISSING
from cmomy.core.typing import GenArrayT
from cmomy.core.utils import mom_shape_to_mom
from cmomy.core.validate import (
    is_dataset,
    is_xarray,
    raise_if_wrong_value,
    validate_floating_dtype,
    validate_mom_and_mom_ndim,
    validate_mom_dims,
    validate_mom_ndim,
)
from cmomy.factory import factory_pusher, parallel_heuristic
from cmomy.utils import assign_moment

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping, Sequence
    from typing import (
        NoReturn,
    )

    import xarray as xr
    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from cmomy.core.typing import (
        ApplyUFuncKwargs,
        ArrayLikeArg,
        ArrayOrder,
        ArrayOrderCF,
        AxisReduce,
        BlockByModes,
        Casting,
        DataT_,
        DimsReduce,
        DTypeLikeArg,
        FloatT_,
        KeepAttrs,
        MissingCoreDimOptions,
        MissingType,
        Mom_NDim,
        MomDims,
        Moments,
        MomentsStrict,
        NDArrayAny,
        NDArrayInt,
        ReduceValsKwargs,
        ResampleValsKwargs,
        RngTypes,
        SelectMoment,
        WrapRawKwargs,
    )
    from cmomy.core.typing_compat import Self, Unpack
    from cmomy.factory import Pusher
    from cmomy.wrapper import CentralMomentsArray, CentralMomentsData


@docfiller.decorate  # noqa: PLR0904
class CentralMomentsABC(ABC, Generic[GenArrayT]):
    r"""
    Wrapper to calculate central moments.

    Parameters
    ----------
    obj : {t_array}
        Central moments array.
    {mom_ndim}
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

    __slots__ = ("_mom_ndim", "_obj")

    _obj: GenArrayT
    _mom_ndim: Mom_NDim

    def __init__(
        self,
        obj: GenArrayT,
        *,
        mom_ndim: Mom_NDim = 1,
        fastpath: bool = False,
    ) -> None:
        self._mom_ndim = mom_ndim if fastpath else validate_mom_ndim(mom_ndim)
        self._obj = obj

        if fastpath:
            return

        # catchall to test ndim < mom_ndim
        if len(self.mom) < self.mom_ndim:
            msg = f"{len(self.mom)=} != {self.mom_ndim=}.  Possibly more mom_ndim than ndim."
            raise ValueError(msg)

        # must have positive moments
        if any(m <= 0 for m in self.mom):
            msg = "Moments must be positive"
            raise ValueError(msg)

        self._validate_dtype()

    # ** Properties -----------------------------------------------------------
    @property
    def obj(self) -> GenArrayT:
        """Underlying object."""
        return self._obj

    @property
    def mom_ndim(self) -> Mom_NDim:
        """Number of moment dimensions."""
        return self._mom_ndim

    @property
    @abstractmethod
    def mom_shape(self) -> MomentsStrict:
        """Shape of moments dimensions."""

    @property
    def mom(self) -> MomentsStrict:
        """Moments tuple."""
        return mom_shape_to_mom(self.mom_shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        """DType of wrapped object."""
        if is_dataset(self._obj):
            self._raise_notimplemented_for_dataset()
        return self._obj.dtype  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of wrapped object."""
        if is_dataset(self._obj):
            self._raise_notimplemented_for_dataset()
        return self._obj.shape

    @property
    def val_shape(self) -> tuple[int, ...]:
        """Shape of values dimensions."""
        if is_dataset(self._obj):
            self._raise_notimplemented_for_dataset()
        return self.shape[: -self._mom_ndim]

    @property
    def ndim(self) -> int:
        """Total number of dimensions."""
        if is_dataset(self._obj):
            self._raise_notimplemented_for_dataset()
        return self._obj.ndim

    @property
    def val_ndim(self) -> int:
        """Number of values dimensions (``ndim - mom_ndim``)."""
        if is_dataset(self._obj):
            self._raise_notimplemented_for_dataset()
        return self.ndim - self._mom_ndim

    def __repr__(self) -> str:
        """Repr for class."""
        name = self.__class__.__name__
        s = f"<{name}(mom_ndim={self.mom_ndim})>\n"
        return s + repr(self._obj)

    def _repr_html_(self) -> str:  # noqa: PLW3201
        from cmomy.core.formatting import (
            repr_html_wrapper,  # pyright: ignore[reportUnknownVariableType]
        )

        return repr_html_wrapper(self)  # type: ignore[no-any-return,no-untyped-call]

    def __array__(  # noqa: PLW3201
        self, dtype: DTypeLike = None, copy: bool | None = None
    ) -> NDArrayAny:
        """Used by np.array(self)."""  # D401
        return np.asarray(self._obj, dtype=dtype)

    def to_numpy(self) -> NDArrayAny:
        """Coerce wrapped data to :class:`~numpy.ndarray` if possible."""
        return np.asarray(self)

    # ** Create/copy/new ------------------------------------------------------
    @abstractmethod
    def _new_like(self, obj: GenArrayT) -> Self:
        """Create new object with same properties (mom_ndim, etc) as self with no checks and fastpath"""

    @abstractmethod
    @docfiller.decorate
    def new_like(
        self,
        obj: GenArrayT | None = None,
        *,
        verify: bool = False,
        copy: bool | None = None,
        dtype: DTypeLike = None,
        order: ArrayOrder = None,
        fastpath: bool = False,
    ) -> Self:
        """
        Create new object like self, with new data.

        Parameters
        ----------
        obj : {t_array}
            Data for new object.  Must be conformable to ``self.obj``.
        {verify}
        {copy}
        {dtype}
        {order}
        {fastpath}

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
        casting: Casting | None = None,
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
        {casting}

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

        return self._new_like(
            obj=self._obj.astype(dtype, **kwargs),  # type: ignore[arg-type]
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
            obj=self._obj,
            verify=False,
            copy=True,
        )

    # ** Utils ----------------------------------------------------------------
    def _raise_if_not_mom_ndim_1(self) -> None:
        if self._mom_ndim != 1:
            msg = "Only implemented for `mom_ndim==1`"
            raise ValueError(msg)

    def _raise_not_implemented(self, method: str = "method") -> NoReturn:
        msg = f"{method} not implemented for wrapped {type(self._obj)}."
        raise NotImplementedError(msg)

    def _raise_if_wrong_mom_shape(self, mom_shape: tuple[int, ...]) -> None:
        raise_if_wrong_value(mom_shape, self.mom_shape, "Wrong moments shape.")

    @staticmethod
    def _raise_notimplemented_for_dataset() -> NoReturn:
        msg = "Not implemented for wrapped Dataset"
        raise NotImplementedError(msg)

    @abstractmethod
    def _validate_dtype(self) -> None:
        """Validate dtype of obj"""

    @staticmethod
    def _check_y(y: tuple[Any, ...], mom_ndim: int) -> None:
        raise_if_wrong_value(
            len(y), mom_ndim - 1, "`len(y)` must equal `mom_ndim - 1`."
        )

    @classmethod
    def _mom_dims_kws(
        cls,
        mom_dims: MomDims | None,
        mom_ndim: Mom_NDim,
        out: Any = None,
    ) -> dict[str, Any]:
        return (
            {"mom_dims": validate_mom_dims(mom_dims, mom_ndim, out)}
            if hasattr(cls, "mom_dims")
            else {}
        )

    # ** Pushing --------------------------------------------------------------
    def _pusher(self, parallel: bool | None = None, size: int | None = None) -> Pusher:
        return factory_pusher(
            mom_ndim=self._mom_ndim, parallel=parallel_heuristic(parallel, size=size)
        )

    @abstractmethod
    @docfiller.decorate
    def push_data(
        self,
        data: Any,
        *,
        casting: Casting = "same_kind",
        parallel: bool | None = False,
        scale: ArrayLike | None = None,
    ) -> Self:
        """
        Push data object to moments.

        Parameters
        ----------
        data : array-like or {t_array}
            Accumulation array conformable to ``self.obj``.
        {casting}
        {parallel}
        scale : array-like
            Scaling to apply to weights of ``data``.  Optional.

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
        casting: Casting = "same_kind",
        parallel: bool | None = None,
    ) -> Self:
        """
        Push and reduce multiple average central moments.

        Parameters
        ----------
        datas : array-like or {t_array}
            Collection of accumulation arrays to push onto ``self``.
        {axis_data_and_dim}
        {casting}
        {parallel}

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
        casting: Casting = "same_kind",
        parallel: bool | None = False,
    ) -> Self:
        """
        Push single sample to central moments.

        Parameters
        ----------
        x : array-like or {t_array}
            Values to push onto ``self``.
        *y : array-like or {t_array}
            Additional values (needed if ``mom_ndim > 1``)
        weight : int, float, array-like or {t_array}
            Weight of each sample.  If scalar, broadcast `w.shape` to `x0.shape`.
        {casting}
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
        casting: Casting = "same_kind",
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
        {casting}
        {parallel}

        Returns
        -------
        output : {klass}
            Same object with pushed data.
        """

    # ** Operators ------------------------------------------------------------
    def _check_other(self, other: Self) -> None:
        """Check other object."""
        if type(self) is not type(other):
            msg = f"{type(self)=} != {type(other)=}"
            raise TypeError(msg)
        self._raise_if_wrong_mom_shape(other.mom_shape)

    def __iadd__(self, other: Self) -> Self:  # noqa: PYI034
        """Self adder."""
        self._check_other(other)
        return self.push_data(other.obj)

    def __add__(self, other: Self) -> Self:
        """Add objects to new object."""
        self._check_other(other)
        return self.copy().push_data(other.obj)

    def __isub__(self, other: Self) -> Self:  # noqa: PYI034
        """Inplace subtraction."""
        self._check_other(other)
        return self.push_data(other.obj, scale=-1.0)

    def __sub__(self, other: Self) -> Self:
        """Subtract objects."""
        self._check_other(other)
        return self.copy().push_data(other.obj, scale=-1.0)

    def __mul__(self, scale: float) -> Self:
        """New object with weight scaled by scale."""  # D401
        return self.assign_moment(weight=self.weight() * scale, copy=True)

    def __imul__(self, scale: float) -> Self:  # noqa: PYI034
        """Inplace multiply."""
        # do this to make sure things work with dask...
        self._obj = self.assign_moment(weight=self.weight() * scale, copy=False)._obj
        return self

    # ** Pipe -----------------------------------------------------------------
    def pipe(
        self,
        func_or_method: Callable[..., Any] | str,
        *args: Any,
        _reorder: bool = True,
        _copy: bool | None = None,
        _verify: bool = False,
        _fastpath: bool = False,
        **kwargs: Any,
    ) -> Self:
        """
        Apply `func_or_method` to underlying data and wrap results in
        new wrapped object.

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
            values = getattr(self._obj, func_or_method)(*args, **kwargs)
        else:
            values = func_or_method(self._obj, *args, **kwargs)

        if _reorder and hasattr(self, "mom_dims"):
            values = values.transpose(..., *self.mom_dims)  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]

        return self.new_like(
            obj=values,
            copy=_copy,
            verify=_verify,
            fastpath=_fastpath,
        )

    # ** Interface to modules -------------------------------------------------
    # *** .utils --------------------------------------------------------------
    @docfiller.decorate
    def moveaxis(
        self,
        axis: int | tuple[int, ...] | MissingType = MISSING,
        dest: int | tuple[int, ...] | MissingType = MISSING,
        *,
        dim: str | Sequence[Hashable] | MissingType = MISSING,
        dest_dim: str | Sequence[Hashable] | MissingType = MISSING,
    ) -> Self:
        """
        Generalized moveaxis

        Parameters
        ----------
        axis : int or sequence of int
            Original positions of axes to move.
        dest : int or sequence of int
            Destination positions for each original axes.
        **kwargs
            Extra arguments to :func:`.utils.moveaxis`

        Returns
        -------
        output : {klass}
            Object with moved axes.  This is a view of the original data.

        See Also
        --------
        .utils.moveaxis
        numpy.moveaxis
        """
        from cmomy.utils import moveaxis

        if is_dataset(self._obj):
            self._raise_notimplemented_for_dataset()

        obj = moveaxis(
            self._obj,
            axis=axis,
            dest=dest,
            dim=dim,
            dest_dim=dest_dim,
            mom_ndim=self._mom_ndim,
        )

        return self.new_like(
            obj=obj,
        )

    @docfiller.decorate
    def select_moment(
        self,
        name: SelectMoment,
        *,
        squeeze: bool = True,
        dim_combined: str = "variable",
        coords_combined: str | Sequence[Hashable] | None = None,
        keep_attrs: KeepAttrs = None,
        on_missing_core_dim: MissingCoreDimOptions = "drop",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> GenArrayT:
        """
        Select specific moments.

        Parameters
        ----------
        {select_moment_name}
        {select_squeeze}
        {select_dim_combined}
        {select_coords_combined}
        {keep_attrs}
        {on_missing_core_dim}
        {apply_ufunc_kwargs}

        Returns
        -------
        output : ndarray or DataArray or Dataset
            Same type as ``self.obj``. If ``name`` is ``ave`` or ``var``, the last
            dimensions of ``output`` has shape ``mom_ndim`` with each element
            corresponding to the `ith` variable. If ``squeeze=True`` and
            `mom_ndim==1`, this last dimension is removed. For all other ``name``
            options, output has shape of input with moment dimensions removed.


        See Also
        --------
        .utils.select_moment
        """
        from cmomy.utils import select_moment

        return select_moment(
            self._obj,
            name=name,
            mom_ndim=self._mom_ndim,
            dim_combined=dim_combined,
            coords_combined=coords_combined,
            squeeze=squeeze,
            keep_attrs=keep_attrs,
            mom_dims=getattr(self, "mom_dims", None),
            on_missing_core_dim=on_missing_core_dim,
            apply_ufunc_kwargs=apply_ufunc_kwargs,
        )

    @docfiller.decorate
    def assign_moment(
        self,
        moment: Mapping[SelectMoment, ArrayLike] | None = None,
        *,
        squeeze: bool = True,
        copy: bool = True,
        dim_combined: Hashable | None = None,
        keep_attrs: KeepAttrs = None,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
        **moment_kwargs: ArrayLike | xr.DataArray | xr.Dataset,
    ) -> Self:
        """
        Create object with update weight, average, etc.

        Parameters
        ----------
        {assign_moment_mapping}
        {select_squeeze}
        copy : bool, default=True
            If ``True`` (the default), return new array with updated weights.
            Otherwise, return the original array with weights updated inplace.
            Note that a copy is always created for a ``dask`` backed object.
        dim_combined : str, optional
            Name of dimensions for multiple values. Must supply if passing in
            multiple values for ``name="ave"`` etc.
        {mom_dims_data}
        {keep_attrs}
        {on_missing_core_dim}
        {apply_ufunc_kwargs}
        **moment_kwargs
            Keyword argument form of ``moment``.  Must provide either ``moment`` or ``moment_kwargs``.

        Returns
        -------
        output : object
            Same type as ``self`` with updated data.

        See Also
        --------
        .utils.assign_moment
        """
        obj = assign_moment(
            data=self._obj,
            moment=moment,
            mom_ndim=self._mom_ndim,
            squeeze=squeeze,
            copy=copy,
            mom_dims=getattr(self, "mom_dims", None),
            dim_combined=dim_combined,
            keep_attrs=keep_attrs,
            on_missing_core_dim=on_missing_core_dim,
            apply_ufunc_kwargs=apply_ufunc_kwargs,
            **moment_kwargs,
        )
        return self._new_like(obj=obj)

    # *** .convert ------------------------------------------------------------
    @docfiller.decorate
    def cumulative(
        self,
        *,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        move_axis_to_end: bool = False,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        casting: Casting = "same_kind",
        order: ArrayOrder = None,
        parallel: bool | None = None,
        keep_attrs: KeepAttrs = None,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> GenArrayT:
        """
        Convert to cumulative moments.

        Parameters
        ----------
        {axis_data_and_dim}
        {move_axis_to_end}
        {out}
        {dtype}
        {casting}
        {order}
        {parallel}
        {keep_attrs}
        {mom_dims_data}
        {on_missing_core_dim}
        {apply_ufunc_kwargs}

        Returns
        -------
        output : {t_array}
            Same type as ``self.obj``, with moments accumulated over ``axis``.

        See Also
        --------
        .convert.cumulative
        """
        from cmomy.convert import cumulative

        return cumulative(  # type: ignore[no-any-return]
            self._obj,
            axis=axis,
            dim=dim,
            mom_ndim=self._mom_ndim,
            inverse=False,
            move_axis_to_end=move_axis_to_end,
            out=out,
            dtype=dtype,
            casting=casting,
            order=order,
            parallel=parallel,
            keep_attrs=keep_attrs,
            mom_dims=getattr(self, "mom_dims", None),
            on_missing_core_dim=on_missing_core_dim,
            apply_ufunc_kwargs=apply_ufunc_kwargs,
        )

    @docfiller.decorate
    def moments_to_comoments(
        self,
        *,
        mom: tuple[int, int],
        mom_dims2: MomDims | None = None,
        dtype: DTypeLike = None,
        order: ArrayOrderCF = None,
        keep_attrs: KeepAttrs = None,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self:
        """
        Convert moments (mom_ndim=1) to comoments (mom_ndim=2).

        Parameters
        ----------
        {mom_moments_to_comoments}
        mom_dims2 : tuple of str
            Moments dimensions for output (``mom_ndim=2``) data.  Defaults to ``("mom_0", "mom_1")``.
        {keep_attrs}
        {on_missing_core_dim}
        {apply_ufunc_kwargs}

        Return
        ------
        output : {klass}
            Same type as ``self`` with ``mom_ndim=2``.

        See Also
        --------
        .convert.moments_to_comoments
        """
        self._raise_if_not_mom_ndim_1()
        from cmomy import convert

        kws: dict[str, Any]
        if is_xarray(self._obj):
            mom_dims2 = validate_mom_dims(mom_dims2, mom_ndim=2)
            kws = {"mom_dims": mom_dims2}
        else:
            mom_dims2 = None
            kws = {}

        return type(self)(
            convert.moments_to_comoments(
                self._obj,
                mom=mom,
                mom_dims=getattr(self, "mom_dims", None),
                mom_dims2=mom_dims2,
                dtype=dtype,
                order=order,
                keep_attrs=keep_attrs,
                on_missing_core_dim=on_missing_core_dim,
                apply_ufunc_kwargs=apply_ufunc_kwargs,
            ),
            mom_ndim=2,
            **kws,
        )

    # *** .resample -----------------------------------------------------------
    @docfiller.decorate
    def resample_and_reduce(  # noqa: PLR0913
        self,
        *,
        freq: ArrayLike | xr.DataArray | GenArrayT | None = None,
        nrep: int | None = None,
        rng: RngTypes | None = None,
        paired: bool = True,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        rep_dim: str = "rep",
        move_axis_to_end: bool = False,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        casting: Casting = "same_kind",
        order: ArrayOrder = None,
        parallel: bool | None = None,
        keep_attrs: KeepAttrs = None,
        # dask specific...
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self:
        """
        Bootstrap resample and reduce.

        Parameters
        ----------
        {freq}
        {nrep_optional}
        {rng}
        {paired}
        {axis_data_and_dim}
        {rep_dim}
        {move_axis_to_end}
        {out}
        {dtype}
        {casting}
        {order}
        {parallel}
        {keep_attrs}
        {on_missing_core_dim}
        {apply_ufunc_kwargs}

        Returns
        -------
        output : object
            Instance of calling class. Note that new object will have
            ``(...,shape[axis-1], nrep, shape[axis+1], ...)``,
            where ``nrep = freq.shape[0]``.

        See Also
        --------
        reduce
        ~.resample.randsamp_freq : random frequency sample
        ~.resample.freq_to_indices : convert frequency sample to index sample
        ~.resample.indices_to_freq : convert index sample to frequency sample
        ~.resample.resample_data : method to perform resampling

        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.default_rng(0)
        >>> da = cmomy.CentralMomentsArray.from_vals(
        ...     rng.random((10, 3)),
        ...     mom=3,
        ...     axis=0,
        ... ).to_x(dims="rec")
        >>> da
        <CentralMomentsData(mom_ndim=1)>
        <xarray.DataArray (rec: 3, mom_0: 4)> Size: 96B
        array([[ 1.0000e+01,  5.2485e-01,  1.1057e-01, -4.6282e-03],
               [ 1.0000e+01,  5.6877e-01,  6.8876e-02, -1.2745e-02],
               [ 1.0000e+01,  5.0944e-01,  1.1978e-01, -1.4644e-02]])
        Dimensions without coordinates: rec, mom_0

        Note that for reproducible results, must set numba random
        seed as well

        >>> freq = cmomy.randsamp_freq(data=da.obj, dim="rec", nrep=5)
        >>> da_resamp = da.resample_and_reduce(
        ...     dim="rec",
        ...     freq=freq,
        ... )
        >>> da_resamp
        <CentralMomentsData(mom_ndim=1)>
        <xarray.DataArray (rep: 5, mom_0: 4)> Size: 160B
        array([[ 3.0000e+01,  5.0944e-01,  1.1978e-01, -1.4644e-02],
               [ 3.0000e+01,  5.3435e-01,  1.0038e-01, -1.2329e-02],
               [ 3.0000e+01,  5.2922e-01,  1.0360e-01, -1.6009e-02],
               [ 3.0000e+01,  5.5413e-01,  8.3204e-02, -1.1267e-02],
               [ 3.0000e+01,  5.4899e-01,  8.6627e-02, -1.5407e-02]])
        Dimensions without coordinates: rep, mom_0

        Alternatively, we can resample and reduce

        >>> indices = cmomy.resample.freq_to_indices(freq)
        >>> da.sel(rec=xr.DataArray(indices, dims=["rep", "rec"])).reduce(dim="rec")
        <CentralMomentsData(mom_ndim=1)>
        <xarray.DataArray (rep: 5, mom_0: 4)> Size: 160B
        array([[ 3.0000e+01,  5.0944e-01,  1.1978e-01, -1.4644e-02],
               [ 3.0000e+01,  5.3435e-01,  1.0038e-01, -1.2329e-02],
               [ 3.0000e+01,  5.2922e-01,  1.0360e-01, -1.6009e-02],
               [ 3.0000e+01,  5.5413e-01,  8.3204e-02, -1.1267e-02],
               [ 3.0000e+01,  5.4899e-01,  8.6627e-02, -1.5407e-02]])
        Dimensions without coordinates: rep, mom_0

        """
        from cmomy.resample import resample_data

        # pyright error due to `freq` above...
        return self._new_like(
            obj=resample_data(  # pyright: ignore[reportCallIssue, reportUnknownArgumentType]
                self._obj,  # pyright: ignore[reportArgumentType]
                mom_ndim=self._mom_ndim,
                freq=freq,
                nrep=nrep,
                rng=rng,
                paired=paired,
                axis=axis,
                dim=dim,
                rep_dim=rep_dim,
                move_axis_to_end=move_axis_to_end,
                dtype=dtype,
                out=out,
                casting=casting,
                order=order,
                parallel=parallel,
                keep_attrs=keep_attrs,
                mom_dims=getattr(self, "mom_dims", None),
                on_missing_core_dim=on_missing_core_dim,
                apply_ufunc_kwargs=apply_ufunc_kwargs,
            )
        )

    @docfiller.decorate
    def jackknife_and_reduce(
        self,
        *,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        data_reduced: Self | GenArrayT | None = None,
        rep_dim: str | None = "rep",
        move_axis_to_end: bool = False,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        casting: Casting = "same_kind",
        order: ArrayOrder = None,
        parallel: bool | None = None,
        keep_attrs: KeepAttrs = None,
        # dask specific...
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self:
        """
        Jackknife resample and reduce

        Parameters
        ----------
        {axis_data_and_dim}
        {parallel}
        data_reduced : array or {klass}
            Data reduced along ``axis``. Array of same type as ``self.obj`` or
            same type as ``self``.
        {rep_dim}
        {move_axis_to_end}
        {out}
        {dtype}
        {casting}
        {order}
        {parallel}
        {keep_attrs}
        {on_missing_core_dim}
        {apply_ufunc_kwargs}


        Returns
        -------
        object
            Instance of calling class with jackknife resampling along ``axis``.

        See Also
        --------
        ~.resample.jackknife_data
        """
        if isinstance(data_reduced, type(self)):
            data_reduced = data_reduced.obj

        from cmomy.resample import jackknife_data

        obj: GenArrayT = jackknife_data(  # pyright: ignore[reportUnknownVariableType, reportCallIssue]
            self._obj,  # pyright: ignore[reportArgumentType]
            mom_ndim=self._mom_ndim,
            axis=axis,
            dim=dim,
            data_reduced=data_reduced,  # pyright: ignore[reportArgumentType]
            rep_dim=rep_dim,
            move_axis_to_end=move_axis_to_end,
            out=out,
            dtype=dtype,
            casting=casting,
            order=order,
            parallel=parallel,
            keep_attrs=keep_attrs,
            on_missing_core_dim=on_missing_core_dim,
            apply_ufunc_kwargs=apply_ufunc_kwargs,
        )
        return self._new_like(obj=obj)  # pyright: ignore[reportUnknownArgumentType]

    # *** .reduction ----------------------------------------------------------
    def _block_by(
        self,
        block: int,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        mode: BlockByModes = "drop_last",
    ) -> NDArrayInt:
        from cmomy.reduction import block_by
        from cmomy.resample import select_ndat

        return block_by(
            ndat=select_ndat(self._obj, axis=axis, dim=dim, mom_ndim=self._mom_ndim),
            block=block,
            mode=mode,
        )

    @abstractmethod
    @docfiller.decorate
    def reduce(
        self,
    ) -> Self:
        """
        Create new object reduce along axis.

        Parameters
        ----------
        {by}
        block : int, optional
            If specified, perform block average reduction with blocks
            of this size.  Negative values are transformed to all data.
        {axis_data_and_dim}
        {move_axis_to_end}
        {out}
        {dtype}
        {casting}
        {order}
        {keepdims}
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
        ~.reduction.reduce_data_indexed
        ~.reduction.block_by
        """

    # ** Access to underlying statistics --------------------------------------
    def weight(self) -> GenArrayT:
        """Weight data."""
        return self.select_moment("weight")

    def mean(self, squeeze: bool = True) -> GenArrayT:
        """Mean (first moment)."""
        return self.select_moment("ave", squeeze=squeeze)

    def var(self, squeeze: bool = True) -> GenArrayT:
        """Variance (second central moment)."""
        return self.select_moment("var", squeeze=squeeze)

    def std(self, squeeze: bool = True) -> GenArrayT:
        """Standard deviation (ddof=0)."""
        if is_dataset(self._obj):
            self._raise_notimplemented_for_dataset()
        return cast("GenArrayT", np.sqrt(self.var(squeeze=squeeze)))

    def cov(self) -> GenArrayT:
        """Covariance (or variance if ``mom_ndim==1``)."""
        return self.select_moment("cov")

    def cmom(self) -> GenArrayT:
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
        return assign_moment(
            self._obj,
            {"weight": 1, "ave": 0},
            mom_ndim=self._mom_ndim,
            copy=True,
        )

    def to_raw(self, *, weight: ArrayLike | None = None) -> GenArrayT:
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
        from cmomy.convert import moments_type

        out = moments_type(self._obj, mom_ndim=self._mom_ndim, to="raw")
        if weight is not None:
            out = assign_moment(out, weight=weight, mom_ndim=self._mom_ndim, copy=False)
        return out

    def rmom(self) -> GenArrayT:
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

    # ** Constructors ---------------------------------------------------------
    @classmethod
    @abstractmethod
    @docfiller.decorate
    def zeros(
        cls,
        *,
        mom: Moments,
    ) -> Self:
        """
        Create a new base object.

        Parameters
        ----------
        {mom}

        Returns
        -------
        output : {klass}
            New instance with zero values.


        See Also
        --------
        numpy.zeros
        """

    @overload
    @classmethod
    def from_vals(
        cls,
        x: DataT_,
        *y: ArrayLike | xr.DataArray | DataT_,
        weight: ArrayLike | xr.DataArray | DataT_ | None = ...,
        out: NDArrayAny | None = ...,
        dtype: DTypeLike = ...,
        **kwargs: Unpack[ReduceValsKwargs],
    ) -> CentralMomentsData[DataT_]: ...
    @overload
    @classmethod
    def from_vals(
        cls,
        x: ArrayLikeArg[FloatT_],
        *y: ArrayLike,
        weight: ArrayLike | None = ...,
        out: None = ...,
        dtype: None = ...,
        **kwargs: Unpack[ReduceValsKwargs],
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    @classmethod
    def from_vals(
        cls,
        x: ArrayLike,
        *y: ArrayLike,
        weight: ArrayLike | None = ...,
        out: NDArray[FloatT_],
        dtype: DTypeLike = ...,
        **kwargs: Unpack[ReduceValsKwargs],
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    @classmethod
    def from_vals(
        cls,
        x: ArrayLike,
        *y: ArrayLike,
        weight: ArrayLike | None = ...,
        out: None = ...,
        dtype: DTypeLikeArg[FloatT_],
        **kwargs: Unpack[ReduceValsKwargs],
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    @classmethod
    def from_vals(
        cls,
        x: Any,
        *y: Any,
        weight: Any = ...,
        out: NDArrayAny | None = ...,
        dtype: DTypeLike = ...,
        **kwargs: Unpack[ReduceValsKwargs],
    ) -> Self: ...

    @classmethod
    @docfiller.decorate
    def from_vals(
        cls,
        x: ArrayLike | DataT_,
        *y: ArrayLike | xr.DataArray | DataT_,
        mom: Moments,
        weight: ArrayLike | xr.DataArray | DataT_ | None = None,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        mom_dims: MomDims | None = None,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        casting: Casting = "same_kind",
        order: ArrayOrderCF = None,
        keepdims: bool = False,
        parallel: bool | None = None,
        keep_attrs: KeepAttrs = None,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self | CentralMomentsABC[Any]:
        """
        Create from observations/values.

        Parameters
        ----------
        x : array-like or {t_array}
            Values to reduce.
        *y : array-like or {t_array}
            Additional values (needed if ``len(mom)==2``).
        weight : scalar or array-like
            Optional weight.  If scalar or array, attempt to
            broadcast to `x0.shape`
        {mom}
        {axis}
        {dim}
        {mom_dims}
        {keepdims}
        {out}
        {dtype}
        {casting}
        {order}
        {keepdims}
        {parallel}
        {keep_attrs}
        {on_missing_core_dim}
        {apply_ufunc_kwargs}

        Returns
        -------
        object
            Instance of calling class.

        See Also
        --------
        push_vals
        ~.reduction.reduce_vals

        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.default_rng(0)
        >>> x = rng.random((100, 3))
        >>> da = cmomy.CentralMomentsArray.from_vals(x, axis=0, mom=2)
        >>> da
        <CentralMomentsArray(mom_ndim=1)>
        array([[1.0000e+02, 5.5313e-01, 8.8593e-02],
               [1.0000e+02, 5.5355e-01, 7.1942e-02],
               [1.0000e+02, 5.1413e-01, 1.0407e-01]])

        """
        from cmomy.reduction import reduce_vals

        mom, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
        kws = cls._mom_dims_kws(mom_dims, mom_ndim)
        obj = reduce_vals(  # type: ignore[type-var, misc]
            x,  # pyright: ignore[reportArgumentType]
            *y,
            mom=mom,
            weight=weight,
            axis=axis,
            dim=dim,
            **kws,
            keepdims=keepdims,
            parallel=parallel,
            out=out,
            dtype=dtype,
            casting=casting,
            order=order,
            keep_attrs=keep_attrs,
            on_missing_core_dim=on_missing_core_dim,
            apply_ufunc_kwargs=apply_ufunc_kwargs,
        )

        return cls(
            obj=obj,  # type: ignore[arg-type]
            mom_ndim=mom_ndim,
            **kws,
            fastpath=True,
        )

    @overload
    @classmethod
    def from_resample_vals(
        cls,
        x: DataT_,
        *y: ArrayLike | xr.DataArray | DataT_,
        weight: ArrayLike | xr.DataArray | DataT_ | None = ...,
        freq: ArrayLike | xr.DataArray | DataT_ | None = ...,
        out: NDArrayAny | None = ...,
        dtype: DTypeLike = ...,
        **kwargs: Unpack[ResampleValsKwargs],
    ) -> CentralMomentsData[DataT_]: ...
    @overload
    @classmethod
    def from_resample_vals(
        cls,
        x: ArrayLikeArg[FloatT_],
        *y: ArrayLike,
        weight: ArrayLike | None = ...,
        freq: ArrayLike | None = ...,
        out: None = ...,
        dtype: None = ...,
        **kwargs: Unpack[ResampleValsKwargs],
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    @classmethod
    def from_resample_vals(
        cls,
        x: ArrayLike,
        *y: ArrayLike,
        weight: ArrayLike | None = ...,
        freq: ArrayLike | None = ...,
        out: NDArray[FloatT_],
        dtype: DTypeLike = ...,
        **kwargs: Unpack[ResampleValsKwargs],
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    @classmethod
    def from_resample_vals(
        cls,
        x: ArrayLike,
        *y: ArrayLike,
        weight: ArrayLike | None = ...,
        freq: ArrayLike | None = ...,
        out: None = ...,
        dtype: DTypeLikeArg[FloatT_],
        **kwargs: Unpack[ResampleValsKwargs],
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    @classmethod
    def from_resample_vals(
        cls,
        x: Any,
        *y: ArrayLike,
        weight: ArrayLike | None = ...,
        freq: ArrayLike | None = ...,
        out: NDArrayAny | None = ...,
        dtype: DTypeLike = ...,
        **kwargs: Unpack[ResampleValsKwargs],
    ) -> Self: ...

    @classmethod
    @docfiller.decorate
    def from_resample_vals(  # noqa: PLR0913
        cls,
        x: ArrayLike | DataT_,
        *y: ArrayLike | xr.DataArray | DataT_,
        mom: Moments,
        weight: ArrayLike | xr.DataArray | DataT_ | None = None,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        freq: ArrayLike | xr.DataArray | DataT_ | None = None,
        nrep: int | None = None,
        rng: RngTypes | None = None,
        paired: bool = True,
        move_axis_to_end: bool = True,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        casting: Casting = "same_kind",
        order: ArrayOrderCF = None,
        parallel: bool | None = None,
        mom_dims: MomDims | None = None,
        rep_dim: str = "rep",
        keep_attrs: KeepAttrs = None,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self | CentralMomentsABC[Any]:
        """
        Create from resample observations/values.

        This effectively resamples `x`.

        Parameters
        ----------
        x : array-like or {t_array}
            Observations.
        *y : array-like or {t_array}
            Additional values (needed if ``len(mom) > 1``).
        {mom}
        {weight}
        {axis}
        {dim}
        {freq}
        {nrep}
        {rng}
        {paired}
        {move_axis_to_end}
        {order}
        {out}
        {dtype}
        {casting}
        {order_cf}
        {parallel}
        {mom_dims}
        {rep_dim}
        {keep_attrs}
        {on_missing_core_dim}
        {apply_ufunc_kwargs}

        Returns
        -------
        object
            Instance of calling class


        See Also
        --------
        ~.resample.resample_vals
        ~.resample.randsamp_freq
        ~.resample.freq_to_indices
        ~.resample.indices_to_freq
        """
        mom, mom_ndim = validate_mom_and_mom_ndim(mom=mom, mom_ndim=None)
        kws = cls._mom_dims_kws(mom_dims, mom_ndim)

        from cmomy.resample import resample_vals

        obj = resample_vals(  # type: ignore[type-var, misc]
            x,  # pyright: ignore[reportArgumentType]
            *y,
            freq=freq,
            nrep=nrep,
            rng=rng,
            paired=paired,
            mom=mom,
            weight=weight,
            axis=axis,
            dim=dim,
            move_axis_to_end=move_axis_to_end,
            parallel=parallel,
            **kws,
            rep_dim=rep_dim,
            keep_attrs=keep_attrs,
            on_missing_core_dim=on_missing_core_dim,
            apply_ufunc_kwargs=apply_ufunc_kwargs,
            dtype=dtype,
            out=out,
            casting=casting,
            order=order,
        )
        return cls(
            obj=obj,  # type: ignore[arg-type]
            mom_ndim=mom_ndim,
            **kws,
            fastpath=True,
        )

    @overload
    @classmethod
    def from_raw(
        cls,
        raw: DataT_,
        *,
        out: NDArrayAny | None = ...,
        dtype: DTypeLike = ...,
        **kwargs: Unpack[WrapRawKwargs],
    ) -> CentralMomentsData[DataT_]: ...
    @overload
    @classmethod
    def from_raw(
        cls,
        raw: ArrayLikeArg[FloatT_],
        *,
        out: None = ...,
        dtype: None = ...,
        **kwargs: Unpack[WrapRawKwargs],
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    @classmethod
    def from_raw(
        cls,
        raw: ArrayLike,
        *,
        out: NDArray[FloatT_],
        dtype: DTypeLike = ...,
        **kwargs: Unpack[WrapRawKwargs],
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    @classmethod
    def from_raw(
        cls,
        raw: ArrayLike,
        *,
        out: None = ...,
        dtype: DTypeLikeArg[FloatT_],
        **kwargs: Unpack[WrapRawKwargs],
    ) -> CentralMomentsArray[FloatT_]: ...
    @overload
    @classmethod
    def from_raw(
        cls,
        raw: ArrayLike,
        *,
        out: NDArrayAny | None = ...,
        dtype: DTypeLike = ...,
        **kwargs: Unpack[WrapRawKwargs],
    ) -> CentralMomentsArray[Any]: ...

    @overload
    @classmethod
    def from_raw(
        cls,
        raw: Any,
        *,
        out: NDArrayAny | None = ...,
        dtype: DTypeLike = ...,
        **kwargs: Unpack[WrapRawKwargs],
    ) -> Any: ...

    @classmethod
    @docfiller.decorate
    def from_raw(
        cls,
        raw: ArrayLike | xr.DataArray | xr.Dataset,
        *,
        mom_ndim: Mom_NDim = 1,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        casting: Casting = "same_kind",
        order: ArrayOrder = None,
        keep_attrs: KeepAttrs = None,
        mom_dims: MomDims | None = None,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self | CentralMomentsABC[Any]:
        """
        Create object from raw moment data.

        raw[..., i, j] = <x**i y**j>.
        raw[..., 0, 0] = `weight`


        Parameters
        ----------
        raw : {t_array}
            Raw moment array.
        {mom_ndim}
        {out}
        {dtype}
        {casting}
        {order}
        {keep_attrs}
        {mom_dims}
        {on_missing_core_dim}
        {apply_ufunc_kwargs}

        Returns
        -------
        output : {klass}

        See Also
        --------
        to_raw
        rmom
        .convert.moments_type

        Notes
        -----
        Weights are taken from ``raw[...,0, 0]``.
        Using raw moments can result in numerical issues, especially for higher moments.  Use with care.

        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.default_rng(0)
        >>> x = rng.random(10)
        >>> raw_x = (x[:, None] ** np.arange(5)).mean(axis=0)

        >>> dx_raw = cmomy.CentralMomentsArray.from_raw(raw_x, mom_ndim=1)
        >>> print(dx_raw.mean())
        0.5505105129032412
        >>> dx_raw.cmom()
        array([ 1.    ,  0.    ,  0.1014, -0.0178,  0.02  ])

        Which is equivalent to creating raw moments from values
        >>> dx_cen = cmomy.CentralMomentsArray.from_vals(x, axis=0, mom=4)
        >>> print(dx_cen.mean())
        0.5505105129032413
        >>> dx_cen.cmom()
        array([ 1.    ,  0.    ,  0.1014, -0.0178,  0.02  ])

        But note that calculating using from_raw can lead to
        numerical issues.  For example

        >>> y = x + 10000
        >>> raw_y = (y[:, None] ** np.arange(5)).mean(axis=0)
        >>> dy_raw = cmomy.CentralMomentsArray.from_raw(raw_y, mom_ndim=1)
        >>> print(dy_raw.mean() - 10000)
        0.5505105129050207

        Note that the central moments don't match!

        >>> np.isclose(dy_raw.cmom(), dx_raw.cmom())
        array([ True,  True,  True, False, False])

        >>> dy_cen = cmomy.CentralMomentsArray.from_vals(y, axis=0, mom=4)
        >>> print(dy_cen.mean() - 10000)
        0.5505105129032017
        >>> dy_cen.cmom()  # this matches above
        array([ 1.    ,  0.    ,  0.1014, -0.0178,  0.02  ])
        """
        from cmomy import convert

        kws = cls._mom_dims_kws(mom_dims, mom_ndim, raw)
        return cls(
            obj=convert.moments_type(  # type: ignore[arg-type]
                raw,
                mom_ndim=mom_ndim,
                to="central",
                out=out,
                dtype=dtype,
                casting=casting,
                order=order,
                keep_attrs=keep_attrs,
                on_missing_core_dim=on_missing_core_dim,
                apply_ufunc_kwargs=apply_ufunc_kwargs,
                **kws,
            ),
            mom_ndim=mom_ndim,
            **kws,
            fastpath=True,
        )

"""
Attempt at a more generic wrapper class

Idea is to Wrap ndarray, xr.DataArray, and xr.Dataset objects...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic

import numpy as np
import xarray as xr

from ._lib.factory import factory_pusher, parallel_heuristic
from .core.array_utils import axes_data_reduction
from .core.docstrings import docfiller
from .core.missing import MISSING
from .core.prepare import (
    prepare_data_for_reduction,
    prepare_values_for_reduction,
)
from .core.typing import GenArrayT
from .core.utils import mom_shape_to_mom
from .core.validate import validate_floating_dtype, validate_mom_ndim
from .utils import assign_moment

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping, Sequence
    from typing import NoReturn

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from ._lib.factory import Pusher
    from .core.typing import (
        ApplyUFuncKwargs,
        ArrayOrder,
        AxisReduce,
        DataCasting,
        Groups,
        KeepAttrs,
        MissingCoreDimOptions,
        MissingType,
        Mom_NDim,
        Moments,
        MomentsStrict,
        NDArrayAny,
        ScalarT,
        SelectMoment,
    )
    from .core.typing_compat import Self


@docfiller.decorate  # noqa: PLR0904
class CentralWrapperABC(ABC, Generic[GenArrayT]):
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

    _cache: dict[str, Any]
    _obj: GenArrayT
    _mom_ndim: Mom_NDim

    def __init__(
        self,
        obj: GenArrayT,
        *,
        mom_ndim: Mom_NDim = 1,
        fastpath: bool = False,
    ) -> None:
        self._cache = {}
        self._mom_ndim = mom_ndim if fastpath else validate_mom_ndim(mom_ndim)
        self._obj = obj

        if fastpath:
            return

        # must have positive moments
        if any(m <= 0 for m in self.mom):
            msg = "moments must be positive"
            raise ValueError(msg)

        self._validate_dtype()

    @abstractmethod
    def _validate_dtype(self) -> None:
        pass

    @property
    def obj(self) -> GenArrayT:
        return self._obj

    @property
    def mom_ndim(self) -> Mom_NDim:
        return self._mom_ndim

    @property
    def mom(self) -> MomentsStrict:
        return mom_shape_to_mom(self.mom_shape)

    @property
    @abstractmethod
    def mom_shape(self) -> MomentsStrict:
        pass

    def __repr__(self) -> str:
        """Repr for class."""
        name = self.__class__.__name__
        s = f"<{name}(mom_ndim={self.mom_ndim})>\n"
        return s + repr(self._obj)

    def _repr_html_(self) -> str:  # noqa: PLW3201
        # TODO(wpk): Check out if this is best way to go...
        from .core.formatting import (
            repr_html_wrapper,  # pyright: ignore[reportUnknownVariableType]
        )

        return repr_html_wrapper(self)  # type: ignore[no-any-return,no-untyped-call]

    def __array__(  # noqa: PLW3201
        self, dtype: DTypeLike = None, copy: bool | None = None
    ) -> NDArrayAny:
        """Used by np.array(self)."""  # D401
        return np.asarray(self._obj, dtype=dtype)

    def __getitem__(self, key: Any) -> Self:
        """
        Get new object by indexing.

        Note that only objects with the same moment(s) shape are allowed.

        If you want to extract data in general, use `self.to_values()[....]`.
        """
        return self._new_like(obj=self.obj[key])

    # ** creation -------------------------------------------------------------
    def _new_like(self, obj: GenArrayT) -> Self:
        """Create new object with same properties (mom_ndim, etc) as self"""

    @abstractmethod
    @docfiller.decorate
    def new_like(
        self,
        obj: GenArrayT | None = None,
        *,
        copy: bool | None = None,
        verify: bool = False,
        dtype: DTypeLike = None,
        fastpath: bool = False,
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
            obj=self._obj.astype(dtype, **kwargs),  # type: ignore[arg-type]
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
        msg = f"{method} not implemented for {type(self._obj)}."
        raise ValueError(msg)

    def _raise_if_wrong_mom_shape(self, mom_shape: tuple[int, ...]) -> None:
        if mom_shape != self.mom_shape:
            msg = f"Moments shape={mom_shape} != {self.mom_shape=}"
            raise ValueError(msg)

    # ** Pushing routines -----------------------------------------------------
    def _pusher(self, parallel: bool | None = None, size: int | None = None) -> Pusher:
        return factory_pusher(
            mom_ndim=self._mom_ndim, parallel=parallel_heuristic(parallel, size=size)
        )

    # Low level pushers
    def _push_data_numpy(
        self,
        out: NDArray[ScalarT],
        data: ArrayLike,
        *,
        parallel: bool | None = None,
    ) -> NDArray[ScalarT]:
        data = np.asarray(data, dtype=out.dtype)
        self._pusher(parallel).data(data, out)
        return out

    def _push_datas_numpy(
        self,
        out: NDArray[ScalarT],
        datas: ArrayLike,
        *,
        axis: AxisReduce | MissingType,
        parallel: bool | None = None,
    ) -> NDArray[ScalarT]:
        axis, datas = prepare_data_for_reduction(
            data=datas,
            axis=axis,
            mom_ndim=self.mom_ndim,
            dtype=out.dtype,
        )
        axes = axes_data_reduction(mom_ndim=self.mom_ndim, axis=axis)

        self._pusher(parallel).datas(
            datas,
            out,
            axes=axes,
        )
        return out

    @staticmethod
    def _check_y(y: tuple[ArrayLike, ...], mom_ndim: int) -> None:
        if len(y) + 1 != mom_ndim:
            msg = f"Number of arrays {len(y) + 1} != {mom_ndim=}"
            raise ValueError(msg)

    def _push_val_numpy(
        self,
        out: NDArray[ScalarT],
        x: ArrayLike,
        weight: ArrayLike | None,
        *y: ArrayLike,
        parallel: bool | None = None,
    ) -> NDArray[ScalarT]:
        self._check_y(y, self.mom_ndim)

        weight = 1.0 if weight is None else weight
        x = np.asarray(x, dtype=out.dtype)
        self._pusher(parallel).val(
            out,
            x,
            *(np.asarray(a, dtype=out.dtype) for a in (weight, *y)),
        )
        return out

    def _push_vals_numpy(
        self,
        out: NDArray[ScalarT],
        x: ArrayLike,
        weight: ArrayLike | None,
        *y: ArrayLike,
        axis: AxisReduce | MissingType,
        parallel: bool | None = None,
    ) -> NDArray[ScalarT]:
        self._check_y(y, self._mom_ndim)

        weight = 1.0 if weight is None else weight
        axis, args = prepare_values_for_reduction(
            x,
            weight,
            *y,
            axis=axis,
            dtype=out.dtype,
            narrays=self._mom_ndim + 1,
        )

        self._pusher(parallel).vals(
            out,
            *args,
        )
        return out

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

    # ** Operators ------------------------------------------------------------
    def _check_other(self, b: Self) -> None:
        """Check other object."""
        if type(self) is not type(b):
            msg = f"{type(self)=} != {type(b)=}"
            raise TypeError(msg)
        if self.mom_ndim != b.mom_ndim:
            msg = f"{self.mom_ndim} != {b.mom_ndim=}"
            raise ValueError(msg)
        self._check_other_conformable(b)

    def _check_other_conformable(self, b: Self) -> None:  # noqa: ARG002, PLR6301
        msg = "Operation not supported."
        raise NotImplementedError(msg)

    def __iadd__(self, b: Self) -> Self:  # noqa: PYI034
        """Self adder."""
        self._check_other(b)
        return self.push_data(b.obj)

    def __add__(self, b: Self) -> Self:
        """Add objects to new object."""
        self._check_other(b)
        return self.copy().push_data(b.obj)

    def __isub__(self, b: Self) -> Self:  # noqa: PYI034
        """Inplace subtraction."""
        # NOTE: consider implementint push_data_scale routine to make this cleaner
        self._check_other(b)
        if not np.all(self.weight() >= b.weight()):
            raise ValueError
        b = b.assign_moment(weight=-b.weight(), copy=True)
        return self.push_data(b.obj)

    def __sub__(self, b: Self) -> Self:
        """Subtract objects."""
        self._check_other(b)
        if not np.all(self.weight() >= b.weight()):
            raise ValueError
        new = b.assign_moment(weight=-b.weight(), copy=True)
        return new.push_data(self.obj)

    def __mul__(self, scale: float) -> Self:
        """New object with weight scaled by scale."""  # D401
        return self.assign_moment(weight=self.weight() * scale, copy=True)

    def __imul__(self, scale: float) -> Self:  # noqa: PYI034
        """Inplace multiply."""
        # do this to make sure things work with dask...
        self._obj = self.assign_moment(weight=self.weight() * scale, copy=False)._obj
        return self

    # ** Apply function to obj ------------------------------------------------
    def pipe(
        self,
        func_or_method: Callable[..., Any] | str,
        *args: Any,
        _reorder: bool = True,
        _copy: bool | None = None,
        _verify: bool = False,
        **kwargs: Any,
    ) -> Self:
        """
        Apply `func_or_method` to underlying data and wrap results in
        class:`.xCentralMoments` object.

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
            values = values.transpose(..., *self.mom_dims)  # pyright: ignore[reportUnknownMemberType,reportGeneralTypeIssues, reportAttributeAccessIssue]

        return self.new_like(
            obj=values,
            copy=_copy,
            verify=_verify,
        )

    # ** Interface to .utils --------------------------------------------------
    @docfiller.decorate
    def moveaxis(
        self,
        axis: int | tuple[int, ...],
        dest: int | tuple[int, ...],
        **kwargs: Any,
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
        from .utils import moveaxis

        if isinstance(self._obj, xr.Dataset):
            self._raise_not_implemented()

        obj = moveaxis(
            self._obj,
            axis=axis,
            dest=dest,
            **kwargs,
            mom_ndim=self._mom_ndim,
        )

        return self.new_like(
            obj=obj,
        )

    def select_moment(
        self,
        name: SelectMoment,
        *,
        squeeze: bool = True,
        dim_combined: str = "variable",
        coords_combined: str | Sequence[Hashable] | None = None,
        keep_attrs: KeepAttrs = None,
        on_missing_core_dim: MissingCoreDimOptions = "copy",
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> GenArrayT:
        """
        Select specific moments.

        See Also
        --------
        .utils.select_moment
        """
        from .utils import select_moment

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
        **kwargs: Any,
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

        Returns
        -------
        output : object
            Same type as ``self`` with updated data.

        Returns
        -------
        output : object
            Same type as ``self``
        """
        obj = assign_moment(
            data=self._obj,
            moment=moment,
            mom_ndim=self._mom_ndim,
            squeeze=squeeze,
            copy=copy,
            mom_dims=getattr(self, "mom_dims", None),
            **kwargs,
        )
        return self.new_like(obj=obj, fastpath=True)

    # ** Interface to .convert ------------------------------------------------
    @docfiller.decorate
    def cumulative(
        self, axis: AxisReduce | MissingType = MISSING, **kwargs: Any
    ) -> GenArrayT:
        """
        Convert to cumulative moments.

        Parameters
        ----------
        **kwargs
            Extra arguments to :func:`.convert.cumulative`

        Returns
        -------
        output : {t_array}
        """
        from cmomy.convert import cumulative

        return cumulative(
            self._obj,
            axis=axis,
            mom_ndim=self._mom_ndim,
            **kwargs,
        )

    @docfiller.decorate
    def moments_to_comoments(self, *, mom: tuple[int, int], **kwargs: Any) -> Self:
        """
        Convert moments (mom_ndim=1) to comoments (mom_ndim=2).

        Parameters
        ----------
        {mom_moments_to_comoments}

        Return
        ------
        object
            New object with ``mom_ndim=2``.

        See Also
        --------
        .convert.moments_to_comoments
        """
        self._raise_if_not_mom_ndim_1()
        from cmomy import convert

        return type(self)(
            convert.moments_to_comoments(  # pyright: ignore[reportArgumentType]
                self._obj,
                mom=mom,
                mom_dims=getattr(self, "mom_dims", None),
                **kwargs,
            ),
            mom_ndim=2,
        )

    # ** Access to underlying statistics ------------------------------------------
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
        """Standard deviation."""  # D401
        raise NotImplementedError

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
        from .convert import moments_type

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

    # * Interface to .resample ------------------------------------------------
    @docfiller.decorate
    def resample_and_reduce(
        self,
        *,
        axis: AxisReduce | MissingType = -1,
        freq: ArrayLike | None = None,
        nrep: int | None = None,
        rng: np.random.Generator | None = None,
        parallel: bool | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Bootstrap resample and reduce.

        Parameters
        ----------
        {axis_data_and_dim}
        {freq}
        {nrep_optional}
        {rng}
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
        ~.resample.randsamp_freq : random frequency sample
        ~.resample.freq_to_indices : convert frequency sample to index sample
        ~.resample.indices_to_freq : convert index sample to frequency sample
        ~.resample.resample_data : method to perform resampling
        """
        from .resample import resample_data

        obj = resample_data(  # pyright: ignore[reportUnknownVariableType]  # Not sure what's going on here...
            self._obj,
            freq=freq,
            nrep=nrep,
            rng=rng,
            mom_ndim=self._mom_ndim,
            axis=axis,
            parallel=parallel,
            **kwargs,
        )

        return self._new_like(obj=obj)  # pyright: ignore[reportUnknownArgumentType]

    @docfiller.decorate
    def jackknife_and_reduce(
        self,
        *,
        axis: AxisReduce | MissingType = -1,
        parallel: bool | None = None,
        data_reduced: GenArrayT | None = None,
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
        from .resample import jackknife_data

        return self._new_like(
            obj=jackknife_data(  # pyright: ignore[reportCallIssue, reportUnknownArgumentType]
                self._obj,  # pyright: ignore[reportArgumentType]
                mom_ndim=self._mom_ndim,
                axis=axis,
                data_reduced=data_reduced,  # pyright: ignore[reportArgumentType]
                parallel=parallel,
                **kwargs,
            )
        )

    # ** Interface to .reduction ----------------------------------------------
    @abstractmethod
    @docfiller.decorate
    def reduce(
        self,
        *,
        by: Groups | None = None,
        axis: AxisReduce = -1,
        keepdims: bool = False,
        move_axis_to_end: bool = False,
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

    # ** Constructors ---------------------------------------------------------
    @classmethod
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
        raise NotImplementedError

    @classmethod
    @abstractmethod
    @docfiller.decorate
    def from_vals(
        cls,
        x: Any,
        *y: Any,
        mom: Moments,
        axis: AxisReduce = -1,
        weight: Any = None,
        keepdims: bool = False,
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
        x: Any,
        *y: Any,
        mom: Moments,
        freq: Any = None,
        nrep: int | None = None,
        rng: np.random.Generator | None = None,
        weight: Any = None,
        axis: AxisReduce = -1,
        move_axis_to_end: bool = True,
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
        {nrep}
        {rng}
        {weight}
        {axis_and_dim}
        {move_axis_to_end}
        {full_output}
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
        raw: GenArrayT,
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
            obj=convert.moments_type(raw, mom_ndim=mom_ndim, to="central"),  # pyright: ignore[reportArgumentType]
            mom_ndim=mom_ndim,
        )

"""
Attempt at a more generic wrapper class

Idea is to Wrap ndarray, xr.DataArray, and xr.Dataset objects...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic

import xarray as xr

from cmomy.core.docstrings import docfiller
from cmomy.core.typing import GenArrayT
from cmomy.core.validate import validate_floating_dtype, validate_mom_ndim
from cmomy.utils import assign_moment

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping, Sequence
    from typing import NoReturn

    import numpy as np
    from numpy.typing import ArrayLike, DTypeLike

    from cmomy.core.typing import (
        ApplyUFuncKwargs,
        ArrayOrder,
        AxisReduce,
        DataCasting,
        Groups,
        KeepAttrs,
        MissingCoreDimOptions,
        MissingType,
        Mom_NDim,
        SelectMoment,
    )
    from cmomy.core.typing_compat import Self


@docfiller.decorate
class CentralWrapperABC(ABC, Generic[GenArrayT]):
    """Generic wrapper class to add functionality to array objects..."""

    _cache: dict[str, Any]
    _obj: GenArrayT
    _mom_ndim: Mom_NDim

    def __init__(
        self,
        obj: GenArrayT,
        mom_ndim: Mom_NDim = 1,
        copy: bool | None = None,
        fastpath: bool = False,
    ) -> None:
        self._cache = {}
        self._mom_ndim = mom_ndim if fastpath else validate_mom_ndim(mom_ndim)
        self.set_obj(obj, copy=copy, fastpath=fastpath)

    @abstractmethod
    def set_obj(self, obj: GenArrayT, copy: bool | None, fastpath: bool) -> None:
        pass

    @property
    def obj(self) -> GenArrayT:
        return self._obj

    @property
    def mom_ndim(self) -> Mom_NDim:
        return self._mom_ndim

    def __repr__(self) -> str:
        """Repr for class."""
        name = self.__class__.__name__
        s = f"<{name}(mom_ndim={self.mom_ndim})>\n"
        return s + repr(self.obj)

    def _repr_html_(self) -> str:  # noqa: PLW3201
        # TODO(wpk): Check out if this is best way to go...
        from cmomy.core.formatting import (
            repr_html_wrapper,  # pyright: ignore[reportUnknownVariableType]
        )

        return repr_html_wrapper(self)  # type: ignore[no-any-return,no-untyped-call]

    # ** creation -------------------------------------------------------------
    @abstractmethod
    @docfiller.decorate
    def new_like(
        self,
        obj: GenArrayT | None = None,
        *,
        copy: bool | None = None,
        order: ArrayOrder | None = None,
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
            obj=self.obj.astype(dtype, **kwargs),  # type: ignore[arg-type]
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
            obj=self.obj,
            verify=False,
            copy=True,
        )

    # ** Utils ----------------------------------------------------------------
    def _raise_if_not_mom_ndim_1(self) -> None:
        if self._mom_ndim != 1:
            msg = "Only implemented for `mom_ndim==1`"
            raise ValueError(msg)

    def _raise_not_implemented(self, method: str = "method") -> NoReturn:
        msg = f"{method} not implemented for {type(self.obj)}."
        raise ValueError(msg)

    # ** Apply function to obj ------------------------------------------------
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
            values = getattr(self.obj, func_or_method)(*args, **kwargs)
        else:
            values = func_or_method(self.obj, *args, **kwargs)

        if _reorder and hasattr(self, "mom_dims"):
            values = values.transpose(..., *self.mom_dims)  # pyright: ignore[reportUnknownMemberType,reportGeneralTypeIssues, reportAttributeAccessIssue]

        return self.new_like(
            obj=values,
            copy=_copy,
            order=_order,
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
        from cmomy.utils import moveaxis

        if isinstance(self.obj, xr.Dataset):
            self._raise_not_implemented()

        obj = moveaxis(
            self.obj,
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
        from cmomy.utils import select_moment

        return select_moment(
            self.obj,
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
            data=self.obj,
            moment=moment,
            mom_ndim=self._mom_ndim,
            squeeze=squeeze,
            copy=copy,
            **kwargs,
        )
        return self.new_like(obj=obj, fastpath=True)

    # ** Interface to .convert ------------------------------------------------
    @abstractmethod
    @docfiller.decorate
    def cumulative(self, **kwargs: Any) -> GenArrayT:
        """Convert to cumulative moments."""

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
                self.obj,
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
            self.obj,
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

        out = moments_type(self.obj, mom_ndim=self._mom_ndim, to="raw")
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
        ~cmomy.resample.randsamp_freq : random frequency sample
        ~cmomy.resample.freq_to_indices : convert frequency sample to index sample
        ~cmomy.resample.indices_to_freq : convert index sample to frequency sample
        ~cmomy.resample.resample_data : method to perform resampling
        """
        from cmomy.resample import resample_data

        obj = resample_data(  # pyright: ignore[reportUnknownVariableType]  # Not sure what's going on here...
            self.obj,
            freq=freq,
            nrep=nrep,
            rng=rng,
            mom_ndim=self._mom_ndim,
            axis=axis,
            parallel=parallel,
            **kwargs,
        )

        return self.new_like(obj=obj)  # pyright: ignore[reportUnknownArgumentType]

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
        from cmomy.resample import jackknife_data

        return self.new_like(
            obj=jackknife_data(  # pyright: ignore[reportCallIssue, reportUnknownArgumentType]
                self.obj,  # pyright: ignore[reportArgumentType]
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

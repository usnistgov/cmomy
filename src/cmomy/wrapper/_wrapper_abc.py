"""
Attempt at a more generic wrapper class

Idea is to Wrap ndarray, xr.DataArray, and xr.Dataset objects...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, cast

import numpy as np

from cmomy.core.docstrings import docfiller
from cmomy.core.missing import MISSING
from cmomy.core.moment_params import factory_mom_params
from cmomy.core.typing_compat import TypeVar, override
from cmomy.core.validate import (
    is_dataset,
    raise_if_wrong_value,
    validate_floating_dtype,
)
from cmomy.factory import factory_pusher, parallel_heuristic
from cmomy.utils import assign_moment

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping, Sequence
    from typing import (
        NoReturn,
    )

    import xarray as xr
    from numpy.typing import ArrayLike, DTypeLike, NDArray  # noqa: F401

    from cmomy.core._typing_kwargs import (
        ApplyUFuncKwargs,
    )
    from cmomy.core.moment_params import MomParamsArray, MomParamsXArray  # noqa: F401
    from cmomy.core.typing import (
        ArrayOrderCF,
        ArrayOrderKACF,
        AxisReduce,
        BlockByModes,
        Casting,
        DimsReduce,
        KeepAttrs,
        MissingType,
        MomAxes,
        MomDims,
        Moments,
        MomentsStrict,
        MomNDim,
        NDArrayAny,
        NDArrayInt,
        SelectMoment,
    )
    from cmomy.core.typing_compat import Self
    from cmomy.factory import Pusher
    from cmomy.resample.typing import SamplerType

#: MomParams type variable
MomParamsT = TypeVar("MomParamsT", "MomParamsArray", "MomParamsXArray")
#: Generic array type variable
GenArrayT = TypeVar("GenArrayT", "NDArray[Any]", "xr.DataArray", "xr.Dataset")


@docfiller.decorate  # noqa: PLR0904
class CentralMomentsABC(ABC, Generic[GenArrayT, MomParamsT]):
    r"""
    Wrapper to calculate central moments.

    Parameters
    ----------
    obj : {t_array}
        Central moments array.
    {mom_params}
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


    See Also
    --------
    cmomy.wrap
    cmomy.wrap_reduce_vals
    cmomy.wrap_resample_vals
    cmomy.wrap_raw
    """

    __slots__ = ("_mom_params", "_obj")

    _obj: GenArrayT
    _mom_params: MomParamsT

    def __init__(
        self,
        obj: GenArrayT,
        *,
        mom_params: MomParamsT,
        fastpath: bool = False,
    ) -> None:
        self._mom_params = mom_params
        self._obj = obj

        if fastpath:
            return

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
    def mom_params(self) -> MomParamsT:
        """Moments parameters object."""
        return self._mom_params

    @property
    def mom_ndim(self) -> MomNDim:
        """Number of moment dimensions."""
        return self._mom_params.ndim

    @property
    def mom_shape(self) -> MomentsStrict:
        """Shape of moments dimensions."""
        return self._mom_params.get_mom_shape(self._obj)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

    @property
    def mom(self) -> MomentsStrict:
        """Moments tuple."""
        return self._mom_params.get_mom(self._obj)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

    @property
    def dtype(self) -> np.dtype[Any]:
        """DType of wrapped object."""
        if is_dataset(self._obj):
            self._raise_notimplemented_for_dataset()
        return self._obj.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of wrapped object."""
        if is_dataset(self._obj):
            self._raise_notimplemented_for_dataset()
        return self._obj.shape  # type: ignore[no-any-return,unused-ignore]

    @property
    def val_shape(self) -> tuple[int, ...]:
        """Shape of values dimensions."""
        if is_dataset(self._obj):
            self._raise_notimplemented_for_dataset()
        return self._mom_params.get_val_shape(self._obj)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

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
        return self.ndim - self.mom_ndim

    @override
    def __repr__(self) -> str:
        """Repr for class."""
        name = self.__class__.__name__
        s = f"<{name}(mom_ndim={self.mom_ndim})>\n"
        return s + repr(self._obj)

    def _repr_html_(self) -> str:  # noqa: PLW3201
        from cmomy.core.formatting import (
            repr_html_wrapper,  # pyright: ignore[reportUnknownVariableType]
        )

        return repr_html_wrapper(self)  # type: ignore[no-any-return,no-untyped-call]  # pyright: ignore[reportUnknownVariableType]

    def __array__(  # noqa: PLW3201
        self, dtype: DTypeLike = None, copy: bool | None = None
    ) -> NDArrayAny:
        """Used by np.array(self)."""  # D401
        return np.asarray(self._obj, dtype=dtype)

    def to_numpy(self) -> NDArrayAny:
        """Coerce wrapped data to :class:`~numpy.ndarray` if possible."""
        return np.asarray(self)

    # ** Create/copy/new ------------------------------------------------------
    def _new_like(self, obj: GenArrayT) -> Self:
        """Create new object with same properties (mom_ndim, etc) as self with no checks and fastpath"""
        return type(self)(
            obj=obj,
            mom_params=self._mom_params,
            fastpath=True,
        )

    @abstractmethod
    @docfiller.decorate
    def new_like(
        self,
        obj: GenArrayT | None = None,
        *,
        verify: bool = False,
        copy: bool | None = None,
        dtype: DTypeLike = None,
        order: ArrayOrderKACF = None,
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
            New {klass} object with data set to zero.
        """

    @docfiller.decorate
    def astype(
        self,
        dtype: DTypeLike,
        *,
        order: ArrayOrderKACF = None,
        casting: Casting | None = None,
        subok: bool | None = None,
        copy: bool = False,
    ) -> Self:
        """
        Underlying data cast to specified type

        Parameters
        ----------
        dtype : str or dtype
            Type code of data-type to cast the array data.  Note that a value of None will
            cast to ``np.float64``.  This is the same behavior as :func:`~numpy.asarray`.
        {order}
        {casting}

        subok : bool, optional
            If True, then sub-classes will be passed-through, otherwise the
            returned array will be forced to be a base-class array.
        copy : bool, optional
            By default, astype always returns a newly allocated array. If this
            is set to False and the `dtype` requirement is satisfied, the input
            array is returned instead of a copy.


        Notes
        -----
        Only ``numpy.float32`` and ``numpy.float64`` dtype are supported.


        See Also
        --------
        numpy.ndarray.astype
        """
        dtype = validate_floating_dtype(dtype)
        kwargs = {"order": order, "casting": casting, "subok": subok, "copy": copy}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        return self._new_like(
            obj=self._obj.astype(dtype, **kwargs),  # type: ignore[arg-type]  # pyright: ignore[reportUnknownMemberType, reportCallIssue, reportArgumentType]
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
        if self.mom_ndim != 1:
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

    # ** Pushing --------------------------------------------------------------
    def _pusher(self, parallel: bool | None = None, size: int | None = None) -> Pusher:
        return factory_pusher(
            mom_ndim=self.mom_ndim, parallel=parallel_heuristic(parallel, size=size)
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
        mom_axes: MomAxes | None = None,
        casting: Casting = "same_kind",
        parallel: bool | None = None,
    ) -> Self:
        """
        Push and reduce multiple average central moments.

        Parameters
        ----------
        datas : array-like or {t_array}
            Collection of accumulation arrays to push onto ``self``.
        {mom_axes}
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
    def pipe(  # pylint: disable=useless-type-doc,useless-param-doc
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
        Use leading underscore for (`_copy`, etc) to avoid possible name
        clashes with ``func_or_method``.


        """
        values = (
            getattr(self._obj, func_or_method)(*args, **kwargs)
            if isinstance(func_or_method, str)
            else func_or_method(self._obj, *args, **kwargs)
        )

        if _reorder and hasattr(self, "mom_dims"):
            values = values.transpose(..., *self.mom_dims)  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]  # pylint: disable=no-member

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
        axes_to_end: bool = False,
    ) -> Self:
        """
        Generalized moveaxis

        Parameters
        ----------
        axis : int or sequence of int
            Original positions of axes to move.
        dest : int or sequence of int
            Destination positions for each original axes.
        dim: str or sequence of str
            Original dimensions.
        dest_dim: str or sequence of str
            Destination position of dimensions.

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
            mom_params=self._mom_params,
            axes_to_end=axes_to_end,
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
            mom_params=self._mom_params,
            dim_combined=dim_combined,
            coords_combined=coords_combined,
            squeeze=squeeze,
            keep_attrs=keep_attrs,
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
            mom_ndim=None,
            mom_axes=None,
            mom_dims=None,
            mom_params=self._mom_params,
            squeeze=squeeze,
            copy=copy,
            dim_combined=dim_combined,
            keep_attrs=keep_attrs,
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
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        casting: Casting = "same_kind",
        order: ArrayOrderKACF = None,
        parallel: bool | None = None,
        axes_to_end: bool = False,
        keep_attrs: KeepAttrs = None,
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> GenArrayT:
        """
        Convert to cumulative moments.

        Parameters
        ----------
        {axis_data_and_dim}
        {axes_to_end}
        {out}
        {dtype}
        {casting}
        {order}
        {parallel}
        {keep_attrs}
        {mom_dims_data}
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
            mom_params=self._mom_params,
            inverse=False,
            axes_to_end=axes_to_end,
            out=out,
            dtype=dtype,
            casting=casting,
            order=order,
            parallel=parallel,
            keep_attrs=keep_attrs,
            apply_ufunc_kwargs=apply_ufunc_kwargs,
        )

    @docfiller.decorate
    def moments_to_comoments(
        self,
        *,
        mom: tuple[int, int],
        mom_dims_out: MomDims | None = None,
        dtype: DTypeLike = None,
        order: ArrayOrderCF = None,
        keep_attrs: KeepAttrs = None,
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self:
        """
        Convert moments (mom_ndim=1) to comoments (mom_ndim=2).

        Parameters
        ----------
        {mom_moments_to_comoments}
        mom_dims_out : tuple of str
            Moments dimensions for output (``mom_ndim=2``) data.  Defaults to ``("mom_0", "mom_1")``.
        {dtype}
        {order_cf}
        {keep_attrs}
        {apply_ufunc_kwargs}

        Return
        ------
        output : {klass}
            Same type as ``self`` with ``mom_ndim=2``.

        See Also
        --------
        .convert.moments_to_comoments
        """  # noqa: DOC102
        self._raise_if_not_mom_ndim_1()
        from cmomy import convert

        mom_params_out = factory_mom_params(target=self._obj, ndim=2, dims=mom_dims_out)

        return type(self)(
            convert.moments_to_comoments(
                self._obj,
                mom=mom,
                mom_params=self._mom_params,
                mom_dims_out=mom_dims_out,
                dtype=dtype,
                order=order,
                keep_attrs=keep_attrs,
                apply_ufunc_kwargs=apply_ufunc_kwargs,
            ),
            mom_params=mom_params_out,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        )

    # *** .resample -----------------------------------------------------------
    @docfiller.decorate
    def resample_and_reduce(
        self,
        *,
        sampler: SamplerType,
        axis: AxisReduce | MissingType = MISSING,
        dim: DimsReduce | MissingType = MISSING,
        rep_dim: str = "rep",
        axes_to_end: bool = False,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        casting: Casting = "same_kind",
        order: ArrayOrderKACF = None,
        parallel: bool | None = None,
        keep_attrs: KeepAttrs = None,
        # dask specific...
        apply_ufunc_kwargs: ApplyUFuncKwargs | None = None,
    ) -> Self:
        """
        Bootstrap resample and reduce.

        Parameters
        ----------
        {sampler}
        {axis_data_and_dim}
        {rep_dim}
        {axes_to_end}
        {out}
        {dtype}
        {casting}
        {order}
        {parallel}
        {keep_attrs}
        {apply_ufunc_kwargs}

        Returns
        -------
        output : object
            Instance of calling class. Note that new object will have
            ``(...,shape[axis-1], nrep, shape[axis+1], ...)``,
            where ``nrep`` is the number of replicates.

        See Also
        --------
        reduce
        ~.resample.factory_sampler
        ~.resample.resample_data : method to perform resampling

        Examples
        --------
        >>> import cmomy
        >>> rng = cmomy.default_rng(0)
        >>> da = cmomy.wrap_reduce_vals(
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

        >>> sampler = cmomy.resample.factory_sampler(data=da.obj, dim="rec", nrep=5)
        >>> da_resamp = da.resample_and_reduce(
        ...     dim="rec",
        ...     sampler=sampler,
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

        """
        from cmomy.resample import resample_data

        return self._new_like(
            obj=resample_data(
                self._obj,
                mom_params=self._mom_params,
                sampler=sampler,
                axis=axis,
                dim=dim,
                rep_dim=rep_dim,
                axes_to_end=axes_to_end,
                dtype=dtype,
                out=out,
                casting=casting,
                order=order,
                parallel=parallel,
                keep_attrs=keep_attrs,
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
        axes_to_end: bool = False,
        out: NDArrayAny | None = None,
        dtype: DTypeLike = None,
        casting: Casting = "same_kind",
        order: ArrayOrderKACF = None,
        parallel: bool | None = None,
        keep_attrs: KeepAttrs = None,
        # dask specific...
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
        {axes_to_end}
        {out}
        {dtype}
        {casting}
        {order}
        {parallel}
        {keep_attrs}
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
            mom_params=self._mom_params,
            axis=axis,
            dim=dim,
            data_reduced=data_reduced,  # pyright: ignore[reportArgumentType]
            rep_dim=rep_dim,
            axes_to_end=axes_to_end,
            out=out,
            dtype=dtype,
            casting=casting,
            order=order,
            parallel=parallel,
            keep_attrs=keep_attrs,
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
        from cmomy.grouped import block_by
        from cmomy.resample import select_ndat

        return block_by(
            ndat=select_ndat(
                self._obj,
                axis=axis,
                dim=dim,
                mom_params=self._mom_params,
            ),
            block=block,
            mode=mode,
        )

    @abstractmethod
    @docfiller.decorate
    def reduce(  # pylint: disable=differing-param-doc,differing-type-doc
        self,
    ) -> Self:
        """
        Create new object reduce along axis.

        Parameters
        ----------
        {axis_data_and_dim}
        {by}
        block : int, optional
            If specified, perform block average reduction with blocks
            of this size.  Negative values are transformed to all data.
        {axes_to_end}
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
        ~.grouped.reduce_data_grouped
        ~.grouped.reduce_data_indexed
        ~.grouped.block_by
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
            mom_params=self._mom_params,
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
        .wrap_raw
        .convert.moments_type
        """
        from cmomy.convert import moments_type

        out = moments_type(self._obj, mom_params=self._mom_params, to="raw")
        if weight is not None:
            out = assign_moment(
                out, weight=weight, mom_params=self._mom_params, copy=False
            )
        return out  # type: ignore[no-any-return]

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

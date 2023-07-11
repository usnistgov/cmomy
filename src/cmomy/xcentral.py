"""Thin wrapper around central routines with xarray support."""


from __future__ import annotations

from typing import (  # TYPE_CHECKING,
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Literal,
    Mapping,
    Sequence,
    cast,
    no_type_check,
)

# overload
# from warnings import warn
from module_utilities import cached

from . import convert
from ._lazy_imports import np, xr
from .abstract_central import CentralMomentsABC
from .central import CentralMoments
from .docstrings import docfiller_xcentral as docfiller
from .docstrings import factory_docfiller_inherit_from_parent
from .utils import _shape_reduce

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from ._typing import Dims, Mom_NDim, MomDims, Moments, T_CentralMoments


# * Utilities
def _select_axis_dim(
    dims: tuple[Hashable, ...],
    axis: int | None = None,
    dim: Hashable | None = None,
    default_axis: int | None = None,
    default_dim: Hashable | None = None,
) -> tuple[int, Hashable]:
    """Produce axis/dim from input."""

    if axis is None and dim is None:
        if default_axis is None and default_dim is None:
            raise ValueError("must specify axis or dim")
        elif default_axis is not None and default_dim is not None:
            raise ValueError("can only specify one of default_axis or default_dim")
        elif default_axis:
            axis = default_axis
        else:
            dim = default_dim

    elif axis is not None and dim is not None:
        raise ValueError("can only specify axis or dim")

    if dim is not None:
        if dim in dims:
            axis = dims.index(dim)
        else:
            raise ValueError(f"did not find '{dim}' in {dims}")
    elif axis is not None:
        if isinstance(axis, str):
            raise ValueError(
                f"Using string value for axis is deprecated.  Please use `dim` option instead.  Passed {axis} of type {type(axis)}"
            )
        else:
            dim = dims[axis]
    else:
        raise ValueError(f"unknown dim {dim} and axis {axis}")

    return axis, dim


def _move_mom_dims_to_end(
    x: xr.DataArray, mom_dims: MomDims, mom_ndim: Mom_NDim | None = None
) -> xr.DataArray:
    if mom_dims is not None:
        if isinstance(mom_dims, str):
            mom_dims = (mom_dims,)
        else:
            mom_dims = tuple(mom_dims)  # type: ignore

        if mom_ndim is not None and len(mom_dims) != mom_ndim:
            raise ValueError(
                "len(mom_dims)={len(mom_dims)} not equal to mom_ndim={mom_ndim}"
            )

        order = (...,) + mom_dims
        x = x.transpose(*order)

    return x


# * xcentral moments/comoments
def _xcentral_moments(
    vals: xr.DataArray,
    mom: Moments,
    w: xr.DataArray | None = None,
    axis: int | None = None,
    dim: Hashable | None = None,
    last: bool = True,
    mom_dims: MomDims | None = None,
) -> xr.DataArray:
    x = vals
    assert isinstance(x, xr.DataArray)

    if isinstance(mom, tuple):
        mom = mom[0]

    if mom_dims is None:
        mom_dims = ("mom_0",)
    elif isinstance(mom_dims, str):
        mom_dims = (mom_dims,)
    assert len(mom_dims) == 1  # type: ignore

    if w is None:
        w = xr.ones_like(x)
    else:
        w = xr.DataArray(w).broadcast_like(x)

        axis, dim = _select_axis_dim(dims=x.dims, axis=axis, dim=dim, default_axis=0)

    if TYPE_CHECKING:
        mom_dims = cast(tuple[Hashable], mom_dims)
        dim = cast(str, dim)

    wsum = w.sum(dim=dim)
    wsum_inv = 1.0 / wsum

    xave = xr.dot(w, x, dims=dim) * wsum_inv

    p = xr.DataArray(np.arange(0, mom + 1), dims=mom_dims)
    dx = (x - xave) ** p
    out = xr.dot(w, dx, dims=dim) * wsum_inv

    out.loc[{mom_dims[0]: 0}] = wsum
    out.loc[{mom_dims[0]: 1}] = xave

    if last:
        out = out.transpose(..., *mom_dims)
    return cast(xr.DataArray, out)


def _xcentral_comoments(
    vals: tuple[xr.DataArray, xr.DataArray],
    mom: Moments,
    w: xr.DataArray | None = None,
    axis: int | None = None,
    dim: Hashable | None = None,
    last: bool = True,
    broadcast: bool = False,
    mom_dims: tuple[Hashable, Hashable] | None = None,
):
    """Calculate central co-mom (covariance, etc) along axis."""

    if isinstance(mom, int):
        mom = (mom,) * 2
    else:
        mom = tuple(mom)  # type: ignore

    assert len(mom) == 2

    assert isinstance(vals, tuple) and len(vals) == 2
    x, y = vals

    assert isinstance(x, xr.DataArray)

    if w is None:
        w = xr.ones_like(x)
    else:
        w = xr.DataArray(w).broadcast_like(x)

    if broadcast:
        y = xr.DataArray(y).broadcast_like(x)
    else:
        assert isinstance(y, xr.DataArray)

        y = y.transpose(*x.dims)
        assert y.shape == x.shape
        assert x.dims == y.dims

    axis, dim = _select_axis_dim(dims=x.dims, axis=axis, dim=dim, default_axis=0)

    if mom_dims is None:
        mom_dims = ("mom_0", "mom_1")

    assert len(mom_dims) == 2

    if TYPE_CHECKING:
        dim = cast(str, dim)

    wsum = w.sum(dim=dim)
    wsum_inv = 1.0 / wsum

    xy = (x, y)

    xave = [xr.dot(w, xx, dims=dim) * wsum_inv for xx in xy]

    p = [
        xr.DataArray(np.arange(0, mom + 1), dims=dim) for mom, dim in zip(mom, mom_dims)
    ]

    dx = [(xx - xxave) ** pp for xx, xxave, pp in zip(xy, xave, p)]

    out = xr.dot(w, dx[0], dx[1], dims=dim) * wsum_inv

    out.loc[{mom_dims[0]: 0, mom_dims[1]: 0}] = wsum
    out.loc[{mom_dims[0]: 1, mom_dims[1]: 0}] = xave[0]
    out.loc[{mom_dims[0]: 0, mom_dims[1]: 1}] = xave[1]

    if last:
        out = out.transpose(..., *mom_dims)
    return out


@docfiller.decorate
def xcentral_moments(
    x: xr.DataArray | tuple[xr.DataArray, xr.DataArray],
    mom: Moments,
    w: xr.DataArray | None = None,
    axis: int | None = None,
    dim: Hashable | None = None,
    last: bool = True,
    mom_dims: MomDims | None = None,
    broadcast: bool = False,
) -> xr.DataArray:
    """
    Calculate central mom along axis.

    Parameters
    ----------
    x : DataArray or tuple of DataArray
        input data
    {mom}
    w : array-like, optional
        if passed, should be able to broadcast to `x`. An exception is if
        w is a 1d array with len(w) == x.shape[axis]. In this case,
        w will be reshaped and broadcast against x
    {axis}
    {dim}
    last : bool, default=True
        if True, put mom as last dimension.
        Otherwise, mom will be in first dimension
    {dtype}
    {mom_dims}
    {broadcast}

    Returns
    -------
    output : DataArray
        array of shape shape + (mom,) or (mom,) + shape depending on
        value of `last`, where `shape` is the shape of `x` with axis removed,
        i.e., shape=x.shape[:axis] + x.shape[axis+1:]. Assuming `last is True`,
        output[...,0] is the total weight (or count), output[...,1] is the mean
        value of x, output[...,n] with n>1 is the nth central moment
    """

    if isinstance(mom, int):
        mom = (mom,)

    kws = dict(vals=x, mom=mom, w=w, axis=axis, dim=dim, last=last, mom_dims=mom_dims)
    if len(mom) == 1:
        out = _xcentral_moments(**kws)  # type: ignore
    else:
        kws["broadcast"] = broadcast
        out = _xcentral_comoments(**kws)  # type: ignore

    return cast(xr.DataArray, out)


# --- * xCentralMoments-----------------------------------------------------------------
from .docstrings import factory_docfiller_from_parent, factory_docinherit_from_parent

docfiller_abc = factory_docfiller_from_parent(
    docfiller=docfiller, cls=CentralMomentsABC
)
docinherit_abc = factory_docinherit_from_parent(cls=CentralMomentsABC)
docfiller_inherit_abc = factory_docfiller_inherit_from_parent(
    cls=CentralMomentsABC, docfiller=docfiller
)


@docfiller(CentralMomentsABC)
class xCentralMoments(CentralMomentsABC[xr.DataArray]):
    """
    Notes
    -----
    Most methods are wrapped to accept :class:`xarray.DataArray` object.
    """

    __slots__ = "_xdata"

    # Override __new__ for signature
    # def __new__(cls, data: xr.DataArray, mom_ndim: Literal[1, 2] = 1):  # noqa: D102
    #     return super().__new__(cls, data=data, mom_ndim=mom_ndim)

    def __init__(self, data: xr.DataArray, mom_ndim: Mom_NDim = 1) -> None:
        if not isinstance(data, xr.DataArray):
            raise ValueError(
                "data must be a xarray.DataArray. "
                "See xCentralMoments.from_data for wrapping numpy arrays"
            )

        self._xdata = data

        if mom_ndim not in (1, 2):
            raise ValueError(
                "mom_ndim must be either 1 (for central moments)"
                "or 2 (for central comoments)"
            )
        self._mom_ndim = mom_ndim

        if data.ndim < self.mom_ndim:
            raise ValueError("not enough dimensions in data")

        # TODO: data.data or data.values?
        self._data = data.data
        self._data_flat = self._data.reshape(self.shape_flat)

        if any(m <= 0 for m in self.mom):
            raise ValueError("moments must be positive")

    # ** xarray attributes
    @property
    def values(self):
        """Underlying data."""
        return self._xdata

    @property
    def attrs(self):
        """Attributes of values."""
        return self._xdata.attrs

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """Dimensions of values."""
        return self._xdata.dims

    @property
    def coords(self):
        """Coordinates of values."""
        return self._xdata.coords

    @property
    def name(self):
        """Name of values."""
        return self._xdata.name

    @property
    def indexes(self):
        """Indexes of values."""
        return self._xdata.indexes

    @property
    def sizes(self):
        """Sizes of values."""
        return self._xdata.sizes

    @property
    def val_dims(self) -> tuple[Hashable, ...]:
        """Names of value dimensions."""
        return self.dims[: -self.mom_ndim]

    @property
    def mom_dims(self) -> tuple[Hashable, ...]:
        """Names of moment dimensions."""
        return self.dims[-self.mom_ndim :]

    # ** top level creation/copy/new
    @cached.prop
    def _template_val(self) -> xr.DataArray:
        """Template for values part of data."""
        return self._xdata[self._weight_index]

    def _wrap_like(self, x: np.ndarray, *args, **kwargs) -> xr.DataArray:
        return self._xdata.copy(data=x)

    @docfiller_abc()
    def new_like(
        self: xCentralMoments,
        data: np.ndarray | xr.DataArray | None = None,
        copy: bool = False,
        copy_kws: Mapping | None = None,
        verify: bool = True,
        check_shape: bool = True,
        dtype: DTypeLike | None = None,
        strict: bool = False,
        **kws,
    ) -> xCentralMoments:
        if data is None:
            data = xr.zeros_like(self._xdata)
            copy = verify = check_shape = False

        kws.setdefault("mom_ndim", self.mom_ndim)

        if strict:
            kws = {
                "mom": self.mom,
                "val_shape": self.val_shape,
                "dtype": self.dtype,
                **kws,
            }

        if isinstance(data, np.ndarray):
            kws.setdefault("template", self._xdata)

        return type(self).from_data(
            data=data,
            copy=copy,
            copy_kws=copy_kws,
            verify=verify,
            check_shape=check_shape,
            **kws,
        )

    # ** Access to underlying statistics
    def _single_index_selector(
        self, val, dim_combined="variable", coords_combined=None, select=True
    ):
        idxs = self._single_index(val)[-self.mom_ndim :]
        if coords_combined is None:
            coords_combined = self.mom_dims

        selector = {
            dim: (idx if self._mom_ndim == 1 else xr.DataArray(idx, dims=dim_combined))
            for dim, idx in zip(self.mom_dims, idxs)
        }
        if select:
            out = self.values.isel(**selector)
            if self._mom_ndim > 1:
                out = out.assign_coords(**{dim_combined: list(coords_combined)})
            return out
        else:
            return selector

    def mean(self, dim_combined="variable", coords_combined=None):
        """Return mean/first moment(s) of data."""
        return self._single_index_selector(
            val=1, dim_combined=dim_combined, coords_combined=coords_combined
        )

    def var(self, dim_combined="variable", coords_combined=None):
        """Return variance (second central moment) of data."""
        return self._single_index_selector(
            val=2, dim_combined=dim_combined, coords_combined=coords_combined
        )

    # ** xarray specific methods
    def _wrap_xarray_method(self, _method, *args, **kwargs):
        xdata = getattr(self._xdata, _method)(*args, **kwargs)
        return self.new_like(data=xdata, strict=False)

    def assign_coords(self, coords=None, **coords_kwargs):
        """Assign coordinates to data and return new object."""
        return self._wrap_xarray_method("assign_coords", coords=coords, **coords_kwargs)

    def assign_attrs(self, *args, **kwargs):
        """Assign attributes to data and return new object."""
        return self._wrap_xarray_method("assign_attrs", *args, **kwargs)

    def rename(self, new_name_or_name_dict=None, **names):
        """Rename object."""
        return self._wrap_xarray_method(
            "rename", new_name_or_name_dict=new_name_or_name_dict, **names
        )

    def stack(
        self,
        dimensions: Mapping[Any, Sequence[Hashable]] | None = None,
        _order: bool = True,
        _verify: bool = False,
        _copy: bool = False,
        _check_mom: bool = True,
        _kws: Mapping | None = None,
        **dimensions_kwargs,
    ) -> xCentralMoments:
        """
        Stack dimensions.

        Returns
        -------
        output : xCentralMoments
            With dimensions stacked.

        See Also
        --------
        pipe
        xarray.DataArray.stack

        Examples
        --------
        >>> np.random.seed(0)
        >>> da = xCentralMoments.from_vals(np.random.rand(10, 2, 3), mom=2, axis=0)
        >>> da
        <xCentralMoments(val_shape=(2, 3), mom=(2,))>
        <xarray.DataArray (dim_0: 2, dim_1: 3, mom_0: 3)>
        array([[[10.    ,  0.4549,  0.044 ],
                [10.    ,  0.6019,  0.0849],
                [10.    ,  0.6049,  0.0911]],
        <BLANKLINE>
               [[10.    ,  0.5372,  0.0591],
                [10.    ,  0.4262,  0.0843],
                [10.    ,  0.4733,  0.0591]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0
        >>> da_stack = da.stack(z=["dim_0", "dim_1"])
        >>> da_stack
        <xCentralMoments(val_shape=(6,), mom=(2,))>
        <xarray.DataArray (z: 6, mom_0: 3)>
        array([[10.    ,  0.4549,  0.044 ],
               [10.    ,  0.6019,  0.0849],
               [10.    ,  0.6049,  0.0911],
               [10.    ,  0.5372,  0.0591],
               [10.    ,  0.4262,  0.0843],
               [10.    ,  0.4733,  0.0591]])
        Coordinates:
          * z        (z) object MultiIndex
          * dim_0    (z) int64 0 0 0 1 1 1
          * dim_1    (z) int64 0 1 2 0 1 2
        Dimensions without coordinates: mom_0

        And unstack

        >>> da_stack.unstack("z")
        <xCentralMoments(val_shape=(2, 3), mom=(2,))>
        <xarray.DataArray (dim_0: 2, dim_1: 3, mom_0: 3)>
        array([[[10.    ,  0.4549,  0.044 ],
                [10.    ,  0.6019,  0.0849],
                [10.    ,  0.6049,  0.0911]],
        <BLANKLINE>
               [[10.    ,  0.5372,  0.0591],
                [10.    ,  0.4262,  0.0843],
                [10.    ,  0.4733,  0.0591]]])
        Coordinates:
          * dim_0    (dim_0) int64 0 1
          * dim_1    (dim_1) int64 0 1 2
        Dimensions without coordinates: mom_0
        """
        return self.pipe(
            "stack",
            dimensions=dimensions,
            _order=_order,
            _copy=_copy,
            _verify=_verify,
            _check_mom=_check_mom,
            _kws=_kws,
            **dimensions_kwargs,
        )

    def unstack(
        self,
        dim: Hashable | Sequence[Hashable] | None = None,
        fill_value: Any = np.nan,
        sparse: bool = False,
        _order=True,
        _copy=False,
        _verify=False,
        _check_mom=True,
        _kws=None,
    ) -> xCentralMoments:
        """
        Unstack dimensions.

        Returns
        -------
        output : xCentralMoments
            With dimensions unstacked

        See Also
        --------
        stack
        xarray.DataArray.unstack

        """
        return self.pipe(
            "unstack",
            _order=_order,
            _copy=_copy,
            _verify=_verify,
            _kws=_kws,
            _check_mom=_check_mom,
            dim=dim,
            fill_value=fill_value,
            sparse=sparse,
        )

    def sel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        method: str | None = None,
        tolerance: Any = None,
        drop: bool = False,
        _order=False,
        _copy=False,
        _verify=False,
        _check_mom=True,
        _kws=None,
        **indexers_kws,
    ) -> xCentralMoments:
        """
        Select subset of data.

        Returns
        -------
        output : xCentralMoments
            With dimensions unstacked

        See Also
        --------
        xarray.DataArray.sel

        Examples
        --------
        >>> np.random.seed(0)
        >>> da = xCentralMoments.from_vals(
        ...     np.random.rand(10, 3), axis=0, dims="x", coords=dict(x=list("abc"))
        ... )
        >>> da
        <xCentralMoments(val_shape=(3,), mom=(2,))>
        <xarray.DataArray (x: 3, mom_0: 3)>
        array([[10.    ,  0.521 ,  0.0703],
               [10.    ,  0.6261,  0.0701],
               [10.    ,  0.5962,  0.0892]])
        Coordinates:
          * x        (x) <U1 'a' 'b' 'c'
        Dimensions without coordinates: mom_0

        Select by value

        >>> da.sel(x="a")
        <xCentralMoments(val_shape=(), mom=(2,))>
        <xarray.DataArray (mom_0: 3)>
        array([10.    ,  0.521 ,  0.0703])
        Coordinates:
            x        <U1 'a'
        Dimensions without coordinates: mom_0
        >>> da.sel(x=["a", "c"])
        <xCentralMoments(val_shape=(2,), mom=(2,))>
        <xarray.DataArray (x: 2, mom_0: 3)>
        array([[10.    ,  0.521 ,  0.0703],
               [10.    ,  0.5962,  0.0892]])
        Coordinates:
          * x        (x) <U1 'a' 'c'
        Dimensions without coordinates: mom_0


        Select by position

        >>> da.isel(x=0)
        <xCentralMoments(val_shape=(), mom=(2,))>
        <xarray.DataArray (mom_0: 3)>
        array([10.    ,  0.521 ,  0.0703])
        Coordinates:
            x        <U1 'a'
        Dimensions without coordinates: mom_0
        >>> da.isel(x=[0, 1])
        <xCentralMoments(val_shape=(2,), mom=(2,))>
        <xarray.DataArray (x: 2, mom_0: 3)>
        array([[10.    ,  0.521 ,  0.0703],
               [10.    ,  0.6261,  0.0701]])
        Coordinates:
          * x        (x) <U1 'a' 'b'
        Dimensions without coordinates: mom_0

        """
        return self.pipe(
            "sel",
            _order=_order,
            _copy=_copy,
            _verify=_verify,
            _check_mom=_check_mom,
            _kws=_kws,
            indexers=indexers,
            method=method,
            tolerance=tolerance,
            drop=drop,
            **indexers_kws,
        )

    def isel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        drop: bool = False,
        _order: bool = False,
        _copy: bool = False,
        _verify: bool = False,
        _check_mom=True,
        _kws: None = None,
        **indexers_kws,
    ) -> xCentralMoments:
        """
        Select subset of data by position.

        Returns
        -------
        output : xCentralMoments
            With dimensions unstacked

        See Also
        --------
        sel
        xarray.DataArray.isel
        """
        return self.pipe(
            "isel",
            _order=_order,
            _copy=_copy,
            _verify=_verify,
            _check_mom=_check_mom,
            _kws=_kws,
            indexers=indexers,
            drop=drop,
            **indexers_kws,
        )

    def transpose(
        self,
        *dims,
        transpose_coords=None,
        missing_dims="raise",
        _order=True,
        _copy=False,
        _verify=False,
        _check_mom=True,
        _kws=None,
        **kws,
    ) -> xCentralMoments:
        """
        Transpose dimensions of data.

        Notes
        -----
        if ``_order = True`` (the default), then make sure mom_dims are last
        regardless of input.

        See Also
        --------
        pipe
        xarray.DataArray.transpose
        """
        # make sure dims are last

        if _order:
            dims = tuple(d for d in dims if d not in self.mom_dims) + self.mom_dims

        return self.pipe(
            "transpose",
            *dims,
            transpose_coords=transpose_coords,
            missing_dims=missing_dims,
            _order=False,
            _copy=_copy,
            _verify=_verify,
            _check_mom=_check_mom,
            _kws=_kws,
        )

    # ** To/from CentralMoments
    def to_centralmoments(self) -> CentralMoments:
        """Create a CentralMoments object from xCentralMoments."""
        return CentralMoments(data=self.data, mom_ndim=self.mom_ndim)

    @cached.prop
    def centralmoments_view(self) -> CentralMoments:
        """
        Create CentralMoments view.

        This object has the same underlying data as `self`, but no
        DataArray attributes.  Useful for some function calls.

        See Also
        --------
        CentralMoments.to_xcentralmoments
        """
        return self.to_centralmoments()

    @classmethod
    @docfiller.decorate
    def from_centralmoments(
        cls,
        obj: CentralMoments,
        dims: Hashable | Sequence[Hashable] | None = None,
        attrs: Mapping | None = None,
        coords: Mapping | None = None,
        name: Hashable | None = None,
        indexes: Any = None,
        mom_dims: Hashable | tuple[Hashable, Hashable] | None = None,
        template: xr.DataArray | None = None,
        copy: bool = False,
    ) -> xCentralMoments:
        """
        Create and xCentralMoments object from CentralMoments.

        Parameters
        ----------
        obj : CentralMoments
            Input object to be converted.
        {xr_params}
        {copy}

        Returns
        -------
        output : xCentralMoments

        See Also
        --------
        CentralMoments.to_xcentralmoments

        """

        data = obj.to_xarray(
            dims=dims,
            attrs=attrs,
            coords=coords,
            name=name,
            indexes=indexes,
            mom_dims=mom_dims,
        )
        return cls(data=data, mom_ndim=obj.mom_ndim)

    @staticmethod
    def _wrap_centralmoments_method(
        _method_name: str,
        *args,
        dims: Dims | None = None,
        mom_dims: MomDims | None = None,
        attrs: Mapping | None = None,
        coords: Mapping | None = None,
        indexes: Any = None,
        name: Hashable | None = None,
        template: xr.DataArray | None = None,
        copy_final: bool = False,
        **kwargs,
    ) -> xCentralMoments:
        method = cast(
            Callable[..., CentralMoments], getattr(CentralMoments, _method_name)
        )

        return method(*args, **kwargs).to_xcentralmoments(
            dims=dims,
            mom_dims=mom_dims,
            attrs=attrs,
            coords=coords,
            indexes=indexes,
            name=name,
            template=template,
            copy=copy_final,
        )

    # ** Push/verify
    def _xverify_value(
        self,
        x: np.ndarray | xr.DataArray | float,
        target: str
        | np.ndarray
        | xr.DataArray
        | tuple[Hashable, ...]
        | str
        | None = None,
        dim: Hashable | None = None,
        axis: int | None = None,
        broadcast: bool = False,
        expand: bool = False,
        shape_flat: Any | None = None,
    ) -> (
        tuple[np.ndarray, xr.DataArray]
        | np.ndarray
        | tuple[float, xr.DataArray]
        | float
    ):
        if isinstance(target, str):
            # if dim is not None:
            #     if isinstance(dim, int):
            #         dim = x.dims[dim]

            if dim is not None and axis is not None:
                axis = None

            if dim is not None or axis is not None:
                assert isinstance(x, xr.DataArray)
                axis, dim = _select_axis_dim(dims=x.dims, axis=axis, dim=dim)

            if target == "val":
                target = self.val_dims
            elif target == "vals":
                target = (dim,) + self.val_dims
            elif target == "data":
                target = self.dims
            elif target == "datas":
                target = (dim,) + self.dims

        if isinstance(target, tuple):
            # no broadcast in this cast
            assert isinstance(x, xr.DataArray)

            target_dims = target

            target_shape = tuple(
                x.sizes[k] if k == dim else self.sizes[k] for k in target_dims
            )

            # make sure in correct order
            x = x.transpose(*target_dims)
            target_output = x
            values = x.values

        else:
            assert isinstance(target, xr.DataArray)
            target_dims = target.dims
            target_shape = target.shape

            target_output = None

            if dim is not None and axis is not None:
                axis = None

            if dim is not None or axis is not None:
                # this is hackish
                # in this case, target should already be in correct order
                # so just steal from target_shape
                dim = target_dims[0]

            if isinstance(x, xr.DataArray):
                if broadcast:
                    x = x.broadcast_like(target)
                else:
                    x = x.transpose(*target_dims)

                values = x.values
            else:
                # only things this can be is either a scalar or
                # array with same size as target
                x = np.asarray(x)
                if x.shape == target.shape:
                    values = x
                    # have x -> target to get correct recs
                    x = target

                elif x.ndim == 0 and broadcast and expand:
                    x = xr.DataArray(x).broadcast_like(target)
                    values = x.values

                elif (
                    x.ndim == 1 and len(x) == target.sizes[dim] and broadcast and expand
                ):
                    x = xr.DataArray(x, dims=dim).broadcast_like(target)
                    values = x.values

        # check shape
        assert values.shape == target_shape
        if dim is None:
            nrec: tuple[int, ...] = ()
        else:
            assert isinstance(x, xr.DataArray)
            nrec = (x.sizes[dim],)

        if shape_flat is not None:
            values = values.reshape(nrec + shape_flat)

        if values.ndim == 0:
            values = values[()]

        if target_output is None:
            return values
        else:
            return values, target_output

    def _verify_value(
        self,
        x: float | np.ndarray | xr.DataArray,
        target: str | np.ndarray | xr.DataArray | tuple[int, ...] | None = None,
        axis: int | None = None,
        dim: Hashable | None = None,
        broadcast: bool = False,
        expand: bool = False,
        other: np.ndarray | None = None,
        shape_flat: tuple[int, ...] | None = None,
        **kwargs,
    ) -> Any:
        if isinstance(x, xr.DataArray) or isinstance(target, xr.DataArray):
            return self._xverify_value(
                x,
                target=target,
                axis=axis,
                dim=dim,
                # dim=axis,
                broadcast=broadcast,
                expand=expand,
                shape_flat=shape_flat,
            )

        else:
            assert axis is None or isinstance(
                axis, int
            ), f"Error with axis value {axis}"

            return self.centralmoments_view._verify_value(
                x,
                target=target,
                axis=axis,
                broadcast=broadcast,
                expand=expand,
                shape_flat=shape_flat,
                other=other,
            )

    # TODO: update signatures for data/x/w
    @docfiller_inherit_abc()
    def push_data(
        self: T_CentralMoments,
        data: np.ndarray | xr.DataArray,
    ) -> T_CentralMoments:
        return super().push_data(data=data)

    @docfiller_inherit_abc()
    def push_datas(
        self: T_CentralMoments,
        datas: np.ndarray | xr.DataArray,
        axis: int | None = None,
        dim: Hashable | None = None,
        **kwargs,
    ) -> T_CentralMoments:
        return super().push_datas(datas=datas, axis=axis, dim=dim, **kwargs)

    @docfiller_inherit_abc()
    def push_val(
        self: T_CentralMoments,
        x: float
        | np.ndarray
        | xr.DataArray
        | tuple[float | np.ndarray | xr.DataArray, float | np.ndarray | xr.DataArray],
        w: float | np.ndarray | xr.DataArray | None = None,
        broadcast: bool = False,
        **kwargs,
    ) -> T_CentralMoments:
        return super().push_val(x=x, w=w, broadcast=broadcast, **kwargs)

    @docfiller_inherit_abc()
    def push_vals(
        self: T_CentralMoments,
        x: np.ndarray
        | xr.DataArray
        | tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray],
        w: float | np.ndarray | xr.DataArray | None = None,
        axis: int | None = None,
        broadcast: bool = False,
        dim: Hashable | None = None,
        **kwargs,
    ) -> T_CentralMoments:
        return super().push_vals(
            x=x, w=w, axis=axis, broadcast=broadcast, dim=dim, **kwargs
        )

    # ** Manipulation
    @docfiller.decorate
    def resample_and_reduce(
        self,
        freq: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        nrep: int | None = None,
        axis: int | None = None,
        dim: Hashable | None = None,
        rep_dim: str = "rep",
        parallel: bool = True,
        resample_kws: Mapping | None = None,
        full_output: bool = False,
        **kws,
    ) -> xCentralMoments | tuple[xCentralMoments, np.ndarray]:
        """

        Parameters
        ----------
        {freq}
        {indices}
        {nrep}
        {axis}
        {dim}
        {rep_dim}
        {parallel}
        {resample_kws}
        {full_output}
        **kws
            Arguments to :meth:`CentralMoments.resample_and_reduce`

        Returns
        -------
        output : {klass}

        Examples
        --------
        >>> np.random.seed(0)
        >>> da = xCentralMoments.from_vals(
        ...     np.random.rand(10, 3), mom=3, axis=0, dims="rec"
        ... )
        >>> da
        <xCentralMoments(val_shape=(3,), mom=(3,))>
        <xarray.DataArray (rec: 3, mom_0: 4)>
        array([[ 1.0000e+01,  5.2102e-01,  7.0287e-02, -3.5494e-03],
               [ 1.0000e+01,  6.2614e-01,  7.0138e-02, -1.7101e-02],
               [ 1.0000e+01,  5.9620e-01,  8.9201e-02, -1.1886e-02]])
        Dimensions without coordinates: rec, mom_0

        Note that for reproducible results, must set numba random
        seed as well

        >>> from cmomy.resample import numba_random_seed
        >>> numba_random_seed(0)
        >>> da_resamp, freq = da.resample_and_reduce(
        ...     nrep=5, dim="rec", full_output=True
        ... )
        >>> da_resamp
        <xCentralMoments(val_shape=(5,), mom=(3,))>
        <xarray.DataArray (rep: 5, mom_0: 4)>
        array([[ 3.0000e+01,  5.5606e-01,  7.2693e-02, -7.9911e-03],
               [ 3.0000e+01,  6.1616e-01,  7.6691e-02, -1.5745e-02],
               [ 3.0000e+01,  5.4608e-01,  7.7848e-02, -5.3486e-03],
               [ 3.0000e+01,  5.4608e-01,  7.7848e-02, -5.3486e-03],
               [ 3.0000e+01,  6.0618e-01,  8.3046e-02, -1.4003e-02]])
        Dimensions without coordinates: rep, mom_0

        Alternatively, we can resample and reduce

        >>> from cmomy.resample import freq_to_indices
        >>> indices = freq_to_indices(freq)
        >>> da.sel(rec=xr.DataArray(indices, dims=["rep", "rec"])).reduce(dim="rec")
        <xCentralMoments(val_shape=(5,), mom=(3,))>
        <xarray.DataArray (rep: 5, mom_0: 4)>
        array([[ 3.0000e+01,  5.5606e-01,  7.2693e-02, -7.9911e-03],
               [ 3.0000e+01,  6.1616e-01,  7.6691e-02, -1.5745e-02],
               [ 3.0000e+01,  5.4608e-01,  7.7848e-02, -5.3486e-03],
               [ 3.0000e+01,  5.4608e-01,  7.7848e-02, -5.3486e-03],
               [ 3.0000e+01,  6.0618e-01,  8.3046e-02, -1.4003e-02]])
        Dimensions without coordinates: rep, mom_0

        """
        self._raise_if_scalar()

        axis, dim = _select_axis_dim(dims=self.dims, axis=axis, dim=dim)

        if dim in self.mom_dims:
            raise ValueError(f"can only resample from value dimensions {self.val_dims}")

        # Final form will move `dim` to front of array.
        # this will be replaced by rep_dimension
        template = self.values.isel({dim: 0})

        out, freq = self.centralmoments_view.resample_and_reduce(
            freq=freq,
            indices=indices,
            nrep=nrep,
            axis=axis,
            parallel=parallel,
            resample_kws=resample_kws,
            full_output=True,
            **kws,
        )

        new = out.to_xcentralmoments(
            dims=(rep_dim,) + template.dims,
            mom_dims=None,
            attrs=template.attrs,
            coords=template.coords,
            name=template.name,
        )

        if full_output:
            return new, freq

        else:
            return new

    def _wrap_axis(
        self, axis: int | None, default: int = 0, ndim: int | None = None, **kws
    ) -> int:
        if isinstance(axis, str):
            raise ValueError("shouldn't get string axis here")
        else:
            return super()._wrap_axis(axis=axis, default=default, ndim=ndim)

    @docfiller.decorate
    def reduce(
        self: xCentralMoments,
        dim: Hashable | None = None,
        axis: int | None = None,
        **kws,
    ) -> xCentralMoments:
        """
        Parameters
        ----------
        {dim}
        {axis}
        **kws
        Extra arguments to :meth:`from_datas`

        Returns
        -------
        output : {klass}
            Reduced along dimension

        See Also
        --------
        from_datas


        Examples
        --------
        >>> np.random.seed(0)
        >>> da = xCentralMoments.from_vals(np.random.rand(10, 2, 3), axis=0)
        >>> da.reduce(dim="dim_0")
        <xCentralMoments(val_shape=(3,), mom=(2,))>
        <xarray.DataArray (dim_1: 3, mom_0: 3)>
        array([[20.    ,  0.4961,  0.0532],
               [20.    ,  0.5141,  0.0923],
               [20.    ,  0.5391,  0.0794]])
        Dimensions without coordinates: dim_1, mom_0
        """

        self._raise_if_scalar()
        axis, dim = _select_axis_dim(dims=self.dims, axis=axis, dim=dim)
        axis = self._wrap_axis(axis)
        return type(self).from_datas(
            self.values, mom_ndim=self.mom_ndim, axis=axis, **kws
        )

    @docfiller.decorate
    def block(
        self,
        block_size: int,
        dim: Hashable | None = None,
        axis: int | None = None,
        coords_policy: Literal["first", "last", None] = "first",
        **kws,
    ) -> xCentralMoments:
        """
        Parameters
        ----------
        block_size : int
            Number of observations to include in a given block.
        {dim}
        {axis}
        coords_policy : {{'first','last',None}}
            Policy for handling coordinates along `axis`.
            If no coordinates do nothing, otherwise use:

            * 'first': select first value of coordinate for each block.
            * 'last': select last value of coordinate for each block.
            * None: drop any coordinates.
        **kws
            Extra arguments to :meth:`CentralMoments.block`

        Returns
        -------
        output : xCentralMoments
            Object with block averaging.

        See Also
        --------
        CentralMoments.block


        Examples
        --------
        >>> np.random.seed(0)
        >>> x = np.random.rand(10, 10)
        >>> da = xCentralMoments.from_vals(x, mom=2, axis=1)
        >>> da
        <xCentralMoments(val_shape=(10,), mom=(2,))>
        <xarray.DataArray (dim_0: 10, mom_0: 3)>
        array([[10.    ,  0.6158,  0.034 ],
               [10.    ,  0.5473,  0.1159],
               [10.    ,  0.5803,  0.0832],
               [10.    ,  0.5554,  0.0608],
               [10.    ,  0.391 ,  0.0469],
               [10.    ,  0.4087,  0.0672],
               [10.    ,  0.3481,  0.0845],
               [10.    ,  0.4623,  0.11  ],
               [10.    ,  0.4443,  0.0651],
               [10.    ,  0.3747,  0.0822]])
        Dimensions without coordinates: dim_0, mom_0

        >>> da.block(block_size=5, dim="dim_0")
        <xCentralMoments(val_shape=(2,), mom=(2,))>
        <xarray.DataArray (dim_0: 2, mom_0: 3)>
        array([[50.    ,  0.538 ,  0.0741],
               [50.    ,  0.4076,  0.0836]])
        Dimensions without coordinates: dim_0, mom_0

        This is equivalent to

        >>> xCentralMoments.from_vals(x.reshape(2, 50), mom=2, axis=1)
        <xCentralMoments(val_shape=(2,), mom=(2,))>
        <xarray.DataArray (dim_0: 2, mom_0: 3)>
        array([[50.    ,  0.538 ,  0.0741],
               [50.    ,  0.4076,  0.0836]])
        Dimensions without coordinates: dim_0, mom_0

        """

        self._raise_if_scalar()
        axis, dim = _select_axis_dim(dims=self.dims, axis=axis, dim=dim, default_axis=0)

        if block_size is None:
            block_size = self.sizes[dim]
            nblock = 1
        else:
            nblock = self.sizes[dim] // block_size

        if coords_policy == "first":
            start = 0
        else:
            start = block_size - 1

        # get template values
        template = self.values.isel(
            {dim: slice(start, block_size * nblock, block_size)}
        ).transpose(dim, ...)

        if coords_policy is None:
            template = template.drop(dim)

        return self.centralmoments_view.block(
            block_size=block_size, axis=axis, **kws
        ).to_xcentralmoments(template=template)

    # ** Constructors
    @classmethod
    @docfiller_inherit_abc()
    def zeros(
        cls: type[xCentralMoments],
        mom: Moments | None = None,
        mom_ndim: Mom_NDim | None = None,
        val_shape: tuple[int, ...] | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        zeros_kws: Mapping | None = None,
        dims: tuple[Hashable, ...] | None = None,
        coords: Mapping | None = None,
        attrs: Mapping | None = None,
        name: Hashable | None = None,
        indexes: Any | None = None,
        template: xr.DataArray | None = None,
        mom_dims: MomDims | None = None,
        **kws: Any,
    ) -> xCentralMoments:
        """
        Parameters
        ----------
        {xr_params}
        {mom_dims}

        See Also
        --------
        CentralMoments.to_xcentralmoments
        """

        if template is None:
            return cls._wrap_centralmoments_method(
                "zeros",
                mom=mom,
                val_shape=val_shape,
                mom_ndim=mom_ndim,
                shape=shape,
                dtype=dtype,
                zeros_kws=zeros_kws,
                dims=dims,
                coords=coords,
                attrs=attrs,
                name=name,
                indexes=indexes,
                mom_dims=mom_dims,
                template=template,
            )

        else:
            kws = {"copy": True, "check_shape": True, "verify": True, **kws}

            return cls.from_data(
                data=template,
                mom_ndim=mom_ndim,
                mom=mom,
                val_shape=val_shape,
                dtype=dtype,
                dims=dims,
                coords=coords,
                attrs=attrs,
                name=name,
                indexes=indexes,
                mom_dims=mom_dims,
                **kws,
            ).zero()

    @classmethod
    @docfiller_inherit_abc()
    def from_data(
        cls: type[xCentralMoments],
        data: np.ndarray | xr.DataArray,
        mom: Moments | None = None,
        mom_ndim: Mom_NDim | None = None,
        val_shape: tuple[int, ...] | None = None,
        copy: bool = True,
        copy_kws: Mapping | None = None,
        verify: bool = True,
        check_shape: bool = True,
        dtype: DTypeLike | None = None,
        dims: Dims | None = None,
        attrs: Mapping | None = None,
        coords: Mapping | None = None,
        name: Hashable | None = None,
        indexes: Any = None,
        mom_dims: MomDims | None = None,
        template: Any | None = None,
    ) -> xCentralMoments:
        """
        Parameters
        ----------
        data : ndarray or DataArray
            If DataArray, use it's attributes in final object.
            If ndarray, use passed xarray parameters (`dims`, `attrs`, etc.).
        {xr_params}
        {mom_dims}

        See Also
        --------
        CentralMoments.from_data
        """

        if isinstance(data, xr.DataArray):
            mom_ndim = cls._choose_mom_ndim(mom, mom_ndim)

            data = _move_mom_dims_to_end(data, mom_dims, mom_ndim)

            if verify:
                data_verified = data.astype(
                    dtype=dtype or data.dtype, order="C", copy=False
                )
            else:
                data_verified = data

            if check_shape:
                if val_shape is None:
                    val_shape = data_verified.shape[:-mom_ndim]
                mom = cls._check_mom(mom, mom_ndim, data_verified.shape)

                if data_verified.shape != val_shape + tuple(x + 1 for x in mom):
                    raise ValueError(
                        f"{data.shape} does not conform to {val_shape} and {mom}"
                    )

            if copy and data_verified is data:
                if copy_kws is None:
                    copy_kws = {}

                # to make sure copy has same format
                data_verified = data_verified.copy(
                    data=data_verified.values.copy(**copy_kws)
                )

            return cls(data=data_verified, mom_ndim=mom_ndim)

        else:
            return cls._wrap_centralmoments_method(
                "from_data",
                data=data,
                mom=mom,
                mom_ndim=mom_ndim,
                val_shape=val_shape,
                dtype=dtype,
                copy=copy,
                copy_kws=copy_kws,
                verify=verify,
                check_shape=check_shape,
                dims=dims,
                attrs=attrs,
                coords=coords,
                indexes=indexes,
                name=name,
                template=template,
                mom_dims=mom_dims,
            )

    @classmethod
    @docfiller_inherit_abc()
    def from_datas(
        cls,
        datas: np.ndarray | xr.DataArray,
        mom_ndim: Mom_NDim | None = None,
        axis: int | None = None,
        mom: Moments | None = None,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        verify: bool = True,
        check_shape: bool = True,
        dim: Hashable | None = None,
        dims: Dims | None = None,
        attrs: Mapping | None = None,
        coords: Mapping | None = None,
        indexes: Any | None = None,
        name: Hashable | None = None,
        mom_dims: MomDims | None = None,
        template: xr.DataArray | None = None,
        **kws,
    ) -> xCentralMoments:
        """

        Parameters
        ----------
        datas : ndarray or DataArray
            If pass in a DataArray, use it's attributes in new object.
            If ndarray, use `dim`, `attrs`, etc, to wrap resulting data.
        {dim}
        {xr_params}


        See Also
        --------
        CentralMoments.from_datas
        CentralMoments.to_xcentralmoments

        Notes
        -----
        If pass in :class:`xarray.DataArray`, then dims, etc, are ignored.
        Note that here, `dims` does not include the dimension reduced over.
        The dimensions are applied after the fact.


        """

        if isinstance(datas, xr.DataArray):
            mom_ndim = cls._choose_mom_ndim(mom, mom_ndim)
            axis, dim = _select_axis_dim(dims=datas.dims, axis=axis, dim=dim)

            datas = _move_mom_dims_to_end(datas, mom_dims, mom_ndim).transpose(dim, ...)

            if verify:
                datas = datas.astype(order="C", dtype=dtype, copy=False)

            if check_shape:
                if val_shape is None:
                    val_shape = datas.shape[1:-mom_ndim]

                mom = cls._check_mom(mom, mom_ndim, datas.shape)
                assert datas.shape[1:] == val_shape + tuple(x + 1 for x in mom)

            new = (
                cls(
                    # template for output data
                    data=datas.isel({dim: 0}).astype(dtype=dtype, copy=True),
                    mom_ndim=mom_ndim,
                )
                .zero()
                .push_datas(datas, dim=dim)
            )

        else:
            new = cls._wrap_centralmoments_method(
                "from_datas",
                datas=datas,
                mom=mom,
                mom_ndim=mom_ndim,
                axis=axis,
                val_shape=val_shape,
                dtype=dtype,
                verify=verify,
                check_shape=check_shape,
                dims=dims,
                attrs=attrs,
                coords=coords,
                indexes=indexes,
                template=template,
                name=name,
                mom_dims=mom_dims,
                **kws,
            )

        return new

    @classmethod
    @docfiller_inherit_abc()
    def from_raw(
        cls,
        raw: np.ndarray | xr.DataArray,
        mom_ndim: Mom_NDim | None = None,
        mom: Moments | None = None,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        convert_kws: Mapping | None = None,
        dims: Dims | None = None,
        attrs: Mapping | None = None,
        coords: Mapping | None = None,
        indexes: Any | None = None,
        name: Hashable | None = None,
        mom_dims: MomDims | None = None,
        template: xr.DataArray | None = None,
        **kws: Any,
    ) -> xCentralMoments:
        """
        Parameters
        ----------
        raw : ndarray or DataArray
            If DataArray, use attributes in final object.
            If ndarray, use `dims`, `attrs`, etc to wrap final result.
        {xr_params}
        {mom_dims}

        See Also
        --------
        CentralMoments.from_raw
        """

        if isinstance(raw, xr.DataArray):
            mom_ndim = cls._choose_mom_ndim(mom, mom_ndim)

            raw = _move_mom_dims_to_end(raw, mom_dims, mom_ndim)

            if convert_kws is None:
                convert_kws = {}

            if mom_ndim == 1:
                data_values = convert.to_central_moments(
                    raw.values, dtype=dtype, **convert_kws
                )
            elif mom_ndim == 2:
                data_values = convert.to_central_comoments(
                    raw.values, dtype=dtype, **convert_kws
                )
            else:
                raise ValueError(f"unknown mom_ndim {mom_ndim}")

            kws = {"copy": False, "check_shape": True, "verify": True, **kws}
            new = cls.from_data(
                data=raw.copy(data=data_values),
                mom=mom,
                mom_ndim=mom_ndim,
                val_shape=val_shape,
                dtype=dtype,
                **kws,
            )

        else:
            new = cls._wrap_centralmoments_method(
                "from_raw",
                raw=raw,
                mom_ndim=mom_ndim,
                mom=mom,
                val_shape=val_shape,
                dtype=dtype,
                convert_kws=convert_kws,
                dims=dims,
                attrs=attrs,
                coords=coords,
                indexes=indexes,
                mom_dims=mom_dims,
                template=template,
                **kws,
            )

        return new

    @classmethod
    @docfiller_inherit_abc()
    def from_raws(
        cls,
        raws: np.ndarray | xr.DataArray,
        mom_ndim: Mom_NDim | None = None,
        mom: Moments | None = None,
        axis: int | None = None,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        convert_kws: Mapping | None = None,
        dim: Hashable | None = None,
        dims: Dims | None = None,
        attrs: Mapping | None = None,
        coords: Mapping | None = None,
        indexes: Mapping | None = None,
        name: Hashable | None = None,
        mom_dims: MomDims = None,
        template: xr.DataArray | None = None,
        **kws,
    ):
        """
        Parameters
        ----------
        raws : ndarray or DataArray
            If DataArray, use attributes in final result
            If ndarray, use `dims`, `attrs`, to wrap final result
        {dim}
        {xr_params}
        {mom_dims}

        See Also
        --------
        CentralMoments.from_raw
        """

        if isinstance(raws, xr.DataArray):
            return cls.from_raw(
                raw=raws,
                mom=mom,
                mom_ndim=mom_ndim,
                val_shape=val_shape,
                dtype=dtype,
                convert_kws=convert_kws,
                mom_dims=mom_dims,
            ).reduce(dim=dim)

        else:
            return cls._wrap_centralmoments_method(
                "from_raws",
                raws=raws,
                mom=mom,
                mom_ndim=mom_ndim,
                axis=axis,
                val_shape=val_shape,
                dtype=dtype,
                convert_kws=convert_kws,
                dims=dims,
                attrs=attrs,
                coords=coords,
                indexes=indexes,
                name=name,
                mom_dims=mom_dims,
                template=template,
                **kws,
            )

    @classmethod
    @docfiller_inherit_abc()
    def from_vals(
        cls,
        x: np.ndarray
        | tuple[np.ndarray, np.ndarray]
        | xr.DataArray
        | tuple[xr.DataArray, xr.DataArray]
        | tuple[xr.DataArray, np.ndarray],
        w: float | np.ndarray | xr.DataArray | None = None,
        axis: int | None = None,
        mom: Moments = 2,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        broadcast: bool = False,
        dim: Hashable | None = None,
        dims: Dims | None = None,
        attrs: Mapping | None = None,
        coords: Mapping | None = None,
        indexes: Any | None = None,
        name: Hashable | None = None,
        mom_dims: MomDims | None = None,
        template: xr.DataArray | None = None,
        **kws,
    ) -> xCentralMoments:
        """
        Parameters
        ----------
        x : array, tuple of array, DataArray, or tuple of DataArray
            For moments, `x=x0`.  For comoments, `x=(x0, x1)`.
            If pass DataArray, inherit attributes from `x0`.  If pass
            ndarray, use `dims`, `attrs`, etc to wrap final result
        {dim}
        {xr_params}
        {mom_dims}

        See Also
        --------
        CentralMoments.from_vals
        """

        if isinstance(x, tuple):
            x0 = x[0]
        else:
            x0 = x

        if isinstance(x0, xr.DataArray):
            mom_ndim = cls._mom_ndim_from_mom(mom)
            axis, dim = _select_axis_dim(dims=x0.dims, axis=axis, dim=dim)

            if val_shape is None:
                val_shape = _shape_reduce(x0.shape, axis)
            if dtype is None:
                dtype = x0.dtype

            template = x0.isel({dim: 0})

            dims = template.dims
            if coords is None:
                coords = {}
            coords = dict(template.coords, **coords)
            if attrs is None:
                attrs = {}
            attrs = dict(template.attrs, **attrs)
            if name is None:
                name = template.name
            new = cls.zeros(
                val_shape=val_shape,
                mom=mom,
                mom_ndim=mom_ndim,
                dtype=dtype,
                dims=dims,
                attrs=attrs,
                coords=coords,
                indexes=indexes,
                name=name,
                mom_dims=mom_dims,
                **kws,
            ).push_vals(x=x, dim=dim, w=w, broadcast=broadcast)

        else:
            new = cls._wrap_centralmoments_method(
                "from_vals",
                dims=dims,
                attrs=attrs,
                coords=coords,
                indexes=indexes,
                name=name,
                mom_dims=mom_dims,
                template=template,
                x=x,
                w=w,
                axis=axis,
                mom=mom,
                val_shape=val_shape,
                dtype=dtype,
                broadcast=broadcast,
                **kws,
            )

        return new

    # @classmethod
    # @overload
    # def from_resample_vals(
    #     cls: type[xCentralMoments],
    #     x: np.ndarray
    #     | tuple[np.ndarray, np.ndarray]
    #     | xr.DataArray
    #     | tuple[xr.DataArray, xr.DataArray],
    #     freq: np.ndarray | None,
    #     indices: np.ndarray | None,
    #     nrep: int | None,
    #     w: np.ndarray | xr.DataArray | None,
    #     axis: int | None,
    #     dim: Hashable | None,
    #     mom: Moments,
    #     rep_dim: str,
    #     dtype: DTypeLike | None,
    #     broadcast: bool,
    #         parallel: bool,
    #         resample_kws: Mapping | None,
    #         dims: Dims | None,
    #         attrs: Mapping | None,
    #         coords: Mapping | None,
    #         indexes: Any | None,
    #         name: Hashable | None,
    #         mom_dims: MomDims | None,
    #     full_output: Literal[False] = False,
    #     **kws,
    # ) -> xCentralMoments: ...

    # @classmethod
    # @overload
    # def from_resample_vals(
    #     cls: type[xCentralMoments],
    #     x: np.ndarray
    #     | tuple[np.ndarray, np.ndarray]
    #     | xr.DataArray
    #     | tuple[xr.DataArray, xr.DataArray],
    #     freq: np.ndarray | None,
    #     indices: np.ndarray | None,
    #     nrep: int | None,
    #     w: np.ndarray | xr.DataArray | None,
    #     axis: int | None,
    #     dim: Hashable | None,
    #     mom: Moments,
    #     rep_dim: str,
    #     dtype: DTypeLike | None,
    #     broadcast: bool,
    #         parallel: bool,
    #         resample_kws: Mapping | None,
    #         dims: Dims | None,
    #         attrs: Mapping | None,
    #         coords: Mapping | None,
    #         indexes: Any | None,
    #         name: Hashable | None,
    #         mom_dims: MomDims | None,
    #     full_output: Literal[True],
    #     **kws,
    # ) -> tuple[xCentralMoments, np.ndarray]: ...

    @classmethod
    @docfiller_inherit_abc()
    def from_resample_vals(
        cls: type[xCentralMoments],
        x: np.ndarray
        | tuple[np.ndarray, np.ndarray]
        | xr.DataArray
        | tuple[xr.DataArray, xr.DataArray],
        freq: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        nrep: int | None = None,
        w: np.ndarray | xr.DataArray | None = None,
        axis: int | None = None,
        dim: Hashable | None = None,
        mom: Moments = 2,
        rep_dim: str = "rep",
        dtype: DTypeLike | None = None,
        broadcast: bool = False,
        parallel: bool = True,
        resample_kws: Mapping | None = None,
        dims: Dims | None = None,
        attrs: Mapping | None = None,
        coords: Mapping | None = None,
        indexes: Any | None = None,
        name: Hashable | None = None,
        mom_dims: MomDims | None = None,
        full_output: bool = False,
        **kws,
    ) -> xCentralMoments | tuple[xCentralMoments, np.ndarray]:
        """
        Parameters
        ----------
        x : array, tuple of array, DataArray, or tuple of DataArray
            For moments, `x=x0`.  For comoments, `x=(x0, x1)`.
            If pass DataArray, inherit attributes from `x0`.  If pass
            ndarray, use `dims`, `attrs`, etc to wrap final result
        {dim}
        {rep_dim}
        {xr_params}
        {mom_dims}

        See Also
        --------
        CentralMoments.from_resample_vals
        """

        if isinstance(x, tuple):
            x0 = x[0]
        else:
            x0 = x

        if isinstance(x0, xr.DataArray):
            axis, dim = _select_axis_dim(dims=x0.dims, axis=axis, dim=dim)
            # TODO: create object, and verify y, and w against x

            # override final xarray stuff:
            template = x0.isel({dim: 0})
            dims = template.dims
            if coords is None:
                coords = {}
            coords = dict(template.coords, **coords)
            if attrs is None:
                attrs = {}
            attrs = dict(template.attrs, **attrs)
            if name is None:
                name = template.name

        w_values: np.ndarray | None
        if isinstance(w, xr.DataArray):
            if isinstance(x0, xr.DataArray):
                w_values = w.broadcast_like(x0).values
            else:
                w_values = w.values
        else:
            w_values = w

        if dims is not None:
            if isinstance(dims, str):
                dims = (dims,)
            dims = (rep_dim,) + tuple(dims)  # type: ignore

        x_array: np.ndarray | tuple[np.ndarray, np.ndarray]
        if isinstance(x, tuple):
            x_array = tuple(np.array(xx, copy=False) for xx in x)  # type: ignore
        else:
            x_array = np.array(x, copy=False)

        out, freq = CentralMoments.from_resample_vals(
            x=x_array,
            freq=freq,
            indices=indices,
            nrep=nrep,
            w=w_values,
            axis=axis,
            mom=mom,
            dtype=dtype,
            broadcast=broadcast,
            parallel=parallel,
            resample_kws=resample_kws,
            full_output=True,
            **kws,
        )

        new = out.to_xcentralmoments(
            dims=dims,
            coords=coords,
            attrs=attrs,
            name=name,
            mom_dims=mom_dims,
            copy=False,
        )

        if full_output:
            return (new, freq)
        else:
            return new

    # @no_type_check
    # @classmethod
    # def from_stat(
    #     cls,
    #     a,
    #     v=0.0,
    #     w=None,
    #     mom=2,
    #     val_shape=None,
    #     dtype=None,
    #     dims=None,
    #     attrs=None,
    #     coords=None,
    #     indexes=None,
    #     name=None,
    #     mom_dims=None,
    #     **kws,
    # ):
    #     """Create from single observation of statisitcs (mean, variance).

    #     See CentralMoment.from_stat
    #     """

    #     kws, *_ = _check_xr_input(
    #         a,
    #         axis=None,
    #         _kws_in=kws,
    #         mom_dims=mom_dims,
    #         dims=dims,
    #         attrs=attrs,
    #         coords=coords,
    #         indexes=indexes,
    #         name=name,
    #     )

    #     return super(xCentralMoments, cls).from_stat(
    #         a=a, v=v, w=w, mom=mom, val_shape=val_shape, dtype=dtype, **kws
    #     )

    # @no_type_check
    # @classmethod
    # def from_stats(
    #     cls,
    #     a,
    #     v=0.0,
    #     w=None,
    #     axis=None,
    #     dim=None,
    #     mom=2,
    #     val_shape=None,
    #     dtype=None,
    #     dims=None,
    #     attrs=None,
    #     coords=None,
    #     indexes=None,
    #     name=None,
    #     mom_dims=None,
    #     **kws,
    # ):
    #     """Create from collection of statisitcs.

    #     See CentralMoments.from_stats
    #     """

    #     assert not (axis is None and dim is None)

    #     kws, axis, dim, values = _check_xr_input(
    #         a,
    #         axis=axis,
    #         dim=dim,
    #         mom_dims=mom_dims,
    #         dims=dims,
    #         attrs=attrs,
    #         coords=coords,
    #         indexes=indexes,
    #         name=name,
    #         _kws_in=kws,
    #     )

    #     return super(xCentralMoments, cls).from_stats(
    #         a=a,
    #         v=v,
    #         w=w,
    #         axis=axis,
    #         dim=dim,
    #         mom=mom,
    #         val_shape=val_shape,
    #         dtype=dtype,
    #         **kws,
    #     )


# Mostly deprecated.  Keeping around for now.


# * Deprecated utilities
def _xr_wrap_like(da, x):
    """Wrap x with xarray like da."""
    x = np.asarray(x)
    assert x.shape == da.shape

    return xr.DataArray(
        x, dims=da.dims, coords=da.coords, name=da.name, indexes=da.indexes
    )


@no_type_check
def _xr_order_like(template, *others):
    """Given dimensions, order in same manner."""

    if not isinstance(template, xr.DataArray):
        out = others

    else:
        dims = template.dims

        key_map = {dim: i for i, dim in enumerate(dims)}

        def key(x):
            return key_map[x]

        out = []
        for other in others:
            if isinstance(other, xr.DataArray):
                # reorder
                order = sorted(other.dims, key=key)

                x = other.transpose(*order)
            else:
                x = other

            out.append(x)

    if len(out) == 1:
        out = out[0]

    return out


def _attributes_from_xr(
    da: xr.DataArray | np.ndarray,
    dim: Hashable | None = None,
    mom_dims: MomDims | None = None,
    **kws,
) -> dict[str, Any]:
    if isinstance(da, xr.DataArray):
        if dim is not None:
            # reduce along this dim
            da = da.isel({dim: 0}, drop=True)
        out = {k: getattr(da, k) if v is None else v for k, v in kws.items()}
    else:
        out = kws.copy()

    out["mom_dims"] = mom_dims
    return out


@no_type_check
def _check_xr_input(
    x: xr.DataArray | np.ndarray,
    axis: int | None = None,
    dim: Hashable | None = None,
    mom_dims: MomDims | None = None,
    _kws_in: dict[Any, Any] | None = None,
    **kws,
) -> Any:
    if isinstance(x, xr.DataArray):
        # MIGRATION DIM
        # axis, dim = _select_axis_dim(x.dims, axis, dim)
        if axis is None and dim is None:
            pass
        else:
            axis, dim = _select_axis_dim(dims=x.dims, axis=axis, dim=dim)
        values = x.values
    else:
        if axis is None:
            dim = None
        else:
            dim = axis
        values = x
    kws = _attributes_from_xr(x, dim=dim, mom_dims=mom_dims, **kws)

    if _kws_in is not None and len(_kws_in) > 0:
        kws = dict(kws, **_kws_in)

    return kws, axis, dim, values


@no_type_check
def _optional_wrap_data(
    data: xr.DataArray | np.ndarray,
    mom_ndim: Mom_NDim,
    template: Any = None,
    dims: tuple[Hashable, ...] | None = None,
    coords: Mapping | None = None,
    name: Hashable | None = None,
    attrs: Mapping | None = None,
    indexes: Any = None,
    mom_dims: MomDims | None = None,
    dtype: DTypeLike | None = None,
    copy: bool = False,
    copy_kws: Mapping | None = None,
    verify: bool = True,
    # verify_mom_dims=True,
) -> xr.DataArray:
    """Wrap data with xarray."""

    if isinstance(data, xr.DataArray):
        if mom_dims is not None:
            if isinstance(mom_dims, str):
                mom_dims = (mom_dims,)
            else:
                mom_dims = tuple(mom_dims)

    elif template is not None:
        data = template.copy(data=data)

    else:
        # wrap data with DataArray
        ndim = data.ndim
        if dims is not None:
            if isinstance(dims, str):
                dims = [dims]
        else:
            dims = [f"dim_{i}" for i in range(ndim - mom_ndim)]
        dims = tuple(dims)

        if len(dims) == ndim:
            dims_total = dims

        elif len(dims) == ndim - mom_ndim:
            if mom_dims is None:
                mom_dims = tuple(f"mom_{i}" for i in range(mom_ndim))
            elif isinstance(mom_dims, str):
                mom_dims = (mom_dims,)
            else:
                mom_dims = tuple(mom_dims)

            dims_total = dims + mom_dims
        else:
            raise ValueError(f"bad dims {dims}, moment_dims {mom_dims}")

        # xarray object
        data = xr.DataArray(
            data,
            dims=dims_total,
            coords=coords,
            attrs=attrs,
            name=name,
            # skip this option.  Breaks with some versions of xarray
            # indexes=indexes,
        )

    # if verify_mom_dims:
    #     if data.dims[-mom_ndim:] != mom_dims:
    #         data = data.transpose(*((..., ) + mom_dims))
    if mom_dims is not None:
        if data.dims[-mom_ndim:] != mom_dims:
            raise ValueError(f"last dimensions {data.dims} do not match {mom_dims}")

    if verify:
        vals = np.asarray(data.values, dtype=dtype, order="c")
    else:
        vals = data.values

    if copy:
        if copy_kws is None:
            copy_kws = {}

        if vals is data.values:
            vals = vals.copy(**copy_kws)

        data = data.copy(data=vals)

    elif vals is not data.values:
        # data.values = vals
        # Above leads to overwriting the data object in cases where we are updating things.
        # Instead, create a new object with the correct data
        data = data.copy(data=vals)
    return data

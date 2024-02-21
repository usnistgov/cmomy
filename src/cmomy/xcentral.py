"""Thin wrapper around central routines with xarray support."""
from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload  # TYPE_CHECKING,

import numpy as np
import xarray as xr
from module_utilities import cached

from . import convert
from ._compat import xr_dot
from .abstract_central import CentralMomentsABC
from .docstrings import docfiller_xcentral as docfiller
from .utils import (
    mom_to_mom_ndim,
    select_mom_ndim,
    shape_reduce,
    validate_mom_and_mom_ndim,
)

if TYPE_CHECKING:
    from typing import (  # TYPE_CHECKING,
        Any,
        Callable,
        Hashable,
        Literal,
        Mapping,
        Sequence,
    )

    # Iterable,
    from numpy.typing import DTypeLike
    from xarray.core import types as xr_types
    from xarray.core.coordinates import DataArrayCoordinates
    from xarray.core.indexes import Indexes

    from ._typing_compat import Self
    from .central import CentralMoments
    from .typing import (
        Mom_NDim,
        MomDims,
        Moments,
        MultiArray,
        MultiArrayVals,
        MyNDArray,
        XArrayAttrsType,
        XArrayCoordsType,
        XArrayDimsType,
        XArrayIndexesType,
        XArrayNameType,
    )


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
            msg = "must specify axis or dim"
            raise ValueError(msg)
        if default_axis is not None and default_dim is not None:
            msg = "can only specify one of default_axis or default_dim"
            raise ValueError(msg)

        if default_axis is not None:
            axis = default_axis
        else:
            dim = default_dim

    elif axis is not None and dim is not None:
        msg = "can only specify axis or dim"
        raise ValueError(msg)

    if dim is not None:
        if dim in dims:
            axis = dims.index(dim)
        else:
            msg = f"did not find '{dim}' in {dims}"
            raise ValueError(msg)
    elif axis is not None:
        if isinstance(axis, str):
            msg = f"Using string value for axis is deprecated.  Please use `dim` option instead.  Passed {axis} of type {type(axis)}"
            raise ValueError(msg)
        dim = dims[axis]
    else:  # pragma: no cover
        msg = f"unknown dim {dim} and axis {axis}"
        raise TypeError(msg)

    return axis, dim


def _move_mom_dims_to_end(
    x: xr.DataArray, mom_dims: MomDims, mom_ndim: Mom_NDim | None = None
) -> xr.DataArray:
    if mom_dims is not None:
        # if isinstance(mom_dims, str):
        #     mom_dims = (mom_dims,)
        # else:
        #     mom_dims = tuple(mom_dims)
        mom_dims = (mom_dims,) if isinstance(mom_dims, str) else tuple(mom_dims)  # type: ignore[arg-type]

        if mom_ndim is not None and len(mom_dims) != mom_ndim:
            msg = f"len(mom_dims)={len(mom_dims)} not equal to mom_ndim={mom_ndim}"
            raise ValueError(msg)

        order = (..., *mom_dims)
        x = x.transpose(*order)

    return x


# * xcentral moments/comoments
def _xcentral_moments(
    vals: xr.DataArray,
    mom: Moments,
    w: xr.DataArray | None = None,
    axis: int | None = None,
    dim: Hashable | None = None,
    mom_dims: MomDims | None = None,
) -> xr.DataArray:
    x = vals
    if not isinstance(x, xr.DataArray):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError

    if isinstance(mom, tuple):
        mom = mom[0]

    if mom_dims is None:
        mom_dims = ("mom_0",)
    elif isinstance(mom_dims, str):
        mom_dims = (mom_dims,)
    if len(mom_dims) != 1:  # type: ignore[arg-type]
        raise ValueError

    if w is None:
        # fmt: off
        w = xr.ones_like(x)  # pyright: ignore[reportUnknownMemberType] # needed for python=3.8
        # fmt: on
    else:
        w = xr.DataArray(w).broadcast_like(x)

        axis, dim = _select_axis_dim(dims=x.dims, axis=axis, dim=dim, default_axis=0)

    if TYPE_CHECKING:
        mom_dims = cast("tuple[Hashable]", mom_dims)
        dim = cast(str, dim)

    wsum = w.sum(dim=dim)
    wsum_inv = 1.0 / wsum

    # fmt: off
    xave = xr_dot(w, x, dim=dim) * wsum_inv  # pyright: ignore[reportUnknownMemberType]
    p = xr.DataArray(
        np.arange(0, mom + 1), dims=mom_dims  # pyright: ignore[reportUnknownMemberType]
    )
    dx = (x - xave) ** p
    out = xr_dot(w, dx, dim=dim) * wsum_inv  # pyright: ignore[reportUnknownMemberType]
    # fmt: on

    out.loc[{mom_dims[0]: 0}] = wsum
    out.loc[{mom_dims[0]: 1}] = xave

    # ensure in correct order
    out = out.transpose(..., *mom_dims)
    return cast(xr.DataArray, out)


def _xcentral_comoments(  # noqa: C901
    vals: tuple[xr.DataArray, xr.DataArray],
    mom: Moments,
    w: xr.DataArray | None = None,
    axis: int | None = None,
    dim: Hashable | None = None,
    broadcast: bool = False,
    mom_dims: tuple[Hashable, Hashable] | None = None,
) -> xr.DataArray:
    """Calculate central co-mom (covariance, etc) along axis."""
    mom = (mom, mom) if isinstance(mom, int) else tuple(mom)  # type: ignore[assignment]

    assert isinstance(mom, tuple)  # noqa: S101  # pragma: no cover

    if len(mom) != 2:
        raise ValueError
    if not isinstance(vals, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError
    if len(vals) != 2:
        raise ValueError

    x, y = vals

    if not isinstance(x, xr.DataArray):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError

    w = xr.ones_like(x) if w is None else xr.DataArray(w).broadcast_like(x)  # pyright: ignore[reportUnknownMemberType] # needed for python=3.8

    if broadcast:
        y = xr.DataArray(y).broadcast_like(x)
    elif not isinstance(y, xr.DataArray):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError
    else:
        y = y.transpose(*x.dims)
        if y.shape != x.shape or x.dims != y.dims:
            raise ValueError

    axis, dim = _select_axis_dim(dims=x.dims, axis=axis, dim=dim, default_axis=0)

    if mom_dims is None:
        mom_dims = ("mom_0", "mom_1")

    if len(mom_dims) != 2:
        raise ValueError

    if TYPE_CHECKING:
        dim = cast(str, dim)

    wsum = w.sum(dim=dim)
    wsum_inv = 1.0 / wsum

    xy = (x, y)

    # fmt: off
    xave = [xr_dot(w, xx, dim=dim) * wsum_inv for xx in xy]  # pyright: ignore[reportUnknownMemberType]
    p = [
        xr.DataArray(np.arange(0, mom + 1), dims=dim)  # type: ignore[arg-type, unused-ignore]  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        for mom, dim in zip(mom, mom_dims)
    ]

    dx = [(xx - xxave) ** pp for xx, xxave, pp in zip(xy, xave, p)]
    out = xr_dot(w, dx[0], dx[1], dim=dim) * wsum_inv  # pyright: ignore[reportUnknownMemberType]
    # fmt: on

    out.loc[{mom_dims[0]: 0, mom_dims[1]: 0}] = wsum
    out.loc[{mom_dims[0]: 1, mom_dims[1]: 0}] = xave[0]
    out.loc[{mom_dims[0]: 0, mom_dims[1]: 1}] = xave[1]

    out = out.transpose(..., *mom_dims)
    return cast(xr.DataArray, out)


@docfiller.decorate
def xcentral_moments(
    x: xr.DataArray | tuple[xr.DataArray, xr.DataArray],
    mom: Moments,
    *,
    w: xr.DataArray | None = None,
    axis: int | None = None,
    dim: Hashable | None = None,
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

    kwargs = {
        "vals": x,
        "mom": mom,
        "w": w,
        "axis": axis,
        "dim": dim,
        "mom_dims": mom_dims,
    }
    if len(mom) == 1:
        out = _xcentral_moments(**kwargs)  # type: ignore[arg-type]
    else:
        kwargs["broadcast"] = broadcast
        out = _xcentral_comoments(**kwargs)  # type: ignore[arg-type]

    return out


# --- * xCentralMoments-----------------------------------------------------------------
docfiller_abc = docfiller.factory_from_parent(CentralMomentsABC)
docfiller_inherit_abc = docfiller.factory_inherit_from_parent(CentralMomentsABC)


@docfiller(CentralMomentsABC)  # noqa: PLR0904
class xCentralMoments(CentralMomentsABC[xr.DataArray]):  # noqa: N801
    """
    Notes
    -----
    Most methods are wrapped to accept :class:`xarray.DataArray` object.
    """

    __slots__ = ("_cache", "_data", "_data_flat", "_mom_ndim", "_xdata")

    def __init__(self, data: xr.DataArray, mom_ndim: Mom_NDim = 1) -> None:
        if mom_ndim not in {1, 2}:
            msg = (
                "mom_ndim must be either 1 (for central moments)"
                "or 2 (for central comoments)"
            )
            raise ValueError(msg)

        if not isinstance(data, xr.DataArray):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = (
                "data must be a xarray.DataArray. "
                "See xCentralMoments.from_data for wrapping numpy arrays"
            )
            raise TypeError(msg)

        if data.ndim < mom_ndim:
            msg = "not enough dimensions in data"
            raise ValueError(msg)

        self._mom_ndim = mom_ndim
        self._data: MyNDArray = data.to_numpy()  # pyright: ignore[reportUnknownMemberType]
        self._data_flat = self._data.reshape(self.shape_flat)
        self._data = self._data_flat.reshape(data.shape)  # ensure same data
        self._xdata = data.copy(data=self._data)

        if any(m <= 0 for m in self.mom):
            msg = "moments must be positive"
            raise ValueError(msg)

        self._validate_data()
        if (
            self._xdata.to_numpy() is not self._data  # pyright: ignore[reportUnknownMemberType]
            or self._xdata.variable._data is not self._data  # noqa: SLF001  # pyright: ignore[reportUnknownMemberType, reportPrivateUsage]
        ):  # pragma: no cover
            raise ValueError

        self._cache: dict[str, Any] = {}

    # ** xarray attributes
    def to_values(self) -> xr.DataArray:
        """Underlying :class:`xarray.DataArray`."""
        return self._xdata

    def to_dataarray(self) -> xr.DataArray:
        """Underlying :class:`xarray.DataArray`."""
        return self._xdata

    @property
    def attrs(self) -> dict[Any, Any]:
        """Attributes of values."""
        return self._xdata.attrs

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """Dimensions of values."""
        return self._xdata.dims

    @property
    def coords(self) -> DataArrayCoordinates[Any]:
        """Coordinates of values."""
        return self._xdata.coords  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

    @property
    def name(self) -> Hashable:
        """Name of values."""
        return self._xdata.name

    @property
    def indexes(self) -> Indexes[Any]:  # pragma: no cover
        """Indexes of values."""
        return self._xdata.indexes  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

    @property
    def sizes(self) -> Mapping[Hashable, int]:
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
    # @cached.prop
    # def _template_val(self) -> xr.DataArray:
    #     """Template for values part of data."""
    #     return self._xdata[self._weight_index]

    def _wrap_like(self, x: MyNDArray) -> xr.DataArray:
        return self._xdata.copy(data=x)

    @docfiller_abc()
    def new_like(
        self,
        data: MyNDArray | xr.DataArray | None = None,
        *,
        copy: bool = False,
        copy_kws: Mapping[str, Any] | None = None,
        verify: bool = False,
        strict: bool = False,
        **kwargs: Any,
    ) -> Self:
        if data is None:
            # fmt: off
            data = xr.zeros_like(self._xdata)  # pyright: ignore[reportUnknownMemberType] # needed for python=3.8
            # fmt: on
            copy = verify = False

        kwargs.setdefault("mom_ndim", self.mom_ndim)

        if strict:
            kwargs = {
                "mom": self.mom,
                "val_shape": self.val_shape,
                "dtype": self.dtype,
                **kwargs,
            }

        if isinstance(data, np.ndarray):
            kwargs.setdefault("template", self._xdata)

        return type(self).from_data(
            data=data,
            copy=copy,
            copy_kws=copy_kws,
            verify=verify,
            **kwargs,
        )

    # ** Access to underlying statistics
    # TODO(wpk): add overload
    def _single_index_selector(
        self,
        val: int,
        dim_combined: str = "variable",
    ) -> dict[Hashable, Any]:
        idxs = self._single_index(val)[-self.mom_ndim :]
        return {
            dim: (idx if self._mom_ndim == 1 else xr.DataArray(idx, dims=dim_combined))
            for dim, idx in zip(self.mom_dims, idxs)
        }

    def _single_index_dataarray(
        self,
        val: int,
        dim_combined: str = "variable",
        coords_combined: str | Sequence[Hashable] | None = None,
    ) -> xr.DataArray:
        if coords_combined is None:
            coords_combined = self.mom_dims

        selector = self._single_index_selector(
            val=val,
            dim_combined=dim_combined,
        )

        out = self.to_dataarray().isel(selector)
        if self._mom_ndim > 1:
            out = out.assign_coords(coords={dim_combined: list(coords_combined)})  # pyright: ignore[reportUnknownMemberType]
        return out

    def mean(
        self,
        dim_combined: str = "variable",
        coords_combined: str | Sequence[Hashable] | None = None,
    ) -> xr.DataArray:
        """Return mean/first moment(s) of data."""
        return self._single_index_dataarray(
            val=1, dim_combined=dim_combined, coords_combined=coords_combined
        )

    def var(
        self,
        dim_combined: str = "variable",
        coords_combined: str | Sequence[Hashable] | None = None,
    ) -> xr.DataArray:
        """Return variance (second central moment) of data."""
        return self._single_index_dataarray(
            val=2, dim_combined=dim_combined, coords_combined=coords_combined
        )

    # ** xarray specific methods
    def _wrap_xarray_method(self, _method: str, *args: Any, **kwargs: Any) -> Self:
        xdata = getattr(self._xdata, _method)(*args, **kwargs)
        return self.new_like(data=xdata, strict=False)

    def assign_coords(
        self, coords: XArrayCoordsType = None, **coords_kwargs: Any
    ) -> Self:
        """Assign coordinates to data and return new object."""
        return self._wrap_xarray_method("assign_coords", coords=coords, **coords_kwargs)

    def assign_attrs(self, *args: Any, **kwargs: Any) -> Self:
        """Assign attributes to data and return new object."""
        return self._wrap_xarray_method("assign_attrs", *args, **kwargs)

    def rename(
        self,
        new_name_or_name_dict: Hashable | Mapping[Any, Hashable] | None = None,
        **names: Hashable,
    ) -> Self:
        """Rename object."""
        return self._wrap_xarray_method(
            "rename", new_name_or_name_dict=new_name_or_name_dict, **names
        )

    def stack(
        self,
        dimensions: Mapping[Any, Sequence[Hashable]] | None = None,
        *,
        _order: bool = True,
        _verify: bool = False,
        _copy: bool = False,
        _check_mom: bool = True,
        _kws: Mapping[str, Any] | None = None,
        **dimensions_kwargs: Any,
    ) -> Self:
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
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> da = xCentralMoments.from_vals(rng.random((10, 2, 3)), mom=2, axis=0)
        >>> da
        <xCentralMoments(val_shape=(2, 3), mom=(2,))>
        <xarray.DataArray (dim_0: 2, dim_1: 3, mom_0: 3)> Size: 144B
        array([[[10.    ,  0.5205,  0.0452],
                [10.    ,  0.4438,  0.0734],
                [10.    ,  0.5038,  0.1153]],
        <BLANKLINE>
               [[10.    ,  0.5238,  0.1272],
                [10.    ,  0.628 ,  0.0524],
                [10.    ,  0.412 ,  0.0865]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0
        >>> da_stack = da.stack(z=["dim_0", "dim_1"])
        >>> da_stack
        <xCentralMoments(val_shape=(6,), mom=(2,))>
        <xarray.DataArray (z: 6, mom_0: 3)> Size: 144B
        array([[10.    ,  0.5205,  0.0452],
               [10.    ,  0.4438,  0.0734],
               [10.    ,  0.5038,  0.1153],
               [10.    ,  0.5238,  0.1272],
               [10.    ,  0.628 ,  0.0524],
               [10.    ,  0.412 ,  0.0865]])
        Coordinates:
          * z        (z) object 48B MultiIndex
          * dim_0    (z) int64 48B 0 0 0 1 1 1
          * dim_1    (z) int64 48B 0 1 2 0 1 2
        Dimensions without coordinates: mom_0

        And unstack

        >>> da_stack.unstack("z")
        <xCentralMoments(val_shape=(2, 3), mom=(2,))>
        <xarray.DataArray (dim_0: 2, dim_1: 3, mom_0: 3)> Size: 144B
        array([[[10.    ,  0.5205,  0.0452],
                [10.    ,  0.4438,  0.0734],
                [10.    ,  0.5038,  0.1153]],
        <BLANKLINE>
               [[10.    ,  0.5238,  0.1272],
                [10.    ,  0.628 ,  0.0524],
                [10.    ,  0.412 ,  0.0865]]])
        Coordinates:
          * dim_0    (dim_0) int64 16B 0 1
          * dim_1    (dim_1) int64 24B 0 1 2
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
        *,
        sparse: bool = False,
        _order: bool = True,
        _copy: bool = False,
        _verify: bool = False,
        _check_mom: bool = True,
        _kws: Mapping[str, Any] | None = None,
    ) -> Self:
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
        *,
        _order: bool = False,
        _copy: bool = False,
        _verify: bool = False,
        _check_mom: bool = True,
        _kws: Mapping[str, Any] | None = None,
        **indexers_kws: Any,
    ) -> Self:
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
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> da = xCentralMoments.from_vals(
        ...     rng.random((10, 3)), axis=0, dims="x", coords=dict(x=list("abc")), mom=2
        ... )
        >>> da
        <xCentralMoments(val_shape=(3,), mom=(2,))>
        <xarray.DataArray (x: 3, mom_0: 3)> Size: 72B
        array([[10.    ,  0.5248,  0.1106],
               [10.    ,  0.5688,  0.0689],
               [10.    ,  0.5094,  0.1198]])
        Coordinates:
          * x        (x) <U1 12B 'a' 'b' 'c'
        Dimensions without coordinates: mom_0

        Select by value

        >>> da.sel(x="a")
        <xCentralMoments(val_shape=(), mom=(2,))>
        <xarray.DataArray (mom_0: 3)> Size: 24B
        array([10.    ,  0.5248,  0.1106])
        Coordinates:
            x        <U1 4B 'a'
        Dimensions without coordinates: mom_0
        >>> da.sel(x=["a", "c"])
        <xCentralMoments(val_shape=(2,), mom=(2,))>
        <xarray.DataArray (x: 2, mom_0: 3)> Size: 48B
        array([[10.    ,  0.5248,  0.1106],
               [10.    ,  0.5094,  0.1198]])
        Coordinates:
          * x        (x) <U1 8B 'a' 'c'
        Dimensions without coordinates: mom_0


        Select by position

        >>> da.isel(x=0)
        <xCentralMoments(val_shape=(), mom=(2,))>
        <xarray.DataArray (mom_0: 3)> Size: 24B
        array([10.    ,  0.5248,  0.1106])
        Coordinates:
            x        <U1 4B 'a'
        Dimensions without coordinates: mom_0
        >>> da.isel(x=[0, 1])
        <xCentralMoments(val_shape=(2,), mom=(2,))>
        <xarray.DataArray (x: 2, mom_0: 3)> Size: 48B
        array([[10.    ,  0.5248,  0.1106],
               [10.    ,  0.5688,  0.0689]])
        Coordinates:
          * x        (x) <U1 8B 'a' 'b'
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
        missing_dims: xr_types.ErrorOptionsWithWarn = "raise",
        *,
        _order: bool = False,
        _copy: bool = False,
        _verify: bool = False,
        _check_mom: bool = True,
        _kws: Mapping[str, Any] | None = None,
        **indexers_kws: Any,
    ) -> Self:
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
            missing_dims=missing_dims,
            **indexers_kws,
        )

    def transpose(
        self,
        *dims: Hashable,
        transpose_coords: bool = True,
        missing_dims: xr_types.ErrorOptionsWithWarn = "raise",
        _copy: bool = False,
        _verify: bool = False,
        _check_mom: bool = True,
        _kws: Mapping[str, Any] | None = None,
    ) -> Self:
        """
        Transpose dimensions of data.

        Notes
        -----
        ``mom_dims`` will always be put at the end of the array.

        See Also
        --------
        pipe
        xarray.DataArray.transpose
        """
        # make sure dims are last

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
        from .central import CentralMoments

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
        xCentralMoments.from_centralmoments
        """
        out = self.to_centralmoments()
        if not np.shares_memory(out.to_numpy(), self._data):  # pragma: no cover
            raise ValueError
        return out

    @classmethod
    @docfiller.decorate
    def from_centralmoments(
        cls,
        obj: CentralMoments,
        *,
        dims: XArrayDimsType = None,
        attrs: XArrayAttrsType = None,
        coords: XArrayCoordsType = None,
        name: XArrayNameType = None,
        indexes: XArrayIndexesType = None,
        mom_dims: MomDims | None = None,
        template: xr.DataArray | None = None,
        copy: bool = False,
    ) -> Self:
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
            template=template,
            copy=copy,
        )
        return cls(data=data, mom_ndim=obj.mom_ndim)

    @classmethod
    def _wrap_centralmoments_method(
        cls,
        _method_name: str,
        *args: Any,
        dims: XArrayDimsType = None,
        mom_dims: MomDims | None = None,
        attrs: XArrayAttrsType = None,
        coords: XArrayCoordsType = None,
        indexes: XArrayIndexesType = None,
        name: XArrayNameType = None,
        template: xr.DataArray | None = None,
        copy_final: bool = False,
        **kwargs: Any,
    ) -> Self:
        from .central import CentralMoments

        method = cast(
            "Callable[..., CentralMoments]", getattr(CentralMoments, _method_name)
        )

        centralmoments = method(*args, **kwargs)

        return cls.from_centralmoments(
            obj=centralmoments,
            dims=dims,
            attrs=attrs,
            coords=coords,
            name=name,
            indexes=indexes,
            mom_dims=mom_dims,
            template=template,
            copy=copy_final,
        )

    # ** Push/verify
    def _xverify_value(  # noqa: C901,PLR0912,PLR0915
        self,
        *,
        x: MultiArray[xr.DataArray],
        target: str | MyNDArray | xr.DataArray | None = None,
        dim: Hashable | None = None,
        axis: int | None = None,
        broadcast: bool = False,
        expand: bool = False,
        shape_flat: Any | None = None,
    ) -> tuple[MyNDArray, xr.DataArray]:
        if isinstance(x, xr.DataArray):
            x = x.astype(dtype=self.dtype, copy=False)  # pyright: ignore[reportUnknownMemberType]
        else:
            x = np.asarray(x, dtype=self.dtype)

        target_dims: None | tuple[Hashable, ...] = None
        if isinstance(target, str):
            if dim is not None and axis is not None:
                # prefer dim over axis
                axis = None

            if dim is not None or axis is not None:
                if not isinstance(
                    x, xr.DataArray
                ):  # pragma: no cover (probably not needed)
                    raise TypeError
                axis, dim = _select_axis_dim(dims=x.dims, axis=axis, dim=dim)

            if target == "val":
                target_dims = self.val_dims
            elif target == "vals":
                target_dims = (dim, *self.val_dims)
            elif target == "data":
                target_dims = self.dims
            elif target == "datas":
                target_dims = (dim, *self.dims)
            else:
                msg = f"unknown option to xverify {target}"
                raise ValueError(msg)

        if target_dims is not None:
            # no broadcast in this cast
            if not isinstance(
                x, xr.DataArray
            ):  # pragma: no cover (might not be needed)
                raise TypeError
            target_shape = tuple(
                x.sizes[k] if k == dim else self.sizes[k] for k in target_dims
            )

            # make sure in correct order
            x = x.transpose(*target_dims)
            target_output = x

        elif not isinstance(target, xr.DataArray):  # pragma: no cover
            raise TypeError
        else:
            target = target.astype(dtype=self.dtype, copy=False)  # pyright: ignore[reportUnknownMemberType]

            target_dims = target.dims
            target_shape = target.shape

            target_output = target

            if dim is not None and axis is not None:  # pragma: no cover
                axis = None

            if dim is not None or axis is not None:
                # this is hackish
                # in this case, target should already be in correct order
                # so just steal from target_shape
                dim = target_dims[0]

            if isinstance(x, xr.DataArray):
                x = x.broadcast_like(target) if broadcast else x.transpose(*target_dims)

            else:
                # only things this can be is either a scalar or
                # array with same size as target
                x = np.asarray(x)
                if x.ndim == 0 and broadcast and expand:
                    x = xr.DataArray(x).broadcast_like(target)

                elif (
                    x.ndim == 1 and len(x) == target.sizes[dim] and broadcast and expand
                ):
                    x = xr.DataArray(x, dims=dim).broadcast_like(target)  # type: ignore[arg-type,unused-ignore] # pyright: ignore[reportArgumentType]

                elif x.shape == target.shape:
                    x = xr.DataArray(x, dims=target.dims)
                else:
                    raise ValueError

        values: MyNDArray = x.to_numpy() if isinstance(x, xr.DataArray) else x  # pyright: ignore[reportUnknownMemberType, reportUnnecessaryIsInstance]

        # check shape
        if values.shape != target_shape:  # pragma: no cover
            raise ValueError
        if dim is None:
            nrec: tuple[int, ...] = ()
        else:
            nrec = (x.sizes[dim],)

        if shape_flat is not None:
            values = values.reshape(nrec + shape_flat)

        if values.ndim == 0:
            values = values[()]

        return values, target_output

    def _verify_value(
        self,
        *,
        x: MultiArray[xr.DataArray],
        target: str | MyNDArray | xr.DataArray,
        shape_flat: tuple[int, ...],
        axis: int | None = None,
        dim: Hashable | None = None,
        broadcast: bool = False,
        expand: bool = False,
        other: MyNDArray | None = None,
    ) -> tuple[MyNDArray, MyNDArray | xr.DataArray]:
        if isinstance(x, xr.DataArray) or isinstance(target, xr.DataArray):
            return self._xverify_value(
                x=x,
                target=target,
                axis=axis,
                dim=dim,
                # dim=axis,
                broadcast=broadcast,
                expand=expand,
                shape_flat=shape_flat,
            )

        if axis is not None and not isinstance(axis, int):  # pyright: ignore[reportUnnecessaryIsInstance]  # pragma: no cover
            msg = f"Error with axis value {axis}"
            raise ValueError(msg)

        return self.centralmoments_view._verify_value(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
            x=x,
            target=target,
            axis=axis,
            broadcast=broadcast,
            expand=expand,
            shape_flat=shape_flat,
            other=other,
        )

    @docfiller_inherit_abc()
    def push_data(
        self,
        data: MultiArrayVals[xr.DataArray],
    ) -> Self:
        return super().push_data(data=data)

    @docfiller_inherit_abc()
    def push_datas(
        self,
        datas: MultiArrayVals[xr.DataArray],
        axis: int | None = None,
        dim: Hashable | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> Self:
        return super().push_datas(datas=datas, axis=axis, dim=dim)

    @docfiller_inherit_abc()
    def push_val(
        self,
        x: MultiArray[xr.DataArray]
        | tuple[MultiArray[xr.DataArray], MultiArray[xr.DataArray]],
        w: MultiArray[xr.DataArray] | None = None,
        broadcast: bool = False,
        **kwargs: Any,
    ) -> Self:
        return super().push_val(x=x, w=w, broadcast=broadcast, **kwargs)

    @docfiller_inherit_abc()
    def push_vals(
        self,
        x: MultiArrayVals[xr.DataArray]
        | tuple[MultiArrayVals[xr.DataArray], MultiArrayVals[xr.DataArray]],
        w: MultiArray[xr.DataArray] | None = None,
        axis: int | None = None,
        broadcast: bool = False,
        dim: Hashable | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> Self:
        return super().push_vals(
            x=x,
            w=w,
            axis=axis,
            broadcast=broadcast,
            dim=dim,
        )

    # ** Manipulation
    @overload
    def resample_and_reduce(
        self,
        nrep: int | None = ...,
        *,
        full_output: Literal[False] = ...,
        dim: Hashable | None = ...,
        rep_dim: str = ...,
        axis: int | None = ...,
        freq: MyNDArray | None = ...,
        indices: MyNDArray | None = ...,
        parallel: bool = ...,
        resample_kws: Mapping[str, Any] | None = ...,
        rng: np.random.Generator | None = ...,
        **kwargs: Any,
    ) -> Self:
        ...

    @overload
    def resample_and_reduce(
        self,
        nrep: int | None = ...,
        *,
        full_output: Literal[True],
        dim: Hashable | None = ...,
        rep_dim: str = ...,
        axis: int | None = ...,
        freq: MyNDArray | None = ...,
        indices: MyNDArray | None = ...,
        parallel: bool = ...,
        resample_kws: Mapping[str, Any] | None = ...,
        rng: np.random.Generator | None = ...,
        **kwargs: Any,
    ) -> tuple[Self, MyNDArray]:
        ...

    @overload
    def resample_and_reduce(
        self,
        nrep: int | None = ...,
        *,
        full_output: bool,
        dim: Hashable | None = ...,
        rep_dim: str = ...,
        axis: int | None = ...,
        freq: MyNDArray | None = ...,
        indices: MyNDArray | None = ...,
        parallel: bool = ...,
        resample_kws: Mapping[str, Any] | None = ...,
        rng: np.random.Generator | None = ...,
        **kwwargs: Any,
    ) -> Self | tuple[Self, MyNDArray]:
        ...

    @docfiller.decorate
    def resample_and_reduce(
        self,
        nrep: int | None = None,
        *,
        full_output: bool = False,
        dim: Hashable | None = None,
        rep_dim: str = "rep",
        axis: int | None = None,
        freq: MyNDArray | None = None,
        indices: MyNDArray | None = None,
        parallel: bool = True,
        resample_kws: Mapping[str, Any] | None = None,
        rng: np.random.Generator | None = None,
        **kwargs: Any,
    ) -> Self | tuple[Self, MyNDArray]:
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
        {rng}
        **kwargs
            Arguments to :meth:`CentralMoments.resample_and_reduce`

        Returns
        -------
        output : {klass}

        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> da = xCentralMoments.from_vals(
        ...     rng.random((10, 3)), mom=3, axis=0, dims="rec"
        ... )
        >>> da
        <xCentralMoments(val_shape=(3,), mom=(3,))>
        <xarray.DataArray (rec: 3, mom_0: 4)> Size: 96B
        array([[ 1.0000e+01,  5.2485e-01,  1.1057e-01, -4.6282e-03],
               [ 1.0000e+01,  5.6877e-01,  6.8876e-02, -1.2745e-02],
               [ 1.0000e+01,  5.0944e-01,  1.1978e-01, -1.4644e-02]])
        Dimensions without coordinates: rec, mom_0

        Note that for reproducible results, must set numba random
        seed as well

        >>> da_resamp, freq = da.resample_and_reduce(
        ...     nrep=5,
        ...     dim="rec",
        ...     full_output=True,
        ...     rng=rng,
        ... )
        >>> da_resamp
        <xCentralMoments(val_shape=(5,), mom=(3,))>
        <xarray.DataArray (rep: 5, mom_0: 4)> Size: 160B
        array([[ 3.0000e+01,  5.0944e-01,  1.1978e-01, -1.4644e-02],
               [ 3.0000e+01,  5.3435e-01,  1.0038e-01, -1.2329e-02],
               [ 3.0000e+01,  5.2922e-01,  1.0360e-01, -1.6009e-02],
               [ 3.0000e+01,  5.5413e-01,  8.3204e-02, -1.1267e-02],
               [ 3.0000e+01,  5.4899e-01,  8.6627e-02, -1.5407e-02]])
        Dimensions without coordinates: rep, mom_0

        Alternatively, we can resample and reduce

        >>> from cmomy.resample import freq_to_indices
        >>> indices = freq_to_indices(freq)
        >>> da.sel(rec=xr.DataArray(indices, dims=["rep", "rec"])).reduce(dim="rec")
        <xCentralMoments(val_shape=(5,), mom=(3,))>
        <xarray.DataArray (rep: 5, mom_0: 4)> Size: 160B
        array([[ 3.0000e+01,  5.0944e-01,  1.1978e-01, -1.4644e-02],
               [ 3.0000e+01,  5.3435e-01,  1.0038e-01, -1.2329e-02],
               [ 3.0000e+01,  5.2922e-01,  1.0360e-01, -1.6009e-02],
               [ 3.0000e+01,  5.5413e-01,  8.3204e-02, -1.1267e-02],
               [ 3.0000e+01,  5.4899e-01,  8.6627e-02, -1.5407e-02]])
        Dimensions without coordinates: rep, mom_0

        """
        self._raise_if_scalar()

        axis, dim = _select_axis_dim(dims=self.dims, axis=axis, dim=dim)

        if dim in self.mom_dims:
            msg = f"can only resample from value dimensions {self.val_dims}"
            raise ValueError(msg)

        # Final form will move `dim` to front of array.
        # this will be replaced by rep_dimension
        template = self.to_dataarray().isel({dim: 0})

        out, freq = self.centralmoments_view.resample_and_reduce(
            freq=freq,
            indices=indices,
            nrep=nrep,
            axis=axis,
            parallel=parallel,
            resample_kws=resample_kws,
            full_output=True,
            rng=rng,
            **kwargs,
        )

        new = self.from_centralmoments(
            obj=out,
            dims=(rep_dim, *template.dims),
            mom_dims=None,
            attrs=template.attrs,
            coords=template.coords,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            name=template.name,
        )

        if full_output:
            return new, freq

        return new

    @docfiller.decorate
    def reduce(
        self, dim: Hashable | None = None, axis: int | None = None, **kwargs: Any
    ) -> Self:
        """
        Parameters
        ----------
        {dim}
        {axis}
        **kwargs
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
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> da = xCentralMoments.from_vals(rng.random((10, 2, 3)), axis=0, mom=2)
        >>> da.reduce(dim="dim_0")
        <xCentralMoments(val_shape=(3,), mom=(2,))>
        <xarray.DataArray (dim_1: 3, mom_0: 3)> Size: 72B
        array([[20.    ,  0.5221,  0.0862],
               [20.    ,  0.5359,  0.0714],
               [20.    ,  0.4579,  0.103 ]])
        Dimensions without coordinates: dim_1, mom_0
        """
        self._raise_if_scalar()
        axis, dim = _select_axis_dim(dims=self.dims, axis=axis, dim=dim)
        return type(self).from_datas(
            self.to_values(), mom_ndim=self.mom_ndim, axis=axis, **kwargs
        )

    @docfiller.decorate
    def block(
        self,
        block_size: int | None,
        dim: Hashable | None = None,
        axis: int | None = None,
        coords_policy: Literal["first", "last", None] = "first",
        **kwargs: Any,
    ) -> Self:
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
        **kwargs
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
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> x = rng.random((10, 10))
        >>> da = xCentralMoments.from_vals(x, mom=2)
        >>> da
        <xCentralMoments(val_shape=(10,), mom=(2,))>
        <xarray.DataArray (dim_0: 10, mom_0: 3)> Size: 240B
        array([[10.    ,  0.6247,  0.0583],
               [10.    ,  0.3938,  0.0933],
               [10.    ,  0.425 ,  0.1003],
               [10.    ,  0.5   ,  0.117 ],
               [10.    ,  0.5606,  0.0446],
               [10.    ,  0.5612,  0.0861],
               [10.    ,  0.531 ,  0.0731],
               [10.    ,  0.8403,  0.0233],
               [10.    ,  0.5097,  0.103 ],
               [10.    ,  0.5368,  0.085 ]])
        Dimensions without coordinates: dim_0, mom_0

        >>> da.block(block_size=5, dim="dim_0")
        <xCentralMoments(val_shape=(2,), mom=(2,))>
        <xarray.DataArray (dim_0: 2, mom_0: 3)> Size: 48B
        array([[50.    ,  0.5008,  0.0899],
               [50.    ,  0.5958,  0.0893]])
        Dimensions without coordinates: dim_0, mom_0

        This is equivalent to

        >>> xCentralMoments.from_vals(x.reshape(2, 50), mom=2, axis=1)
        <xCentralMoments(val_shape=(2,), mom=(2,))>
        <xarray.DataArray (dim_0: 2, mom_0: 3)> Size: 48B
        array([[50.    ,  0.5268,  0.0849],
               [50.    ,  0.5697,  0.0979]])
        Dimensions without coordinates: dim_0, mom_0


        The coordinate policy can be useful to keep coordinates:

        >>> da2 = da.assign_coords(dim_0=range(10))
        >>> da2.block(5, dim="dim_0", coords_policy="first")
        <xCentralMoments(val_shape=(2,), mom=(2,))>
        <xarray.DataArray (dim_0: 2, mom_0: 3)> Size: 48B
        array([[50.    ,  0.5008,  0.0899],
               [50.    ,  0.5958,  0.0893]])
        Coordinates:
          * dim_0    (dim_0) int64 16B 0 5
        Dimensions without coordinates: mom_0
        >>> da2.block(5, dim="dim_0", coords_policy="last")
        <xCentralMoments(val_shape=(2,), mom=(2,))>
        <xarray.DataArray (dim_0: 2, mom_0: 3)> Size: 48B
        array([[50.    ,  0.5008,  0.0899],
               [50.    ,  0.5958,  0.0893]])
        Coordinates:
          * dim_0    (dim_0) int64 16B 4 9
        Dimensions without coordinates: mom_0
        >>> da2.block(5, dim="dim_0", coords_policy=None)
        <xCentralMoments(val_shape=(2,), mom=(2,))>
        <xarray.DataArray (dim_0: 2, mom_0: 3)> Size: 48B
        array([[50.    ,  0.5008,  0.0899],
               [50.    ,  0.5958,  0.0893]])
        Dimensions without coordinates: dim_0, mom_0

        """
        self._raise_if_scalar()
        axis, dim = _select_axis_dim(dims=self.dims, axis=axis, dim=dim, default_axis=0)

        if block_size is None:
            block_size = self.sizes[dim]
            nblock = 1
        else:
            nblock = self.sizes[dim] // block_size

        start = 0 if coords_policy == "first" else block_size - 1

        # get template values
        template = (
            self.to_dataarray()
            .isel({dim: slice(start, block_size * nblock, block_size)})
            .transpose(dim, ...)
        )

        if coords_policy is None:
            if not isinstance(dim, str):  # pragma: no cover
                raise TypeError
            template = template.drop_vars(dim)

        central = self.centralmoments_view.block(
            block_size=block_size, axis=axis, **kwargs
        )

        return self.from_centralmoments(obj=central, template=template)

    # ** Constructors
    @classmethod
    @docfiller_inherit_abc()
    def zeros(
        cls,
        *,
        mom: Moments | None = None,
        mom_ndim: Mom_NDim | None = None,
        val_shape: tuple[int, ...] | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        zeros_kws: Mapping[str, Any] | None = None,
        dims: XArrayDimsType = None,
        coords: XArrayCoordsType = None,
        attrs: XArrayAttrsType = None,
        name: XArrayNameType = None,
        indexes: XArrayIndexesType = None,
        template: xr.DataArray | None = None,
        mom_dims: MomDims | None = None,
        **kwargs: Any,
    ) -> Self:
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

        kwargs = {"copy": True, "verify": True, **kwargs}

        return cls.from_data(
            data=template,
            mom=mom,
            mom_ndim=mom_ndim,
            val_shape=val_shape,
            dtype=dtype,
            dims=dims,
            coords=coords,
            attrs=attrs,
            name=name,
            indexes=indexes,
            mom_dims=mom_dims,
            **kwargs,
        ).zero()

    @classmethod
    @docfiller_inherit_abc()
    def from_data(  # noqa: PLR0913
        cls,
        data: MyNDArray | xr.DataArray,
        *,
        mom: Moments | None = None,
        mom_ndim: Mom_NDim | None = None,
        val_shape: tuple[int, ...] | None = None,
        copy: bool = True,
        copy_kws: Mapping[str, Any] | None = None,
        verify: bool = False,
        dtype: DTypeLike | None = None,
        dims: XArrayDimsType = None,
        attrs: XArrayAttrsType = None,
        coords: XArrayCoordsType = None,
        name: XArrayNameType = None,
        indexes: XArrayIndexesType = None,
        mom_dims: MomDims | None = None,
        template: Any | None = None,
    ) -> Self:
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
            mom_ndim = select_mom_ndim(mom=mom, mom_ndim=mom_ndim)

            data = _move_mom_dims_to_end(data, mom_dims, mom_ndim)

            if verify:
                data_verified = data.astype(  # pyright: ignore[reportUnknownMemberType]
                    dtype=dtype or data.dtype,  # pyright: ignore[reportUnknownMemberType]
                    order="C",
                    copy=False,  # pyright: ignore[reportUnknownMemberType]
                )
            else:
                data_verified = data

            mom, mom_ndim = validate_mom_and_mom_ndim(
                mom=mom, mom_ndim=mom_ndim, shape=data_verified.shape
            )

            if val_shape is None:
                val_shape = data_verified.shape[:-mom_ndim]

            if data_verified.shape != val_shape + tuple(x + 1 for x in mom):
                msg = f"{data.shape} does not conform to {val_shape} and {mom}"
                raise ValueError(msg)

            if copy and data_verified is data:  # pragma: no cover
                if copy_kws is None:  # pragma: no cover
                    copy_kws = {}

                # to make sure copy has same format
                data_verified = data_verified.copy(
                    data=data_verified.to_numpy().copy(**copy_kws)  # pyright: ignore[reportUnknownMemberType]
                )

            return cls(data=data_verified, mom_ndim=mom_ndim)

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
    def from_datas(  # noqa: PLR0913
        cls,
        datas: MyNDArray | xr.DataArray,
        *,
        mom: Moments | None = None,
        mom_ndim: Mom_NDim | None = None,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        verify: bool = False,
        axis: int | None = None,
        dim: Hashable | None = None,
        dims: XArrayDimsType = None,
        attrs: XArrayAttrsType = None,
        coords: XArrayCoordsType = None,
        indexes: XArrayIndexesType = None,
        name: XArrayNameType = None,
        mom_dims: MomDims | None = None,
        template: xr.DataArray | None = None,
        **kwargs: Any,
    ) -> Self:
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
            axis, dim = _select_axis_dim(dims=datas.dims, axis=axis, dim=dim)
            mom_ndim = select_mom_ndim(mom=mom, mom_ndim=mom_ndim)
            # move moments to end and dim to beginning
            datas = _move_mom_dims_to_end(datas, mom_dims, mom_ndim).transpose(dim, ...)

            if verify:
                datas = datas.astype(order="C", dtype=dtype, copy=False)  # pyright: ignore[reportUnknownMemberType]

            new = cls.from_data(
                data=xr.zeros_like(datas.isel({dim: 0})),  # pyright: ignore[reportUnknownMemberType]
                mom=mom,
                mom_ndim=mom_ndim,
                val_shape=val_shape,
                dtype=dtype,
                verify=False,  # already did this...
                **kwargs,
            ).push_datas(datas, dim=dim)

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
                dims=dims,
                attrs=attrs,
                coords=coords,
                indexes=indexes,
                template=template,
                name=name,
                mom_dims=mom_dims,
                **kwargs,
            )

        return new

    @classmethod
    @docfiller_inherit_abc()
    def from_vals(  # noqa: PLR0913
        cls,
        x: MyNDArray
        | tuple[MyNDArray, MyNDArray]
        | xr.DataArray
        | tuple[xr.DataArray, xr.DataArray]
        | tuple[xr.DataArray, MyNDArray],
        mom: Moments,
        *,
        w: float | MyNDArray | xr.DataArray | None = None,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        broadcast: bool = False,
        axis: int | None = None,
        dim: Hashable | None = None,
        dims: XArrayDimsType = None,
        attrs: XArrayAttrsType = None,
        coords: XArrayCoordsType = None,
        indexes: XArrayIndexesType = None,
        name: XArrayNameType = None,
        mom_dims: MomDims | None = None,
        template: xr.DataArray | None = None,
        **kwargs: Any,
    ) -> Self:
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
        x0 = x[0] if isinstance(x, tuple) else x

        if isinstance(x0, xr.DataArray):
            mom_ndim = mom_to_mom_ndim(mom)
            axis, dim = _select_axis_dim(dims=x0.dims, axis=axis, dim=dim)

            if val_shape is None:
                val_shape = shape_reduce(shape=x0.shape, axis=axis)
            if dtype is None:
                dtype = x0.dtype  # pyright: ignore[reportUnknownMemberType]

            template = x0.isel({dim: 0})

            dims = template.dims
            if coords is None:
                coords = {}
            coords = {**template.coords, **coords}  # type: ignore[dict-item] # pyright: ignore[reportUnknownMemberType,reportGeneralTypeIssues]
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
                **kwargs,
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
                **kwargs,
            )

        return new

    @overload
    @classmethod
    def from_resample_vals(
        cls,
        x: MyNDArray
        | tuple[MyNDArray, MyNDArray]
        | xr.DataArray
        | tuple[xr.DataArray, xr.DataArray],
        mom: Moments,
        *,
        full_output: Literal[False] = ...,
        nrep: int | None = ...,
        freq: MyNDArray | None = ...,
        indices: MyNDArray | None = ...,
        w: MyNDArray | xr.DataArray | None = ...,
        dim: Hashable | None = ...,
        rep_dim: str = ...,
        dtype: DTypeLike | None = ...,
        broadcast: bool = ...,
        parallel: bool = ...,
        resample_kws: Mapping[str, Any] | None = ...,
        rng: np.random.Generator | None = ...,
        axis: int | None = ...,
        # xarray specific
        dims: XArrayDimsType = ...,
        attrs: XArrayAttrsType = ...,
        coords: XArrayCoordsType = ...,
        indexes: XArrayIndexesType = ...,
        name: XArrayNameType = ...,
        mom_dims: MomDims | None = ...,
        **kwargs: Any,
    ) -> Self:
        ...

    @overload
    @classmethod
    def from_resample_vals(
        cls,
        x: MyNDArray
        | tuple[MyNDArray, MyNDArray]
        | xr.DataArray
        | tuple[xr.DataArray, xr.DataArray],
        mom: Moments,
        *,
        full_output: Literal[True],
        nrep: int | None = ...,
        freq: MyNDArray | None = ...,
        indices: MyNDArray | None = ...,
        w: MyNDArray | xr.DataArray | None = ...,
        dim: Hashable | None = ...,
        rep_dim: str = ...,
        dtype: DTypeLike | None = ...,
        broadcast: bool = ...,
        parallel: bool = ...,
        resample_kws: Mapping[str, Any] | None = ...,
        rng: np.random.Generator | None = ...,
        axis: int | None = ...,
        # xarray specific
        dims: XArrayDimsType = ...,
        attrs: XArrayAttrsType = ...,
        coords: XArrayCoordsType = ...,
        indexes: XArrayIndexesType = ...,
        name: XArrayNameType = ...,
        mom_dims: MomDims | None = ...,
        **kwargs: Any,
    ) -> tuple[Self, MyNDArray]:
        ...

    @overload
    @classmethod
    def from_resample_vals(
        cls,
        x: MyNDArray
        | tuple[MyNDArray, MyNDArray]
        | xr.DataArray
        | tuple[xr.DataArray, xr.DataArray],
        mom: Moments,
        *,
        full_output: bool,
        nrep: int | None = ...,
        freq: MyNDArray | None = ...,
        indices: MyNDArray | None = ...,
        w: MyNDArray | xr.DataArray | None = ...,
        dim: Hashable | None = ...,
        rep_dim: str = ...,
        dtype: DTypeLike | None = ...,
        broadcast: bool = ...,
        parallel: bool = ...,
        resample_kws: Mapping[str, Any] | None = ...,
        rng: np.random.Generator | None = ...,
        axis: int | None = ...,
        # xarray specific
        dims: XArrayDimsType = ...,
        attrs: XArrayAttrsType = ...,
        coords: XArrayCoordsType = ...,
        indexes: XArrayIndexesType = ...,
        name: XArrayNameType = ...,
        mom_dims: MomDims | None = ...,
        **kwargs: Any,
    ) -> Self | tuple[Self, MyNDArray]:
        ...

    @classmethod
    @docfiller_inherit_abc()
    def from_resample_vals(  # noqa: C901,PLR0913,PLR0912
        cls,
        x: MyNDArray
        | tuple[MyNDArray, MyNDArray]
        | xr.DataArray
        | tuple[xr.DataArray, xr.DataArray],
        mom: Moments,
        *,
        full_output: bool = False,
        nrep: int | None = None,
        freq: MyNDArray | None = None,
        indices: MyNDArray | None = None,
        w: MyNDArray | xr.DataArray | None = None,
        dim: Hashable | None = None,
        rep_dim: str = "rep",
        dtype: DTypeLike | None = None,
        broadcast: bool = False,
        parallel: bool = True,
        resample_kws: Mapping[str, Any] | None = None,
        rng: np.random.Generator | None = None,
        axis: int | None = None,
        # xarray specific
        dims: XArrayDimsType = None,
        attrs: XArrayAttrsType = None,
        coords: XArrayCoordsType = None,
        indexes: XArrayIndexesType = None,  # noqa: ARG003
        name: XArrayNameType = None,
        mom_dims: MomDims | None = None,
        **kwargs: Any,
    ) -> Self | tuple[Self, MyNDArray]:
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
        from .central import CentralMoments

        x0 = x[0] if isinstance(x, tuple) else x

        if isinstance(x0, xr.DataArray):
            axis, dim = _select_axis_dim(dims=x0.dims, axis=axis, dim=dim)
            # TODO(wpk): create object, and verify y, and w against x
            # override final xarray stuff:
            template = x0.isel({dim: 0})
            dims = template.dims
            if coords is None:
                coords = {}
            coords = {**template.coords, **coords}  # type: ignore[dict-item]  # pyright: ignore[reportUnknownMemberType,reportGeneralTypeIssues]
            if attrs is None:
                attrs = {}
            attrs = dict(template.attrs, **attrs)
            if name is None:
                name = template.name

        w_values: MyNDArray | None
        if isinstance(w, xr.DataArray):
            if isinstance(x0, xr.DataArray):
                w_values = w.broadcast_like(x0).to_numpy()  # pyright: ignore[reportUnknownMemberType]

            else:  # pragma: no cover
                w_values = w.to_numpy()  # pyright: ignore[reportUnknownMemberType]
        else:
            w_values = w

        if dims is not None:
            if isinstance(dims, str):  # pragma: no cover
                dims = (dims,)
            dims = (rep_dim, *dims)  # type: ignore[misc]

        x_array: MyNDArray | tuple[MyNDArray, MyNDArray]
        if isinstance(x, tuple):
            x_array = tuple(np.array(xx, copy=False) for xx in x)  # type: ignore[assignment]
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
            rng=rng,
            **kwargs,
        )

        new = cls.from_centralmoments(
            obj=out,
            dims=dims,  # pyright: ignore[reportUnknownArgumentType]
            coords=coords,
            attrs=attrs,
            name=name,
            mom_dims=mom_dims,
            copy=False,
        )

        if full_output:
            return (new, freq)
        return new

    @classmethod
    @docfiller_inherit_abc()
    def from_raw(
        cls,
        raw: MyNDArray | xr.DataArray,
        *,
        mom_ndim: Mom_NDim | None = None,
        mom: Moments | None = None,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        convert_kws: Mapping[str, Any] | None = None,
        dims: XArrayDimsType = None,
        attrs: XArrayAttrsType = None,
        coords: XArrayCoordsType = None,
        indexes: XArrayIndexesType = None,
        name: XArrayNameType = None,
        mom_dims: MomDims | None = None,
        template: xr.DataArray | None = None,
        **kwargs: Any,
    ) -> Self:
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
            mom_ndim = select_mom_ndim(mom=mom, mom_ndim=mom_ndim)

            raw = _move_mom_dims_to_end(raw, mom_dims, mom_ndim)

            if convert_kws is None:
                convert_kws = {}

            values = cast("MyNDArray", raw.to_numpy())  # pyright: ignore[reportUnknownMemberType]

            if mom_ndim == 1:
                data_values = convert.to_central_moments(
                    values, dtype=dtype, **convert_kws
                )
            elif mom_ndim == 2:
                data_values = convert.to_central_comoments(
                    values, dtype=dtype, **convert_kws
                )
            else:  # pragma: no cover
                msg = f"unknown mom_ndim {mom_ndim}"
                raise ValueError(msg)

            kwargs = {"copy": False, **kwargs}
            new = cls.from_data(
                data=raw.copy(data=data_values),
                mom=mom,
                mom_ndim=mom_ndim,
                val_shape=val_shape,
                dtype=dtype,
                **kwargs,
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
                name=name,
                attrs=attrs,
                coords=coords,
                indexes=indexes,
                mom_dims=mom_dims,
                template=template,
                **kwargs,
            )

        return new

    @classmethod
    @docfiller_inherit_abc()
    def from_raws(  # noqa: PLR0913
        cls,
        raws: MyNDArray | xr.DataArray,
        *,
        mom_ndim: Mom_NDim | None = None,
        mom: Moments | None = None,
        axis: int | None = None,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        convert_kws: Mapping[str, Any] | None = None,
        dim: Hashable | None = None,
        dims: XArrayDimsType = None,
        attrs: XArrayAttrsType = None,
        coords: XArrayCoordsType = None,
        indexes: XArrayIndexesType = None,
        name: XArrayNameType = None,
        mom_dims: MomDims = None,
        template: xr.DataArray | None = None,
        **kwargs: Any,
    ) -> Self:
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
            **kwargs,
        )

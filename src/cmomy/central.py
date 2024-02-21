"""
Central moments/comoments routines from :class:`np.ndarray` objects
-------------------------------------------------------------------.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

import numpy as np
import xarray as xr

from . import convert
from .abstract_central import CentralMomentsABC
from .docstrings import docfiller_central as docfiller
from .utils import (
    axis_expand_broadcast,
    mom_to_mom_ndim,
    select_mom_ndim,
    shape_reduce,
    validate_mom_and_mom_ndim,
)

if TYPE_CHECKING:
    from typing import Any, Hashable, Literal, Mapping

    from numpy.typing import ArrayLike, DTypeLike

    from ._typing_compat import Self
    from .typing import (
        ArrayOrder,
        Mom_NDim,
        MomDims,
        Moments,
        MultiArray,
        MultiArrayVals,
        VerifyValuesStyles,
        XArrayAttrsType,
        XArrayCoordsType,
        XArrayDimsType,
        XArrayIndexesType,
        XArrayNameType,
    )
    from .xcentral import xCentralMoments


from .typing import MyNDArray


###############################################################################
# central mom/comoments routines
###############################################################################
def _central_moments(
    vals: ArrayLike,
    mom: Moments,
    w: MyNDArray | None = None,
    axis: int = 0,
    last: bool = True,
    dtype: DTypeLike | None = None,
    order: ArrayOrder | None = None,
    out: MyNDArray | None = None,
) -> MyNDArray:
    """Calculate central mom along axis."""
    if isinstance(mom, tuple):  # pragma: no cover
        mom = mom[0]

    x = np.asarray(vals, dtype=dtype, order=order)
    if dtype is None:
        dtype = x.dtype

    if w is None:
        w = np.ones_like(x)
    else:
        w = axis_expand_broadcast(
            w, shape=x.shape, axis=axis, roll=False, dtype=dtype, order=order
        )

    # if axis < 0:
    #     axis += x.ndim
    if axis != 0:
        x = np.moveaxis(x, axis, 0)
        w = np.moveaxis(w, axis, 0)

    shape = (mom + 1,) + x.shape[1:]
    if out is None:
        out = np.empty(shape, dtype=dtype)
    elif out.shape != shape:
        # try rolling
        out = np.moveaxis(out, -1, 0)
        if out.shape != shape:
            raise ValueError

    wsum = w.sum(axis=0)  # pyright: ignore[reportUnknownMemberType]
    wsum_inv = 1.0 / wsum
    xave = np.einsum("r...,r...->...", w, x) * wsum_inv  # pyright: ignore[reportUnknownMemberType]

    shape = (-1,) + (1,) * (x.ndim)
    p = np.arange(2, mom + 1).reshape(*shape)  # pyright: ignore[reportUnknownMemberType]

    dx = (x[None, ...] - xave) ** p

    out[0, ...] = wsum
    out[1, ...] = xave
    out[2:, ...] = np.einsum("r..., mr...->m...", w, dx) * wsum_inv  # pyright: ignore[reportUnknownMemberType]

    if last:
        out = np.moveaxis(out, 0, -1)
    return out


def _central_comoments(  # noqa: C901, PLR0912
    vals: tuple[MyNDArray, MyNDArray],
    mom: tuple[int, int],
    w: MyNDArray | None = None,
    axis: int = 0,
    last: bool = True,
    broadcast: bool = False,
    dtype: DTypeLike | None = None,
    order: ArrayOrder | None = None,
    out: MyNDArray | None = None,
) -> MyNDArray:
    """Calculate central co-mom (covariance, etc) along axis."""
    if not isinstance(
        mom, tuple
    ):  # pragma: no cover  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError

    if len(mom) != 2:
        raise ValueError

    # change x to tuple of inputs
    if not isinstance(vals, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError
    if len(vals) != 2:
        raise ValueError
    x, y = vals

    x = np.asarray(x, dtype=dtype, order=order)
    if dtype is None:
        dtype = x.dtype

    y = axis_expand_broadcast(
        y,
        shape=x.shape,
        axis=axis,
        roll=False,
        broadcast=broadcast,
        expand=broadcast,
        dtype=dtype,
        order=order,
    )

    if w is None:
        w = np.ones_like(x)
    else:
        w = axis_expand_broadcast(
            w, shape=x.shape, axis=axis, roll=False, dtype=dtype, order=order
        )

    if w.shape != x.shape or y.shape != x.shape:
        raise ValueError

    if axis != 0:
        x = np.moveaxis(x, axis, 0)
        y = np.moveaxis(y, axis, 0)
        w = np.moveaxis(w, axis, 0)

    shape = tuple(x + 1 for x in mom) + x.shape[1:]
    if out is None:
        out = np.empty(shape, dtype=dtype)
    elif out.shape != shape:
        # try moving axis
        out = np.moveaxis(out, [-2, -1], [0, 1])
        if out.shape != shape:
            raise ValueError

    wsum = w.sum(axis=0)  # pyright: ignore[reportUnknownMemberType]
    wsum_inv = 1.0 / wsum

    xave = np.einsum("r...,r...->...", w, x) * wsum_inv  # pyright: ignore[reportUnknownMemberType]
    yave = np.einsum("r...,r...->...", w, y) * wsum_inv  # pyright: ignore[reportUnknownMemberType]

    shape = (-1,) + (1,) * (x.ndim)
    p0 = np.arange(0, mom[0] + 1).reshape(*shape)  # pyright: ignore[reportUnknownMemberType]
    p1 = np.arange(0, mom[1] + 1).reshape(*shape)  # pyright: ignore[reportUnknownMemberType]

    dx = (x[None, ...] - xave) ** p0
    dy = (y[None, ...] - yave) ** p1

    out[...] = (
        np.einsum("r...,ir...,jr...->ij...", w, dx, dy) * wsum_inv  # pyright: ignore[reportUnknownMemberType]
    )

    out[0, 0, ...] = wsum
    out[1, 0, ...] = xave
    out[0, 1, ...] = yave

    if last:
        out = np.moveaxis(out, [0, 1], [-2, -1])
    return out


@docfiller.decorate
def central_moments(
    x: MyNDArray | tuple[MyNDArray, MyNDArray],
    mom: Moments,
    *,
    w: MyNDArray | None = None,
    axis: int = 0,
    last: bool = True,
    dtype: DTypeLike | None = None,
    order: ArrayOrder | None = None,
    out: MyNDArray | None = None,
    broadcast: bool = False,
) -> MyNDArray:
    """
    Calculate central moments or comoments along axis.

    Parameters
    ----------
    x : array-like or tuple of array-like
        if calculating moments, then this is the input array.
        if calculating comoments, then pass in tuple of values of form (x, y)
    {mom}
    w : array-like, optional
        Weights. If passed, should be able to broadcast to `x`. An exception is if
        w is a 1d array with len(w) == x.shape[axis]. In this case,
        w will be reshaped and broadcast against x
    {axis}
    last : bool, default=True
        if True, put mom as last dimension.
        Otherwise, mom will be in first dimension
    {dtype}
    {broadcast}
    out : array
        if present, use this for output data
        Needs to have shape of either (mom,) + shape or shape + (mom,)
        where shape is the shape of x with axis removed

    Returns
    -------
    output : array
        array of shape=shape + mom_shape or mom_shape + shape depending on
        value of `last`, where `shape` is the shape of `x` with axis removed,
        i.e., shape=x.shape[:axis] + x.shape[axis+1:], and `mom_shape` is the shape of
        the moment part, either (mom+1,) or (mom0+1, mom1+1).  Assuming `last is True`,
        output[...,0] is the total weight (or count), output[...,1] is the mean
        value of x, output[...,n] with n>1 is the nth central moment.

    See Also
    --------
    CentralMoments


    Examples
    --------
    create data:

    >>> from cmomy.random import default_rng
    >>> rng = default_rng(0)
    >>> x = rng.random(10)

    Generate first 2 central moments:

    >>> moments = central_moments(x=x, mom=2)
    >>> print(moments)
    [10.      0.5505  0.1014]

    Generate moments with weights

    >>> w = rng.random(10)
    >>> central_moments(x=x, w=w, mom=2)
    array([4.7419, 0.5877, 0.0818])


    Generate co-moments

    >>> y = rng.random(10)
    >>> central_moments(x=(x, y), w=w, mom=(2, 2))
    array([[ 4.7419e+00,  6.3452e-01,  1.0383e-01],
           [ 5.8766e-01, -5.1403e-03,  6.1079e-03],
           [ 8.1817e-02,  1.5621e-03,  7.7609e-04]])

    """
    if isinstance(mom, int):
        mom = (mom,)

    if len(mom) == 1:
        return _central_moments(
            vals=x,
            mom=mom,
            w=w,
            axis=axis,
            last=last,
            dtype=dtype,
            order=order,
            out=out,
        )
    return _central_comoments(
        vals=x,  # type: ignore[arg-type]
        mom=mom,
        w=w,
        axis=axis,
        last=last,
        dtype=dtype,
        order=order,
        broadcast=broadcast,
        out=out,
    )


###############################################################################
# Classes
###############################################################################

docfiller_abc = docfiller.factory_from_parent(CentralMomentsABC)
docfiller_inherit_abc = docfiller.factory_inherit_from_parent(CentralMomentsABC)


@docfiller(CentralMomentsABC)  # noqa: PLR0904
class CentralMoments(CentralMomentsABC[MyNDArray]):  # noqa: D101
    # def __new__(cls, data: MyNDArray, mom_ndim: Literal[1, 2] = 1):
    #     return super().__new__(cls, data=data, mom_ndim=mom_ndim)

    def __init__(self, data: MyNDArray, mom_ndim: Mom_NDim = 1) -> None:
        if mom_ndim not in {1, 2}:
            msg = (
                "mom_ndim must be either 1 (for central moments)"
                "or 2 (for central comoments)"
            )
            raise ValueError(msg)

        if not isinstance(data, np.ndarray):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"data must be an np.ndarray.  Passed type {type(data)}"
            raise TypeError(msg)

        if data.ndim < mom_ndim:
            msg = "not enough dimensions in data"
            raise ValueError(msg)

        self._mom_ndim = mom_ndim
        self._data = data
        self._data_flat = self._data.reshape(self.shape_flat)
        self._data = self._data_flat.reshape(self.shape)  # ensure same data

        if any(m <= 0 for m in self.mom):
            msg = "moments must be positive"
            raise ValueError(msg)

        self._validate_data()  # pragma: no cover

        self._cache: dict[str, Any] = {}

    def to_values(self) -> MyNDArray:
        """Accesses for self.data."""
        return self._data

    ###########################################################################
    # SECTION: top level creation/copy/new
    ###########################################################################
    # @docfiller(CentralMomentsABC.new_like)
    @docfiller_abc()
    def new_like(
        self,
        data: MyNDArray | None = None,
        *,
        copy: bool = False,
        copy_kws: Mapping[str, Any] | None = None,
        verify: bool = False,
        strict: bool = False,
        **kwargs: Any,
    ) -> Self:
        """
        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> da = CentralMoments.from_vals(rng.random(10), mom=3, axis=0)
        >>> da
        <CentralMoments(val_shape=(), mom=(3,))>
        array([10.    ,  0.5505,  0.1014, -0.0178])

        >>> da2 = da.new_like().zero()
        >>> da2
        <CentralMoments(val_shape=(), mom=(3,))>
        array([0., 0., 0., 0.])

        >>> da
        <CentralMoments(val_shape=(), mom=(3,))>
        array([10.    ,  0.5505,  0.1014, -0.0178])

        """
        if data is None:
            data = np.zeros_like(self._data, order="C")
            copy = verify = False

        kwargs.setdefault("mom_ndim", self.mom_ndim)

        if strict:
            kwargs = dict(
                {
                    "mom": self.mom,
                    "val_shape": self.val_shape,
                    "dtype": self.dtype,
                },
                **kwargs,
            )

        return type(self).from_data(
            data=data,
            copy=copy,
            copy_kws=copy_kws,
            verify=verify,
            **kwargs,
        )

    ###########################################################################
    # SECTION: To/from xarray
    ###########################################################################
    @docfiller.decorate
    def to_xarray(  # noqa: PLR0912
        self,
        *,
        dims: XArrayDimsType = None,
        attrs: XArrayAttrsType = None,
        coords: XArrayCoordsType = None,
        name: XArrayNameType = None,
        indexes: XArrayIndexesType = None,  # noqa: ARG002
        mom_dims: MomDims | None = None,
        template: xr.DataArray | None = None,
        copy: bool = False,
    ) -> xr.DataArray:
        """
        Create a :class:`xarray.DataArray` representation of underlying data.

        Parameters
        ----------
        {xr_params}
        {copy}

        Returns
        -------
        output : DataArray


        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> da = CentralMoments.from_vals(rng.random((10, 1, 2)), axis=0, mom=2)
        >>> da
        <CentralMoments(val_shape=(1, 2), mom=(2,))>
        array([[[10.    ,  0.6207,  0.0647],
                [10.    ,  0.404 ,  0.1185]]])

        Default constructor

        >>> da.to_xarray()
        <xarray.DataArray (dim_0: 1, dim_1: 2, mom_0: 3)> Size: 48B
        array([[[10.    ,  0.6207,  0.0647],
                [10.    ,  0.404 ,  0.1185]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0

        Setting attributes

        >>> da.to_xarray()
        <xarray.DataArray (dim_0: 1, dim_1: 2, mom_0: 3)> Size: 48B
        array([[[10.    ,  0.6207,  0.0647],
                [10.    ,  0.404 ,  0.1185]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0
        >>> da
        <CentralMoments(val_shape=(1, 2), mom=(2,))>
        array([[[10.    ,  0.6207,  0.0647],
                [10.    ,  0.404 ,  0.1185]]])

        """
        if template is not None:
            out = template.copy(data=self.data)
        else:
            if dims is None:
                dims = tuple(f"dim_{i}" for i in range(self.val_ndim))
            elif isinstance(dims, str):
                dims = (dims,)
            else:
                # try to convert to tuple
                dims = tuple(dims)  # type: ignore[arg-type]

            if len(dims) == self.ndim:
                dims_output = dims

            elif len(dims) == self.val_ndim:
                if mom_dims is None:
                    mom_dims = tuple(f"mom_{i}" for i in range(self.mom_ndim))
                elif isinstance(mom_dims, str):
                    mom_dims = (mom_dims,)
                else:
                    # try to convert to tuple
                    mom_dims = tuple(mom_dims)  # type: ignore[arg-type]

                if len(mom_dims) != self.mom_ndim:
                    msg = f"{mom_dims=} != {self.mom_ndim=}"
                    raise ValueError(msg)

                dims_output = dims + mom_dims

            else:
                msg = f"Problem with {dims}, {mom_dims}.  Total length should be {self.ndim}"
                raise ValueError(msg)
            out = xr.DataArray(
                self.data, dims=dims_output, coords=coords, attrs=attrs, name=name
            )

        if copy:
            out = out.copy()

        return out

    @docfiller.decorate
    def to_xcentralmoments(
        self,
        *,
        dims: XArrayDimsType = None,
        attrs: XArrayAttrsType = None,
        coords: XArrayCoordsType = None,
        name: XArrayNameType = None,
        indexes: XArrayIndexesType = None,
        mom_dims: MomDims | None = None,
        template: xr.DataArray | None = None,
        copy: bool = False,
    ) -> xCentralMoments:
        """
        Create an :class:`xarray.DataArray` representation of underlying data.

        Parameters
        ----------
        {xr_params}
        {copy}

        Returns
        --------
        output : xCentralMoments

        See Also
        --------
        to_xarray

        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> da = CentralMoments.from_vals(rng.random((10, 1, 2)), axis=0, mom=2)
        >>> da
        <CentralMoments(val_shape=(1, 2), mom=(2,))>
        array([[[10.    ,  0.6207,  0.0647],
                [10.    ,  0.404 ,  0.1185]]])

        Default constructor

        >>> da.to_xcentralmoments()
        <xCentralMoments(val_shape=(1, 2), mom=(2,))>
        <xarray.DataArray (dim_0: 1, dim_1: 2, mom_0: 3)> Size: 48B
        array([[[10.    ,  0.6207,  0.0647],
                [10.    ,  0.404 ,  0.1185]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0

        Setting attributes

        >>> da.to_xcentralmoments()
        <xCentralMoments(val_shape=(1, 2), mom=(2,))>
        <xarray.DataArray (dim_0: 1, dim_1: 2, mom_0: 3)> Size: 48B
        array([[[10.    ,  0.6207,  0.0647],
                [10.    ,  0.404 ,  0.1185]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0
        >>> da
        <CentralMoments(val_shape=(1, 2), mom=(2,))>
        array([[[10.    ,  0.6207,  0.0647],
                [10.    ,  0.404 ,  0.1185]]])

        """  # noqa: D409
        from .xcentral import xCentralMoments

        return xCentralMoments.from_centralmoments(
            obj=self,
            dims=dims,
            attrs=attrs,
            coords=coords,
            name=name,
            indexes=indexes,
            mom_dims=mom_dims,
            template=template,
            copy=copy,
        )

        # data = self.to_xarray(
        #     dims=dims,
        #     attrs=attrs,
        #     coords=coords,
        #     name=name,
        #     indexes=indexes,
        #     mom_dims=mom_dims,
        #     template=template,
        #     copy=copy,
        # )
        # return xCentralMoments(data=data, mom_ndim=self.mom_ndim)

    ###########################################################################
    # SECTION: pushing routines
    ###########################################################################
    #  -> np.ndarray | float | Tuple[float|np.ndarray, None|float|np.ndarray] :

    def _get_target_shape(
        self,
        x: MyNDArray,
        style: VerifyValuesStyles,
        axis: int | None = None,
        other: MyNDArray | None = None,
    ) -> tuple[int, ...]:
        from .utils import shape_insert_axis

        if style == "val":
            target_shape = self.val_shape
        elif style == "vals":
            if axis is None:
                raise ValueError
            target_shape = shape_insert_axis(
                shape=self.val_shape, axis=axis, new_size=x.shape[axis]
            )
        elif style == "data":
            target_shape = self.shape
        elif style == "datas":
            # make sure axis in limits
            if axis is None:
                raise ValueError
            target_shape = shape_insert_axis(
                shape=self.shape, axis=axis, new_size=x.shape[axis]
            )
        elif style == "var":
            target_shape = self.shape_var
        elif style == "vars":
            if other is None or axis is None:
                raise ValueError
            target_shape = shape_insert_axis(
                shape=self.shape_var, axis=axis, new_size=other.shape[axis]
            )
        else:
            msg = f"unknown string style name {style}"
            raise ValueError(msg)

        return target_shape

    def _verify_value(
        self,
        *,
        x: MultiArray[MyNDArray],
        target: str | MyNDArray,
        shape_flat: tuple[int, ...],
        axis: int | None = None,
        dim: Hashable | None = None,  # included here for consistency  # noqa: ARG002
        broadcast: bool = False,
        expand: bool = False,
        other: MyNDArray | None = None,
    ) -> tuple[MyNDArray, MyNDArray]:
        """
        Verify input values.

        Parameters
        ----------
        x : array
        target : tuple or array
            If tuple, this is the target shape to be used to Make target.
            If array, this is the target array
        Optional target that has already been rolled.  If this is passed, and
        x will be broadcast/expanded, can expand to this shape without the need
        to reorder,
        """
        x = np.asarray(x, dtype=self.dtype)

        if isinstance(target, str):
            target_shape = self._get_target_shape(
                x=x, style=cast("VerifyValuesStyles", target), axis=axis, other=other
            )
            target_output = x

        elif isinstance(
            target, np.ndarray
        ):  # pragma: no cover  # pyright: ignore[reportUnnecessaryIsInstance]
            target_shape = target.shape
            target_output = target

        else:
            msg = "unknown target type"
            raise TypeError(msg)

        x = axis_expand_broadcast(
            x,
            shape=target_shape,
            axis=axis,
            verify=False,
            expand=expand,
            broadcast=broadcast,
            dtype=self.dtype,
            roll=False,
        )

        # check shape:
        if x.shape != target_shape:
            msg = f"{x.shape=} not equal {target_shape=}"
            raise ValueError(msg)

        # move axis
        nrec: tuple[int, ...]
        if axis is not None:
            if axis != 0:
                x = np.moveaxis(x, axis, 0)
            nrec = (x.shape[0],)
        else:
            nrec = ()

        x = x.reshape(nrec + shape_flat)  # pyright: ignore[reportUnknownArgumentType]

        if x.ndim == 0:
            x = x[()]

        return x, target_output

    @docfiller_inherit_abc()
    def push_data(self, data: MultiArrayVals[MyNDArray]) -> Self:
        """
        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> xs = rng.random((2, 10))
        >>> datas = [central_moments(x=x, mom=2) for x in xs]
        >>> da = CentralMoments.from_data(datas[0], mom_ndim=1)
        >>> da
        <CentralMoments(val_shape=(), mom=(2,))>
        array([10.    ,  0.5505,  0.1014])


        >>> da.push_data(datas[1])
        <CentralMoments(val_shape=(), mom=(2,))>
        array([20.    ,  0.5124,  0.1033])


        Which is equivalent to

        >>> CentralMoments.from_vals(xs.reshape(-1), mom=2)
        <CentralMoments(val_shape=(), mom=(2,))>
        array([20.    ,  0.5124,  0.1033])

        """
        return super().push_data(data=data)

    @docfiller_inherit_abc()
    def push_datas(
        self,
        datas: MultiArray[MyNDArray],
        axis: int | None = 0,
        **kwargs: Any,  # noqa: ARG002
    ) -> Self:
        """
        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> xs = rng.random((2, 10))
        >>> datas = np.array([central_moments(x=x, mom=2) for x in xs])
        >>> da = CentralMoments.zeros(mom=2)
        >>> da.push_datas(datas, axis=0)
        <CentralMoments(val_shape=(), mom=(2,))>
        array([20.    ,  0.5124,  0.1033])


        Which is equivalent to

        >>> CentralMoments.from_vals(xs.reshape(-1), mom=2)
        <CentralMoments(val_shape=(), mom=(2,))>
        array([20.    ,  0.5124,  0.1033])
        """
        return super().push_datas(datas=datas, axis=axis or 0)

    @docfiller_inherit_abc()
    def push_val(
        self,
        x: MultiArray[MyNDArray] | tuple[MultiArray[MyNDArray], MultiArray[MyNDArray]],
        w: MultiArray[MyNDArray] | None = None,
        broadcast: bool = False,
    ) -> Self:
        """
        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> x = rng.random((10, 2))
        >>> y = rng.random(10)
        >>> w = rng.random(10)

        >>> da = CentralMoments.zeros(val_shape=(2,), mom=(2, 2))
        >>> for xx, yy, ww in zip(x, y, w):
        ...     _ = da.push_val(x=(xx, yy), w=ww, broadcast=True)

        >>> da
        <CentralMoments(val_shape=(2,), mom=(2, 2))>
        array([[[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 6.4741e-01,  3.3791e-02, -5.1117e-03],
                [ 5.0888e-02, -1.0060e-02,  7.0290e-03]],
        <BLANKLINE>
               [[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 3.9793e-01,  6.3224e-03, -2.2669e-02],
                [ 9.3979e-02,  9.9433e-04,  6.5765e-03]]])


        Which is the same as

        >>> CentralMoments.from_vals(x=(x, y), w=w, mom=(2, 2), broadcast=True)
        <CentralMoments(val_shape=(2,), mom=(2, 2))>
        array([[[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 6.4741e-01,  3.3791e-02, -5.1117e-03],
                [ 5.0888e-02, -1.0060e-02,  7.0290e-03]],
        <BLANKLINE>
               [[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 3.9793e-01,  6.3224e-03, -2.2669e-02],
                [ 9.3979e-02,  9.9433e-04,  6.5765e-03]]])

        """
        return super().push_val(x=x, w=w, broadcast=broadcast)

    @docfiller_inherit_abc()
    def push_vals(
        self,
        x: MultiArrayVals[MyNDArray]
        | tuple[MultiArrayVals[MyNDArray], MultiArrayVals[MyNDArray]],
        w: MultiArray[MyNDArray] | None = None,
        axis: int | None = 0,
        broadcast: bool = False,
        **kwargs: Any,  # noqa: ARG002
    ) -> Self:
        """
        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> x = rng.random((10, 2))
        >>> y = rng.random(10)
        >>> w = rng.random(10)

        >>> da = CentralMoments.zeros(val_shape=(2,), mom=(2, 2))
        >>> da.push_vals(x=(x, y), w=w, broadcast=True)
        <CentralMoments(val_shape=(2,), mom=(2, 2))>
        array([[[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 6.4741e-01,  3.3791e-02, -5.1117e-03],
                [ 5.0888e-02, -1.0060e-02,  7.0290e-03]],
        <BLANKLINE>
               [[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 3.9793e-01,  6.3224e-03, -2.2669e-02],
                [ 9.3979e-02,  9.9433e-04,  6.5765e-03]]])


        Which is the same as

        >>> CentralMoments.from_vals(x=(x, y), w=w, mom=(2, 2), broadcast=True)
        <CentralMoments(val_shape=(2,), mom=(2, 2))>
        array([[[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 6.4741e-01,  3.3791e-02, -5.1117e-03],
                [ 5.0888e-02, -1.0060e-02,  7.0290e-03]],
        <BLANKLINE>
               [[ 5.4367e+00,  6.0656e-01,  9.9896e-02],
                [ 3.9793e-01,  6.3224e-03, -2.2669e-02],
                [ 9.3979e-02,  9.9433e-04,  6.5765e-03]]])

        """
        return super().push_vals(
            x=x,
            w=w,
            axis=axis or 0,
            broadcast=broadcast,
        )

    ###########################################################################
    # SECTION: Manipulation
    ###########################################################################
    @overload
    def resample_and_reduce(
        self,
        nrep: int | None = ...,
        *,
        full_output: Literal[False] = ...,
        freq: MyNDArray | None = ...,
        indices: MyNDArray | None = ...,
        axis: int | None = ...,
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
        freq: MyNDArray | None = ...,
        indices: MyNDArray | None = ...,
        axis: int | None = ...,
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
        freq: MyNDArray | None = ...,
        indices: MyNDArray | None = ...,
        axis: int | None = ...,
        parallel: bool = ...,
        resample_kws: Mapping[str, Any] | None = ...,
        rng: np.random.Generator | None = ...,
        **kwargs: Any,
    ) -> Self | tuple[Self, MyNDArray]:
        ...

    @docfiller.decorate
    def resample_and_reduce(
        self,
        nrep: int | None = None,
        *,
        full_output: bool = False,
        freq: MyNDArray | None = None,
        indices: MyNDArray | None = None,
        axis: int | None = None,
        parallel: bool = True,
        resample_kws: Mapping[str, Any] | None = None,
        rng: np.random.Generator | None = None,
        **kwargs: Any,
    ) -> Self | tuple[Self, MyNDArray]:
        """
        Bootstrap resample and reduce.

        Parameters
        ----------
        {nrep}
        {full_output}
        {freq}
        {indices}
        {axis}
        {parallel}
        {resample_kws}
        {rng}
        **kwargs
            Extra key-word arguments to :meth:`from_data`

        Returns
        -------
        output : object
            Instance of calling class
            Note that new object will have val_shape = (nrep,) +
            val_shape[:axis] + val_shape[axis+1:]

        See Also
        --------
        resample
        reduce
        ~cmomy.resample.randsamp_freq : random frequency sample
        ~cmomy.resample.freq_to_indices : convert frequency sample to index sample
        ~cmomy.resample.indices_to_freq : convert index sample to frequency sample
        ~cmomy.resample.resample_data : method to perform resampling
        """
        from .resample import randsamp_freq, resample_data

        self._raise_if_scalar()
        axis = self._wrap_axis(axis)
        if resample_kws is None:
            resample_kws = {}

        freq = randsamp_freq(
            nrep=nrep,
            ndat=self.val_shape[axis],
            indices=indices,
            freq=freq,
            check=True,
            rng=rng,
        )
        data = resample_data(
            self.data, freq, mom=self.mom, axis=axis, parallel=parallel, **resample_kws
        )
        out = type(self).from_data(data, mom_ndim=self.mom_ndim, copy=False, **kwargs)

        if full_output:
            return out, freq
        return out

    @docfiller.decorate
    def reduce(self, axis: int | None = None, **kwargs: Any) -> Self:
        """
        Create new object reduce along axis.

        Parameters
        ----------
        {axis}
        **kwargs
            Extra parameters to :meth:`from_datas`

        Returns
        -------
        output : {klass}

        See Also
        --------
        from_datas
        """
        self._raise_if_scalar()
        axis = self._wrap_axis(axis)
        return type(self).from_datas(
            self.to_values(), mom_ndim=self.mom_ndim, axis=axis, **kwargs
        )

    @docfiller.decorate
    def block(
        self,
        block_size: int | None = None,
        axis: int | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Block average reduction.

        Parameters
        ----------
        block_size : int
            number of consecutive records to combine
        {axis}
        **kwargs
            Extra key word arguments to :meth:`from_datas` method

        Returns
        -------
        output : object
            New instance of calling class
            Shape of output will be
            `(nblock,) + self.shape[:axis] + self.shape[axis+1:]`.

        Notes
        -----
        The block averaged `axis` will be moved to the front of the output data.

        See Also
        --------
        reshape
        `moveaxis
        :meth:`reduce`
        """
        self._raise_if_scalar()

        axis = self._wrap_axis(axis)
        data = self.data

        # move axis to first
        if axis != 0:
            data = np.moveaxis(data, axis, 0)

        n = data.shape[0]

        if block_size is None:
            block_size = n
            nblock = 1

        else:
            nblock = n // block_size

        datas = data[: (nblock * block_size), ...].reshape(
            (nblock, block_size) + data.shape[1:]
        )
        return type(self).from_datas(
            datas=datas, mom_ndim=self.mom_ndim, axis=1, **kwargs
        )

    @docfiller.decorate
    def reshape(
        self,
        shape: tuple[int, ...],
        *,
        copy: bool = True,
        copy_kws: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Create a new object with reshaped data.

        Parameters
        ----------
        shape : tuple
            shape of values part of data.
        {copy}
        {copy_kws}
        **kwargs
            Parameters to :meth:`from_data`

        Returns
        -------
        output : CentralMoments
            output object with reshaped data

        See Also
        --------
        numpy.reshape
        from_data

        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> da = CentralMoments.from_vals(rng.random((10, 2, 3)), mom=2)
        >>> da
        <CentralMoments(val_shape=(2, 3), mom=(2,))>
        array([[[10.    ,  0.5205,  0.0452],
                [10.    ,  0.4438,  0.0734],
                [10.    ,  0.5038,  0.1153]],
        <BLANKLINE>
               [[10.    ,  0.5238,  0.1272],
                [10.    ,  0.628 ,  0.0524],
                [10.    ,  0.412 ,  0.0865]]])

               [[10.        ,  0.53720667,  0.05909394],
                [10.        ,  0.42622908,  0.08434857],
                [10.        ,  0.47326641,  0.05907737]]])

        >>> da.reshape(shape=(-1,))
        <CentralMoments(val_shape=(6,), mom=(2,))>
        array([[10.    ,  0.5205,  0.0452],
               [10.    ,  0.4438,  0.0734],
               [10.    ,  0.5038,  0.1153],
               [10.    ,  0.5238,  0.1272],
               [10.    ,  0.628 ,  0.0524],
               [10.    ,  0.412 ,  0.0865]])
        """
        self._raise_if_scalar()
        new_shape = shape + self.mom_shape
        data = self._data.reshape(new_shape)

        return type(self).from_data(
            data=data,
            mom_ndim=self.mom_ndim,
            mom=self.mom,
            val_shape=None,
            copy=copy,
            copy_kws=copy_kws,
            dtype=self.dtype,
            **kwargs,
        )

    @docfiller.decorate
    def moveaxis(
        self,
        source: int | tuple[int, ...],
        destination: int | tuple[int, ...],
        *,
        copy: bool = True,
        copy_kws: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Move axis from source to destination.

        Parameters
        ----------
        source : int or sequence of int
            Original positions of the axes to move. These must be unique.
        destination : int or sequence of int
            Destination positions for each of the original axes. These must also be
            unique.
        {copy}
        {copy_kws}

        Returns
        -------
        result : CentralMoments
            CentralMoments object with with moved axes. This array is a view of the input array.


        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> da = CentralMoments.from_vals(rng.random((10, 1, 2, 3)), axis=0, mom=2)
        >>> da.moveaxis((2, 1), (0, 2))
        <CentralMoments(val_shape=(3, 1, 2), mom=(2,))>
        array([[[[10.    ,  0.5205,  0.0452],
                 [10.    ,  0.5238,  0.1272]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[10.    ,  0.4438,  0.0734],
                 [10.    ,  0.628 ,  0.0524]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[10.    ,  0.5038,  0.1153],
                 [10.    ,  0.412 ,  0.0865]]]])

        """
        self._raise_if_scalar()

        def _internal_check_val(v: int | tuple[int, ...]) -> tuple[int, ...]:
            v = (v,) if isinstance(v, int) else tuple(v)
            return tuple(self._wrap_axis(x) for x in v)

        source = _internal_check_val(source)
        destination = _internal_check_val(destination)
        data = np.moveaxis(self.data, source, destination)

        # use from data for extra checks
        # return self.new_like(data=data, copy=copy, *args, **kwargs)
        return type(self).from_data(
            data,
            mom=self.mom,
            mom_ndim=self.mom_ndim,
            val_shape=data.shape[: -self.mom_ndim],
            copy=copy,
            copy_kws=copy_kws,
            **kwargs,
        )

    ###########################################################################
    # SECTION: Constructors
    ###########################################################################
    @classmethod
    @docfiller_abc()
    def zeros(
        cls,
        *,
        mom: Moments | None = None,
        mom_ndim: Mom_NDim | None = None,
        val_shape: tuple[int, ...] | int | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        zeros_kws: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        mom, mom_ndim = validate_mom_and_mom_ndim(
            mom=mom, mom_ndim=mom_ndim, shape=shape
        )

        if shape is None:
            if val_shape is None:
                val_shape = ()
            elif isinstance(val_shape, int):
                val_shape = (val_shape,)

            shape = cast("tuple[int, ...]", val_shape) + tuple(x + 1 for x in mom)  # type: ignore[redundant-cast]

        if dtype is None:
            dtype = float

        if zeros_kws is None:
            zeros_kws = {}
        data = np.zeros(shape=shape, dtype=dtype, **zeros_kws)

        kwargs = dict(kwargs, verify=False, copy=False)
        return cls.from_data(data=data, mom_ndim=mom_ndim, **kwargs)

    @classmethod
    @docfiller_abc()
    def from_data(
        cls,
        data: MyNDArray,
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
        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> x = rng.random(20)
        >>> data = central_moments(x=x, mom=2)

        >>> da = CentralMoments.from_data(data=data, mom_ndim=1)
        >>> da
        <CentralMoments(val_shape=(), mom=(2,))>
        array([20.    ,  0.5124,  0.1033])
        """
        data_verified = np.asarray(data, dtype=dtype, order="C") if verify else data
        mom, mom_ndim = validate_mom_and_mom_ndim(
            mom=mom, mom_ndim=mom_ndim, shape=data_verified.shape
        )

        if val_shape is None:
            val_shape = data_verified.shape[:-mom_ndim]

        if data_verified.shape != val_shape + tuple(x + 1 for x in mom):
            msg = f"{data.shape} does not conform to {val_shape} and {mom}"
            raise ValueError(msg)

        if copy and data_verified is data:
            if copy_kws is None:
                copy_kws = {}
            data_verified = data_verified.copy(**copy_kws)

        return cls(data=data_verified, mom_ndim=mom_ndim)

    @classmethod
    @docfiller_inherit_abc()
    def from_datas(
        cls,
        datas: MyNDArray,
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
        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> x = rng.random((10, 2))
        >>> datas = central_moments(x=x, mom=2, axis=0)
        >>> datas
        array([[10.    ,  0.6207,  0.0647],
               [10.    ,  0.404 ,  0.1185]])

        Reduce along a dimension
        >>> da = CentralMoments.from_datas(datas, axis=0, mom=2)
        >>> da
        <CentralMoments(val_shape=(), mom=(2,))>
        array([20.    ,  0.5124,  0.1033])

        """
        mom_ndim = select_mom_ndim(mom=mom, mom_ndim=mom_ndim)

        datas, axis = cls._datas_axis_to_first(
            datas,
            axis=axis or 0,
            mom_ndim=mom_ndim,
        )

        if verify:
            datas = np.asarray(datas, dtype=dtype, order="C")

        return cls.from_data(
            data=np.zeros(datas.shape[1:], dtype=dtype, order="C"),
            mom=mom,
            mom_ndim=mom_ndim,
            val_shape=val_shape,
            dtype=None,
            verify=False,
            **kwargs,
        ).push_datas(datas=datas, axis=0)

    @classmethod
    @docfiller_inherit_abc()
    def from_vals(
        cls,
        x: MyNDArray | tuple[MyNDArray, MyNDArray],
        mom: Moments,
        *,
        w: float | MyNDArray | None = None,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        broadcast: bool = False,
        axis: int | None = 0,
        **kwargs: Any,
    ) -> Self:
        """
        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> x = rng.random((100, 3))
        >>> da = CentralMoments.from_vals(x, axis=0, mom=2)
        >>> da
        <CentralMoments(val_shape=(3,), mom=(2,))>
        array([[1.0000e+02, 5.5313e-01, 8.8593e-02],
               [1.0000e+02, 5.5355e-01, 7.1942e-02],
               [1.0000e+02, 5.1413e-01, 1.0407e-01]])
        """
        axis = axis or 0
        mom_ndim = mom_to_mom_ndim(mom)

        x0 = x if mom_ndim == 1 else x[0]
        x0 = cast("MyNDArray", x0)

        if val_shape is None:
            val_shape = shape_reduce(shape=x0.shape, axis=axis)
        if dtype is None:
            dtype = x0.dtype

        return cls.zeros(val_shape=val_shape, mom=mom, dtype=dtype, **kwargs).push_vals(
            x=x, axis=axis, w=w, broadcast=broadcast
        )

    @overload
    @classmethod
    def from_resample_vals(
        cls,
        x: MyNDArray | tuple[MyNDArray, MyNDArray],
        mom: Moments,
        *,
        full_output: Literal[False] = ...,
        nrep: int | None = ...,
        freq: MyNDArray | None = ...,
        indices: MyNDArray | None = ...,
        w: MyNDArray | None = ...,
        axis: int | None = ...,
        dtype: DTypeLike | None = ...,
        broadcast: bool = ...,
        parallel: bool = ...,
        resample_kws: Mapping[str, Any] | None = ...,
        **kwargs: Any,
    ) -> Self:
        ...

    @overload
    @classmethod
    def from_resample_vals(
        cls,
        x: MyNDArray | tuple[MyNDArray, MyNDArray],
        mom: Moments,
        *,
        full_output: Literal[True],
        nrep: int | None = ...,
        freq: MyNDArray | None = ...,
        indices: MyNDArray | None = ...,
        w: MyNDArray | None = ...,
        axis: int | None = ...,
        dtype: DTypeLike | None = ...,
        broadcast: bool = ...,
        parallel: bool = ...,
        resample_kws: Mapping[str, Any] | None = ...,
        **kwargs: Any,
    ) -> tuple[Self, MyNDArray]:
        ...

    @overload
    @classmethod
    def from_resample_vals(
        cls,
        x: MyNDArray | tuple[MyNDArray, MyNDArray],
        mom: Moments,
        *,
        full_output: bool,
        nrep: int | None = ...,
        freq: MyNDArray | None = ...,
        indices: MyNDArray | None = ...,
        w: MyNDArray | None = ...,
        axis: int | None = ...,
        dtype: DTypeLike | None = ...,
        broadcast: bool = ...,
        parallel: bool = ...,
        resample_kws: Mapping[str, Any] | None = ...,
        **kwargs: Any,
    ) -> Self | tuple[Self, MyNDArray]:
        ...

    @classmethod
    @docfiller_inherit_abc()
    def from_resample_vals(
        cls,
        x: MyNDArray | tuple[MyNDArray, MyNDArray],
        mom: Moments,
        *,
        full_output: bool = False,
        nrep: int | None = None,
        freq: MyNDArray | None = None,
        indices: MyNDArray | None = None,
        w: MyNDArray | None = None,
        axis: int | None = 0,
        dtype: DTypeLike | None = None,
        broadcast: bool = False,
        parallel: bool = True,
        resample_kws: Mapping[str, Any] | None = None,
        rng: np.random.Generator | None = None,
        **kwargs: Any,
    ) -> Self | tuple[Self, MyNDArray]:
        """
        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> ndat, nrep = 10, 3
        >>> x = rng.random(ndat)
        >>> da, freq = CentralMoments.from_resample_vals(
        ...     x, nrep=nrep, axis=0, full_output=True, mom=2
        ... )
        >>> da
        <CentralMoments(val_shape=(3,), mom=(2,))>
        array([[10.    ,  0.5397,  0.0757],
               [10.    ,  0.5848,  0.0618],
               [10.    ,  0.5768,  0.0564]])

        Note that this is equivalent to (though in general faster than)

        >>> from cmomy.resample import freq_to_indices
        >>> indices = freq_to_indices(freq)
        >>> x_resamp = np.take(x, indices, axis=0)
        >>> da = CentralMoments.from_vals(x_resamp, axis=1, mom=2)
        >>> da
        <CentralMoments(val_shape=(3,), mom=(2,))>
        array([[10.    ,  0.5397,  0.0757],
               [10.    ,  0.5848,  0.0618],
               [10.    ,  0.5768,  0.0564]])

        """
        from .resample import randsamp_freq, resample_vals

        axis = axis or 0
        mom_ndim = mom_to_mom_ndim(mom)

        x0 = x if mom_ndim == 1 else x[0]
        x0 = cast("MyNDArray", x0)
        freq = randsamp_freq(
            nrep=nrep,
            ndat=x0.shape[axis],
            freq=freq,
            indices=indices,
            check=True,
            rng=rng,
        )

        if resample_kws is None:
            resample_kws = {}

        data = resample_vals(
            x,
            freq=freq,
            mom=mom,
            axis=axis,
            w=w,
            mom_ndim=mom_ndim,
            dtype=dtype,
            parallel=parallel,
            **resample_kws,
            broadcast=broadcast,
        )
        out = cls.from_data(
            data,
            mom_ndim=mom_ndim,
            mom=mom,
            copy=False,
            **kwargs,
        )

        if full_output:
            return out, freq
        return out

    @classmethod
    @docfiller_inherit_abc()
    def from_raw(
        cls,
        raw: MyNDArray,
        *,
        mom_ndim: Mom_NDim | None = None,
        mom: Moments | None = None,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        convert_kws: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> x = rng.random(10)
        >>> raw_x = (x[:, None] ** np.arange(5)).mean(axis=0)

        >>> dx_raw = CentralMoments.from_raw(raw_x, mom_ndim=1)
        >>> dx_raw.mean()
        0.5505105129032412
        >>> dx_raw.cmom()
        array([ 1.    ,  0.    ,  0.1014, -0.0178,  0.02  ])

        Which is equivalent to creating raw moments from values
        >>> dx_cen = CentralMoments.from_vals(x, axis=0, mom=4)
        >>> dx_cen.mean()
        0.5505105129032413
        >>> dx_cen.cmom()
        array([ 1.    ,  0.    ,  0.1014, -0.0178,  0.02  ])


        But note that calculating using from_raw can lead to
        numerical issues.  For example

        >>> y = x + 10000
        >>> raw_y = (y[:, None] ** np.arange(5)).mean(axis=0)
        >>> dy_raw = CentralMoments.from_raw(raw_y, mom_ndim=1)
        >>> dy_raw.mean() - 10000
        0.5505105129050207

        Note that the central moments don't match!

        >>> np.isclose(dy_raw.cmom(), dx_raw.cmom())
        array([ True,  True,  True, False, False])

        >>> dy_cen = CentralMoments.from_vals(y, axis=0, mom=4)
        >>> dy_cen.mean() - 10000
        0.5505105129032017
        >>> dy_cen.cmom()  # this matches above
        array([ 1.    ,  0.    ,  0.1014, -0.0178,  0.02  ])
        """
        mom_ndim = select_mom_ndim(mom=mom, mom_ndim=mom_ndim)

        if convert_kws is None:
            convert_kws = {}

        if mom_ndim == 1:
            data = convert.to_central_moments(raw, dtype=dtype, **convert_kws)
        elif mom_ndim == 2:
            data = convert.to_central_comoments(raw, dtype=dtype, **convert_kws)
        else:  # pragma: no cover
            msg = f"unknown mom_ndim {mom_ndim}"
            raise ValueError(msg)

        return cls.from_data(
            data,
            mom_ndim=mom_ndim,
            mom=mom,
            val_shape=val_shape,
            dtype=dtype,
            copy=False,
            **kwargs,
        )

    @classmethod
    @docfiller_inherit_abc()
    def from_raws(
        cls,
        raws: MyNDArray,
        *,
        mom_ndim: Mom_NDim | None = None,
        mom: Moments | None = None,
        axis: int | None = 0,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        convert_kws: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Examples
        --------
        >>> from cmomy.random import default_rng
        >>> rng = default_rng(0)
        >>> x = rng.random((10, 2))
        >>> raws = (x[..., None] ** np.arange(4)).mean(axis=0)
        >>> raws
        array([[1.    , 0.6207, 0.45  , 0.3412],
               [1.    , 0.404 , 0.2817, 0.2226]])
        >>> dx = CentralMoments.from_raws(raws, axis=0, mom_ndim=1)
        >>> dx.mean()
        0.5123518678291825
        >>> dx.cmom()
        array([ 1.    ,  0.    ,  0.1033, -0.0114])

        This is equivalent to

        >>> da = CentralMoments.from_vals(x.reshape(-1), axis=0, mom=3)
        >>> da.mean()
        0.5123518678291825
        >>> da.cmom()
        array([ 1.    ,  0.    ,  0.1033, -0.0114])

        """
        mom_ndim = select_mom_ndim(mom=mom, mom_ndim=mom_ndim)
        axis = axis or 0

        if convert_kws is None:
            convert_kws = {}
        if mom_ndim == 1:
            datas = convert.to_central_moments(raws, dtype=dtype, **convert_kws)
        elif mom_ndim == 2:
            datas = convert.to_central_comoments(raws, dtype=dtype, **convert_kws)
        else:  # pragma: no cover
            msg = f"unknown mom_ndim {mom_ndim}"
            raise ValueError(msg)

        return cls.from_datas(
            datas=datas,
            axis=axis,
            mom_ndim=mom_ndim,
            mom=mom,
            val_shape=val_shape,
            dtype=dtype,
            **kwargs,
        )

    # --------------------------------------------------
    # mom_ndim == 1 specific
    # --------------------------------------------------
    @staticmethod
    def _raise_if_not_1d(mom_ndim: Mom_NDim) -> None:
        if mom_ndim != 1:
            msg = "only available for mom_ndim == 1"
            raise NotImplementedError(msg)

    # special, 1d only methods
    def push_stat(
        self,
        a: MultiArray[MyNDArray],
        v: MultiArray[MyNDArray] = 0.0,
        w: MultiArray[MyNDArray] | None = None,
        broadcast: bool = True,
    ) -> Self:
        """Push statistics onto self."""
        self._raise_if_not_1d(self.mom_ndim)

        ar, target = self._check_val(x=a, target="val")
        vr = self._check_var(v=v, broadcast=broadcast)
        wr = self._check_weight(w=w, target=target)
        self._push.stat(self._data_flat, wr, ar, vr)  # type: ignore[misc]
        return self

    def push_stats(
        self,
        a: MultiArrayVals[MyNDArray],
        v: MultiArray[MyNDArray] = 0.0,
        w: MultiArray[MyNDArray] | None = None,
        axis: int = 0,
        broadcast: bool = True,
    ) -> Self:
        """Push multiple statistics onto self."""
        self._raise_if_not_1d(self.mom_ndim)

        ar, target = self._check_vals(x=a, target="vals", axis=axis)
        vr = self._check_vars(v=v, target=target, axis=axis, broadcast=broadcast)
        wr = self._check_weights(w=w, target=target, axis=axis)
        self._push.stats(self._data_flat, wr, ar, vr)  # type: ignore[misc]
        return self

    @classmethod
    def from_stat(
        cls,
        a: ArrayLike | float,
        v: MyNDArray | float = 0.0,
        w: MyNDArray | float | None = None,
        mom: Moments = 2,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        order: ArrayOrder | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create object from single weight, average, variance/covariance."""
        mom_ndim = mom_to_mom_ndim(mom)
        cls._raise_if_not_1d(mom_ndim)

        a = np.asarray(a, dtype=dtype, order=order)

        if val_shape is None:
            val_shape = a.shape
        if dtype is None:
            dtype = a.dtype

        return cls.zeros(val_shape=val_shape, mom=mom, dtype=dtype, **kwargs).push_stat(
            w=w, a=a, v=v
        )

    @classmethod
    def from_stats(
        cls,
        a: MyNDArray,
        v: MyNDArray,
        w: MyNDArray | float | None = None,
        axis: int = 0,
        mom: Moments = 2,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        order: ArrayOrder | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Create object from several statistics.

        Weights, averages, variances/covariances along
        axis.
        """
        mom_ndim = mom_to_mom_ndim(mom)
        cls._raise_if_not_1d(mom_ndim)

        a = np.asarray(a, dtype=dtype, order=order)

        # get val_shape
        if val_shape is None:
            val_shape = shape_reduce(shape=a.shape, axis=axis)
        return cls.zeros(
            val_shape=val_shape, dtype=dtype, mom=mom, **kwargs
        ).push_stats(
            a=a,
            v=v,
            w=w,
            axis=axis,
        )

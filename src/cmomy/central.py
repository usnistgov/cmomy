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
from .utils import axis_expand_broadcast, shape_reduce

if TYPE_CHECKING:
    from typing import Any, Hashable, Literal, Mapping

    from numpy.typing import ArrayLike, DTypeLike
    from typing_extensions import Self

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

    if isinstance(mom, tuple):
        mom = mom[0]

    x = np.asarray(vals, dtype=dtype, order=order)
    if dtype is None:
        dtype = x.dtype

    if w is None:
        w = np.ones_like(x)
    else:
        w = axis_expand_broadcast(
            w, x.shape, axis, roll=False, dtype=dtype, order=order
        )

    # if axis < 0:
    #     axis += x.ndim
    if axis != 0:
        x = np.moveaxis(x, axis, 0)
        w = np.moveaxis(w, axis, 0)

    shape = (mom + 1,) + x.shape[1:]
    if out is None:
        out = np.empty(shape, dtype=dtype)
    else:
        if out.shape != shape:
            # try rolling
            out = np.moveaxis(out, -1, 0)
        assert out.shape == shape

    wsum = w.sum(axis=0)  # pyright: ignore
    wsum_inv = 1.0 / wsum
    xave = np.einsum("r...,r...->...", w, x) * wsum_inv  # pyright: ignore

    shape = (-1,) + (1,) * (x.ndim)
    p = np.arange(2, mom + 1).reshape(*shape)  # pyright: ignore

    dx = (x[None, ...] - xave) ** p

    out[0, ...] = wsum
    out[1, ...] = xave
    out[2:, ...] = np.einsum("r..., mr...->m...", w, dx) * wsum_inv  # pyright: ignore

    if last:
        out = np.moveaxis(out, 0, -1)
    return out


def _central_comoments(
    vals: tuple[MyNDArray, MyNDArray],
    mom: int | tuple[int, int],
    w: MyNDArray | None = None,
    axis: int = 0,
    last: bool = True,
    broadcast: bool = False,
    dtype: DTypeLike | None = None,
    order: ArrayOrder | None = None,
    out: MyNDArray | None = None,
) -> MyNDArray:
    """Calculate central co-mom (covariance, etc) along axis."""

    if isinstance(mom, int):
        mom = (mom,) * 2

    assert len(mom) == 2
    mom = tuple(mom)  # type: ignore

    # change x to tuple of inputs
    assert isinstance(vals, tuple) and len(vals) == 2
    x, y = vals

    x = np.asarray(x, dtype=dtype, order=order)
    if dtype is None:
        dtype = x.dtype

    y = axis_expand_broadcast(
        y,
        x.shape,
        axis,
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
            w, x.shape, axis, roll=False, dtype=dtype, order=order
        )

    assert w.shape == x.shape
    assert y.shape == x.shape

    # if axis < 0:
    #     axis += x.ndim
    if axis != 0:
        x = np.moveaxis(x, axis, 0)
        y = np.moveaxis(y, axis, 0)
        w = np.moveaxis(w, axis, 0)

    shape = tuple(x + 1 for x in mom) + x.shape[1:]
    if out is None:
        out = np.empty(shape, dtype=dtype)
    else:
        if out.shape != shape:
            # try moving axis
            out = np.moveaxis(out, [-2, -1], [0, 1])
        assert out.shape == shape

    wsum = w.sum(axis=0)  # pyright: ignore
    wsum_inv = 1.0 / wsum

    xave = np.einsum("r...,r...->...", w, x) * wsum_inv  # pyright: ignore
    yave = np.einsum("r...,r...->...", w, y) * wsum_inv  # pyright: ignore

    shape = (-1,) + (1,) * (x.ndim)
    p0 = np.arange(0, mom[0] + 1).reshape(*shape)  # pyright: ignore
    p1 = np.arange(0, mom[1] + 1).reshape(*shape)  # pyright: ignore

    dx = (x[None, ...] - xave) ** p0
    dy = (y[None, ...] - yave) ** p1

    out[...] = (
        np.einsum("r...,ir...,jr...->ij...", w, dx, dy) * wsum_inv  # pyright: ignore
    )  # pyright: ignore

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

    >>> np.random.seed(0)
    >>> x = np.random.rand(10)

    Generate first 2 central moments:

    >>> moments = central_moments(x=x, mom=2)
    >>> print(moments)
    [10.      0.6158  0.034 ]

    Generate moments with weights

    >>> w = np.random.rand(10)
    >>> central_moments(x=x, w=w, mom=2)
    array([5.4734, 0.6542, 0.039 ])


    Generate co-moments

    >>> y = np.random.rand(10)
    >>> central_moments(x=(x, y), w=w, mom=(2, 2))
    array([[ 5.4734e+00,  6.9472e-01,  5.1306e-02],
           [ 6.5420e-01,  1.1600e-02, -2.6317e-03],
           [ 3.8979e-02, -3.3614e-03,  2.3024e-03]])

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
    else:
        return _central_comoments(
            vals=x,  # type: ignore
            mom=mom,  # type: ignore
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


@docfiller(CentralMomentsABC)
class CentralMoments(CentralMomentsABC[MyNDArray]):  # noqa: D101
    # def __new__(cls, data: MyNDArray, mom_ndim: Literal[1, 2] = 1):  # noqa: D102
    #     return super().__new__(cls, data=data, mom_ndim=mom_ndim)

    def __init__(self, data: MyNDArray, mom_ndim: Mom_NDim = 1) -> None:
        if mom_ndim not in (1, 2):
            raise ValueError(
                "mom_ndim must be either 1 (for central moments)"
                "or 2 (for central comoments)"
            )

        if not isinstance(data, np.ndarray):  # pyright: ignore
            raise ValueError(f"data must be an np.ndarray.  Passed type {type(data)}")

        self._mom_ndim = mom_ndim

        if data.ndim < self.mom_ndim:
            raise ValueError("not enough dimensions in data")

        self._data = data
        self._data_flat = self._data.reshape(self.shape_flat)

        if any(m <= 0 for m in self.mom):
            raise ValueError("moments must be positive")

        self._cache: dict[str, Any] = {}

    @property
    def values(self) -> MyNDArray:
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
        copy: bool = False,
        copy_kws: Mapping[str, Any] | None = None,
        verify: bool = True,
        check_shape: bool = True,
        strict: bool = False,
        **kws: Any,
    ) -> Self:
        """
        Examples
        --------
        >>> np.random.seed(0)
        >>> da = CentralMoments.from_vals(np.random.rand(10), mom=3, axis=0)
        >>> da
        <CentralMoments(val_shape=(), mom=(3,))>
        array([1.0000e+01, 6.1577e-01, 3.4031e-02, 3.8198e-03])

        >>> da2 = da.new_like().zero()
        >>> da2
        <CentralMoments(val_shape=(), mom=(3,))>
        array([0., 0., 0., 0.])

        >>> da
        <CentralMoments(val_shape=(), mom=(3,))>
        array([1.0000e+01, 6.1577e-01, 3.4031e-02, 3.8198e-03])

        """

        if data is None:
            data = np.zeros_like(self._data, order="C")
            copy = verify = check_shape = False

        kws.setdefault("mom_ndim", self.mom_ndim)

        if strict:
            kws = dict(
                dict(
                    mom=self.mom,
                    val_shape=self.val_shape,
                    dtype=self.dtype,
                ),
                **kws,
            )

        return type(self).from_data(
            data=data,
            copy=copy,
            copy_kws=copy_kws,
            verify=verify,
            check_shape=check_shape,
            **kws,
        )

    ###########################################################################
    # SECTION: To/from xarray
    ###########################################################################
    @docfiller.decorate
    def to_xarray(
        self,
        dims: XArrayDimsType = None,
        attrs: XArrayAttrsType = None,
        coords: XArrayCoordsType = None,
        name: XArrayNameType = None,
        indexes: XArrayIndexesType = None,  # pyright: ignore
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
        >>> np.random.seed(0)
        >>> da = CentralMoments.from_vals(np.random.rand(10, 1, 2), axis=0, mom=2)
        >>> da
        <CentralMoments(val_shape=(1, 2), mom=(2,))>
        array([[[10.    ,  0.5206,  0.0815],
                [10.    ,  0.6425,  0.0633]]])

        Default constructor

        >>> da.to_xarray()
        <xarray.DataArray (dim_0: 1, dim_1: 2, mom_0: 3)>
        array([[[10.    ,  0.5206,  0.0815],
                [10.    ,  0.6425,  0.0633]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0

        Setting attributes

        >>> da.to_xarray()
        <xarray.DataArray (dim_0: 1, dim_1: 2, mom_0: 3)>
        array([[[10.    ,  0.5206,  0.0815],
                [10.    ,  0.6425,  0.0633]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0
        >>> da
        <CentralMoments(val_shape=(1, 2), mom=(2,))>
        array([[[10.    ,  0.5206,  0.0815],
                [10.    ,  0.6425,  0.0633]]])

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
                dims = tuple(dims)  # type: ignore

            if len(dims) == self.ndim:
                dims_output = dims

            elif len(dims) == self.val_ndim:
                if mom_dims is None:
                    mom_dims = tuple(f"mom_{i}" for i in range(self.mom_ndim))
                elif isinstance(mom_dims, str):
                    mom_dims = (mom_dims,)
                else:
                    # try to convert to tuple
                    mom_dims = tuple(mom_dims)  # type: ignore

                assert (
                    len(mom_dims) == self.mom_ndim
                ), f"mom_dims={mom_dims} has wrong length?"

                dims_output = dims + mom_dims

            else:
                raise ValueError(
                    f"Problem with {dims}, {mom_dims}.  Total length should be {self.ndim}"
                )
            out = xr.DataArray(
                self.data, dims=dims_output, coords=coords, attrs=attrs, name=name
            )

        if copy:
            out = out.copy()

        return out

    @docfiller.decorate
    def to_xcentralmoments(
        self,
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
        >>> np.random.seed(0)
        >>> da = CentralMoments.from_vals(np.random.rand(10, 1, 2), axis=0, mom=2)
        >>> da
        <CentralMoments(val_shape=(1, 2), mom=(2,))>
        array([[[10.    ,  0.5206,  0.0815],
                [10.    ,  0.6425,  0.0633]]])

        Default constructor

        >>> da.to_xcentralmoments()
        <xCentralMoments(val_shape=(1, 2), mom=(2,))>
        <xarray.DataArray (dim_0: 1, dim_1: 2, mom_0: 3)>
        array([[[10.    ,  0.5206,  0.0815],
                [10.    ,  0.6425,  0.0633]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0

        Setting attributes

        >>> da.to_xcentralmoments()
        <xCentralMoments(val_shape=(1, 2), mom=(2,))>
        <xarray.DataArray (dim_0: 1, dim_1: 2, mom_0: 3)>
        array([[[10.    ,  0.5206,  0.0815],
                [10.    ,  0.6425,  0.0633]]])
        Dimensions without coordinates: dim_0, dim_1, mom_0
        >>> da
        <CentralMoments(val_shape=(1, 2), mom=(2,))>
        array([[[10.    ,  0.5206,  0.0815],
                [10.    ,  0.6425,  0.0633]]])

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
            assert axis is not None
            target_shape = shape_insert_axis(self.val_shape, axis, x.shape[axis])
        elif style == "data":
            target_shape = self.shape
        elif style == "datas":
            # make sure axis in limits
            assert axis is not None
            target_shape = shape_insert_axis(self.shape, axis, x.shape[axis])
        elif style == "var":
            target_shape = self.shape_var
        elif style == "vars":
            assert other is not None and axis is not None
            target_shape = shape_insert_axis(self.shape_var, axis, other.shape[axis])
        else:
            raise ValueError(f"unknown string style name {style}")

        return target_shape

    def _verify_value(
        self,
        *,
        x: MultiArray[MyNDArray],
        target: str | MyNDArray,
        shape_flat: tuple[int, ...],
        axis: int | None = None,
        dim: Hashable | None = None,  # pyright: ignore # included here for consistency
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

        elif isinstance(target, np.ndarray):  # pyright: ignore
            target_shape = target.shape
            target_output = target

        else:
            raise ValueError("unknown target type")

        x = axis_expand_broadcast(
            x,
            target_shape,
            axis,
            verify=False,
            expand=expand,
            broadcast=broadcast,
            dtype=self.dtype,
            roll=False,
        )

        # check shape:
        assert (
            x.shape == target_shape
        ), f"x.shape = {x.shape} not equal target_shape={target_shape}"

        # move axis
        nrec: tuple[int, ...]
        if axis is not None:
            if axis != 0:
                x = np.moveaxis(x, axis, 0)
            nrec = (x.shape[0],)
        else:
            nrec = ()

        x = x.reshape(nrec + shape_flat)

        if x.ndim == 0:
            x = x[()]

        return x, target_output

    @docfiller_inherit_abc()
    def push_data(self, data: MultiArrayVals[MyNDArray]) -> Self:
        """
        Examples
        --------
        >>> np.random.seed(0)
        >>> xs = np.random.rand(2, 10)
        >>> datas = [central_moments(x=x, mom=2) for x in xs]
        >>> da = CentralMoments.from_data(datas[0], mom_ndim=1)
        >>> da
        <CentralMoments(val_shape=(), mom=(2,))>
        array([10.    ,  0.6158,  0.034 ])


        >>> da.push_data(datas[1])
        <CentralMoments(val_shape=(), mom=(2,))>
        array([20.    ,  0.5816,  0.0761])


        Which is equivalent to

        >>> CentralMoments.from_vals(xs.reshape(-1), mom=2)
        <CentralMoments(val_shape=(), mom=(2,))>
        array([20.    ,  0.5816,  0.0761])

        """
        return super().push_data(data=data)

    @docfiller_inherit_abc()
    def push_datas(
        self,
        datas: MultiArray[MyNDArray],
        axis: int | None = 0,
        **kwargs: Any,  # pyrigt: ignore
    ) -> Self:
        """
        Examples
        --------
        >>> np.random.seed(0)
        >>> xs = np.random.rand(2, 10)
        >>> datas = np.array([central_moments(x=x, mom=2) for x in xs])
        >>> da = CentralMoments.zeros(mom=2)
        >>> da.push_datas(datas, axis=0)
        <CentralMoments(val_shape=(), mom=(2,))>
        array([20.    ,  0.5816,  0.0761])


        Which is equivalent to

        >>> CentralMoments.from_vals(xs.reshape(-1), mom=2)
        <CentralMoments(val_shape=(), mom=(2,))>
        array([20.    ,  0.5816,  0.0761])
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
        >>> np.random.seed(0)
        >>> x = np.random.rand(10, 2)
        >>> y = np.random.rand(10)
        >>> w = np.random.rand(10)

        >>> da = CentralMoments.zeros(val_shape=(2,), mom=(2, 2))
        >>> for xx, yy, ww in zip(x, y, w):
        ...     _ = da.push_val(x=(xx, yy), w=ww, broadcast=True)
        ...

        >>> da
        <CentralMoments(val_shape=(2,), mom=(2, 2))>
        array([[[ 5.5544e+00,  6.0763e-01,  5.9601e-02],
                [ 4.5048e-01, -1.5438e-02,  7.5730e-04],
                [ 7.9759e-02,  2.0716e-04,  3.2777e-03]],
        <BLANKLINE>
               [[ 5.5544e+00,  6.0763e-01,  5.9601e-02],
                [ 6.7438e-01, -4.0366e-02, -3.8163e-04],
                [ 6.4172e-02,  9.6487e-03,  6.3702e-03]]])


        Which is the same as

        >>> CentralMoments.from_vals(x=(x, y), w=w, mom=(2, 2), broadcast=True)
        <CentralMoments(val_shape=(2,), mom=(2, 2))>
        array([[[ 5.5544e+00,  6.0763e-01,  5.9601e-02],
                [ 4.5048e-01, -1.5438e-02,  7.5730e-04],
                [ 7.9759e-02,  2.0716e-04,  3.2777e-03]],
        <BLANKLINE>
               [[ 5.5544e+00,  6.0763e-01,  5.9601e-02],
                [ 6.7438e-01, -4.0366e-02, -3.8163e-04],
                [ 6.4172e-02,  9.6487e-03,  6.3702e-03]]])

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
        **kwargs: Any,  # pyright: ignore
    ) -> Self:
        """
        Examples
        --------
        >>> np.random.seed(0)
        >>> x = np.random.rand(10, 2)
        >>> y = np.random.rand(10)
        >>> w = np.random.rand(10)

        >>> da = CentralMoments.zeros(val_shape=(2,), mom=(2, 2))
        >>> da.push_vals(x=(x, y), w=w, broadcast=True)
        <CentralMoments(val_shape=(2,), mom=(2, 2))>
        array([[[ 5.5544e+00,  6.0763e-01,  5.9601e-02],
                [ 4.5048e-01, -1.5438e-02,  7.5730e-04],
                [ 7.9759e-02,  2.0716e-04,  3.2777e-03]],
        <BLANKLINE>
               [[ 5.5544e+00,  6.0763e-01,  5.9601e-02],
                [ 6.7438e-01, -4.0366e-02, -3.8163e-04],
                [ 6.4172e-02,  9.6487e-03,  6.3702e-03]]])


        Which is the same as

        >>> CentralMoments.from_vals(x=(x, y), w=w, mom=(2, 2), broadcast=True)
        <CentralMoments(val_shape=(2,), mom=(2, 2))>
        array([[[ 5.5544e+00,  6.0763e-01,  5.9601e-02],
                [ 4.5048e-01, -1.5438e-02,  7.5730e-04],
                [ 7.9759e-02,  2.0716e-04,  3.2777e-03]],
        <BLANKLINE>
               [[ 5.5544e+00,  6.0763e-01,  5.9601e-02],
                [ 6.7438e-01, -4.0366e-02, -3.8163e-04],
                [ 6.4172e-02,  9.6487e-03,  6.3702e-03]]])

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
        **kws: Any,
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
        **kws: Any,
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
        **kws: Any,
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
        **kws: Any,
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
        **kws
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
            nrep=nrep, indices=indices, freq=freq, size=self.val_shape[axis], check=True
        )
        data = resample_data(
            self.data, freq, mom=self.mom, axis=axis, parallel=parallel, **resample_kws
        )
        out = type(self).from_data(data, mom_ndim=self.mom_ndim, copy=False, **kws)

        if full_output:
            return out, freq
        else:
            return out

    @docfiller.decorate
    def reduce(self, axis: int | None = None, **kws: Any) -> Self:
        """
        Create new object reduce along axis.

        Parameters
        ----------
        {axis}
        **kws
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
            self.values, mom_ndim=self.mom_ndim, axis=axis, **kws
        )

    @docfiller.decorate
    def block(
        self,
        block_size: int | None = None,
        axis: int | None = None,
        **kws: Any,
    ) -> Self:
        """
        Block average reduction.

        Parameters
        ----------
        block_size : int
            number of consecutive records to combine
        {axis}
        **kws
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
        return type(self).from_datas(datas=datas, mom_ndim=self.mom_ndim, axis=1, **kws)

    @docfiller.decorate
    def reshape(
        self,
        shape: tuple[int, ...],
        copy: bool = True,
        copy_kws: Mapping[str, Any] | None = None,
        **kws: Any,
    ) -> Self:
        """
        Create a new object with reshaped data.

        Parameters
        ----------
        shape : tuple
            shape of values part of data.
        {copy}
        {copy_kws}
        **kws
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
        >>> x = np.random.seed(0)
        >>> da = CentralMoments.from_vals(np.random.rand(10, 2, 3), mom=2)
        >>> da
        <CentralMoments(val_shape=(2, 3), mom=(2,))>
        array([[[10.    ,  0.4549,  0.044 ],
                [10.    ,  0.6019,  0.0849],
                [10.    ,  0.6049,  0.0911]],
        <BLANKLINE>
               [[10.    ,  0.5372,  0.0591],
                [10.    ,  0.4262,  0.0843],
                [10.    ,  0.4733,  0.0591]]])

               [[10.        ,  0.53720667,  0.05909394],
                [10.        ,  0.42622908,  0.08434857],
                [10.        ,  0.47326641,  0.05907737]]])

        >>> da.reshape(shape=(-1,))
        <CentralMoments(val_shape=(6,), mom=(2,))>
        array([[10.    ,  0.4549,  0.044 ],
               [10.    ,  0.6019,  0.0849],
               [10.    ,  0.6049,  0.0911],
               [10.    ,  0.5372,  0.0591],
               [10.    ,  0.4262,  0.0843],
               [10.    ,  0.4733,  0.0591]])
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
            verify=True,
            check_shape=True,
            dtype=self.dtype,
            **kws,
        )

    @docfiller.decorate
    def moveaxis(
        self,
        source: int | tuple[int, ...],
        destination: int | tuple[int, ...],
        copy: bool = True,
        copy_kws: Mapping[str, Any] | None = None,
        **kws: Any,
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
        >>> np.random.seed(0)
        >>> da = CentralMoments.from_vals(np.random.rand(10, 1, 2, 3), axis=0)
        >>> da.moveaxis((2, 1), (0, 2))
        <CentralMoments(val_shape=(3, 1, 2), mom=(2,))>
        array([[[[10.    ,  0.4549,  0.044 ],
                 [10.    ,  0.5372,  0.0591]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[10.    ,  0.6019,  0.0849],
                 [10.    ,  0.4262,  0.0843]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[10.    ,  0.6049,  0.0911],
                 [10.    ,  0.4733,  0.0591]]]])

        """
        self._raise_if_scalar()

        def _internal_check_val(v: int | tuple[int, ...]) -> tuple[int, ...]:
            if isinstance(v, int):
                v = (v,)
            else:
                v = tuple(v)
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
            **kws,
        )

    ###########################################################################
    # SECTION: Constructors
    ###########################################################################
    @classmethod
    @docfiller_abc()
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
        if shape is None:
            assert mom is not None
            if isinstance(mom, int):
                mom = (mom,)
            if mom_ndim is None:
                mom_ndim = len(mom)  # type: ignore
            assert len(mom) == mom_ndim

            if val_shape is None:
                val_shape = ()
            elif isinstance(val_shape, int):
                val_shape = (val_shape,)
            shape = val_shape + tuple(x + 1 for x in mom)

        else:
            assert mom_ndim is not None

        if dtype is None:
            dtype = float

        if zeros_kws is None:
            zeros_kws = {}
        data = np.zeros(shape=shape, dtype=dtype, **zeros_kws)

        kws = dict(kws, verify=False, copy=False, check_shape=False)
        return cls.from_data(data=data, mom_ndim=mom_ndim, **kws)

    @classmethod
    @docfiller_abc()
    def from_data(
        cls,
        data: MyNDArray,
        mom: Moments | None = None,
        mom_ndim: Mom_NDim | None = None,
        val_shape: tuple[int, ...] | None = None,
        copy: bool = True,
        copy_kws: Mapping[str, Any] | None = None,
        verify: bool = True,
        check_shape: bool = True,
        dtype: DTypeLike | None = None,
        # **kws: Any,
    ) -> Self:
        """
        Examples
        --------
        >>> np.random.seed(0)
        >>> x = np.random.rand(20)
        >>> data = central_moments(x=x, mom=2)

        >>> da = CentralMoments.from_data(data=data, mom_ndim=1)
        >>> da
        <CentralMoments(val_shape=(), mom=(2,))>
        array([20.    ,  0.5816,  0.0761])
        """
        mom_ndim = cls._choose_mom_ndim(mom, mom_ndim)

        if verify:
            data_verified = np.asarray(data, dtype=dtype, order="C")
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
            data_verified = data_verified.copy(**copy_kws)

        return cls(data=data_verified, mom_ndim=mom_ndim)

    @classmethod
    @docfiller_inherit_abc()
    def from_datas(
        cls,
        datas: MyNDArray,
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
        Examples
        --------
        >>> np.random.seed(0)
        >>> x = np.random.rand(10, 2)
        >>> datas = central_moments(x=x, mom=2, axis=0)
        >>> datas
        array([[10.    ,  0.5206,  0.0815],
               [10.    ,  0.6425,  0.0633]])

        Reduce along a dimension
        >>> da = CentralMoments.from_datas(datas, axis=0, mom=2)
        >>> da
        <CentralMoments(val_shape=(), mom=(2,))>
        array([20.    ,  0.5816,  0.0761])

        """

        axis = axis or 0
        mom_ndim = cls._choose_mom_ndim(mom, mom_ndim)

        if verify:
            datas = np.asarray(datas, dtype=dtype)
        datas, axis = cls._datas_axis_to_first(datas, axis=axis, mom_ndim=mom_ndim)
        if check_shape:
            if val_shape is None:
                val_shape = datas.shape[1:-mom_ndim]

            mom = cls._check_mom(mom, mom_ndim, datas.shape)
            assert datas.shape[1:] == val_shape + tuple(x + 1 for x in mom)

        if dtype is None:
            dtype = datas.dtype

        return cls.zeros(
            shape=datas.shape[1:], mom_ndim=mom_ndim, dtype=dtype, **kws
        ).push_datas(datas=datas, axis=0)

    @classmethod
    @docfiller_inherit_abc()
    def from_vals(
        cls,
        x: MyNDArray | tuple[MyNDArray, MyNDArray],
        w: float | MyNDArray | None = None,
        axis: int | None = 0,
        # dim: Hashable | None = None,
        mom: Moments = 2,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        broadcast: bool = False,
        **kws: Any,
    ) -> Self:
        """
        Examples
        --------
        >>> np.random.seed(0)
        >>> x = np.random.rand(100, 3)
        >>> da = CentralMoments.from_vals(x, axis=0, mom=2)
        >>> da
        <CentralMoments(val_shape=(3,), mom=(2,))>
        array([[1.0000e+02, 5.0168e-01, 8.7872e-02],
               [1.0000e+02, 4.8657e-01, 8.5287e-02],
               [1.0000e+02, 5.2226e-01, 7.8481e-02]])
        """

        axis = axis or 0
        mom_ndim = cls._mom_ndim_from_mom(mom)
        x0 = x if mom_ndim == 1 else x[0]
        x0 = cast("MyNDArray", x0)
        if val_shape is None:
            val_shape = shape_reduce(x0.shape, axis)
        if dtype is None:
            dtype = x0.dtype

        return cls.zeros(val_shape=val_shape, mom=mom, dtype=dtype, **kws).push_vals(
            x=x, axis=axis, w=w, broadcast=broadcast
        )

    @overload
    @classmethod
    def from_resample_vals(
        cls,
        x: MyNDArray | tuple[MyNDArray, MyNDArray],
        nrep: int | None = ...,
        *,
        full_output: Literal[False] = ...,
        freq: MyNDArray | None = ...,
        indices: MyNDArray | None = ...,
        w: MyNDArray | None = ...,
        axis: int | None = ...,
        mom: Moments = ...,
        dtype: DTypeLike | None = ...,
        broadcast: bool = ...,
        parallel: bool = ...,
        resample_kws: Mapping[str, Any] | None = ...,
        **kws: Any,
    ) -> Self:
        ...

    @overload
    @classmethod
    def from_resample_vals(
        cls,
        x: MyNDArray | tuple[MyNDArray, MyNDArray],
        nrep: int | None = ...,
        *,
        full_output: Literal[True],
        freq: MyNDArray | None = ...,
        indices: MyNDArray | None = ...,
        w: MyNDArray | None = ...,
        axis: int | None = ...,
        mom: Moments = ...,
        dtype: DTypeLike | None = ...,
        broadcast: bool = ...,
        parallel: bool = ...,
        resample_kws: Mapping[str, Any] | None = ...,
        **kws: Any,
    ) -> tuple[Self, MyNDArray]:
        ...

    @overload
    @classmethod
    def from_resample_vals(
        cls,
        x: MyNDArray | tuple[MyNDArray, MyNDArray],
        nrep: int | None = ...,
        *,
        full_output: bool,
        axis: int | None = ...,
        freq: MyNDArray | None = ...,
        indices: MyNDArray | None = ...,
        w: MyNDArray | None = ...,
        mom: Moments = ...,
        dtype: DTypeLike | None = ...,
        broadcast: bool = ...,
        parallel: bool = ...,
        resample_kws: Mapping[str, Any] | None = ...,
        **kws: Any,
    ) -> Self | tuple[Self, MyNDArray]:
        ...

    @classmethod
    @docfiller_inherit_abc()
    def from_resample_vals(
        cls,
        x: MyNDArray | tuple[MyNDArray, MyNDArray],
        nrep: int | None = None,
        *,
        full_output: bool = False,
        freq: MyNDArray | None = None,
        indices: MyNDArray | None = None,
        w: MyNDArray | None = None,
        axis: int | None = 0,
        mom: Moments = 2,
        dtype: DTypeLike | None = None,
        broadcast: bool = False,
        parallel: bool = True,
        resample_kws: Mapping[str, Any] | None = None,
        **kws: Any,
    ) -> Self | tuple[Self, MyNDArray]:
        """
        Examples
        --------
        >>> np.random.seed(0)
        >>> ndat, nrep = 10, 3
        >>> x = np.random.rand(ndat)
        >>> from cmomy.resample import numba_random_seed
        >>> numba_random_seed(0)
        >>> da, freq = CentralMoments.from_resample_vals(
        ...     x, nrep=nrep, axis=0, full_output=True
        ... )
        >>> da
        <CentralMoments(val_shape=(3,), mom=(2,))>
        array([[10.    ,  0.5777,  0.0174],
               [10.    ,  0.7872,  0.0381],
               [10.    ,  0.5633,  0.0259]])

        Note that this is equivalent to (though in general faster than)

        >>> from cmomy.resample import freq_to_indices
        >>> indices = freq_to_indices(freq)
        >>> x_resamp = np.take(x, indices, axis=0)
        >>> da = CentralMoments.from_vals(x_resamp, axis=1, mom=2)
        >>> da
        <CentralMoments(val_shape=(3,), mom=(2,))>
        array([[10.    ,  0.5777,  0.0174],
               [10.    ,  0.7872,  0.0381],
               [10.    ,  0.5633,  0.0259]])

        """

        from .resample import randsamp_freq, resample_vals

        axis = axis or 0
        mom_ndim = cls._mom_ndim_from_mom(mom)

        x0 = x if mom_ndim == 1 else x[0]
        x0 = cast("MyNDArray", x0)
        freq = randsamp_freq(
            nrep=nrep,
            freq=freq,
            indices=indices,
            size=x0.shape[axis],
            check=True,
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
            verify=True,
            check_shape=True,
            copy=False,
            **kws,
        )

        if full_output:
            return out, freq
        else:
            return out

    @classmethod
    @docfiller_inherit_abc()
    def from_raw(
        cls,
        raw: MyNDArray,
        mom_ndim: Mom_NDim | None = None,
        mom: Moments | None = None,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        convert_kws: Mapping[str, Any] | None = None,
        **kws: Any,
    ) -> Self:
        """
        Examples
        --------
        >>> np.random.seed(0)
        >>> x = np.random.rand(10)
        >>> raw_x = (x[:, None] ** np.arange(5)).mean(axis=0)

        >>> dx_raw = CentralMoments.from_raw(raw_x, mom_ndim=1)
        >>> dx_raw.mean()
        0.6157662833145425
        >>> dx_raw.cmom()
        array([1.    , 0.    , 0.034 , 0.0038, 0.0026])

        Which is equivalent to creating raw moments from values
        >>> dx_cen = CentralMoments.from_vals(x, axis=0, mom=4)
        >>> dx_cen.mean()
        0.6157662833145425
        >>> dx_cen.cmom()
        array([1.    , 0.    , 0.034 , 0.0038, 0.0026])


        But note that calculating using from_raw can lead to
        numerical issues.  For example

        >>> y = x + 10000
        >>> raw_y = (y[:, None] ** np.arange(5)).mean(axis=0)
        >>> dy_raw = CentralMoments.from_raw(raw_y, mom_ndim=1)
        >>> dy_raw.mean() - 10000
        0.6157662833156792
        >>> dy_raw.cmom()  # note that these don't match dx_raw, which they should
        array([ 1.0000e+00,  0.0000e+00,  3.4031e-02,  4.7744e-03, -1.8350e+01])

        >>> dy_cen = CentralMoments.from_vals(y, axis=0, mom=4)
        >>> dy_cen.mean() - 10000
        0.6157662833156792
        >>> dy_cen.cmom()  # this matches above
        array([1.    , 0.    , 0.034 , 0.0038, 0.0026])
        """

        mom_ndim = cls._choose_mom_ndim(mom, mom_ndim)

        if convert_kws is None:
            convert_kws = {}

        if mom_ndim == 1:
            data = convert.to_central_moments(raw, dtype=dtype, **convert_kws)
        elif mom_ndim == 2:
            data = convert.to_central_comoments(raw, dtype=dtype, **convert_kws)
        else:
            raise ValueError(f"unknown mom_ndim {mom_ndim}")

        kws = dict(dict(verify=True, check_shape=True), **kws)

        return cls.from_data(
            data,
            mom_ndim=mom_ndim,
            mom=mom,
            val_shape=val_shape,
            dtype=dtype,
            copy=False,
            **kws,
        )

    @classmethod
    @docfiller_inherit_abc()
    def from_raws(
        cls,
        raws: MyNDArray,
        mom_ndim: Mom_NDim | None = None,
        mom: Moments | None = None,
        axis: int | None = 0,
        val_shape: tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        convert_kws: Mapping[str, Any] | None = None,
        **kws: Any,
    ) -> Self:
        """
        Examples
        --------
        >>> np.random.seed(0)
        >>> x = np.random.rand(10, 2)
        >>> raws = (x[..., None] ** np.arange(4)).mean(axis=0)
        >>> raws
        array([[1.    , 0.5206, 0.3525, 0.259 ],
               [1.    , 0.6425, 0.4762, 0.374 ]])
        >>> dx = CentralMoments.from_raws(raws, axis=0, mom_ndim=1)
        >>> dx.mean()
        0.5815548245225974
        >>> dx.cmom()
        array([ 1.    ,  0.    ,  0.0761, -0.013 ])

        This is equivalent to

        >>> da = CentralMoments.from_vals(x.reshape(-1), axis=0, mom=3)
        >>> da.mean()
        0.5815548245225974
        >>> da.cmom()
        array([ 1.    ,  0.    ,  0.0761, -0.013 ])

        """
        mom_ndim = cls._choose_mom_ndim(mom, mom_ndim)
        axis = axis or 0

        if convert_kws is None:
            convert_kws = {}
        if mom_ndim == 1:
            datas = convert.to_central_moments(raws, dtype=dtype, **convert_kws)
        elif mom_ndim == 2:
            datas = convert.to_central_comoments(raws, dtype=dtype, **convert_kws)
        else:
            raise ValueError(f"unknown mom_ndim {mom_ndim}")

        return cls.from_datas(
            datas=datas,
            axis=axis,
            mom_ndim=mom_ndim,
            mom=mom,
            val_shape=val_shape,
            dtype=dtype,
            **kws,
        )

    # --------------------------------------------------
    # mom_ndim == 1 specific
    # --------------------------------------------------
    @staticmethod
    def _raise_if_not_1d(mom_ndim: Mom_NDim) -> None:
        if mom_ndim != 1:
            raise NotImplementedError("only available for mom_ndim == 1")

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
        self._push.stat(self._data_flat, wr, ar, vr)  # type: ignore
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
        self._push.stats(self._data_flat, wr, ar, vr)  # type: ignore
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
        **kws: Any,
    ) -> Self:
        """Create object from single weight, average, variance/covariance."""
        mom_ndim = cls._mom_ndim_from_mom(mom)
        cls._raise_if_not_1d(mom_ndim)

        a = np.asarray(a, dtype=dtype, order=order)

        if val_shape is None:
            val_shape = a.shape
        if dtype is None:
            dtype = a.dtype

        return cls.zeros(val_shape=val_shape, mom=mom, dtype=dtype, **kws).push_stat(
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
        **kws: Any,
    ) -> Self:
        """
        Create object from several statistics.

        Weights, averages, variances/covariances along
        axis.
        """

        mom_ndim = cls._mom_ndim_from_mom(mom)
        cls._raise_if_not_1d(mom_ndim)

        a = np.asarray(a, dtype=dtype, order=order)

        # get val_shape
        if val_shape is None:
            val_shape = shape_reduce(a.shape, axis)
        return cls.zeros(val_shape=val_shape, dtype=dtype, mom=mom, **kws).push_stats(
            a=a,
            v=v,
            w=w,
            axis=axis,
        )

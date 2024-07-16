"""
Moving averages (:mod:`~cmomy.moving`)
======================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ._utils import (
    MISSING,
    axes_data_reduction,
    normalize_axis_index,
    parallel_heuristic,
    select_axis_dim_mult,
    select_dtype,
    validate_axis,
    validate_mom_ndim,
)
from .docstrings import docfiller

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from cmomy.typing import DimsReduceMult

    from .typing import (
        AxisReduce,
        AxisReduceMult,
        FloatT,
        MissingType,
        Mom_NDim,
        NDArrayAny,
    )


# * Moving average
# def construct_rolling_window_array(
#     x: NDArray[FloatT],
#     axis: AxisReduceMult,
#     window: int | Sequence[int],
#     center: bool | Sequence[bool] = False,
#     fill_value: ArrayLike = np.nan,
#     mom_ndim: Mom_NDim | None = None,
#     **kwargs: Any,
# ) -> NDArray[FloatT]:
#     """
#     Convert an array to one with rolling windows.

#     Parameters
#     ----------
#     x : array
#     axis : int or iterable of int
#     window : int or sequence of int
#     center : bool

#     Returns
#     -------
#     output : array
#         Array of shape ``(*shape, window)`` if ``mom_ndim = None`` or
#         ``(*shape[:-mom_ndim], window, *shape[-mom_ndim:])`` if ``mom_ndim is
#         not None``. That is, the new window dimension is placed at the end, but
#         before any moment dimensions if they are specified.
#     """
#     ndim = x.ndim - (0 if mom_ndim is None else validate_mom_ndim(mom_ndim))
#     if axis is None:
#         axis = tuple(range(ndim))
#     axis = normalize_axis_tuple(axis, ndim, msg_prefix="rolling")

#     nroll = len(axis)
#     window = (window,) * nroll if isinstance(window, int) else tuple(window)
#     center = (center,) * nroll if isinstance(center, bool) else tuple(center)
#     if any(len(x) != nroll for x in (window, center)):
#         msg = f"{axis=}, {window=}, {center=} must have same length"
#         raise ValueError(msg)

#     pads = [(0, 0)] * x.ndim
#     for a, win, cent in zip(axis, window, center):
#         if cent:
#             start = win // 2
#             end = win - 1 - start
#             pads[a] = (start, end)
#         else:
#             pads[a] = (win - 1, 0)

#     padded: NDArray[FloatT] = np.pad(
#         x, pads, mode="constant", constant_values=fill_value, **kwargs
#     )

#     from numpy.lib.stride_tricks import sliding_window_view

#     out = cast(
#         "NDArray[FloatT]",
#         sliding_window_view(  # type: ignore[call-overload]
#             padded,
#             window_shape=window if isinstance(window, int) else tuple(window),
#             axis=axis,  # pyright: ignore[reportArgumentType]
#         ),
#     )

#     if mom_ndim is not None:
#         out = np.moveaxis(out, range(-len(axis), 0), range(ndim, ndim + len(axis)))  # pyright: ignore[reportUnknownArgumentType]
#     return out


@docfiller.decorate
def construct_rolling_window_array(
    x: NDArray[FloatT] | xr.DataArray,
    window: int | Sequence[int],
    axis: AxisReduceMult | MissingType = MISSING,
    center: bool | Sequence[bool] = False,
    stride: int | Sequence[int] = 1,
    fill_value: ArrayLike = np.nan,
    mom_ndim: Mom_NDim | None = None,
    # xarray specific
    dim: DimsReduceMult | MissingType = MISSING,
    window_dim: str | Sequence[str] | None = None,
    keep_attrs: bool | None = None,
    **kwargs: Any,
) -> NDArray[FloatT] | xr.DataArray:
    """
    Convert an array to one with rolling windows.

    Parameters
    ----------
    x : array
    axis : int or iterable of int
    window : int or sequence of int
    center : bool
    fill_value : scalar

    Returns
    -------
    output : array
        Array of shape ``(*shape, window)`` if ``mom_ndim = None`` or
        ``(*shape[:-mom_ndim], window, *shape[-mom_ndim:])`` if ``mom_ndim is
        not None``. That is, the new window dimension is placed at the end, but
        before any moment dimensions if they are specified.

    Notes
    -----
    This function uses different syntax compared to
    :meth:`~xarray.DataArray.rolling`. Instead of mappings ffor ``center``,
    etc, here you pass scalar or sequence values corresponding to axis/dim.
    Corresponding mappings are created from, for example ``center=dict(zip(dim,
    center))``.

    See Also
    --------
    xarray.DataArray.rolling
    xarray.core.rolling.DataArrayRolling.construct
    """
    if isinstance(x, xr.DataArray):
        mom_ndim = validate_mom_ndim(mom_ndim) if mom_ndim is not None else mom_ndim
        axis, dim = select_axis_dim_mult(x, axis=axis, dim=dim, mom_ndim=mom_ndim)

        nroll = len(axis)
        window = (window,) * nroll if isinstance(window, int) else window
        center = (center,) * nroll if isinstance(center, bool) else center
        stride = (stride,) * nroll if isinstance(stride, int) else stride

        window_dim = (
            tuple(f"rolling_{d}" for d in dim)
            if window_dim is None
            else (window_dim,)
            if isinstance(window_dim, str)
            else window_dim
        )

        if any(len(x) != nroll for x in (window, center, stride, window_dim)):
            msg = f"{axis=}, {window=}, {center=}, {stride=}, {window_dim=} must have same length"
            raise ValueError(msg)

        xout = x.rolling(
            dict(zip(dim, window)),
            center=dict(zip(dim, center)),
            **kwargs,
        ).construct(
            window_dim=dict(zip(dim, window_dim)),
            stride=dict(zip(dim, stride)),
            fill_value=fill_value,
            keep_attrs=keep_attrs,
        )

        if mom_ndim is not None:
            xout = xout.transpose(..., *x.dims[-mom_ndim:])

        return xout

    return construct_rolling_window_array(
        x=xr.DataArray(x),
        window=window,
        axis=axis,
        center=center,
        stride=stride,
        fill_value=fill_value,
        mom_ndim=mom_ndim,
        **kwargs,
    ).to_numpy()


def move_data(
    data: ArrayLike,
    *,
    axis: AxisReduce | MissingType = MISSING,
    mom_ndim: Mom_NDim = 1,
    window: int,
    min_count: int | None = None,
    center: bool = False,
    parallel: bool | None = None,
    dtype: DTypeLike = None,
    out: NDArrayAny | None = None,
) -> NDArrayAny:
    """
    Moving average of central moments array.

    Parameters
    ----------
    data : array-like
    """
    min_count = window if min_count is None else min_count

    mom_ndim = validate_mom_ndim(mom_ndim)
    dtype = select_dtype(data, out=out, dtype=dtype)

    # special to support multiple reduction dimensions...
    data = np.asarray(data, dtype=dtype)
    axis = normalize_axis_index(validate_axis(axis), data.ndim, mom_ndim, "move_data")

    valid: list[slice] | None = None
    if center:
        shift = (-window // 2) + 1

        valid = [slice(None)] * data.ndim
        valid[axis] = slice(-shift, None)

        pads = [(0, 0)] * data.ndim
        pads[axis] = (0, -shift)

        data = np.pad(
            data,
            pads,
            mode="constant",
            constant_values=0.0,
        )

    axes = axes_data_reduction(
        # add in data_tmp, window, count
        tuple(range(-mom_ndim, 0)),
        (),
        (),
        mom_ndim=mom_ndim,
        axis=axis,
        out_has_axis=True,
    )

    from ._lib.factory import factory_move_data

    data_tmp = np.zeros(data.shape[-mom_ndim:], dtype=dtype)
    out = factory_move_data(
        mom_ndim=mom_ndim,
        parallel=parallel_heuristic(parallel, data.size * mom_ndim),
    )(data, data_tmp, window, min_count, out=out, axes=axes)

    if valid is not None:
        out = out[tuple(valid)]  # pyright: ignore[]

    return out

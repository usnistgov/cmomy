"""
Routines to convert central (co)moments to raw (co)moments. (:mod:`cmomy.convert`)
==================================================================================
"""
from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import numpy as np
from module_utilities.docfiller import DocFiller

if TYPE_CHECKING:
    from typing import Any, Callable, Sequence

    from numpy.typing import ArrayLike, DTypeLike

    from .typing import ArrayOrder, MyNDArray

_shared_docs = r"""
Parameters
----------
x_cmom | x : ndarray
    Central moments array.  The expected structure is:

    * ``x[..., 0]`` : weight
    * ``x[..., 1]`` : mean
    * ``x[..., k]`` : kth central moment

x_cocmom | x : ndarray
    Central comoments array.  The expected structure is:

    * ``x[..., 0, 0]`` : weight
    * ``x[..., 1, 0]`` : mean of `a`
    * ``x[....,0, 1]`` : mean of `b`
    * ``x[..., i, j]``: :math:`\langle (\delta a)^i (\delta b)^j \rangle`,
      where `a` and `b` are the variables being considered.

x_rmom | x : ndarray
    Raw moments array.  The expected structure is:

    * ``x[..., 0]`` : weight
    * ``x[..., k]`` : kth moment :math:`\langle a^k \rangle`

x_cormom | x : ndarray
    Raw comoments array.  The expected structure is:

    * ``x[..., 0, 0]``: weight
    * ``x[..., i, j]`` : :math:`\langle a^i b^j \rangle`,
      where `a` and `b` are the variables being considered.


out_cmom | out : ndarray
    Central moments array.  The expected structure is:

    * ``out[..., 0]`` : weight
    * ``out[..., 1]`` : mean
    * ``out[..., k]`` : kth central moment

out_cocmom | out : ndarray
    Central comoments array.  The expected structure is:

    * ``out[..., 0, 0]`` : weight
    * ``out[..., 1, 0]`` : mean of `a`
    * ``out[....,0, 1]`` : mean of `b`
    * ``out[..., i, j]`` : :math:`\langle (\delta a)^i (\delta b)^j \rangle`,
      where `a` and `b` are the variables being considered.

out_rmom | out : ndarray
    Raw moments array.  The expected structure is:

    * ``out[..., 0]`` : weight
    * ``out[..., k]`` : kth moment :math:`\langle a^k \rangle`

out_cormom | out : ndarray
    Raw comoments array.  The expected structure is:

    * ``out[..., 0, 0]`` : weight
    * ``out[..., i, j]`` : :math:`\langle a^i b^j \rangle`,
      where `a` and `b` are the variables being considered.

axis_mom | axis : int, default=-1
    Axis location of moments in ``x``.
axis_comom | axis : tuple of int, default=``(-2,-1)``
    Axis locations of moments in comoments array ``x``
order : str, optional
    Optional ordering ('c', 'f', etc) to apply to output.
dtype : str, optional
    Optional :mod:`numpy` data type to apply to output.
out : ndarray, optional
    Optional numpy output array.  Should have same shape as ``x``.

"""

docfiller_decorate = DocFiller.from_docstring(_shared_docs, combine_keys="parameters")()


def _convert_moments(
    data: ArrayLike,
    axis: int | Sequence[int],
    target_axis: int | Sequence[int],
    func: Callable[..., Any],
    dtype: DTypeLike | None = None,
    order: ArrayOrder | None = None,
    out: MyNDArray | None = None,
) -> MyNDArray:
    def _verify_tuple(x: int | Sequence[int]) -> tuple[int, ...]:
        if isinstance(x, int):
            return (x,)
        return tuple(x)

    axis = _verify_tuple(axis)
    target_axis = _verify_tuple(target_axis)

    if len(axis) != len(target_axis):
        raise ValueError

    data = np.asarray(data, dtype=dtype, order=order)
    if out is None:
        out = np.zeros_like(data)
    elif out.shape != data.shape:
        raise ValueError
    else:
        out[...] = 0.0

    if axis != target_axis:
        data_r = np.moveaxis(data, axis, target_axis)
        out_r = np.moveaxis(out, axis, target_axis)
    else:
        data_r = data
        out_r = out

    # make sure out_r is in correct order
    out_r = np.asarray(out_r, order="C")

    shape = data_r.shape
    mom_shape = shape[-len(axis) :]
    val_shape = shape[: -len(axis)]

    reshape = (1, *mom_shape) if val_shape == () else (prod(val_shape), *mom_shape)

    data_r = data_r.reshape(reshape)
    out_r = out_r.reshape(reshape)

    func(data_r, out_r)

    return out_r.reshape(shape)


@docfiller_decorate
def to_raw_moments(
    x: MyNDArray,
    axis: int | None = -1,
    dtype: DTypeLike | None = None,
    order: ArrayOrder | None = None,
    out: MyNDArray | None = None,
) -> MyNDArray:
    r"""
    Convert central moments to raw moments.

    Parameters
    ----------
    {x_cmom}
    {axis_mom}
    {dtype}
    {order}
    {out}

    Returns
    -------
    {out_rmom}
    """
    from ._lib.convert import (
        central_to_raw_moments,  # pyright: ignore[reportUnknownVariableType]
    )

    if axis is None:
        axis = -1

    return _convert_moments(
        data=x,
        axis=axis,
        target_axis=-1,
        func=central_to_raw_moments,  # pyright: ignore[reportUnknownArgumentType]
        dtype=dtype,
        order=order,
        out=out,
    )


@docfiller_decorate
def to_raw_comoments(
    x: MyNDArray,
    axis: tuple[int, int] | None = (-2, -1),
    dtype: DTypeLike | None = None,
    order: ArrayOrder | None = None,
    out: MyNDArray | None = None,
) -> MyNDArray:
    r"""
    Convert central moments to raw moments.

    Parameters
    ----------
    {x_cocmom}
    {axis_comom}
    {dtype}
    {order}
    {out}

    Returns
    -------
    {out_cormom}
    """
    from ._lib.convert import (
        central_to_raw_comoments,  # pyright: ignore[reportUnknownVariableType]
    )

    if axis is None:
        axis = (-2, -1)

    return _convert_moments(
        data=x,
        axis=axis,
        target_axis=(-2, -1),
        func=central_to_raw_comoments,  # pyright: ignore[reportUnknownArgumentType]
        dtype=dtype,
        order=order,
        out=out,
    )


@docfiller_decorate
def to_central_moments(
    x: MyNDArray,
    axis: int | None = -1,
    dtype: DTypeLike | None = None,
    order: ArrayOrder | None = None,
    out: MyNDArray | None = None,
) -> MyNDArray:
    r"""
    Convert central moments to raw moments.

    Parameters
    ----------
    {x_rmom}
    {axis_mom}
    {dtype}
    {order}
    {out}

    Returns
    -------
    {out_cmom}
    """
    from ._lib.convert import (
        raw_to_central_moments,  # pyright: ignore[reportUnknownVariableType]
    )

    if axis is None:
        axis = -1

    return _convert_moments(
        data=x,
        axis=axis,
        target_axis=-1,
        func=raw_to_central_moments,  # pyright: ignore[reportUnknownArgumentType]
        dtype=dtype,
        order=order,
        out=out,
    )


@docfiller_decorate
def to_central_comoments(
    x: MyNDArray,
    axis: tuple[int, int] | None = (-2, -1),
    dtype: DTypeLike | None = None,
    order: ArrayOrder | None = None,
    out: MyNDArray | None = None,
) -> MyNDArray:
    r"""
    Convert raw comoments to central comoments.

    Parameters
    ----------
    {x_cormom}
    {axis_comom}
    {dtype}
    {order}
    {out}

    Returns
    -------
    {out_cocmom}
    """
    from ._lib.convert import (
        raw_to_central_comoments,  # pyright: ignore[reportUnknownVariableType]
    )

    if axis is None:
        axis = (-2, -1)

    return _convert_moments(
        data=x,
        axis=axis,
        target_axis=(-2, -1),
        func=raw_to_central_comoments,  # pyright: ignore[reportUnknownArgumentType]
        dtype=dtype,
        order=order,
        out=out,
    )

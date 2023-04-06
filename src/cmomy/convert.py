# type: ignore
"""
Routines to convert central (co)moments to raw (co)moments. (:mod:`cmomy.convert`)
==================================================================================
"""
from __future__ import annotations

from typing import Any, Callable, Sequence, no_type_check

import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike, DTypeLike

from ._docstrings import DocFiller
from ._typing import ArrayOrder
from .options import OPTIONS
from .utils import factory_binomial, myjit

_bfac = factory_binomial(OPTIONS["nmax"])


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

docfiller_shared = DocFiller.from_docstring(_shared_docs, combine_keys="parameters")()


@myjit
def _central_to_raw_moments(central, raw):
    nv = central.shape[0]
    order = central.shape[1] - 1

    for v in range(nv):
        ave = central[v, 1]

        raw[v, 0] = central[v, 0]
        raw[v, 1] = ave

        for n in range(2, order + 1):
            tmp = 0.0
            ave_i = 1.0
            for i in range(0, n - 1):
                tmp += central[v, n - i] * ave_i * _bfac[n, i]
                ave_i *= ave

            # last two
            # <dx> = 0 so skip i = n-1
            # i = n
            tmp += ave_i * ave
            raw[v, n] = tmp


@myjit
def _raw_to_central_moments(raw, central):
    nv = central.shape[0]
    order = central.shape[1] - 1

    for v in range(nv):
        ave = raw[v, 1]

        central[v, 0] = raw[v, 0]
        central[v, 1] = ave

        for n in range(2, order + 1):
            tmp = 0.0
            ave_i = 1.0
            for i in range(0, n - 1):
                tmp += raw[v, n - i] * ave_i * _bfac[n, i]
                ave_i *= -ave

            # last two
            # right now, ave_i = (-ave)**(n-1)
            # i = n-1
            # ave * ave_i * n
            # i = n
            # 1 * (-ave) * ave_i
            tmp += ave * ave_i * (n - 1)
            central[v, n] = tmp


# comoments
@myjit
def _central_to_raw_comoments(central, raw):
    nv = central.shape[0]
    order0 = central.shape[1] - 1
    order1 = central.shape[2] - 1

    for v in range(nv):
        ave0 = central[v, 1, 0]
        ave1 = central[v, 0, 1]

        for n in range(0, order0 + 1):
            for m in range(0, order1 + 1):
                nm = n + m
                if nm <= 1:
                    raw[v, n, m] = central[v, n, m]
                else:
                    tmp = 0.0
                    ave_i = 1.0
                    for i in range(n + 1):
                        ave_j = 1.0
                        for j in range(m + 1):
                            nm_ij = nm - (i + j)
                            if nm_ij == 0:
                                # both zero order
                                tmp += ave_i * ave_j
                            elif nm_ij == 1:
                                # <dx**0 * dy**1> = 0
                                pass
                            else:
                                tmp += (
                                    central[v, n - i, m - j]
                                    * ave_i
                                    * ave_j
                                    * _bfac[n, i]
                                    * _bfac[m, j]
                                )
                            ave_j *= ave1
                        ave_i *= ave0
                    raw[v, n, m] = tmp


@myjit
def _raw_to_central_comoments(raw, central):
    nv = central.shape[0]
    order0 = central.shape[1] - 1
    order1 = central.shape[2] - 1

    for v in range(nv):
        ave0 = raw[v, 1, 0]
        ave1 = raw[v, 0, 1]

        for n in range(0, order0 + 1):
            for m in range(0, order1 + 1):
                nm = n + m
                if nm <= 1:
                    central[v, n, m] = raw[v, n, m]
                else:
                    tmp = 0.0
                    ave_i = 1.0
                    for i in range(n + 1):
                        ave_j = 1.0
                        for j in range(m + 1):
                            nm_ij = nm - (i + j)
                            if nm_ij == 0:
                                # both zero order
                                tmp += ave_i * ave_j
                            else:
                                tmp += (
                                    raw[v, n - i, m - j]
                                    * ave_i
                                    * ave_j
                                    * _bfac[n, i]
                                    * _bfac[m, j]
                                )
                            ave_j *= -ave1
                        ave_i *= -ave0
                    central[v, n, m] = tmp


@no_type_check
def _convert_moments(
    data: ArrayLike,
    axis: int | Sequence[int],
    target_axis: int | Sequence[int],
    func: Callable[..., Any],
    dtype: DTypeLike | None = None,
    order: ArrayOrder | None = None,
    out: np.ndarray | None = None,
) -> ndarray:
    if isinstance(axis, int):
        axis = (axis,)
    if isinstance(target_axis, int):
        target_axis = (target_axis,)

    axis = tuple(axis)
    target_axis = tuple(target_axis)

    assert len(axis) == len(target_axis)

    data = np.asarray(data, dtype=dtype, order=order)
    if out is None:
        out = np.zeros_like(data)
    else:
        assert out.shape == data.shape
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

    if val_shape == ():
        reshape = (1,) + mom_shape
    else:
        reshape = (np.prod(val_shape),) + mom_shape

    data_r = data_r.reshape(reshape)
    out_r = out_r.reshape(reshape)

    func(data_r, out_r)
    return out_r.reshape(shape)


@docfiller_shared
def to_raw_moments(
    x: ndarray,
    axis: int = -1,
    dtype: DTypeLike | None = None,
    order: ArrayOrder | None = None,
    out: np.ndarray | None = None,
) -> ndarray:
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
    if axis is None:
        axis = -1

    return _convert_moments(
        data=x,
        axis=axis,
        target_axis=-1,
        func=_central_to_raw_moments,
        dtype=dtype,
        order=order,
        out=out,
    )


@docfiller_shared
def to_raw_comoments(
    x: ndarray,
    axis: tuple[int, int] = (-2, -1),
    dtype: DTypeLike | None = None,
    order: ArrayOrder | None = None,
    out: np.ndarray | None = None,
) -> ndarray:
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

    if axis is None:
        axis = (-2, -1)

    return _convert_moments(
        data=x,
        axis=axis,
        target_axis=(-2, -1),
        func=_central_to_raw_comoments,
        dtype=dtype,
        order=order,
        out=out,
    )


@docfiller_shared
def to_central_moments(
    x: ndarray,
    axis: int = -1,
    dtype: DTypeLike | None = None,
    order: ArrayOrder | None = None,
    out: np.ndarray | None = None,
) -> ndarray:
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

    if axis is None:
        axis = -1

    return _convert_moments(
        data=x,
        axis=axis,
        target_axis=-1,
        func=_raw_to_central_moments,
        dtype=dtype,
        order=order,
        out=out,
    )


@docfiller_shared
def to_central_comoments(
    x: ndarray,
    axis: tuple[int, int] = (-2, -1),
    dtype: DTypeLike | None = None,
    order: ArrayOrder | None = None,
    out: np.ndarray | None = None,
) -> ndarray:
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

    if axis is None:
        axis = (-2, -1)

    return _convert_moments(
        data=x,
        axis=axis,
        target_axis=(-2, -1),
        func=_raw_to_central_comoments,
        dtype=dtype,
        order=order,
        out=out,
    )

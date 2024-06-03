# mypy: disable-error-code="no-untyped-def, no-untyped-call"

import numpy as np


# Dumb calculations
def get_cmom(w, x, moments, axis=0, last=True):
    """Calculate central moments"""
    if w is None:
        w = np.array(1.0, dtype=x.dtype)

    if w.ndim == 1 and w.ndim != x.ndim and len(w) == x.shape[axis]:
        shape = [1] * x.ndim
        shape[axis] = -1
        w = w.reshape(*shape)

    if w.shape != x.shape:
        w = np.broadcast_to(w, x.shape)

    wsum_keep = w.sum(axis, keepdims=True)
    wsum_keep_inv = 1.0 / wsum_keep

    wsum = w.sum(axis)
    wsum_inv = 1.0 / wsum

    # get moments
    xave = (w * x).sum(axis, keepdims=True) * wsum_keep_inv
    dx = x - xave

    xmean = (w * x).sum(axis) * wsum_inv
    weight = wsum
    data = [weight, xmean]

    for n in range(2, moments + 1):
        y = (w * dx**n).sum(axis) * wsum_inv
        data.append(y)

    data_array = np.array(data, dtype=x.dtype)
    if last:
        data_array = np.moveaxis(data_array, 0, -1)
    return data_array


def get_comom(w, x, y, moments, axis=0, broadcast=True):
    """Calculate central co-moments."""
    if w is None:
        w = np.array(1.0, dtype=x.dtype)

    if w.ndim == 1 and w.ndim != x.ndim and len(w) == x.shape[axis]:
        shape_list = [1] * x.ndim
        shape_list[axis] = -1
        w = w.reshape(*shape_list)

    if w.shape != x.shape:
        w = np.broadcast_to(w, x.shape)

    if y.ndim != x.ndim and y.ndim == 1 and len(y) == x.shape[axis]:
        shape_list = [1] * x.ndim
        shape_list[axis] = -1
        y = y.reshape(*shape_list)

    if broadcast and y.shape != x.shape:
        y = np.broadcast_to(y, x.shape)

    assert w.shape == x.shape
    assert y.shape == x.shape

    shape_list = list(x.shape)
    shape_list.pop(axis)
    shape = tuple(shape_list) + tuple(x + 1 for x in moments)

    out = np.zeros(shape)
    wsum = w.sum(axis)
    wsum_inv = 1.0 / wsum

    wsum_keep = w.sum(axis, keepdims=True)
    wsum_keep_inv = 1.0 / wsum_keep

    xave = (w * x).sum(axis, keepdims=True) * wsum_keep_inv
    dx = x - xave

    yave = (w * y).sum(axis, keepdims=True) * wsum_keep_inv
    dy = y - yave

    for i in range(moments[0] + 1):
        for j in range(moments[1] + 1):
            if i == 0 and j == 0:
                val = wsum

            elif i + j == 1:
                val = (w * x**i * y**j).sum(axis) * wsum_inv
            else:
                val = (w * dx**i * dy**j).sum(axis) * wsum_inv

            out[..., i, j] = val
    return out

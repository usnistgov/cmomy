# mypy: disable-error-code="no-untyped-def, no-untyped-call, call-overload, assignment, arg-type"
# pyright: reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

import cmomy
from cmomy.core.validate import (
    is_dataarray,
    is_xarray,
)


# * General
def remove_dim_from_kwargs(kwargs):
    kwargs = kwargs.copy()
    kwargs.pop("dim")
    return kwargs


def moments_to_comoments_kwargs(kwargs):
    kwargs = kwargs.copy()
    for k in ("dim", "mom_ndim"):
        kwargs.pop(k)
    kwargs["mom"] = (1, -1)
    return kwargs


def get_by(n):
    n0 = n // 2
    return [0] * n0 + [1] * (n - n0)


def _get_axis_size(data, **kwargs):
    if is_xarray(data):
        return data.sizes[kwargs["dim"]]
    return data.shape[kwargs["axis"]]


def do_reduce_data_grouped(data, **kwargs):
    by = get_by(_get_axis_size(data, **kwargs))
    return cmomy.reduce_data_grouped(data, by=by, **kwargs)


def do_reduce_data_indexed(data, **kwargs):
    by = get_by(_get_axis_size(data, **kwargs))
    _, index, start, end = cmomy.reduction.factor_by_to_index(by)

    coords_policy = kwargs.pop("coords_policy", "first")
    if is_dataarray(data) and coords_policy in {"first", "last"}:
        coords_policy = None

    return cmomy.reduction.reduce_data_indexed(
        data,
        index=index,
        group_start=start,
        group_end=end,
        coords_policy=coords_policy,
        **kwargs,
    )


def do_bootstrap_data(data, nrep, method, **kwargs):
    kwargs = kwargs.copy()
    kwargs.pop("move_axis_to_end", None)

    args = [cmomy.resample_data(data, nrep=nrep, rng=0, **kwargs)]
    if method in {"basic", "bca"}:
        args.append(cmomy.reduce_data(data, **kwargs))
    if method == "bca":
        args.append(cmomy.resample.jackknife_data(data, **kwargs))

    kwargs = {"dim": "rep"} if is_xarray(data) else {"axis": kwargs["axis"]}
    return cmomy.bootstrap_confidence_interval(*args, method=method, **kwargs)


def do_wrap(*args, **kwargs):
    return cmomy.wrap(*args, **kwargs).obj


def do_wrap_reduce_vals(*args, **kwargs):
    return cmomy.wrap_reduce_vals(*args, **kwargs).obj


def do_wrap_resample_vals(*args, nrep=20, rng=0, **kwargs):
    return cmomy.wrap_resample_vals(*args, nrep=nrep, rng=rng, **kwargs).obj


def do_moveaxis(data, **kwargs):  # noqa: ARG001
    return cmomy.moveaxis(data, 0, 0)


def do_wrap_raw(data, **kwargs):
    return cmomy.wrap_raw(data, **kwargs).obj


def _get_wrapped(data, **kwargs):
    kws = kwargs.copy()
    params = {
        "mom_ndim": kws.pop("mom_ndim", 1),
        "mom_dims": kws.pop("mom_dims", None),
        "dtype": kws.get("dtype", None),
    }
    params = {k: v for k, v in params.items() if v is not None}
    return cmomy.wrap(data, **params), kws


def do_wrap_method(method, as_array=True):
    def func(data, **kwargs):
        c, kws = _get_wrapped(data, **kwargs)
        out = getattr(c, method)(**kws)
        if as_array and hasattr(out, "obj"):
            return out.obj
        return out

    return func

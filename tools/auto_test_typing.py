# mypy: disable-error-code="no-untyped-def, no-untyped-call, call-overload, var-annotated, arg-type, operator"
# pyright: reportCallIssue=false, reportArgumentType=false
"""Create test_typing_auto.py file"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

FORMAT = "[%(name)s - %(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger("make_parallel_files")


TEMPLATE_GENERAL_TEST = """\
    check(
        assert_type(
            {func_name}({data}{args}),
            {type_},
        ),
        {klass},
        {dtype},
        {second_klass},
    )\
"""


TEMPLATE_METHOD_TEST = """\
    check(
        assert_type(
            {data}.{func_name}({args}),
            {type_},
        ),
        {klass},
        {dtype},
        {second_klass},
    )\
"""


@dataclass
class GeneralTest:
    """General test creation"""

    func_name: str
    data: str
    args: str
    type_: str
    klass: str
    dtype: str
    second_klass: str = "None"
    template: str = field(default=TEMPLATE_GENERAL_TEST, repr=False)

    def __str__(self) -> str:
        return self.template.format_map(self.__dict__)

    @classmethod
    def from_params(
        cls,
        func_name: str,
        data: str,
        out_prefix: str | None,
        base_args: str | None,
        dtype_arg: str | None,
        out_dtype: str | None,
        type_: str,
        dtype: str,
        klass: str = "np.ndarray",
        second_klass: str = "None",
        method: bool = False,
        astype: bool = False,
        newlike: bool = False,
        template: str | None = None,
    ) -> GeneralTest:
        """Create object from params."""
        if template is None:
            template = TEMPLATE_METHOD_TEST if method else TEMPLATE_GENERAL_TEST

        args = base_args
        if args is not None:
            args = args.format_map(
                {
                    "axis_dim": "axis=0"
                    if klass in {"np.ndarray", "CentralMomentsArray"}
                    else 'dim="dim_0"'
                }
            )
        else:
            args = ""

        if astype:
            if dtype_arg is None:
                dtype = "None"
            args = f"{dtype}, {args}" if args else dtype
        elif dtype_arg is not None:
            n = f"dtype={dtype_arg}"
            args = f"{args}, {n}" if args else n

        if out_dtype is not None:
            if newlike:
                n = f"{out_prefix}{out_dtype}"
                args = f"{n}, {args}" if args else n
            else:
                n = f"out={out_prefix}{out_dtype}"
                args = f"{args}, {n}" if args else n

        if (not method) and args:
            args = f", {args}"

        return cls(
            func_name=func_name,
            data=data,
            args=args,
            type_=type_,
            klass=klass,
            second_klass=second_klass,
            dtype=dtype,
            template=template,
        )


def get_list(
    funcs_list: list[tuple[Any, ...]], params_list: list[tuple[Any, ...]], **kwargs: Any
) -> list[str]:
    """Get function list"""
    s = len(set(params_list))
    if s != len(params_list):
        logger.info("odd params list %s, %s", s, len(params_list))

    s = len(set(funcs_list))
    if s != len(funcs_list):
        logger.info("odd funcs list %s %s", s, len(funcs_list))

    out: list[str] = []
    for func_name, data_prefix, out_prefix, base_args in funcs_list:
        # add in a test function
        name_norm = func_name.replace(".", "_")
        out.append(f"\n\ndef test_{name_norm}() -> None:")

        for data_suffix, *args in params_list:
            out.append(
                str(
                    GeneralTest.from_params(
                        func_name,
                        data_prefix + data_suffix,
                        out_prefix,
                        base_args,
                        *args,
                        **kwargs,
                    )
                )
            )
    return out


# fmt: off
HEADER = """\
# This file is autogenerated.  Do not edit by hand.
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, cast, Iterator, Hashable

import numpy as np
from numpy import float32, float64

import xarray as xr
import pytest

import cmomy
from cmomy import CentralMomentsArray, CentralMomentsData

MYPY_ONLY = True

if sys.version_info < (3, 11):
    from typing_extensions import assert_type
else:
    from typing import assert_type

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike

    from cmomy.core.typing_compat import TypeVar
    from cmomy.core.typing import NDArrayAny, CentralMomentsArrayAny, CentralMomentsDataAny

    T = TypeVar("T")

from numpy.typing import NDArray

# So can exclude from coverage
pytestmark = pytest.mark.typing


def check(
    actual: T,
    klass: type[Any],
    dtype: DTypeLike | None = None,
    obj_class: type[Any] | None = None,
) -> T:
    assert isinstance(actual, klass)
    if dtype is not None:
        assert actual.dtype is np.dtype(dtype)  # pyright: ignore[reportAttributeAccessIssue]

    if obj_class is None and klass is CentralMomentsArray:
        obj_class = np.ndarray

    if obj_class is not None:
        assert isinstance(actual.obj, obj_class)  # pyright: ignore[reportAttributeAccessIssue]
    return actual  # type: ignore[no-any-return]


# * Parameters
vals_float32: NDArray[np.float32] = np.zeros(10, dtype=np.float32)
vals_float64: NDArray[np.float64] = np.zeros(10, dtype=np.float64)
vals_arrayany: NDArrayAny = cast("NDArrayAny", vals_float64)
vals_arraylike: ArrayLike = cast("ArrayLike", vals_float64)
vals_any: Any = cast("Any", vals_float64)

data_float32: NDArray[np.float32] = np.zeros((10, 3), dtype=np.float32)
data_float64: NDArray[np.float64] = np.zeros((10, 3), dtype=np.float64)
data_arrayany: NDArrayAny = cast("NDArrayAny", data_float64)
data_arraylike: ArrayLike = cast("ArrayLike", data_float64)
data_any: Any = cast("Any", data_float64)

# For reduction
reduce_out_float32: NDArray[np.float32] = np.zeros(3, dtype=np.float32)
reduce_out_float64: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
reduce_out_arrayany: NDArrayAny = np.zeros(3, dtype="f8")
reduce_out_any = cast("Any", np.zeros_like(reduce_out_float64))

# For transform
transform_out_float32: NDArray[np.float32] = np.zeros((10, 3), dtype=np.float32)
transform_out_float64: NDArray[np.float64] = np.zeros((10, 3), dtype=np.float64)
transform_out_arrayany: NDArrayAny = np.zeros((10, 3), dtype="f8")
transform_out_any = cast("Any", np.zeros_like(transform_out_float64))

# For group/resample
group_out_float32: NDArray[np.float32] = np.zeros((2, 3), dtype=np.float32)
group_out_float64: NDArray[np.float64] = np.zeros((2, 3), dtype=np.float64)
group_out_arrayany: NDArrayAny = np.zeros((2, 3), dtype="f8")
group_out_any = cast("Any", np.zeros_like(group_out_float64))

vals_dataarray = xr.DataArray(vals_float64, name="x")
vals_dataset = xr.Dataset({"x": vals_dataarray})
vals_dataarray_or_set: xr.DataArray | xr.Dataset = cast("xr.DataArray | xr.Dataset", vals_dataarray)
vals_dataarray_any: Any = cast("Any", vals_dataarray)
vals_dataset_any: Any = cast("Any", vals_dataset)

data_dataarray: xr.DataArray = xr.DataArray(data_float64, name="data")
data_dataset: xr.Dataset = xr.Dataset({"data": data_dataarray})
data_dataarray_any: Any = cast("Any", data_dataarray)
data_dataset_any: Any = cast("Any", data_dataset)
data_dataarray_or_sdata: xr.DataArray | xr.Dataset = cast("xr.DataArray | xr.Dataset", data_dataarray)


central_float32 = CentralMomentsArray(data_float32)
central_float64 = CentralMomentsArray(data_float64)
central_arraylike: CentralMomentsArrayAny = CentralMomentsArray(data_arraylike)
central_arrayany: CentralMomentsArrayAny = CentralMomentsArray(data_arrayany)
central_any: Any = CentralMomentsArray(data_any)

central_dataarray = CentralMomentsData(data_dataarray)
central_dataset = CentralMomentsData(data_dataset)
central_dataarray_any: CentralMomentsDataAny = CentralMomentsData(data_dataarray_any)
central_dataset_any: CentralMomentsDataAny = CentralMomentsData(data_dataset_any)
# ca_or_cs = cast("CentralMomentsData[xr.DataArray] | CentralMomentsData[xr.DataArray]", CentralMomentsData(data_dataarray_or_sdata))  # noqa: ERA001

freq = cmomy.random_freq(ndat=10, nrep=2)
sampler = cmomy.resample.IndexSampler(freq=freq)
by = [0] * 5 + [1] * 5
_, index, group_start, group_end = cmomy.grouped.factor_by_to_index(by)
"""


out: list[str] = []


# func_name, data_prefix, out_prefix, "base_args"
# * GenArray to GenArray
# ** Just arrays
funcs_genarraylike_to_genarray_dtype_out = [
    ("cmomy.reduce_data", "data_", "reduce_out_", "{axis_dim}, mom_ndim=1"),
    ("cmomy.reduce_vals", "vals_", "reduce_out_", "{axis_dim}, mom=2"),
    ("cmomy.reduce_data_grouped", "data_", "group_out_", "{axis_dim}, mom_ndim=1, by=by"),
    ("cmomy.reduce_data_indexed", "data_", "group_out_", "{axis_dim}, mom_ndim=1, index=index, group_start=group_start, group_end=group_end"),
    ("cmomy.resample_data", "data_", "group_out_", "{axis_dim}, mom_ndim=1, sampler=sampler"),
    ("cmomy.resample_vals", "vals_", "group_out_", "{axis_dim}, mom=2, sampler=sampler"),
    ("cmomy.resample.jackknife_data", "data_", "transform_out_", "{axis_dim}, mom_ndim=1"),
    ("cmomy.resample.jackknife_vals", "vals_", "transform_out_", "{axis_dim}, mom=2"),
    ("cmomy.convert.moments_type", "data_", "transform_out_", "mom_ndim=1"),
    ("cmomy.convert.cumulative", "data_", "transform_out_", "{axis_dim}, mom_ndim=1"),
    ("cmomy.utils.vals_to_data", "vals_", "transform_out_", "mom=2"),
    ("cmomy.rolling.rolling_vals", "vals_", "transform_out_", "{axis_dim}, mom=2, window=3"),
    ("cmomy.rolling.rolling_data", "data_", "transform_out_", "{axis_dim}, mom_ndim=1, window=3"),
    ("cmomy.rolling.rolling_exp_vals", "vals_", "transform_out_", "{axis_dim}, mom=2, alpha=0.2"),
    ("cmomy.rolling.rolling_exp_data", "data_", "transform_out_", "{axis_dim}, mom_ndim=1, alpha=0.2"),
]
funcs_genarraylike_to_genarray_dtype = [
    ("cmomy.convert.moments_to_comoments", "data_", None, "mom=(1, -1)"),
    ("cmomy.convert.comoments_to_moments", "data_", None, "")
]
funcs_genarray_to_genarray = [
    ("cmomy.utils.select_moment", "data_", None, '"weight", mom_ndim=1'),
    ("cmomy.utils.assign_moment", "data_", None, "weight=1, mom_ndim=1"),
    ("cmomy.bootstrap_confidence_interval", "data_", None, '{axis_dim}, method="percentile"'),
]

# data_suffix, dtype_arg, out_dtype, type, dtype, klass, second_class
params_genarray_to_genarray = [
    ("float32", None, None, "NDArray[float32]", "float32"),
    ("float64", None, None, "NDArray[float64]", "float64"),
    ("arrayany", None, None, "NDArray[Any]", "float64"),
    ("any", None, None, "Any", "float64"),
    ("dataarray", None, None, "xr.DataArray", "float64", "xr.DataArray"),
    ("dataset", None, None, "xr.Dataset", None, "xr.Dataset"),
    ("dataarray_any", None, None, "Any", "float64", "xr.DataArray"),
    ("dataset_any", None, None, "Any", None, "xr.Dataset"),
]

params_genarraylike_to_genarray_dtype = [
    *params_genarray_to_genarray,
    # array only
    ("arraylike", None, None, "NDArray[Any]", "float64"),
    # dtype
    ("float32", "float64", None, "NDArray[float64]", "float64"),
    ("float64", "float32", None, "NDArray[float32]", "float32"),
    ("arrayany", "float32", None, "NDArray[float32]", "float32"),
    ("arraylike", "float32", None, "NDArray[float32]", "float32"),
    ("any", "float32", None, "Any", "float32"),
    ("float32", '"f8"', None, "NDArray[Any]", "float64"),
    ("dataarray", "float32", None, "xr.DataArray", "float32", "xr.DataArray"),
    ("dataset", "float32", None, "xr.Dataset", None, "xr.Dataset"),
    ("dataarray_any", "float32", None, "Any", "float32", "xr.DataArray"),
]

params_genarraylike_to_genarray_dtype_out = [
    *params_genarraylike_to_genarray_dtype,
    # out
    ("float32", None, "float64", "NDArray[float64]", "float64"),
    ("arraylike", None, "float64", "NDArray[float64]", "float64"),
    ("any", None, "float32", "Any", "float32"),
    ("float32", None, "arrayany", "NDArray[Any]", "float64"),
    # ("float32", None, "any", "NDArray[Any]", "float64"),  # noqa: ERA001
    ("dataarray", None, "float32", "xr.DataArray", "float32", "xr.DataArray"),
    ("dataarray_any", None, "float32", "Any", "float32", "xr.DataArray"),
    # with out and dtype
    ("float32", "float32", "float64", "NDArray[float64]", "float64"),
    ("arraylike", "float32", "float64", "NDArray[float64]", "float64"),
    ("any", "float64", "float32", "Any", "float32"),
    ("float32", "float64", "arrayany", "NDArray[Any]", "float64"),
    ("dataarray", '"f8"', "float32", "xr.DataArray", "float32", "xr.DataArray"),

]

out = []
for funcs, params in [
        (funcs_genarray_to_genarray, params_genarray_to_genarray),
        (funcs_genarraylike_to_genarray_dtype, params_genarraylike_to_genarray_dtype),
        (funcs_genarraylike_to_genarray_dtype_out, params_genarraylike_to_genarray_dtype_out),
]:
    out.extend(get_list(funcs, params))  # type: ignore[arg-type]


# * Move axis
params_moveaxis = list(filter(lambda x: "dataset" not in x[0] and "arraylike" not in x[0] and x[1] is None, params_genarraylike_to_genarray_dtype))  # type: ignore[var-annotated,arg-type, operator]
funcs_moveaxis = [
    ("cmomy.moveaxis", "data_", None, "0, 1"),
]

out.extend(get_list(funcs_moveaxis, params_moveaxis))


# * GenArray to wrapped
params_genarraylike_to_wrapped_dtype = [
    ("float32", None, None, "CentralMomentsArray[float32]", "float32", "CentralMomentsArray"),
    ("float64", None, None, "CentralMomentsArray[float64]", "float64", "CentralMomentsArray"),
    ("arrayany", None, None, "CentralMomentsArray[Any]", "float64", "CentralMomentsArray"),
    ("arraylike", None, None, "CentralMomentsArray[Any]", "float64", "CentralMomentsArray"),
    ("any", None, None, "Any", "float64", "CentralMomentsArray"),
    ("dataarray", None, None, "CentralMomentsData[xr.DataArray]", "float64", "CentralMomentsData", "xr.DataArray"),
    ("dataset", None, None, "CentralMomentsData[xr.Dataset]", None, "CentralMomentsData", "xr.Dataset"),
    ("dataarray_any", None, None, "Any", "float64", "CentralMomentsData", "xr.DataArray"),
    ("dataset_any", None, None, "Any", None, "CentralMomentsData", "xr.Dataset"),
    ("float32", "float64", None, "CentralMomentsArray[float64]", "float64", "CentralMomentsArray"),
    ("arraylike", "float32", None, "CentralMomentsArray[float32]", "float32", "CentralMomentsArray"),
    ("any", "float32", None, "Any", "float32", "CentralMomentsArray"),
    ("dataarray", "float32", None, "CentralMomentsData[xr.DataArray]", "float32", "CentralMomentsData", "xr.DataArray"),
    ("dataset", "float32", None, "CentralMomentsData[xr.Dataset]", None, "CentralMomentsData", "xr.Dataset"),
    ("dataarray_any", "float32", None, "Any", "float32", "CentralMomentsData", "xr.DataArray"),
]
funcs_genarraylike_to_wrapped_dtype = [
    ("cmomy.wrap", "data_", None, ""),
]

out.extend(get_list(funcs_genarraylike_to_wrapped_dtype, params_genarraylike_to_wrapped_dtype))


params_genarraylike_to_wrapped_dtype_out = [
    *params_genarraylike_to_wrapped_dtype,
    ("float32", None, "float64", "CentralMomentsArray[float64]", "float64", "CentralMomentsArray"),
    ("float64", None, "float32", "CentralMomentsArray[float32]", "float32", "CentralMomentsArray"),
    ("arraylike", None, "float64", "CentralMomentsArray[float64]", "float64", "CentralMomentsArray"),
    ("float32", None, "arrayany", "CentralMomentsArray[Any]", "float64", "CentralMomentsArray"),
    ("any", None, "float32", "Any", "float32", "CentralMomentsArray"),
    ("dataarray", None, "float32", "CentralMomentsData[xr.DataArray]", "float32", "CentralMomentsData", "xr.DataArray"),
    ("dataarray", None, "arrayany", "CentralMomentsData[xr.DataArray]", "float64", "CentralMomentsData", "xr.DataArray"),
    ("dataarray_any", None, "float32", "Any", "float32", "CentralMomentsData", "xr.DataArray"),
]

funcs_genarraylike_to_wrapped_dtype_out = [
    ("cmomy.wrap_reduce_vals", "vals_", "reduce_out_", "{axis_dim}, mom=2"),
    ("cmomy.wrap_resample_vals", "vals_", "group_out_", "{axis_dim}, mom=2, sampler=sampler"),
    ("cmomy.wrap_raw", "data_", "transform_out_", ""),
]

out.extend(get_list(funcs_genarraylike_to_wrapped_dtype_out, params_genarraylike_to_wrapped_dtype_out))


# * Arraylike to CentralMomentsArray
params_arraylike_to_class = [
    ("float32", None, None, "CentralMomentsArray[float32]", "float32", "CentralMomentsArray"),
    ("float64", None, None, "CentralMomentsArray[float64]", "float64", "CentralMomentsArray"),
    ("arrayany", None, None, "CentralMomentsArray[Any]", "float64", "CentralMomentsArray"),
    ("arraylike", None, None, "CentralMomentsArray[Any]", "float64", "CentralMomentsArray"),
    ("any", None, None, "CentralMomentsArray[Any]", "float64", "CentralMomentsArray"),
    ("float32", "float64", None, "CentralMomentsArray[float64]", "float64", "CentralMomentsArray"),
    ("arraylike", "float32", None, "CentralMomentsArray[float32]", "float32", "CentralMomentsArray"),
    ("any", "float32", None, "CentralMomentsArray[float32]", "float32", "CentralMomentsArray"),
]
funcs_arraylike_to_class = [
    ("cmomy.CentralMomentsArray", "data_", None, ""),
]
out.extend(get_list(funcs_arraylike_to_class, params_arraylike_to_class))


# * XArray to CentralMomentsData
params_xarray_to_class = [
    ("dataarray", None, None, "CentralMomentsData[xr.DataArray]", "float64", "CentralMomentsData", "xr.DataArray"),
    ("dataset", None, None, "CentralMomentsData[xr.Dataset]", None, "CentralMomentsData", "xr.Dataset"),
    ("dataarray_any", None, None, "CentralMomentsData[Any]", "float64", "CentralMomentsData", "xr.DataArray"),
    ("dataset_any", None, None, "CentralMomentsData[Any]", None, "CentralMomentsData", "xr.Dataset"),
]
funcs_xarray_to_class = [
    ("cmomy.CentralMomentsData", "data_", None, ""),
]
out.extend(get_list(funcs_xarray_to_class, params_xarray_to_class))


# * Class zeros
params_class_zeros = [
    ("CentralMomentsArray", None, None, "CentralMomentsArray[np.float64]", "float64", "CentralMomentsArray"),
    ("CentralMomentsArray", "float32", None, "CentralMomentsArray[np.float32]", "float32", "CentralMomentsArray"),
    ("CentralMomentsArray", '"f4"', None, "CentralMomentsArray[Any]", "float32", "CentralMomentsArray"),
    ("CentralMomentsData", None, None, "CentralMomentsData[xr.DataArray]", "float64", "CentralMomentsData", "xr.DataArray"),
    ("CentralMomentsData", '"f4"', None, "CentralMomentsData[xr.DataArray]", "float32", "CentralMomentsData", "xr.DataArray"),
]
funcs_class_zeros = [
    ("zeros", "", None, "mom=2")
]
out.extend(get_list(funcs_class_zeros, params_class_zeros, method=True))


# * class to class
params_class_to_class = [
    ("float32", None, None, "CentralMomentsArray[float32]", "float32", "CentralMomentsArray"),
    ("float64", None, None, "CentralMomentsArray[float64]", "float64", "CentralMomentsArray"),
    ("arrayany", None, None, "CentralMomentsArray[Any]", "float64", "CentralMomentsArray"),
    ("arraylike", None, None, "CentralMomentsArray[Any]", "float64", "CentralMomentsArray"),
    ("any", None, None, "Any", "float64", "CentralMomentsArray"),
    ("dataarray", None, None, "CentralMomentsData[xr.DataArray]", "float64", "CentralMomentsData", "xr.DataArray"),
    ("dataset", None, None, "CentralMomentsData[xr.Dataset]", None, "CentralMomentsData", "xr.Dataset"),
    ("dataarray_any", None, None, "CentralMomentsData[Any]", "float64", "CentralMomentsData", "xr.DataArray"),
    ("dataset_any", None, None, "CentralMomentsData[Any]", None, "CentralMomentsData", "xr.Dataset"),
    ("float32", "float64", None, "CentralMomentsArray[float64]", "float64", "CentralMomentsArray"),
    ("arraylike", "float32", None, "CentralMomentsArray[float32]", "float32", "CentralMomentsArray"),
    ("any", "float32", None, "Any", "float32", "CentralMomentsArray"),
    ("dataarray", "float32", None, "CentralMomentsData[xr.DataArray]", "float32", "CentralMomentsData", "xr.DataArray"),
    ("dataset", "float32", None, "CentralMomentsData[xr.Dataset]", None, "CentralMomentsData", "xr.Dataset"),
    ("dataarray_any", "float32", None, "CentralMomentsData[Any]", "float32", "CentralMomentsData", "xr.DataArray"),
]
funcs_class_to_class = [
    ("cmomy.zeros_like", "central_", None, ""),
]
out.extend(get_list(funcs_class_to_class, params_class_to_class))


# * Class methods
# ** to class
# *** astype
params_class_astype = [
    ("float32", None, None, "CentralMomentsArray[float64]", "float64", "CentralMomentsArray"),
    ("float64", None, None, "CentralMomentsArray[float64]", "float64", "CentralMomentsArray"),
    ("float64", "float32", None, "CentralMomentsArray[float32]", "float32", "CentralMomentsArray"),
    ("arraylike", None, None, "CentralMomentsArray[float64]", "float64", "CentralMomentsArray"),
    ("arraylike", "float32", None, "CentralMomentsArray[float32]", "float32", "CentralMomentsArray"),
    ("any", "float32", None, "Any", "float32", "CentralMomentsArray"),
    ("dataarray", "float32", None, "CentralMomentsData[xr.DataArray]", "float32", "CentralMomentsData", "xr.DataArray"),
    ("dataset", "float32", None, "CentralMomentsData[xr.Dataset]", None, "CentralMomentsData", "xr.Dataset"),
    ("dataarray_any", "float32", None, "CentralMomentsData[Any]", "float32", "CentralMomentsData", "xr.DataArray"),
]
funcs_class_astype = [
    ("astype", "central_", None, "")
]
out.extend(get_list(funcs_class_astype, params_class_astype, method=True, astype=True))

# *** methods
params_class_method = [
    ("float64", None, None, "CentralMomentsArray[float64]", "float64", "CentralMomentsArray"),
    ("arraylike", None, None, "CentralMomentsArray[Any]", "float64", "CentralMomentsArray"),
    ("any", None, None, "Any", "float64", "CentralMomentsArray"),
    ("dataarray", None, None, "CentralMomentsData[xr.DataArray]", "float64", "CentralMomentsData", "xr.DataArray"),
    ("dataset", None, None, "CentralMomentsData[xr.Dataset]", None, "CentralMomentsData", "xr.Dataset"),
    ("dataarray_any", None, None, "CentralMomentsData[Any]", "float64", "CentralMomentsData", "xr.DataArray"),
    ("dataset_any", None, None, "CentralMomentsData[Any]", None, "CentralMomentsData", "xr.Dataset"),
]
params_class_method_dtype = [
    *params_class_method,
    ("float32", "float64", None, "CentralMomentsArray[float64]", "float64", "CentralMomentsArray"),
    ("float64", '"f4"', None, "CentralMomentsArray[Any]", "float32", "CentralMomentsArray"),
    ("arraylike", "float32", None, "CentralMomentsArray[float32]", "float32", "CentralMomentsArray"),
    ("dataarray", "float32", None, "CentralMomentsData[xr.DataArray]", "float32", "CentralMomentsData", "xr.DataArray"),
]
params_class_method_dtype_out = [
    *params_class_method_dtype,
    ("float32", None, "float64", "CentralMomentsArray[float64]", "float64", "CentralMomentsArray"),
    ("float32", None, "arrayany", "CentralMomentsArray[Any]", "float64", "CentralMomentsArray"),
    ("arraylike", "float64", "float32", "CentralMomentsArray[float32]", "float32", "CentralMomentsArray"),
    ("dataarray", None, "float32", "CentralMomentsData[xr.DataArray]", "float32", "CentralMomentsData", "xr.DataArray"),

]
funcs_class_method = [
    ("assign_moment", "central_", None, "weight=1"),
]
funcs_class_method_dtype = [
    ("moments_to_comoments", "central_", None, "mom=(1, -1)"),
]
funcs_class_method_dtype_out = [
    ("resample_and_reduce", "central_", "group_out_", "{axis_dim}, sampler=sampler"),
    ("jackknife_and_reduce", "central_", "transform_out_", "{axis_dim}"),
    ("reduce", "central_", "reduce_out_", "{axis_dim}"),
]
out.extend(get_list(funcs_class_method, params_class_method, method=True))
out.extend(get_list(funcs_class_method_dtype, params_class_method_dtype, method=True))
out.extend(get_list(funcs_class_method_dtype_out, params_class_method_dtype_out, method=True))

# *** moveaxis
params_class_method_moveaxis = list(filter(lambda x: "dataset" not in x[0], params_class_method))
funcs_class_method_moveaxis = [
    ("moveaxis", "central_", None, "0, 0"),
]
out.extend(get_list(funcs_class_method_moveaxis, params_class_method_moveaxis, method=True))

# *** reshape
params_class_method_reshape = list(filter(lambda x: "data" not in x[0], params_class_method))
funcs_class_method_reshape = [
    ("reshape", "central_", None, "(2, 5)")
]
out.extend(get_list(funcs_class_method_reshape, params_class_method_reshape, method=True))


# *** new_like
params_class_newlike = [
    *params_class_method_dtype,
    ("float32", None, "float64", "CentralMomentsArray[float64]", "float64", "CentralMomentsArray"),
    ("float32", None, "arrayany", "CentralMomentsArray[Any]", "float64", "CentralMomentsArray"),
    # This is different because out here is data input
    ("arraylike", "float64", "float32", "CentralMomentsArray[float64]", "float64", "CentralMomentsArray"),
    ("dataarray", None, "float32", "CentralMomentsData[xr.DataArray]", "float32", "CentralMomentsData", "xr.DataArray"),
]
funcs_class_newlike = [
    ("new_like", "central_", "transform_out_", ""),
]
out.extend(get_list(funcs_class_newlike, params_class_newlike, method=True, newlike=True))


# ** To Array
params_class_methodtoarray = [
    ("float64", None, None, "NDArray[float64]", "float64"),
    ("arraylike", None, None, "NDArray[Any]", "float64"),
    ("any", None, None, "Any", "float64"),
    ("dataarray", None, None, "xr.DataArray", "float64", "xr.DataArray"),
    ("dataset", None, None, "xr.Dataset", None, "xr.Dataset"),
    ("dataarray_any", None, None, "Any", "float64", "xr.DataArray"),
    ("dataset_any", None, None, "Any", None, "xr.Dataset"),
]
params_class_methodtoarray_dtype = [
    *params_class_methodtoarray,
    ("float32", "float64", None, "NDArray[float64]", "float64"),
    ("float64", '"f4"', None, "NDArray[Any]", "float32"),
    ("arraylike", "float32", None, "NDArray[float32]", "float32"),
    ("dataarray", "float32", None, "xr.DataArray", "float32", "xr.DataArray"),
]
params_class_methodtoarray_dtype_out = [
    *params_class_methodtoarray_dtype,
    ("float32", None, "float64", "NDArray[float64]", "float64"),
    ("float32", None, "arrayany", "NDArray[Any]", "float64"),
    ("arraylike", "float64", "float32", "NDArray[float32]", "float32"),
    ("dataarray", None, "float32", "xr.DataArray", "float32", "xr.DataArray"),
]
funcs_class_methodtoarray = [
]
funcs_class_methodtoarray_dtype = [
]
funcs_class_methodtoarray_dtype_out = [
    ("cumulative", "central_", "transform_out_", "{axis_dim}")
]
# out.extend(get_list(funcs_class_methodtoarray, params_class_methodtoarray, method=True))  # noqa: ERA001
# out.extend(get_list(funcs_class_methodtoarray_dtype, params_class_methodtoarray_dtype, method=True))  # noqa: ERA001
out.extend(get_list(funcs_class_methodtoarray_dtype_out, params_class_methodtoarray_dtype_out, method=True))


params_class_method_to_dataarray = [
    ("float64", None, None, "CentralMomentsData[xr.DataArray]", None, "CentralMomentsData", "xr.DataArray"),
    ("arraylike", None, None, "CentralMomentsData[xr.DataArray]", None, "CentralMomentsData", "xr.DataArray"),
    ("dataarray", None, None, "CentralMomentsData[xr.DataArray]", None, "CentralMomentsData", "xr.DataArray"),
    ("dataarray_any", None, None, "CentralMomentsData[xr.DataArray]", None, "CentralMomentsData", "xr.DataArray"),
    ("dataset", None, None, "CentralMomentsData[xr.DataArray]", None, "CentralMomentsData", "xr.DataArray"),
    ("dataset_any", None, None, "CentralMomentsData[xr.DataArray]", None, "CentralMomentsData", "xr.DataArray"),
]
func_class_method_to_dataarray = [
    ("to_dataarray", "central_", None, ""),
]

out.extend(get_list(func_class_method_to_dataarray, params_class_method_to_dataarray, method=True))

params_class_method_to_dataset = [
    ("dataarray", None, None, "CentralMomentsData[xr.Dataset]", None, "CentralMomentsData", "xr.Dataset"),
    ("dataarray_any", None, None, "CentralMomentsData[xr.Dataset]", None, "CentralMomentsData", "xr.Dataset"),
    ("dataset", None, None, "CentralMomentsData[xr.Dataset]", None, "CentralMomentsData", "xr.Dataset"),
    ("dataset_any", None, None, "CentralMomentsData[xr.Dataset]", None, "CentralMomentsData", "xr.Dataset"),
]
func_class_method_to_dataset = [
    ("to_dataset", "central_", None, ""),
]

out.extend(get_list(func_class_method_to_dataset, params_class_method_to_dataset, method=True))


# iterators
out.append("""

def test_iterators() -> None:
    from collections.abc import ItemsView, KeysView, ValuesView

    assert_type(iter(central_float64), Iterator[CentralMomentsArray[np.float64]])
    assert_type(iter(central_dataarray), Iterator[CentralMomentsData[xr.DataArray]])
    # TODO(wpk): problem with mypy and __iter__ overload....
    # assert_type(iter(central_dataset), Iterator[Hashable])  # noqa: ERA001
    assert_type(central_dataarray.__iter__(), Iterator[CentralMomentsData[xr.DataArray]])
    assert_type(central_dataset.__iter__(), Iterator[Hashable])
    assert_type(central_dataarray.iter(), Iterator[CentralMomentsData[xr.DataArray]])
    assert_type(central_dataset.iter(), Iterator[Hashable])
    assert_type(central_dataset.keys(), KeysView[Hashable])
    assert_type(central_dataset.values(), ValuesView[CentralMomentsData[xr.DataArray]])
    assert_type(central_dataset.items(), ItemsView[Hashable, CentralMomentsData[xr.DataArray]])
""")


# sampler
out.append("""

def _check_typing_sampler(
    idx_array: NDArrayAny,
    idx_dataarray: xr.DataArray,
    idx_dataset: xr.Dataset,
    freq_array: NDArrayAny,
    freq_dataarray: xr.DataArray,
    freq_dataset: xr.Dataset,
    data_array: NDArrayAny,
    data_dataarray: xr.DataArray,
    data_dataset: xr.Dataset,
) -> None:
    from cmomy import IndexSampler

    assert_type(IndexSampler.from_params(10, 20), IndexSampler[NDArrayAny])

    assert_type(IndexSampler(indices=idx_array), IndexSampler[NDArrayAny])
    assert_type(IndexSampler(indices=idx_dataarray), IndexSampler[xr.DataArray])
    assert_type(IndexSampler(indices=idx_dataset), IndexSampler[xr.Dataset])

    assert_type(IndexSampler(freq=freq_array), IndexSampler[NDArrayAny])
    assert_type(IndexSampler(freq=freq_dataarray), IndexSampler[xr.DataArray])
    assert_type(IndexSampler(freq=freq_dataset), IndexSampler[xr.Dataset])

    a = IndexSampler(indices=idx_array)
    assert_type(a.freq, NDArrayAny)
    assert_type(a.indices, NDArrayAny)

    b = IndexSampler(indices=idx_dataarray)
    assert_type(b.freq, xr.DataArray)
    assert_type(b.indices, xr.DataArray)

    c = IndexSampler(indices=idx_dataset)
    assert_type(c.freq, xr.Dataset)
    assert_type(c.indices, xr.Dataset)

    assert_type(IndexSampler.from_data(data_array, nrep=100), IndexSampler[NDArrayAny])
    assert_type(
        IndexSampler.from_data(data_dataarray, nrep=100), IndexSampler[xr.DataArray]
    )
    assert_type(
        IndexSampler.from_data(data_dataset, nrep=100), IndexSampler[xr.DataArray]
    )

    d = IndexSampler.from_data(data_dataset, nrep=100, paired=False)
    assert_type(d, IndexSampler["xr.DataArray | xr.Dataset"])

    assert_type(d.indices, "xr.DataArray | xr.Dataset")
    assert_type(d.indices, "xr.DataArray | xr.Dataset")
""")


# * convert.concat
params_array = [
    ("float32", None, None, "NDArray[float32]", "float32"),
    ("float64", None, None, "NDArray[float64]", "float64"),
    ("arrayany", None, None, "NDArray[Any]", "float64"),
    ("any", None, None, "Any", "float64"),
]
for (dtype_seq, func_name, data_prefix, out_prefix, base_args) in [
        (params_array, "cmomy.convert.concat", "data_", "reduce_out_", "{axis_dim}"),
]:
    for (data_suffix, *args) in dtype_seq:
        d = data_prefix + data_suffix
        out.append(str(
            GeneralTest.from_params(
                func_name,
                f"({d}, {d})",
                out_prefix,
                base_args,
                *args
            )
        ))
# fmt: on
# * Create test fie
out.insert(0, HEADER)
s = len(set(out))
if s != len(out):
    logger.info("duplicate in out %s %s", s, len(out))

with open("./tests/test_typing_auto.py", "w", encoding="utf-8") as f:  # noqa: PTH123
    f.write("\n".join(out))
    f.write("\n")

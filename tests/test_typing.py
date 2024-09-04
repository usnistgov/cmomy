from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Union, cast

import numpy as np
import xarray as xr

import cmomy
from cmomy import CentralMomentsArray, CentralMomentsXArray, convert, rolling
from cmomy.reduction import (
    # reduce_data,
    reduce_data_grouped,
    reduce_data_indexed,
    # reduce_vals,
)
from cmomy.resample import resample_vals

MYPY_ONLY = True

if sys.version_info < (3, 11):
    from typing_extensions import assert_type
else:
    from typing import assert_type

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike

    from cmomy.core.typing_compat import TypeVar

    T = TypeVar("T")


from numpy.typing import NDArray

from cmomy.core.typing import (  # keep this in typecheck block...
    CentralMomentsArrayAny,
    CentralMomentsDataAny,
    CentralMomentsDataArray,
    CentralMomentsDataset,
    NDArrayAny,
    NDArrayInt,
)

NDArrayFloat32 = NDArray[np.float32]
NDArrayFloat64 = NDArray[np.float64]
CentralMomentsArrayFloat32 = CentralMomentsArray[np.float32]
CentralMomentsArrayFloat64 = CentralMomentsArray[np.float64]


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
x32: NDArrayFloat32 = np.zeros(10, dtype=np.float32)
x64: NDArrayFloat64 = np.zeros(10, dtype=np.float64)
x_arrayany: NDArrayAny = cast("NDArrayAny", x64)
x_arraylike: ArrayLike = cast("ArrayLike", x64)
x_any: Any = cast("Any", x64)

data32: NDArrayFloat32 = np.zeros((10, 3), dtype=np.float32)
data64: NDArrayFloat64 = np.zeros((10, 3), dtype=np.float64)
data_arrayany: NDArrayAny = cast("NDArrayAny", data64)
data_arraylike: ArrayLike = cast("ArrayLike", data64)
data_any: Any = cast("Any", data64)

out32: NDArrayFloat32 = np.zeros(3, dtype=np.float32)
out64: NDArrayFloat64 = np.zeros(3, dtype=np.float64)
out_any: NDArrayAny = cast("Any", np.zeros_like(out64))

same_out32: NDArrayFloat32 = np.zeros((10, 3), dtype=np.float32)
same_out64: NDArrayFloat64 = np.zeros((10, 3), dtype=np.float64)
same_out_any: NDArrayAny = cast("Any", np.zeros_like(same_out64))


group_out32: NDArrayFloat32 = np.zeros((2, 3), dtype=np.float32)
group_out64: NDArrayFloat64 = np.zeros((2, 3), dtype=np.float64)
group_out_any: NDArrayAny = cast("Any", np.zeros_like(group_out64))


da = xr.DataArray(x64, name="x")
ds = xr.Dataset({"x": da})
da_or_ds: xr.DataArray | xr.Dataset = cast("xr.DataArray | xr.Dataset", da)
da_any: Any = cast("Any", da)
ds_any: Any = cast("Any", ds)

xdata: xr.DataArray = xr.DataArray(data64, name="data")
sdata: xr.Dataset = xr.Dataset({"data": xdata})
xdata_any: Any = cast("Any", xdata)
sdata_any: Any = cast("Any", sdata)
xdata_or_sdata: xr.DataArray | xr.Dataset = cast("xr.DataArray | xr.Dataset", xdata)


c32 = CentralMomentsArray(data32)
c64 = CentralMomentsArray(data64)
c_any: CentralMomentsArrayAny = CentralMomentsArray(data_any)

ca = CentralMomentsXArray(xdata)
cs = CentralMomentsXArray(sdata)
ca_any: CentralMomentsDataAny = CentralMomentsXArray(xdata_any)
cs_any: CentralMomentsDataAny = CentralMomentsXArray(sdata_any)
# ca_or_cs: CentralMomentsDataset | CentralMomentsDataArray = CentralMomentsXArray(xdata_or_sdata)  # noqa: ERA001

freq = cmomy.random_freq(ndat=10, nrep=2)
by = [0] * 5 + [1] * 5


# * reduction -----------------------------------------------------------------
def test_reduce_vals() -> None:
    check(
        assert_type(cmomy.reduce_vals(x32, mom=3, axis=0), NDArrayFloat32),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(cmomy.reduce_vals(x64, mom=3, axis=0), NDArrayFloat64),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(cmomy.reduce_vals(x_any, mom=3, axis=0), Any),
        np.ndarray,
        np.float64,
    )

    # dtype override x
    check(
        assert_type(
            cmomy.reduce_vals(x32, mom=3, axis=0, dtype=np.float64), NDArrayFloat64
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.reduce_vals(x64, mom=3, axis=0, dtype=np.float32), NDArrayFloat32
        ),
        np.ndarray,
        np.float32,
    )

    # out override x
    check(
        assert_type(cmomy.reduce_vals(x32, mom=3, axis=0, out=out64), NDArrayFloat64),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(cmomy.reduce_vals(x64, mom=3, axis=0, out=out32), NDArrayFloat32),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(cmomy.reduce_vals(x32, mom=3, axis=0, out=out_any), NDArrayAny),
        np.ndarray,
        np.float64,
    )

    # out override x and dtype
    check(
        assert_type(
            cmomy.reduce_vals(x32, mom=3, axis=0, out=out64, dtype=np.float32),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.reduce_vals(x64, mom=3, axis=0, out=out32, dtype=np.float64),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # Would like this to default to np.float64
    check(
        assert_type(cmomy.reduce_vals(x_arrayany, mom=3, axis=0), NDArrayAny),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.reduce_vals(x_arrayany, mom=3, axis=0, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(cmomy.reduce_vals(x_arraylike, mom=3, axis=0), NDArrayAny),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.reduce_vals(x_arraylike, mom=3, axis=0, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.reduce_vals(x_arraylike, mom=3, axis=0, dtype=np.float64, out=out32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    check(
        assert_type(
            cmomy.reduce_vals(x_arraylike, mom=3, axis=0, dtype=np.dtype("f8")),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.reduce_vals(da, mom=3, dim="dim_0", dtype=np.float32), xr.DataArray
        ),
        xr.DataArray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.reduce_vals(da_any, mom=3, dim="dim_0", dtype=np.float64, out=out32),
            Any,
        ),
        xr.DataArray,
        np.float32,
    )

    check(
        assert_type(cmomy.reduce_vals(ds, mom=3, dim="dim_0"), xr.Dataset),
        xr.Dataset,
    )

    if MYPY_ONLY:
        # TODO(wpk): would love to figure out how to get pyright to give this
        # as the fallback overload...
        g: ArrayLike | xr.DataArray | xr.Dataset = x_arrayany
        assert_type(
            cmomy.reduce_vals(g, mom=3, axis=0), xr.DataArray | xr.Dataset | NDArrayAny
        )


def test_reduce_data() -> None:
    check(
        assert_type(cmomy.reduce_data(data32, mom_ndim=1, axis=0), NDArrayFloat32),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(cmomy.reduce_data(data64, mom_ndim=1, axis=0), NDArrayFloat64),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.reduce_data(data_any, mom_ndim=1, axis=0, dtype=np.float64), Any
        ),
        np.ndarray,
        np.float64,
    )

    # dtype override x
    check(
        assert_type(
            cmomy.reduce_data(data32, mom_ndim=1, axis=0, dtype=np.float64),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.reduce_data(data64, mom_ndim=1, axis=0, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # out override x
    check(
        assert_type(
            cmomy.reduce_data(data32, mom_ndim=1, axis=0, out=out64), NDArrayFloat64
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.reduce_data(data64, mom_ndim=1, axis=0, out=out32), NDArrayFloat32
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.reduce_data(data32, mom_ndim=1, axis=0, out=out_any), NDArrayAny
        ),
        np.ndarray,
        np.float64,
    )

    # out override x and dtype
    check(
        assert_type(
            cmomy.reduce_data(data32, mom_ndim=1, axis=0, out=out64, dtype=np.float32),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.reduce_data(data64, mom_ndim=1, axis=0, out=out32, dtype=np.float64),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # Would like this to default to np.float64
    check(
        assert_type(cmomy.reduce_data(data_arrayany, mom_ndim=1, axis=0), NDArrayAny),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.reduce_data(data_arrayany, mom_ndim=1, axis=0, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(cmomy.reduce_data(data_arraylike, mom_ndim=1, axis=0), NDArrayAny),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.reduce_data(data_arraylike, mom_ndim=1, axis=0, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.reduce_data(
                data_arraylike, mom_ndim=1, axis=0, dtype=np.float64, out=out32
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # unknown dtype
    check(
        assert_type(
            cmomy.reduce_data(data32, mom_ndim=1, axis=0, dtype="f8"), NDArrayAny
        ),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.reduce_data(xdata, mom_ndim=1, dim="dim_0", dtype=np.float32),
            xr.DataArray,
        ),
        xr.DataArray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.reduce_data(
                xdata_any, mom_ndim=1, dim="dim_0", dtype=np.float32, out=out32
            ),
            Any,
        ),
        xr.DataArray,
        np.float32,
    )

    check(
        assert_type(cmomy.reduce_data(sdata, mom_ndim=1, dim="dim_0"), xr.Dataset),
        xr.Dataset,
    )

    if MYPY_ONLY:
        # TODO(wpk): would love to figure out how to get pyright to give this
        # as the fallback overload...
        g: ArrayLike | xr.DataArray | xr.Dataset = data_arrayany
        assert_type(
            cmomy.reduce_data(g, mom_ndim=1, axis=0),
            xr.DataArray | xr.Dataset | NDArrayAny,
        )


def test_reduce_data_grouped() -> None:
    check(
        assert_type(
            reduce_data_grouped(data32, mom_ndim=1, by=by, axis=0), NDArrayFloat32
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            reduce_data_grouped(data64, mom_ndim=1, by=by, axis=0), NDArrayFloat64
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(reduce_data_grouped(data_any, mom_ndim=1, by=by, axis=0), Any),
        np.ndarray,
        np.float64,
    )

    # dtype override x
    check(
        assert_type(
            reduce_data_grouped(data32, mom_ndim=1, by=by, axis=0, dtype=np.float64),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            reduce_data_grouped(data64, mom_ndim=1, by=by, axis=0, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # out override x
    check(
        assert_type(
            reduce_data_grouped(data32, mom_ndim=1, by=by, axis=0, out=group_out64),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            reduce_data_grouped(data64, mom_ndim=1, by=by, axis=0, out=group_out32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            reduce_data_grouped(data32, mom_ndim=1, by=by, axis=0, out=group_out_any),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )

    # out override x and dtype
    check(
        assert_type(
            reduce_data_grouped(
                data32, mom_ndim=1, by=by, axis=0, out=group_out64, dtype=np.float32
            ),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            reduce_data_grouped(
                data64, mom_ndim=1, by=by, axis=0, out=group_out32, dtype=np.float64
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # Would like this to default to np.float64
    check(
        assert_type(
            reduce_data_grouped(data_arrayany, mom_ndim=1, by=by, axis=0), NDArrayAny
        ),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            reduce_data_grouped(
                data_arrayany, mom_ndim=1, by=by, axis=0, dtype=np.float32
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            reduce_data_grouped(data_arraylike, mom_ndim=1, by=by, axis=0), NDArrayAny
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            reduce_data_grouped(
                data_arraylike, mom_ndim=1, by=by, axis=0, dtype=np.float32
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            reduce_data_grouped(
                data_arraylike,
                mom_ndim=1,
                by=by,
                axis=0,
                dtype=np.float64,
                out=group_out32,
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # unknown dtype
    check(
        assert_type(
            reduce_data_grouped(data32, mom_ndim=1, by=by, axis=0, dtype="f8"),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            reduce_data_grouped(
                xdata, mom_ndim=1, by=by, dim="dim_0", dtype=np.float32
            ),
            xr.DataArray,
        ),
        xr.DataArray,
        np.float32,
    )
    check(
        assert_type(
            reduce_data_grouped(
                xdata_any,
                mom_ndim=1,
                by=by,
                dim="dim_0",
                dtype=np.float64,
                out=group_out32,
            ),
            Any,
        ),
        xr.DataArray,
        np.float32,
    )

    check(
        assert_type(
            reduce_data_grouped(sdata, mom_ndim=1, by=by, dim="dim_0"), xr.Dataset
        ),
        xr.Dataset,
    )

    if MYPY_ONLY:
        # TODO(wpk): would love to figure out how to get pyright to give this
        # as the fallback overload...
        g: ArrayLike | xr.DataArray | xr.Dataset = data_arrayany
        assert_type(
            reduce_data_grouped(g, mom_ndim=1, by=by, axis=0),
            xr.DataArray | xr.Dataset | NDArrayAny,
        )


def test_reduce_data_indexed() -> None:
    _, index, group_start, group_end = cmomy.reduction.factor_by_to_index(by)

    check(
        assert_type(
            reduce_data_indexed(
                data32,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                axis=0,
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            reduce_data_indexed(
                data64,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                axis=0,
            ),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            reduce_data_indexed(
                data_any,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                axis=0,
            ),
            Any,
        ),
        np.ndarray,
        np.float64,
    )

    # dtype override x
    check(
        assert_type(
            reduce_data_indexed(
                data32,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                axis=0,
                dtype=np.float64,
            ),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            reduce_data_indexed(
                data64,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                axis=0,
                dtype=np.float32,
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # out override x
    check(
        assert_type(
            reduce_data_indexed(
                data32,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                axis=0,
                out=group_out64,
            ),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            reduce_data_indexed(
                data64,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                axis=0,
                out=group_out32,
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            reduce_data_indexed(
                data32,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                axis=0,
                out=group_out_any,
            ),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )

    # out override x and dtype
    check(
        assert_type(
            reduce_data_indexed(
                data32,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                axis=0,
                out=group_out64,
                dtype=np.float32,
            ),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            reduce_data_indexed(
                data64,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                axis=0,
                out=group_out32,
                dtype=np.float64,
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # Would like this to default to np.float64
    check(
        assert_type(
            reduce_data_indexed(
                data_arrayany,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                axis=0,
            ),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            reduce_data_indexed(
                data_arrayany,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                axis=0,
                dtype=np.float32,
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            reduce_data_indexed(
                data_arraylike,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                axis=0,
            ),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            reduce_data_indexed(
                data_arraylike,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                axis=0,
                dtype=np.float32,
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            reduce_data_indexed(
                data_arraylike,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                axis=0,
                dtype=np.float64,
                out=group_out32,
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # unknown dtype
    check(
        assert_type(
            reduce_data_indexed(
                data32,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                axis=0,
                dtype="f8",
            ),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            reduce_data_indexed(
                xdata,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                dim="dim_0",
                dtype=np.float32,
            ),
            xr.DataArray,
        ),
        xr.DataArray,
        np.float32,
    )
    check(
        assert_type(
            reduce_data_indexed(
                xdata_any,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                dim="dim_0",
                out=group_out32,
            ),
            Any,
        ),
        xr.DataArray,
        np.float32,
    )

    check(
        assert_type(
            reduce_data_indexed(
                sdata,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                dim="dim_0",
            ),
            xr.Dataset,
        ),
        xr.Dataset,
    )

    if MYPY_ONLY:
        # TODO(wpk): would love to figure out how to get pyright to give this
        # as the fallback overload...
        g: ArrayLike | xr.DataArray | xr.Dataset = data_arrayany
        assert_type(
            reduce_data_indexed(
                g,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                axis=0,
            ),
            xr.DataArray | xr.Dataset | NDArrayAny,
        )


# * resample ------------------------------------------------------------------
def test_resample_data() -> None:
    check(
        assert_type(
            cmomy.resample_data(data32, mom_ndim=1, freq=freq, axis=0), NDArrayFloat32
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.resample_data(data64, mom_ndim=1, freq=freq, axis=0), NDArrayFloat64
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(cmomy.resample_data(data_any, mom_ndim=1, freq=freq, axis=0), Any),
        np.ndarray,
        np.float64,
    )

    # dtype override x
    check(
        assert_type(
            cmomy.resample_data(
                data32, mom_ndim=1, freq=freq, axis=0, dtype=np.float64
            ),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.resample_data(
                data64, mom_ndim=1, freq=freq, axis=0, dtype=np.float32
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # out override x
    check(
        assert_type(
            cmomy.resample_data(data32, mom_ndim=1, freq=freq, axis=0, out=group_out64),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.resample_data(data64, mom_ndim=1, freq=freq, axis=0, out=group_out32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.resample_data(
                data32, mom_ndim=1, freq=freq, axis=0, out=group_out_any
            ),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )

    # out override x and dtype
    check(
        assert_type(
            cmomy.resample_data(
                data32, mom_ndim=1, freq=freq, axis=0, out=group_out64, dtype=np.float32
            ),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.resample_data(
                data64, mom_ndim=1, freq=freq, axis=0, out=group_out32, dtype=np.float64
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # Would like this to default to np.float64
    check(
        assert_type(
            cmomy.resample_data(data_arrayany, mom_ndim=1, freq=freq, axis=0),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.resample_data(
                data_arrayany, mom_ndim=1, freq=freq, axis=0, dtype=np.float32
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.resample_data(data_arraylike, mom_ndim=1, freq=freq, axis=0),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.resample_data(
                data_arraylike, mom_ndim=1, freq=freq, axis=0, dtype=np.float32
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.resample_data(
                data_arraylike,
                mom_ndim=1,
                freq=freq,
                axis=0,
                dtype=np.float64,
                out=group_out32,
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # unknown dtype
    check(
        assert_type(
            cmomy.resample_data(data32, mom_ndim=1, freq=freq, axis=0, dtype="f8"),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.resample_data(
                xdata, mom_ndim=1, freq=freq, dim="dim_0", dtype=np.float32
            ),
            xr.DataArray,
        ),
        xr.DataArray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.resample_data(
                xdata_any,
                mom_ndim=1,
                freq=freq,
                dim="dim_0",
                dtype=np.float64,
                out=group_out32,
            ),
            Any,
        ),
        xr.DataArray,
        np.float32,
    )

    check(
        assert_type(
            cmomy.resample_data(sdata, mom_ndim=1, freq=freq, dim="dim_0"), xr.Dataset
        ),
        xr.Dataset,
    )

    if MYPY_ONLY:
        # TODO(wpk): would love to figure out how to get pyright to give this
        # as the fallback overload...
        g: ArrayLike | xr.DataArray | xr.Dataset = data_arrayany
        assert_type(
            cmomy.resample_data(g, mom_ndim=1, freq=freq, axis=0),
            xr.DataArray | xr.Dataset | NDArrayAny,
        )


def test_resample_vals() -> None:
    check(
        assert_type(cmomy.resample_vals(x32, mom=3, freq=freq, axis=0), NDArrayFloat32),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(cmomy.resample_vals(x64, mom=3, freq=freq, axis=0), NDArrayFloat64),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(cmomy.resample_vals(x_any, mom=3, freq=freq, axis=0), Any),
        np.ndarray,
        np.float64,
    )

    # dtype override x
    check(
        assert_type(
            cmomy.resample_vals(x32, mom=3, freq=freq, axis=0, dtype=np.float64),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.resample_vals(x64, mom=3, freq=freq, axis=0, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # out override x
    check(
        assert_type(
            cmomy.resample_vals(x32, mom=3, freq=freq, axis=0, out=group_out64),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.resample_vals(x64, mom=3, freq=freq, axis=0, out=group_out32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.resample_vals(x32, mom=3, freq=freq, axis=0, out=group_out_any),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )

    # out override x and dtype
    check(
        assert_type(
            cmomy.resample_vals(
                x32, mom=3, freq=freq, axis=0, out=group_out64, dtype=np.float32
            ),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.resample_vals(
                x64, mom=3, freq=freq, axis=0, out=group_out32, dtype=np.float64
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # Would like this to default to np.float64
    check(
        assert_type(
            cmomy.resample_vals(x_arrayany, mom=3, freq=freq, axis=0),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.resample_vals(x_arrayany, mom=3, freq=freq, axis=0, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.resample_vals(x_arraylike, mom=3, freq=freq, axis=0),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.resample_vals(
                x_arraylike, mom=3, freq=freq, axis=0, dtype=np.float32
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.resample_vals(
                x_arraylike,
                mom=3,
                freq=freq,
                axis=0,
                dtype=np.float64,
                out=group_out32,
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # unknown dtype
    check(
        assert_type(
            cmomy.resample_vals(x32, mom=3, freq=freq, axis=0, dtype="f8"),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.resample_vals(da, mom=3, freq=freq, dim="dim_0", dtype=np.float32),
            xr.DataArray,
        ),
        xr.DataArray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.resample_vals(
                da_any,
                mom=3,
                freq=freq,
                dim="dim_0",
                dtype=np.float64,
                out=group_out32,
            ),
            Any,
        ),
        xr.DataArray,
        np.float32,
    )

    check(
        assert_type(cmomy.resample_vals(ds, mom=3, freq=freq, dim="dim_0"), xr.Dataset),
        xr.Dataset,
    )

    if MYPY_ONLY:
        # TODO(wpk): would love to figure out how to get pyright to give this
        # as the fallback overload...
        g: ArrayLike | xr.DataArray | xr.Dataset = x_arrayany
        assert_type(
            cmomy.resample_vals(g, mom=3, freq=freq, axis=0),
            xr.DataArray | xr.Dataset | NDArrayAny,
        )


def test_jackknife_data() -> None:
    check(
        assert_type(
            cmomy.resample.jackknife_data(data32, mom_ndim=1, axis=0), NDArrayFloat32
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.resample.jackknife_data(data64, mom_ndim=1, axis=0), NDArrayFloat64
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(cmomy.resample.jackknife_data(data_any, mom_ndim=1, axis=0), Any),
        np.ndarray,
        np.float64,
    )

    # dtype override x
    check(
        assert_type(
            cmomy.resample.jackknife_data(data32, mom_ndim=1, axis=0, dtype=np.float64),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.resample.jackknife_data(data64, mom_ndim=1, axis=0, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # out override x
    check(
        assert_type(
            cmomy.resample.jackknife_data(data32, mom_ndim=1, axis=0, out=same_out64),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.resample.jackknife_data(data64, mom_ndim=1, axis=0, out=same_out32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.resample.jackknife_data(data32, mom_ndim=1, axis=0, out=same_out_any),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )

    # out override x and dtype
    check(
        assert_type(
            cmomy.resample.jackknife_data(
                data32, mom_ndim=1, axis=0, out=same_out64, dtype=np.float32
            ),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.resample.jackknife_data(
                data64, mom_ndim=1, axis=0, out=same_out32, dtype=np.float64
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # Would like this to default to np.float64
    check(
        assert_type(
            cmomy.resample.jackknife_data(data_arrayany, mom_ndim=1, axis=0),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.resample.jackknife_data(
                data_arrayany, mom_ndim=1, axis=0, dtype=np.float32
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.resample.jackknife_data(data_arraylike, mom_ndim=1, axis=0),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.resample.jackknife_data(
                data_arraylike, mom_ndim=1, axis=0, dtype=np.float32
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.resample.jackknife_data(
                data_arraylike,
                mom_ndim=1,
                axis=0,
                dtype=np.float64,
                out=same_out32,
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # unknown dtype
    check(
        assert_type(
            cmomy.resample.jackknife_data(data32, mom_ndim=1, axis=0, dtype="f8"),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.resample.jackknife_data(
                xdata, mom_ndim=1, dim="dim_0", dtype=np.float32
            ),
            xr.DataArray,
        ),
        xr.DataArray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.resample.jackknife_data(
                xdata_any,
                mom_ndim=1,
                dim="dim_0",
                dtype=np.float64,
                out=same_out32,
            ),
            Any,
        ),
        xr.DataArray,
        np.float32,
    )

    check(
        assert_type(
            cmomy.resample.jackknife_data(sdata, mom_ndim=1, dim="dim_0"), xr.Dataset
        ),
        xr.Dataset,
    )

    if MYPY_ONLY:
        # TODO(wpk): would love to figure out how to get pyright to give this
        # as the fallback overload...
        g: ArrayLike | xr.DataArray | xr.Dataset = data_arrayany
        assert_type(
            cmomy.resample.jackknife_data(g, mom_ndim=1, axis=0),
            xr.DataArray | xr.Dataset | NDArrayAny,
        )


def test_jackknife_vals() -> None:
    check(
        assert_type(cmomy.resample.jackknife_vals(x32, mom=2, axis=0), NDArrayFloat32),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(cmomy.resample.jackknife_vals(x64, mom=2, axis=0), NDArrayFloat64),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(cmomy.resample.jackknife_vals(x_any, mom=2, axis=0), Any),
        np.ndarray,
        np.float64,
    )

    # dtype override x
    check(
        assert_type(
            cmomy.resample.jackknife_vals(x32, mom=2, axis=0, dtype=np.float64),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.resample.jackknife_vals(x64, mom=2, axis=0, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # out override x
    check(
        assert_type(
            cmomy.resample.jackknife_vals(x32, mom=2, axis=0, out=same_out64),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.resample.jackknife_vals(x64, mom=2, axis=0, out=same_out32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.resample.jackknife_vals(x32, mom=2, axis=0, out=same_out_any),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )

    # out override x and dtype
    check(
        assert_type(
            cmomy.resample.jackknife_vals(
                x32, mom=2, axis=0, out=same_out64, dtype=np.float32
            ),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.resample.jackknife_vals(
                x64, mom=2, axis=0, out=same_out32, dtype=np.float64
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # Would like this to default to np.float64
    check(
        assert_type(
            cmomy.resample.jackknife_vals(x_arrayany, mom=2, axis=0),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.resample.jackknife_vals(x_arrayany, mom=2, axis=0, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.resample.jackknife_vals(x_arraylike, mom=2, axis=0),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.resample.jackknife_vals(x_arraylike, mom=2, axis=0, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.resample.jackknife_vals(
                x_arraylike,
                mom=2,
                axis=0,
                dtype=np.float64,
                out=same_out32,
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # unknown dtype
    check(
        assert_type(
            cmomy.resample.jackknife_vals(x32, mom=2, axis=0, dtype="f8"),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.resample.jackknife_vals(da, mom=2, dim="dim_0", dtype=np.float32),
            xr.DataArray,
        ),
        xr.DataArray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.resample.jackknife_vals(
                da_any,
                mom=2,
                dim="dim_0",
                dtype=np.float64,
                out=same_out32,
            ),
            Any,
        ),
        xr.DataArray,
        np.float32,
    )

    check(
        assert_type(cmomy.resample.jackknife_vals(ds, mom=2, dim="dim_0"), xr.Dataset),
        xr.Dataset,
    )

    if MYPY_ONLY:
        # TODO(wpk): would love to figure out how to get pyright to give this
        # as the fallback overload...
        g: ArrayLike | xr.DataArray | xr.Dataset = x_arrayany
        assert_type(
            cmomy.resample.jackknife_vals(g, mom=2, axis=0),
            xr.DataArray | xr.Dataset | NDArrayAny,
        )


# * convert -------------------------------------------------------------------
def test_convert_moments_type() -> None:
    check(
        assert_type(cmomy.convert.moments_type(data32, mom_ndim=1), NDArrayFloat32),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(cmomy.convert.moments_type(data64, mom_ndim=1), NDArrayFloat64),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(cmomy.convert.moments_type(data_any, mom_ndim=1), Any),
        np.ndarray,
        np.float64,
    )
    # dtype override x
    check(
        assert_type(
            cmomy.convert.moments_type(data32, mom_ndim=1, dtype=np.float64),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.convert.moments_type(data64, mom_ndim=1, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # out override x
    check(
        assert_type(
            cmomy.convert.moments_type(data32, mom_ndim=1, out=same_out64),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.convert.moments_type(data64, mom_ndim=1, out=same_out32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.convert.moments_type(data32, mom_ndim=1, out=same_out_any), NDArrayAny
        ),
        np.ndarray,
        np.float64,
    )

    # out override x and dtype
    check(
        assert_type(
            cmomy.convert.moments_type(
                data32, mom_ndim=1, out=same_out64, dtype=np.float32
            ),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.convert.moments_type(
                data64, mom_ndim=1, out=same_out32, dtype=np.float64
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # Would like this to default to np.float64
    check(
        assert_type(cmomy.convert.moments_type(data_arrayany, mom_ndim=1), NDArrayAny),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.convert.moments_type(data_arrayany, mom_ndim=1, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(cmomy.convert.moments_type(data_arraylike, mom_ndim=1), NDArrayAny),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.convert.moments_type(data_arraylike, mom_ndim=1, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.convert.moments_type(
                data_arraylike, mom_ndim=1, dtype=np.float64, out=same_out32
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # unknown dtype
    check(
        assert_type(
            cmomy.convert.moments_type(data32, mom_ndim=1, dtype="f8"), NDArrayAny
        ),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(cmomy.convert.moments_type(xdata, mom_ndim=1), xr.DataArray),
        xr.DataArray,
        np.float64,
    )
    check(
        assert_type(cmomy.convert.moments_type(xdata_any, mom_ndim=1), Any),
        xr.DataArray,
        np.float64,
    )

    check(
        assert_type(cmomy.convert.moments_type(sdata, mom_ndim=1), xr.Dataset),
        xr.Dataset,
    )

    if MYPY_ONLY:
        # TODO(wpk): would love to figure out how to get pyright to give this
        # as the fallback overload...
        g: ArrayLike | xr.DataArray | xr.Dataset = data_arrayany
        assert_type(
            cmomy.convert.moments_type(g, mom_ndim=1),
            xr.DataArray | xr.Dataset | NDArrayAny,
        )


def test_cumulative() -> None:
    check(
        assert_type(
            cmomy.convert.cumulative(data32, mom_ndim=1, axis=0), NDArrayFloat32
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.convert.cumulative(data64, mom_ndim=1, axis=0), NDArrayFloat64
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(cmomy.convert.cumulative(data_any, mom_ndim=1, axis=0), Any),
        np.ndarray,
        np.float64,
    )

    # dtype override x
    check(
        assert_type(
            cmomy.convert.cumulative(data32, mom_ndim=1, axis=0, dtype=np.float64),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.convert.cumulative(data64, mom_ndim=1, axis=0, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # out override x
    check(
        assert_type(
            cmomy.convert.cumulative(data32, mom_ndim=1, axis=0, out=same_out64),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.convert.cumulative(data64, mom_ndim=1, axis=0, out=same_out32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.convert.cumulative(data32, mom_ndim=1, axis=0, out=same_out_any),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )

    # out override x and dtype
    check(
        assert_type(
            cmomy.convert.cumulative(
                data32, mom_ndim=1, axis=0, out=same_out64, dtype=np.float32
            ),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.convert.cumulative(
                data64, mom_ndim=1, axis=0, out=same_out32, dtype=np.float64
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # Would like this to default to np.float64
    check(
        assert_type(
            cmomy.convert.cumulative(data_arrayany, mom_ndim=1, axis=0), NDArrayAny
        ),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.convert.cumulative(
                data_arrayany, mom_ndim=1, axis=0, dtype=np.float32
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.convert.cumulative(data_arraylike, mom_ndim=1, axis=0), NDArrayAny
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.convert.cumulative(
                data_arraylike, mom_ndim=1, axis=0, dtype=np.float32
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.convert.cumulative(
                data_arraylike, mom_ndim=1, axis=0, dtype=np.float64, out=same_out32
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # unknown dtype
    check(
        assert_type(
            cmomy.convert.cumulative(data32, mom_ndim=1, axis=0, dtype="f8"), NDArrayAny
        ),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(cmomy.convert.cumulative(xdata, mom_ndim=1, axis=0), xr.DataArray),
        xr.DataArray,
        np.float64,
    )
    check(
        assert_type(cmomy.convert.cumulative(xdata_any, mom_ndim=1, axis=0), Any),
        xr.DataArray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.convert.cumulative(sdata, mom_ndim=1, dim="dim_0"), xr.Dataset
        ),
        xr.Dataset,
    )

    if MYPY_ONLY:
        # TODO(wpk): would love to figure out how to get pyright to give this
        # as the fallback overload...
        g: ArrayLike | xr.DataArray | xr.Dataset = data_arrayany
        assert_type(
            cmomy.convert.cumulative(g, mom_ndim=1, axis=0),
            xr.DataArray | xr.Dataset | NDArrayAny,
        )


def test_convert_moments_to_comoments() -> None:
    check(
        assert_type(
            cmomy.convert.moments_to_comoments(data32, mom=(1, -1)), NDArrayFloat32
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.convert.moments_to_comoments(data64, mom=(1, -1)), NDArrayFloat64
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(cmomy.convert.moments_to_comoments(data_any, mom=(1, -1)), Any),
        np.ndarray,
        np.float64,
    )

    # dtype override x
    check(
        assert_type(
            cmomy.convert.moments_to_comoments(data32, mom=(1, -1), dtype=np.float64),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.convert.moments_to_comoments(data64, mom=(1, -1), dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # Would like this to default to np.float64
    check(
        assert_type(
            cmomy.convert.moments_to_comoments(data_arrayany, mom=(1, -1)), NDArrayAny
        ),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.convert.moments_to_comoments(
                data_arrayany, mom=(1, -1), dtype=np.float32
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.convert.moments_to_comoments(data_arraylike, mom=(1, -1)), NDArrayAny
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.convert.moments_to_comoments(
                data_arraylike, mom=(1, -1), dtype=np.float32
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # unknown dtype
    check(
        assert_type(
            cmomy.convert.moments_to_comoments(data32, mom=(1, -1), dtype="f8"),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.convert.moments_to_comoments(xdata, mom=(1, -1)), xr.DataArray
        ),
        xr.DataArray,
        np.float64,
    )
    check(
        assert_type(cmomy.convert.moments_to_comoments(xdata_any, mom=(1, -1)), Any),
        xr.DataArray,
        np.float64,
    )

    check(
        assert_type(cmomy.convert.moments_to_comoments(sdata, mom=(1, -1)), xr.Dataset),
        xr.Dataset,
    )

    if MYPY_ONLY:
        # TODO(wpk): would love to figure out how to get pyright to give this
        # as the fallback overload...
        g: ArrayLike | xr.DataArray | xr.Dataset = data_arrayany
        assert_type(
            cmomy.convert.moments_to_comoments(g, mom=(1, -1)),
            xr.DataArray | xr.Dataset | NDArrayAny,
        )


def typecheck_concat() -> None:
    check(
        assert_type(convert.concat((x32, x32), axis=0), NDArrayFloat32),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(convert.concat((x64, x64), axis=0), NDArrayFloat64),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(convert.concat((x_arrayany, x_arrayany), axis=0), NDArrayAny),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(convert.concat((x_any, x_any), axis=0), Any), np.ndarray, np.float64
    )

    check(
        assert_type(convert.concat((da, da), dim="new"), xr.DataArray),
        xr.DataArray,
        np.float64,
    )
    check(
        assert_type(convert.concat((ds, ds), dim="new"), xr.Dataset),
        xr.Dataset,
    )
    check(
        assert_type(convert.concat((da_any, da_any), dim="new"), Any),
        xr.DataArray,
        np.float64,
    )

    check(
        assert_type(convert.concat((c32, c32), axis=0), CentralMomentsArrayFloat32),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(convert.concat((c64, c64), axis=0), CentralMomentsArrayFloat64),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(convert.concat((c_any, c_any), axis=0), CentralMomentsArrayAny),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(
            convert.concat((ca, ca), dim="new"),
            CentralMomentsDataArray,
        ),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )
    check(
        assert_type(
            convert.concat((cs, cs), dim="new"),
            CentralMomentsDataset,
        ),
        CentralMomentsXArray,
        None,
        xr.Dataset,
    )
    check(
        assert_type(convert.concat((ca_any, ca_any), axis=0), CentralMomentsDataAny),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )

    obj = cast("Any", c32)
    check(
        assert_type(convert.concat((obj, obj), axis=0), Any),
        CentralMomentsArray,
        np.float32,
    )


# * utils ---------------------------------------------------------------------
def test_vals_to_data() -> None:
    check(
        assert_type(cmomy.utils.vals_to_data(x32, mom=2), NDArrayFloat32),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(cmomy.utils.vals_to_data(x64, mom=2), NDArrayFloat64),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(cmomy.utils.vals_to_data(x_any, mom=2), Any),
        np.ndarray,
        np.float64,
    )

    # dtype override x
    check(
        assert_type(
            cmomy.utils.vals_to_data(x32, mom=2, dtype=np.float64),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.utils.vals_to_data(x64, mom=2, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # out override x
    check(
        assert_type(
            cmomy.utils.vals_to_data(x32, mom=2, out=same_out64),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.utils.vals_to_data(x64, mom=2, out=same_out32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(cmomy.utils.vals_to_data(x32, mom=2, out=same_out_any), NDArrayAny),
        np.ndarray,
        np.float64,
    )

    # out override x and dtype
    check(
        assert_type(
            cmomy.utils.vals_to_data(x32, mom=2, out=same_out64, dtype=np.float32),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.utils.vals_to_data(x64, mom=2, out=same_out32, dtype=np.float64),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # Would like this to default to np.float64
    check(
        assert_type(cmomy.utils.vals_to_data(x_arrayany, mom=2), NDArrayAny),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.utils.vals_to_data(x_arrayany, mom=2, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(cmomy.utils.vals_to_data(x_arraylike, mom=2), NDArrayAny),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.utils.vals_to_data(x_arraylike, mom=2, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.utils.vals_to_data(
                x_arraylike, mom=2, dtype=np.float64, out=same_out32
            ),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    # unknown dtype
    check(
        assert_type(cmomy.utils.vals_to_data(x32, mom=2, dtype="f8"), NDArrayAny),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(cmomy.utils.vals_to_data(da, mom=2), xr.DataArray),
        xr.DataArray,
        np.float64,
    )
    check(
        assert_type(cmomy.utils.vals_to_data(da_any, mom=2), Any),
        xr.DataArray,
        np.float64,
    )

    check(
        assert_type(cmomy.utils.vals_to_data(ds, mom=2), xr.Dataset),
        xr.Dataset,
    )

    if MYPY_ONLY:
        # TODO(wpk): would love to figure out how to get pyright to give this
        # as the fallback overload...
        g: ArrayLike | xr.DataArray | xr.Dataset = x_arrayany
        assert_type(
            cmomy.utils.vals_to_data(g, mom=2),
            xr.DataArray | xr.Dataset | NDArrayAny,
        )


def test_moveaxis() -> None:
    check(
        assert_type(cmomy.moveaxis(data32, 0, 1), NDArrayFloat32),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(cmomy.moveaxis(data64, 0, 1), NDArrayFloat64),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(cmomy.moveaxis(data_any, 0, 1), Any),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(cmomy.moveaxis(data_arrayany, 0, 1), NDArrayAny),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(cmomy.moveaxis(xdata, 0, 1), xr.DataArray),
        xr.DataArray,
        np.float64,
    )
    check(assert_type(cmomy.moveaxis(xdata_any, 0, 1), Any), xr.DataArray, np.float64)


def test_select_moment() -> None:
    check(
        assert_type(
            cmomy.utils.select_moment(data32, "weight", mom_ndim=1), NDArrayFloat32
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.utils.select_moment(data64, "weight", mom_ndim=1), NDArrayFloat64
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(cmomy.utils.select_moment(data_any, "weight", mom_ndim=1), Any),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.utils.select_moment(data_arrayany, "weight", mom_ndim=1), NDArrayAny
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(cmomy.utils.select_moment(data_any, "weight", mom_ndim=1), Any),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.utils.select_moment(xdata, "weight", mom_ndim=1),
            xr.DataArray,
        ),
        xr.DataArray,
    )
    check(
        assert_type(
            cmomy.utils.select_moment(sdata, "weight", mom_ndim=1),
            xr.Dataset,
        ),
        xr.Dataset,
    )


def test_assign_moment() -> None:
    check(
        assert_type(
            cmomy.utils.assign_moment(data32, weight=1, mom_ndim=1), NDArrayFloat32
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.utils.assign_moment(data64, weight=1, mom_ndim=1), NDArrayFloat64
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(cmomy.utils.assign_moment(data_any, weight=1, mom_ndim=1), Any),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.utils.assign_moment(data_arrayany, weight=1, mom_ndim=1), NDArrayAny
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(cmomy.utils.assign_moment(data_any, weight=1, mom_ndim=1), Any),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.utils.assign_moment(xdata, weight=1, mom_ndim=1),
            xr.DataArray,
        ),
        xr.DataArray,
    )
    check(
        assert_type(
            cmomy.utils.assign_moment(sdata, weight=1, mom_ndim=1),
            xr.Dataset,
        ),
        xr.Dataset,
    )


# * rolling -------------------------------------------------------------------
# * wrap ----------------------------------------------------------------------
# ** constructors
def test_wrap() -> None:
    check(
        assert_type(cmomy.wrap(data_any, dtype=np.float32), Any),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(cmomy.wrap(data_arraylike), CentralMomentsArrayAny),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.wrap(data_arraylike, dtype=np.float32), CentralMomentsArrayFloat32
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(cmomy.wrap(data32), CentralMomentsArrayFloat32),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(cmomy.wrap(data64), CentralMomentsArrayFloat64),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(cmomy.wrap(data64, dtype=np.float32), CentralMomentsArrayFloat32),
        CentralMomentsArray,
        np.float32,
    )

    check(
        assert_type(cmomy.wrap(xdata), CentralMomentsDataArray),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )
    check(
        assert_type(cmomy.wrap(sdata), CentralMomentsDataset),
        CentralMomentsXArray,
        None,
        xr.Dataset,
    )
    check(
        assert_type(
            cmomy.wrap(xdata_or_sdata),
            "CentralMomentsDataset | CentralMomentsDataArray",
        ),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )


def test_wrap_reduce_vals() -> None:
    check(
        assert_type(
            cmomy.wrap_reduce_vals(x32, mom=3, axis=0), CentralMomentsArrayFloat32
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.wrap_reduce_vals(x64, mom=3, axis=0), CentralMomentsArrayFloat64
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(cmomy.wrap_reduce_vals(x_any, mom=3, axis=0), Any),
        CentralMomentsArray,
        np.float64,
    )

    # dtype override x
    check(
        assert_type(
            cmomy.wrap_reduce_vals(x32, mom=3, axis=0, dtype=np.float64),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.wrap_reduce_vals(x64, mom=3, axis=0, dtype=np.float32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )

    # out override x
    check(
        assert_type(
            cmomy.wrap_reduce_vals(x32, mom=3, axis=0, out=out64),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.wrap_reduce_vals(x64, mom=3, axis=0, out=out32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.wrap_reduce_vals(x32, mom=3, axis=0, out=out_any),
            CentralMomentsArrayAny,
        ),
        CentralMomentsArray,
        np.float64,
    )

    # out override x and dtype
    check(
        assert_type(
            cmomy.wrap_reduce_vals(x32, mom=3, axis=0, out=out64, dtype=np.float32),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.wrap_reduce_vals(x64, mom=3, axis=0, out=out32, dtype=np.float64),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )

    # Would like this to default to np.float64
    check(
        assert_type(
            cmomy.wrap_reduce_vals(x_arrayany, mom=3, axis=0), CentralMomentsArrayAny
        ),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.wrap_reduce_vals(x_arrayany, mom=3, axis=0, dtype=np.float32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.wrap_reduce_vals(x_arraylike, mom=3, axis=0), CentralMomentsArrayAny
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.wrap_reduce_vals(x_arraylike, mom=3, axis=0, dtype=np.float32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.wrap_reduce_vals(
                x_arraylike, mom=3, axis=0, dtype=np.float64, out=out32
            ),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )

    check(
        assert_type(
            cmomy.wrap_reduce_vals(x_arraylike, mom=3, axis=0, dtype=np.dtype("f8")),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.wrap_reduce_vals(da, mom=3, dim="dim_0", dtype=np.float32),
            CentralMomentsDataArray,
        ),
        CentralMomentsXArray,
        np.float32,
        xr.DataArray,
    )
    check(
        assert_type(
            cmomy.wrap_reduce_vals(
                da_any, mom=3, dim="dim_0", dtype=np.float64, out=out32
            ),
            Any,
        ),
        CentralMomentsXArray,
        np.float32,
        xr.DataArray,
    )

    check(
        assert_type(
            cmomy.wrap_reduce_vals(ds, mom=3, dim="dim_0"), CentralMomentsDataset
        ),
        CentralMomentsXArray,
        None,
        xr.Dataset,
    )

    if MYPY_ONLY:
        # TODO(wpk): would love to figure out how to get pyright to give this
        # as the fallback overload...
        g: ArrayLike | xr.DataArray | xr.Dataset = x_arrayany
        assert_type(
            cmomy.wrap_reduce_vals(g, mom=3, axis=0),
            "CentralMomentsArrayAny | CentralMomentsDataArray | CentralMomentsDataset",
        )


def test_wrap_resample_vals() -> None:
    check(
        assert_type(
            cmomy.wrap_resample_vals(x32, mom=3, freq=freq, axis=0),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.wrap_resample_vals(x64, mom=3, freq=freq, axis=0),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(cmomy.wrap_resample_vals(x_any, mom=3, freq=freq, axis=0), Any),
        CentralMomentsArray,
        np.float64,
    )

    # dtype override x
    check(
        assert_type(
            cmomy.wrap_resample_vals(x32, mom=3, freq=freq, axis=0, dtype=np.float64),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.wrap_resample_vals(x64, mom=3, freq=freq, axis=0, dtype=np.float32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )

    # out override x
    check(
        assert_type(
            cmomy.wrap_resample_vals(x32, mom=3, freq=freq, axis=0, out=group_out64),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.wrap_resample_vals(x64, mom=3, freq=freq, axis=0, out=group_out32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.wrap_resample_vals(x32, mom=3, freq=freq, axis=0, out=group_out_any),
            CentralMomentsArrayAny,
        ),
        CentralMomentsArray,
        np.float64,
    )

    # out override x and dtype
    check(
        assert_type(
            cmomy.wrap_resample_vals(
                x32, mom=3, freq=freq, axis=0, out=group_out64, dtype=np.float32
            ),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.wrap_resample_vals(
                x64, mom=3, freq=freq, axis=0, out=group_out32, dtype=np.float64
            ),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )

    # Would like this to default to np.float64
    check(
        assert_type(
            cmomy.wrap_resample_vals(x_arrayany, mom=3, freq=freq, axis=0),
            CentralMomentsArrayAny,
        ),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.wrap_resample_vals(
                x_arrayany, mom=3, freq=freq, axis=0, dtype=np.float32
            ),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.wrap_resample_vals(x_arraylike, mom=3, freq=freq, axis=0),
            CentralMomentsArrayAny,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.wrap_resample_vals(
                x_arraylike, mom=3, freq=freq, axis=0, dtype=np.float32
            ),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.wrap_resample_vals(
                x_arraylike, mom=3, freq=freq, axis=0, dtype=np.float64, out=group_out32
            ),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )

    check(
        assert_type(
            cmomy.wrap_resample_vals(
                x_arraylike, mom=3, freq=freq, axis=0, dtype=np.dtype("f8")
            ),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.wrap_resample_vals(
                da, mom=3, freq=freq, dim="dim_0", dtype=np.float32
            ),
            CentralMomentsDataArray,
        ),
        CentralMomentsXArray,
        np.float32,
        xr.DataArray,
    )
    check(
        assert_type(
            cmomy.wrap_resample_vals(
                da_any, mom=3, freq=freq, dim="dim_0", dtype=np.float64, out=group_out32
            ),
            Any,
        ),
        CentralMomentsXArray,
        np.float32,
        xr.DataArray,
    )

    check(
        assert_type(
            cmomy.wrap_resample_vals(ds, mom=3, freq=freq, dim="dim_0"),
            CentralMomentsDataset,
        ),
        CentralMomentsXArray,
        None,
        xr.Dataset,
    )

    if MYPY_ONLY:
        # TODO(wpk): would love to figure out how to get pyright to give this
        # as the fallback overload...
        g: ArrayLike | xr.DataArray | xr.Dataset = x_arrayany
        assert_type(
            cmomy.wrap_resample_vals(g, mom=3, freq=freq, axis=0),
            "CentralMomentsArrayAny | CentralMomentsDataArray | CentralMomentsDataset",
        )


def test_wrap_raw() -> None:
    check(
        assert_type(cmomy.wrap_raw(data32), CentralMomentsArrayFloat32),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(cmomy.wrap_raw(data64), CentralMomentsArrayFloat64),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(cmomy.wrap_raw(data_any), Any),
        CentralMomentsArray,
        np.float64,
    )

    # dtype override x
    check(
        assert_type(
            cmomy.wrap_raw(data32, dtype=np.float64), CentralMomentsArrayFloat64
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.wrap_raw(data64, dtype=np.float32), CentralMomentsArrayFloat32
        ),
        CentralMomentsArray,
        np.float32,
    )

    # out override x
    check(
        assert_type(cmomy.wrap_raw(data32, out=same_out64), CentralMomentsArrayFloat64),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(cmomy.wrap_raw(data64, out=same_out32), CentralMomentsArrayFloat32),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(cmomy.wrap_raw(data32, out=same_out_any), CentralMomentsArrayAny),
        CentralMomentsArray,
        np.float64,
    )

    # out override x and dtype
    check(
        assert_type(
            cmomy.wrap_raw(data32, out=same_out64, dtype=np.float32),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.wrap_raw(data64, out=same_out32, dtype=np.float64),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )

    # Would like this to default to np.float64
    check(
        assert_type(cmomy.wrap_raw(data_arrayany), CentralMomentsArrayAny),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.wrap_raw(data_arrayany, dtype=np.float32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(cmomy.wrap_raw(data_arraylike), CentralMomentsArrayAny),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.wrap_raw(data_arraylike, dtype=np.float32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            cmomy.wrap_raw(data_arraylike, dtype=np.float64, out=same_out32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )

    check(
        assert_type(
            cmomy.wrap_raw(data_arraylike, dtype=np.dtype("f8")),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(cmomy.wrap_raw(xdata, dtype=np.float32), CentralMomentsDataArray),
        CentralMomentsXArray,
        np.float32,
        xr.DataArray,
    )
    check(
        assert_type(
            cmomy.wrap_raw(xdata_any, dtype=np.float64, out=same_out32),
            Any,
        ),
        CentralMomentsXArray,
        np.float32,
        xr.DataArray,
    )

    check(
        assert_type(cmomy.wrap_raw(sdata), CentralMomentsDataset),
        CentralMomentsXArray,
        None,
        xr.Dataset,
    )

    if MYPY_ONLY:
        # TODO(wpk): would love to figure out how to get pyright to give this
        # as the fallback overload...
        g: ArrayLike | xr.DataArray | xr.Dataset = data_arrayany
        assert_type(
            cmomy.wrap_raw(g),
            "CentralMomentsArrayAny | CentralMomentsDataArray | CentralMomentsDataset",
        )


def test_wrap_zeros_like() -> None:
    check(
        assert_type(cmomy.zeros_like(c32), CentralMomentsArrayFloat32),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(cmomy.zeros_like(c64), CentralMomentsArrayFloat64),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(cmomy.zeros_like(c_any), CentralMomentsArrayAny),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(
            cmomy.zeros_like(c32, dtype=np.float64), CentralMomentsArrayFloat64
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            cmomy.zeros_like(c_any, dtype=np.float64), CentralMomentsArrayFloat64
        ),
        CentralMomentsArray,
        np.float64,
    )

    obj = cast("Any", c64)
    check(
        assert_type(cmomy.zeros_like(obj), Any),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(cmomy.zeros_like(ca), CentralMomentsDataArray),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )
    check(
        assert_type(cmomy.zeros_like(cs), CentralMomentsDataset),
        CentralMomentsXArray,
        None,
        xr.Dataset,
    )

    check(
        assert_type(cmomy.zeros_like(ca_any), CentralMomentsDataAny),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )

    ca_or_cs = cast("CentralMomentsDataArray | CentralMomentsDataset", ca)
    check(
        assert_type(
            cmomy.zeros_like(ca_or_cs),
            "CentralMomentsDataArray | CentralMomentsDataset",
        ),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )


# ** class
def test_wrapped_check_types() -> None:
    # testing that object have correct type
    check(assert_type(c32, CentralMomentsArrayFloat32), CentralMomentsArray, np.float32)
    check(assert_type(c64, CentralMomentsArrayFloat64), CentralMomentsArray, np.float64)
    check(assert_type(c_any, CentralMomentsArrayAny), CentralMomentsArray, np.float64)
    check(
        assert_type(CentralMomentsArray(data_arraylike), CentralMomentsArrayAny),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(CentralMomentsArray(data_arrayany), CentralMomentsArrayAny),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(ca, CentralMomentsDataArray),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )
    check(
        assert_type(cs, CentralMomentsDataset),
        CentralMomentsXArray,
        None,
        xr.Dataset,
    )
    check(
        assert_type(ca_any, CentralMomentsDataAny),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )
    check(
        assert_type(cs_any, CentralMomentsDataAny),
        CentralMomentsXArray,
        None,
        xr.Dataset,
    )
    # check(
    #     assert_type(ca_or_cs, CentralMomentsDataAny),  # noqa: ERA001
    #     CentralMomentsXArray, np.float64, xr.DataArray)


def test_wrapped_init() -> None:
    check(
        assert_type(
            CentralMomentsArray(data32, mom_ndim=1), CentralMomentsArrayFloat32
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            CentralMomentsArray(data64, mom_ndim=1), CentralMomentsArrayFloat64
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            CentralMomentsArray(data64, mom_ndim=1, dtype=np.float32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            CentralMomentsArray(data32, mom_ndim=1, dtype=np.float64),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(
            CentralMomentsArray(data_arraylike, mom_ndim=1), CentralMomentsArrayAny
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            CentralMomentsArray(data_arraylike, mom_ndim=1, dtype=np.float32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            CentralMomentsArray(data_arraylike, mom_ndim=1, dtype=np.dtype("f8")),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            CentralMomentsArray(data_arraylike, mom_ndim=1, dtype="f8"),
            CentralMomentsArrayAny,
        ),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(
            CentralMomentsArray(x_arrayany, mom_ndim=1), CentralMomentsArrayAny
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            CentralMomentsArray(x_arrayany, mom_ndim=1, dtype=np.float32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            CentralMomentsArray(x_arrayany, mom_ndim=1, dtype=np.dtype("f8")),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            CentralMomentsArray(x_arrayany, mom_ndim=1, dtype="f8"),
            CentralMomentsArrayAny,
        ),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(CentralMomentsArray(x_any, mom_ndim=1), CentralMomentsArrayAny),
        CentralMomentsArray,
        np.float64,
    )

    # xarray
    check(
        assert_type(CentralMomentsXArray(xdata, mom_ndim=1), CentralMomentsDataArray),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )
    check(
        assert_type(CentralMomentsXArray(sdata, mom_ndim=1), CentralMomentsDataset),
        CentralMomentsXArray,
        obj_class=xr.Dataset,
    )
    check(
        assert_type(CentralMomentsXArray(xdata_any, mom_ndim=1), CentralMomentsDataAny),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )
    # TODO(wpk): think about what this should do...
    # reveal_type(CentralMomentsXArray(xdata_or_sdata, mom_ndim=1))  # noqa: ERA001


def test_wrapped_astype() -> None:
    check(
        assert_type(c32.astype(np.float64), CentralMomentsArrayFloat64),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(c64.astype(np.float32), CentralMomentsArrayFloat32),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(c32.astype(np.dtype("f8")), CentralMomentsArrayFloat64),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(c64.astype("f4"), CentralMomentsArrayAny),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(c64.astype("f4"), CentralMomentsArrayAny),
        CentralMomentsArray,
        np.float32,
    )

    check(
        assert_type(c_any.astype(np.float32), CentralMomentsArrayFloat32),
        CentralMomentsArray,
        np.float32,
    )

    check(
        assert_type(c32.astype(None), CentralMomentsArrayFloat64),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(c64.astype(None), CentralMomentsArrayFloat64),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(c_any.astype(None), CentralMomentsArrayFloat64),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(ca.astype(None), CentralMomentsDataArray),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )
    check(
        assert_type(ca.astype(np.float32), CentralMomentsDataArray),
        CentralMomentsXArray,
        np.float32,
        xr.DataArray,
    )
    check(
        assert_type(ca_any.astype(np.float32), CentralMomentsDataAny),
        CentralMomentsXArray,
        np.float32,
        xr.DataArray,
    )

    check(
        assert_type(cs.astype(None), CentralMomentsDataset),
        CentralMomentsXArray,
        None,
        xr.Dataset,
    )
    check(
        assert_type(cs_any.astype(None), CentralMomentsDataAny),
        CentralMomentsXArray,
        None,
        xr.Dataset,
    )


def test_wrapped_moments_to_comoments() -> None:
    check(
        assert_type(c32.moments_to_comoments(mom=(1, -1)), CentralMomentsArrayFloat32),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            c32.moments_to_comoments(mom=(1, -1), dtype=np.float64),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            c32.moments_to_comoments(mom=(1, -1), dtype="f8"), CentralMomentsArrayAny
        ),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(c64.moments_to_comoments(mom=(1, -1)), CentralMomentsArrayFloat64),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(c_any.moments_to_comoments(mom=(1, -1)), CentralMomentsArrayAny),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            c_any.moments_to_comoments(mom=(1, -1), dtype=np.float32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )

    check(
        assert_type(
            ca.moments_to_comoments(mom=(1, -1), dtype=np.float32),
            CentralMomentsDataArray,
        ),
        CentralMomentsXArray,
        np.float32,
        xr.DataArray,
    )
    check(
        assert_type(
            cs.moments_to_comoments(mom=(1, -1)),
            CentralMomentsDataset,
        ),
        CentralMomentsXArray,
        None,
        xr.Dataset,
    )
    check(
        assert_type(
            ca_any.moments_to_comoments(mom=(1, -1)),
            CentralMomentsDataAny,
        ),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )


def test_wrapped_assign_moment_central() -> None:
    check(
        assert_type(c32.assign_moment(weight=1.0), CentralMomentsArrayFloat32),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(c64.assign_moment(weight=1.0), CentralMomentsArrayFloat64),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(c_any.assign_moment(weight=1.0), CentralMomentsArrayAny),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(
            ca.assign_moment(weight=1.0),
            CentralMomentsDataArray,
        ),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )
    check(
        assert_type(
            cs.assign_moment(weight=1.0),
            CentralMomentsDataset,
        ),
        CentralMomentsXArray,
        None,
        xr.Dataset,
    )

    check(
        assert_type(ca_any.assign_moment(weight=1.0), CentralMomentsDataAny),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )


def test_wrapped_to_dataarray_to_dataset() -> None:
    check(
        assert_type(ca.to_dataset(), CentralMomentsDataset),
        CentralMomentsXArray,
        None,
        xr.Dataset,
    )
    check(
        assert_type(ca_any.to_dataset(), CentralMomentsDataset),
        CentralMomentsXArray,
        None,
        xr.Dataset,
    )
    check(
        assert_type(cs.to_dataarray(), CentralMomentsDataArray),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )
    check(
        assert_type(cs_any.to_dataarray(), CentralMomentsDataArray),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )


def test_wrapped_resample_and_reduce() -> None:
    check(
        assert_type(
            c32.resample_and_reduce(axis=0, freq=freq), CentralMomentsArrayFloat32
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            c64.resample_and_reduce(axis=0, freq=freq), CentralMomentsArrayFloat64
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            c_any.resample_and_reduce(axis=0, freq=freq), CentralMomentsArrayAny
        ),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(
            c32.resample_and_reduce(axis=0, freq=freq, dtype=np.float64),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            c64.resample_and_reduce(axis=0, freq=freq, dtype=np.float32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            c_any.resample_and_reduce(axis=0, freq=freq, dtype=np.float32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )

    check(
        assert_type(
            c32.resample_and_reduce(axis=0, freq=freq, out=group_out64),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            c64.resample_and_reduce(axis=0, freq=freq, out=group_out32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            c_any.resample_and_reduce(axis=0, freq=freq, out=group_out32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )

    check(
        assert_type(
            c32.resample_and_reduce(
                axis=0, freq=freq, dtype=np.float32, out=group_out64
            ),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            c64.resample_and_reduce(
                axis=0, freq=freq, dtype=np.float64, out=group_out32
            ),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            c_any.resample_and_reduce(
                axis=0, freq=freq, dtype=np.float64, out=group_out32
            ),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )

    check(
        assert_type(
            c32.resample_and_reduce(axis=0, freq=freq, out=group_out_any),
            CentralMomentsArrayAny,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            c32.resample_and_reduce(axis=0, freq=freq, dtype="f4"),
            CentralMomentsArrayAny,
        ),
        CentralMomentsArray,
        np.float32,
    )

    check(
        assert_type(
            ca.resample_and_reduce(dim="dim_0", freq=freq), CentralMomentsDataArray
        ),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )
    check(
        assert_type(
            cs.resample_and_reduce(dim="dim_0", freq=freq), CentralMomentsDataset
        ),
        CentralMomentsXArray,
        None,
        xr.Dataset,
    )
    check(
        assert_type(
            ca_any.resample_and_reduce(dim="dim_0", freq=freq), CentralMomentsDataAny
        ),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )
    check(
        assert_type(
            ca.resample_and_reduce(dim="dim_0", freq=freq, dtype=np.float32),
            CentralMomentsDataArray,
        ),
        CentralMomentsXArray,
        np.float32,
        xr.DataArray,
    )
    check(
        assert_type(
            ca.resample_and_reduce(dim="dim_0", freq=freq, out=group_out_any),
            CentralMomentsDataArray,
        ),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )


def test_wrapped_jackknife_and_reduce() -> None:
    check(
        assert_type(c32.jackknife_and_reduce(axis=0), CentralMomentsArrayFloat32),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(c64.jackknife_and_reduce(axis=0), CentralMomentsArrayFloat64),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            c_any.jackknife_and_reduce(axis=0, data_reduced=c_any.reduce(axis=0)),
            CentralMomentsArrayAny,
        ),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(
            c32.jackknife_and_reduce(axis=0, dtype=np.float64),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            c64.jackknife_and_reduce(axis=0, dtype=np.float32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            c_any.jackknife_and_reduce(axis=0, dtype=np.float32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )

    check(
        assert_type(
            c32.jackknife_and_reduce(axis=0, out=same_out64),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            c64.jackknife_and_reduce(axis=0, out=same_out32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            c_any.jackknife_and_reduce(axis=0, out=same_out32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )

    check(
        assert_type(
            c32.jackknife_and_reduce(axis=0, dtype=np.float32, out=same_out64),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            c64.jackknife_and_reduce(axis=0, dtype=np.float64, out=same_out32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            c_any.jackknife_and_reduce(axis=0, dtype=np.float64, out=same_out32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )

    check(
        assert_type(
            c32.jackknife_and_reduce(axis=0, out=same_out_any),
            CentralMomentsArrayAny,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            c32.jackknife_and_reduce(axis=0, dtype="f4"),
            CentralMomentsArrayAny,
        ),
        CentralMomentsArray,
        np.float32,
    )

    check(
        assert_type(ca.jackknife_and_reduce(dim="dim_0"), CentralMomentsDataArray),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )
    check(
        assert_type(cs.jackknife_and_reduce(dim="dim_0"), CentralMomentsDataset),
        CentralMomentsXArray,
        None,
        xr.Dataset,
    )
    check(
        assert_type(
            ca_any.jackknife_and_reduce(
                dim="dim_0", data_reduced=ca_any.reduce(dim="dim_0")
            ),
            CentralMomentsDataAny,
        ),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )
    check(
        assert_type(
            ca.jackknife_and_reduce(dim="dim_0", dtype=np.float32),
            CentralMomentsDataArray,
        ),
        CentralMomentsXArray,
        np.float32,
        xr.DataArray,
    )
    check(
        assert_type(
            ca.jackknife_and_reduce(dim="dim_0", out=same_out_any),
            CentralMomentsDataArray,
        ),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )


def test_wrapped_reduce() -> None:
    check(
        assert_type(c32.reduce(axis=0), CentralMomentsArrayFloat32),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(c64.reduce(axis=0), CentralMomentsArrayFloat64),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            c_any.reduce(axis=0),
            CentralMomentsArrayAny,
        ),
        CentralMomentsArray,
        np.float64,
    )

    check(
        assert_type(
            c32.reduce(axis=0, dtype=np.float64),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            c64.reduce(axis=0, dtype=np.float32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            c_any.reduce(axis=0, dtype=np.float32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )

    check(
        assert_type(
            c32.reduce(axis=0, out=out64),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            c64.reduce(axis=0, out=out32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            c_any.reduce(axis=0, out=out32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )

    check(
        assert_type(
            c32.reduce(axis=0, dtype=np.float32, out=out64),
            CentralMomentsArrayFloat64,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            c64.reduce(axis=0, dtype=np.float64, out=out32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )
    check(
        assert_type(
            c_any.reduce(axis=0, dtype=np.float64, out=out32),
            CentralMomentsArrayFloat32,
        ),
        CentralMomentsArray,
        np.float32,
    )

    check(
        assert_type(
            c32.reduce(axis=0, out=out_any),
            CentralMomentsArrayAny,
        ),
        CentralMomentsArray,
        np.float64,
    )
    check(
        assert_type(
            c32.reduce(axis=0, dtype="f4"),
            CentralMomentsArrayAny,
        ),
        CentralMomentsArray,
        np.float32,
    )

    check(
        assert_type(ca.reduce(dim="dim_0"), CentralMomentsDataArray),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )
    check(
        assert_type(cs.reduce(dim="dim_0"), CentralMomentsDataset),
        CentralMomentsXArray,
        None,
        xr.Dataset,
    )
    check(
        assert_type(
            ca_any.reduce(dim="dim_0"),
            CentralMomentsDataAny,
        ),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )
    check(
        assert_type(
            ca.reduce(dim="dim_0", dtype=np.float32),
            CentralMomentsDataArray,
        ),
        CentralMomentsXArray,
        np.float32,
        xr.DataArray,
    )
    check(
        assert_type(
            ca.reduce(dim="dim_0", out=out_any),
            CentralMomentsDataArray,
        ),
        CentralMomentsXArray,
        np.float64,
        xr.DataArray,
    )


def test_wrapped_cumulative() -> None:
    check(
        assert_type(c32.cumulative(axis=0), NDArrayFloat32),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(c64.cumulative(axis=0), NDArrayFloat64),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            c_any.cumulative(axis=0),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )

    check(
        assert_type(
            c32.cumulative(axis=0, dtype=np.float64),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            c64.cumulative(axis=0, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            c_any.cumulative(axis=0, dtype=np.float32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    check(
        assert_type(
            c32.cumulative(axis=0, out=same_out64),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            c64.cumulative(axis=0, out=same_out32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            c_any.cumulative(axis=0, out=same_out32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    check(
        assert_type(
            c32.cumulative(axis=0, dtype=np.float32, out=same_out64),
            NDArrayFloat64,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            c64.cumulative(axis=0, dtype=np.float64, out=same_out32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )
    check(
        assert_type(
            c_any.cumulative(axis=0, dtype=np.float64, out=same_out32),
            NDArrayFloat32,
        ),
        np.ndarray,
        np.float32,
    )

    check(
        assert_type(
            c32.cumulative(axis=0, out=same_out_any),
            NDArrayAny,
        ),
        np.ndarray,
        np.float64,
    )
    check(
        assert_type(
            c32.cumulative(axis=0, dtype="f4"),
            NDArrayAny,
        ),
        np.ndarray,
        np.float32,
    )

    check(
        assert_type(ca.cumulative(dim="dim_0"), xr.DataArray),
        xr.DataArray,
        np.float64,
    )
    check(
        assert_type(cs.cumulative(dim="dim_0"), xr.Dataset),
        xr.Dataset,
    )
    check(
        assert_type(
            ca_any.cumulative(dim="dim_0"),
            Any,
        ),
        xr.DataArray,
        np.float64,
    )
    check(
        assert_type(
            ca.cumulative(dim="dim_0", dtype=np.float32),
            xr.DataArray,
        ),
        xr.DataArray,
        np.float32,
    )
    check(
        assert_type(
            ca.cumulative(dim="dim_0", out=same_out_any),
            xr.DataArray,
        ),
        xr.DataArray,
        np.float64,
    )


# * tests
# TODO(wpk): start back here.
# if mypy properly supports partial will add this...
# def typecheck_convert_central_to_raw() -> None:
def typecheck_bootstrap_confidence_interval(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
) -> None:
    assert_type(
        cmomy.bootstrap_confidence_interval(x32),
        NDArrayFloat32,
    )

    assert_type(
        cmomy.bootstrap_confidence_interval(x64),
        NDArrayFloat64,
    )

    da = xr.DataArray(x32)
    assert_type(
        cmomy.bootstrap_confidence_interval(da),
        xr.DataArray,
    )
    assert_type(
        cmomy.bootstrap_confidence_interval(da.to_dataset()),
        xr.Dataset,
    )


def typecheck_resample_vals(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    freq: NDArrayInt,
    xany: NDArrayAny,
) -> None:
    assert_type(resample_vals(x32, freq=freq, mom=3), NDArrayFloat32)
    assert_type(resample_vals(x64, freq=freq, mom=3), NDArrayFloat64)

    assert_type(resample_vals(x32, freq=freq, mom=3, dtype=np.float64), NDArrayFloat64)
    assert_type(resample_vals(x64, freq=freq, mom=3, dtype=np.float32), NDArrayFloat32)

    assert_type(resample_vals(x32, freq=freq, mom=3, out=out64), NDArrayFloat64)
    assert_type(resample_vals(x64, freq=freq, mom=3, out=out32), NDArrayFloat32)

    assert_type(
        resample_vals(x32, freq=freq, mom=3, out=out64, dtype=np.float32),
        NDArrayFloat64,
    )
    assert_type(
        resample_vals(x64, freq=freq, mom=3, out=out32, dtype=np.float64),
        NDArrayFloat32,
    )

    assert_type(xany, NDArrayAny)

    # Would like this to default to np.float64
    assert_type(resample_vals(xany, freq=freq, mom=3), NDArrayAny)
    assert_type(resample_vals(xany, freq=freq, mom=3, dtype=np.float32), NDArrayFloat32)
    assert_type(
        resample_vals([1, 2, 3], freq=freq, mom=3, dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        resample_vals([1, 2, 3], freq=freq, mom=3, dtype=np.float64),
        NDArrayFloat64,
    )

    assert_type(resample_vals([1.0, 2.0, 3.0], freq=freq, mom=3), NDArrayAny)
    assert_type(resample_vals(x32, freq=freq, mom=3, dtype="f8"), NDArrayAny)

    xx = xr.DataArray(x32)
    assert_type(resample_vals(xx, freq=freq, mom=3), xr.DataArray)

    assert_type(resample_vals(xx.to_dataset(), freq=freq, mom=3), xr.Dataset)

    if MYPY_ONLY:
        g: ArrayLike | xr.DataArray | xr.Dataset = xany
        assert_type(
            resample_vals(g, freq=freq, mom=3),
            Union[xr.DataArray, xr.Dataset, NDArrayAny],
        )


def typecheck_centralmoments_zeros() -> None:
    assert_type(CentralMomentsArray.zeros(mom=3), CentralMomentsArrayFloat64)
    assert_type(
        CentralMomentsArray.zeros(mom=3, dtype=None),
        CentralMomentsArrayFloat64,
    )
    assert_type(
        CentralMomentsArray.zeros(mom=3, dtype=np.float32),
        CentralMomentsArrayFloat32,
    )
    assert_type(
        CentralMomentsArray.zeros(mom=3, dtype=np.dtype("f8")),
        CentralMomentsArrayFloat64,
    )
    assert_type(
        CentralMomentsArray.zeros(mom=3, dtype="f8"),
        CentralMomentsArrayAny,
    )

    assert_type(
        CentralMomentsArray[np.float32].zeros(mom=3, dtype="f4"),
        CentralMomentsArrayFloat32,
    )

    assert_type(
        CentralMomentsArray.zeros(mom=3, dtype=np.dtype("f4")),
        CentralMomentsArrayFloat32,
    )


def typecheck_xcentralmoments_zeros() -> None:
    assert_type(CentralMomentsXArray.zeros(mom=3), CentralMomentsDataArray)
    assert_type(
        CentralMomentsXArray.zeros(mom=3, dtype=None),
        CentralMomentsDataArray,
    )


def typecheck_centralmoments_from_vals(
    data32: NDArrayFloat32,
    data64: NDArrayFloat64,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    xany: NDArrayAny,
) -> None:
    assert_type(
        CentralMomentsArray.from_vals(data32, mom=3, axis=0),
        CentralMomentsArrayFloat32,
    )
    assert_type(
        CentralMomentsArray.from_vals(data64, mom=3, axis=0),
        CentralMomentsArrayFloat64,
    )
    assert_type(
        CentralMomentsArray.from_vals(data32, mom=3, axis=0, dtype=np.float64),
        CentralMomentsArrayFloat64,
    )
    assert_type(
        CentralMomentsArray.from_vals(data32, mom=3, axis=0, dtype=np.dtype("f8")),
        CentralMomentsArrayFloat64,
    )
    assert_type(
        CentralMomentsArray.from_vals(data32, mom=3, axis=0, dtype="f8"),
        CentralMomentsArrayAny,
    )

    # using out
    assert_type(
        CentralMomentsArray.from_vals(xany, mom=3, axis=0, dtype=np.float32, out=out64),
        CentralMomentsArrayFloat64,
    )
    assert_type(
        CentralMomentsArray.from_vals(xany, mom=3, axis=0, dtype=np.float64, out=out32),
        CentralMomentsArrayFloat32,
    )

    assert_type(
        CentralMomentsArray.from_vals(xany, mom=3, axis=0), CentralMomentsArrayAny
    )

    assert_type(
        CentralMomentsArray.from_vals([1, 2, 3], mom=3, axis=0, dtype="f8"),
        CentralMomentsArrayAny,
    )
    assert_type(
        CentralMomentsArray.from_vals([1, 2, 3], mom=3, axis=0, dtype=np.float32),
        CentralMomentsArrayFloat32,
    )


def typecheck_xcentralmoments_from_vals(
    da: xr.DataArray,
    ds: xr.Dataset,
    out: NDArrayFloat64,
) -> None:
    ca = CentralMomentsXArray.from_vals(da, mom=3, axis=0, out=out)
    cs = CentralMomentsXArray.from_vals(ds, mom=3, dim="a")

    assert_type(
        ca,
        CentralMomentsDataArray,
    )
    assert_type(
        cs,
        CentralMomentsDataset,
    )


def typecheck_classmethod_from_raw(
    xany: NDArrayAny,
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    da: xr.DataArray,
    ds: xr.Dataset,
    a: Any,
) -> None:
    assert_type(CentralMomentsArray.from_raw(xany, mom_ndim=1), CentralMomentsArrayAny)
    assert_type(
        CentralMomentsArray.from_raw(x32, mom_ndim=1), CentralMomentsArrayFloat32
    )
    assert_type(
        CentralMomentsArray.from_raw(x64, mom_ndim=1), CentralMomentsArrayFloat64
    )

    assert_type(
        CentralMomentsArray.from_raw(a, mom_ndim=1),
        Any,
    )

    assert_type(
        CentralMomentsXArray.from_raw(da, mom_ndim=1),
        CentralMomentsDataArray,
    )
    assert_type(
        CentralMomentsXArray.from_raw(ds, mom_ndim=1),
        CentralMomentsDataset,
    )

    assert_type(CentralMomentsXArray.from_raw(a, mom_ndim=1), Any)


def typecheck_centralmoments_newlike(
    data32: NDArrayFloat32,
    data64: NDArrayFloat64,
) -> None:
    c32 = CentralMomentsArray.zeros(mom=3, dtype=np.float32)
    c64 = CentralMomentsArray.zeros(mom=3, dtype=np.float64)
    assert_type(c32, CentralMomentsArrayFloat32)
    assert_type(c64, CentralMomentsArrayFloat64)
    assert_type(c32.new_like(), CentralMomentsArrayFloat32)
    assert_type(c32.new_like(dtype=np.float64), CentralMomentsArrayFloat64)
    assert_type(c64.new_like(), CentralMomentsArrayFloat64)
    assert_type(c32.new_like(dtype=np.float32), CentralMomentsArrayFloat32)

    assert_type(c32.new_like(data32), CentralMomentsArrayFloat32)
    assert_type(c32.new_like(data64), CentralMomentsArrayFloat64)
    assert_type(c64.new_like(data32), CentralMomentsArrayFloat32)

    assert_type(c32.new_like(data32, dtype=np.float64), CentralMomentsArrayFloat64)
    assert_type(c32.new_like(data64, dtype=np.float32), CentralMomentsArrayFloat32)
    assert_type(c64.new_like(data32, dtype=np.float64), CentralMomentsArrayFloat64)

    assert_type(c64.new_like(data64, dtype="f4"), CentralMomentsArrayAny)
    assert_type(c64.new_like(dtype="f4"), CentralMomentsArrayAny)


def typecheck_xcentralmoments_newlike(
    data: NDArrayFloat64,
) -> None:
    xr.DataArray(data)
    ca = CentralMomentsXArray.zeros(mom=3, dtype=np.float64)
    cs = ca.to_dataset()

    assert_type(ca.new_like(), CentralMomentsDataArray)
    assert_type(ca.new_like(dtype=np.float32), CentralMomentsDataArray)
    assert_type(cs.new_like(), CentralMomentsDataset)
    assert_type(cs.new_like(dtype=np.float32), CentralMomentsDataset)


def typecheck_central_resample_vals(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    freq: NDArrayInt,
    xany: NDArrayAny,
) -> None:
    assert_type(
        CentralMomentsArray.from_resample_vals(x32, freq=freq, mom=3, axis=0),
        CentralMomentsArrayFloat32,
    )
    assert_type(
        CentralMomentsArray.from_resample_vals(x64, freq=freq, mom=3, axis=0),
        CentralMomentsArrayFloat64,
    )

    assert_type(
        CentralMomentsArray.from_resample_vals(
            xany, freq=freq, mom=3, axis=0, dtype=np.float64
        ),
        CentralMomentsArrayFloat64,
    )
    assert_type(
        CentralMomentsArray.from_resample_vals(
            xany, freq=freq, mom=3, axis=0, dtype=np.float32
        ),
        CentralMomentsArrayFloat32,
    )
    assert_type(
        CentralMomentsArray.from_resample_vals(
            x64,
            freq=freq,
            mom=3,
            axis=0,
            dtype="f8",
        ),
        CentralMomentsArrayAny,
    )

    assert_type(
        CentralMomentsArray.from_resample_vals(
            x64,
            freq=freq,
            mom=3,
            axis=0,
            out=out32,
        ),
        CentralMomentsArrayFloat32,
    )

    assert_type(
        CentralMomentsArray.from_resample_vals(
            x64,
            freq=freq,
            mom=3,
            axis=0,
            out=out64,
        ),
        CentralMomentsArrayFloat64,
    )
    assert_type(
        CentralMomentsArray.from_resample_vals(
            xany, freq=freq, mom=3, axis=0, out=out32, dtype=np.float64
        ),
        CentralMomentsArrayFloat32,
    )

    assert_type(
        CentralMomentsArray.from_resample_vals(
            xany, freq=freq, mom=3, axis=0, out=out64, dtype=np.float32
        ),
        CentralMomentsArrayFloat64,
    )

    assert_type(xany, NDArrayAny)

    # Would like this to default to np.float64
    assert_type(
        CentralMomentsArray.from_resample_vals(xany, freq=freq, mom=3, axis=0),
        CentralMomentsArrayAny,
    )
    assert_type(
        CentralMomentsArray.from_resample_vals(
            xany, freq=freq, mom=3, axis=0, dtype=np.float32
        ),
        CentralMomentsArrayFloat32,
    )
    assert_type(
        CentralMomentsArray.from_resample_vals(
            [1, 2, 3], freq=freq, mom=3, axis=0, dtype=np.float32
        ),
        CentralMomentsArrayFloat32,
    )
    assert_type(
        CentralMomentsArray.from_resample_vals([1, 2, 3], freq=freq, mom=3, axis=0),
        CentralMomentsArrayAny,
    )
    assert_type(
        CentralMomentsArray.from_resample_vals(
            [1, 2, 3], freq=freq, mom=3, axis=0, dtype=np.float64
        ),
        CentralMomentsArrayFloat64,
    )

    assert_type(
        CentralMomentsArray.from_resample_vals(
            [1.0, 2.0, 3.0], freq=freq, mom=3, axis=0
        ),
        CentralMomentsArrayAny,
    )
    assert_type(
        CentralMomentsArray.from_resample_vals(
            [1.0, 2.0, 3.0], freq=freq, mom=3, axis=0, dtype="f4"
        ),
        CentralMomentsArrayAny,
    )
    assert_type(
        CentralMomentsArray.from_resample_vals(
            x32, freq=freq, mom=3, axis=0, dtype="f4"
        ),
        CentralMomentsArrayAny,
    )


def typecheck_xcentral_resample_vals(
    da: xr.DataArray,
    ds: xr.Dataset,
    freq: NDArrayInt,
) -> None:
    assert_type(
        CentralMomentsXArray.from_resample_vals(da, freq=freq, mom=3),
        CentralMomentsDataArray,
    )
    assert_type(
        CentralMomentsXArray.from_resample_vals(ds, freq=freq, mom=3),
        CentralMomentsDataset,
    )


# * Moving
def typecheck_rolling_data(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    outany: NDArrayAny,
    xany: NDArrayAny,
) -> None:
    assert_type(rolling.rolling_data(x32, mom_ndim=1, window=3), NDArrayFloat32)
    assert_type(rolling.rolling_data(x64, mom_ndim=1, window=3), NDArrayFloat64)

    assert_type(
        rolling.rolling_data(x32, mom_ndim=1, window=3, dtype=np.float64),
        NDArrayFloat64,
    )
    assert_type(
        rolling.rolling_data(x64, mom_ndim=1, window=3, dtype=np.float32),
        NDArrayFloat32,
    )

    assert_type(
        rolling.rolling_data(x32, mom_ndim=1, window=3, out=out64),
        NDArrayFloat64,
    )
    assert_type(
        rolling.rolling_data(x64, mom_ndim=1, window=3, out=out32),
        NDArrayFloat32,
    )

    assert_type(
        rolling.rolling_data(x32, mom_ndim=1, window=3, out=out64, dtype=np.float32),
        NDArrayFloat64,
    )
    assert_type(
        rolling.rolling_data(x64, mom_ndim=1, window=3, out=out32, dtype=np.float64),
        NDArrayFloat32,
    )
    assert_type(
        rolling.rolling_data(x64, mom_ndim=1, window=3, out=outany, dtype=np.float64),
        NDArrayAny,
    )

    assert_type(xany, NDArrayAny)

    # Would like this to default to np.float64
    assert_type(rolling.rolling_data(xany, mom_ndim=1, window=3), NDArrayAny)
    assert_type(
        rolling.rolling_data(xany, mom_ndim=1, window=3, dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        rolling.rolling_data([1, 2, 3], mom_ndim=1, window=3, dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        rolling.rolling_data([1, 2, 3], mom_ndim=1, window=3, dtype=np.float64),
        NDArrayFloat64,
    )

    assert_type(
        rolling.rolling_data([1, 2, 3], mom_ndim=1, window=3, dtype="f8"),
        NDArrayAny,
    )
    assert_type(
        rolling.rolling_data(x32, mom_ndim=1, window=3, dtype="f8"),
        NDArrayAny,
    )
    assert_type(
        rolling.rolling_data(xany, mom_ndim=1, window=3, dtype="f8"),
        NDArrayAny,
    )

    assert_type(rolling.rolling_data([1.0, 2.0, 3.0], mom_ndim=1, window=3), NDArrayAny)

    xx = xr.DataArray(x32)
    assert_type(rolling.rolling_data(xx, mom_ndim=1, window=3), xr.DataArray)
    assert_type(rolling.rolling_data(xx.to_dataset(), mom_ndim=1, window=3), xr.Dataset)

    if MYPY_ONLY:
        g: ArrayLike | xr.DataArray | xr.Dataset = xany
        assert_type(
            rolling.rolling_data(g, mom_ndim=1, window=3),
            Union[xr.DataArray, xr.Dataset, NDArrayAny],
        )


def typecheck_rolling_vals(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    outany: NDArrayAny,
    xany: NDArrayAny,
) -> None:
    assert_type(rolling.rolling_vals(x32, mom=3, window=3), NDArrayFloat32)
    assert_type(rolling.rolling_vals(x64, mom=3, window=3), NDArrayFloat64)

    assert_type(
        rolling.rolling_vals(x32, mom=3, window=3, dtype=np.float64),
        NDArrayFloat64,
    )
    assert_type(
        rolling.rolling_vals(x64, mom=3, window=3, dtype=np.float32),
        NDArrayFloat32,
    )

    assert_type(rolling.rolling_vals(x32, mom=3, window=3, out=out64), NDArrayFloat64)
    assert_type(rolling.rolling_vals(x64, mom=3, window=3, out=out32), NDArrayFloat32)

    assert_type(
        rolling.rolling_vals(x32, mom=3, window=3, out=out64, dtype=np.float32),
        NDArrayFloat64,
    )
    assert_type(
        rolling.rolling_vals(x64, mom=3, window=3, out=out32, dtype=np.float64),
        NDArrayFloat32,
    )
    assert_type(
        rolling.rolling_vals(x64, mom=3, window=3, out=outany, dtype=np.float64),
        NDArrayAny,
    )

    assert_type(xany, NDArrayAny)

    # Would like this to default to np.float64
    assert_type(rolling.rolling_vals(xany, mom=3, window=3), NDArrayAny)
    assert_type(
        rolling.rolling_vals(xany, mom=3, window=3, dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        rolling.rolling_vals([1, 2, 3], mom=3, window=3, dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        rolling.rolling_vals([1, 2, 3], mom=3, window=3, dtype=np.float64),
        NDArrayFloat64,
    )

    assert_type(
        rolling.rolling_vals([1, 2, 3], mom=3, window=3, dtype="f8"),
        NDArrayAny,
    )
    assert_type(
        rolling.rolling_vals(x32, mom=3, window=3, dtype="f8"),
        NDArrayAny,
    )
    assert_type(
        rolling.rolling_vals(xany, mom=3, window=3, dtype="f8"),
        NDArrayAny,
    )

    assert_type(rolling.rolling_vals([1.0, 2.0, 3.0], mom=3, window=3), NDArrayAny)

    xx = xr.DataArray(x32)
    assert_type(rolling.rolling_vals(xx, mom=3, window=3), xr.DataArray)
    assert_type(rolling.rolling_vals(xx.to_dataset(), mom=3, window=3), xr.Dataset)
    if MYPY_ONLY:
        g: ArrayLike | xr.DataArray | xr.Dataset = xany
        assert_type(
            rolling.rolling_vals(g, mom=3, window=3),
            Union[xr.DataArray, xr.Dataset, NDArrayAny],
        )


def typecheck_rolling_exp_data(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    outany: NDArrayAny,
    xany: NDArrayAny,
) -> None:
    assert_type(rolling.rolling_exp_data(x32, mom_ndim=1, alpha=0.2), NDArrayFloat32)
    assert_type(rolling.rolling_exp_data(x64, mom_ndim=1, alpha=0.2), NDArrayFloat64)

    assert_type(
        rolling.rolling_exp_data(x32, mom_ndim=1, alpha=0.2, dtype=np.float64),
        NDArrayFloat64,
    )
    assert_type(
        rolling.rolling_exp_data(x64, mom_ndim=1, alpha=0.2, dtype=np.float32),
        NDArrayFloat32,
    )

    assert_type(
        rolling.rolling_exp_data(x32, mom_ndim=1, alpha=0.2, out=out64),
        NDArrayFloat64,
    )
    assert_type(
        rolling.rolling_exp_data(x64, mom_ndim=1, alpha=0.2, out=out32),
        NDArrayFloat32,
    )

    assert_type(
        rolling.rolling_exp_data(
            x32, mom_ndim=1, alpha=0.2, out=out64, dtype=np.float32
        ),
        NDArrayFloat64,
    )
    assert_type(
        rolling.rolling_exp_data(
            x64, mom_ndim=1, alpha=0.2, out=out32, dtype=np.float64
        ),
        NDArrayFloat32,
    )
    assert_type(
        rolling.rolling_exp_data(
            x64, mom_ndim=1, alpha=0.2, out=outany, dtype=np.float64
        ),
        NDArrayAny,
    )

    assert_type(xany, NDArrayAny)

    # Would like this to default to np.float64
    assert_type(rolling.rolling_exp_data(xany, mom_ndim=1, alpha=0.2), NDArrayAny)
    assert_type(
        rolling.rolling_exp_data(xany, mom_ndim=1, alpha=0.2, dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        rolling.rolling_exp_data([1, 2, 3], mom_ndim=1, alpha=0.2, dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        rolling.rolling_exp_data([1, 2, 3], mom_ndim=1, alpha=0.2, dtype=np.float64),
        NDArrayFloat64,
    )

    assert_type(
        rolling.rolling_exp_data([1, 2, 3], mom_ndim=1, alpha=0.2, dtype="f8"),
        NDArrayAny,
    )
    assert_type(
        rolling.rolling_exp_data(x32, mom_ndim=1, alpha=0.2, dtype="f8"),
        NDArrayAny,
    )
    assert_type(
        rolling.rolling_exp_data(xany, mom_ndim=1, alpha=0.2, dtype="f8"),
        NDArrayAny,
    )

    assert_type(
        rolling.rolling_exp_data([1.0, 2.0, 3.0], mom_ndim=1, alpha=0.2),
        NDArrayAny,
    )

    xx = xr.DataArray(x32)
    assert_type(rolling.rolling_exp_data(xx, mom_ndim=1, alpha=0.2), xr.DataArray)
    assert_type(
        rolling.rolling_exp_data(xx.to_dataset(), mom_ndim=1, alpha=0.2), xr.Dataset
    )
    if MYPY_ONLY:
        g: ArrayLike | xr.DataArray | xr.Dataset = xany
        assert_type(
            rolling.rolling_exp_data(g, mom_ndim=1, alpha=0.2),
            Union[xr.DataArray, xr.Dataset, NDArrayAny],
        )


def typecheck_rolling_exp_vals(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    outany: NDArrayAny,
    xany: NDArrayAny,
) -> None:
    assert_type(rolling.rolling_exp_vals(x32, mom=3, alpha=0.2), NDArrayFloat32)
    assert_type(rolling.rolling_exp_vals(x64, mom=3, alpha=0.2), NDArrayFloat64)

    assert_type(
        rolling.rolling_exp_vals(x32, mom=3, alpha=0.2, dtype=np.float64),
        NDArrayFloat64,
    )
    assert_type(
        rolling.rolling_exp_vals(x64, mom=3, alpha=0.2, dtype=np.float32),
        NDArrayFloat32,
    )

    assert_type(
        rolling.rolling_exp_vals(x32, mom=3, alpha=0.2, out=out64),
        NDArrayFloat64,
    )
    assert_type(
        rolling.rolling_exp_vals(x64, mom=3, alpha=0.2, out=out32),
        NDArrayFloat32,
    )

    assert_type(
        rolling.rolling_exp_vals(x32, mom=3, alpha=0.2, out=out64, dtype=np.float32),
        NDArrayFloat64,
    )
    assert_type(
        rolling.rolling_exp_vals(x64, mom=3, alpha=0.2, out=out32, dtype=np.float64),
        NDArrayFloat32,
    )
    assert_type(
        rolling.rolling_exp_vals(x64, mom=3, alpha=0.2, out=outany, dtype=np.float64),
        NDArrayAny,
    )

    assert_type(xany, NDArrayAny)

    # Would like this to default to np.float64
    assert_type(rolling.rolling_exp_vals(xany, mom=3, alpha=0.2), NDArrayAny)
    assert_type(
        rolling.rolling_exp_vals(xany, mom=3, alpha=0.2, dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        rolling.rolling_exp_vals([1, 2, 3], mom=3, alpha=0.2, dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        rolling.rolling_exp_vals([1, 2, 3], mom=3, alpha=0.2, dtype=np.float64),
        NDArrayFloat64,
    )

    assert_type(
        rolling.rolling_exp_vals([1, 2, 3], mom=3, alpha=0.2, dtype="f8"),
        NDArrayAny,
    )
    assert_type(
        rolling.rolling_exp_vals(x32, mom=3, alpha=0.2, dtype="f8"),
        NDArrayAny,
    )
    assert_type(
        rolling.rolling_exp_vals(xany, mom=3, alpha=0.2, dtype="f8"),
        NDArrayAny,
    )

    assert_type(rolling.rolling_exp_vals([1.0, 2.0, 3.0], mom=3, alpha=0.2), NDArrayAny)

    xx = xr.DataArray(x32)
    assert_type(rolling.rolling_exp_vals(xx, mom=3, alpha=0.2), xr.DataArray)
    assert_type(rolling.rolling_exp_vals(xx.to_dataset(), mom=3, alpha=0.2), xr.Dataset)

    if MYPY_ONLY:
        g: ArrayLike | xr.DataArray | xr.Dataset = xany
        assert_type(
            rolling.rolling_exp_vals(g, mom=3, alpha=0.2),
            Union[xr.DataArray, xr.Dataset, NDArrayAny],
        )

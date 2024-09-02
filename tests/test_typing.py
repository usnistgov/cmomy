from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import xarray as xr

import cmomy
from cmomy import CentralMomentsArray, CentralMomentsXArray, convert, rolling
from cmomy.reduction import (
    reduce_data,
    reduce_data_grouped,
    reduce_data_indexed,
    reduce_vals,
)
from cmomy.resample import resample_vals

MYPY_ONLY = True

if sys.version_info < (3, 11):
    from typing_extensions import assert_type
else:
    from typing import assert_type

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from cmomy.core.typing import (  # noqa: TCH004  # keep this in typecheck block...
        CentralMomentsArrayAny,
        CentralMomentsDataAny,
        CentralMomentsDataArray,
        CentralMomentsDataset,
        Groups,
        NDArrayAny,
        NDArrayInt,
    )

    NDArrayFloat32 = NDArray[np.float32]
    NDArrayFloat64 = NDArray[np.float64]
    CentralMomentsArrayFloat32 = CentralMomentsArray[np.float32]
    CentralMomentsArrayFloat64 = CentralMomentsArray[np.float64]


def typecheck_centralmoments_init(x32: NDArrayFloat32, x64: NDArrayFloat64) -> None:
    assert_type(CentralMomentsArray(x32, mom_ndim=1), CentralMomentsArrayFloat32)
    assert_type(CentralMomentsArray(x64, mom_ndim=1), CentralMomentsArrayFloat64)
    assert_type(
        CentralMomentsArray(x64, mom_ndim=1, dtype=np.float32),
        CentralMomentsArrayFloat32,
    )
    assert_type(
        CentralMomentsArray(x32, mom_ndim=1, dtype=np.float64),
        CentralMomentsArrayFloat64,
    )
    assert_type(CentralMomentsArray([1, 2, 3], mom_ndim=1), CentralMomentsArrayAny)

    assert_type(
        CentralMomentsArray([1, 2, 3], mom_ndim=1, dtype=np.float32),
        CentralMomentsArrayFloat32,
    )
    assert_type(
        CentralMomentsArray([1, 2, 3], mom_ndim=1, dtype=np.dtype("f8")),
        CentralMomentsArrayFloat64,
    )
    assert_type(
        CentralMomentsArray([1, 2, 3], mom_ndim=1, dtype="f8"),
        CentralMomentsArrayAny,
    )

    assert_type(CentralMomentsArrayFloat32([1, 2, 3]), CentralMomentsArrayFloat32)
    assert_type(CentralMomentsArrayFloat64([1, 2, 3]), CentralMomentsArrayFloat64)


def typecheck_xcentralmoments_init(data: xr.DataArray) -> None:
    dataAny: Any = data
    assert_type(CentralMomentsXArray(data, mom_ndim=1), CentralMomentsDataArray)
    assert_type(CentralMomentsXArray(dataAny, mom_ndim=1), CentralMomentsDataAny)
    assert_type(CentralMomentsXArray(dataAny, mom_ndim=1), CentralMomentsDataAny)
    assert_type(
        CentralMomentsXArray(data.to_dataset(), mom_ndim=1), CentralMomentsDataset
    )


def typecheck_xcentral_to_dataarray_dataset(da: xr.DataArray) -> None:
    ds = da.to_dataset()
    ca = CentralMomentsXArray(da)
    cs = CentralMomentsXArray(ds)

    assert_type(ca, CentralMomentsDataArray)
    assert_type(cs, CentralMomentsDataset)

    assert_type(ca.to_dataset(), CentralMomentsDataset)
    assert_type(cs.to_dataarray(), CentralMomentsDataArray)


def typecheck_cls_astype(x32: NDArrayFloat32, x64: NDArrayFloat64) -> None:
    c32 = CentralMomentsArray(x32, mom_ndim=1)
    c64 = CentralMomentsArray(x64, mom_ndim=1)

    assert_type(c32, CentralMomentsArrayFloat32)
    assert_type(c64, CentralMomentsArrayFloat64)
    assert_type(c32.astype(np.float64), CentralMomentsArrayFloat64)
    assert_type(c64.astype(np.float32), CentralMomentsArrayFloat32)
    assert_type(c32.astype(np.dtype("f8")), CentralMomentsArrayFloat64)
    assert_type(c64.astype(np.dtype("f4")), CentralMomentsArrayFloat32)
    assert_type(c64.astype("f4"), CentralMomentsArrayAny)

    assert_type(c32.astype(None), CentralMomentsArrayFloat64)
    assert_type(c64.astype(None), CentralMomentsArrayFloat64)
    assert_type(c32.astype(c32.dtype), CentralMomentsArrayFloat32)
    assert_type(c64.astype(c64.dtype), CentralMomentsArrayFloat64)

    cx = c32.to_x()
    cs = cx.to_dataset()
    cany: CentralMomentsDataAny = cs

    assert_type(cx.astype(np.float64), CentralMomentsDataArray)
    assert_type(cs.astype(np.float64), CentralMomentsDataset)
    assert_type(cany.astype("f4"), CentralMomentsDataAny)

    cc = cx.to_c()
    assert_type(cc, CentralMomentsArrayAny)
    assert_type(cc.astype(np.float32), CentralMomentsArrayFloat32)


def typecheck_reduce_vals(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    xany: NDArrayAny,
) -> None:
    assert_type(reduce_vals(x32, mom=3), NDArrayFloat32)
    assert_type(reduce_vals(x64, mom=3), NDArrayFloat64)

    # dtype override x
    assert_type(reduce_vals(x32, mom=3, dtype=np.float64), NDArrayFloat64)
    assert_type(reduce_vals(x64, mom=3, dtype=np.float32), NDArrayFloat32)

    # out override x
    assert_type(reduce_vals(x32, mom=3, out=out64), NDArrayFloat64)
    assert_type(reduce_vals(x64, mom=3, out=out32), NDArrayFloat32)

    # out override x and dtype
    assert_type(reduce_vals(x32, mom=3, out=out64, dtype=np.float32), NDArrayFloat64)
    assert_type(reduce_vals(x64, mom=3, out=out32, dtype=np.float64), NDArrayFloat32)

    assert_type(xany, NDArrayAny)

    # Would like this to default to np.float64
    assert_type(reduce_vals(xany, mom=3), NDArrayAny)
    assert_type(reduce_vals(xany, mom=3, dtype=np.float32), NDArrayFloat32)
    assert_type(reduce_vals([1, 2, 3], mom=3, dtype=np.float32), NDArrayFloat32)
    assert_type(reduce_vals([1, 2, 3], mom=3, dtype=np.float64), NDArrayFloat64)

    assert_type(reduce_vals([1, 2, 3], mom=3, dtype=np.dtype("f8")), NDArrayFloat64)

    # unknown dtype
    assert_type(reduce_vals(x32, mom=3, dtype="f8"), NDArrayAny)
    assert_type(reduce_vals([1, 2, 3], mom=3, dtype="f8"), NDArrayAny)
    assert_type(reduce_vals([1.0, 2.0, 3.0], mom=3), NDArrayAny)

    xx = xr.DataArray(x32)
    assert_type(reduce_vals(xx, mom=3), xr.DataArray)
    assert_type(reduce_vals(xx, xx, mom=(3, 3)), xr.DataArray)
    assert_type(reduce_vals(xx, xx, weight=xx, mom=(3, 3)), xr.DataArray)

    ds = xx.to_dataset()
    assert_type(reduce_vals(ds, mom=3), xr.Dataset)
    assert_type(reduce_vals(ds, xx, mom=(3, 3)), xr.Dataset)
    assert_type(reduce_vals(ds, ds, mom=(3, 3)), xr.Dataset)
    assert_type(reduce_vals(ds, ds, weight=xx, mom=(3, 3)), xr.Dataset)

    if MYPY_ONLY:
        # TODO(wpk): would love to figure out how to get pyright to give this
        # as the fallback overload...
        g: ArrayLike | xr.DataArray | xr.Dataset = xany
        assert_type(reduce_vals(g, mom=3), Union[xr.DataArray, xr.Dataset, NDArrayAny])


def typecheck_reduce_data(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    xany: NDArrayAny,
) -> None:
    assert_type(reduce_data(x32, mom_ndim=1), NDArrayFloat32)
    assert_type(reduce_data(x64, mom_ndim=1), NDArrayFloat64)

    assert_type(reduce_data(xany, mom_ndim=1), NDArrayAny)

    assert_type(reduce_data(x32, mom_ndim=1, dtype=np.float64), NDArrayFloat64)
    assert_type(reduce_data(x64, mom_ndim=1, dtype=np.float32), NDArrayFloat32)

    assert_type(reduce_data(x32, mom_ndim=1, out=out64), NDArrayFloat64)
    assert_type(reduce_data(x64, mom_ndim=1, out=out32), NDArrayFloat32)

    assert_type(
        reduce_data(x32, mom_ndim=1, out=out64, dtype=np.float32),
        NDArrayFloat64,
    )
    assert_type(
        reduce_data(x64, mom_ndim=1, out=out32, dtype=np.float64),
        NDArrayFloat32,
    )

    assert_type(xany, NDArrayAny)

    # Would like this to default to np.float64
    assert_type(reduce_data(xany, mom_ndim=1), NDArrayAny)
    assert_type(reduce_data(xany, mom_ndim=1, dtype=np.float32), NDArrayFloat32)
    assert_type(reduce_data([1, 2, 3], mom_ndim=1, dtype=np.float32), NDArrayFloat32)
    assert_type(reduce_data([1, 2, 3], mom_ndim=1, dtype=np.float64), NDArrayFloat64)

    assert_type(
        reduce_data(x32, mom_ndim=1, dtype="f8"),
        NDArrayAny,
    )
    assert_type(
        reduce_data([1, 2, 3], mom_ndim=1, dtype="f8"),
        NDArrayAny,
    )
    assert_type(reduce_data([1.0, 2.0, 3.0], mom_ndim=1), NDArrayAny)

    xx = xr.DataArray(x32)
    assert_type(reduce_data(xx, mom_ndim=1), xr.DataArray)

    ds = xx.to_dataset(name="hello")
    assert_type(reduce_data(ds, mom_ndim=1), xr.Dataset)

    if MYPY_ONLY:
        g: ArrayLike | xr.DataArray | xr.Dataset = xany
        assert_type(
            reduce_data(g, mom_ndim=1),
            Union[xr.DataArray, xr.Dataset, NDArrayAny],
        )


def typecheck_convert(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    xany: NDArrayAny,
) -> None:
    assert_type(convert.moments_type(x32, mom_ndim=1), NDArrayFloat32)
    assert_type(convert.moments_type(x64, mom_ndim=1), NDArrayFloat64)

    assert_type(convert.moments_type(x32, mom_ndim=1, dtype=np.float64), NDArrayFloat64)
    assert_type(convert.moments_type(x64, mom_ndim=1, dtype=np.float32), NDArrayFloat32)

    assert_type(convert.moments_type(x32, mom_ndim=1, out=out64), NDArrayFloat64)
    assert_type(convert.moments_type(x64, mom_ndim=1, out=out32), NDArrayFloat32)

    assert_type(
        convert.moments_type(x32, mom_ndim=1, out=out64, dtype=np.float32),
        NDArrayFloat64,
    )
    assert_type(
        convert.moments_type(x64, mom_ndim=1, out=out32, dtype=np.float64),
        NDArrayFloat32,
    )

    assert_type(xany, NDArrayAny)

    # Would like this to default to np.float64
    assert_type(convert.moments_type(xany, mom_ndim=1), NDArrayAny)
    assert_type(
        convert.moments_type(xany, mom_ndim=1, dtype=np.float32), NDArrayFloat32
    )
    assert_type(
        convert.moments_type([1, 2, 3], mom_ndim=1, dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        convert.moments_type([1, 2, 3], mom_ndim=1, dtype=np.float64),
        NDArrayFloat64,
    )

    assert_type(
        convert.moments_type([1, 2, 3], mom_ndim=1, dtype="f8"),
        NDArrayAny,
    )
    assert_type(
        convert.moments_type(x32, mom_ndim=1, dtype="f8"),
        NDArrayAny,
    )
    assert_type(
        convert.moments_type(xany, mom_ndim=1, dtype="f8"),
        NDArrayAny,
    )

    assert_type(convert.moments_type([1.0, 2.0, 3.0], mom_ndim=1), NDArrayAny)

    xx = xr.DataArray(x32)
    assert_type(convert.moments_type(xx, mom_ndim=1), xr.DataArray)
    assert_type(convert.moments_type(xx.to_dataset(), mom_ndim=1), xr.Dataset)
    if MYPY_ONLY:
        g: ArrayLike | xr.DataArray | xr.Dataset = xany
        assert_type(
            convert.moments_type(g, mom_ndim=1),
            Union[xr.DataArray, xr.Dataset, NDArrayAny],
        )


def typecheck_moveaxis(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    xint: NDArrayInt,
    xany: NDArrayAny,
) -> None:
    assert_type(x32, NDArrayFloat32)
    assert_type(cmomy.moveaxis(x32, mom_ndim=1), NDArrayFloat32)
    assert_type(cmomy.moveaxis(x64, mom_ndim=1), NDArrayFloat64)
    assert_type(cmomy.moveaxis(xint, mom_ndim=1), NDArrayInt)
    assert_type(cmomy.moveaxis(xany, mom_ndim=1), NDArrayAny)
    assert_type(cmomy.moveaxis(xr.DataArray(xany), mom_ndim=1), xr.DataArray)


def typecheck_select_moment(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    xint: NDArrayInt,
    xany: NDArrayAny,
) -> None:
    assert_type(cmomy.utils.select_moment(x32, "weight", mom_ndim=1), NDArrayFloat32)
    assert_type(cmomy.utils.select_moment(x64, "weight", mom_ndim=1), NDArrayFloat64)
    assert_type(cmomy.utils.select_moment(xint, "weight", mom_ndim=1), NDArrayInt)
    assert_type(cmomy.utils.select_moment(xany, "weight", mom_ndim=1), NDArrayAny)
    assert_type(
        cmomy.utils.select_moment(xr.DataArray(xany), "weight", mom_ndim=1),
        xr.DataArray,
    )
    assert_type(
        cmomy.utils.select_moment(
            xr.DataArray(xany).to_dataset(), "weight", mom_ndim=1
        ),
        xr.Dataset,
    )


def typecheck_cumulative(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    outany: NDArrayAny,
    xany: NDArrayAny,
) -> None:
    assert_type(convert.cumulative(x32, mom_ndim=1), NDArrayFloat32)
    assert_type(convert.cumulative(x64, mom_ndim=1), NDArrayFloat64)

    assert_type(convert.cumulative(x32, mom_ndim=1, dtype=np.float64), NDArrayFloat64)
    assert_type(convert.cumulative(x64, mom_ndim=1, dtype=np.float32), NDArrayFloat32)

    assert_type(convert.cumulative(x32, mom_ndim=1, out=out64), NDArrayFloat64)
    assert_type(convert.cumulative(x64, mom_ndim=1, out=out32), NDArrayFloat32)

    assert_type(
        convert.cumulative(x32, mom_ndim=1, out=out64, dtype=np.float32),
        NDArrayFloat64,
    )
    assert_type(
        convert.cumulative(x64, mom_ndim=1, out=out32, dtype=np.float64),
        NDArrayFloat32,
    )

    assert_type(
        convert.cumulative(x32, mom_ndim=1, out=outany, dtype=np.float32),
        NDArrayAny,
    )

    assert_type(xany, NDArrayAny)

    # Would like this to default to np.float64
    assert_type(convert.cumulative(xany, mom_ndim=1), NDArrayAny)
    assert_type(convert.cumulative(xany, mom_ndim=1, dtype=np.float32), NDArrayFloat32)
    assert_type(
        convert.cumulative([1, 2, 3], mom_ndim=1, dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        convert.cumulative([1, 2, 3], mom_ndim=1, dtype=np.float64),
        NDArrayFloat64,
    )

    assert_type(
        convert.cumulative([1, 2, 3], mom_ndim=1, dtype="f8"),
        NDArrayAny,
    )
    assert_type(
        convert.cumulative(x32, mom_ndim=1, dtype="f8"),
        NDArrayAny,
    )
    assert_type(
        convert.cumulative(xany, mom_ndim=1, dtype="f8"),
        NDArrayAny,
    )

    assert_type(convert.cumulative([1.0, 2.0, 3.0], mom_ndim=1), NDArrayAny)

    xx = xr.DataArray(x32)
    assert_type(convert.cumulative(xx, mom_ndim=1), xr.DataArray)
    assert_type(convert.cumulative(xx.to_dataset(), mom_ndim=1), xr.Dataset)

    if MYPY_ONLY:
        g: ArrayLike | xr.DataArray | xr.Dataset = xany
        assert_type(
            convert.cumulative(g, mom_ndim=1),
            Union[xr.DataArray, xr.Dataset, NDArrayAny],
        )


def typecheck_vals_to_data(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    outany: NDArrayAny,
    xany: NDArrayAny,
) -> None:
    assert_type(cmomy.utils.vals_to_data(x32, mom=1), NDArrayFloat32)
    assert_type(cmomy.utils.vals_to_data(x64, mom=1), NDArrayFloat64)

    assert_type(cmomy.utils.vals_to_data(x32, mom=1, dtype=np.float64), NDArrayFloat64)
    assert_type(cmomy.utils.vals_to_data(x64, mom=1, dtype=np.float32), NDArrayFloat32)

    assert_type(cmomy.utils.vals_to_data(x32, mom=1, out=out64), NDArrayFloat64)
    assert_type(cmomy.utils.vals_to_data(x64, mom=1, out=out32), NDArrayFloat32)

    assert_type(
        cmomy.utils.vals_to_data(x32, mom=1, out=out64, dtype=np.float32),
        NDArrayFloat64,
    )
    assert_type(
        cmomy.utils.vals_to_data(x64, mom=1, out=out32, dtype=np.float64),
        NDArrayFloat32,
    )

    assert_type(
        cmomy.utils.vals_to_data(x32, mom=1, out=outany, dtype=np.float32),
        NDArrayAny,
    )

    assert_type(xany, NDArrayAny)

    # Would like this to default to np.float64
    assert_type(cmomy.utils.vals_to_data(xany, mom=1), NDArrayAny)
    assert_type(cmomy.utils.vals_to_data(xany, mom=1, dtype=np.float32), NDArrayFloat32)
    assert_type(
        cmomy.utils.vals_to_data([1, 2, 3], mom=1, dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        cmomy.utils.vals_to_data([1, 2, 3], mom=1, dtype=np.float64),
        NDArrayFloat64,
    )

    assert_type(
        cmomy.utils.vals_to_data([1, 2, 3], mom=1, dtype="f8"),
        NDArrayAny,
    )
    assert_type(
        cmomy.utils.vals_to_data(x32, mom=1, dtype="f8"),
        NDArrayAny,
    )
    assert_type(
        cmomy.utils.vals_to_data(xany, mom=1, dtype="f8"),
        NDArrayAny,
    )

    assert_type(cmomy.utils.vals_to_data([1.0, 2.0, 3.0], mom=1), NDArrayAny)

    xx = xr.DataArray(x32)
    assert_type(cmomy.utils.vals_to_data(xx, mom=1), xr.DataArray)
    assert_type(cmomy.utils.vals_to_data(xx.to_dataset(), mom=1), xr.Dataset)

    if MYPY_ONLY:
        g: ArrayLike | xr.DataArray | xr.Dataset = xany
        assert_type(
            cmomy.utils.vals_to_data(g, mom=1),
            Union[xr.DataArray, xr.Dataset, NDArrayAny],
        )


def typecheck_convert_moments_to_comoments(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    xany: NDArrayAny,
) -> None:
    assert_type(convert.moments_to_comoments(x32, mom=(2, -1)), NDArrayFloat32)
    assert_type(convert.moments_to_comoments(x64, mom=(2, -1)), NDArrayFloat64)

    assert_type(
        convert.moments_to_comoments(x32, mom=(2, -1), dtype=np.float64),
        NDArrayFloat64,
    )
    assert_type(
        convert.moments_to_comoments(x64, mom=(2, -1), dtype=np.float32),
        NDArrayFloat32,
    )

    assert_type(xany, NDArrayAny)

    # Would like this to default to np.float64
    assert_type(convert.moments_to_comoments(xany, mom=(2, -1)), NDArrayAny)
    assert_type(
        convert.moments_to_comoments(xany, mom=(2, -1), dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        convert.moments_to_comoments([1, 2, 3, 4], mom=(2, -1), dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        convert.moments_to_comoments([1, 2, 3, 4], mom=(2, -1), dtype=np.float64),
        NDArrayFloat64,
    )

    assert_type(
        convert.moments_to_comoments([1, 2, 3, 4], mom=(2, -1), dtype="f8"),
        NDArrayAny,
    )
    assert_type(
        convert.moments_to_comoments(x32, mom=(2, -1), dtype="f8"),
        NDArrayAny,
    )
    assert_type(
        convert.moments_to_comoments(xany, mom=(2, -1), dtype="f8"),
        NDArrayAny,
    )

    assert_type(convert.moments_to_comoments([1.0, 2.0, 3.0], mom=(2, -1)), NDArrayAny)

    xx = xr.DataArray(x32)
    assert_type(convert.moments_to_comoments(xx, mom=(2, -1)), xr.DataArray)
    assert_type(convert.moments_to_comoments(xx.to_dataset(), mom=(2, -1)), xr.Dataset)

    if MYPY_ONLY:
        g: ArrayLike | xr.DataArray | xr.Dataset = xany
        assert_type(
            convert.moments_to_comoments(g, mom=(2, -1)),
            Union[xr.DataArray, xr.Dataset, NDArrayAny],
        )


def typecheck_moments_to_comoments(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    xany: NDArrayAny,
) -> None:
    c32 = CentralMomentsArray(x32, mom_ndim=1)
    c64 = CentralMomentsArray(x64, mom_ndim=1)
    cAny = CentralMomentsArray(xany, mom_ndim=1)

    assert_type(c32.moments_to_comoments(mom=(2, -1)), CentralMomentsArrayFloat32)
    assert_type(c64.moments_to_comoments(mom=(2, -1)), CentralMomentsArrayFloat64)
    assert_type(cAny.moments_to_comoments(mom=(2, -1)), CentralMomentsArrayAny)

    assert_type(
        c32.to_x().moments_to_comoments(mom=(2, -1)),
        CentralMomentsDataArray,
    )
    assert_type(
        c32.to_x().to_dataset().moments_to_comoments(mom=(2, -1)),
        CentralMomentsDataset,
    )


def typecheck_assign_moment_central(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    xany: NDArrayAny,
) -> None:
    c32 = CentralMomentsArray(x32, mom_ndim=1)
    c64 = CentralMomentsArray(x64, mom_ndim=1)
    cAny = CentralMomentsArray(xany, mom_ndim=1)

    assert_type(c32.assign_moment(weight=1.0), CentralMomentsArrayFloat32)
    assert_type(c64.assign_moment(weight=1.0), CentralMomentsArrayFloat64)
    assert_type(cAny.assign_moment(weight=1.0), CentralMomentsArrayAny)

    assert_type(
        c32.to_x().assign_moment(weight=1.0),
        CentralMomentsDataArray,
    )
    assert_type(
        c64.to_x().to_dataset().assign_moment(weight=1.0),
        CentralMomentsDataset,
    )

    cxany: CentralMomentsDataAny = c32.to_x()
    assert_type(cxany.assign_moment(weight=1.0), CentralMomentsDataAny)


def typecheck_concat(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    xany: NDArrayAny,
) -> None:
    c32 = CentralMomentsArray(x32)
    c64 = CentralMomentsArray(x64)

    dx = xr.DataArray(x64, dims=["a", "b", "mom"])
    dc = CentralMomentsXArray(dx)
    dcs = dc.to_dataset()

    assert_type(convert.concat((x32, x32), axis=0), NDArrayFloat32)
    assert_type(convert.concat((x64, x64), axis=0), NDArrayFloat64)

    assert_type(convert.concat((dx, dx), axis=0), xr.DataArray)

    assert_type(convert.concat((c32, c32), axis=0), CentralMomentsArrayFloat32)

    assert_type(convert.concat((c64, c64), axis=0), CentralMomentsArrayFloat64)
    assert_type(
        convert.concat((dc, dc), axis=0),
        CentralMomentsDataArray,
    )
    assert_type(
        convert.concat((dcs, dcs), axis=0),
        CentralMomentsDataset,
    )

    dxany = xr.DataArray(xany)
    assert_type(convert.concat((xany, xany), axis=0), NDArrayAny)
    assert_type(convert.concat((dxany, dxany), axis=0), xr.DataArray)

    cany = CentralMomentsArray(xany)
    assert_type(convert.concat((cany, cany), axis=0), CentralMomentsArrayAny)

    dcany = CentralMomentsXArray(dxany)
    assert_type(convert.concat((dcany, dcany), axis=0), CentralMomentsDataArray)


# if mypy properly supports partial will add this...
# def typecheck_convert_central_to_raw() -> None:


def typecheck_reduce_data_grouped(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    by: NDArrayInt,
    xany: NDArrayAny,
) -> None:
    assert_type(reduce_data_grouped(x32, mom_ndim=1, by=by), NDArrayFloat32)
    assert_type(reduce_data_grouped(x64, mom_ndim=1, by=by), NDArrayFloat64)

    assert_type(
        reduce_data_grouped(x32, mom_ndim=1, by=by, dtype=np.float64),
        NDArrayFloat64,
    )
    assert_type(
        reduce_data_grouped(x64, mom_ndim=1, by=by, dtype=np.float32),
        NDArrayFloat32,
    )

    assert_type(reduce_data_grouped(x32, mom_ndim=1, by=by, out=out64), NDArrayFloat64)
    assert_type(reduce_data_grouped(x64, mom_ndim=1, by=by, out=out32), NDArrayFloat32)

    assert_type(
        reduce_data_grouped(x32, mom_ndim=1, by=by, out=out64, dtype=np.float32),
        NDArrayFloat64,
    )
    assert_type(
        reduce_data_grouped(x64, mom_ndim=1, by=by, out=out32, dtype=np.float64),
        NDArrayFloat32,
    )

    assert_type(xany, NDArrayAny)

    # Would like this to default to np.float64
    assert_type(reduce_data_grouped(xany, mom_ndim=1, by=by), NDArrayAny)
    assert_type(
        reduce_data_grouped(xany, mom_ndim=1, by=by, dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        reduce_data_grouped([1, 2, 3], mom_ndim=1, by=by, dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        reduce_data_grouped([1, 2, 3], mom_ndim=1, by=by, dtype=np.float64),
        NDArrayFloat64,
    )

    assert_type(reduce_data_grouped(x64, mom_ndim=1, by=by, dtype="f8"), NDArrayAny)
    assert_type(reduce_data_grouped([1.0, 2.0, 3.0], mom_ndim=1, by=by), NDArrayAny)

    xx = xr.DataArray(x32)
    assert_type(reduce_data_grouped(xx, mom_ndim=1, by=by), xr.DataArray)

    assert_type(reduce_data_grouped(xx.to_dataset(), mom_ndim=1, by=by), xr.Dataset)

    if MYPY_ONLY:
        g: ArrayLike | xr.DataArray | xr.Dataset = xany
        assert_type(
            reduce_data_grouped(g, mom_ndim=1, by=by),
            Union[xr.DataArray, xr.Dataset, NDArrayAny],
        )


def typecheck_reduce_data_indexed(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    xany: NDArrayAny,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    index: NDArrayInt,
    group_start: NDArrayInt,
    group_end: NDArrayInt,
) -> None:
    assert_type(
        reduce_data_indexed(
            x32,
            mom_ndim=1,
            index=index,
            group_start=group_start,
            group_end=group_end,
        ),
        NDArrayFloat32,
    )
    assert_type(
        reduce_data_indexed(
            x64,
            mom_ndim=1,
            index=index,
            group_start=group_start,
            group_end=group_end,
        ),
        NDArrayFloat64,
    )

    assert_type(
        reduce_data_indexed(
            x32,
            mom_ndim=1,
            index=index,
            group_start=group_start,
            group_end=group_end,
            dtype=np.float64,
        ),
        NDArrayFloat64,
    )
    assert_type(
        reduce_data_indexed(
            x64,
            mom_ndim=1,
            index=index,
            group_start=group_start,
            group_end=group_end,
            dtype=np.float32,
        ),
        NDArrayFloat32,
    )

    assert_type(
        reduce_data_indexed(
            x32,
            mom_ndim=1,
            index=index,
            group_start=group_start,
            group_end=group_end,
            out=out64,
        ),
        NDArrayFloat64,
    )
    assert_type(
        reduce_data_indexed(
            x64,
            mom_ndim=1,
            index=index,
            group_start=group_start,
            group_end=group_end,
            out=out32,
        ),
        NDArrayFloat32,
    )

    assert_type(
        reduce_data_indexed(
            x32,
            mom_ndim=1,
            index=index,
            group_start=group_start,
            group_end=group_end,
            out=out64,
            dtype=np.float32,
        ),
        NDArrayFloat64,
    )
    assert_type(
        reduce_data_indexed(
            x64,
            mom_ndim=1,
            index=index,
            group_start=group_start,
            group_end=group_end,
            out=out32,
            dtype=np.float64,
        ),
        NDArrayFloat32,
    )

    assert_type(xany, NDArrayAny)

    # Would like this to default to np.float64
    assert_type(
        reduce_data_indexed(
            xany,
            mom_ndim=1,
            index=index,
            group_start=group_start,
            group_end=group_end,
        ),
        NDArrayAny,
    )
    assert_type(
        reduce_data_indexed(
            xany,
            mom_ndim=1,
            index=index,
            group_start=group_start,
            group_end=group_end,
            dtype=np.float32,
        ),
        NDArrayFloat32,
    )
    assert_type(
        reduce_data_indexed(
            [1, 2, 3],
            mom_ndim=1,
            index=index,
            group_start=group_start,
            group_end=group_end,
            dtype=np.float32,
        ),
        NDArrayFloat32,
    )
    assert_type(
        reduce_data_indexed(
            [1, 2, 3],
            mom_ndim=1,
            index=index,
            group_start=group_start,
            group_end=group_end,
            dtype=np.float64,
        ),
        NDArrayFloat64,
    )

    assert_type(
        reduce_data_indexed(
            [1.0, 2.0, 3.0],
            mom_ndim=1,
            index=index,
            group_start=group_start,
            group_end=group_end,
        ),
        NDArrayAny,
    )

    assert_type(
        reduce_data_indexed(
            x32,
            mom_ndim=1,
            index=index,
            group_start=group_start,
            group_end=group_end,
            dtype="f8",
        ),
        NDArrayAny,
    )

    xx = xr.DataArray(x32)
    assert_type(
        reduce_data_indexed(
            xx,
            mom_ndim=1,
            index=index,
            group_start=group_start,
            group_end=group_end,
        ),
        xr.DataArray,
    )
    assert_type(
        reduce_data_indexed(
            xx.to_dataset(),
            mom_ndim=1,
            index=index,
            group_start=group_start,
            group_end=group_end,
        ),
        xr.Dataset,
    )

    if MYPY_ONLY:
        g: ArrayLike | xr.DataArray | xr.Dataset = xany
        assert_type(
            reduce_data_indexed(
                g,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
            ),
            Union[xr.DataArray, xr.Dataset, NDArrayAny],
        )


def typecheck_resample_data(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    freq: NDArrayInt,
    xany: NDArrayAny,
) -> None:
    from cmomy.resample import resample_data

    assert_type(resample_data(x32, freq=freq, mom_ndim=1), NDArrayFloat32)
    assert_type(resample_data(x64, freq=freq, mom_ndim=1), NDArrayFloat64)

    assert_type(
        resample_data(x32, freq=freq, mom_ndim=1, dtype=np.float64),
        NDArrayFloat64,
    )
    assert_type(
        resample_data(x64, freq=freq, mom_ndim=1, dtype=np.float32),
        NDArrayFloat32,
    )

    assert_type(resample_data(x32, freq=freq, mom_ndim=1, out=out64), NDArrayFloat64)
    assert_type(resample_data(x64, freq=freq, mom_ndim=1, out=out32), NDArrayFloat32)

    assert_type(
        resample_data(x32, freq=freq, mom_ndim=1, out=out64, dtype=np.float32),
        NDArrayFloat64,
    )
    assert_type(
        resample_data(x64, freq=freq, mom_ndim=1, out=out32, dtype=np.float64),
        NDArrayFloat32,
    )

    assert_type(xany, NDArrayAny)

    # Would like this to default to np.float64
    assert_type(resample_data(xany, freq=freq, mom_ndim=1), NDArrayAny)
    assert_type(
        resample_data(xany, freq=freq, mom_ndim=1, dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        resample_data([1, 2, 3], freq=freq, mom_ndim=1, dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        resample_data([1, 2, 3], freq=freq, mom_ndim=1, dtype=np.float64),
        NDArrayFloat64,
    )

    assert_type(resample_data([1.0, 2.0, 3.0], freq=freq, mom_ndim=1), NDArrayAny)
    assert_type(resample_data(x32, freq=freq, mom_ndim=1, dtype="f8"), NDArrayAny)

    xx = xr.DataArray(x32)
    assert_type(resample_data(xx, freq=freq, mom_ndim=1), xr.DataArray)
    assert_type(resample_data(xx.to_dataset(), freq=freq, mom_ndim=1), xr.Dataset)

    if MYPY_ONLY:
        g: ArrayLike | xr.DataArray | xr.Dataset = xany
        assert_type(
            resample_data(g, freq=freq, mom_ndim=1),
            Union[xr.DataArray, xr.Dataset, NDArrayAny],
        )


def typecheck_wrapped_resample_and_reduce(
    c32: CentralMomentsArrayFloat32,
    c64: CentralMomentsArrayFloat64,
    ca: CentralMomentsDataArray,
    cs: CentralMomentsDataset,
    cx: CentralMomentsDataAny,
    freq: NDArrayInt,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    outany: NDArrayAny,
) -> None:
    assert_type(c32.resample_and_reduce(axis=0, freq=freq), CentralMomentsArrayFloat32)
    assert_type(c64.resample_and_reduce(axis=0, freq=freq), CentralMomentsArrayFloat64)

    assert_type(
        c32.resample_and_reduce(axis=0, freq=freq, dtype=np.float64),
        CentralMomentsArrayFloat64,
    )
    assert_type(
        c64.resample_and_reduce(axis=0, freq=freq, dtype=np.float32),
        CentralMomentsArrayFloat32,
    )

    assert_type(
        c32.resample_and_reduce(axis=0, freq=freq, out=out64),
        CentralMomentsArrayFloat64,
    )
    assert_type(
        c64.resample_and_reduce(axis=0, freq=freq, out=out32),
        CentralMomentsArrayFloat32,
    )

    assert_type(
        c32.resample_and_reduce(axis=0, freq=freq, dtype=np.float32, out=out64),
        CentralMomentsArrayFloat64,
    )
    assert_type(
        c64.resample_and_reduce(axis=0, freq=freq, dtype=np.float64, out=out32),
        CentralMomentsArrayFloat32,
    )

    assert_type(
        c32.resample_and_reduce(axis=0, freq=freq, out=outany), CentralMomentsArrayAny
    )
    assert_type(
        c32.resample_and_reduce(axis=0, freq=freq, dtype="f4"), CentralMomentsArrayAny
    )

    assert_type(ca.resample_and_reduce(dim="a", nrep=10), CentralMomentsDataArray)
    assert_type(cs.resample_and_reduce(dim="a", nrep=10), CentralMomentsDataset)
    assert_type(cx.resample_and_reduce(dim="a", nrep=10), CentralMomentsDataAny)
    assert_type(
        ca.resample_and_reduce(dim="a", nrep=10, dtype=np.float32),
        CentralMomentsDataArray,
    )
    assert_type(
        ca.resample_and_reduce(dim="a", nrep=10, out=outany), CentralMomentsDataArray
    )


def typecheck_wrapped_jackknife_and_reduce(
    c32: CentralMomentsArrayFloat32,
    c64: CentralMomentsArrayFloat64,
    ca: CentralMomentsDataArray,
    cs: CentralMomentsDataset,
    cx: CentralMomentsDataAny,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    outany: NDArrayAny,
    data32: NDArrayFloat32,
    datalike: ArrayLike,
) -> None:
    assert_type(c32.jackknife_and_reduce(axis=0), CentralMomentsArrayFloat32)
    assert_type(c64.jackknife_and_reduce(axis=0), CentralMomentsArrayFloat64)
    assert_type(
        c32.jackknife_and_reduce(axis=0, data_reduced=data32),
        CentralMomentsArrayFloat32,
    )
    assert_type(
        c64.jackknife_and_reduce(axis=0, data_reduced=datalike),
        CentralMomentsArrayFloat64,
    )
    assert_type(
        c64.jackknife_and_reduce(axis=0, data_reduced=c64.reduce(axis=0)),
        CentralMomentsArrayFloat64,
    )

    assert_type(
        c32.jackknife_and_reduce(axis=0, dtype=np.float64), CentralMomentsArrayFloat64
    )
    assert_type(
        c64.jackknife_and_reduce(axis=0, dtype=np.float32), CentralMomentsArrayFloat32
    )

    assert_type(c32.jackknife_and_reduce(axis=0, out=out64), CentralMomentsArrayFloat64)
    assert_type(c64.jackknife_and_reduce(axis=0, out=out32), CentralMomentsArrayFloat32)

    assert_type(
        c32.jackknife_and_reduce(axis=0, dtype=np.float32, out=out64),
        CentralMomentsArrayFloat64,
    )
    assert_type(
        c64.jackknife_and_reduce(axis=0, dtype=np.float64, out=out32),
        CentralMomentsArrayFloat32,
    )

    assert_type(c32.jackknife_and_reduce(axis=0, out=outany), CentralMomentsArrayAny)
    assert_type(c32.jackknife_and_reduce(axis=0, dtype="f4"), CentralMomentsArrayAny)

    assert_type(ca.jackknife_and_reduce(dim="a"), CentralMomentsDataArray)
    assert_type(
        ca.jackknife_and_reduce(dim="a", data_reduced=ca.reduce(dim="a")),
        CentralMomentsDataArray,
    )
    assert_type(cs.jackknife_and_reduce(dim="a"), CentralMomentsDataset)
    assert_type(cx.jackknife_and_reduce(dim="a"), CentralMomentsDataAny)
    assert_type(
        ca.jackknife_and_reduce(dim="a", dtype=np.float32), CentralMomentsDataArray
    )
    assert_type(ca.jackknife_and_reduce(dim="a", out=outany), CentralMomentsDataArray)


def typecheck_wrapped_reduce(
    c32: CentralMomentsArrayFloat32,
    c64: CentralMomentsArrayFloat64,
    ca: CentralMomentsDataArray,
    cs: CentralMomentsDataset,
    cx: CentralMomentsDataAny,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    outany: NDArrayAny,
    by: Groups,
) -> None:
    assert_type(c32.reduce(axis=0), CentralMomentsArrayFloat32)
    assert_type(c64.reduce(axis=0), CentralMomentsArrayFloat64)
    assert_type(c64.reduce(axis=0, by=by), CentralMomentsArrayFloat64)

    assert_type(c32.reduce(axis=0, dtype=np.float64), CentralMomentsArrayFloat64)
    assert_type(c64.reduce(axis=0, dtype=np.float32), CentralMomentsArrayFloat32)
    assert_type(c64.reduce(axis=0, by=by, dtype=np.float32), CentralMomentsArrayFloat32)

    assert_type(c32.reduce(axis=0, out=out64), CentralMomentsArrayFloat64)
    assert_type(c64.reduce(axis=0, out=out32), CentralMomentsArrayFloat32)

    assert_type(
        c32.reduce(axis=0, dtype=np.float32, out=out64), CentralMomentsArrayFloat64
    )
    assert_type(
        c64.reduce(axis=0, dtype=np.float64, out=out32), CentralMomentsArrayFloat32
    )
    assert_type(
        c64.reduce(axis=0, by=by, dtype=np.float64, out=out32),
        CentralMomentsArrayFloat32,
    )

    assert_type(c32.reduce(axis=0, out=outany), CentralMomentsArrayAny)
    assert_type(c32.reduce(axis=0, dtype="f4"), CentralMomentsArrayAny)

    assert_type(ca.reduce(dim="a"), CentralMomentsDataArray)
    assert_type(cs.reduce(dim="a"), CentralMomentsDataset)
    assert_type(cx.reduce(dim="a"), CentralMomentsDataAny)
    assert_type(ca.reduce(dim="a", dtype=np.float32), CentralMomentsDataArray)
    assert_type(ca.reduce(dim="a", out=outany), CentralMomentsDataArray)


def typecheck_wrapped_cumulative(
    c32: CentralMomentsArrayFloat32,
    c64: CentralMomentsArrayFloat64,
    ca: CentralMomentsDataArray,
    cs: CentralMomentsDataset,
    cx: CentralMomentsDataAny,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    outany: NDArrayAny,
) -> None:
    assert_type(c32.cumulative(axis=0), NDArrayFloat32)
    assert_type(c64.cumulative(axis=0), NDArrayFloat64)

    assert_type(c32.cumulative(axis=0, dtype=np.float64), NDArrayFloat64)
    assert_type(c64.cumulative(axis=0, dtype=np.float32), NDArrayFloat32)

    assert_type(c32.cumulative(axis=0, out=out64), NDArrayFloat64)
    assert_type(c64.cumulative(axis=0, out=out32), NDArrayFloat32)

    assert_type(c32.cumulative(axis=0, dtype=np.float32, out=out64), NDArrayFloat64)
    assert_type(c64.cumulative(axis=0, dtype=np.float64, out=out32), NDArrayFloat32)
    assert_type(
        c64.cumulative(axis=0, dtype=np.float64, out=out32),
        NDArrayFloat32,
    )

    assert_type(c32.cumulative(axis=0, out=outany), NDArrayAny)
    assert_type(c32.cumulative(axis=0, dtype="f4"), NDArrayAny)

    assert_type(ca.cumulative(dim="a"), xr.DataArray)
    assert_type(cs.cumulative(dim="a"), xr.Dataset)
    assert_type(cx.cumulative(dim="a"), Any)
    assert_type(ca.cumulative(dim="a", dtype=np.float32), xr.DataArray)
    assert_type(ca.cumulative(dim="a", out=outany), xr.DataArray)


def typecheck_jackknife_data(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    xany: NDArrayAny,
) -> None:
    from cmomy.resample import jackknife_data

    assert_type(jackknife_data(x32, mom_ndim=1), NDArrayFloat32)
    assert_type(jackknife_data(x64, mom_ndim=1), NDArrayFloat64)

    assert_type(
        jackknife_data(x32, mom_ndim=1, dtype=np.float64),
        NDArrayFloat64,
    )
    assert_type(
        jackknife_data(x64, mom_ndim=1, dtype=np.float32),
        NDArrayFloat32,
    )

    assert_type(jackknife_data(x32, mom_ndim=1, out=out64), NDArrayFloat64)
    assert_type(jackknife_data(x64, mom_ndim=1, out=out32), NDArrayFloat32)

    assert_type(
        jackknife_data(x32, mom_ndim=1, out=out64, dtype=np.float32),
        NDArrayFloat64,
    )
    assert_type(
        jackknife_data(x64, mom_ndim=1, out=out32, dtype=np.float64),
        NDArrayFloat32,
    )

    assert_type(xany, NDArrayAny)

    # Would like this to default to np.float64
    assert_type(jackknife_data(xany, mom_ndim=1), NDArrayAny)
    assert_type(
        jackknife_data(xany, mom_ndim=1, dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        jackknife_data([1, 2, 3], mom_ndim=1, dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        jackknife_data([1, 2, 3], mom_ndim=1, dtype=np.float64),
        NDArrayFloat64,
    )

    assert_type(jackknife_data([1.0, 2.0, 3.0], mom_ndim=1), NDArrayAny)
    assert_type(jackknife_data(x32, mom_ndim=1, dtype="f8"), NDArrayAny)

    xx = xr.DataArray(x32)
    assert_type(jackknife_data(xx, mom_ndim=1), xr.DataArray)
    assert_type(jackknife_data(xx.to_dataset(), mom_ndim=1), xr.Dataset)

    if MYPY_ONLY:
        g: ArrayLike | xr.DataArray | xr.Dataset = xany
        assert_type(
            jackknife_data(g, mom_ndim=1),
            Union[xr.DataArray, xr.Dataset, NDArrayAny],
        )


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


def typecheck_jackknife_vals(
    x32: NDArrayFloat32,
    x64: NDArrayFloat64,
    out32: NDArrayFloat32,
    out64: NDArrayFloat64,
    xany: NDArrayAny,
) -> None:
    from cmomy.resample import jackknife_vals

    assert_type(jackknife_vals(x32, mom=3), NDArrayFloat32)
    assert_type(jackknife_vals(x64, mom=3), NDArrayFloat64)

    assert_type(jackknife_vals(x32, mom=3, dtype=np.float64), NDArrayFloat64)
    assert_type(jackknife_vals(x64, mom=3, dtype=np.float32), NDArrayFloat32)

    assert_type(jackknife_vals(x32, mom=3, out=out64), NDArrayFloat64)
    assert_type(jackknife_vals(x64, mom=3, out=out32), NDArrayFloat32)

    assert_type(
        jackknife_vals(x32, mom=3, out=out64, dtype=np.float32),
        NDArrayFloat64,
    )
    assert_type(
        jackknife_vals(x64, mom=3, out=out32, dtype=np.float64),
        NDArrayFloat32,
    )

    assert_type(xany, NDArrayAny)

    # Would like this to default to np.float64
    assert_type(jackknife_vals(xany, mom=3), NDArrayAny)
    assert_type(jackknife_vals(xany, mom=3, dtype=np.float32), NDArrayFloat32)
    assert_type(
        jackknife_vals([1, 2, 3], mom=3, dtype=np.float32),
        NDArrayFloat32,
    )
    assert_type(
        jackknife_vals([1, 2, 3], mom=3, dtype=np.float64),
        NDArrayFloat64,
    )

    assert_type(jackknife_vals([1.0, 2.0, 3.0], mom=3), NDArrayAny)
    assert_type(jackknife_vals(x32, mom=3, dtype="f8"), NDArrayAny)

    xx = xr.DataArray(x32)
    assert_type(jackknife_vals(xx, mom=3), xr.DataArray)
    assert_type(jackknife_vals(xx.to_dataset(), mom=3), xr.Dataset)

    if MYPY_ONLY:
        g: ArrayLike | xr.DataArray | xr.Dataset = xany
        assert_type(
            jackknife_vals(g, mom=3), Union[xr.DataArray, xr.Dataset, NDArrayAny]
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


def typecheck_from_raw(
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

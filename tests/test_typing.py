from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from cmomy import CentralMoments, xCentralMoments
from cmomy.resample import random_freq, resample_vals

MYPY_ONLY = True

if TYPE_CHECKING:
    import sys
    from typing import Any, Union

    from numpy.typing import ArrayLike, NDArray

    import cmomy
    from cmomy import convert, rolling
    from cmomy.reduction import (
        reduce_data,
        reduce_data_grouped,
        reduce_data_indexed,
        reduce_vals,
    )

    if sys.version_info < (3, 11):
        from typing_extensions import assert_type
    else:
        from typing import assert_type


def test_centralmoments_init() -> None:
    x32 = np.array([1, 2, 3], dtype=np.float32)
    x64 = np.array([1, 2, 3], dtype=np.float64)

    if TYPE_CHECKING:
        assert_type(CentralMoments(x32, mom_ndim=1), CentralMoments[np.float32])
        assert_type(CentralMoments(x64, mom_ndim=1), CentralMoments[np.float64])
        assert_type(
            CentralMoments(x64, mom_ndim=1, dtype=np.float32),
            CentralMoments[np.float32],
        )
        assert_type(
            CentralMoments(x32, mom_ndim=1, dtype=np.float64),
            CentralMoments[np.float64],
        )
        assert_type(CentralMoments([1, 2, 3], mom_ndim=1), CentralMoments[Any])

        assert_type(
            CentralMoments([1, 2, 3], mom_ndim=1, dtype=np.float32),
            CentralMoments[np.float32],
        )
        assert_type(
            CentralMoments([1, 2, 3], mom_ndim=1, dtype=np.dtype("f8")),
            CentralMoments[np.float64],
        )
        assert_type(
            CentralMoments([1, 2, 3], mom_ndim=1, dtype="f8"), CentralMoments[Any]
        )

        assert_type(CentralMoments[np.float32]([1, 2, 3]), CentralMoments[np.float32])
        assert_type(CentralMoments[np.float64]([1, 2, 3]), CentralMoments[np.float64])


def test_xcentralmoments_init() -> None:
    data32 = xr.DataArray(np.zeros((10, 3, 4), dtype=np.float32))
    data64 = xr.DataArray(np.zeros((10, 3, 4), dtype=np.float64))

    if TYPE_CHECKING:
        assert_type(xCentralMoments(data32, mom_ndim=1), xCentralMoments[Any])
        assert_type(xCentralMoments(data64, mom_ndim=1), xCentralMoments[Any])
        assert_type(
            xCentralMoments[np.float32](data32, mom_ndim=1), xCentralMoments[np.float32]
        )
        assert_type(
            xCentralMoments[np.float64](data64, mom_ndim=1), xCentralMoments[np.float64]
        )

        assert_type(
            xCentralMoments(data32, mom_ndim=1, dtype=np.float32),
            xCentralMoments[np.float32],
        )

        assert_type(
            xCentralMoments(data32, mom_ndim=1, dtype=np.float64),
            xCentralMoments[np.float64],
        )
        assert_type(
            xCentralMoments(data32, mom_ndim=1, dtype=np.dtype("f8")),
            xCentralMoments[np.float64],
        )
        assert_type(
            xCentralMoments(data32, mom_ndim=1, dtype="f8"),
            xCentralMoments[Any],
        )


def test_astype() -> None:
    x32 = np.array([1, 2, 3], dtype=np.float32)
    x64 = np.array([1, 2, 3], dtype=np.float64)
    c32 = CentralMoments(x32, mom_ndim=1)
    c64 = CentralMoments(x64, mom_ndim=1)

    cx32 = c32.to_x()
    cx64 = c64.to_x()

    cc32 = cx32.to_c()
    cc64 = cx64.to_c()

    if TYPE_CHECKING:
        assert_type(c32, CentralMoments[np.float32])
        assert_type(c64, CentralMoments[np.float64])
        assert_type(c32.astype(np.float64), CentralMoments[np.float64])
        assert_type(c64.astype(np.float32), CentralMoments[np.float32])
        assert_type(c32.astype(np.dtype("f8")), CentralMoments[np.float64])
        assert_type(c64.astype(np.dtype("f4")), CentralMoments[np.float32])
        assert_type(c64.astype("f4"), CentralMoments[Any])

        assert_type(c32.astype(None), CentralMoments[np.float64])
        assert_type(c64.astype(None), CentralMoments[np.float64])
        assert_type(c32.astype(c32.dtype), CentralMoments[np.float32])
        assert_type(c64.astype(c64.dtype), CentralMoments[np.float64])

        assert_type(cx32, xCentralMoments[np.float32])
        assert_type(cx64, xCentralMoments[np.float64])
        assert_type(cx32.astype(np.float64), xCentralMoments[np.float64])
        assert_type(cx64.astype(np.float32), xCentralMoments[np.float32])
        assert_type(cx32.astype(np.dtype("f8")), xCentralMoments[np.float64])
        assert_type(cx64.astype(np.dtype("f4")), xCentralMoments[np.float32])
        assert_type(cx64.astype("f4"), xCentralMoments[Any])

        assert_type(cx32.astype(None), xCentralMoments[np.float64])
        assert_type(cx64.astype(None), xCentralMoments[np.float64])
        assert_type(cx32.astype(cx32.dtype), xCentralMoments[np.float32])
        assert_type(cx64.astype(cx64.dtype), xCentralMoments[np.float64])

        assert_type(cc32, CentralMoments[np.float32])
        assert_type(cc64, CentralMoments[np.float64])
        assert_type(cc32.astype(np.float64), CentralMoments[np.float64])
        assert_type(cc64.astype(np.float32), CentralMoments[np.float32])
        assert_type(cc32.astype(np.dtype("f8")), CentralMoments[np.float64])
        assert_type(cc64.astype(np.dtype("f4")), CentralMoments[np.float32])
        assert_type(cc64.astype("f4"), CentralMoments[Any])

        assert_type(cc32.astype(None), CentralMoments[np.float64])
        assert_type(cc64.astype(None), CentralMoments[np.float64])
        assert_type(cc32.astype(cc32.dtype), CentralMoments[np.float32])
        assert_type(cc64.astype(cc64.dtype), CentralMoments[np.float64])


def test_reduce_vals() -> None:
    x32 = np.array([1, 2, 3], dtype=np.float32)
    x64 = np.array([1, 2, 3], dtype=np.float64)

    out32 = np.zeros((4,), dtype=np.float32)
    out64 = np.zeros((4,), dtype=np.float64)

    if TYPE_CHECKING:
        assert_type(reduce_vals(x32, mom=3), NDArray[np.float32])
        assert_type(reduce_vals(x64, mom=3), NDArray[np.float64])

        # dtype override x
        assert_type(reduce_vals(x32, mom=3, dtype=np.float64), NDArray[np.float64])
        assert_type(reduce_vals(x64, mom=3, dtype=np.float32), NDArray[np.float32])

        # out override x
        assert_type(reduce_vals(x32, mom=3, out=out64), NDArray[np.float64])
        assert_type(reduce_vals(x64, mom=3, out=out32), NDArray[np.float32])

        # out override x and dtype
        assert_type(
            reduce_vals(x32, mom=3, out=out64, dtype=np.float32), NDArray[np.float64]
        )
        assert_type(
            reduce_vals(x64, mom=3, out=out32, dtype=np.float64), NDArray[np.float32]
        )

        xc = np.array([1, 2, 3])
        assert_type(xc, NDArray[Any])

        # Would like this to default to np.float64
        assert_type(reduce_vals(xc, mom=3), NDArray[Any])
        assert_type(reduce_vals(xc, mom=3, dtype=np.float32), NDArray[np.float32])
        assert_type(
            reduce_vals([1, 2, 3], mom=3, dtype=np.float32), NDArray[np.float32]
        )
        assert_type(
            reduce_vals([1, 2, 3], mom=3, dtype=np.float64), NDArray[np.float64]
        )

        assert_type(
            reduce_vals([1, 2, 3], mom=3, dtype=np.dtype("f8")), NDArray[np.float64]
        )

        # unknown dtype
        assert_type(reduce_vals(x32, mom=3, dtype="f8"), NDArray[Any])
        assert_type(reduce_vals([1, 2, 3], mom=3, dtype="f8"), NDArray[Any])
        assert_type(reduce_vals([1.0, 2.0, 3.0], mom=3), NDArray[Any])

        xx = xr.DataArray(x32)
        assert_type(reduce_vals(xx, mom=3), xr.DataArray)

        ds = xx.to_dataset()
        assert_type(reduce_vals(ds, mom=3), xr.Dataset)

        if MYPY_ONLY:
            # TODO(wpk): would love to figure out how to get pyright to give this
            # as the fallback overload...
            g: ArrayLike | xr.DataArray | xr.Dataset = xc
            assert_type(
                reduce_vals(g, mom=3), Union[xr.DataArray, xr.Dataset, NDArray[Any]]
            )


def test_reduce_data() -> None:
    x32 = np.array([1, 2, 3], dtype=np.float32)
    x64 = np.array([1, 2, 3], dtype=np.float64)

    out32 = np.zeros((4,), dtype=np.float32)
    out64 = np.zeros((4,), dtype=np.float64)
    if TYPE_CHECKING:
        assert_type(reduce_data(x32, mom_ndim=1), NDArray[np.float32])
        assert_type(reduce_data(x64, mom_ndim=1), NDArray[np.float64])

        assert_type(reduce_data(x32, mom_ndim=1, dtype=np.float64), NDArray[np.float64])
        assert_type(reduce_data(x64, mom_ndim=1, dtype=np.float32), NDArray[np.float32])

        assert_type(reduce_data(x32, mom_ndim=1, out=out64), NDArray[np.float64])
        assert_type(reduce_data(x64, mom_ndim=1, out=out32), NDArray[np.float32])

        assert_type(
            reduce_data(x32, mom_ndim=1, out=out64, dtype=np.float32),
            NDArray[np.float64],
        )
        assert_type(
            reduce_data(x64, mom_ndim=1, out=out32, dtype=np.float64),
            NDArray[np.float32],
        )

        xc = np.array([1, 2, 3])
        assert_type(xc, NDArray[Any])

        # Would like this to default to np.float64
        assert_type(reduce_data(xc, mom_ndim=1), NDArray[Any])
        assert_type(reduce_data(xc, mom_ndim=1, dtype=np.float32), NDArray[np.float32])
        assert_type(
            reduce_data([1, 2, 3], mom_ndim=1, dtype=np.float32), NDArray[np.float32]
        )
        assert_type(
            reduce_data([1, 2, 3], mom_ndim=1, dtype=np.float64), NDArray[np.float64]
        )

        assert_type(
            reduce_data(x32, mom_ndim=1, dtype="f8"),
            NDArray[Any],
        )
        assert_type(
            reduce_data([1, 2, 3], mom_ndim=1, dtype="f8"),
            NDArray[Any],
        )
        assert_type(reduce_data([1.0, 2.0, 3.0], mom_ndim=1), NDArray[Any])

        xx = xr.DataArray(x32)
        assert_type(reduce_data(xx, mom_ndim=1), xr.DataArray)

        ds = xx.to_dataset(name="hello")
        assert_type(reduce_data(ds, mom_ndim=1), xr.Dataset)

        if MYPY_ONLY:
            g: ArrayLike | xr.DataArray | xr.Dataset = xc
            assert_type(
                reduce_data(g, mom_ndim=1),
                Union[xr.DataArray, xr.Dataset, NDArray[Any]],
            )


def test_convert() -> None:
    x32 = np.array([1, 2, 3], dtype=np.float32)
    x64 = np.array([1, 2, 3], dtype=np.float64)

    out32 = np.zeros((4,), dtype=np.float32)
    out64 = np.zeros((4,), dtype=np.float64)
    if TYPE_CHECKING:
        assert_type(convert.moments_type(x32, mom_ndim=1), NDArray[np.float32])
        assert_type(convert.moments_type(x64, mom_ndim=1), NDArray[np.float64])

        assert_type(
            convert.moments_type(x32, mom_ndim=1, dtype=np.float64), NDArray[np.float64]
        )
        assert_type(
            convert.moments_type(x64, mom_ndim=1, dtype=np.float32), NDArray[np.float32]
        )

        assert_type(
            convert.moments_type(x32, mom_ndim=1, out=out64), NDArray[np.float64]
        )
        assert_type(
            convert.moments_type(x64, mom_ndim=1, out=out32), NDArray[np.float32]
        )

        assert_type(
            convert.moments_type(x32, mom_ndim=1, out=out64, dtype=np.float32),
            NDArray[np.float64],
        )
        assert_type(
            convert.moments_type(x64, mom_ndim=1, out=out32, dtype=np.float64),
            NDArray[np.float32],
        )

        xc = np.array([1, 2, 3])
        assert_type(xc, NDArray[Any])

        # Would like this to default to np.float64
        assert_type(convert.moments_type(xc, mom_ndim=1), NDArray[Any])
        assert_type(
            convert.moments_type(xc, mom_ndim=1, dtype=np.float32), NDArray[np.float32]
        )
        assert_type(
            convert.moments_type([1, 2, 3], mom_ndim=1, dtype=np.float32),
            NDArray[np.float32],
        )
        assert_type(
            convert.moments_type([1, 2, 3], mom_ndim=1, dtype=np.float64),
            NDArray[np.float64],
        )

        assert_type(
            convert.moments_type([1, 2, 3], mom_ndim=1, dtype="f8"),
            NDArray[Any],
        )
        assert_type(
            convert.moments_type(x32, mom_ndim=1, dtype="f8"),
            NDArray[Any],
        )
        assert_type(
            convert.moments_type(xc, mom_ndim=1, dtype="f8"),
            NDArray[Any],
        )

        assert_type(convert.moments_type([1.0, 2.0, 3.0], mom_ndim=1), NDArray[Any])

        xx = xr.DataArray(x32)
        assert_type(convert.moments_type(xx, mom_ndim=1), xr.DataArray)
        assert_type(convert.moments_type(xx.to_dataset(), mom_ndim=1), xr.Dataset)
        if MYPY_ONLY:
            g: ArrayLike | xr.DataArray | xr.Dataset = xc
            assert_type(
                convert.moments_type(g, mom_ndim=1),
                Union[xr.DataArray, xr.Dataset, NDArray[Any]],
            )


def test_moveaxis() -> None:
    x32 = np.array([1, 2, 3], dtype=np.float32)
    x64 = np.array([1, 2, 3], dtype=np.float64)
    xint = np.array([1, 2, 3], dtype=np.int64)
    xAny: NDArray[Any] = xint

    if TYPE_CHECKING:
        assert_type(cmomy.moveaxis(x32, mom_ndim=1), NDArray[np.float32])
        assert_type(cmomy.moveaxis(x64, mom_ndim=1), NDArray[np.float64])
        assert_type(cmomy.moveaxis(xint, mom_ndim=1), NDArray[np.int64])
        assert_type(cmomy.moveaxis(xAny, mom_ndim=1), NDArray[Any])
        assert_type(cmomy.moveaxis(xr.DataArray(xAny), mom_ndim=1), xr.DataArray)


def test_select_moment() -> None:
    x32 = np.array([1, 2, 3], dtype=np.float32)
    x64 = np.array([1, 2, 3], dtype=np.float64)
    xint = np.array([1, 2, 3], dtype=np.int64)
    xAny: NDArray[Any] = xint

    if TYPE_CHECKING:
        assert_type(
            cmomy.utils.select_moment(x32, "weight", mom_ndim=1), NDArray[np.float32]
        )
        assert_type(
            cmomy.utils.select_moment(x64, "weight", mom_ndim=1), NDArray[np.float64]
        )
        assert_type(
            cmomy.utils.select_moment(xint, "weight", mom_ndim=1), NDArray[np.int64]
        )
        assert_type(cmomy.utils.select_moment(xAny, "weight", mom_ndim=1), NDArray[Any])
        assert_type(
            cmomy.utils.select_moment(xr.DataArray(xAny), "weight", mom_ndim=1),
            xr.DataArray,
        )
        assert_type(
            cmomy.utils.select_moment(
                xr.DataArray(xAny).to_dataset(), "weight", mom_ndim=1
            ),
            xr.Dataset,
        )


def test_cumulative() -> None:
    x32 = np.array([1, 2, 3], dtype=np.float32)
    x64 = np.array([1, 2, 3], dtype=np.float64)

    out32 = np.zeros((4,), dtype=np.float32)
    out64 = np.zeros((4,), dtype=np.float64)
    outAny: NDArray[Any] = np.zeros((4,), dtype=np.float32)

    if TYPE_CHECKING:
        assert_type(convert.cumulative(x32, mom_ndim=1), NDArray[np.float32])
        assert_type(convert.cumulative(x64, mom_ndim=1), NDArray[np.float64])

        assert_type(
            convert.cumulative(x32, mom_ndim=1, dtype=np.float64), NDArray[np.float64]
        )
        assert_type(
            convert.cumulative(x64, mom_ndim=1, dtype=np.float32), NDArray[np.float32]
        )

        assert_type(convert.cumulative(x32, mom_ndim=1, out=out64), NDArray[np.float64])
        assert_type(convert.cumulative(x64, mom_ndim=1, out=out32), NDArray[np.float32])

        assert_type(
            convert.cumulative(x32, mom_ndim=1, out=out64, dtype=np.float32),
            NDArray[np.float64],
        )
        assert_type(
            convert.cumulative(x64, mom_ndim=1, out=out32, dtype=np.float64),
            NDArray[np.float32],
        )

        assert_type(
            convert.cumulative(x32, mom_ndim=1, out=outAny, dtype=np.float32),
            NDArray[Any],
        )

        xc = np.array([1, 2, 3])
        assert_type(xc, NDArray[Any])

        # Would like this to default to np.float64
        assert_type(convert.cumulative(xc, mom_ndim=1), NDArray[Any])
        assert_type(
            convert.cumulative(xc, mom_ndim=1, dtype=np.float32), NDArray[np.float32]
        )
        assert_type(
            convert.cumulative([1, 2, 3], mom_ndim=1, dtype=np.float32),
            NDArray[np.float32],
        )
        assert_type(
            convert.cumulative([1, 2, 3], mom_ndim=1, dtype=np.float64),
            NDArray[np.float64],
        )

        assert_type(
            convert.cumulative([1, 2, 3], mom_ndim=1, dtype="f8"),
            NDArray[Any],
        )
        assert_type(
            convert.cumulative(x32, mom_ndim=1, dtype="f8"),
            NDArray[Any],
        )
        assert_type(
            convert.cumulative(xc, mom_ndim=1, dtype="f8"),
            NDArray[Any],
        )

        assert_type(convert.cumulative([1.0, 2.0, 3.0], mom_ndim=1), NDArray[Any])

        xx = xr.DataArray(x32)
        assert_type(convert.cumulative(xx, mom_ndim=1), xr.DataArray)
        assert_type(convert.cumulative(xx.to_dataset(), mom_ndim=1), xr.Dataset)

        if MYPY_ONLY:
            g: ArrayLike | xr.DataArray | xr.Dataset = xc
            assert_type(
                convert.cumulative(g, mom_ndim=1),
                Union[xr.DataArray, xr.Dataset, NDArray[Any]],
            )


def test_vals_to_data() -> None:
    x32 = np.array([1, 2, 3], dtype=np.float32)
    x64 = np.array([1, 2, 3], dtype=np.float64)

    out32 = np.zeros((4,), dtype=np.float32)
    out64 = np.zeros((4,), dtype=np.float64)
    outAny: NDArray[Any] = np.zeros((4,), dtype=np.float32)

    if TYPE_CHECKING:
        assert_type(cmomy.utils.vals_to_data(x32, mom=1), NDArray[np.float32])
        assert_type(cmomy.utils.vals_to_data(x64, mom=1), NDArray[np.float64])

        assert_type(
            cmomy.utils.vals_to_data(x32, mom=1, dtype=np.float64), NDArray[np.float64]
        )
        assert_type(
            cmomy.utils.vals_to_data(x64, mom=1, dtype=np.float32), NDArray[np.float32]
        )

        assert_type(
            cmomy.utils.vals_to_data(x32, mom=1, out=out64), NDArray[np.float64]
        )
        assert_type(
            cmomy.utils.vals_to_data(x64, mom=1, out=out32), NDArray[np.float32]
        )

        assert_type(
            cmomy.utils.vals_to_data(x32, mom=1, out=out64, dtype=np.float32),
            NDArray[np.float64],
        )
        assert_type(
            cmomy.utils.vals_to_data(x64, mom=1, out=out32, dtype=np.float64),
            NDArray[np.float32],
        )

        assert_type(
            cmomy.utils.vals_to_data(x32, mom=1, out=outAny, dtype=np.float32),
            NDArray[Any],
        )

        xc = np.array([1, 2, 3])
        assert_type(xc, NDArray[Any])

        # Would like this to default to np.float64
        assert_type(cmomy.utils.vals_to_data(xc, mom=1), NDArray[Any])
        assert_type(
            cmomy.utils.vals_to_data(xc, mom=1, dtype=np.float32), NDArray[np.float32]
        )
        assert_type(
            cmomy.utils.vals_to_data([1, 2, 3], mom=1, dtype=np.float32),
            NDArray[np.float32],
        )
        assert_type(
            cmomy.utils.vals_to_data([1, 2, 3], mom=1, dtype=np.float64),
            NDArray[np.float64],
        )

        assert_type(
            cmomy.utils.vals_to_data([1, 2, 3], mom=1, dtype="f8"),
            NDArray[Any],
        )
        assert_type(
            cmomy.utils.vals_to_data(x32, mom=1, dtype="f8"),
            NDArray[Any],
        )
        assert_type(
            cmomy.utils.vals_to_data(xc, mom=1, dtype="f8"),
            NDArray[Any],
        )

        assert_type(cmomy.utils.vals_to_data([1.0, 2.0, 3.0], mom=1), NDArray[Any])

        xx = xr.DataArray(x32)
        assert_type(cmomy.utils.vals_to_data(xx, mom=1), xr.DataArray)
        assert_type(cmomy.utils.vals_to_data(xx.to_dataset(), mom=1), xr.Dataset)

        if MYPY_ONLY:
            g: ArrayLike | xr.DataArray | xr.Dataset = xc
            assert_type(
                cmomy.utils.vals_to_data(g, mom=1),
                Union[xr.DataArray, xr.Dataset, NDArray[Any]],
            )


def test_convert_moments_to_comoments() -> None:
    x32 = np.array([1, 2, 3, 4], dtype=np.float32)
    x64 = np.array([1, 2, 3, 4], dtype=np.float64)

    if TYPE_CHECKING:
        assert_type(convert.moments_to_comoments(x32, mom=(2, -1)), NDArray[np.float32])
        assert_type(convert.moments_to_comoments(x64, mom=(2, -1)), NDArray[np.float64])

        assert_type(
            convert.moments_to_comoments(x32, mom=(2, -1), dtype=np.float64),
            NDArray[np.float64],
        )
        assert_type(
            convert.moments_to_comoments(x64, mom=(2, -1), dtype=np.float32),
            NDArray[np.float32],
        )

        xc = np.array([1, 2, 3, 4])
        assert_type(xc, NDArray[Any])

        # Would like this to default to np.float64
        assert_type(convert.moments_to_comoments(xc, mom=(2, -1)), NDArray[Any])
        assert_type(
            convert.moments_to_comoments(xc, mom=(2, -1), dtype=np.float32),
            NDArray[np.float32],
        )
        assert_type(
            convert.moments_to_comoments([1, 2, 3, 4], mom=(2, -1), dtype=np.float32),
            NDArray[np.float32],
        )
        assert_type(
            convert.moments_to_comoments([1, 2, 3, 4], mom=(2, -1), dtype=np.float64),
            NDArray[np.float64],
        )

        assert_type(
            convert.moments_to_comoments([1, 2, 3, 4], mom=(2, -1), dtype="f8"),
            NDArray[Any],
        )
        assert_type(
            convert.moments_to_comoments(x32, mom=(2, -1), dtype="f8"),
            NDArray[Any],
        )
        assert_type(
            convert.moments_to_comoments(xc, mom=(2, -1), dtype="f8"),
            NDArray[Any],
        )

        assert_type(
            convert.moments_to_comoments([1.0, 2.0, 3.0], mom=(2, -1)), NDArray[Any]
        )

        xx = xr.DataArray(x32)
        assert_type(convert.moments_to_comoments(xx, mom=(2, -1)), xr.DataArray)
        assert_type(
            convert.moments_to_comoments(xx.to_dataset(), mom=(2, -1)), xr.Dataset
        )

        if MYPY_ONLY:
            g: ArrayLike | xr.DataArray | xr.Dataset = xc
            assert_type(
                convert.moments_to_comoments(g, mom=(2, -1)),
                Union[xr.DataArray, xr.Dataset, NDArray[Any]],
            )


def test_moments_to_comoments() -> None:
    if TYPE_CHECKING:
        x32 = np.array([1, 2, 3, 4], dtype=np.float32)
        x64 = np.array([1, 2, 3, 4], dtype=np.float64)
        c32 = CentralMoments(x32, mom_ndim=1)
        c64 = CentralMoments(x64, mom_ndim=1)
        cAny = CentralMoments([1, 2, 3, 4], mom_ndim=1)

        assert_type(c32.moments_to_comoments(mom=(2, -1)), CentralMoments[np.float32])
        assert_type(c64.moments_to_comoments(mom=(2, -1)), CentralMoments[np.float64])
        assert_type(cAny.moments_to_comoments(mom=(2, -1)), CentralMoments[Any])

        assert_type(
            c32.to_x().moments_to_comoments(mom=(2, -1)), xCentralMoments[np.float32]
        )
        assert_type(
            c64.to_x().moments_to_comoments(mom=(2, -1)), xCentralMoments[np.float64]
        )
        assert_type(cAny.to_x().moments_to_comoments(mom=(2, -1)), xCentralMoments[Any])


def test_assign_moment_central() -> None:
    if TYPE_CHECKING:
        x32 = np.array([1, 2, 3, 4], dtype=np.float32)
        x64 = np.array([1, 2, 3, 4], dtype=np.float64)
        c32 = CentralMoments(x32, mom_ndim=1)
        c64 = CentralMoments(x64, mom_ndim=1)
        cAny = CentralMoments([1, 2, 3, 4], mom_ndim=1)

        assert_type(c32.assign_moment("weight", 1.0), CentralMoments[np.float32])
        assert_type(c64.assign_moment("weight", 1.0), CentralMoments[np.float64])
        assert_type(cAny.assign_moment("weight", 1.0), CentralMoments[Any])

        assert_type(
            c32.to_x().assign_moment("weight", 1.0), xCentralMoments[np.float32]
        )
        assert_type(
            c64.to_x().assign_moment("weight", 1.0), xCentralMoments[np.float64]
        )
        assert_type(cAny.to_x().assign_moment("weight", 1.0), xCentralMoments[Any])


def test_concat() -> None:
    if TYPE_CHECKING:
        x32 = np.zeros((10, 2, 3), dtype=np.float32)
        x64 = np.zeros((10, 2, 3), dtype=np.float64)

        dx32 = xr.DataArray(x32, dims=["a", "b", "mom"])
        dx64 = xr.DataArray(x64, dims=["a", "b", "mom"])

        c32 = CentralMoments(x32)
        c64 = CentralMoments(x64)

        dc32: xCentralMoments[np.float32] = xCentralMoments(dx32)
        dc64: xCentralMoments[np.float64] = xCentralMoments(dx64)

        assert_type(convert.concat((x32, x32), axis=0), NDArray[np.float32])
        assert_type(convert.concat((x64, x64), axis=0), NDArray[np.float64])

        assert_type(convert.concat((dx32, dx32), axis=0), xr.DataArray)

        assert_type(convert.concat((c32, c32), axis=0), CentralMoments[np.float32])
        assert_type(convert.concat((dc32, dc32), axis=0), xCentralMoments[np.float32])

        assert_type(convert.concat((c64, c64), axis=0), CentralMoments[np.float64])
        assert_type(convert.concat((dc64, dc64), axis=0), xCentralMoments[np.float64])

        xany: NDArray[Any] = np.zeros([1, 2, 3])
        dxany = xr.DataArray(xany)
        assert_type(convert.concat((xany, xany), axis=0), NDArray[Any])
        assert_type(convert.concat((dxany, dxany), axis=0), xr.DataArray)

        cany = CentralMoments(xany)
        assert_type(convert.concat((cany, cany), axis=0), CentralMoments[Any])

        dcany = xCentralMoments(dxany)
        assert_type(convert.concat((dcany, dcany), axis=0), xCentralMoments[Any])


# if mypy properly supports partial will add this...
# def test_convert_central_to_raw() -> None:


def test_reduce_data_grouped() -> None:
    x32 = np.array([1, 2, 3], dtype=np.float32)
    x64 = np.array([1, 2, 3], dtype=np.float64)

    out32 = np.zeros((4,), dtype=np.float32)
    out64 = np.zeros((4,), dtype=np.float64)

    by = np.zeros((10,), dtype=np.int64)

    if TYPE_CHECKING:
        assert_type(reduce_data_grouped(x32, mom_ndim=1, by=by), NDArray[np.float32])
        assert_type(reduce_data_grouped(x64, mom_ndim=1, by=by), NDArray[np.float64])

        assert_type(
            reduce_data_grouped(x32, mom_ndim=1, by=by, dtype=np.float64),
            NDArray[np.float64],
        )
        assert_type(
            reduce_data_grouped(x64, mom_ndim=1, by=by, dtype=np.float32),
            NDArray[np.float32],
        )

        assert_type(
            reduce_data_grouped(x32, mom_ndim=1, by=by, out=out64), NDArray[np.float64]
        )
        assert_type(
            reduce_data_grouped(x64, mom_ndim=1, by=by, out=out32), NDArray[np.float32]
        )

        assert_type(
            reduce_data_grouped(x32, mom_ndim=1, by=by, out=out64, dtype=np.float32),
            NDArray[np.float64],
        )
        assert_type(
            reduce_data_grouped(x64, mom_ndim=1, by=by, out=out32, dtype=np.float64),
            NDArray[np.float32],
        )

        xc = np.array([1, 2, 3])
        assert_type(xc, NDArray[Any])

        # Would like this to default to np.float64
        assert_type(reduce_data_grouped(xc, mom_ndim=1, by=by), NDArray[Any])
        assert_type(
            reduce_data_grouped(xc, mom_ndim=1, by=by, dtype=np.float32),
            NDArray[np.float32],
        )
        assert_type(
            reduce_data_grouped([1, 2, 3], mom_ndim=1, by=by, dtype=np.float32),
            NDArray[np.float32],
        )
        assert_type(
            reduce_data_grouped([1, 2, 3], mom_ndim=1, by=by, dtype=np.float64),
            NDArray[np.float64],
        )

        assert_type(
            reduce_data_grouped(x64, mom_ndim=1, by=by, dtype="f8"), NDArray[Any]
        )
        assert_type(
            reduce_data_grouped([1.0, 2.0, 3.0], mom_ndim=1, by=by), NDArray[Any]
        )

        xx = xr.DataArray(x32)
        assert_type(reduce_data_grouped(xx, mom_ndim=1, by=by), xr.DataArray)

        assert_type(reduce_data_grouped(xx.to_dataset(), mom_ndim=1, by=by), xr.Dataset)

        if MYPY_ONLY:
            g: ArrayLike | xr.DataArray | xr.Dataset = xc
            assert_type(
                reduce_data_grouped(g, mom_ndim=1, by=by),
                Union[xr.DataArray, xr.Dataset, NDArray[Any]],
            )


def test_reduce_data_indexed() -> None:
    x32 = np.array([1, 2, 3], dtype=np.float32)
    x64 = np.array([1, 2, 3], dtype=np.float64)

    out32 = np.zeros((4,), dtype=np.float32)
    out64 = np.zeros((4,), dtype=np.float64)

    index = np.zeros((10,), dtype=np.int64)
    group_start = index
    group_end = index

    if TYPE_CHECKING:
        assert_type(
            reduce_data_indexed(
                x32,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
            ),
            NDArray[np.float32],
        )
        assert_type(
            reduce_data_indexed(
                x64,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
            ),
            NDArray[np.float64],
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
            NDArray[np.float64],
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
            NDArray[np.float32],
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
            NDArray[np.float64],
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
            NDArray[np.float32],
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
            NDArray[np.float64],
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
            NDArray[np.float32],
        )

        xc = np.array([1, 2, 3])
        assert_type(xc, NDArray[Any])

        # Would like this to default to np.float64
        assert_type(
            reduce_data_indexed(
                xc,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
            ),
            NDArray[Any],
        )
        assert_type(
            reduce_data_indexed(
                xc,
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
                dtype=np.float32,
            ),
            NDArray[np.float32],
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
            NDArray[np.float32],
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
            NDArray[np.float64],
        )

        assert_type(
            reduce_data_indexed(
                [1.0, 2.0, 3.0],
                mom_ndim=1,
                index=index,
                group_start=group_start,
                group_end=group_end,
            ),
            NDArray[Any],
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
            NDArray[Any],
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
            g: ArrayLike | xr.DataArray | xr.Dataset = xc
            assert_type(
                reduce_data_indexed(
                    g,
                    mom_ndim=1,
                    index=index,
                    group_start=group_start,
                    group_end=group_end,
                ),
                Union[xr.DataArray, xr.Dataset, NDArray[Any]],
            )


def test_resample_data() -> None:
    from cmomy.resample import random_freq, resample_data

    x32 = np.zeros((10, 3, 3), dtype=np.float32)
    x64 = np.zeros((10, 3, 3), dtype=np.float64)

    out32 = np.zeros((4,), dtype=np.float32)
    out64 = np.zeros((4,), dtype=np.float64)

    freq = random_freq(20, 10)

    if TYPE_CHECKING:
        assert_type(resample_data(x32, freq=freq, mom_ndim=1), NDArray[np.float32])
        assert_type(resample_data(x64, freq=freq, mom_ndim=1), NDArray[np.float64])

        assert_type(
            resample_data(x32, freq=freq, mom_ndim=1, dtype=np.float64),
            NDArray[np.float64],
        )
        assert_type(
            resample_data(x64, freq=freq, mom_ndim=1, dtype=np.float32),
            NDArray[np.float32],
        )

        assert_type(
            resample_data(x32, freq=freq, mom_ndim=1, out=out64), NDArray[np.float64]
        )
        assert_type(
            resample_data(x64, freq=freq, mom_ndim=1, out=out32), NDArray[np.float32]
        )

        assert_type(
            resample_data(x32, freq=freq, mom_ndim=1, out=out64, dtype=np.float32),
            NDArray[np.float64],
        )
        assert_type(
            resample_data(x64, freq=freq, mom_ndim=1, out=out32, dtype=np.float64),
            NDArray[np.float32],
        )

        xc = np.array([1, 2, 3])
        assert_type(xc, NDArray[Any])

        # Would like this to default to np.float64
        assert_type(resample_data(xc, freq=freq, mom_ndim=1), NDArray[Any])
        assert_type(
            resample_data(xc, freq=freq, mom_ndim=1, dtype=np.float32),
            NDArray[np.float32],
        )
        assert_type(
            resample_data([1, 2, 3], freq=freq, mom_ndim=1, dtype=np.float32),
            NDArray[np.float32],
        )
        assert_type(
            resample_data([1, 2, 3], freq=freq, mom_ndim=1, dtype=np.float64),
            NDArray[np.float64],
        )

        assert_type(resample_data([1.0, 2.0, 3.0], freq=freq, mom_ndim=1), NDArray[Any])
        assert_type(resample_data(x32, freq=freq, mom_ndim=1, dtype="f8"), NDArray[Any])

        xx = xr.DataArray(x32)
        assert_type(resample_data(xx, freq=freq, mom_ndim=1), xr.DataArray)
        assert_type(resample_data(xx.to_dataset(), freq=freq, mom_ndim=1), xr.Dataset)

        if MYPY_ONLY:
            g: ArrayLike | xr.DataArray | xr.Dataset = xc
            assert_type(
                resample_data(g, freq=freq, mom_ndim=1),
                Union[xr.DataArray, xr.Dataset, NDArray[Any]],
            )


def test_jackknife_data() -> None:
    from cmomy.resample import jackknife_data

    x32 = np.zeros((10, 3, 3), dtype=np.float32)
    x64 = np.zeros((10, 3, 3), dtype=np.float64)

    out32 = np.zeros((4,), dtype=np.float32)
    out64 = np.zeros((4,), dtype=np.float64)

    random_freq(20, 10)

    if TYPE_CHECKING:
        assert_type(jackknife_data(x32, mom_ndim=1), NDArray[np.float32])
        assert_type(jackknife_data(x64, mom_ndim=1), NDArray[np.float64])

        assert_type(
            jackknife_data(x32, mom_ndim=1, dtype=np.float64),
            NDArray[np.float64],
        )
        assert_type(
            jackknife_data(x64, mom_ndim=1, dtype=np.float32),
            NDArray[np.float32],
        )

        assert_type(jackknife_data(x32, mom_ndim=1, out=out64), NDArray[np.float64])
        assert_type(jackknife_data(x64, mom_ndim=1, out=out32), NDArray[np.float32])

        assert_type(
            jackknife_data(x32, mom_ndim=1, out=out64, dtype=np.float32),
            NDArray[np.float64],
        )
        assert_type(
            jackknife_data(x64, mom_ndim=1, out=out32, dtype=np.float64),
            NDArray[np.float32],
        )

        xc = np.array([1, 2, 3])
        assert_type(xc, NDArray[Any])

        # Would like this to default to np.float64
        assert_type(jackknife_data(xc, mom_ndim=1), NDArray[Any])
        assert_type(
            jackknife_data(xc, mom_ndim=1, dtype=np.float32),
            NDArray[np.float32],
        )
        assert_type(
            jackknife_data([1, 2, 3], mom_ndim=1, dtype=np.float32),
            NDArray[np.float32],
        )
        assert_type(
            jackknife_data([1, 2, 3], mom_ndim=1, dtype=np.float64),
            NDArray[np.float64],
        )

        assert_type(jackknife_data([1.0, 2.0, 3.0], mom_ndim=1), NDArray[Any])
        assert_type(jackknife_data(x32, mom_ndim=1, dtype="f8"), NDArray[Any])

        xx = xr.DataArray(x32)
        assert_type(jackknife_data(xx, mom_ndim=1), xr.DataArray)
        assert_type(jackknife_data(xx.to_dataset(), mom_ndim=1), xr.Dataset)

        if MYPY_ONLY:
            g: ArrayLike | xr.DataArray | xr.Dataset = xc
            assert_type(
                jackknife_data(g, mom_ndim=1),
                Union[xr.DataArray, xr.Dataset, NDArray[Any]],
            )


def test_resample_vals() -> None:
    x32 = np.zeros((10, 3, 3), dtype=np.float32)
    x64 = np.zeros((10, 3, 3), dtype=np.float64)

    out32 = np.zeros((4,), dtype=np.float32)
    out64 = np.zeros((4,), dtype=np.float64)

    freq = random_freq(20, 10)

    if TYPE_CHECKING:
        assert_type(resample_vals(x32, freq=freq, mom=3), NDArray[np.float32])
        assert_type(resample_vals(x64, freq=freq, mom=3), NDArray[np.float64])

        assert_type(
            resample_vals(x32, freq=freq, mom=3, dtype=np.float64), NDArray[np.float64]
        )
        assert_type(
            resample_vals(x64, freq=freq, mom=3, dtype=np.float32), NDArray[np.float32]
        )

        assert_type(
            resample_vals(x32, freq=freq, mom=3, out=out64), NDArray[np.float64]
        )
        assert_type(
            resample_vals(x64, freq=freq, mom=3, out=out32), NDArray[np.float32]
        )

        assert_type(
            resample_vals(x32, freq=freq, mom=3, out=out64, dtype=np.float32),
            NDArray[np.float64],
        )
        assert_type(
            resample_vals(x64, freq=freq, mom=3, out=out32, dtype=np.float64),
            NDArray[np.float32],
        )

        xc = np.array([1, 2, 3])
        assert_type(xc, NDArray[Any])

        # Would like this to default to np.float64
        assert_type(resample_vals(xc, freq=freq, mom=3), NDArray[Any])
        assert_type(
            resample_vals(xc, freq=freq, mom=3, dtype=np.float32), NDArray[np.float32]
        )
        assert_type(
            resample_vals([1, 2, 3], freq=freq, mom=3, dtype=np.float32),
            NDArray[np.float32],
        )
        assert_type(
            resample_vals([1, 2, 3], freq=freq, mom=3, dtype=np.float64),
            NDArray[np.float64],
        )

        assert_type(resample_vals([1.0, 2.0, 3.0], freq=freq, mom=3), NDArray[Any])
        assert_type(resample_vals(x32, freq=freq, mom=3, dtype="f8"), NDArray[Any])

        xx = xr.DataArray(x32)
        assert_type(resample_vals(xx, freq=freq, mom=3), xr.DataArray)

        assert_type(resample_vals(xx.to_dataset(), freq=freq, mom=3), xr.Dataset)

        if MYPY_ONLY:
            g: ArrayLike | xr.DataArray | xr.Dataset = xc
            assert_type(
                resample_vals(g, freq=freq, mom=3),
                Union[xr.DataArray, xr.Dataset, NDArray[Any]],
            )


def test_jackknife_vals() -> None:
    from cmomy.resample import jackknife_vals

    x32 = np.zeros((10, 3, 3), dtype=np.float32)
    x64 = np.zeros((10, 3, 3), dtype=np.float64)

    out32 = np.zeros((4,), dtype=np.float32)
    out64 = np.zeros((4,), dtype=np.float64)

    random_freq(20, 10)

    if TYPE_CHECKING:
        assert_type(jackknife_vals(x32, mom=3), NDArray[np.float32])
        assert_type(jackknife_vals(x64, mom=3), NDArray[np.float64])

        assert_type(jackknife_vals(x32, mom=3, dtype=np.float64), NDArray[np.float64])
        assert_type(jackknife_vals(x64, mom=3, dtype=np.float32), NDArray[np.float32])

        assert_type(jackknife_vals(x32, mom=3, out=out64), NDArray[np.float64])
        assert_type(jackknife_vals(x64, mom=3, out=out32), NDArray[np.float32])

        assert_type(
            jackknife_vals(x32, mom=3, out=out64, dtype=np.float32),
            NDArray[np.float64],
        )
        assert_type(
            jackknife_vals(x64, mom=3, out=out32, dtype=np.float64),
            NDArray[np.float32],
        )

        xc = np.array([1, 2, 3])
        assert_type(xc, NDArray[Any])

        # Would like this to default to np.float64
        assert_type(jackknife_vals(xc, mom=3), NDArray[Any])
        assert_type(jackknife_vals(xc, mom=3, dtype=np.float32), NDArray[np.float32])
        assert_type(
            jackknife_vals([1, 2, 3], mom=3, dtype=np.float32),
            NDArray[np.float32],
        )
        assert_type(
            jackknife_vals([1, 2, 3], mom=3, dtype=np.float64),
            NDArray[np.float64],
        )

        assert_type(jackknife_vals([1.0, 2.0, 3.0], mom=3), NDArray[Any])
        assert_type(jackknife_vals(x32, mom=3, dtype="f8"), NDArray[Any])

        xx = xr.DataArray(x32)
        assert_type(jackknife_vals(xx, mom=3), xr.DataArray)
        assert_type(jackknife_vals(xx.to_dataset(), mom=3), xr.Dataset)

        if MYPY_ONLY:
            g: ArrayLike | xr.DataArray | xr.Dataset = xc
            assert_type(
                jackknife_vals(g, mom=3), Union[xr.DataArray, xr.Dataset, NDArray[Any]]
            )


def test_centralmoments_zeros() -> None:
    if TYPE_CHECKING:
        assert_type(CentralMoments.zeros(mom=3), CentralMoments[np.float64])
        assert_type(CentralMoments.zeros(mom=3, dtype=None), CentralMoments[np.float64])
        assert_type(
            CentralMoments.zeros(mom=3, dtype=np.float32), CentralMoments[np.float32]
        )
        assert_type(
            CentralMoments.zeros(mom=3, dtype=np.dtype("f8")),
            CentralMoments[np.float64],
        )
        assert_type(
            CentralMoments.zeros(mom=3, dtype="f8"),
            CentralMoments[Any],
        )

        assert_type(
            CentralMoments.zeros(mom=3, dtype=np.dtype("f4")),
            CentralMoments[np.float32],
        )


def test_xcentralmoments_zeros() -> None:
    if TYPE_CHECKING:
        assert_type(xCentralMoments.zeros(mom=3), xCentralMoments[np.float64])
        assert_type(
            xCentralMoments.zeros(mom=3, dtype=None), xCentralMoments[np.float64]
        )
        assert_type(
            xCentralMoments.zeros(mom=3, dtype=np.float32), xCentralMoments[np.float32]
        )
        assert_type(
            xCentralMoments.zeros(mom=3, dtype=np.dtype("f8")),
            xCentralMoments[np.float64],
        )
        assert_type(
            xCentralMoments.zeros(mom=3, dtype="f8"),
            xCentralMoments[Any],
        )

        assert_type(
            xCentralMoments.zeros(mom=3, dtype=np.dtype("f4")),
            xCentralMoments[np.float32],
        )


def test_centralmoments_from_vals() -> None:
    data32 = np.zeros((10, 3, 4), dtype=np.float32)
    data64 = np.zeros((10, 3, 4), dtype=np.float64)

    xAny: NDArray[Any] = data32

    out32 = np.zeros((4,), dtype=np.float32)
    out64 = np.zeros((4,), dtype=np.float64)

    if TYPE_CHECKING:
        assert_type(
            CentralMoments.from_vals(data32, mom=3, axis=0), CentralMoments[np.float32]
        )
        assert_type(
            CentralMoments.from_vals(data64, mom=3, axis=0), CentralMoments[np.float64]
        )
        assert_type(
            CentralMoments.from_vals(data32, mom=3, axis=0, dtype=np.float64),
            CentralMoments[np.float64],
        )
        assert_type(
            CentralMoments.from_vals(data32, mom=3, axis=0, dtype=np.dtype("f8")),
            CentralMoments[np.float64],
        )
        assert_type(
            CentralMoments.from_vals(data32, mom=3, axis=0, dtype="f8"),
            CentralMoments[Any],
        )

        # using out
        assert_type(
            CentralMoments.from_vals(xAny, mom=3, axis=0, dtype=np.float32, out=out64),
            CentralMoments[np.float64],
        )
        assert_type(
            CentralMoments.from_vals(xAny, mom=3, axis=0, dtype=np.float64, out=out32),
            CentralMoments[np.float32],
        )

        assert_type(CentralMoments.from_vals(xAny, mom=3, axis=0), CentralMoments[Any])
        assert_type(
            CentralMoments.from_vals([1, 2, 3], mom=3, axis=0, dtype="f8"),
            CentralMoments[Any],
        )
        assert_type(
            CentralMoments.from_vals([1, 2, 3], mom=3, axis=0, dtype=np.float32),
            CentralMoments[np.float32],
        )


def test_xcentralmoments_from_vals() -> None:
    data32 = xr.DataArray(np.zeros((10, 3, 4), dtype=np.float32))
    data64 = xr.DataArray(np.zeros((10, 3, 4), dtype=np.float64))

    out32 = np.zeros((4,), dtype=np.float32)
    out64 = np.zeros((4,), dtype=np.float64)

    if TYPE_CHECKING:
        assert_type(
            xCentralMoments.from_vals(data32, mom=3, axis=0), xCentralMoments[Any]
        )
        assert_type(
            xCentralMoments.from_vals(data64, mom=3, axis=0), xCentralMoments[Any]
        )
        assert_type(
            xCentralMoments.from_vals(data32, mom=3, axis=0, dtype=np.float64),
            xCentralMoments[np.float64],
        )

        assert_type(
            xCentralMoments.from_vals(data64, mom=3, axis=0, dtype=np.float32),
            xCentralMoments[np.float32],
        )

        assert_type(
            xCentralMoments.from_vals(
                data32, mom=3, axis=0, out=out32, dtype=np.float64
            ),
            xCentralMoments[np.float32],
        )

        assert_type(
            xCentralMoments.from_vals(
                data64, mom=3, axis=0, out=out64, dtype=np.float32
            ),
            xCentralMoments[np.float64],
        )

        assert_type(
            xCentralMoments.from_vals(data32, mom=3, axis=0, dtype=np.dtype("f8")),
            xCentralMoments[np.float64],
        )
        assert_type(
            xCentralMoments.from_vals(data32, mom=3, axis=0, dtype="f8"),
            xCentralMoments[Any],
        )


def test_centralmoments_newlike() -> None:
    data32 = np.zeros((4,), dtype=np.float32)
    data64 = np.zeros((4,), dtype=np.float64)

    c32 = CentralMoments.zeros(mom=3, dtype=np.float32)
    c64 = CentralMoments.zeros(mom=3, dtype=np.float64)

    if TYPE_CHECKING:
        assert_type(c32, CentralMoments[np.float32])
        assert_type(c64, CentralMoments[np.float64])
        assert_type(c32.new_like(), CentralMoments[np.float32])
        assert_type(c32.new_like(dtype=np.float64), CentralMoments[np.float64])
        assert_type(c64.new_like(), CentralMoments[np.float64])
        assert_type(c32.new_like(dtype=np.float32), CentralMoments[np.float32])

        assert_type(c32.new_like(data32), CentralMoments[np.float32])
        assert_type(c32.new_like(data64), CentralMoments[np.float64])
        assert_type(c64.new_like(data32), CentralMoments[np.float32])

        assert_type(c32.new_like(data32, dtype=np.float64), CentralMoments[np.float64])
        assert_type(c32.new_like(data64, dtype=np.float32), CentralMoments[np.float32])
        assert_type(c64.new_like(data32, dtype=np.float64), CentralMoments[np.float64])

        assert_type(c64.new_like(data64, dtype="f4"), CentralMoments[Any])
        assert_type(c64.new_like(dtype="f4"), CentralMoments[Any])


def test_xcentralmoments_newlike() -> None:
    data32 = np.zeros((4,), dtype=np.float32)
    data64 = np.zeros((4,), dtype=np.float64)

    xdata32 = xr.DataArray(data32)
    xdata64 = xr.DataArray(data64)

    c32 = xCentralMoments.zeros(mom=3, dtype=np.float32)
    c64 = xCentralMoments.zeros(mom=3, dtype=np.float64)

    if TYPE_CHECKING:
        assert_type(c32.new_like(), xCentralMoments[np.float32])
        assert_type(c32.new_like(dtype=np.float64), xCentralMoments[np.float64])
        assert_type(c64.new_like(), xCentralMoments[np.float64])
        assert_type(c32.new_like(dtype=np.float32), xCentralMoments[np.float32])

        assert_type(c32.new_like(data32), xCentralMoments[np.float32])
        assert_type(c32.new_like(data64), xCentralMoments[np.float64])
        assert_type(c64.new_like(data32), xCentralMoments[np.float32])

        assert_type(c32.new_like(xdata32), xCentralMoments[Any])
        assert_type(c32.new_like(xdata64), xCentralMoments[Any])
        assert_type(c64.new_like(xdata32), xCentralMoments[Any])

        assert_type(c32.new_like(data32, dtype=np.float64), xCentralMoments[np.float64])
        assert_type(c32.new_like(data64, dtype=np.float32), xCentralMoments[np.float32])
        assert_type(c64.new_like(data32, dtype=np.float64), xCentralMoments[np.float64])

        assert_type(
            c32.new_like(xdata32, dtype=np.float64), xCentralMoments[np.float64]
        )
        assert_type(
            c32.new_like(xdata64, dtype=np.float32), xCentralMoments[np.float32]
        )
        assert_type(
            c64.new_like(xdata32, dtype=np.float64), xCentralMoments[np.float64]
        )

        assert_type(c64.new_like(data64, dtype="f4"), xCentralMoments[Any])
        assert_type(c64.new_like(dtype="f4"), xCentralMoments[Any])


def test_central_resample_vals() -> None:
    x32 = np.zeros((10, 3, 3), dtype=np.float32)
    x64 = np.zeros((10, 3, 3), dtype=np.float64)
    xAny: NDArray[Any] = x32

    out32 = np.zeros((4,), dtype=np.float32)
    out64 = np.zeros((4,), dtype=np.float64)

    freq = random_freq(20, 10)

    if TYPE_CHECKING:
        assert_type(
            CentralMoments.from_resample_vals(x32, freq=freq, mom=3, axis=0),
            CentralMoments[np.float32],
        )
        assert_type(
            CentralMoments.from_resample_vals(x64, freq=freq, mom=3, axis=0),
            CentralMoments[np.float64],
        )

        assert_type(
            CentralMoments.from_resample_vals(
                xAny, freq=freq, mom=3, axis=0, dtype=np.float64
            ),
            CentralMoments[np.float64],
        )
        assert_type(
            CentralMoments.from_resample_vals(
                xAny, freq=freq, mom=3, axis=0, dtype=np.float32
            ),
            CentralMoments[np.float32],
        )
        assert_type(
            CentralMoments.from_resample_vals(
                x64,
                freq=freq,
                mom=3,
                axis=0,
                dtype="f8",
            ),
            CentralMoments[Any],
        )

        assert_type(
            CentralMoments.from_resample_vals(
                x64,
                freq=freq,
                mom=3,
                axis=0,
                out=out32,
            ),
            CentralMoments[np.float32],
        )

        assert_type(
            CentralMoments.from_resample_vals(
                x64,
                freq=freq,
                mom=3,
                axis=0,
                out=out64,
            ),
            CentralMoments[np.float64],
        )
        assert_type(
            CentralMoments.from_resample_vals(
                xAny, freq=freq, mom=3, axis=0, out=out32, dtype=np.float64
            ),
            CentralMoments[np.float32],
        )

        assert_type(
            CentralMoments.from_resample_vals(
                xAny, freq=freq, mom=3, axis=0, out=out64, dtype=np.float32
            ),
            CentralMoments[np.float64],
        )

        xc = np.array([1, 2, 3])
        assert_type(xc, NDArray[Any])

        # Would like this to default to np.float64
        assert_type(
            CentralMoments.from_resample_vals(xc, freq=freq, mom=3, axis=0),
            CentralMoments[Any],
        )
        assert_type(
            CentralMoments.from_resample_vals(
                xc, freq=freq, mom=3, axis=0, dtype=np.float32
            ),
            CentralMoments[np.float32],
        )
        assert_type(
            CentralMoments.from_resample_vals(
                [1, 2, 3], freq=freq, mom=3, axis=0, dtype=np.float32
            ),
            CentralMoments[np.float32],
        )
        assert_type(
            CentralMoments.from_resample_vals([1, 2, 3], freq=freq, mom=3, axis=0),
            CentralMoments[Any],
        )
        assert_type(
            CentralMoments.from_resample_vals(
                [1, 2, 3], freq=freq, mom=3, axis=0, dtype=np.float64
            ),
            CentralMoments[np.float64],
        )

        assert_type(
            CentralMoments.from_resample_vals(
                [1.0, 2.0, 3.0], freq=freq, mom=3, axis=0
            ),
            CentralMoments[Any],
        )
        assert_type(
            CentralMoments.from_resample_vals(
                [1.0, 2.0, 3.0], freq=freq, mom=3, axis=0, dtype="f4"
            ),
            CentralMoments[Any],
        )
        assert_type(
            CentralMoments.from_resample_vals(
                x32, freq=freq, mom=3, axis=0, dtype="f4"
            ),
            CentralMoments[Any],
        )


def test_xcentral_resample_vals() -> None:
    x32 = xr.DataArray(np.zeros((10, 3, 3), dtype=np.float32))
    x64 = xr.DataArray(np.zeros((10, 3, 3), dtype=np.float64))

    out32 = np.zeros((4,), dtype=np.float32)
    out64 = np.zeros((4,), dtype=np.float64)
    freq = random_freq(20, 10)

    if TYPE_CHECKING:
        assert_type(
            xCentralMoments.from_resample_vals(x32, freq=freq, mom=3),
            xCentralMoments[Any],
        )
        assert_type(
            xCentralMoments.from_resample_vals(x64, freq=freq, mom=3),
            xCentralMoments[Any],
        )

        assert_type(
            xCentralMoments.from_resample_vals(x32, freq=freq, mom=3, dtype=np.float64),
            xCentralMoments[np.float64],
        )
        assert_type(
            xCentralMoments.from_resample_vals(x64, freq=freq, mom=3, dtype=np.float32),
            xCentralMoments[np.float32],
        )

        assert_type(
            xCentralMoments.from_resample_vals(
                x32, freq=freq, mom=3, out=out32, dtype=np.float64
            ),
            xCentralMoments[np.float32],
        )
        assert_type(
            xCentralMoments.from_resample_vals(
                x64, freq=freq, mom=3, out=out64, dtype=np.float32
            ),
            xCentralMoments[np.float64],
        )
        assert_type(
            xCentralMoments.from_resample_vals(x64, freq=freq, mom=3, dtype="f8"),
            xCentralMoments[Any],
        )


# * Moving
def test_rolling_data() -> None:
    x32 = np.array([1, 2, 3], dtype=np.float32)
    x64 = np.array([1, 2, 3], dtype=np.float64)

    out32 = np.zeros((4,), dtype=np.float32)
    out64 = np.zeros((4,), dtype=np.float64)
    outAny: NDArray[Any] = np.zeros((4,), dtype=np.float32)
    if TYPE_CHECKING:
        assert_type(
            rolling.rolling_data(x32, mom_ndim=1, window=3), NDArray[np.float32]
        )
        assert_type(
            rolling.rolling_data(x64, mom_ndim=1, window=3), NDArray[np.float64]
        )

        assert_type(
            rolling.rolling_data(x32, mom_ndim=1, window=3, dtype=np.float64),
            NDArray[np.float64],
        )
        assert_type(
            rolling.rolling_data(x64, mom_ndim=1, window=3, dtype=np.float32),
            NDArray[np.float32],
        )

        assert_type(
            rolling.rolling_data(x32, mom_ndim=1, window=3, out=out64),
            NDArray[np.float64],
        )
        assert_type(
            rolling.rolling_data(x64, mom_ndim=1, window=3, out=out32),
            NDArray[np.float32],
        )

        assert_type(
            rolling.rolling_data(
                x32, mom_ndim=1, window=3, out=out64, dtype=np.float32
            ),
            NDArray[np.float64],
        )
        assert_type(
            rolling.rolling_data(
                x64, mom_ndim=1, window=3, out=out32, dtype=np.float64
            ),
            NDArray[np.float32],
        )
        assert_type(
            rolling.rolling_data(
                x64, mom_ndim=1, window=3, out=outAny, dtype=np.float64
            ),
            NDArray[Any],
        )

        xc = np.array([1, 2, 3])
        assert_type(xc, NDArray[Any])

        # Would like this to default to np.float64
        assert_type(rolling.rolling_data(xc, mom_ndim=1, window=3), NDArray[Any])
        assert_type(
            rolling.rolling_data(xc, mom_ndim=1, window=3, dtype=np.float32),
            NDArray[np.float32],
        )
        assert_type(
            rolling.rolling_data([1, 2, 3], mom_ndim=1, window=3, dtype=np.float32),
            NDArray[np.float32],
        )
        assert_type(
            rolling.rolling_data([1, 2, 3], mom_ndim=1, window=3, dtype=np.float64),
            NDArray[np.float64],
        )

        assert_type(
            rolling.rolling_data([1, 2, 3], mom_ndim=1, window=3, dtype="f8"),
            NDArray[Any],
        )
        assert_type(
            rolling.rolling_data(x32, mom_ndim=1, window=3, dtype="f8"),
            NDArray[Any],
        )
        assert_type(
            rolling.rolling_data(xc, mom_ndim=1, window=3, dtype="f8"),
            NDArray[Any],
        )

        assert_type(
            rolling.rolling_data([1.0, 2.0, 3.0], mom_ndim=1, window=3), NDArray[Any]
        )

        xx = xr.DataArray(x32)
        assert_type(rolling.rolling_data(xx, mom_ndim=1, window=3), xr.DataArray)
        assert_type(
            rolling.rolling_data(xx.to_dataset(), mom_ndim=1, window=3), xr.Dataset
        )

        if MYPY_ONLY:
            g: ArrayLike | xr.DataArray | xr.Dataset = xc
            assert_type(
                rolling.rolling_data(g, mom_ndim=1, window=3),
                Union[xr.DataArray, xr.Dataset, NDArray[Any]],
            )


def test_rolling_vals() -> None:
    x32 = np.array([1, 2, 3], dtype=np.float32)
    x64 = np.array([1, 2, 3], dtype=np.float64)

    out32 = np.zeros((4,), dtype=np.float32)
    out64 = np.zeros((4,), dtype=np.float64)
    outAny: NDArray[Any] = np.zeros((4,), dtype=np.float32)
    if TYPE_CHECKING:
        assert_type(rolling.rolling_vals(x32, mom=3, window=3), NDArray[np.float32])
        assert_type(rolling.rolling_vals(x64, mom=3, window=3), NDArray[np.float64])

        assert_type(
            rolling.rolling_vals(x32, mom=3, window=3, dtype=np.float64),
            NDArray[np.float64],
        )
        assert_type(
            rolling.rolling_vals(x64, mom=3, window=3, dtype=np.float32),
            NDArray[np.float32],
        )

        assert_type(
            rolling.rolling_vals(x32, mom=3, window=3, out=out64), NDArray[np.float64]
        )
        assert_type(
            rolling.rolling_vals(x64, mom=3, window=3, out=out32), NDArray[np.float32]
        )

        assert_type(
            rolling.rolling_vals(x32, mom=3, window=3, out=out64, dtype=np.float32),
            NDArray[np.float64],
        )
        assert_type(
            rolling.rolling_vals(x64, mom=3, window=3, out=out32, dtype=np.float64),
            NDArray[np.float32],
        )
        assert_type(
            rolling.rolling_vals(x64, mom=3, window=3, out=outAny, dtype=np.float64),
            NDArray[Any],
        )

        xc = np.array([1, 2, 3])
        assert_type(xc, NDArray[Any])

        # Would like this to default to np.float64
        assert_type(rolling.rolling_vals(xc, mom=3, window=3), NDArray[Any])
        assert_type(
            rolling.rolling_vals(xc, mom=3, window=3, dtype=np.float32),
            NDArray[np.float32],
        )
        assert_type(
            rolling.rolling_vals([1, 2, 3], mom=3, window=3, dtype=np.float32),
            NDArray[np.float32],
        )
        assert_type(
            rolling.rolling_vals([1, 2, 3], mom=3, window=3, dtype=np.float64),
            NDArray[np.float64],
        )

        assert_type(
            rolling.rolling_vals([1, 2, 3], mom=3, window=3, dtype="f8"),
            NDArray[Any],
        )
        assert_type(
            rolling.rolling_vals(x32, mom=3, window=3, dtype="f8"),
            NDArray[Any],
        )
        assert_type(
            rolling.rolling_vals(xc, mom=3, window=3, dtype="f8"),
            NDArray[Any],
        )

        assert_type(
            rolling.rolling_vals([1.0, 2.0, 3.0], mom=3, window=3), NDArray[Any]
        )

        xx = xr.DataArray(x32)
        assert_type(rolling.rolling_vals(xx, mom=3, window=3), xr.DataArray)
        assert_type(rolling.rolling_vals(xx.to_dataset(), mom=3, window=3), xr.Dataset)
        if MYPY_ONLY:
            g: ArrayLike | xr.DataArray | xr.Dataset = xc
            assert_type(
                rolling.rolling_vals(g, mom=3, window=3),
                Union[xr.DataArray, xr.Dataset, NDArray[Any]],
            )


def test_rolling_exp_data() -> None:
    x32 = np.array([1, 2, 3], dtype=np.float32)
    x64 = np.array([1, 2, 3], dtype=np.float64)

    out32 = np.zeros((4,), dtype=np.float32)
    out64 = np.zeros((4,), dtype=np.float64)
    outAny: NDArray[Any] = np.zeros((4,), dtype=np.float32)
    if TYPE_CHECKING:
        assert_type(
            rolling.rolling_exp_data(x32, mom_ndim=1, alpha=0.2), NDArray[np.float32]
        )
        assert_type(
            rolling.rolling_exp_data(x64, mom_ndim=1, alpha=0.2), NDArray[np.float64]
        )

        assert_type(
            rolling.rolling_exp_data(x32, mom_ndim=1, alpha=0.2, dtype=np.float64),
            NDArray[np.float64],
        )
        assert_type(
            rolling.rolling_exp_data(x64, mom_ndim=1, alpha=0.2, dtype=np.float32),
            NDArray[np.float32],
        )

        assert_type(
            rolling.rolling_exp_data(x32, mom_ndim=1, alpha=0.2, out=out64),
            NDArray[np.float64],
        )
        assert_type(
            rolling.rolling_exp_data(x64, mom_ndim=1, alpha=0.2, out=out32),
            NDArray[np.float32],
        )

        assert_type(
            rolling.rolling_exp_data(
                x32, mom_ndim=1, alpha=0.2, out=out64, dtype=np.float32
            ),
            NDArray[np.float64],
        )
        assert_type(
            rolling.rolling_exp_data(
                x64, mom_ndim=1, alpha=0.2, out=out32, dtype=np.float64
            ),
            NDArray[np.float32],
        )
        assert_type(
            rolling.rolling_exp_data(
                x64, mom_ndim=1, alpha=0.2, out=outAny, dtype=np.float64
            ),
            NDArray[Any],
        )

        xc = np.array([1, 2, 3])
        assert_type(xc, NDArray[Any])

        # Would like this to default to np.float64
        assert_type(rolling.rolling_exp_data(xc, mom_ndim=1, alpha=0.2), NDArray[Any])
        assert_type(
            rolling.rolling_exp_data(xc, mom_ndim=1, alpha=0.2, dtype=np.float32),
            NDArray[np.float32],
        )
        assert_type(
            rolling.rolling_exp_data(
                [1, 2, 3], mom_ndim=1, alpha=0.2, dtype=np.float32
            ),
            NDArray[np.float32],
        )
        assert_type(
            rolling.rolling_exp_data(
                [1, 2, 3], mom_ndim=1, alpha=0.2, dtype=np.float64
            ),
            NDArray[np.float64],
        )

        assert_type(
            rolling.rolling_exp_data([1, 2, 3], mom_ndim=1, alpha=0.2, dtype="f8"),
            NDArray[Any],
        )
        assert_type(
            rolling.rolling_exp_data(x32, mom_ndim=1, alpha=0.2, dtype="f8"),
            NDArray[Any],
        )
        assert_type(
            rolling.rolling_exp_data(xc, mom_ndim=1, alpha=0.2, dtype="f8"),
            NDArray[Any],
        )

        assert_type(
            rolling.rolling_exp_data([1.0, 2.0, 3.0], mom_ndim=1, alpha=0.2),
            NDArray[Any],
        )

        xx = xr.DataArray(x32)
        assert_type(rolling.rolling_exp_data(xx, mom_ndim=1, alpha=0.2), xr.DataArray)
        assert_type(
            rolling.rolling_exp_data(xx.to_dataset(), mom_ndim=1, alpha=0.2), xr.Dataset
        )
        if MYPY_ONLY:
            g: ArrayLike | xr.DataArray | xr.Dataset = xc
            assert_type(
                rolling.rolling_exp_data(g, mom_ndim=1, alpha=0.2),
                Union[xr.DataArray, xr.Dataset, NDArray[Any]],
            )


def test_rolling_exp_vals() -> None:
    x32 = np.array([1, 2, 3], dtype=np.float32)
    x64 = np.array([1, 2, 3], dtype=np.float64)

    out32 = np.zeros((4,), dtype=np.float32)
    out64 = np.zeros((4,), dtype=np.float64)
    outAny: NDArray[Any] = np.zeros((4,), dtype=np.float32)
    if TYPE_CHECKING:
        assert_type(
            rolling.rolling_exp_vals(x32, mom=3, alpha=0.2), NDArray[np.float32]
        )
        assert_type(
            rolling.rolling_exp_vals(x64, mom=3, alpha=0.2), NDArray[np.float64]
        )

        assert_type(
            rolling.rolling_exp_vals(x32, mom=3, alpha=0.2, dtype=np.float64),
            NDArray[np.float64],
        )
        assert_type(
            rolling.rolling_exp_vals(x64, mom=3, alpha=0.2, dtype=np.float32),
            NDArray[np.float32],
        )

        assert_type(
            rolling.rolling_exp_vals(x32, mom=3, alpha=0.2, out=out64),
            NDArray[np.float64],
        )
        assert_type(
            rolling.rolling_exp_vals(x64, mom=3, alpha=0.2, out=out32),
            NDArray[np.float32],
        )

        assert_type(
            rolling.rolling_exp_vals(
                x32, mom=3, alpha=0.2, out=out64, dtype=np.float32
            ),
            NDArray[np.float64],
        )
        assert_type(
            rolling.rolling_exp_vals(
                x64, mom=3, alpha=0.2, out=out32, dtype=np.float64
            ),
            NDArray[np.float32],
        )
        assert_type(
            rolling.rolling_exp_vals(
                x64, mom=3, alpha=0.2, out=outAny, dtype=np.float64
            ),
            NDArray[Any],
        )

        xc = np.array([1, 2, 3])
        assert_type(xc, NDArray[Any])

        # Would like this to default to np.float64
        assert_type(rolling.rolling_exp_vals(xc, mom=3, alpha=0.2), NDArray[Any])
        assert_type(
            rolling.rolling_exp_vals(xc, mom=3, alpha=0.2, dtype=np.float32),
            NDArray[np.float32],
        )
        assert_type(
            rolling.rolling_exp_vals([1, 2, 3], mom=3, alpha=0.2, dtype=np.float32),
            NDArray[np.float32],
        )
        assert_type(
            rolling.rolling_exp_vals([1, 2, 3], mom=3, alpha=0.2, dtype=np.float64),
            NDArray[np.float64],
        )

        assert_type(
            rolling.rolling_exp_vals([1, 2, 3], mom=3, alpha=0.2, dtype="f8"),
            NDArray[Any],
        )
        assert_type(
            rolling.rolling_exp_vals(x32, mom=3, alpha=0.2, dtype="f8"),
            NDArray[Any],
        )
        assert_type(
            rolling.rolling_exp_vals(xc, mom=3, alpha=0.2, dtype="f8"),
            NDArray[Any],
        )

        assert_type(
            rolling.rolling_exp_vals([1.0, 2.0, 3.0], mom=3, alpha=0.2), NDArray[Any]
        )

        xx = xr.DataArray(x32)
        assert_type(rolling.rolling_exp_vals(xx, mom=3, alpha=0.2), xr.DataArray)
        assert_type(
            rolling.rolling_exp_vals(xx.to_dataset(), mom=3, alpha=0.2), xr.Dataset
        )

        if MYPY_ONLY:
            g: ArrayLike | xr.DataArray | xr.Dataset = xc
            assert_type(
                rolling.rolling_exp_vals(g, mom=3, alpha=0.2),
                Union[xr.DataArray, xr.Dataset, NDArray[Any]],
            )

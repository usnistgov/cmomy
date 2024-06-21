# import xarray as xr
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from cmomy import CentralMoments, xCentralMoments
from cmomy.resample import random_freq, resample_vals

if TYPE_CHECKING:
    import sys
    from typing import Any

    from numpy.typing import NDArray

    from cmomy import convert
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
        # reveal_type(c32)
        # reveal_type(c64)
        # reveal_type(c32.astype(np.float64))
        # reveal_type(c64.astype(np.float32))
        # reveal_type(c32.astype(np.dtype("f8")))
        # reveal_type(c64.astype(np.dtype("f4")))

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

        # reveal_type(cx32)
        # reveal_type(cx64)
        # reveal_type(cx32.astype(np.float64))
        # reveal_type(cx64.astype(np.float32))
        # reveal_type(cx32.astype(np.dtype("f8")))
        # reveal_type(cx64.astype(np.dtype("f4")))

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

        # reveal_type(cc32)
        # reveal_type(cc64)
        # reveal_type(cc32.astype(np.float64))
        # reveal_type(cc64.astype(np.float32))
        # reveal_type(cc32.astype(np.dtype("f8")))
        # reveal_type(cc64.astype(np.dtype("f4")))

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

        # reveal_type(reduce_vals([1,2,3], mom=3))
        # reveal_type(reduce_vals([1,2,3], mom=3, dtype=np.float32))

        xx = xr.DataArray(x32)
        assert_type(reduce_vals(xx, mom=3), xr.DataArray)


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

        # reveal_type(reduce_data([1,2,3], mom_ndim=1))
        # reveal_type(reduce_data([1,2,3], mom_ndim=1, dtype=np.float32))

        xx = xr.DataArray(x32)
        assert_type(reduce_data(xx, mom_ndim=1), xr.DataArray)


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

        # reveal_type(convert.moments_type([1,2,3], mom_ndim=1))
        # reveal_type(convert.moments_type([1,2,3], mom_ndim=1, dtype=np.float32))

        xx = xr.DataArray(x32)
        assert_type(convert.moments_type(xx, mom_ndim=1), xr.DataArray)


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

        # reveal_type(convert.moments_to_comoments([1,2,3], mom=(2, -1)))
        # reveal_type(convert.moments_to_comoments([1,2,3], mom=(2, -1), dtype=np.float32))

        xx = xr.DataArray(x32)
        assert_type(convert.moments_to_comoments(xx, mom=(2, -1)), xr.DataArray)


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


def test_update_weight() -> None:
    if TYPE_CHECKING:
        x32 = np.array([1, 2, 3, 4], dtype=np.float32)
        x64 = np.array([1, 2, 3, 4], dtype=np.float64)
        c32 = CentralMoments(x32, mom_ndim=1)
        c64 = CentralMoments(x64, mom_ndim=1)
        cAny = CentralMoments([1, 2, 3, 4], mom_ndim=1)

        assert_type(c32.assign_weight(1.0), CentralMoments[np.float32])
        assert_type(c64.assign_weight(1.0), CentralMoments[np.float64])
        assert_type(cAny.assign_weight(1.0), CentralMoments[Any])

        assert_type(c32.to_x().assign_weight(1.0), xCentralMoments[np.float32])
        assert_type(c64.to_x().assign_weight(1.0), xCentralMoments[np.float64])
        assert_type(cAny.to_x().assign_weight(1.0), xCentralMoments[Any])


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
#     x32 = np.array([1, 2, 3], dtype=np.float32)
#     x64 = np.array([1, 2, 3], dtype=np.float64)

#     out32 = np.zeros((4,), dtype=np.float32)
#     out64 = np.zeros((4,), dtype=np.float64)
#     if TYPE_CHECKING:
#         assert_type(convert.raw_to_central(x32, mom_ndim=1), NDArray[np.float32])
#         assert_type(convert.raw_to_central(x64, mom_ndim=1), NDArray[np.float64])

#         assert_type(convert.raw_to_central(x32, mom_ndim=1, dtype=np.float64), NDArray[np.float64])
#         assert_type(convert.raw_to_central(x64, mom_ndim=1, dtype=np.float32), NDArray[np.float32])

#         assert_type(convert.raw_to_central(x32, mom_ndim=1, out=out64), NDArray[np.float64])
#         assert_type(convert.raw_to_central(x64, mom_ndim=1, out=out32), NDArray[np.float32])

#         assert_type(
#             convert.raw_to_central(x32, mom_ndim=1, out=out64, dtype=np.float32),
#             NDArray[np.float64],
#         )
#         assert_type(
#             convert.raw_to_central(x64, mom_ndim=1, out=out32, dtype=np.float64),
#             NDArray[np.float32],
#         )

#         xc = np.array([1, 2, 3])
#         assert_type(xc, NDArray[Any])

#         # Would like this to default to np.float64
#         assert_type(convert.raw_to_central(xc, mom_ndim=1), NDArray[Any])
#         assert_type(convert.raw_to_central(xc, mom_ndim=1, dtype=np.float32), NDArray[np.float32])
#         assert_type(
#             convert.raw_to_central([1, 2, 3], mom_ndim=1, dtype=np.float32), NDArray[np.float32]
#         )
#         assert_type(
#             convert.raw_to_central([1, 2, 3], mom_ndim=1, dtype=np.float64), NDArray[np.float64]
#         )

#         assert_type(
#             convert.raw_to_central([1, 2, 3], mom_ndim=1, dtype="f8"),
#             NDArray[Any],
#         )
#         assert_type(
#             convert.raw_to_central(x32, mom_ndim=1, dtype="f8"),
#             NDArray[Any],
#         )
#         assert_type(
#             convert.raw_to_central(xc, mom_ndim=1, dtype="f8"),
#             NDArray[Any],
#         )

#         assert_type(convert.raw_to_central([1.0, 2.0, 3.0], mom_ndim=1), NDArray[Any])

#         # reveal_type(convert.raw_to_central([1,2,3], mom_ndim=1))
#         # reveal_type(convert.raw_to_central([1,2,3], mom_ndim=1, dtype=np.float32))

#         xx = xr.DataArray(x32)
#         assert_type(convert.raw_to_central(xx, mom_ndim=1), xr.DataArray)


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

        # reveal_type(reduce_data_grouped([1,2,3], mom_ndim=1, by=by))
        # reveal_type(reduce_data_grouped([1,2,3], mom_ndim=1, by=by, dtype=np.float32))

        xx = xr.DataArray(x32)
        assert_type(reduce_data_grouped(xx, mom_ndim=1, by=by), xr.DataArray)


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

        # reveal_type(resample_data([1,2,3], freq=freq, mom_ndim=1))
        # reveal_type(resample_data([1,2,3], freq=freq, mom_ndim=1, dtype=np.float32))

        xx = xr.DataArray(x32)
        assert_type(resample_data(xx, freq=freq, mom_ndim=1), xr.DataArray)


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

        # reveal_type(resample_vals([1,2,3], freq=freq, mom=3))
        # reveal_type(resample_vals([1,2,3], freq=freq, mom=3, dtype=np.float32))

        xx = xr.DataArray(x32)
        assert_type(resample_vals(xx, freq=freq, mom=3), xr.DataArray)


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


# def test_centralmoments_from_data() -> None:
#     data32 = np.zeros((10, 3, 4), dtype=np.float32)
#     data64 = np.zeros((10, 3, 4), dtype=np.float64)

#     if TYPE_CHECKING:
#         assert_type(
#             CentralMoments.from_data(data32, mom_ndim=1), CentralMoments[np.float32]
#         )
#         assert_type(
#             CentralMoments.from_data(data64, mom_ndim=1), CentralMoments[np.float64]
#         )
#         assert_type(
#             CentralMoments.from_data(data32, mom_ndim=1, dtype=np.float64),
#             CentralMoments[np.float64],
#         )
#         assert_type(
#             CentralMoments.from_data(data32, mom_ndim=1, dtype=np.dtype("f8")),
#             CentralMoments[np.float64],
#         )
#         assert_type(
#             CentralMoments.from_data(data32, mom_ndim=1, dtype="f8"),
#             CentralMoments[Any],
#         )
#         assert_type(
#             CentralMoments.from_data([1, 2, 3], mom_ndim=1), CentralMoments[Any]
#         )
#         assert_type(
#             CentralMoments.from_data([1, 2, 3], mom_ndim=1, dtype="f8"),
#             CentralMoments[Any],
#         )
#         assert_type(
#             CentralMoments.from_data([1, 2, 3], mom_ndim=1, dtype=np.float32),
#             CentralMoments[np.float32],
#         )

#         # z = np.array([1, 2, 3])
#         # reveal_type(z)
#         # reveal_type(CentralMoments(z, mom_ndim=1))
#         # reveal_type(CentralMoments[np.float32](z, mom_ndim=1))

#         # reveal_type(CentralMoments.from_data(z, mom_ndim=1))
#         # reveal_type(CentralMoments.from_data(z, mom_ndim=1, dtype=np.float32))
#         # reveal_type(CentralMoments.from_data(z, mom_ndim=1, dtype=np.float64))

#         # reveal_type(CentralMoments[np.float32].from_data(z, mom_ndim=1))
#         # reveal_type(CentralMoments[np.float64].from_data(z, mom_ndim=1))

#         # reveal_type(CentralMoments[np.float32].from_data([1,2,3], mom_ndim=1))
#         # reveal_type(CentralMoments[np.float64].from_data([1,2,3], mom_ndim=1))

#         # # reveal_type(CentralMoments[np.float32].from_data(z, dtype=np.float32))
#         # # reveal_type(CentralMoments.test(z))
#         # reveal_type(CentralMoments.test2(z))
#         # reveal_type(CentralMoments[np.float32].test(z))
#         # reveal_type(CentralMoments[np.float64].test2(z))


# def test_xcentralmoments_from_data() -> None:
#     data32 = xr.DataArray(np.zeros((10, 3, 4), dtype=np.float32))
#     data64 = xr.DataArray(np.zeros((10, 3, 4), dtype=np.float64))

#     if TYPE_CHECKING:
#         assert_type(xCentralMoments.from_data(data32, mom_ndim=1), xCentralMoments[Any])
#         # assert_type(xCentralMoments[np.float32].from_data(data32, mom_ndim=1), xCentralMoments[np.float32])
#         assert_type(xCentralMoments.from_data(data64, mom_ndim=1), xCentralMoments[Any])
#         assert_type(
#             xCentralMoments.from_data(data32, mom_ndim=1, dtype=np.float32),
#             xCentralMoments[np.float32],
#         )

#         assert_type(
#             xCentralMoments.from_data(data32, mom_ndim=1, dtype=np.float64),
#             xCentralMoments[np.float64],
#         )
#         assert_type(
#             xCentralMoments.from_data(data32, mom_ndim=1, dtype=np.dtype("f8")),
#             xCentralMoments[np.float64],
#         )
#         assert_type(
#             xCentralMoments.from_data(data32, mom_ndim=1, dtype="f8"),
#             xCentralMoments[Any],
#         )


def test_centralmoments_from_vals() -> None:
    data32 = np.zeros((10, 3, 4), dtype=np.float32)
    data64 = np.zeros((10, 3, 4), dtype=np.float64)

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
            CentralMoments.from_vals([1, 2, 3], mom=3, axis=0), CentralMoments[Any]
        )
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

    if TYPE_CHECKING:
        assert_type(
            xCentralMoments.from_vals(data32, mom=3, axis=0), xCentralMoments[Any]
        )
        # assert_type(xCentralMoments[np.float32].from_vals(data32, mom=3, axis=0), xCentralMoments[np.float32])
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

        # assert_type(c32.new_like(np.array([1,2,3], dtype=np.float64)), CentralMoments[np.float64])
        # assert_type(c32.new_like(np.array([1,2,3], dtype=np.float32), dtype=np.float64), np.float64)


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
                x32, freq=freq, mom=3, axis=0, dtype=np.float64
            ),
            CentralMoments[np.float64],
        )
        assert_type(
            CentralMoments.from_resample_vals(
                x64, freq=freq, mom=3, axis=0, dtype=np.float32
            ),
            CentralMoments[np.float32],
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
            xCentralMoments.from_resample_vals(x64, freq=freq, mom=3, dtype="f8"),
            xCentralMoments[Any],
        )

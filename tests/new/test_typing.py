# import xarray as xr
from typing import TYPE_CHECKING, Any, assert_type  # , reveal_type

import numpy as np

from cmomy.new.central_numpy import CentralMoments

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import NDArray

    from cmomy.new.central_dataarray import xCentralMoments
    from cmomy.new.reduction import (
        reduce_data,
        reduce_data_grouped,
        reduce_data_indexed,
        reduce_vals,
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

        assert_type(reduce_data([1.0, 2.0, 3.0], mom_ndim=1), NDArray[Any])

        # reveal_type(reduce_data([1,2,3], mom_ndim=1))
        # reveal_type(reduce_data([1,2,3], mom_ndim=1, dtype=np.float32))

        xx = xr.DataArray(x32)
        assert_type(reduce_data(xx, mom_ndim=1), xr.DataArray)


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

        # reveal_type(reduce_data_indexed([1,2,3], mom_ndim=1, index=index, group_start=group_start, group_end=group_end))
        # reveal_type(reduce_data_indexed([1,2,3], mom_ndim=1, index=index, group_start=group_start, group_end=group_end, dtype=np.float32))

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

# import xarray as xr
from typing import TYPE_CHECKING, Any, assert_type  # , reveal_type

import numpy as np

from cmomy.new.central_numpy import CentralMoments

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import NDArray

    from cmomy.new.central_dataarray import xCentralMoments
    from cmomy.new.reduction import reduce_vals


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

        assert_type(reduce_vals(x32, mom=3, dtype=np.float64), NDArray[np.float64])
        assert_type(reduce_vals(x64, mom=3, dtype=np.float32), NDArray[np.float32])

        assert_type(reduce_vals(x32, mom=3, out=out64), NDArray[np.float64])
        assert_type(reduce_vals(x64, mom=3, out=out32), NDArray[np.float32])

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
        # assert_type(reduce_vals([1,2,3], mom=3, dtype=np.float32), NDArray[np.float32])
        # assert_type(reduce_vals([1,2,3], mom=3, dtype=np.float64), NDArray[np.float64])

        # reveal_type(reduce_vals([1,2,3], mom=3))
        # reveal_type(reduce_vals([1,2,3], mom=3, dtype=np.float32))

        xx = xr.DataArray(x32)
        assert_type(reduce_vals(xx, mom=3), xr.DataArray)

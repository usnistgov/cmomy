# mypy: disable-error-code="no-untyped-def, no-untyped-call"
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cmomy import CentralMoments, xCentralMoments

# * Dtypes
dtype_mark = pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.float32, np.float32),
        (np.dtype(np.float32), np.float32),
        ("f4", np.float32),
        (np.dtype("f4"), np.float32),
        (np.float64, np.float64),
        (np.dtype(np.float64), np.float64),
        ("f8", np.float64),
        (np.dtype("f8"), np.float64),
        (None, np.float64),
        (np.float16, "error"),
    ],
)

dtype_base_mark = pytest.mark.parametrize("dtype_base", [np.float32, np.float64])
cls_mark = pytest.mark.parametrize("cls", [CentralMoments, xCentralMoments])


@dtype_base_mark
@dtype_mark
def test_astype(dtype_base, dtype, expected) -> None:
    c = CentralMoments.zeros(mom=3, val_shape=(2, 3), dtype=dtype_base)
    cx = c.to_x()
    cc = cx.to_c()

    objs: list[CentralMoments | xCentralMoments] = [c, cx, cc]

    for obj in objs:
        assert obj.dtype.type == dtype_base
        if expected == "error":
            with pytest.raises(ValueError, match=".*not supported.*"):
                obj.astype(dtype)
        else:
            assert obj.astype(dtype).dtype.type == expected


@dtype_mark
@cls_mark
def test_zeros_dtype(cls, dtype, expected) -> None:
    if expected == "error":
        with pytest.raises(ValueError, match=".*not supported.*"):
            cls.zeros(mom=3, val_shape=(2, 3), dtype=dtype)

    else:
        assert cls.zeros(mom=3, val_shape=(2, 3), dtype=dtype).dtype.type == expected


@cls_mark
@dtype_base_mark
@dtype_mark
def test_init(cls, dtype_base, dtype, expected) -> None:
    data = np.zeros((2, 3, 4), dtype=dtype_base)

    if cls == xCentralMoments:
        data = xr.DataArray(data)  # type: ignore[assignment]

    if expected == "error":
        with pytest.raises(ValueError, match=".*not supported.*"):
            cls(data, mom_ndim=1, dtype=dtype)
    else:
        c = cls(data, mom_ndim=1, dtype=dtype)

        if dtype is None:
            assert c.dtype.type == dtype_base
        else:
            assert c.dtype.type == expected


@cls_mark
@dtype_base_mark
@dtype_mark
def test_from_vals(cls, dtype_base, dtype, expected) -> None:
    data = np.zeros((2, 3, 4), dtype=dtype_base)

    if cls == xCentralMoments:
        data = xr.DataArray(data)  # type: ignore[assignment]

    if expected == "error":
        with pytest.raises(ValueError, match=".*not supported.*"):
            cls.from_vals(data, mom=3, axis=0, dtype=dtype)
    else:
        c = cls.from_vals(data, mom=3, axis=0, dtype=dtype)

        if dtype is None:
            assert c.dtype.type == dtype_base
        else:
            assert c.dtype.type == expected


@cls_mark
@dtype_base_mark
@dtype_mark
def test_new_like(cls, dtype_base, dtype, expected) -> None:
    data = np.zeros((2, 3, 4), dtype=dtype)
    data_base = np.zeros((2, 3, 4), dtype=dtype_base)
    if cls == xCentralMoments:
        xdata = xr.DataArray(data)
        # xdata_base = xr.DataArray(data_base)

    c = cls.zeros(mom=3, val_shape=(2, 3), dtype=dtype_base)

    assert c.dtype.type == dtype_base

    if expected == "error":
        with pytest.raises(ValueError, match=".*not supported.*"):
            c.new_like(data, dtype=dtype)
    elif dtype is None:
        assert c.new_like().dtype.type == dtype_base
        assert c.new_like(dtype=dtype).dtype.type == dtype_base
        assert c.new_like(data=data).dtype.type == expected
        assert c.new_like(data=data, dtype=dtype).dtype.type == expected
        assert c.new_like(data=data_base).dtype.type == dtype_base
        assert c.new_like(data=data_base, dtype=dtype).dtype.type == dtype_base
        if cls == xCentralMoments:
            assert c.new_like(data=xdata, dtype=dtype).dtype.type == expected

    else:
        assert c.new_like().dtype.type == dtype_base
        assert c.new_like(dtype=dtype).dtype.type == expected
        assert c.new_like(data=data).dtype.type == expected
        assert c.new_like(data=data, dtype=dtype).dtype.type == expected
        assert c.new_like(data=data_base).dtype.type == dtype_base
        assert c.new_like(data=data_base, dtype=dtype).dtype.type == expected
        if cls == xCentralMoments:
            assert c.new_like(data=xdata).dtype.type == expected
            assert c.new_like(data=xdata, dtype=dtype).dtype.type == expected
